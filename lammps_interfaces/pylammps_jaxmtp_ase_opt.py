from lammps import lammps
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
import time
import pickle
from ase import Atoms
from ase.neighborlist import PrimitiveNeighborList
import numpy.typing as npt
from functools import partial
import os

from motep_jax.motep_original_files.mtp import read_mtp
from motep_jax.motep_original_files.calculator import MTP


class MTPLammpsInterface:
    def __init__(self, mtp_file):
        """Initialize the MTP-LAMMPS interface"""
        self.mtp_file = mtp_file
        self._initialize_mtp()
        
        # Pre-compile the JAX function once
        self._compiled = False
        
    def _initialize_mtp(self):
        """Initialize the MTP model with memory optimization"""
        
        # Clear JAX cache to free memory
        jax.clear_caches()
        
        self.mtp_data = read_mtp(self.mtp_file)
        
        self.species = [28,13]
        self.mtp_data.species = self.species
       	 
        if self.species is None:
        	self.mtp_data.species_count = 1
        else:
        	self.mtp_data.species_count = len(self.species) 

        self.params = {
            'species': self.mtp_data.species_coeffs,
            'radial': self.mtp_data.radial_coeffs,
            'basis': self.mtp_data.moment_coeffs
        }
        
        self.mtp_instance = MTP(self.mtp_data, engine="jax_new", is_trained=True)
        
        # Pre-compile with dummy data to avoid compilation during simulation
        self._warmup_jax_compilation()
    
    def _warmup_jax_compilation(self):
        """Pre-compile JAX functions with dummy data"""
        try:
            # Create minimal dummy data for compilation
            dummy_itypes = jnp.array([0, 1])
            dummy_js = jnp.array([[-1], [-1]])
            dummy_rijs = jnp.array([[[10.0, 10.0, 10.0]], [[10.0, 10.0, 10.0]]])
            dummy_jtypes = jnp.array([[-1], [-1]])
            
            # Compile once
            _ = self._calculate_mtp(dummy_itypes, dummy_js, dummy_rijs, 
                                 dummy_jtypes, 3, 1000.0)
            self._compiled = True
            print("JAX compilation completed successfully")
        except Exception as e:
            print(f"JAX compilation warning: {e}")
            self._compiled = False
    
    def _get_all_distances(self, atoms: Atoms, mtp_data, all_js, all_offsets) -> tuple[np.ndarray, np.ndarray]:
        '''Calculate interatomic displacement vectors (optimized)'''
        max_dist = mtp_data.max_dist
        positions = atoms.positions
        offsets = all_offsets
        
        if all_js.shape[1] == 0: 
            all_r_ijs = np.zeros((len(atoms), 0, 3))
        else:
            all_r_ijs = positions[all_js] + offsets - positions[:, None, :]
            mask = all_js < 0
            all_r_ijs[mask, :] = max_dist 
        return all_js, all_r_ijs
    
    def _get_types(self, atoms: Atoms, species: list[int] = None) -> npt.NDArray[np.int64]:
        if species is None:
            return np.array(atoms.numbers, dtype=int)
        else:
            return np.fromiter((species.index(_) for _ in atoms.numbers), dtype=int)
    
    def _compute_all_offsets(self, nl: PrimitiveNeighborList, atoms: Atoms):
        '''Process neighbor lists with memory optimization'''
        cell = atoms.cell
        js = [nl.get_neighbors(i)[0] for i in range(len(atoms))]
        offsets = [nl.get_neighbors(i)[1] @ cell for i in range(len(atoms))]
        num_js = [_.shape[0] for _ in js]
        max_num_js = np.max([_.shape[0] for _ in offsets]) if offsets else 0 
        pads = [(0, max_num_js - n) for n in num_js]
        
        padded_js = [
            np.pad(js_, pad_width=pad, constant_values=-1) for js_, pad in zip(js, pads)
        ]
        padded_offsets = [
            np.pad(offset, pad_width=(pad, (0, 0)), constant_values=0.0) 
            for offset, pad in zip(offsets, pads) 
        ]

        if not padded_js:
            return np.empty((len(atoms), 0), dtype=int), np.empty((len(atoms), 0, 3), dtype=float)
        return np.array(padded_js, dtype=int), np.array(padded_offsets)
    
    def _compute_neighbor_data(self, atoms, mtp_data):
        '''Compute neighbor data with caching'''
        nl = PrimitiveNeighborList(
            cutoffs=[0.5 * mtp_data.max_dist] * len(atoms),
            skin=0.3,
            self_interaction=False,
            bothways=True,
        )
        if len(atoms) > 0:
            nl.update(atoms.pbc, atoms.cell, atoms.positions)
        else:
            return np.empty((0, 0), dtype=int), np.empty((0, 0, 3), dtype=float)

        all_js, all_offsets = self._compute_all_offsets(nl, atoms)
        return all_js, all_offsets
    

    #@partial(jax.jit, static_argnums=(0,1,2,3))
    def _extract_jaxmtp_input(self, atoms, species, mtp_data, all_js, all_offsets):
        '''Extract JAX-MTP input data'''
        itypes = jnp.array(self._get_types(atoms, species))
        all_js, all_rijs = self._get_all_distances(atoms, mtp_data, all_js, all_offsets)
        all_js, all_rijs = jnp.array(all_js), jnp.array(all_rijs)
    
        if all_js.shape[1] > 0:
            all_jtypes = itypes[jnp.asarray(all_js)]
            all_jtypes = jnp.where(all_js >= 0, all_jtypes, -1)
        else:
            all_jtypes = jnp.empty((len(atoms), 0), dtype=itypes.dtype)
    
        cell_rank = atoms.cell.rank
        volume = atoms.get_volume()
    
        return itypes, all_js, all_rijs, all_jtypes, cell_rank, volume
    
    def force_callback(self, caller, ntimestep, nlocal, tag, x, f):
        """Optimized callback function for LAMMPS"""
        
        # Reduce print frequency to save time
        if ntimestep % 1000 == 0:
            print(f"Step {ntimestep}: Computing forces")
                
        start_time = time.time()
        type_to_symbol = {1: 'Al', 2: 'Ni' }
        types = [1] * len(x)
        symbols = [type_to_symbol[t] for t in types]           
        cell = [30.0, 30.0, 30.0]           
        pbc = True  
        
        atoms = Atoms(symbols=symbols, positions=x, cell=cell, pbc=pbc)
        
        start_time_neigh = time.time()
        all_js, all_offsets = self._compute_neighbor_data(atoms, self.mtp_data)
        itypes, all_js, all_rijs, all_jtypes, cell_rank, volume = self._extract_jaxmtp_input(
            atoms, self.species, self.mtp_data, all_js, all_offsets
        )
        end_time_neigh = time.time()
        elapsed_time_neigh = end_time_neigh - start_time_neigh
        print('Neighbor time')
        print(elapsed_time_neigh)

        start_time_mtp = time.time()
        energy, forces, stress = self._calculate_mtp(
            itypes, all_js, all_rijs, all_jtypes, cell_rank, volume
        )
        end_time_mtp = time.time()
        elapsed_time_mtp = end_time_mtp - start_time_mtp
        print('MTP time')
        print(elapsed_time_mtp)
        
        # Convert to numpy efficiently
        np_forces = np.asarray(forces)
                        
        # Vectorized assignment (faster than loop)
        f[:nlocal, :] = np_forces[:nlocal, :]
        
        if ntimestep % 1000 == 0:
            elapsed = time.time() - start_time
            print(f"Force calculation took {elapsed:.4f} seconds")
        
        return 0

    @partial(jax.jit, static_argnums=(0))
    def _calculate_mtp(self, itypes, all_js, all_rijs, all_jtypes, cell_rank, volume):
        """JAX function to calculate energy, forces, and stress using MTP"""
        targets = self.mtp_instance.calculate_jax_new(
            itypes, all_js, all_rijs, all_jtypes, cell_rank, volume, self.params
        )
        
        energy = targets['energy']
        forces = targets['forces']
        stress = targets['stress']
        
        return energy, forces, stress


def main():
    """Main function with memory optimization"""
    import gc
    gc.collect()
    
    lmp = lammps()  

    lmp.command("units metal")
    lmp.command("dimension 3")
    lmp.command("boundary p p p")
    lmp.command("atom_style atomic")
    #lmp.command("read_data lammps_data_file.data")
    lmp.command("read_data equil-400K-2dKWL.data")

    lmp.command("mass 1 26.98") 
    lmp.command("mass 2 58.69") 

    lmp.command("variable temperature equal 1300")
    
    lmp.command("reset_timestep 0")
    lmp.command("timestep 0.001")
    lmp.command("velocity all create ${temperature} 12345")

    print("Initializing MTP interface...")
    mtp_interface = MTPLammpsInterface('Ni3Al-12g.mtp')
    print("MTP interface initialized successfully")
    
    lmp.command("fix ext_force all external pf/callback 1 1")
    
    lmp.set_fix_external_callback("ext_force", mtp_interface.force_callback)
    
    lmp.command("fix integrator all nve")
    
    lmp.command("thermo 200")
    lmp.command("thermo_style custom step dt temp vol lx ly lz press pxx pyy pzz pe ke etotal pxy pxz pyz")
    
    lmp.command("dump trajectory all atom 100 lammps_jaxmtp.dump")
    lmp.command("dump_modify trajectory sort id")

    print("Starting simulation...")
    lmp.command('run 2')
    print("Simulation completed successfully")


if __name__ == "__main__":
    main()
