from lammps import lammps
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
import time
import pickle
from mpi4py import MPI
from ase import Atoms
from ase.neighborlist import PrimitiveNeighborList
import numpy.typing as npt


from motep_jax.motep_original_files.mtp import read_mtp
from motep_jax.motep_original_files.calculator import MTP


class MTPLammpsInterface:
    def __init__(self, lmp, mtp_filename):
        """Initialize the MTP-LAMMPS interface"""
        self.lmp = lmp
        self.mtp_filename = mtp_filename
        
        self._initialize_mtp()
        
        self._compile_jax_functions()
        
        
    
    def _initialize_mtp(self):
        """Initialize the MTP model"""
    
        self.mtp_data = read_mtp(self.mtp_filename)
        
        self.mtp_instance = MTP(self.mtp_data, engine="jax_new", is_trained=True)
        
        print("MTP model initialized successfully")
    
    def _compile_jax_functions(self):
        """Pre-compile JAX functions for better performance"""
        self.jitted_calc = jax.jit(self._calculate_mtp)
    

    
    def _get_all_distances(self, atoms: Atoms, mtp_data, all_js, all_offsets) -> tuple[np.ndarray, np.ndarray]:
        '''
        Internal helper function to calculate interatomic displacement vectors.

        Given an Atoms object and the pre-computed padded neighbor indices (`all_js`)
        and offset vectors (`all_offsets`), this function calculates the displacement
        vector from each atom `i` to its neighbors `j`. Handles padding by assigning
        a large distance vector for padded neighbor slots.

        :param atoms: ASE Atoms object.
        :param mtp_data: Object containing MTP parameters, including `max_dist`.
        :param all_js: Padded array of neighbor indices (N_atoms, max_neighbors).
        :param all_offsets: Padded array of offset vectors (N_atoms, max_neighbors, 3).
        :return: Tuple containing:
                 - np.ndarray: The input `all_js` array (passed through).
                 - np.ndarray: Array of interatomic displacement vectors `r_ij`
                               (N_atoms, max_neighbors, 3). Padded entries are
                               assigned large values.
        '''
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
        '''
        Internal helper function to process neighbor lists from ASE.

        Takes the raw neighbor list and computes the neighbor indices and the
        corresponding offset vectors due to periodic boundary conditions.
        Pads the results so that each atom has the same number of neighbors
        (filling with -1 for indices and zero vectors for offsets), suitable
        for batch processing in JAX.

        :param nl: ASE PrimitiveNeighborList containing neighbor information.
        :param atoms: ASE Atoms object for which the neighbors were computed.
        :return: Tuple containing:
                 - np.ndarray: Padded array of neighbor indices (N_atoms, max_neighbors).
                 - np.ndarray: Padded array of offset vectors (N_atoms, max_neighbors, 3).
        '''
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
            np.pad(offset, pad_width=(pad, (0, 0)), constant_values=0.0) for offset, pad in zip(offsets, pads) 
        ]

        if not padded_js:
            return np.empty((len(atoms), 0), dtype=int), np.empty((len(atoms), 0, 3), dtype=float)
        return np.array(padded_js, dtype=int), np.array(padded_offsets)
    
    
    def _compute_neighbor_data(self, atoms, mtp_data):
        '''
        Computes the neighbor list indices and offset vectors for a given Atoms object.

        Uses ASE's PrimitiveNeighborList based on the MTP potential's cutoff
        radius (`mtp_data.max_dist`). It then formats this information using
        `_compute_all_offsets` into padded numpy arrays. This data is essential
        for calculating interatomic distances and MTP features.

        :param atoms: ASE Atoms object representing the atomic configuration.
        :param mtp_data: Object containing MTP potential parameters, including `max_dist`.
        :return: Tuple containing:
                 - np.ndarray: Padded array of neighbor indices (N_atoms, max_neighbors).
                 - np.ndarray: Padded array of offset vectors (N_atoms, max_neighbors, 3).
        '''
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
        
        
    def _extract_jaxmtp_input(self, atoms, species, mtp_data, all_js, all_offsets):
        '''
        Extracts and computes all necessary data for a single atomic configuration.
    
        This function takes a single `Atoms` object and pre-computed neighbor data,
        calculates the MTP-specific inputs (atom types, neighbor types, displacement
        vectors using `_get_all_distances`), and retrieves the target properties
        (energy, forces, stress) from the `Atoms` object. Converts results to JAX arrays.
    
        :param atoms: ASE Atoms object for the configuration.
        :param species: List of atomic numbers defining the species types for the MTP.
        :param mtp_data: Object containing MTP potential parameters.
        :param all_js: Padded array of neighbor indices from `compute_neighbor_data`.
        :param all_offsets: Padded array of offset vectors from `compute_neighbor_data`.
        :return: Tuple containing JAX arrays for the configuration:
                 (itypes, all_js, all_rijs, all_jtypes, cell_rank, volume, E, F, sigma).
        '''
    
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
        """Callback function for LAMMPS to get forces from MTP potential"""
        
        if ntimestep % 100 == 0:
            print(f)
                
        start_time = time.time()
        
        symbols = ['Al', 'Ni']           
        cell = [30, 30, 30]           
        pbc = True  
        
        atoms = Atoms(symbols=symbols,positions=x,cell=cell,pbc=pbc)
        
        all_js, all_offsets = self._compute_neighbor_data(atoms, self.mtp_data)
        itypes, all_js, all_rijs, all_jtypes, cell_rank, volume = self._extract_jaxmtp_input(atoms, self.species, self.mtp_data, all_js, all_offsets)
        
        energy, forces, stress = self._calculate_mtp(
            itypes, all_js, all_rijs, all_jtypes, cell_rank, volume
        )
        
        np_forces = np.array(forces) 
                        
        for i in range(nlocal):
            f[i][0] = np_forces[i][0]
            f[i][1] = np_forces[i][1]
            f[i][2] = np_forces[i][2]
        
        elapsed = time.time() - start_time
        self.total_time += elapsed
        self.call_count += 1
        
        return 0
    
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
    """Main function to run LAMMPS with MTP potential"""
    
    comm = MPI.COMM_WORLD
    lmp = lammps(comm=comm)  

    species = [0,1] 
    mtp_interface = MTPLammpsInterface(lmp, 'Ni3Al-12g.mtp')
    
    mtp_interface.setup_simulation(input_file='run_pylammps_int.in')
    lmp.set_fix_external_callback("ext_force", mtp_interface.force_callback)
    
    lmp.command('run 1')

if __name__ == "__main__":
    main()
