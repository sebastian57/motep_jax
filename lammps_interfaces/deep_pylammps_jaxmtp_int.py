from lammps import lammps
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax, pmap, vmap
import time
import pickle
from ase.neighborlist import PrimitiveNeighborList
import numpy.typing as npt
from functools import partial
import os
from mpi4py import MPI

from motep_jax.motep_original_files.mtp import read_mtp
from motep_jax.motep_original_files.calculator import MTP


class OptimizedMTPLammpsInterface:
    def __init__(self, mtp_file, chunk_size=8192, use_pmap=True):
        """Initialize the optimized MTP-LAMMPS interface
        
        Args:
            mtp_file: Path to MTP potential file
            chunk_size: Number of atoms per JAX batch (must be power of 2 for efficiency)
            use_pmap: Whether to use pmap for multi-GPU within node
        """
        self.mtp_file = mtp_file
        self.chunk_size = chunk_size
        self.use_pmap = use_pmap
        
        # MPI setup
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        
        # JAX device setup
        self._setup_jax_devices()
        self._initialize_mtp()
        
        # Pre-compile functions
        self._precompile_jax_functions()
        
        # Type mapping for LAMMPS types to atomic numbers
        self.type_map = {1: 13, 2: 28}  # 1->Al, 2->Ni
        self.species_to_index = {13: 0, 28: 1}  # Al->0, Ni->1
        
        # Performance tracking
        self.timing_stats = {
            'neighbor_time': [],
            'force_time': [],
            'total_time': []
        }
        
        # Cache for neighbor lists to avoid recomputation
        self.neighbor_cache = {}
        self.last_positions = None
        
    def _setup_jax_devices(self):
        """Setup JAX devices for optimal GPU utilization"""
        # Get available devices
        self.devices = jax.devices()
        self.n_devices = len(self.devices)
        
        if self.rank == 0:
            print(f"Available JAX devices: {self.n_devices}")
            for i, device in enumerate(self.devices):
                print(f"  Device {i}: {device}")
        
        # For pmap, we need to ensure chunk_size is divisible by n_devices
        if self.use_pmap and self.n_devices > 1:
            if self.chunk_size % self.n_devices != 0:
                # Adjust chunk_size to be divisible
                self.chunk_size = ((self.chunk_size // self.n_devices) + 1) * self.n_devices
                if self.rank == 0:
                    print(f"Adjusted chunk_size to {self.chunk_size} for pmap compatibility")
        
    def _initialize_mtp(self):
        """Initialize the MTP model with memory optimization"""
        jax.clear_caches()
        
        self.mtp_data = read_mtp(self.mtp_file)
        self.species = [28, 13]  # Ni, Al
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
        
    def _precompile_jax_functions(self):
        """Pre-compile all JAX functions with fixed shapes"""
        if self.rank == 0:
            print("Pre-compiling JAX functions...")
        
        start_time = time.time()
        
        # Create dummy data with exact chunk size
        dummy_size = self.chunk_size
        dummy_itypes = jnp.zeros(dummy_size, dtype=jnp.int32)
        dummy_js = jnp.full((dummy_size, 64), -1, dtype=jnp.int32)  # Assume max 64 neighbors
        dummy_rijs = jnp.ones((dummy_size, 64, 3), dtype=jnp.float32) * 10.0
        dummy_jtypes = jnp.full((dummy_size, 64), -1, dtype=jnp.int32)
        
        # Pre-compile single chunk function
        try:
            _ = self._calculate_mtp_chunk(dummy_itypes, dummy_js, dummy_rijs, 
                                       dummy_jtypes, 3, 1000.0)
            if self.rank == 0:
                print("✓ Single chunk compilation successful")
        except Exception as e:
            if self.rank == 0:
                print(f"✗ Single chunk compilation failed: {e}")
        
        # Pre-compile pmap version if using multiple devices
        if self.use_pmap and self.n_devices > 1:
            try:
                # Reshape for pmap (batch dimension for devices)
                pmap_size = dummy_size // self.n_devices
                pmap_itypes = dummy_itypes.reshape(self.n_devices, pmap_size)
                pmap_js = dummy_js.reshape(self.n_devices, pmap_size, 64)
                pmap_rijs = dummy_rijs.reshape(self.n_devices, pmap_size, 64, 3)
                pmap_jtypes = dummy_jtypes.reshape(self.n_devices, pmap_size, 64)
                
                _ = self._calculate_mtp_pmap(pmap_itypes, pmap_js, pmap_rijs, 
                                          pmap_jtypes, 3, 1000.0)
                if self.rank == 0:
                    print("✓ Pmap compilation successful")
            except Exception as e:
                if self.rank == 0:
                    print(f"✗ Pmap compilation failed, falling back to single device: {e}")
                self.use_pmap = False
        
        compilation_time = time.time() - start_time
        if self.rank == 0:
            print(f"Compilation completed in {compilation_time:.2f} seconds")
    
    @partial(jax.jit, static_argnums=(0,))
    def _calculate_mtp_chunk(self, itypes, all_js, all_rijs, all_jtypes, cell_rank, volume):
        """JAX function to calculate forces for a single chunk"""
        targets = self.mtp_instance.calculate_jax_new(
            itypes, all_js, all_rijs, all_jtypes, cell_rank, volume, self.params
        )
        return targets['forces']
    
    @partial(pmap, static_broadcasted_argnums=(0,))
    def _calculate_mtp_pmap(self, itypes, all_js, all_rijs, all_jtypes, cell_rank, volume):
        """Pmap version for multi-device computation"""
        targets = self.mtp_instance.calculate_jax_new(
            itypes, all_js, all_rijs, all_jtypes, cell_rank, volume, self.params
        )
        return targets['forces']
    
    def _extract_lammps_data(self, caller, nlocal):
        """Extract necessary data directly from LAMMPS without ASE objects"""
        # Extract positions - this returns a direct reference to LAMMPS data
        positions = caller.numpy.extract_atom("x")[:nlocal]
        
        # Extract atom types
        types = caller.numpy.extract_atom("type")[:nlocal]
        
        # Convert LAMMPS types to atomic numbers
        atomic_numbers = np.array([self.type_map[t] for t in types])
        
        # Extract box information
        boxlo, boxhi, xy, yz, xz, periodicity, box_change = caller.extract_box()
        
        # Calculate cell vectors
        xlo, ylo, zlo = boxlo
        xhi, yhi, zhi = boxhi
        
        # Handle triclinic boxes
        if xy != 0.0 or yz != 0.0 or xz != 0.0:
            # Triclinic box
            cell = np.array([
                [xhi - xlo, 0.0, 0.0],
                [xy, yhi - ylo, 0.0], 
                [xz, yz, zhi - zlo]
            ])
            cell_rank = 6
        else:
            # Orthogonal box
            cell = np.array([
                [xhi - xlo, 0.0, 0.0],
                [0.0, yhi - ylo, 0.0],
                [0.0, 0.0, zhi - zlo]
            ])
            cell_rank = 3
            
        # Calculate volume
        volume = caller.get_volume()
        
        # PBC settings
        pbc = np.array(periodicity, dtype=bool)
        
        return positions, atomic_numbers, cell, volume, cell_rank, pbc
    
    def _build_neighbor_list_fast(self, positions, cell, pbc, cutoff=None):
        """Build neighbor list using efficient numpy operations"""
        if cutoff is None:
            cutoff = self.mtp_data.max_dist
            
        nlocal = len(positions)
        
        # Use a simple cutoff-based neighbor list for now
        # In production, you might want to use spatial hashing or cell lists
        all_js = []
        all_offsets = []
        
        # Calculate minimum image distances
        inv_cell = np.linalg.inv(cell)
        
        for i in range(nlocal):
            neighbors = []
            offsets = []
            
            for j in range(nlocal):
                if i == j:
                    continue
                    
                rij = positions[j] - positions[i]
                
                # Apply minimum image convention for PBC
                if np.any(pbc):
                    # Convert to fractional coordinates
                    frac_rij = np.dot(rij, inv_cell)
                    
                    # Apply minimum image
                    for dim in range(3):
                        if pbc[dim]:
                            frac_rij[dim] = frac_rij[dim] - np.round(frac_rij[dim])
                    
                    # Convert back to Cartesian
                    rij = np.dot(frac_rij, cell)
                
                distance = np.linalg.norm(rij)
                
                if distance < cutoff:
                    neighbors.append(j)
                    offsets.append(rij)
            
            all_js.append(np.array(neighbors, dtype=int))
            all_offsets.append(np.array(offsets) if len(offsets) > 0 else np.zeros((0, 3)))
        
        return all_js, all_offsets
    
    def _process_chunk_data_fast(self, chunk_start_idx, all_js, all_offsets):
        """Process neighbor data for a single chunk without ASE"""
        chunk_end_idx = min(chunk_start_idx + self.chunk_size, len(all_js))
        
        # Extract neighbor data for this chunk
        chunk_js = all_js[chunk_start_idx:chunk_end_idx]
        chunk_offsets = all_offsets[chunk_start_idx:chunk_end_idx]
        
        # Pad to fixed size for JAX compilation
        max_neighbors = 64  # Fixed maximum
        padded_js = []
        padded_offsets = []
        
        for js, offsets in zip(chunk_js, chunk_offsets):
            if len(js) > max_neighbors:
                # Truncate if too many neighbors
                js = js[:max_neighbors]
                offsets = offsets[:max_neighbors]
            
            pad_size = max_neighbors - len(js)
            padded_js.append(np.pad(js, (0, pad_size), constant_values=-1))
            padded_offsets.append(np.pad(offsets, ((0, pad_size), (0, 0)), constant_values=0.0))
        
        # Pad chunk to fixed size
        while len(padded_js) < self.chunk_size:
            padded_js.append(np.full(max_neighbors, -1, dtype=int))
            padded_offsets.append(np.zeros((max_neighbors, 3)))
        
        return np.array(padded_js), np.array(padded_offsets)
    
    def _extract_chunk_input_fast(self, atomic_numbers, positions, chunk_start_idx, 
                                  all_js, all_offsets):
        """Extract JAX-MTP input for a chunk without ASE objects"""
        chunk_end_idx = min(chunk_start_idx + self.chunk_size, len(atomic_numbers))
        actual_chunk_size = chunk_end_idx - chunk_start_idx
        
        # Get atom types for this chunk and convert to species indices
        chunk_atomic_numbers = atomic_numbers[chunk_start_idx:chunk_end_idx]
        itypes = np.array([self.species_to_index[num] for num in chunk_atomic_numbers])
        
        # Pad to fixed chunk size
        if actual_chunk_size < self.chunk_size:
            padding = self.chunk_size - actual_chunk_size
            itypes = np.pad(itypes, (0, padding), constant_values=0)
        
        # Get neighbor data for this chunk
        chunk_js, chunk_offsets = self._process_chunk_data_fast(
            chunk_start_idx, all_js, all_offsets
        )
        
        # Calculate displacement vectors
        chunk_positions = positions[chunk_start_idx:chunk_end_idx]
        
        # Pad chunk positions
        if actual_chunk_size < self.chunk_size:
            padding = self.chunk_size - actual_chunk_size
            chunk_positions = np.pad(chunk_positions, ((0, padding), (0, 0)), constant_values=0.0)
        
        # Calculate rijs using the precomputed offsets
        all_rijs = []
        for i, (js, offsets) in enumerate(zip(chunk_js, chunk_offsets)):
            if i < actual_chunk_size:
                valid_mask = js >= 0
                rijs = np.zeros((len(js), 3))
                # Use precomputed offsets directly
                rijs[valid_mask] = offsets[valid_mask]
                rijs[~valid_mask] = self.mtp_data.max_dist
            else:
                rijs = np.full((len(js), 3), self.mtp_data.max_dist)
            all_rijs.append(rijs)
        
        all_rijs = np.array(all_rijs)
        
        # Get neighbor types
        all_jtypes = np.where(chunk_js >= 0, 
                             np.array([self.species_to_index[atomic_numbers[j]] if j >= 0 else -1 
                                     for j in chunk_js.flatten()]).reshape(chunk_js.shape),
                             -1)
        
        return (jnp.array(itypes), jnp.array(chunk_js), 
                jnp.array(all_rijs), jnp.array(all_jtypes))
        
    def force_callback(self, caller, ntimestep, nlocal, tag, x, f):
        """Optimized callback with direct LAMMPS data extraction"""
        start_time = time.time()
        
        if ntimestep % 1000 == 0 and self.rank == 0:
            print(f"Step {ntimestep}: Computing forces (Rank {self.rank}, {nlocal} atoms)")
        
        # Extract data directly from LAMMPS - much faster than creating ASE objects
        positions, atomic_numbers, cell, volume, cell_rank, pbc = self._extract_lammps_data(
            caller, nlocal
        )
        
        neighbor_start = time.time()
        
        # Build neighbor list efficiently
        all_js, all_offsets = self._build_neighbor_list_fast(
            positions, cell, pbc, self.mtp_data.max_dist
        )
        
        neighbor_time = time.time() - neighbor_start
        force_start = time.time()
        
        total_forces = np.zeros_like(positions)
        
        n_chunks = (nlocal + self.chunk_size - 1) // self.chunk_size
        
        for chunk_idx in range(n_chunks):
            chunk_start = chunk_idx * self.chunk_size
            chunk_end = min(chunk_start + self.chunk_size, nlocal)
            actual_chunk_size = chunk_end - chunk_start
            
            if actual_chunk_size == 0:
                continue
            
            itypes, js, rijs, jtypes = self._extract_chunk_input_fast(
                atomic_numbers, positions, chunk_start, all_js, all_offsets
            )
            
            if self.use_pmap and self.n_devices > 1:
                device_batch_size = self.chunk_size // self.n_devices
                itypes_pmap = itypes.reshape(self.n_devices, device_batch_size)
                js_pmap = js.reshape(self.n_devices, device_batch_size, -1)
                rijs_pmap = rijs.reshape(self.n_devices, device_batch_size, -1, 3)
                jtypes_pmap = jtypes.reshape(self.n_devices, device_batch_size, -1)
                
                chunk_forces = self._calculate_mtp_pmap(
                    itypes_pmap, js_pmap, rijs_pmap, jtypes_pmap, 
                    cell_rank, volume
                )
                chunk_forces = chunk_forces.reshape(self.chunk_size, 3)
            else:
                chunk_forces = self._calculate_mtp_chunk(
                    itypes, js, rijs, jtypes, cell_rank, volume
                )
            
            chunk_forces_np = np.asarray(chunk_forces)
            total_forces[chunk_start:chunk_end] = chunk_forces_np[:actual_chunk_size]
        
        force_time = time.time() - force_start
        
        # Set forces in LAMMPS
        f[:nlocal, :] = total_forces
        
        total_time = time.time() - start_time
        
        if ntimestep % 1000 == 0:
            self.timing_stats['neighbor_time'].append(neighbor_time)
            self.timing_stats['force_time'].append(force_time)
            self.timing_stats['total_time'].append(total_time)
            
            if self.rank == 0:
                print(f"Timing - Neighbor: {neighbor_time:.4f}s, Force: {force_time:.4f}s, Total: {total_time:.4f}s")
                if len(self.timing_stats['total_time']) > 1:
                    avg_time = np.mean(self.timing_stats['total_time'][-10:])
                    print(f"Average time (last 10): {avg_time:.4f}s")
        
        return 0


def main():
    """Main function with MPI-aware initialization"""
    import gc
    gc.collect()
    
    # MPI setup
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print(f"Starting simulation with {size} MPI ranks")
    
    # Initialize LAMMPS
    lmp = lammps()

    lmp.command("units metal")
    lmp.command("dimension 3")
    lmp.command("boundary p p p")
    lmp.command("atom_style atomic")
    lmp.command("read_data equil-400K-2dKWL.data")

    lmp.command("mass 1 26.98") 
    lmp.command("mass 2 58.69") 

    lmp.command("variable temperature equal 1300")
    
    lmp.command("reset_timestep 0")
    lmp.command("timestep 0.001")
    lmp.command("velocity all create ${temperature} 12345")

    if rank == 0:
        print("Initializing optimized MTP interface...")
    
    # Initialize with appropriate chunk size
    chunk_size = 8192  # Start conservative, can increase
    mtp_interface = OptimizedMTPLammpsInterface(
        'Ni3Al-12g.mtp', 
        chunk_size=chunk_size,
        use_pmap=True  # Enable multi-GPU within node
    )
    
    if rank == 0:
        print("MTP interface initialized successfully")
    
    lmp.command("fix ext_force all external pf/callback 1 1")
    lmp.set_fix_external_callback("ext_force", mtp_interface.force_callback)
    lmp.command("fix integrator all nve")
    
    lmp.command("thermo 200")
    lmp.command("thermo_style custom step dt temp vol lx ly lz press pxx pyy pzz pe ke etotal pxy pxz pyz")
    
    lmp.command("dump trajectory all atom 100 lammps_jaxmtp_optimized.dump")
    lmp.command("dump_modify trajectory sort id")

    if rank == 0:
        print("Starting optimized simulation...")
    
    lmp.command('run 10000')
    
    if rank == 0:
        print("Simulation completed successfully")
        
        # Print performance summary
        avg_neighbor = np.mean(mtp_interface.timing_stats['neighbor_time'])
        avg_force = np.mean(mtp_interface.timing_stats['force_time'])
        avg_total = np.mean(mtp_interface.timing_stats['total_time'])
        print(f"\nPerformance Summary:")
        print(f"Average neighbor time: {avg_neighbor:.4f}s")
        print(f"Average force time: {avg_force:.4f}s")
        print(f"Average total time: {avg_total:.4f}s")
        print(f"Force computation efficiency: {avg_force/avg_total*100:.1f}%")


if __name__ == "__main__":
    main()