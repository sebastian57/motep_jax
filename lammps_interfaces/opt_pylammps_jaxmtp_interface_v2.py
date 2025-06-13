# JAX device setup for MI300A - No mpi4py version
import os
os.environ['XLA_FLAGS'] = '--xla_gpu_enable_triton_softmax_fusion=true --xla_gpu_triton_gemm_any=True'

from lammps import lammps
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax, pmap, local_device_count
import time
import pickle
from ase import Atoms
from ase.neighborlist import PrimitiveNeighborList
import numpy.typing as npt
from functools import partial, lru_cache
import h5py

from motep_jax.motep_original_files.mtp import read_mtp
from motep_jax.motep_original_files.calculator import MTP


class HighlyOptimizedMTPLammpsInterface:
    def __init__(self, mtp_file, cache_dir="./jax_cache", 
                 chunk_size=50000, memory_fraction=0.8):
        self.mtp_file = mtp_file
        self.cache_dir = cache_dir
        self.chunk_size = chunk_size
        self.memory_fraction = memory_fraction
        
        # Get process info from environment variables set by mpirun/mpiexec
        self.rank = int(os.environ.get('OMPI_COMM_WORLD_RANK', 
                                      os.environ.get('PMI_RANK', 
                                      os.environ.get('SLURM_PROCID', '0'))))
        self.size = int(os.environ.get('OMPI_COMM_WORLD_SIZE', 
                                      os.environ.get('PMI_SIZE', 
                                      os.environ.get('SLURM_NPROCS', '1'))))
        
        print(f"Process {self.rank} of {self.size} starting...")
        
        # JAX device setup
        self._setup_jax_devices()
        
        # Initialize MTP
        self._initialize_mtp_cached()
        
        # Setup memory management
        self._setup_memory_pools()
        
        # Setup performance monitoring
        self._setup_performance_monitoring()

    def _setup_jax_devices(self):
        """Setup JAX devices with proper distribution across processes"""
        self.devices = jax.local_devices()
        self.n_devices = len(self.devices)
        
        print(f"Rank {self.rank}: Found {self.n_devices} JAX devices: {self.devices}")
        
        # Distribute devices across MPI processes
        if self.size > 1:
            device_per_rank = max(1, self.n_devices // self.size)
            start_device = self.rank * device_per_rank
            end_device = min((self.rank + 1) * device_per_rank, self.n_devices)
            self.assigned_devices = self.devices[start_device:end_device]
        else:
            self.assigned_devices = self.devices
        
        if self.assigned_devices:
            jax.config.update('jax_default_device', self.assigned_devices[0])
            print(f"Rank {self.rank}: Using devices {self.assigned_devices}")

    def _initialize_mtp_cached(self):
        """Initialize MTP with file-based caching for multi-process"""
        os.makedirs(self.cache_dir, exist_ok=True)
        cache_file = os.path.join(self.cache_dir, f"mtp_cache_shared.pkl")
        
        # Use file locking for cache coordination between processes
        lock_file = cache_file + ".lock"
        
        # Process 0 builds the cache, others wait
        if self.rank == 0:
            if os.path.exists(cache_file):
                print("Loading cached MTP compilation...")
                try:
                    with open(cache_file, 'rb') as f:
                        cached_data = pickle.load(f)
                        self.mtp_data = cached_data['mtp_data']
                        self.params = cached_data['params']
                        self.mtp_instance = cached_data['mtp_instance']
                        print("Successfully loaded cached MTP data")
                except Exception as e:
                    print(f"Cache load failed: {e}, rebuilding...")
                    self._build_mtp_from_scratch()
                    self._save_mtp_cache(cache_file)
            else:
                self._build_mtp_from_scratch()
                self._save_mtp_cache(cache_file)
            
            # Signal other processes that cache is ready
            with open(lock_file, 'w') as f:
                f.write("ready")
        else:
            # Wait for process 0 to finish building cache
            print(f"Rank {self.rank}: Waiting for cache to be built...")
            while not os.path.exists(lock_file):
                time.sleep(0.1)
            
            # Load the cache
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                self.mtp_data = cached_data['mtp_data']
                self.params = cached_data['params']
                self.mtp_instance = cached_data['mtp_instance']
        
        # Compile JAX functions
        self._parallel_jax_compilation()

    def _build_mtp_from_scratch(self):
        """Build MTP from scratch"""
        jax.clear_caches()
        
        self.mtp_data = read_mtp(self.mtp_file)
        self.species = [28, 13]  # Ni, Al
        self.mtp_data.species = self.species
        self.mtp_data.species_count = len(self.species)

        self.params = {
            'species': self.mtp_data.species_coeffs,
            'radial': self.mtp_data.radial_coeffs,
            'basis': self.mtp_data.moment_coeffs
        }
        
        self.mtp_instance = MTP(self.mtp_data, engine="jax_new", is_trained=True)

    def _save_mtp_cache(self, cache_file):
        """Save MTP cache to file"""
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'mtp_data': self.mtp_data,
                    'params': self.params,
                    'mtp_instance': self.mtp_instance
                }, f)
            print(f"Saved MTP cache to {cache_file}")
        except Exception as e:
            print(f"Failed to save cache: {e}")

    def _parallel_jax_compilation(self):
        """Compile JAX functions for different input sizes"""
        print(f"Rank {self.rank}: Starting parallel JAX compilation...")
        
        # Each process compiles for its expected workload
        test_sizes = [100, 1000, 10000, self.chunk_size]
        
        if len(self.assigned_devices) > 1:
            self._compile_with_pmap(test_sizes)
        else:
            self._compile_single_device(test_sizes)
        
        print(f"Rank {self.rank}: JAX compilation completed")

    def _compile_with_pmap(self, test_sizes):
        """Compile with pmap for multi-device"""
        n_devices = len(self.assigned_devices)
        
        for size in test_sizes:
            size_per_device = max(1, size // n_devices)
            
            dummy_data = {
                'itypes': jnp.array([[i % 2] * size_per_device for i in range(n_devices)]),
                'js': jnp.array([[[-1] * 10] * size_per_device for _ in range(n_devices)]),
                'rijs': jnp.array([[[10.0, 10.0, 10.0]] * 10 * size_per_device for _ in range(n_devices)]),
                'jtypes': jnp.array([[[-1] * 10] * size_per_device for _ in range(n_devices)])
            }
            
            try:
                _ = self._calculate_mtp_pmap(
                    dummy_data['itypes'], dummy_data['js'], 
                    dummy_data['rijs'], dummy_data['jtypes'],
                    3, 1000.0
                )
                print(f"Rank {self.rank}: Compiled pmap for size {size}")
            except Exception as e:
                print(f"Rank {self.rank}: pmap compilation warning for size {size}: {e}")

    def _compile_single_device(self, test_sizes):
        """Compile for single device"""
        for size in test_sizes:
            dummy_itypes = jnp.array([i % 2 for i in range(size)])
            dummy_js = jnp.array([[-1] * 10 for _ in range(size)])
            dummy_rijs = jnp.array([[[10.0, 10.0, 10.0]] * 10 for _ in range(size)])
            dummy_jtypes = jnp.array([[-1] * 10 for _ in range(size)])
            
            try:
                _ = self._calculate_mtp_jit(
                    dummy_itypes, dummy_js, dummy_rijs, 
                    dummy_jtypes, 3, 1000.0
                )
                print(f"Rank {self.rank}: Compiled JIT for size {size}")
            except Exception as e:
                print(f"Rank {self.rank}: single device compilation warning for size {size}: {e}")

    def _setup_memory_pools(self):
        """Setup pre-allocated memory pools"""
        max_atoms = 1536000
        max_neighbors = 100  
        
        self.position_buffer = np.empty((max_atoms, 3), dtype=np.float32)
        self.force_buffer = np.empty((max_atoms, 3), dtype=np.float32)
        self.type_buffer = np.empty(max_atoms, dtype=np.int32)
        self.neighbor_buffer = np.empty((max_atoms, max_neighbors), dtype=np.int32)
        self.offset_buffer = np.empty((max_atoms, max_neighbors, 3), dtype=np.float32)
        
        self.jax_position_buffer = jnp.empty((max_atoms, 3), dtype=jnp.float32)
        self.jax_force_buffer = jnp.empty((max_atoms, 3), dtype=jnp.float32)
        
        print(f"Rank {self.rank}: Allocated memory pools for {max_atoms} atoms")

    def _setup_performance_monitoring(self):
        """Setup performance monitoring"""
        self._timing_stats = {
            'data_transfer_in': [],
            'neighbor_calc': [],
            'jax_prep': [],
            'mtp_calc': [],
            'data_transfer_out': [],
            'total': []
        }
        self._memory_stats = {'peak_gpu': 0, 'peak_cpu': 0}

    def force_callback(self, caller, ntimestep, nlocal, tag, x, f):
        """Main force callback function called by LAMMPS"""
        if ntimestep % 100 == 0 and self.rank == 0:
            print(f"Step {ntimestep}: Computing forces for {nlocal} atoms (Rank {self.rank})")
        
        total_start = time.time()
        
        # Fast data transfer in
        transfer_in_start = time.time()
        self._fast_data_transfer_in(x, nlocal)
        transfer_in_time = time.time() - transfer_in_start
        
        # Process atoms (chunked if necessary)
        if nlocal > self.chunk_size:
            forces = self._process_chunked(nlocal)
        else:
            forces = self._process_single_chunk(nlocal)
        
        # Fast data transfer out
        transfer_out_start = time.time()
        self._fast_data_transfer_out(f, forces, nlocal)
        transfer_out_time = time.time() - transfer_out_start
        
        total_time = time.time() - total_start
        
        # Update timing statistics
        self._timing_stats['data_transfer_in'].append(transfer_in_time)
        self._timing_stats['data_transfer_out'].append(transfer_out_time)
        self._timing_stats['total'].append(total_time)
        
        if ntimestep % 100 == 0 and self.rank == 0:
            self._print_performance_snapshot(ntimestep, total_time, nlocal)
        
        return 0

    def _fast_data_transfer_in(self, x, nlocal):
        """Fast data transfer from LAMMPS to numpy"""
        try:
            # Try direct numpy array conversion first (most reliable)
            x_array = np.array(x, copy=False)
            if x_array.shape == (nlocal, 3):
                self.current_positions = x_array.astype(np.float32)
                return
        except:
            pass
        
        try:
            # Try array interface with proper pointer handling
            if hasattr(x, '__array_interface__'):
                array_interface = x.__array_interface__
                data_ptr = array_interface['data']
                
                # Handle both tuple (ptr, read_only) and direct pointer cases
                if isinstance(data_ptr, tuple):
                    ptr_address = data_ptr[0]
                else:
                    ptr_address = data_ptr
                
                # Create array from memory address
                import ctypes
                data_type = ctypes.c_double * (nlocal * 3)
                data_array = data_type.from_address(ptr_address)
                self.current_positions = np.frombuffer(data_array, dtype=np.float64).reshape((nlocal, 3)).astype(np.float32)
                return
        except Exception as e:
            print(f"Array interface method failed: {e}")
        
        # Fallback to safe copy method
        print(f"Rank {self.rank}: Using fallback copy method for data transfer")
        if nlocal <= len(self.position_buffer):
            self.current_positions = self.position_buffer[:nlocal]
            for i in range(nlocal):
                for j in range(3):
                    self.current_positions[i, j] = x[i][j]
        else:
            # Create new array if buffer is too small
            self.current_positions = np.zeros((nlocal, 3), dtype=np.float32)
            for i in range(nlocal):
                for j in range(3):
                    self.current_positions[i, j] = x[i][j]

    def _fast_data_transfer_out(self, f, forces, nlocal):
        """Fast data transfer from numpy to LAMMPS"""
        try:
            # Try direct assignment first
            f_array = np.array(f, copy=False)
            if f_array.shape == (nlocal, 3):
                f_array[:] = forces[:nlocal].astype(f_array.dtype)
                return
        except:
            pass
        
        try:
            # Try array interface with proper pointer handling
            if hasattr(f, '__array_interface__'):
                array_interface = f.__array_interface__
                data_ptr = array_interface['data']
                
                # Handle both tuple (ptr, read_only) and direct pointer cases
                if isinstance(data_ptr, tuple):
                    ptr_address = data_ptr[0]
                else:
                    ptr_address = data_ptr
                
                # Create array from memory address
                import ctypes
                data_type = ctypes.c_double * (nlocal * 3)
                data_array = data_type.from_address(ptr_address)
                f_view = np.frombuffer(data_array, dtype=np.float64).reshape((nlocal, 3))
                f_view[:] = forces[:nlocal].astype(np.float64)
                return
        except Exception as e:
            print(f"Array interface method failed for output: {e}")
        
        # Fallback to safe copy method
        print(f"Rank {self.rank}: Using fallback copy method for force output")
        for i in range(nlocal):
            for j in range(3):
                f[i][j] = float(forces[i, j])

    def _process_chunked(self, nlocal):
        """Process atoms in chunks for memory efficiency"""
        n_chunks = (nlocal + self.chunk_size - 1) // self.chunk_size
        all_forces = np.zeros((nlocal, 3), dtype=np.float32)
        
        for i in range(n_chunks):
            start_idx = i * self.chunk_size
            end_idx = min((i + 1) * self.chunk_size, nlocal)
            chunk_size = end_idx - start_idx
            
            chunk_forces = self._process_single_chunk(chunk_size, start_idx)
            all_forces[start_idx:end_idx] = chunk_forces[:chunk_size]
        
        return all_forces

    def _process_single_chunk(self, chunk_size, offset=0):
        """Process a single chunk of atoms"""
        # Neighbor calculation
        neighbor_start = time.time()
        atoms = self._create_atoms_object_fast(chunk_size, offset)
        all_js, all_offsets = self._compute_neighbors_optimized(atoms)
        neighbor_time = time.time() - neighbor_start
        
        # JAX input preparation
        jax_prep_start = time.time()
        jax_inputs = self._prepare_jax_inputs_vectorized(
            atoms, all_js, all_offsets, chunk_size
        )
        jax_prep_time = time.time() - jax_prep_start
        
        # MTP calculation
        mtp_start = time.time()
        if len(self.assigned_devices) > 1 and chunk_size > 1000:
            energy, forces, stress = self._calculate_mtp_pmap(*jax_inputs)
        else:
            energy, forces, stress = self._calculate_mtp_jit(*jax_inputs)
        mtp_time = time.time() - mtp_start
        
        # Update timing stats
        self._timing_stats['neighbor_calc'].append(neighbor_time)
        self._timing_stats['jax_prep'].append(jax_prep_time)
        self._timing_stats['mtp_calc'].append(mtp_time)
        
        return np.asarray(forces)

    def _create_atoms_object_fast(self, chunk_size, offset=0):
        """Fast atoms object creation for chunks"""
        positions = self.current_positions[offset:offset+chunk_size]
        
        # Simple alternating Ni/Al pattern
        symbols = ['Ni' if i % 4 != 3 else 'Al' for i in range(chunk_size)]
        
        return Atoms(
            symbols=symbols,
            positions=positions,
            cell=[30,30,30],
            pbc=True
        )

    def _compute_neighbors_optimized(self, atoms):
        """Optimized neighbor computation"""
        cutoff = 0.5 * self.mtp_data.max_dist
        
        nl = PrimitiveNeighborList(
            cutoffs=[cutoff] * len(atoms),
            skin=0.5, 
            self_interaction=False,
            bothways=True,
        )
        
        nl.update(atoms.pbc, atoms.cell, atoms.positions)
        
        return self._vectorized_neighbor_processing(nl, atoms)

    def _vectorized_neighbor_processing(self, nl, atoms):
        """Vectorized neighbor list processing"""
        n_atoms = len(atoms)
        if n_atoms == 0:
            return np.empty((0, 0), dtype=int), np.empty((0, 0, 3), dtype=float)
        
        estimated_max_neighbors = min(100, n_atoms)
        
        all_js = np.full((n_atoms, estimated_max_neighbors), -1, dtype=int)
        all_offsets = np.zeros((n_atoms, estimated_max_neighbors, 3), dtype=float)
        
        cell = atoms.cell
        
        for i in range(n_atoms):
            js, offsets = nl.get_neighbors(i)
            if len(js) > 0:
                n_neighbors = min(len(js), estimated_max_neighbors)
                all_js[i, :n_neighbors] = js[:n_neighbors]
                all_offsets[i, :n_neighbors] = offsets[:n_neighbors] @ cell
        
        return all_js, all_offsets

    def _prepare_jax_inputs_vectorized(self, atoms, all_js, all_offsets, chunk_size):
        """Prepare inputs for JAX computation"""
        itypes = np.array([self.species[i % 2] for i in range(chunk_size)], dtype=int)
        
        positions = atoms.positions
        all_rijs = self._compute_distances_vectorized(positions, all_js, all_offsets)
        
        itypes_jax = jnp.asarray(itypes)
        all_js_jax = jnp.asarray(all_js)
        all_rijs_jax = jnp.asarray(all_rijs)
        
        all_jtypes = jnp.where(
            all_js_jax >= 0,
            itypes_jax[jnp.clip(all_js_jax, 0, len(itypes_jax)-1)],
            -1
        )
        
        cell_rank = atoms.cell.rank
        volume = atoms.get_volume()
        
        return itypes_jax, all_js_jax, all_rijs_jax, all_jtypes, cell_rank, volume

    def _compute_distances_vectorized(self, positions, all_js, all_offsets):
        """Vectorized distance computation"""
        valid_mask = all_js >= 0
        
        all_rijs = np.full((*all_js.shape, 3), self.mtp_data.max_dist, dtype=np.float32)
        
        for i in range(len(positions)):
            valid_j = all_js[i][valid_mask[i]]
            if len(valid_j) > 0:
                valid_offsets = all_offsets[i][valid_mask[i]]
                all_rijs[i, valid_mask[i]] = (
                    positions[valid_j] + valid_offsets - positions[i]
                )
        
        return all_rijs

    @partial(jax.jit, static_argnums=(0,))
    def _calculate_mtp_jit(self, itypes, all_js, all_rijs, all_jtypes, cell_rank, volume):
        """JIT-compiled MTP calculation"""
        targets = self.mtp_instance.calculate_jax_new(
            itypes, all_js, all_rijs, all_jtypes, cell_rank, volume, self.params
        )
        return targets['energy'], targets['forces'], targets['stress']

    @partial(pmap, static_broadcasted_argnums=(0,))
    def _calculate_mtp_pmap(self, itypes, all_js, all_rijs, all_jtypes, cell_rank, volume):
        """pmap-parallelized MTP calculation"""
        targets = self.mtp_instance.calculate_jax_new(
            itypes, all_js, all_rijs, all_jtypes, cell_rank, volume, self.params
        )
        return targets['energy'], targets['forces'], targets['stress']

    def _print_performance_snapshot(self, timestep, total_time, nlocal):
        """Print performance statistics"""
        if len(self._timing_stats['total']) < 10:
            return
            
        recent_stats = {k: v[-10:] for k, v in self._timing_stats.items() if v}
        
        print(f"\n=== PERFORMANCE SNAPSHOT (Step {timestep}, Rank {self.rank}) ===")
        print(f"Atoms: {nlocal}, Total Ranks: {self.size}, Devices: {len(self.assigned_devices)}")
        
        for key, times in recent_stats.items():
            if times:
                avg_time = np.mean(times)
                print(f"{key:20s}: {avg_time:.4f}s avg")
        
        atoms_per_second = nlocal / np.mean(recent_stats['total'])
        print(f"Performance: {atoms_per_second:.0f} atoms/second/rank")

    def print_final_performance_summary(self):
        """Print final performance summary"""
        if not self._timing_stats['total']:
            return
            
        print(f"\n{'='*60}")
        print(f"FINAL PERFORMANCE SUMMARY - RANK {self.rank}")
        print(f"{'='*60}")
        
        for key, times in self._timing_stats.items():
            if times:
                times_array = np.array(times)
                print(f"{key:20s}: {np.mean(times_array):.4f}s avg, "
                      f"{np.std(times_array):.4f}s std, "
                      f"{np.sum(times_array):.2f}s total")
        
        if self.rank == 0:
            total_time = np.sum(self._timing_stats['total'])
            total_calls = len(self._timing_stats['total'])
            print(f"\nGlobal Summary:")
            print(f"Total simulation time: {total_time:.2f}s")
            print(f"Total force calls: {total_calls}")
            print(f"Average per call: {total_time/total_calls:.4f}s")


def main():
    """Main simulation function"""
    import gc
    
    # Get rank info from environment
    rank = int(os.environ.get('OMPI_COMM_WORLD_RANK', 
                             os.environ.get('PMI_RANK', 
                             os.environ.get('SLURM_PROCID', '0'))))
    size = int(os.environ.get('OMPI_COMM_WORLD_SIZE', 
                             os.environ.get('PMI_SIZE', 
                             os.environ.get('SLURM_NPROCS', '1'))))
    
    if rank == 0:
        print(f"Starting simulation with {size} MPI ranks (no mpi4py)")
    
    # Optional: Start performance monitoring if available
    try:
        from performance_monitor import PerformanceMonitor
        monitor = PerformanceMonitor(log_interval=30)
        monitor.start_monitoring()
    except ImportError:
        monitor = None
        if rank == 0:
            print("Performance monitor not available, continuing without it")
    
    try:
        # Clean up memory
        gc.collect()
        
        # Initialize LAMMPS
        lmp = lammps()
        
        # LAMMPS setup commands
        lmp.command("units metal")
        lmp.command("dimension 3")
        lmp.command("boundary p p s")
        lmp.command("atom_style atomic")
        lmp.command("read_data equil-400K-2dKWL.data")
        
        lmp.command("mass 1 26.98")  # Al
        lmp.command("mass 2 58.69")  # Ni
        
        lmp.command("variable temperature equal 300")
        lmp.command("reset_timestep 0")
        lmp.command("timestep 0.0005")
        lmp.command("velocity all create ${temperature} 12345")
        
        if rank == 0:
            print("Initializing highly optimized MTP interface...")
        
        # Initialize MTP interface (no MPI communication)
        mtp_interface = HighlyOptimizedMTPLammpsInterface(
            'Ni3Al-12g.mtp',
            chunk_size=50000
        )
        
        if rank == 0:
            print("MTP interface initialized successfully")
        
        # Setup LAMMPS force calculation
        lmp.command("fix ext_force all external pf/callback 1 1")
        lmp.set_fix_external_callback("ext_force", mtp_interface.force_callback)
        lmp.command("fix integrator all nve")
        
        # Neighbor list and communication settings
        lmp.command("comm_modify cutoff 3.0 vel yes")
        lmp.command("neigh_modify delay 0 every 1 check yes page 1000000 one 100000")
        
        # Output settings
        lmp.command("thermo 50")
        lmp.command("thermo_style custom step dt temp vol press pe ke etotal")
        
        if rank == 0:
            lmp.command("dump trajectory all atom 100 trajectory.lammpstrj")
            lmp.command("dump_modify trajectory sort id")
        
        if rank == 0:
            print("Starting optimized simulation for large system...")
        
        # Run simulation
        lmp.command('run 100')
        
        if rank == 0:
            print("Simulation completed successfully")
        
        # Print performance summary
        mtp_interface.print_final_performance_summary()
        
        if rank == 0:
            print("All ranks completed successfully")

    except Exception as e:
        print(f"Rank {rank}: Error during simulation: {e}")
        raise
    finally:
        if monitor:
            monitor.stop_monitoring()


if __name__ == "__main__":
    main()
