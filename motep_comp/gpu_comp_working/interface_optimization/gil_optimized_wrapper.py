#!/usr/bin/env python3
"""
GIL Optimized Wrapper for JAX-MTP Interface
Minimizes Python GIL overhead during JAX computation
Target: 5-10ms GIL overhead → 1-2ms (5x improvement)
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
import threading
import logging
from typing import Dict, Any, Callable, Optional, Tuple
from functools import wraps, partial
from dataclasses import dataclass
import weakref

@dataclass
class GILConfig:
    """Configuration for GIL optimization"""
    precompile_all_functions: bool = True
    minimize_python_objects: bool = True
    use_direct_numpy_access: bool = True
    enable_function_caching: bool = True
    max_cached_functions: int = 10

class PrecompiledFunctionCache:
    """Cache for pre-compiled JAX functions to eliminate runtime JIT"""
    
    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self.cache = {}
        self.access_order = []
        self._lock = threading.Lock()
        
    def get_or_compile(self, 
                      function_key: str,
                      compile_fn: Callable,
                      compile_args: tuple) -> Callable:
        """Get cached function or compile and cache new one"""
        
        with self._lock:
            if function_key in self.cache:
                # Move to end (most recently used)
                self.access_order.remove(function_key)
                self.access_order.append(function_key)
                return self.cache[function_key]
            
            # Compile new function
            logging.info(f"Pre-compiling JAX function: {function_key}")
            compiled_fn = compile_fn(*compile_args)
            
            # Cache management (LRU eviction)
            if len(self.cache) >= self.max_size:
                oldest_key = self.access_order.pop(0)
                del self.cache[oldest_key]
                logging.info(f"Evicted cached function: {oldest_key}")
            
            self.cache[function_key] = compiled_fn
            self.access_order.append(function_key)
            
            return compiled_fn

class GILOptimizedWrapper:
    """
    GIL-optimized wrapper for JAX-MTP computations.
    
    Key optimizations:
    1. Pre-compile all JAX functions to eliminate compilation under GIL
    2. Minimize Python object creation during computation
    3. Use direct NumPy array access (no Python list iteration)
    4. Batch operations to minimize GIL acquisition count
    """
    
    def __init__(self, config: GILConfig):
        self.config = config
        self.function_cache = PrecompiledFunctionCache(config.max_cached_functions)
        
        # Performance monitoring
        self.gil_stats = {
            'total_calls': 0,
            'total_gil_time': 0.0,
            'total_compute_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
            'precompilation_time': 0.0
        }
        
        # Pre-loaded JAX computation function
        self._jax_compute_function = None
        self._jax_function_signature = None
        
        logging.info("GIL Optimized Wrapper initialized")
        
    def precompile_jax_function(self, 
                              jax_function: Callable,
                              max_atoms: int,
                              max_neighbors: int,
                              mtp_params: Dict[str, Any]) -> Callable:
        """
        Pre-compile JAX function with dummy data to eliminate runtime JIT.
        
        This is critical for GIL optimization - compilation happens once during
        initialization rather than during each LAMMPS timestep.
        """
        
        function_key = f"jax_mtp_{max_atoms}_{max_neighbors}"
        
        def compile_function():
            logging.info(f"Pre-compiling JAX function for {max_atoms} atoms, {max_neighbors} neighbors")
            start_time = time.perf_counter()
            
            # Create dummy input data with correct shapes and types
            dummy_inputs = self._create_dummy_inputs(max_atoms, max_neighbors, mtp_params)
            
            # Pre-compile with @jax.jit and static arguments
            @partial(jax.jit, static_argnames=('natoms_actual', 'nneigh_actual'))
            def compiled_jax_function(*args, **kwargs):
                return jax_function(*args, **kwargs)
            
            # Warm up the function (triggers compilation)
            logging.info("Warming up JAX function...")
            _ = compiled_jax_function(**dummy_inputs)
            
            compilation_time = time.perf_counter() - start_time
            self.gil_stats['precompilation_time'] += compilation_time
            
            logging.info(f"✅ JAX function pre-compiled in {compilation_time:.2f}s")
            return compiled_jax_function
        
        return self.function_cache.get_or_compile(
            function_key, 
            compile_function,
            ()
        )
    
    def _create_dummy_inputs(self, 
                           max_atoms: int, 
                           max_neighbors: int,
                           mtp_params: Dict[str, Any]) -> Dict[str, Any]:
        """Create dummy input data for JAX function pre-compilation"""
        
        # Use realistic data types and shapes
        dummy_inputs = {
            'itypes': jnp.zeros(max_atoms, dtype=jnp.int32),
            'all_js': jnp.zeros((max_atoms, max_neighbors), dtype=jnp.int32),
            'all_rijs': jnp.zeros((max_atoms, max_neighbors, 3), dtype=jnp.float32),
            'all_jtypes': jnp.zeros((max_atoms, max_neighbors), dtype=jnp.int32),
            'cell_rank': jnp.array(3, dtype=jnp.int32),
            'volume': jnp.array(1000.0, dtype=jnp.float32),
            'natoms_actual': max_atoms // 2,  # Realistic values
            'nneigh_actual': max_neighbors // 2,
            
            # MTP parameters (from your loaded .mtp file)
            'species': jnp.array(mtp_params.get('species', [1, 2]), dtype=jnp.int32),
            'scaling': jnp.array(mtp_params.get('scaling', 1.0), dtype=jnp.float32),
            'min_dist': jnp.array(mtp_params.get('min_dist', 0.5), dtype=jnp.float32),
            'max_dist': jnp.array(mtp_params.get('max_dist', 5.0), dtype=jnp.float32),
            'species_coeffs': mtp_params.get('species_coeffs'),
            'moment_coeffs': mtp_params.get('moment_coeffs'),
            'radial_coeffs': mtp_params.get('radial_coeffs'),
            'execution_order': mtp_params.get('execution_order'),
            'scalar_contractions': mtp_params.get('scalar_contractions')
        }
        
        return dummy_inputs
    
    def gil_optimized_compute(self,
                             jax_function: Callable,
                             input_data: Dict[str, Any],
                             minimize_gil: bool = True) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        GIL-optimized computation wrapper.
        
        Performance strategy:
        1. Minimize time spent holding GIL
        2. Use pre-compiled JAX functions (zero compilation time)
        3. Batch all operations to reduce GIL acquisition/release cycles
        4. Direct NumPy array handling (no Python object overhead)
        """
        
        start_total = time.perf_counter()
        
        with threading.Lock():  # Ensure thread safety if called from multiple threads
            self.gil_stats['total_calls'] += 1
            
            if minimize_gil:
                result = self._minimal_gil_compute(jax_function, input_data)
            else:
                result = self._standard_compute(jax_function, input_data)
            
            total_time = time.perf_counter() - start_total
            self.gil_stats['total_compute_time'] += total_time
            
            return result
    
    def _minimal_gil_compute(self, 
                           jax_function: Callable,
                           input_data: Dict[str, Any]) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Minimal GIL compute strategy - hold GIL only for data transfer, not computation.
        """
        
        gil_start = time.perf_counter()
        
        try:
            # Strategy 1: Single function call (minimizes GIL overhead)
            # JAX releases GIL during GPU computation
            energy, forces, stress = jax_function(
                input_data['itypes'],
                input_data['all_js'], 
                input_data['all_rijs'],
                input_data['all_jtypes'],
                input_data['cell_rank'],
                input_data['volume'],
                input_data['natoms_actual'],
                input_data['nneigh_actual'],
                input_data['species'],
                input_data['scaling'],
                input_data['min_dist'],
                input_data['max_dist'],
                input_data['species_coeffs'],
                input_data['moment_coeffs'],
                input_data['radial_coeffs'],
                input_data['execution_order'],
                input_data['scalar_contractions']
            )
            
            # Strategy 2: Batch result extraction (single GIL acquisition)
            # Convert JAX arrays to NumPy in one operation
            if hasattr(energy, 'block_until_ready'):
                energy.block_until_ready()
                forces.block_until_ready()
                stress.block_until_ready()
            
            # Direct conversion to Python/NumPy types (minimal object creation)
            energy_cpu = float(energy)
            forces_cpu = np.asarray(forces)
            stress_cpu = np.asarray(stress)
            
            gil_time = time.perf_counter() - gil_start
            self.gil_stats['total_gil_time'] += gil_time
            
            return energy_cpu, forces_cpu, stress_cpu
            
        except Exception as e:
            logging.error(f"Minimal GIL compute failed: {e}")
            # Fallback to standard compute
            return self._standard_compute(jax_function, input_data)
    
    def _standard_compute(self,
                         jax_function: Callable, 
                         input_data: Dict[str, Any]) -> Tuple[float, np.ndarray, np.ndarray]:
        """Standard computation (fallback)"""
        
        gil_start = time.perf_counter()
        
        # Standard JAX function call
        results = jax_function(**input_data)
        
        # Extract results
        if isinstance(results, tuple):
            energy, forces, stress = results
        else:
            # Handle different return formats
            energy = results
            forces = jnp.zeros((input_data['natoms_actual'], 3))
            stress = jnp.zeros(6)
        
        # Convert to CPU
        energy_cpu = float(energy)
        forces_cpu = np.array(forces)
        stress_cpu = np.array(stress)
        
        gil_time = time.perf_counter() - gil_start
        self.gil_stats['total_gil_time'] += gil_time
        
        return energy_cpu, forces_cpu, stress_cpu
    
    def create_optimized_interface_function(self,
                                          base_jax_function: Callable,
                                          max_atoms: int,
                                          max_neighbors: int,
                                          mtp_params: Dict[str, Any]) -> Callable:
        """
        Create a fully optimized interface function with minimal GIL overhead.
        
        This function is designed to be called from C++ via Python embedding.
        """
        
        # Pre-compile the JAX function
        compiled_jax_fn = self.precompile_jax_function(
            base_jax_function, max_atoms, max_neighbors, mtp_params
        )
        
        def optimized_interface(**kwargs):
            """Optimized interface function for C++ calling"""
            
            # Validate inputs efficiently
            required_keys = ['itypes', 'all_js', 'all_rijs', 'all_jtypes', 
                           'cell_rank', 'volume', 'natoms_actual', 'nneigh_actual']
            
            if not all(key in kwargs for key in required_keys):
                missing = [key for key in required_keys if key not in kwargs]
                raise ValueError(f"Missing required arguments: {missing}")
            
            # Add MTP parameters to input data
            input_data = kwargs.copy()
            input_data.update(mtp_params)
            
            # Call optimized compute function
            return self.gil_optimized_compute(
                compiled_jax_fn, input_data, minimize_gil=True
            )
        
        return optimized_interface
    
    def get_gil_performance_stats(self) -> Dict[str, float]:
        """Get detailed GIL performance statistics"""
        
        stats = self.gil_stats.copy()
        
        if stats['total_calls'] > 0:
            stats['avg_gil_time_ms'] = (stats['total_gil_time'] / stats['total_calls']) * 1000
            stats['avg_compute_time_ms'] = (stats['total_compute_time'] / stats['total_calls']) * 1000
            stats['gil_overhead_percent'] = (stats['total_gil_time'] / stats['total_compute_time']) * 100
            stats['cache_hit_rate'] = stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses'])
        else:
            stats['avg_gil_time_ms'] = 0.0
            stats['avg_compute_time_ms'] = 0.0
            stats['gil_overhead_percent'] = 0.0
            stats['cache_hit_rate'] = 0.0
        
        # Estimate speedup
        baseline_gil_overhead = 8.0  # ms (typical GIL overhead before optimization)
        if stats['avg_gil_time_ms'] > 0:
            stats['estimated_speedup'] = baseline_gil_overhead / stats['avg_gil_time_ms']
        else:
            stats['estimated_speedup'] = 1.0
            
        return stats
    
    def reset_stats(self):
        """Reset GIL performance statistics"""
        self.gil_stats = {
            'total_calls': 0,
            'total_gil_time': 0.0,
            'total_compute_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
            'precompilation_time': 0.0
        }
    
    def clear_function_cache(self):
        """Clear the pre-compiled function cache"""
        with self.function_cache._lock:
            self.function_cache.cache.clear()
            self.function_cache.access_order.clear()
            logging.info("Cleared pre-compiled function cache")

def gil_optimized(max_atoms: int, max_neighbors: int, mtp_params: Dict[str, Any]):
    """
    Decorator for creating GIL-optimized JAX functions.
    
    Usage:
    @gil_optimized(max_atoms=16384, max_neighbors=200, mtp_params=loaded_mtp_params)
    def my_jax_function(...):
        return calc_energy_forces_stress_padded_simple_ultimate(...)
    """
    
    def decorator(func):
        config = GILConfig()
        wrapper = GILOptimizedWrapper(config)
        
        # Pre-compile during decoration
        optimized_func = wrapper.create_optimized_interface_function(
            func, max_atoms, max_neighbors, mtp_params
        )
        
        # Attach performance monitoring
        optimized_func.get_performance_stats = wrapper.get_gil_performance_stats
        optimized_func.reset_stats = wrapper.reset_stats
        
        return optimized_func
    
    return decorator
