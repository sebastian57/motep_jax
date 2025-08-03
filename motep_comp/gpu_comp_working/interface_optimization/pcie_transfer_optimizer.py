#!/usr/bin/env python3
"""
PCIe Transfer Optimizer for JAX-MTP Interface
Optimizes GPU data transfers to minimize PCIe latency
Target: 20ms multiple transfers → 5-8ms batched transfer (2.5-4x improvement)
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Any, Optional, Tuple
import time
import logging
from dataclasses import dataclass
from functools import partial

@dataclass
class TransferConfig:
    """Configuration for PCIe transfer optimization"""
    enable_memory_pinning: bool = True
    enable_async_transfer: bool = True
    batch_all_transfers: bool = True
    transfer_device: Optional[jax.Device] = None
    
class PCIeTransferOptimizer:
    """
    High-performance PCIe transfer optimization for LAMMPS-JAX interface.
    
    Key optimizations:
    1. Batch all GPU transfers into single jax.device_put() call
    2. Enable memory pinning for maximum PCIe bandwidth utilization  
    3. Use pre-compiled transfer functions to eliminate JIT overhead
    4. Monitor transfer rates to detect sub-optimal patterns
    """
    
    def __init__(self, config: TransferConfig):
        self.config = config
        self.device = config.transfer_device or jax.devices()[0]
        
        # Performance monitoring
        self.transfer_stats = {
            'total_transfers': 0,
            'total_transfer_time': 0.0,
            'total_bytes_transferred': 0,
            'peak_bandwidth_gbps': 0.0,
            'avg_bandwidth_gbps': 0.0
        }
        
        # Pre-compiled transfer functions for zero JIT overhead
        self._initialize_transfer_functions()
        
        # Configure JAX for optimal PCIe performance
        self._configure_jax_for_pcie()
        
        logging.info(f"PCIe Transfer Optimizer initialized for device: {self.device}")
        
    def _configure_jax_for_pcie(self):
        """Configure JAX for optimal PCIe transfer performance"""
        
        if self.config.enable_memory_pinning:
            # Enable pinned memory for faster CPU-GPU transfers
            try:
                jax.config.update('jax_cpu_enable_pinned_memory', True)
                logging.info("✅ JAX pinned memory enabled")
            except Exception as e:
                logging.warning(f"⚠️  Could not enable pinned memory: {e}")
        
        # Configure optimal memory allocation 
        jax.config.update('jax_enable_x64', False)  # Use float32/bfloat16 for bandwidth
        
        # Disable pre-allocation for dynamic memory management
        import os
        os.environ.update({
            'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
            'XLA_PYTHON_CLIENT_MEM_FRACTION': '0.9',
            'XLA_PYTHON_CLIENT_ALLOCATOR': 'platform'
        })
        
    def _initialize_transfer_functions(self):
        """Pre-compile transfer functions to eliminate JIT overhead during inference"""
        
        logging.info("Pre-compiling GPU transfer functions...")
        
        # Strategy 1: Single batched transfer function (fastest)
        @jax.jit
        def batched_transfer_to_gpu(data_dict):
            """Transfer all data in a single batch to minimize PCIe latency"""
            return jax.device_put(data_dict, device=self.device)
        
        # Strategy 2: Individual optimized transfers (fallback)
        @jax.jit 
        def single_array_transfer(array):
            """Optimized single array transfer"""
            return jax.device_put(array, device=self.device)
        
        # Strategy 3: Async transfer (experimental)
        @jax.jit
        def async_transfer_to_gpu(data_dict):
            """Asynchronous transfer for overlapping computation"""
            # Note: JAX 0.6.2 async support may be limited
            return jax.device_put(data_dict, device=self.device)
        
        # Pre-compile with dummy data to eliminate first-call JIT overhead
        dummy_data = {
            'itypes': np.zeros(1000, dtype=np.int32),
            'all_js': np.zeros((1000, 100), dtype=np.int32),
            'all_rijs': np.zeros((1000, 100, 3), dtype=np.float32),
            'all_jtypes': np.zeros((1000, 100), dtype=np.int32),
            'cell_rank': np.array(3, dtype=np.int32),
            'volume': np.array(1000.0, dtype=np.float32),
            'natoms_actual': np.array(1000, dtype=np.int32),
            'nneigh_actual': np.array(100, dtype=np.int32)
        }
        
        # Warm up transfer functions
        _ = batched_transfer_to_gpu(dummy_data)
        _ = single_array_transfer(dummy_data['itypes'])
        
        # Store compiled functions
        self.transfer_functions = {
            'batched': batched_transfer_to_gpu,
            'single': single_array_transfer,
            'async': async_transfer_to_gpu
        }
        
        logging.info("✅ GPU transfer functions pre-compiled")
        
    def transfer_to_gpu(self, cpu_data: Dict[str, np.ndarray]) -> Dict[str, jnp.ndarray]:
        """
        Optimized CPU → GPU transfer with batching and performance monitoring.
        
        Performance Strategy:
        1. Batch all arrays into single dictionary transfer
        2. Use pre-compiled JAX function (zero JIT overhead)
        3. Monitor bandwidth to detect sub-optimal patterns
        4. Automatic fallback for compatibility
        """
        
        start_time = time.perf_counter()
        
        # Calculate transfer size for bandwidth monitoring
        total_bytes = sum(
            array.nbytes for array in cpu_data.values() 
            if hasattr(array, 'nbytes')
        )
        
        try:
            if self.config.batch_all_transfers:
                # Strategy 1: Single batched transfer (fastest)
                gpu_data = self.transfer_functions['batched'](cpu_data)
                transfer_method = "batched"
                
            else:
                # Strategy 2: Individual transfers (fallback)
                gpu_data = {}
                for key, array in cpu_data.items():
                    gpu_data[key] = self.transfer_functions['single'](array)
                transfer_method = "individual"
                
        except Exception as e:
            logging.warning(f"⚠️  Optimized transfer failed, using fallback: {e}")
            # Fallback: Standard JAX device_put
            gpu_data = jax.device_put(cpu_data, device=self.device)
            transfer_method = "fallback"
        
        # Wait for transfer completion (for accurate timing)
        if gpu_data:
            # Block until transfer is complete
            first_array = next(iter(gpu_data.values()))
            if hasattr(first_array, 'block_until_ready'):
                first_array.block_until_ready()
        
        transfer_time = time.perf_counter() - start_time
        
        # Update performance statistics
        self._update_transfer_stats(total_bytes, transfer_time, transfer_method)
        
        return gpu_data
    
    def transfer_from_gpu(self, gpu_results: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Optimized GPU → CPU transfer for results extraction.
        
        Performance Strategy:
        1. Extract results in single operation where possible
        2. Convert JAX arrays to NumPy efficiently
        3. Minimize Python object creation
        """
        
        start_time = time.perf_counter()
        
        energy_jax, forces_jax, stress_jax = gpu_results
        
        # Strategy 1: Block and convert in single operation
        try:
            # Ensure GPU computation is complete
            energy_jax.block_until_ready()
            forces_jax.block_until_ready() 
            stress_jax.block_until_ready()
            
            # Convert to CPU arrays (efficient)
            energy = float(energy_jax)
            forces = np.asarray(forces_jax)
            stress = np.asarray(stress_jax)
            
        except Exception as e:
            logging.warning(f"⚠️  Optimized GPU→CPU transfer failed: {e}")
            # Fallback
            energy = float(energy_jax)
            forces = np.array(forces_jax)
            stress = np.array(stress_jax)
        
        transfer_time = time.perf_counter() - start_time
        
        # Update stats (estimate bytes transferred)
        total_bytes = forces.nbytes + stress.nbytes + 8  # 8 bytes for energy
        self._update_transfer_stats(total_bytes, transfer_time, "gpu_to_cpu")
        
        return energy, forces, stress
    
    def _update_transfer_stats(self, bytes_transferred: int, transfer_time: float, method: str):
        """Update transfer performance statistics"""
        
        self.transfer_stats['total_transfers'] += 1
        self.transfer_stats['total_transfer_time'] += transfer_time
        self.transfer_stats['total_bytes_transferred'] += bytes_transferred
        
        # Calculate bandwidth (GB/s)
        if transfer_time > 0:
            bandwidth_gbps = (bytes_transferred / transfer_time) / (1024**3)
            
            if bandwidth_gbps > self.transfer_stats['peak_bandwidth_gbps']:
                self.transfer_stats['peak_bandwidth_gbps'] = bandwidth_gbps
            
            # Update rolling average
            total_time = self.transfer_stats['total_transfer_time']
            total_bytes = self.transfer_stats['total_bytes_transferred']
            self.transfer_stats['avg_bandwidth_gbps'] = (total_bytes / total_time) / (1024**3)
            
            # Log slow transfers for debugging
            if bandwidth_gbps < 2.0 and bytes_transferred > 1024*1024:  # < 2 GB/s for >1MB
                logging.warning(f"⚠️  Slow transfer detected: {bandwidth_gbps:.2f} GB/s using {method}")
    
    def get_transfer_performance(self) -> Dict[str, float]:
        """Get detailed transfer performance metrics"""
        
        stats = self.transfer_stats.copy()
        
        # Add derived metrics
        if stats['total_transfers'] > 0:
            stats['avg_transfer_time_ms'] = (stats['total_transfer_time'] / stats['total_transfers']) * 1000
            stats['avg_transfer_size_mb'] = (stats['total_bytes_transferred'] / stats['total_transfers']) / (1024**2)
        else:
            stats['avg_transfer_time_ms'] = 0.0
            stats['avg_transfer_size_mb'] = 0.0
        
        # PCIe efficiency metrics
        theoretical_pcie_bandwidth = self._get_theoretical_bandwidth()
        if theoretical_pcie_bandwidth > 0:
            stats['pcie_efficiency_percent'] = (stats['avg_bandwidth_gbps'] / theoretical_pcie_bandwidth) * 100
        else:
            stats['pcie_efficiency_percent'] = 0.0
            
        # Speedup estimation
        baseline_transfer_time = 20.0  # ms (current multiple transfer overhead)
        if stats['avg_transfer_time_ms'] > 0:
            stats['estimated_speedup'] = baseline_transfer_time / stats['avg_transfer_time_ms']
        else:
            stats['estimated_speedup'] = 1.0
            
        return stats
    
    def _get_theoretical_bandwidth(self) -> float:
        """Estimate theoretical PCIe bandwidth based on device"""
        
        device_name = str(self.device).lower()
        
        # RTX 3060 Ti typically uses PCIe 4.0 x16
        if 'rtx 3060' in device_name or 'geforce' in device_name:
            return 32.0  # GB/s (PCIe 4.0 x16 theoretical)
        
        # AMD MI300A uses high-bandwidth interconnect
        elif 'mi300' in device_name or 'amd' in device_name:
            return 100.0  # GB/s (estimated for MI300A)
        
        # Generic estimates
        elif 'tesla' in device_name or 'quadro' in device_name:
            return 32.0  # GB/s (PCIe 4.0 x16)
        
        else:
            return 16.0  # GB/s (conservative PCIe 3.0 x16)
    
    def benchmark_transfer_patterns(self, test_sizes: list = None) -> Dict[str, Dict[str, float]]:
        """
        Benchmark different transfer patterns to optimize configuration.
        
        Useful for identifying optimal transfer strategies for your hardware.
        """
        
        if test_sizes is None:
            test_sizes = [1000, 5000, 10000, 20000]  # atoms
        
        benchmark_results = {}
        
        for natoms in test_sizes:
            nneigh = min(natoms // 10, 200)  # Realistic neighbor count
            
            # Create test data
            test_data = {
                'itypes': np.random.randint(0, 4, natoms, dtype=np.int32),
                'all_js': np.random.randint(0, natoms, (natoms, nneigh), dtype=np.int32),
                'all_rijs': np.random.randn(natoms, nneigh, 3).astype(np.float32),
                'all_jtypes': np.random.randint(0, 4, (natoms, nneigh), dtype=np.int32),
                'cell_rank': np.array(3, dtype=np.int32),
                'volume': np.array(1000.0, dtype=np.float32),
                'natoms_actual': np.array(natoms, dtype=np.int32),
                'nneigh_actual': np.array(nneigh, dtype=np.int32)
            }
            
            # Benchmark batched vs individual transfers
            results = {}
            
            # Test batched transfer
            start_time = time.perf_counter()
            for _ in range(10):  # Average over multiple runs
                gpu_data = self.transfer_functions['batched'](test_data)
                gpu_data['itypes'].block_until_ready()
            batched_time = (time.perf_counter() - start_time) / 10
            
            # Test individual transfers
            start_time = time.perf_counter()
            for _ in range(10):
                for key, array in test_data.items():
                    gpu_array = self.transfer_functions['single'](array)
                    gpu_array.block_until_ready()
            individual_time = (time.perf_counter() - start_time) / 10
            
            results[f'{natoms}_atoms'] = {
                'batched_time_ms': batched_time * 1000,
                'individual_time_ms': individual_time * 1000,
                'speedup': individual_time / batched_time if batched_time > 0 else 1.0,
                'data_size_mb': sum(arr.nbytes for arr in test_data.values()) / (1024**2)
            }
            
            benchmark_results.update(results)
        
        return benchmark_results
    
    def reset_stats(self):
        """Reset transfer performance statistics"""
        self.transfer_stats = {
            'total_transfers': 0,
            'total_transfer_time': 0.0,
            'total_bytes_transferred': 0,
            'peak_bandwidth_gbps': 0.0,
            'avg_bandwidth_gbps': 0.0
        }
