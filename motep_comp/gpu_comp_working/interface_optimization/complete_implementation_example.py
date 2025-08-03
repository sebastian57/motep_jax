#!/usr/bin/env python3
"""
Complete Implementation Example: JAX-MTP CPU Interface Optimization
==================================================================================

This file demonstrates the complete integration of CPU optimization with your 
existing JAX-MTP system. It shows how to:

1. Set up the optimization system
2. Integrate with existing jax_pad_pmap_mixed_2.py
3. Load and use your .bin files  
4. Monitor performance improvements
5. Validate correctness

Compatible with JAX 0.6.2 and your existing CUDA setup on RTX 3060 Ti.
"""

import os
import sys
import time
import logging
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add your project directories to path
project_root = Path(__file__).parent
sys.path.append(str(project_root / "motep_comp" / "gpu_comp_working"))
sys.path.append(str(project_root))

class JAXMTPOptimizedSystem:
    """
    Complete optimized JAX-MTP system integrating all CPU optimizations
    with your existing GPU-optimized implementation.
    """
    
    def __init__(self, 
                 max_atoms: int = 16384,
                 max_neighbors: int = 200,
                 enable_optimizations: bool = True):
        
        self.max_atoms = max_atoms
        self.max_neighbors = max_neighbors
        self.enable_optimizations = enable_optimizations
        
        logger.info(f"Initializing JAX-MTP system for {max_atoms} atoms, {max_neighbors} neighbors")
        logger.info(f"JAX version: {jax.__version__}")
        logger.info(f"Available devices: {jax.devices()}")
        
        # Initialize optimization components
        self.optimizer = None
        self.baseline_function = None
        self.mtp_params = None
        
        # Performance tracking
        self.performance_history = []
        self.baseline_times = []
        self.optimized_times = []
        
        self._setup_system()
    
    def _setup_system(self):
        """Set up the optimization system"""
        
        if self.enable_optimizations:
            try:
                # Import optimization system
                from ultimate_interface_optimizer import create_ultimate_optimizer
                
                self.optimizer = create_ultimate_optimizer(
                    self.max_atoms, 
                    self.max_neighbors,
                    enable_all_optimizations=True
                )
                
                logger.info("✅ Ultimate interface optimizer initialized")
                
            except ImportError as e:
                logger.error(f"❌ Failed to import optimization system: {e}")
                logger.info("Using baseline implementation only")
                self.enable_optimizations = False
        
        # Always set up baseline for comparison
        self._setup_baseline_function()
    
    def _setup_baseline_function(self):
        """Set up baseline JAX function for comparison"""
        
        try:
            # Import your existing optimized JAX implementation
            from jax_pad_pmap_mixed_2 import calc_energy_forces_stress_padded_simple_ultimate
            self.baseline_function = calc_energy_forces_stress_padded_simple_ultimate
            logger.info("✅ Baseline JAX function loaded")
            
        except ImportError:
            try:
                # Fallback to other implementations
                from jax_pad_strategy2_mixed import calc_energy_forces_stress_padded_simple_strategy2_mixed
                self.baseline_function = calc_energy_forces_stress_padded_simple_strategy2_mixed
                logger.info("✅ Strategy 2 JAX function loaded as baseline")
                
            except ImportError:
                logger.error("❌ Could not load any JAX implementation")
                raise
    
    def load_mtp_parameters(self, mtp_file_path: str):
        """Load MTP parameters from .mtp file"""
        
        try:
            # Import your MTP loading utilities
            sys.path.append(str(project_root / "motep_original_files"))
            from mtp import MTPData
            
            mtp_data = MTPData(mtp_file_path)
            
            # Convert to format expected by JAX functions
            self.mtp_params = {
                'species': np.array(mtp_data.species, dtype=np.int32),
                'scaling': np.array(mtp_data.scaling, dtype=np.float32),
                'min_dist': np.array(mtp_data.min_dist, dtype=np.float32),
                'max_dist': np.array(mtp_data.max_dist, dtype=np.float32),
                'species_coeffs': mtp_data.species_coeffs,
                'moment_coeffs': mtp_data.moment_coeffs,
                'radial_coeffs': mtp_data.radial_coeffs,
                'execution_order': mtp_data.execution_order,
                'scalar_contractions': mtp_data.scalar_contractions
            }
            
            logger.info(f"✅ MTP parameters loaded from {mtp_file_path}")
            
            # Load into optimizer if available
            if self.optimizer:
                self.optimizer.load_jax_function("loaded_mtp", self.mtp_params)
                logger.info("✅ MTP parameters loaded into optimizer")
                
        except Exception as e:
            logger.error(f"❌ Failed to load MTP parameters: {e}")
            
            # Create mock parameters for testing
            self.mtp_params = self._create_mock_mtp_params()
            logger.warning("⚠️  Using mock MTP parameters for testing")
    
    def _create_mock_mtp_params(self) -> Dict[str, Any]:
        """Create mock MTP parameters for testing when .mtp file unavailable"""
        
        return {
            'species': np.array([1, 2], dtype=np.int32),
            'scaling': np.array(1.0, dtype=np.float32),
            'min_dist': np.array(0.5, dtype=np.float32),
            'max_dist': np.array(5.0, dtype=np.float32),
            'species_coeffs': np.random.randn(4, 10, 5).astype(np.float32),
            'moment_coeffs': np.random.randn(4, 15, 8).astype(np.float32),
            'radial_coeffs': np.random.randn(4, 10, 5, 20).astype(np.float32),
            'execution_order': [1, 2, 3],
            'scalar_contractions': [True, False, True]
        }
    
    def create_test_system(self, natoms: int, nneigh: int) -> Dict[str, np.ndarray]:
        """Create realistic test system for benchmarking"""
        
        np.random.seed(42)  # Reproducible tests
        
        # Realistic atomic system
        system_data = {
            'itypes': np.random.randint(0, len(self.mtp_params['species']), natoms, dtype=np.int32),
            'all_js': np.random.randint(0, natoms, (natoms, nneigh), dtype=np.int32),
            'all_rijs': (np.random.randn(natoms, nneigh, 3) * 2.5).astype(np.float32),
            'all_jtypes': np.random.randint(0, len(self.mtp_params['species']), (natoms, nneigh), dtype=np.int32),
            'cell_rank': 3,
            'volume': 1000.0 + natoms * 0.1,
            'natoms_actual': natoms,
            'nneigh_actual': nneigh
        }
        
        # Ensure neighbor distances are realistic (0.5 to 5.0 Angstrom)
        distances = np.linalg.norm(system_data['all_rijs'], axis=2)
        system_data['all_rijs'] = system_data['all_rijs'] * (
            (self.mtp_params['max_dist'] - self.mtp_params['min_dist']) / 
            np.maximum(distances[:, :, np.newaxis], 0.1) + self.mtp_params['min_dist']
        )
        
        return system_data
    
    def benchmark_baseline(self, system_data: Dict[str, np.ndarray], 
                          num_iterations: int = 20) -> Tuple[Tuple[float, np.ndarray, np.ndarray], float]:
        """Benchmark baseline implementation"""
        
        logger.info(f"Benchmarking baseline implementation ({num_iterations} iterations)...")
        
        # Warm up
        for _ in range(5):
            _ = self.baseline_function(
                system_data['itypes'],
                system_data['all_js'],
                system_data['all_rijs'],
                system_data['all_jtypes'],
                system_data['cell_rank'],
                system_data['volume'],
                system_data['natoms_actual'],
                system_data['nneigh_actual'],
                self.mtp_params['species'],
                self.mtp_params['scaling'],
                self.mtp_params['min_dist'],
                self.mtp_params['max_dist'],
                self.mtp_params['species_coeffs'],
                self.mtp_params['moment_coeffs'],
                self.mtp_params['radial_coeffs'],
                self.mtp_params['execution_order'],
                self.mtp_params['scalar_contractions']
            )
        
        # Benchmark
        times = []
        for i in range(num_iterations):
            start_time = time.perf_counter()
            
            results = self.baseline_function(
                system_data['itypes'],
                system_data['all_js'],
                system_data['all_rijs'],
                system_data['all_jtypes'],
                system_data['cell_rank'],
                system_data['volume'],
                system_data['natoms_actual'],
                system_data['nneigh_actual'],
                self.mtp_params['species'],
                self.mtp_params['scaling'],
                self.mtp_params['min_dist'],
                self.mtp_params['max_dist'],
                self.mtp_params['species_coeffs'],
                self.mtp_params['moment_coeffs'],
                self.mtp_params['radial_coeffs'],
                self.mtp_params['execution_order'],
                self.mtp_params['scalar_contractions']
            )
            
            elapsed = time.perf_counter() - start_time
            times.append(elapsed)
            
            if i % 5 == 0:
                logger.info(f"  Baseline iteration {i+1}/{num_iterations}: {elapsed*1000:.1f} ms")
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        logger.info(f"✅ Baseline benchmark complete: {avg_time*1000:.1f} ± {std_time*1000:.1f} ms")
        
        return results, avg_time
    
    def benchmark_optimized(self, system_data: Dict[str, np.ndarray],
                           num_iterations: int = 20) -> Tuple[Tuple[float, np.ndarray, np.ndarray], float]:
        """Benchmark optimized implementation"""
        
        if not self.optimizer:
            raise RuntimeError("Optimizer not available")
        
        logger.info(f"Benchmarking optimized implementation ({num_iterations} iterations)...")
        
        # Warm up
        for _ in range(5):
            _ = self.optimizer.compute_energy_forces_stress(
                system_data['itypes'],
                system_data['all_js'],
                system_data['all_rijs'],
                system_data['all_jtypes'],
                system_data['cell_rank'],
                system_data['volume'],
                system_data['natoms_actual'],
                system_data['nneigh_actual'],
                self.mtp_params
            )
        
        # Benchmark
        times = []
        for i in range(num_iterations):
            start_time = time.perf_counter()
            
            results = self.optimizer.compute_energy_forces_stress(
                system_data['itypes'],
                system_data['all_js'],
                system_data['all_rijs'],
                system_data['all_jtypes'],
                system_data['cell_rank'],
                system_data['volume'],
                system_data['natoms_actual'],
                system_data['nneigh_actual'],
                self.mtp_params
            )
            
            elapsed = time.perf_counter() - start_time
            times.append(elapsed)
            
            if i % 5 == 0:
                logger.info(f"  Optimized iteration {i+1}/{num_iterations}: {elapsed*1000:.1f} ms")
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        logger.info(f"✅ Optimized benchmark complete: {avg_time*1000:.1f} ± {std_time*1000:.1f} ms")
        
        return results, avg_time
    
    def validate_correctness(self, baseline_results: Tuple, optimized_results: Tuple,
                           tolerance: float = 1e-6) -> bool:
        """Validate that optimized results match baseline"""
        
        logger.info("Validating correctness...")
        
        base_energy, base_forces, base_stress = baseline_results
        opt_energy, opt_forces, opt_stress = optimized_results
        
        # Energy validation
        energy_error = abs(opt_energy - base_energy)
        energy_valid = energy_error < tolerance
        
        # Forces validation
        forces_error = np.max(np.abs(opt_forces - base_forces))
        forces_valid = forces_error < tolerance
        
        # Stress validation
        stress_error = np.max(np.abs(opt_stress - base_stress))
        stress_valid = stress_error < tolerance
        
        is_valid = energy_valid and forces_valid and stress_valid
        
        logger.info(f"Energy error: {energy_error:.2e} ({'✅ PASS' if energy_valid else '❌ FAIL'})")
        logger.info(f"Forces error: {forces_error:.2e} ({'✅ PASS' if forces_valid else '❌ FAIL'})")
        logger.info(f"Stress error: {stress_error:.2e} ({'✅ PASS' if stress_valid else '❌ FAIL'})")
        logger.info(f"Overall validation: {'✅ PASS' if is_valid else '❌ FAIL'}")
        
        return is_valid
    
    def run_comprehensive_test(self, test_systems: list = None) -> Dict[str, Any]:
        """Run comprehensive performance test"""
        
        if test_systems is None:
            test_systems = [
                (1000, 100),   # Small system
                (2500, 150),   # Medium system  
                (5000, 200),   # Large system
                (7500, 200),   # Very large system
                (10000, 200),  # Maximum system
            ]
        
        logger.info("=" * 60)
        logger.info("RUNNING COMPREHENSIVE OPTIMIZATION TEST")
        logger.info("=" * 60)
        
        results = {}
        
        for natoms, nneigh in test_systems:
            test_name = f"{natoms}_atoms_{nneigh}_neighbors"
            logger.info(f"\nTesting {test_name}...")
            
            try:
                # Create test system
                system_data = self.create_test_system(natoms, nneigh)
                
                # Benchmark baseline
                baseline_results, baseline_time = self.benchmark_baseline(system_data)
                
                # Benchmark optimized (if available)
                if self.optimizer:
                    optimized_results, optimized_time = self.benchmark_optimized(system_data)
                    
                    # Validate correctness
                    is_valid = self.validate_correctness(baseline_results, optimized_results)
                    
                    # Calculate performance metrics
                    speedup = baseline_time / optimized_time if optimized_time > 0 else 1.0
                    time_saved_ms = (baseline_time - optimized_time) * 1000
                    
                    results[test_name] = {
                        'natoms': natoms,
                        'nneigh': nneigh,
                        'baseline_time_ms': baseline_time * 1000,
                        'optimized_time_ms': optimized_time * 1000,
                        'speedup': speedup,
                        'time_saved_ms': time_saved_ms,
                        'is_valid': is_valid,
                        'status': 'success'
                    }
                    
                    logger.info(f"📊 Results: {speedup:.1f}x speedup, {time_saved_ms:.1f}ms saved per call")
                    
                else:
                    results[test_name] = {
                        'natoms': natoms,
                        'nneigh': nneigh,
                        'baseline_time_ms': baseline_time * 1000,
                        'status': 'baseline_only'
                    }
                
            except Exception as e:
                logger.error(f"❌ Test {test_name} failed: {e}")
                results[test_name] = {
                    'natoms': natoms,
                    'nneigh': nneigh,
                    'error': str(e),
                    'status': 'failed'
                }
        
        return results
    
    def generate_performance_report(self, results: Dict[str, Any]):
        """Generate performance report"""
        
        logger.info("\n" + "=" * 60)
        logger.info("PERFORMANCE OPTIMIZATION REPORT")
        logger.info("=" * 60)
        
        successful_tests = [r for r in results.values() if r.get('status') == 'success']
        
        if successful_tests:
            speedups = [r['speedup'] for r in successful_tests]
            time_savings = [r['time_saved_ms'] for r in successful_tests]
            
            avg_speedup = np.mean(speedups)
            best_speedup = np.max(speedups)
            avg_time_saved = np.mean(time_savings)
            
            logger.info(f"✅ Successful tests: {len(successful_tests)}/{len(results)}")
            logger.info(f"📈 Average speedup: {avg_speedup:.1f}x")
            logger.info(f"🚀 Best speedup: {best_speedup:.1f}x")
            logger.info(f"⏱️  Average time saved: {avg_time_saved:.1f} ms per call")
            logger.info(f"🎯 Target achieved: {'✅ YES' if avg_speedup >= 2.0 else '❌ NO'}")
            
            # Detailed breakdown
            logger.info("\nDetailed Results:")
            for test_name, result in results.items():
                if result.get('status') == 'success':
                    logger.info(f"  {test_name}: {result['speedup']:.1f}x speedup "
                              f"({result['baseline_time_ms']:.1f}ms → {result['optimized_time_ms']:.1f}ms)")
        else:
            logger.warning("❌ No successful optimization tests")
        
        # Get optimizer performance breakdown if available
        if self.optimizer:
            try:
                perf_stats = self.optimizer.get_performance_summary()
                
                logger.info("\nOptimization Breakdown:")
                logger.info(f"  Buffer optimization speedup: {perf_stats.get('buffer_speedup', 1.0):.1f}x")
                logger.info(f"  PCIe optimization speedup: {perf_stats.get('pcie_speedup', 1.0):.1f}x")
                logger.info(f"  GIL optimization speedup: {perf_stats.get('gil_speedup', 1.0):.1f}x")
                
            except Exception as e:
                logger.warning(f"Could not get detailed performance stats: {e}")

# Main execution function
def main():
    """Main function demonstrating complete optimization system"""
    
    logger.info("JAX-MTP CPU Interface Optimization - Complete Implementation")
    logger.info(f"JAX version: {jax.__version__}")
    logger.info(f"NumPy version: {np.__version__}")
    
    # Initialize system
    system = JAXMTPOptimizedSystem(
        max_atoms=16384,
        max_neighbors=200,
        enable_optimizations=True
    )
    
    # Load MTP parameters (replace with your actual .mtp file)
    mtp_file = "path/to/your/potential.mtp"  # Update this path
    if os.path.exists(mtp_file):
        system.load_mtp_parameters(mtp_file)
    else:
        logger.warning(f"MTP file {mtp_file} not found, using mock parameters")
        system.load_mtp_parameters("")  # Will create mock parameters
    
    # Run comprehensive test
    test_results = system.run_comprehensive_test()
    
    # Generate report
    system.generate_performance_report(test_results)
    
    # Save results
    import json
    with open("optimization_results.json", 'w') as f:
        json.dump(test_results, f, indent=2)
    
    logger.info("\n✅ Complete optimization test finished!")
    logger.info("Results saved to optimization_results.json")

# Integration example for C++ usage
def create_optimized_interface_for_cpp():
    """
    Create optimized interface function for C++ integration.
    This is the function that gets called from pair_jax_mtp_direct.cpp
    """
    
    # Global optimizer instance (initialized once)
    global _global_optimizer
    
    if '_global_optimizer' not in globals():
        from ultimate_interface_optimizer import create_ultimate_optimizer
        _global_optimizer = create_ultimate_optimizer(16384, 200)
        logger.info("Global optimizer created for C++ integration")
    
    def optimized_compute_for_cpp(itypes, all_js, all_rijs, all_jtypes,
                                 cell_rank, volume, natoms_actual, nneigh_actual,
                                 mtp_params_dict):
        """
        Optimized compute function for C++ calling.
        This matches the signature expected by your modified pair_jax_mtp_direct.cpp
        """
        
        return _global_optimizer.compute_energy_forces_stress(
            itypes, all_js, all_rijs, all_jtypes,
            cell_rank, volume, natoms_actual, nneigh_actual,
            mtp_params_dict
        )
    
    return optimized_compute_for_cpp

if __name__ == "__main__":
    main()
