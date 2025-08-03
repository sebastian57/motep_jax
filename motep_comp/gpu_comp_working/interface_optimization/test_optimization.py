#!/usr/bin/env python3
"""Test script for CPU interface optimization"""

import sys
import os
import time
import numpy as np

# Add current directory to path
sys.path.insert(0, '.')

def test_import():
    """Test importing optimization modules"""
    print("Testing imports...")
    
    try:
        from cpu_buffer_manager import CPUBufferManager, BufferConfig
        print("  ✅ cpu_buffer_manager imported")
    except ImportError as e:
        print(f"  ❌ cpu_buffer_manager import failed: {e}")
        return False
    
    try:
        from pcie_transfer_optimizer import PCIeTransferOptimizer, TransferConfig
        print("  ✅ pcie_transfer_optimizer imported")
    except ImportError as e:
        print(f"  ❌ pcie_transfer_optimizer import failed: {e}")
        return False
    
    try:
        from gil_optimized_wrapper import GILOptimizedWrapper, GILConfig
        print("  ✅ gil_optimized_wrapper imported")
    except ImportError as e:
        print(f"  ❌ gil_optimized_wrapper import failed: {e}")
        return False
    
    try:
        from ultimate_interface_optimizer import create_ultimate_optimizer
        print("  ✅ ultimate_interface_optimizer imported")
    except ImportError as e:
        print(f"  ❌ ultimate_interface_optimizer import failed: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality of optimization system"""
    print("\nTesting basic functionality...")
    
    try:
        from ultimate_interface_optimizer import create_ultimate_optimizer
        
        # Create optimizer
        optimizer = create_ultimate_optimizer(1024, 100)
        print("  ✅ Optimizer created successfully")
        
        # Create mock MTP parameters
        mock_params = {
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
        
        # Load mock function
        optimizer.load_jax_function("mock_function", mock_params)
        print("  ✅ Mock function loaded")
        
        # Get performance stats
        stats = optimizer.get_performance_summary()
        print(f"  ✅ Performance stats: {len(stats)} metrics")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Basic functionality test failed: {e}")
        return False

def test_buffer_manager():
    """Test buffer manager specifically"""
    print("\nTesting buffer manager...")
    
    try:
        from cpu_buffer_manager import CPUBufferManager, BufferConfig
        
        config = BufferConfig(max_atoms=1000, max_neighbors=100)
        manager = CPUBufferManager(config)
        
        # Test data preparation
        test_data = {
            'lammps_itypes': np.random.randint(0, 4, 500, dtype=np.int32),
            'lammps_js': np.random.randint(0, 500, (500, 50), dtype=np.int32),
            'lammps_rijs': np.random.randn(500, 50, 3).astype(np.float32),
            'lammps_jtypes': np.random.randint(0, 4, (500, 50), dtype=np.int32),
        }
        
        input_data = manager.prepare_input_data(
            test_data['lammps_itypes'],
            test_data['lammps_js'],
            test_data['lammps_rijs'],
            test_data['lammps_jtypes'],
            3, 1000.0, 500, 50
        )
        
        print("  ✅ Buffer manager test passed")
        
        # Get performance stats
        stats = manager.get_performance_stats()
        print(f"  ✅ Buffer stats: avg time {stats.get('avg_time_per_conversion_ms', 0):.2f} ms")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Buffer manager test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("JAX-MTP CPU Interface Optimization - Test Suite")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_import),
        ("Basic Functionality", test_basic_functionality),
        ("Buffer Manager", test_buffer_manager),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} FAILED")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Optimization system is ready.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
