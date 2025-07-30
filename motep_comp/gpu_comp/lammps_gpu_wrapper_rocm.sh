#!/bin/bash
# LAMMPS ROCm GPU Wrapper - Fixes JAX GPU detection in embedded Python
# Usage: ./lammps_gpu_wrapper_rocm.sh -in your_input.in

echo "🚀 LAMMPS JAX ROCm GPU Wrapper"
echo "Setting up ROCm GPU environment..."

# Critical: Set environment BEFORE LAMMPS starts (matching submit script)
export ROCR_VISIBLE_DEVICES=0,1,2,3
export HIP_VISIBLE_DEVICES=0,1,2,3
export JAX_PLATFORMS=rocm
export JAX_PLATFORM_NAME=rocm
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_autotune_level=4"

# Additional ROCm optimizations (from submit script)
export JAX_ENABLE_X64=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0

# Memory optimizations for MI300A
export MALLOC_MMAP_THRESHOLD_=65536
export MALLOC_TRIM_THRESHOLD_=131072

# Verify ROCm GPU is available
echo "Checking ROCm GPU availability..."
if command -v rocm-smi &> /dev/null; then
    GPU_INFO=$(rocm-smi --showmemuse --csv | head -2 | tail -1)
    echo "✅ ROCm GPU detected: $GPU_INFO"
else
    echo "⚠️  rocm-smi not found - ROCm GPU may not be available"
fi

# Quick JAX ROCm GPU test
echo "Testing JAX ROCm GPU access..."
python3 -c "
import os
import jax
devices = jax.devices()
gpu_devices = [d for d in devices if 'gpu' in str(d).lower() or 'rocm' in str(d).lower() or 'hip' in str(d).lower()]
if gpu_devices:
    print(f'✅ JAX ROCm GPU ready: {gpu_devices[0]}')
else:
    print(f'❌ JAX ROCm GPU not available. Devices: {devices}')
    print('Will fall back to CPU (slower but functional)')
" 2>/dev/null

echo ""
echo "🔥 Starting LAMMPS with ROCm-enabled JAX..."
echo "Command: mpirun -n 4 ../lammps/build/lmp $@"
echo ""

# Execute LAMMPS with all arguments passed through
exec mpirun -n 4 ../lammps/build/lmp "$@"