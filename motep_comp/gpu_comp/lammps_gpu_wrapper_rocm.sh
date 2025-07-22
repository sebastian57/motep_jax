#!/bin/bash
# LAMMPS ROCm Wrapper - Fixes JAX ROCm detection in embedded Python
# Usage: ./lammps_rocm_wrapper.sh -in your_input.in

echo "🚀 LAMMPS JAX ROCm Wrapper"
echo "Setting up ROCm GPU environment..."

# Critical: Set ROCm environment BEFORE LAMMPS starts
export HIP_VISIBLE_DEVICES=0
export ROCR_VISIBLE_DEVICES=0
export HSA_OVERRIDE_GFX_VERSION=9.4.2      # MI300A architecture
export JAX_PLATFORMS=rocm
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_autotune_level=4 --xla_gpu_enable_async_collectives=true"

# Additional ROCm optimizations
export HIP_LAUNCH_BLOCKING=0               # Allow async kernel launches
export ROCM_PATH=/opt/rocm                 # ROCm installation path
export HCC_AMDGPU_TARGET=gfx942           # MI300A architecture

# Memory optimizations
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

# Verify ROCm GPU is available
echo "Checking ROCm GPU availability..."
if command -v rocm-smi &> /dev/null; then
    GPU_INFO=$(rocm-smi --showproductname | grep "Card series" | head -1)
    echo "✅ ROCm GPU detected: $GPU_INFO"
    
    # Show memory info
    MEMORY_INFO=$(rocm-smi --showmeminfo vram | grep "Used Memory" | head -1)
    echo "   Memory status: $MEMORY_INFO"
else
    echo "⚠️  rocm-smi not found - ROCm GPU may not be available"
fi

# Quick JAX ROCm test
echo "Testing JAX ROCm access..."
python3 -c "
import os
import jax
devices = jax.devices()
rocm_devices = [d for d in devices if 'rocm' in str(d).lower() or 'hip' in str(d).lower()]
gpu_devices = [d for d in devices if 'gpu' in str(d).lower()]

if rocm_devices:
    print(f'✅ JAX ROCm ready: {rocm_devices[0]}')
elif gpu_devices:
    print(f'✅ JAX GPU ready: {gpu_devices[0]}')
else:
    print(f'❌ JAX ROCm not available. Devices: {devices}')
    print('Will fall back to CPU (slower but functional)')
" 2>/dev/null

echo ""
echo "🔥 Starting LAMMPS with ROCm-enabled JAX..."
echo "Command: mpirun -n 4 ../lammps/build/lmp $@"
echo ""

# Execute LAMMPS with all arguments passed through
exec mpirun -n 4 ../lammps/build/lmp "$@"
