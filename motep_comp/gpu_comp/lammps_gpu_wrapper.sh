#!/bin/bash
# LAMMPS GPU Wrapper - Fixes JAX GPU detection in embedded Python
# Usage: ./lammps_gpu_wrapper.sh -in your_input.in

echo "🚀 LAMMPS JAX GPU Wrapper"
echo "Setting up GPU environment..."

# Critical: Set environment BEFORE LAMMPS starts
export CUDA_VISIBLE_DEVICES=0
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_autotune_level=4"

# Additional GPU optimizations
export TF_FORCE_GPU_ALLOW_GROWTH=true
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

# Verify GPU is available
echo "Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1)
    echo "✅ GPU detected: $GPU_INFO"
else
    echo "⚠️  nvidia-smi not found - GPU may not be available"
fi

# Quick JAX GPU test
echo "Testing JAX GPU access..."
python3 -c "
import os
import jax
devices = jax.devices()
gpu_devices = [d for d in devices if 'gpu' in str(d).lower() or 'cuda' in str(d).lower()]
if gpu_devices:
    print(f'✅ JAX GPU ready: {gpu_devices[0]}')
else:
    print(f'❌ JAX GPU not available. Devices: {devices}')
    print('Will fall back to CPU (slower but functional)')
" 2>/dev/null

echo ""
echo "🔥 Starting LAMMPS with GPU-enabled JAX..."
echo "Command: ~/jax_lammps/bin/lmp $@"
echo ""

# Execute LAMMPS with all arguments passed through
exec mpirun -n 4 ../lammps/build/lmp "$@"
