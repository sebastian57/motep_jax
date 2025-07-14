#!/bin/bash
#PBS -N mtp_lammps_test
#PBS -l select=4:node_type=mi300a:mpiprocs=4
#PBS -l walltime=00:10:00

# Load required modules (matching your setup)
module load craype-x86-genoa
module load PrgEnv-cray/8.5.0
module load rocm/6.2.2
module load cray-mpich/8.1.30
module load cce/18.0.1
module load perftools-base/24.07.0
module load craype-accel-amd-gfx942
module load amd-mixed/6.2.2
module load cray-pals/1.3.2

# Activate your JAX environment
source /zhome/academic/HLRS/imw/imwseb/jax_lammps_env/bin/activate
PYTHON=/zhome/academic/HLRS/imw/imwseb/jax_lammps_env/bin/python3

echo "=== After venv activation ==="
echo "Which python: $(which python)"
python --version
echo $PYTHON
cd $PBS_O_WORKDIR/simulation_data_lmpjax
echo "Working directory: $PBS_O_WORKDIR"
pwd

export CRAY_ACCEL_TARGET=mi300a
export MPICH_GPU_SUPPORT_ENABLED=0
export MPICH_GTL_PROVIDER=rocm
export MPICH_GPU_SUPPORT_LEVEL=1
export HIP_VISIBLE_DEVICES=0,1,2,3
export JAX_ENABLE_X64=false
export OMP_NUM_THREADS=24
export ROCR_VISIBLE_DEVICES=0,1,2,3
export MPICH_OFI_SKIP_NIC_SYMMETRY_TEST=1

export PYTHONPATH=$PYTHONPATH:$PWD

export JAX_PLATFORM_NAME=rocm
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export JAX_ENABLE_X64=false
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0

export MALLOC_MMAP_THRESHOLD_=65536
export MALLOC_TRIM_THRESHOLD_=131072


echo "Starting LAMMPS simulation at $(date)..."
rocm-smi --showmemuse
mpirun -np 16 -ppn 4 --cpu-bind list:0-23:24-47:48-71:72-95 --gpu-bind list:0:1:2:3 ~/lammps/build/lmp -in lammps_jaxmtp.in
