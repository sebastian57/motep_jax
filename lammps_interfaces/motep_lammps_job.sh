#!/bin/bash
#PBS -N mtp_lammps_test
#PBS -l select=4:node_type=mi300a:mpiprocs=96
#PBS -l walltime=02:00:00

# Load required modules (matching your setup)
module load craype-x86-genoa
module load PrgEnv-cray/8.5.0
module load rocm/6.2.2
module load cray-mpich/8.1.30
module load cce/18.0.1
module load perftools-base/24.07.0
module load craype-accel-amd-gfx90a
module load amd-mixed/6.2.2
module load cray-pals/1.3.2
#module load cray-python/3.11.7
#module load python/3.11

# Activate your JAX environment
source ~/jax_lammps_env/bin/activate
PYTHON=~/jax_lammps_env/bin/python3

echo "=== After venv activation ==="
echo "Which python: $(which python)"
python --version
echo $PYTHON

cd $PBS_O_WORKDIR/simulation_data
echo "Working directory: $PBS_O_WORKDIR"
pwd

# Set environment variables (matching your setup)
export CRAY_ACCEL_TARGET=mi300a
export MPICH_GPU_SUPPORT_ENABLED=0
export MPICH_GTL_PROVIDER=rocm
export MPICH_GPU_SUPPORT_LEVEL=1

export PYTHONPATH=$PYTHONPATH:$PWD

export JAX_PLATFORM_NAME=rocm
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
export JAX_ENABLE_X64=false  
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0

export MALLOC_MMAP_THRESHOLD_=65536
export MALLOC_TRIM_THRESHOLD_=131072

# if ok to use cpu
#export JAX_PLATFORMS=""


# Number of processes
PROCS=96

echo "Starting LAMMPS simulation at $(date)..."
rocm-smi --showmemuse
mpirun -n $PROCS --cpu-bind list:0-23:24-47:48-71:72-95 --gpu-bind list:0:1:2:3 python3 ./pylammps_jaxmtp_ase_opt2.py

echo "Job completed at $(date)"
