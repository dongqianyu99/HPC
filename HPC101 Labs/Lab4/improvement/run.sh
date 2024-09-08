#!/bin/bash
#SBATCH --job-name=solver
#SBATCH --output=run.out
#SBATCH --error=run.err
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=52
#SBATCH --time=00:30:00
#SBATCH --partition=Solver
source /opt/intel/oneapi/setvars.sh
module load openmpi

export OMP_NUM_THREADS=52
export OMP_PROC_BIND=close

# Run BICGSTAB
# ./build/bicgstab $1
mpirun -np 6 --bind-to core --map-by NUMA ./build/bicgstab $1
