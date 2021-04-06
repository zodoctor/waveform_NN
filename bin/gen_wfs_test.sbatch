#!/bin/bash
#SBATCH --partition=short      ### Partition
#SBATCH --job-name=ArrayJob    ### Job Name
#SBATCH --output=testset.out   ### file in which to store job stdout
#SBATCH --error=testset.err    ### file in which to store job stderr
#SBATCH --time=00:20:00        ### WallTime
#SBATCH --nodes=1              ### Number of Nodes
#SBATCH --ntasks=16             ### Number of tasks per array job
#SBATCH --cpus-per-task=1       ### number of cores for each task
#SBATCH --account=farr_lab

module load anaconda3
conda activate wfenv
./gen_wfs.py wf_test_data wftestdata 12 --ncores 16 --seed 1
#echo "I am Slurm job ${SLURM_JOB_ID}, array job ${SLURM_ARRAY_JOB_ID}, and array task ${SLURM_ARRAY_TASK_ID}."
