#!/bin/bash
#SBATCH --partition=short      ### Partition
#SBATCH --job-name=ArrayJob    ### Job Name
#SBATCH --time=01:00:00        ### WallTime
#SBATCH --nodes=1              ### Number of Nodes
#SBATCH --ntasks=16             ### Number of tasks per array job
#SBATCH --cpus-per-task=1       ### number of cores for each task
#SBATCH --array=0-9           ### Array index
#SBATCH --account=farr_lab

module load anaconda3
conda activate wfenv
./gen_wfs.py wf_data "wfdata_${SLURM_ARRAY_TASK_ID}" 16 --ncores 16 --seed ${SLURM_ARRAY_TASK_ID}
#echo "I am Slurm job ${SLURM_JOB_ID}, array job ${SLURM_ARRAY_JOB_ID}, and array task ${SLURM_ARRAY_TASK_ID}."
