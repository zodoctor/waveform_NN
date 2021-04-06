#!/bin/bash
#SBATCH --partition=long      ### Partition
#SBATCH --job-name=ArrayJob    ### Job Name
#SBATCH --time=05:00:00        ### WallTime
#SBATCH --nodes=1              ### Number of Nodes
#SBATCH --ntasks=1             ### Number of tasks per array job
#SBATCH --cpus-per-task=1       ### number of cores for each task
#SBATCH --account=farr_lab

module load anaconda3
conda activate wfenv
./train_model.py model1 wf_data 100 --neurons_per_layer 8 128 256 --activations tanh tanh tanh --batch_size 32 
#echo "I am Slurm job ${SLURM_JOB_ID}, array job ${SLURM_ARRAY_JOB_ID}, and array task ${SLURM_ARRAY_TASK_ID}."
