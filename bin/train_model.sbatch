#!/bin/bash
#SBATCH --partition=short    ### Partition
#SBATCH --job-name=shortcut    ### Job Name
#SBATCH --time=04:00:00        ### WallTime
#SBATCH --nodes=1              ### Number of Nodes
#SBATCH --ntasks=1             ### Number of tasks per array job
#SBATCH --cpus-per-task=1       ### number of cores for each task
#SBATCH --account=farr_lab

module load anaconda3
conda activate wfenv
./train_model_conv.py model_shortcut wf_data 50 --neurons_per_layer 2000 2000 --activations relu relu --batch_size 64
#echo "I am Slurm job ${SLURM_JOB_ID}, array job ${SLURM_ARRAY_JOB_ID}, and array task ${SLURM_ARRAY_TASK_ID}."
