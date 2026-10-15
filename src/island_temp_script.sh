#!/bin/bash
#SBATCH --job-name=IslandsController
#SBATCH -N1 --ntasks-per-node=16
#SBATCH --mem-per-gpu=16G
#SBATCH --time=8:00:00
#SBATCH --output=run_job_outputs/islands/Report_islands-%j.out
#SBATCH --gres=gpu:1
#SBATCH -C intel
#SBATCH --exclude=atl1-1-03-014-16-0
cd $SLURM_SUBMIT_DIR
echo "launching AIsurBL"
echo "Started on `/bin/hostname`"

module load cuda

export HF_HOME=/storage/ice-shared/vip-vvk/llm_storage/

# Run Python script
uv run python run_coevolution.py --checkpoints /home/hice1/mgullapalli6/scratch/llm-guided-evolution-infrastructure/titanic_islands_run
