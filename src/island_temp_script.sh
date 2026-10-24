#!/bin/bash
#SBATCH --job-name=IslandsController
#SBATCH -N1 --ntasks-per-node=16
#SBATCH --mem-per-gpu=16G
#SBATCH --time=4:00:00
#SBATCH --output=run_job_outputs/islands/Report_islands-%j.out
#SBATCH --gres=gpu:1
#SBATCH -C intel
#SBATCH --exclude=atl1-1-03-014-16-0,atl1-1-03-011-13-0
cd $SLURM_SUBMIT_DIR
echo "launching AIsurBL"
echo "Started on `/bin/hostname`"

module load cuda

export HF_HOME=/storage/ice-shared/vip-vvk/llm_storage/
# Bash defaults; braces doubled for Python str.format.
export LLMGE_EVOLUTION_MODE="${LLMGE_EVOLUTION_MODE:-SURROGATE}"
export LLMGE_PORT="${LLMGE_PORT:-18631}"
export LLMGE_HOSTNAME_FILE="${LLMGE_HOSTNAME_FILE:-hostname.log}"
export LLMGE_SEARCH_SEED="${LLMGE_SEARCH_SEED:-242}"

# Run Python script
uv run python run_improved.py --checkpoints surrogate_precision_seed1_run/island_llama3_naslib-efficiency --global_path surrogate_precision_seed1_run/global_data --llm_model llama3 --prompt_group naslib/efficiency
