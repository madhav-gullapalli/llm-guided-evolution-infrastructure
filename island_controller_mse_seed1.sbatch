#!/bin/bash
#SBATCH --job-name=Seed1MseCtrl
#SBATCH -N1 --ntasks-per-node=16
#SBATCH --mem-per-gpu=16G
#SBATCH --time=4:00:00
#SBATCH --output=run_job_outputs/islands/Report_mse_seed1-%j.out
#SBATCH --gres=gpu:1
#SBATCH -C intel
#SBATCH --exclude=atl1-1-03-014-16-0,atl1-1-03-011-13-0
# Matched τ+MSE baseline: num_init=20, BANANAS seed 1, standard naslib prompts.
# $1 COUNT, $2 PREV_SERVER_JOB_ID
COUNT=$1
PREV_SERVER_JOB_ID=${2:-}

cd $SLURM_SUBMIT_DIR
echo "launching seed-1 τ+MSE islands"
echo "Started on `/bin/hostname`"
echo "$COUNT iterations remaining"

module load cuda
module load uv

export HF_HOME=/storage/ice-shared/vip-vvk/llm_storage/
# ARCHIVE / ablation twin: solo SURROGATE τ+MSE on seed 1
export LLMGE_EVOLUTION_MODE=SURROGATE
export LLMGE_FITNESS_MODE=mse
export LLMGE_SEARCH_SEED=1
export LLMGE_PORT=18634
export LLMGE_HOSTNAME_FILE=hostname_mse_seed1.log

cd /storage/ice1/1/9/schen981/llm-guided-evolution-infrastructure

uv run python islands_wrapper.py surrogate_mse_seed1_run \
    --num_islands 3 \
    --llms llama3 \
    --prompt_groups "naslib/general,naslib/ranking,naslib/efficiency"

if (( COUNT > 1 )); then
    NEXT_COUNT=$((COUNT - 1))
    if [[ -n "$PREV_SERVER_JOB_ID" ]]; then
        echo "Stopping previous server job: $PREV_SERVER_JOB_ID"
        scancel "$PREV_SERVER_JOB_ID"
    fi
    echo "Controller completed; launching next MSE seed-1 server with count: $NEXT_COUNT"
    sbatch -p coe-gpu server_mse_seed1.sh "$NEXT_COUNT"
fi
