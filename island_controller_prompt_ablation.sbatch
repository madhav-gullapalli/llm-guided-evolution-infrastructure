#!/bin/bash
#SBATCH --job-name=PromptAblationCtrl
#SBATCH -N1 --ntasks-per-node=16
#SBATCH --mem-per-gpu=16G
#SBATCH --time=4:00:00
#SBATCH --output=run_job_outputs/islands/Report_prompt_ablation-%j.out
#SBATCH --gres=gpu:1
#SBATCH -C intel
#SBATCH --exclude=atl1-1-03-014-16-0,atl1-1-03-011-13-0
# Parallel ablation: τ+Precision fitness WITH prompts that name those objectives.
# Uses a separate run dir / LLM port / hostname file so it can run beside surrogate_precision_run.
# $1 (COUNT): remaining restart iterations
# $2 (PREV_SERVER_JOB_ID): server job to cancel when chaining
COUNT=$1
PREV_SERVER_JOB_ID=${2:-}

cd $SLURM_SUBMIT_DIR
echo "launching prompt-ablation islands"
echo "Started on `/bin/hostname`"
echo "$COUNT iterations remaining"

module load cuda
module load uv

export HF_HOME=/storage/ice-shared/vip-vvk/llm_storage/
export LLMGE_EVOLUTION_MODE=SURROGATE
# Match server_prompt_ablation.sh
export LLMGE_PORT=18632
export LLMGE_HOSTNAME_FILE=hostname_prompt_ablation.log

cd /storage/ice1/1/9/schen981/llm-guided-evolution-infrastructure

uv run python islands_wrapper.py surrogate_precision_prompt_run \
    --num_islands 3 \
    --llms llama3 \
    --prompt_groups "naslib_precision/general,naslib_precision/ranking,naslib_precision/efficiency"

if (( COUNT > 1 )); then
    NEXT_COUNT=$((COUNT - 1))
    if [[ -n "$PREV_SERVER_JOB_ID" ]]; then
        echo "Stopping previous server job: $PREV_SERVER_JOB_ID"
        scancel "$PREV_SERVER_JOB_ID"
    fi
    echo "Controller completed; launching next prompt-ablation server with count: $NEXT_COUNT"
    sbatch -p coe-gpu server_prompt_ablation.sh "$NEXT_COUNT"
fi
