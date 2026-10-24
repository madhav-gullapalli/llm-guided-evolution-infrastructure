#!/bin/bash
#SBATCH --job-name=IslandsController
#SBATCH -N1 --ntasks-per-node=16
#SBATCH --mem-per-gpu=16G
#SBATCH --time=4:00:00
#SBATCH --output=run_job_outputs/islands/Report_islands-%j.out
#SBATCH --gres=gpu:1
#SBATCH -C intel
#SBATCH --exclude=atl1-1-03-014-16-0,atl1-1-03-011-13-0
# Island controller arguments:
# $1 (COUNT): Number of remaining restart iterations
# $2 (PREV_SERVER_JOB_ID): Job ID of the currently-running server (to cancel)
COUNT=$1
PREV_SERVER_JOB_ID=${2:-}

cd $SLURM_SUBMIT_DIR
echo "launching AIsurBL"
echo "Started on `/bin/hostname`"
echo "$COUNT iterations remaining"

module load cuda
module load uv

# LLM_Storage
export HF_HOME=/storage/ice-shared/vip-vvk/llm_storage/
# Evolve the surrogate model seed (not usage_bananas); fitness = τ + Precision@10%
export LLMGE_EVOLUTION_MODE=SURROGATE

# Change to the repository root
cd /storage/ice1/1/9/schen981/llm-guided-evolution-infrastructure

# Fresh run dir for τ+Precision fitness (do not resume old surrogate_islands_run)
uv run python islands_wrapper.py surrogate_precision_run \
    --num_islands 3 \
    --llms llama3 \
    --prompt_groups "naslib/general,naslib/ranking,naslib/efficiency"

if (( COUNT > 1 )); then
    NEXT_COUNT=$((COUNT - 1))
    if [[ -n "$PREV_SERVER_JOB_ID" ]]; then
        echo "Stopping previous server job: $PREV_SERVER_JOB_ID"
        scancel "$PREV_SERVER_JOB_ID"
    fi
    echo "Controller completed; launching next server iteration with count: $NEXT_COUNT"
    sbatch server.sh "$NEXT_COUNT"
fi
