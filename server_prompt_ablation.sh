#!/bin/bash
#SBATCH --job-name=PromptAblationSrv
#SBATCH -t 2:00:00
#SBATCH --nodes=1
#SBATCH -G 2
#SBATCH -C "H200"
#SBATCH --mem 160G
#SBATCH -c 16
#SBATCH --exclude=atl1-1-03-014-16-0,atl1-1-03-011-13-0
#SBATCH --output=run_job_outputs/server/slurm-prompt-ablation-%j.out
echo "launching LLM Server (prompt ablation — τ + Precision prompts)"

COUNT=${1:-1}
EXTRA_FILE=/storage/ice1/1/9/schen981/llm-guided-evolution-infrastructure/extra_segments_prompt_ablation.txt
if [[ -f "$EXTRA_FILE" ]]; then
    EXTRA=$(cat "$EXTRA_FILE")
    rm -f "$EXTRA_FILE"
    COUNT=$((COUNT + EXTRA))
    echo "Added $EXTRA extra segments from $EXTRA_FILE (count now $COUNT)"
fi

hostname

module load cuda
module load uv

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1
export UV_CACHE_DIR="${TMPDIR:-${SLURM_TMPDIR:-/tmp}}/uv-cache-${SLURM_JOB_ID:-$$}"
mkdir -p "$UV_CACHE_DIR"
echo "Using UV cache: $UV_CACHE_DIR"

echo "=== GPU preflight (nvidia-smi) ==="
nvidia-smi || { echo "FATAL: nvidia-smi failed — GPUs not usable"; exit 1; }

echo "=== GPU preflight (torch.cuda) ==="
uv run python - <<'PY'
import torch, sys
print(f"torch={torch.__version__} cuda_build={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()} device_count={torch.cuda.device_count()}")
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    print("FATAL: PyTorch cannot see CUDA devices", file=sys.stderr)
    sys.exit(1)
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {p.name} ({p.total_memory/(1024**3):.1f} GiB)")
x = torch.zeros(1, device="cuda")
print(f"cuda tensor ok on {x.device}")
PY

export SERVER_HOSTNAME=$(hostname)
# Separate from the main surrogate_precision_run server (18631 / hostname.log)
export LLMGE_PORT=18632
export LLMGE_HOSTNAME_FILE=hostname_prompt_ablation.log
export LLMGE_EVOLUTION_MODE=SURROGATE

HOSTNAME_FILE="$(pwd)/hostname_prompt_ablation.log"

echo "Writing server hostname '$SERVER_HOSTNAME' to file: $HOSTNAME_FILE"
echo "$SERVER_HOSTNAME" > "$HOSTNAME_FILE"
echo "Starting LLM server on host: $SERVER_HOSTNAME (count=$COUNT, port=$LLMGE_PORT)"

echo "Submitting prompt-ablation island controller (count=$COUNT)"
sbatch island_controller_prompt_ablation.sbatch "$COUNT" "$SLURM_JOB_ID"

uv run uvicorn server:app --host $SERVER_HOSTNAME --port $LLMGE_PORT --workers 1
