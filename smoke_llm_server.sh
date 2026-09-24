#!/bin/bash
#SBATCH --job-name=LLMGE_SmokeServer
#SBATCH -t 2:00:00
#SBATCH --nodes=1
#SBATCH -G 2
#SBATCH -C "H200"
#SBATCH --mem 160G
#SBATCH -c 16
#SBATCH --output=run_job_outputs/server/smoke-%j.out
# Server-only smoke test: GPU preflight + model load. Does NOT launch islands.

echo "launching LLM Smoke Server"
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
HOSTNAME_FILE="$(pwd)/hostname.log"
echo "Writing server hostname '$SERVER_HOSTNAME' to file: $HOSTNAME_FILE"
echo "$SERVER_HOSTNAME" > "$HOSTNAME_FILE"
echo "Starting smoke LLM server on host: $SERVER_HOSTNAME (NO islands)"

uv run uvicorn server:app --host "$SERVER_HOSTNAME" --port 8137 --workers 1
