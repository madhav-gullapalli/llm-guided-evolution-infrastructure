#!/bin/bash
#SBATCH --job-name=LLMGE_SmokeServer
#SBATCH -t 2:00:00
#SBATCH --nodes=1
#SBATCH -G 2
#SBATCH -C "H200"
#SBATCH --mem 160G
#SBATCH -c 16
#SBATCH --exclude=atl1-1-03-014-16-0
#SBATCH --output=run_job_outputs/server/smoke-%j.out
# Server-only smoke test: GPU preflight + model load. Does NOT launch islands.

# Stop before publishing the hostname or submitting islands if preflight fails.
set -euo pipefail

# Defense against submission filters or overrides dropping the exclusion.
case "$(hostname -s)" in
    atl1-1-03-014-16-0)
        echo "FATAL: allocated blocked node atl1-1-03-014-16-0; refusing to start" >&2
        exit 1
        ;;
esac

echo "launching LLM Smoke Server"
hostname

module load cuda
module load uv

# Keep the GPU visibility and ordering assigned by Slurm.
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
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
if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
    print("FATAL: Expected two usable CUDA GPUs from the Slurm allocation", file=sys.stderr)
    sys.exit(1)
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {p.name} ({p.total_memory/(1024**3):.1f} GiB)")
    x = torch.ones(1, device=f"cuda:{i}")
    assert (x + 1).item() == 2
    torch.cuda.synchronize(i)
    print(f"cuda allocation and kernel ok on {x.device}")
PY

export SERVER_HOSTNAME=$(hostname)
HOSTNAME_FILE="$(pwd)/hostname.log"
echo "Writing server hostname '$SERVER_HOSTNAME' to file: $HOSTNAME_FILE"
echo "$SERVER_HOSTNAME" > "$HOSTNAME_FILE"
echo "Starting smoke LLM server on host: $SERVER_HOSTNAME (NO islands)"

uv run uvicorn server:app --host "$SERVER_HOSTNAME" --port 8137 --workers 1
