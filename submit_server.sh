#!/bin/bash
# Submit held so scheduler-side exclusion can be checked before allocation.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
blocked_node=atl1-1-03-014-16-0,atl1-1-03-011-13-0,atl1-1-03-011-28-0
job_id=$(sbatch --parsable --hold --exclude="$blocked_node" server.sh "${1:-1}")
job_id=${job_id%%;*}
if [[ ! "$job_id" =~ ^[0-9]+$ ]]; then
    echo "FATAL: unexpected sbatch response: $job_id" >&2
    exit 1
fi
cleanup() {
    echo "Submission verification failed; cancelling held job $job_id" >&2
    scancel "$job_id"
}
trap cleanup EXIT
# Update after submission to bypass any submission-time rewriting of exclusions.
scontrol update JobId="$job_id" ExcNodeList="$blocked_node"
record=$(scontrol show job -o "$job_id")
actual_exclusions=${record#* ExcNodeList=}
actual_exclusions=${actual_exclusions%% *}
expanded=$(scontrol show hostnames "$actual_exclusions")
IFS=, read -r -a required_nodes <<< "$blocked_node"
for node in "${required_nodes[@]}"; do
    if ! [[ $'\n'"$expanded"$'\n' == *$'\n'"$node"$'\n'* ]]; then
        echo "FATAL: Slurm did not retain exclusion $node: $record" >&2
        exit 1
    fi
done
scontrol release "$job_id"
trap - EXIT
echo "Submitted job $job_id with verified exclusion: $blocked_node"
