#!/bin/bash
# Extract structured findings from the radiology reports with a local LLM (vLLM),
# as an 8-way shard array, one GPU per shard.
#
#   sbatch slurm/extract_reports.sbatch
#   sbatch --array=0 slurm/extract_reports.sbatch --limit 20     # smoke test
#   sbatch --array=0-3%2 slurm/extract_reports.sbatch            # submit in halves if the
#   sbatch --array=4-7%2 slurm/extract_reports.sbatch            # submit limit refuses 8 tasks
#
# Expects: KAPPA_SCRATCH (shell or .env); KAPPA_VENV, default $KAPPA_SCRATCH/envs/kappa-vllm
# (slurm/env_build.sh vllm). Model weights must already be in $KAPPA_SCRATCH/hf_cache
# (slurm/stage_weights.sh). Extra arguments go to scripts/extract_reports.py.
#
# Notes:
# - The vLLM env uses cu130 wheels, which need NVIDIA driver 580 or newer. On a
#   partition with mixed drivers, pin a suitable node: sbatch --nodelist=<node> ...
# - The shard count is fixed (KAPPA_N_SHARDS, default 8), not read from the array
#   size, so resubmitting part of the array cannot re-partition the corpus.
# - Resumable: records already written are skipped on resubmission.

#SBATCH --job-name=kappa-extract-reports
#SBATCH --partition=sharing
#SBATCH --gres=gpu:h200:1
#SBATCH --array=0-7%2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:45:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
source slurm/env_common.sh --settings-only

module load cuda/12.8.0 cuDNN/9.10.2

KAPPA_VENV="${KAPPA_VENV:-$KAPPA_SCRATCH/envs/kappa-vllm}"

# vLLM runs `ninja` as a subprocess, so the venv's bin/ must be on PATH.
export PATH="$KAPPA_VENV/bin:$PATH"

# Use vLLM's native sampler instead of JIT-compiling flashinfer's on the node;
# decoding is greedy, so the two are equivalent here.
export VLLM_USE_FLASHINFER_SAMPLER=0

export HF_HOME="$KAPPA_SCRATCH/hf_cache"
export HF_HUB_OFFLINE=1          # no outbound internet on compute nodes
export VLLM_CACHE_ROOT="$KAPPA_SCRATCH/vllm_cache"
export XDG_CACHE_HOME="$KAPPA_SCRATCH/.cache"
mkdir -p "$VLLM_CACHE_ROOT" "$XDG_CACHE_HOME"
export PYTHONPATH="$PWD/src"     # the repo is used from PYTHONPATH, not installed
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline        # upload later with slurm/sync_wandb.sh

PY="$KAPPA_VENV/bin/python"

N_SHARDS="${KAPPA_N_SHARDS:-8}"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "error: no SLURM_ARRAY_TASK_ID. Submit this as an array job, e.g." >&2
  echo "  sbatch --array=0-$((N_SHARDS - 1))%2 slurm/extract_reports.sbatch" >&2
  exit 2
fi
if (( SLURM_ARRAY_TASK_ID >= N_SHARDS )); then
  echo "error: task $SLURM_ARRAY_TASK_ID is out of range for $N_SHARDS shards" >&2
  exit 2
fi

printf 'venv    : %s\n' "$KAPPA_VENV"
printf 'node    : %s\n' "$(hostname)"
printf 'gpus    : %s\n' "${SLURM_JOB_GPUS:-?}"
printf 'shard   : %s/%s\n' "$SLURM_ARRAY_TASK_ID" "$N_SHARDS"
printf 'started : %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

$PY scripts/extract_reports.py \
  --shard "${SLURM_ARRAY_TASK_ID}/${N_SHARDS}" \
  --resume \
  "$@"

printf 'finished: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
