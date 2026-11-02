#!/bin/bash
# Build the uint8 series cache the dataloader reads, as a resumable 64-way array (CPU).
#
#   sbatch slurm/cache.sbatch                        # all train studies
#   sbatch --array=0-3 slurm/cache.sbatch --limit 2  # smoke test
#   sbatch slurm/cache.sbatch --split test
#
# Expects: KAPPA_SCRATCH (shell or .env); KAPPA_VENV, default $KAPPA_SCRATCH/envs/kappa-vllm.
# Reads the extracted DICOMs under KAPPA_DATA_ROOT (shell or .env, default <repo>/data)
# and writes $KAPPA_DATA_ROOT/processed/cache. Extra arguments go to
# scripts/build_cache.py.
#
# Notes:
# - Resumable: a task killed at walltime can be resubmitted and skips finished series.
# - Requeue failed tasks with the same array width, or set KAPPA_N_SHARDS to match;
#   the shard count is fixed so a partial requeue cannot re-partition the work.

#SBATCH --job-name=kappa-cache
#SBATCH --partition=short
#SBATCH --array=0-63
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=2:00:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source slurm/env_common.sh --settings-only

KAPPA_VENV="${KAPPA_VENV:-$KAPPA_SCRATCH/envs/kappa-vllm}"
PY="$KAPPA_VENV/bin/python"

export PYTHONPATH="$PWD/src"     # the repo is used from PYTHONPATH, not installed

# One math thread per task: the parallelism is the array, and NumPy otherwise sizes
# its pool from the whole machine.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

printf 'node    : %s\n' "$(hostname)"
printf 'array   : %s of %s (min %s)\n' \
  "${SLURM_ARRAY_TASK_ID:-?}" "${SLURM_ARRAY_TASK_COUNT:-?}" "${SLURM_ARRAY_TASK_MIN:-?}"
printf 'cpus    : %s\n' "${SLURM_CPUS_PER_TASK:-?}"
printf 'venv    : %s\n' "$KAPPA_VENV"
printf 'started : %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# build_cache.py takes the shard index from SLURM_ARRAY_TASK_ID; do not pass it here.
N_SHARDS="${KAPPA_N_SHARDS:-64}"
$PY scripts/build_cache.py --shard-count "$N_SHARDS" "$@"

printf 'finished: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
