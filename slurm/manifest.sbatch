#!/bin/bash
# Build the series manifest: one DICOM header read per series, as a 16-way array (CPU).
#
#   sbatch slurm/manifest.sbatch
#   sbatch --array=0-31 slurm/manifest.sbatch --n-shards 32
#   python scripts/build_manifest.py --merge    # afterwards, once every task has finished
#
# Expects: KAPPA_SCRATCH (shell or .env); KAPPA_VENV, default $KAPPA_SCRATCH/envs/kappa-vllm;
# KAPPA_DATA_ROOT, default $KAPPA_SCRATCH/kappa-data, holding the extracted archive.
# Extra arguments go to scripts/build_manifest.py.
#
# Notes:
# - --require-all is passed: on the full tree a missing series directory means the
#   extraction dropped files, and a partial manifest would look complete.

#SBATCH --job-name=kappa-manifest
#SBATCH --partition=short
#SBATCH --array=0-15
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=2:00:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source slurm/env_common.sh --settings-only

KAPPA_VENV="${KAPPA_VENV:-$KAPPA_SCRATCH/envs/kappa-vllm}"
PY="$KAPPA_VENV/bin/python"

export PYTHONPATH="$PWD/src"     # the repo is used from PYTHONPATH, not installed

# kappa.data.paths resolves every input path from KAPPA_DATA_ROOT.
export KAPPA_DATA_ROOT="${KAPPA_DATA_ROOT:-$KAPPA_SCRATCH/kappa-data}"

N_SHARDS="${N_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-16}}"
SHARD="${SLURM_ARRAY_TASK_ID:-0}"

printf 'venv    : %s\n' "$KAPPA_VENV"
printf 'node    : %s\n' "$(hostname)"
printf 'data    : %s\n' "$KAPPA_DATA_ROOT"
printf 'shard   : %s of %s\n' "$SHARD" "$N_SHARDS"
printf 'started : %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"

$PY scripts/build_manifest.py \
    --shard "$SHARD" \
    --n-shards "$N_SHARDS" \
    --require-all \
    "$@"

printf 'finished: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
