#!/bin/bash
# Parallel, resumable extraction of the competition archive (CPU).
#
#   sbatch slurm/extract.sbatch
#
# Expects: KAPPA_SCRATCH (shell or .env); KAPPA_VENV, default $KAPPA_SCRATCH/envs/kappa-vllm.
# The archive must already be downloaded (scripts/get_data.sh --fetch-only, on the
# login node). Extra arguments go to scripts/extract_archive.py.
#
# Notes:
# - The archive holds ~820k small files, so cost is per-file metadata on the shared
#   filesystem rather than bandwidth; the workers parallelise that.
# - Resumable: cancel and resubmit without losing work.

#SBATCH --job-name=kappa-extract
#SBATCH --partition=short
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=8:00:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source slurm/env_common.sh --settings-only

KAPPA_VENV="${KAPPA_VENV:-$KAPPA_SCRATCH/envs/kappa-vllm}"
PY="$KAPPA_VENV/bin/python"

printf 'venv    : %s\n' "$KAPPA_VENV"
printf 'node    : %s\n' "$(hostname)"
printf 'cpus    : %s\n' "${SLURM_CPUS_PER_TASK:-?}"
printf 'started : %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"

$PY scripts/extract_archive.py --workers "${SLURM_CPUS_PER_TASK:-16}" "$@"

printf 'finished: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
