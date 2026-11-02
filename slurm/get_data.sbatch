#!/bin/bash
# Verify and extract the downloaded competition archive on a compute node (CPU).
#
#   scripts/get_data.sh --fetch-only    # first, on the login node (needs the network)
#   sbatch slurm/get_data.sbatch
#
# Expects: KAPPA_DATA_ROOT (shell or .env), read by scripts/get_data.sh; it defaults
# to <repo>/data. Extra arguments go to scripts/get_data.sh.
#
# Notes:
# - Compute nodes have no outbound internet, so the download cannot run here.
# - CRC verification decompresses the whole archive; keep it off the login node.

#SBATCH --job-name=kappa-extract
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

printf 'node    : %s\n' "$(hostname)"
printf 'started : %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf 'cpus    : %s\n' "${SLURM_CPUS_PER_TASK:-?}"

scripts/get_data.sh --extract-only "$@"

printf 'finished: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
