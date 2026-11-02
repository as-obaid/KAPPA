#!/bin/bash
# Shared environment for the KAPPA cluster scripts. Source it; do not execute it.
#
#   source slurm/env_common.sh                   # full environment (login-node scripts)
#   source slurm/env_common.sh --settings-only   # KAPPA_* settings only (batch jobs)
#
# Settings are KAPPA_* variables, taken from the calling shell first and then
# from the repository's .env. KAPPA_SCRATCH is required: a large per-user
# directory (for example /scratch/$USER). Every other path defaults under it,
# and every variable below is `:-` defaulted, so any of them can be overridden.
#
# Nothing is written to $HOME except trained weights ($KAPPA_WEIGHTS_DIR).

# ------------------------------------------------------------------- settings
# KAPPA_* keys from .env, never overriding a variable the shell already set.
# Same rule as the loader in src/kappa/data/paths.py, restricted to KAPPA_*.
_kappa_env_file="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.env"
if [ -f "$_kappa_env_file" ]; then
  while IFS= read -r _kappa_line || [ -n "$_kappa_line" ]; do
    _kappa_line="${_kappa_line%$'\r'}"
    _kappa_line="${_kappa_line#export }"
    case "$_kappa_line" in KAPPA_*=*) ;; *) continue ;; esac
    _kappa_key="${_kappa_line%%=*}"
    _kappa_val="${_kappa_line#*=}"
    case "$_kappa_key" in *[!A-Za-z0-9_]*) continue ;; esac
    _kappa_val="${_kappa_val%\"}"; _kappa_val="${_kappa_val#\"}"
    _kappa_val="${_kappa_val%\'}"; _kappa_val="${_kappa_val#\'}"
    [ -n "${!_kappa_key:-}" ] || export "$_kappa_key=$_kappa_val"
  done < "$_kappa_env_file"
fi
unset _kappa_env_file _kappa_line _kappa_key _kappa_val

if [ -z "${KAPPA_SCRATCH:-}" ]; then
  echo "KAPPA_SCRATCH is not set. Export it, or add a line such as" >&2
  echo "  KAPPA_SCRATCH=/scratch/<your-user>" >&2
  echo "to the repository's .env. See slurm/README.md." >&2
  return 1
fi
export KAPPA_SCRATCH

if [ "${1:-}" = "--settings-only" ]; then
  return 0
fi

# --------------------------------------------------------------- environments
# Two envs, not interchangeable (see slurm/env_build.sh):
#   KAPPA_VENV        torch cu130 + vLLM, report extraction and CPU jobs; needs driver 580+
#   KAPPA_TRAIN_VENV  torch cu126 + timm, training
export KAPPA_VENV="${KAPPA_VENV:-$KAPPA_SCRATCH/envs/kappa-vllm}"
export KAPPA_TRAIN_VENV="${KAPPA_TRAIN_VENV:-$KAPPA_SCRATCH/envs/kappa-train}"

# --------------------------------------------------------------------- caches
# Every client below defaults its cache into $HOME.
export HF_HOME="${HF_HOME:-$KAPPA_SCRATCH/hf_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$KAPPA_SCRATCH/.cache}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$KAPPA_SCRATCH/vllm_cache}"
export TORCH_HOME="${TORCH_HOME:-$KAPPA_SCRATCH/torch_home}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$KAPPA_SCRATCH/triton_cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$KAPPA_SCRATCH/inductor_cache}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$KAPPA_SCRATCH/nv_compute_cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$KAPPA_SCRATCH/uv-cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$KAPPA_SCRATCH/uv-python}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$KAPPA_SCRATCH/pip-cache}"
# Forced, not `:-` defaulted: Slurm sets TMPDIR itself, and /tmp on a node is small.
export TMPDIR="${KAPPA_TMPDIR:-$KAPPA_SCRATCH/tmp}"

# -------------------------------------------------------------------- outputs
export KAPPA_WORK="${KAPPA_WORK:-$KAPPA_SCRATCH/kappa-work}"
export KAPPA_DATA_ROOT="${KAPPA_DATA_ROOT:-$KAPPA_SCRATCH/kappa-data}"
export WANDB_DIR="${WANDB_DIR:-$KAPPA_WORK/wandb}"
# Scratch is purgeable; trained weights are worth keeping.
export KAPPA_WEIGHTS_DIR="${KAPPA_WEIGHTS_DIR:-$HOME/kappa-weights}"

# ------------------------------------------------------------------ behaviour
# Compute nodes have no outbound internet. Scripts that need the network unset
# this explicitly.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
# Log W&B offline; slurm/sync_wandb.sh uploads from the login node.
export WANDB_MODE="${WANDB_MODE:-offline}"

mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$VLLM_CACHE_ROOT" "$TORCH_HOME" \
         "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH" \
         "$TMPDIR" "$KAPPA_WORK/logs" "$WANDB_DIR" 2>/dev/null || true

# ------------------------------------------------------------------ utilities

# Put a venv's bin/ on PATH without activating it, so tools the process shells
# out to (vLLM runs `ninja`) are found.
kappa_use_venv() {
  local venv="$1"
  if [ ! -x "$venv/bin/python" ]; then
    echo "no interpreter at $venv/bin/python -- run slurm/env_build.sh" >&2
    return 1
  fi
  export PATH="$venv/bin:$PATH"
  export VIRTUAL_ENV="$venv"
}

# Identifying lines at the top of every log: env, node, job, commit.
kappa_banner() {
  printf 'venv    : %s\n' "${VIRTUAL_ENV:-<none>}"
  printf 'node    : %s\n' "$(hostname)"
  printf 'job     : %s%s\n' "${SLURM_JOB_ID:-none}" \
         "${SLURM_ARRAY_TASK_ID:+ (array task ${SLURM_ARRAY_TASK_ID})}"
  printf 'gpus    : %s\n' "${SLURM_JOB_GPUS:-none}"
  printf 'commit  : %s\n' "$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
  printf 'scratch : %s\n' "${KAPPA_SCRATCH:-<unset>}"
  printf 'started : %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  # Must return 0 on nodes without nvidia-smi: callers run under `set -e`.
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
  else
    printf 'gpu     : no nvidia-smi on this node\n'
  fi
  return 0
}
