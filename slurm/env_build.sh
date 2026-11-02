#!/bin/bash
# Build the KAPPA Python environments with uv. Login node only: it needs the network.
#
#   slurm/env_build.sh train           # torch cu126 + timm, for training
#   slurm/env_build.sh vllm            # torch cu130 + vLLM 0.28.0, for report extraction
#   slurm/env_build.sh all
#   slurm/env_build.sh train --force   # delete and rebuild that env
#   tmux new -s envbuild 'slurm/env_build.sh all'   # the vLLM env is ~8 GiB
#
# Expects: KAPPA_SCRATCH (shell or .env); uv at UV_BIN, default $HOME/.local/bin/uv.
# Writes: KAPPA_TRAIN_VENV, default $KAPPA_SCRATCH/envs/kappa-train;
# KAPPA_VLLM_VENV, default $KAPPA_SCRATCH/envs/kappa-vllm; and a `uv pip freeze`
# lock per env in slurm/locks/.
#
# Notes:
# - Two envs because vLLM 0.28 pins torch cu130, which needs NVIDIA driver 580+.
#   Training uses cu126, which also runs on 570-series drivers and supports H200.
# - A cu130 torch on an older driver imports fine but reports no GPU, so the
#   check that matters runs on a GPU node (slurm/train.sbatch does it).
# - Re-running is safe: an existing env is reused unless --force is given.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
source "$REPO/slurm/env_common.sh"

# Installing needs the network, so lift the offline default env_common.sh sets.
unset HF_HUB_OFFLINE

TORCH_CU126=https://download.pytorch.org/whl/cu126
PYVER="${KAPPA_PYVER:-3.11}"
LOCK_DIR="$REPO/slurm/locks"
FORCE=0
TARGETS=()

# Parse arguments before the uv check, so a typo reports a usage error.
for a in "$@"; do
  case "$a" in
    train|vllm) TARGETS+=("$a") ;;
    all)        TARGETS+=(train vllm) ;;
    --force)    FORCE=1 ;;
    *) echo "usage: $0 [train|vllm|all] [--force]" >&2; exit 2 ;;
  esac
done
[ ${#TARGETS[@]} -eq 0 ] && TARGETS=(train)

UV="${UV_BIN:-$HOME/.local/bin/uv}"
[ -x "$UV" ] || { echo "no uv at $UV -- set UV_BIN" >&2; exit 1; }

mkdir -p "$LOCK_DIR" "$KAPPA_SCRATCH/envs"

# Shared runtime deps. Not read from pyproject.toml: on the cluster the repo is
# used from PYTHONPATH rather than installed.
COMMON_PKGS=(numpy pandas pyarrow scikit-learn pydicom pydantic wandb)

ensure_venv() {
  # Reuse an existing venv unless --force, so a partial run can be completed by
  # running the same command again.
  local env_dir="$1"
  if [ "$FORCE" = 1 ]; then
    printf 'force   : removing %s\n' "$env_dir"
    rm -rf -- "$env_dir"
  elif [ -x "$env_dir/bin/python" ]; then
    printf 'reuse   : existing venv at %s (--force to rebuild)\n' "$env_dir"
    return 0
  elif [ -e "$env_dir" ]; then
    # Present but with no interpreter: a previous run died mid-creation.
    printf 'repair  : %s exists with no interpreter, recreating\n' "$env_dir"
    rm -rf -- "$env_dir"
  fi
  "$UV" venv --python "$PYVER" "$env_dir"
}

verify_train() {
  # Check which torch wheel landed by reading its metadata, without importing
  # torch: login nodes may kill large imports, and the CUDA tag is what matters.
  local env_dir="$1"
  local site
  site=$(echo "$env_dir"/lib/python*/site-packages)
  "$env_dir/bin/python" -c 'import sys; print("python  :", sys.version.split()[0])'

  local ver
  ver=$(sed -nE "s/^__version__ = '(.*)'\$/\1/p" "$site/torch/version.py" 2>/dev/null)
  local cu
  cu=$(sed -nE "s/^cuda: Optional\[str\] = '(.*)'\$/\1/p" "$site/torch/version.py" 2>/dev/null)
  printf 'torch   : %s | built for CUDA %s\n' "${ver:-MISSING}" "${cu:-MISSING}"

  case "$ver" in
    *+cu126) ;;
    *) echo "FAIL: expected a +cu126 torch, got '${ver:-nothing}'. The resolver" >&2
       echo "      took torch from somewhere other than $TORCH_CU126." >&2
       return 1 ;;
  esac

  local timm_ver
  timm_ver=$(sed -nE 's/^Version: (.*)$/\1/p' "$site"/timm-*.dist-info/METADATA 2>/dev/null | head -1)
  printf 'timm    : %s\n' "${timm_ver:-MISSING}"
  [ -n "$timm_ver" ] || { echo "FAIL: timm did not install" >&2; return 1; }

  echo "NOTE  : nothing above touched a GPU. slurm/train.sbatch checks"
  echo "        torch.cuda.is_available() on the compute node and exits if it is False."
}

build_train() {
  local env_dir="$KAPPA_TRAIN_VENV"
  printf '\n=== kappa-train (torch cu126) -> %s ===\n' "$env_dir"
  ensure_venv "$env_dir"
  export VIRTUAL_ENV="$env_dir"

  # torch from the cu126 index alone, in its own call: with PyPI also in scope the
  # resolver may pick a torch built for a different CUDA, which verify_train catches.
  "$UV" pip install --index-url "$TORCH_CU126" torch torchvision

  # A second call, so resolving the rest cannot move the torch build.
  "$UV" pip install timm pyyaml "${COMMON_PKGS[@]}"

  verify_train "$env_dir"
  "$UV" pip freeze > "$LOCK_DIR/kappa-train.lock"
  printf 'lock    : %s (%s lines)\n' "$LOCK_DIR/kappa-train.lock" \
         "$(wc -l < "$LOCK_DIR/kappa-train.lock")"
}

build_vllm() {
  local env_dir="${KAPPA_VLLM_VENV:-$KAPPA_SCRATCH/envs/kappa-vllm}"
  printf '\n=== kappa-vllm (torch cu130 + vLLM 0.28.0) -> %s ===\n' "$env_dir"
  printf 'NOTE   : needs NVIDIA driver 580 or newer at run time. See the header.\n'
  ensure_venv "$env_dir"
  export VIRTUAL_ENV="$env_dir"

  # vLLM pins its own torch, so it goes first and alone. 0.28.0 supports the
  # Qwen3.5 and Gemma 4 architectures used for extraction.
  "$UV" pip install "vllm==0.28.0"

  # vLLM runs ninja as a subprocess to build flashinfer kernels.
  "$UV" pip install ninja "${COMMON_PKGS[@]}"

  "$env_dir/bin/python" -c "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__, '| cuda', torch.version.cuda)"
  "$UV" pip freeze > "$LOCK_DIR/kappa-vllm.lock"
  printf 'lock    : %s (%s lines)\n' "$LOCK_DIR/kappa-vllm.lock" \
         "$(wc -l < "$LOCK_DIR/kappa-vllm.lock")"
}

kappa_banner
for t in "${TARGETS[@]}"; do
  case "$t" in
    train) build_train ;;
    vllm)  build_vllm ;;
  esac
done

printf '\nfinished: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf 'NEXT: a one-fold smoke run, e.g. sbatch --array=0 slurm/train.sbatch --config <cfg> --set optim.epochs=1\n'
