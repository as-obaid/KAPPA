#!/bin/bash
# Train one config with five-fold CV: one array task per fold, one GPU per task.
#
#   sbatch slurm/train.sbatch --config configs/baseline_axial_fs1.yaml --track
#   sbatch --array=0 slurm/train.sbatch --config configs/baseline_axial_fs1.yaml \
#          --set optim.epochs=1                       # smoke test, one fold
#   sbatch --partition=gpu-short --time=2:00:00 slurm/train.sbatch --config ...
#
# Expects: KAPPA_SCRATCH (shell or .env); KAPPA_TRAIN_VENV, default
# $KAPPA_SCRATCH/envs/kappa-train (slurm/env_build.sh train); KAPPA_DATA_ROOT, default
# $KAPPA_SCRATCH/kappa-data, holding the manifest, cache and labels. Pretrained weights
# must already be in $KAPPA_SCRATCH/hf_cache. Arguments go to scripts/train.py; the fold
# is the array index. Outputs land in data/runs/<name>/; W&B runs in $WANDB_DIR.
#
# Notes:
# - One GPU per fold. Folds share nothing, so they run as independent array tasks.
# - The GPU model is pinned because the partition mixes cards and bf16 needs sm_80+.
# - Training reads KAPPA_TRAIN_VENV, never KAPPA_VENV: the vLLM env is cu130, which on
#   a 570-series driver (e.g. 570.86.15) imports torch but finds no GPU.
# - A shorter partition needs an explicit --time within its limit.
# - 3D configs need torchvision weights: export TORCH_HOME=$KAPPA_SCRATCH/torch_home
#   before submitting (slurm/stage_torch_weights.sh fills it).

#SBATCH --job-name=kappa-train
#SBATCH --partition=gpu
#SBATCH --array=0-4
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
# SIGUSR1 to this script five minutes before the walltime kill, so the trap below
# can tell scripts/train.py to commit its W&B run.
#SBATCH --signal=B:USR1@300

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source slurm/env_common.sh --settings-only

KAPPA_VENV="${KAPPA_TRAIN_VENV:-$KAPPA_SCRATCH/envs/kappa-train}"
PY="$KAPPA_VENV/bin/python"

if [[ ! -x "$PY" ]]; then
  printf 'error: no interpreter at %s\n' "$PY" >&2
  printf 'build it on the LOGIN node first: slurm/env_build.sh train\n' >&2
  exit 2
fi

export PYTHONPATH="$PWD/src"     # the repo is used from PYTHONPATH, not installed
export KAPPA_DATA_ROOT="${KAPPA_DATA_ROOT:-$KAPPA_SCRATCH/kappa-data}"

# Configs name repo-relative data paths, so link the bulk-data directories in from
# KAPPA_DATA_ROOT. data/folds stays a real directory: fold assignments are tracked.
for sub in interim processed raw; do
  if [ ! -e "data/$sub" ] && [ -d "$KAPPA_DATA_ROOT/$sub" ]; then
    ln -s "$KAPPA_DATA_ROOT/$sub" "data/$sub"
    printf 'linked  : data/%s -> %s/%s\n' "$sub" "$KAPPA_DATA_ROOT" "$sub"
  fi
done

# Preflight: check the inputs the config names before waiting on torch.
CFG=""
prev=""
for arg in "$@"; do
  case "$prev" in --config) CFG="$arg";; esac
  prev="$arg"
done
if [ -n "$CFG" ] && [ -f "$CFG" ]; then
  MISSING=0
  while read -r want; do
    [ -z "$want" ] && continue
    if [ ! -e "$want" ]; then echo "MISSING INPUT: $want" >&2; MISSING=$((MISSING + 1)); fi
  done <<EOF
$(sed -n 's/^ *cache_dir: *//p;s/^ *manifest: *//p;s/^ *labels: *//p;s/^ *file: *//p' "$CFG")
EOF
  if [ "$MISSING" -gt 0 ]; then
    echo "FAILED: $MISSING input(s) named by $CFG do not exist." >&2
    echo "  manifest -> sbatch slurm/manifest.sbatch  then  build_manifest.py --merge" >&2
    echo "  cache    -> sbatch slurm/cache.sbatch" >&2
    echo "  labels   -> scripts/derive_labels.py" >&2
    echo "  folds    -> scripts/build_folds.py" >&2
    exit 1
  fi
  echo "preflight: all inputs named by $CFG exist"
fi

# Compute nodes have no outbound internet: fail fast on a missing weight file
# instead of hanging. Stage weights on the login node first.
export HF_HOME="$KAPPA_SCRATCH/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# W&B logs offline; slurm/sync_wandb.sh uploads from the login node and looks in
# the same WANDB_DIR. Artifact blobs stage in WANDB_DATA_DIR, outside the run
# directory, so keep it on scratch and sync before anything purges it.
export WANDB_MODE=offline
export WANDB_DIR="${WANDB_DIR:-$KAPPA_SCRATCH/kappa-work/wandb}"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-$KAPPA_SCRATCH/kappa-work/wandb-staging}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$KAPPA_SCRATCH/kappa-work/wandb-cache}"
mkdir -p "$WANDB_DIR" "$WANDB_DATA_DIR" "$WANDB_CACHE_DIR"

# Checkpoints are not uploaded as artifacts unless scripts/train.py gets
# --log-weights or KAPPA_WANDB_LOG_WEIGHTS=1.

export TOKENIZERS_PARALLELISM=false
# One BLAS thread per dataloader worker; NumPy otherwise sizes its pool from the machine.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

FOLD="${SLURM_ARRAY_TASK_ID:-0}"

printf 'venv    : %s\n' "$KAPPA_VENV"
printf 'node    : %s\n' "$(hostname)"
printf 'gpus    : %s\n' "${SLURM_JOB_GPUS:-?}"
printf 'cpus    : %s\n' "${SLURM_CPUS_PER_TASK:-?}"
printf 'data    : %s\n' "$KAPPA_DATA_ROOT"
printf 'fold    : %s (array element; folds are parallel jobs, never ranks)\n' "$FOLD"
printf 'started : %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

# A CUDA build newer than the driver imports cleanly and trains on CPU; refuse that.
"$PY" - <<'PYEOF'
import torch
print("torch   :", torch.__version__, "| built for CUDA", torch.version.cuda)
print("cuda    :", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit(
        "FAILED: torch.cuda.is_available() is False on a GPU node. This is the cu130 "
        "wheel against a 570 driver -- the run would train on CPU for the whole "
        "walltime and still report a macro. Use the cu126 env from "
        "slurm/env_build.sh train, and check KAPPA_TRAIN_VENV."
    )
PYEOF

# scripts/train.py reads the fold from the array index; do not pass --fold here.
# It runs in the background so this shell can receive USR1 and forward it; the
# handler in scripts/train.py commits the W&B run and exits.
"$PY" scripts/train.py "$@" &
TRAIN_PID=$!
trap 'echo "forwarding USR1 to $TRAIN_PID"; kill -USR1 "$TRAIN_PID" 2>/dev/null || true' USR1
trap 'echo "forwarding TERM to $TRAIN_PID"; kill -TERM "$TRAIN_PID" 2>/dev/null || true' TERM

# `wait` returns early when a trapped signal arrives, so retry until the child exits.
set +e
wait "$TRAIN_PID"; RC=$?
while kill -0 "$TRAIN_PID" 2>/dev/null; do wait "$TRAIN_PID"; RC=$?; done
set -e

printf 'finished: %s (rc=%s)\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$RC"
printf 'sync    : run slurm/sync_wandb.sh on the LOGIN node (WANDB_DIR=%s)\n' "$WANDB_DIR"
exit "$RC"
