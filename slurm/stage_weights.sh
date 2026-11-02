#!/bin/bash
# Download Hugging Face model weights into $HF_HOME. Login node only: it needs the network.
#
#   slurm/stage_weights.sh                         # the two default extraction models
#   slurm/stage_weights.sh google/gemma-4-31B-it
#   tmux new -s stage 'slurm/stage_weights.sh'     # the defaults are ~111 GiB
#
# Expects: KAPPA_SCRATCH (shell or .env); KAPPA_VENV with the `hf` CLI installed.
# HF_HOME defaults to $KAPPA_SCRATCH/hf_cache, the same cache jobs read with
# HF_HUB_OFFLINE=1. HF_TOKEN, from the shell or .env, lifts the anonymous rate limit.
#
# Notes:
# - Compute nodes have no outbound internet, so weights must be staged before submitting.

set -euo pipefail

source "$(dirname "$0")/env_common.sh"
unset HF_HUB_OFFLINE

# Read HF_TOKEN from .env if the shell does not set it. Never echo it.
ENV_FILE="$(dirname "$0")/../.env"
if [ -z "${HF_TOKEN:-}" ] && [ -f "$ENV_FILE" ]; then
  _tok=$(sed -nE 's/^(export )?HF_TOKEN=["'"'"']?([^"'"'"' ]+)["'"'"']?.*/\2/p' "$ENV_FILE" | head -1)
  [ -n "$_tok" ] && export HF_TOKEN="$_tok"
  unset _tok
fi
if [ -n "${HF_TOKEN:-}" ]; then
  echo "auth: HF_TOKEN loaded (${#HF_TOKEN} chars)"
else
  echo "auth: no HF_TOKEN -- anonymous, rate-limited" >&2
fi

VENV="$KAPPA_VENV"
PY=$VENV/bin/python
HF=$VENV/bin/hf

# huggingface_hub raises if HF_HUB_ENABLE_HF_TRANSFER is set without hf_transfer,
# so only enable it when the package is importable.
if $PY -c "import hf_transfer" 2>/dev/null; then
  export HF_HUB_ENABLE_HF_TRANSFER=1
else
  echo "note: hf_transfer absent, using the slower pure-python downloader" >&2
fi

MODELS=("$@")
if [ ${#MODELS[@]} -eq 0 ]; then
  MODELS=("Qwen/Qwen3.5-27B" "google/gemma-4-31B-it")
fi

# `hf download` takes one pattern per --exclude; repeat the flag, never pass a list.
EXCLUDE=(--exclude "*.pth" --exclude "*.bin" --exclude "original/*")

for model in "${MODELS[@]}"; do
  echo "=== $model ==="
  $HF download "$model" "${EXCLUDE[@]}" \
    || { echo "FAILED: $model" >&2; exit 1; }
  echo
done

echo "=== cache size ==="
du -sh "$HF_HOME"
