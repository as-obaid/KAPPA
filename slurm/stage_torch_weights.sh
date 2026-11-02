#!/bin/bash
# Download torchvision's 3D video weights into $TORCH_HOME. Login node only: it needs the network.
#
#   slurm/stage_torch_weights.sh              # r3d_18
#   slurm/stage_torch_weights.sh r3d_18 mc3_18
#   slurm/stage_torch_weights.sh --check      # verify only, download nothing
#
# Expects: KAPPA_SCRATCH (shell or .env). TORCH_HOME defaults to
# $KAPPA_SCRATCH/torch_home; files land in $TORCH_HOME/hub/checkpoints.
#
# Notes:
# - torchvision does not use the Hugging Face cache, so slurm/stage_weights.sh does
#   not cover these, and torchvision's downloader hangs rather than fails offline.
# - Uses curl and a sha256 prefix check instead of importing torch, so it fits in a
#   memory-capped login node. torch.hub validates the same prefix on load.

set -euo pipefail
cd "$(dirname "$0")/.."

# Same TORCH_HOME the jobs read from. Staging needs the network.
source "$(dirname "$0")/env_common.sh"
unset HF_HUB_OFFLINE

CHECK_ONLY=0
MODELS=()
for arg in "$@"; do
  case "$arg" in
    --check) CHECK_ONLY=1;;
    -*) printf 'unknown argument: %s\n' "$arg" >&2; exit 2;;
    *) MODELS+=("$arg");;
  esac
done
# r3d_18 by default; mc3_18 and r2plus1d_18 are also valid `model.backbone` values.
if [ ${#MODELS[@]} -eq 0 ]; then
  MODELS=("r3d_18")
fi

# Mirrors kappa.models.volume_encoder._weights_enum so torch is never imported;
# tests/test_volume.py checks the URLs against torchvision's enums.
url_for() {
  case "$1" in
    r3d_18)      printf 'https://download.pytorch.org/models/r3d_18-b3b3357e.pth';;
    mc3_18)      printf 'https://download.pytorch.org/models/mc3_18-a90a0ba3.pth';;
    r2plus1d_18) printf 'https://download.pytorch.org/models/r2plus1d_18-91a641e6.pth';;
    *)           printf '';;
  esac
}

# sha256 of a file, with GNU coreutils or macOS shasum.
sha_of() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
  else shasum -a 256 "$1" | cut -d' ' -f1
  fi
}

DEST="$TORCH_HOME/hub/checkpoints"
mkdir -p "$DEST"
printf 'TORCH_HOME : %s\n' "$TORCH_HOME"
printf 'dest       : %s\n' "$DEST"
printf 'models     : %s\n' "${MODELS[*]}"
printf 'mode       : %s\n\n' "$([ "$CHECK_ONLY" -eq 1 ] && echo verify-only || echo download)"

FAILED=0
for model in "${MODELS[@]}"; do
  url="$(url_for "$model")"
  if [ -z "$url" ]; then
    printf '  FAILED  : unknown backbone %s\n' "$model" >&2
    FAILED=$((FAILED + 1))
    continue
  fi
  file="${url##*/}"
  target="$DEST/$file"
  # `r3d_18-b3b3357e.pth` -> `b3b3357e`, the prefix torch.hub validates on load.
  want="${file#*-}"; want="${want%.pth}"

  if [ ! -f "$target" ]; then
    if [ "$CHECK_ONLY" -eq 1 ]; then
      printf '  MISSING : %-12s %s\n' "$model" "$target"
      FAILED=$((FAILED + 1))
      continue
    fi
    printf '  fetching: %-12s <- %s\n' "$model" "$url"
    # --fail so an error page is never saved as a .pth; download to a temporary
    # name so a killed transfer cannot leave a half-file that looks done.
    if ! curl -fSL --retry 3 --retry-delay 2 -o "$target.partial" "$url"; then
      printf '  FAILED  : %s could not be downloaded\n' "$model" >&2
      rm -f "$target.partial"
      FAILED=$((FAILED + 1))
      continue
    fi
    mv "$target.partial" "$target"
  fi

  got="$(sha_of "$target")"
  case "$got" in
    "$want"*)
      printf '  ok      : %-12s %s (%s MB, sha %s...)\n' \
             "$model" "$target" "$(( $(wc -c <"$target") / 1000000 ))" "$want";;
    *)
      printf '  FAILED  : %s sha256 is %s, expected to start %s -- corrupt download\n' \
             "$target" "${got:0:16}" "$want" >&2
      printf '            delete it and re-run.\n' >&2
      FAILED=$((FAILED + 1));;
  esac
done

if [ "$FAILED" -gt 0 ]; then
  printf '\nFAILED for %s model(s).\n' "$FAILED" >&2
  [ "$CHECK_ONLY" -eq 1 ] && printf 'Run without --check on the LOGIN node to download.\n' >&2
  exit 1
fi

printf '\n=== cache ===\n'
du -sh "$TORCH_HOME" 2>/dev/null || true
printf '\nJobs find these weights when TORCH_HOME=%s is exported before sbatch.\n' "$TORCH_HOME"
