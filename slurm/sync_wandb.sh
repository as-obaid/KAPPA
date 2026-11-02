#!/bin/bash
# Upload offline W&B runs. Login node only: it needs the network.
#
#   slurm/sync_wandb.sh
#   KAPPA_WANDB_RESYNC=1 slurm/sync_wandb.sh    # re-upload runs already marked synced
#
# Expects: KAPPA_SCRATCH (shell or .env); WANDB_API_KEY (shell or .env); a wandb client
# in $KAPPA_VENV. Runs are read from $WANDB_DIR/wandb/offline-run-*, with WANDB_DIR
# defaulting to $KAPPA_SCRATCH/kappa-work/wandb, as in slurm/train.sbatch.
#
# Notes:
# - Uses `wandb sync --legacy`; with wandb 0.29 the default sync path can hang.
# - Exit status is 1 only when something was lost or failed since the last pass. A
#   run with a sub-1 KiB transaction log was killed before wandb committed it; it is
#   reported once, recorded in $WANDB_DIR/lost_runs.ledger (KAPPA_WANDB_LEDGER), and
#   not re-reported. Delete the ledger to re-arm.
# - Artifact blobs are staged outside the run directory (WANDB_DATA_DIR) and recorded
#   by absolute path, so sync before anything purges that directory.

set -euo pipefail
cd "$(dirname "$0")/.."

source "$(dirname "$0")/env_common.sh"
unset HF_HUB_OFFLINE
export WANDB_MODE=online

# WANDB_API_KEY from .env if the shell does not set it. Never echo it.
if [ -z "${WANDB_API_KEY:-}" ] && [ -f ./.env ]; then
  _k=$(sed -nE 's/^(export )?WANDB_API_KEY=["'"'"']?([^"'"'"' ]+)["'"'"']?.*/\2/p' ./.env | head -1)
  [ -n "$_k" ] && export WANDB_API_KEY="$_k"
  unset _k
fi
[ -z "${WANDB_API_KEY:-}" ] && { echo "no WANDB_API_KEY; nothing to sync with" >&2; exit 1; }

WANDB="$KAPPA_VENV/bin/wandb"
[ -x "$WANDB" ] || { echo "no wandb client at $WANDB (check KAPPA_VENV)" >&2; exit 1; }

shopt -s nullglob
runs=("$WANDB_DIR"/wandb/offline-run-*)
if [ ${#runs[@]} -eq 0 ]; then echo "no offline runs under $WANDB_DIR"; exit 0; fi

echo "syncing ${#runs[@]} offline run(s)"

EMPTY_NEW=0
EMPTY_KNOWN=0
INPROGRESS=0
FAILED=0
SKIPPED=0
ATTEMPTED_DIRS=()

# Lost runs stay on disk, so each is reported once and then recorded here. The
# ledger lives beside the runs, not in the repo, so it never blocks a git pull.
LEDGER="${KAPPA_WANDB_LEDGER:-$WANDB_DIR/lost_runs.ledger}"
mkdir -p "$(dirname "$LEDGER")"
[ -f "$LEDGER" ] || : >"$LEDGER"

# A run still being written also has a small log, so only call it lost once it
# has gone untouched for KAPPA_WANDB_STALE_H hours.
STALE_SECONDS=$(( ${KAPPA_WANDB_STALE_H:-2} * 3600 ))
NOW=$(date +%s)

_triage_lost() {
  # $1 run directory, $2 human reason, $3 path whose mtime decides staleness
  local d="$1" reason="$2" probe="$3" base age mtime
  base=$(basename "$d")
  mtime=$(stat -c %Y "$probe" 2>/dev/null || stat -f %m "$probe" 2>/dev/null || printf '%s' "$NOW")
  age=$(( NOW - mtime ))
  if [ "$age" -lt "$STALE_SECONDS" ]; then
    printf 'IN PROGRESS (%s, %ds since last write -- not a loss yet): %s\n' "$reason" "$age" "$d"
    INPROGRESS=$((INPROGRESS + 1))
    return 0
  fi
  if grep -qxF "$base" "$LEDGER" 2>/dev/null; then
    printf 'lost, already reported (%s): %s\n' "$reason" "$d"
    EMPTY_KNOWN=$((EMPTY_KNOWN + 1))
    return 0
  fi
  printf 'LOST -- NEW (%s): %s\n' "$reason" "$d"
  printf '%s\n' "$base" >>"$LEDGER"
  EMPTY_NEW=$((EMPTY_NEW + 1))
}

for d in "${runs[@]}"; do
  log=("$d"/run-*.wandb)
  if [ ${#log[@]} -eq 0 ]; then
    _triage_lost "$d" "no transaction log" "$d"; continue
  fi
  bytes=$(wc -c < "${log[0]}")
  if [ "$bytes" -lt 1024 ]; then
    _triage_lost "$d" "${bytes} bytes, killed before wandb committed" "${log[0]}"
    continue
  fi
  # Skip runs already synced. wandb writes `run-*.wandb.synced` beside the log on
  # success (not `*.synced`). Check the array length first: under `set -u` an
  # empty array cannot be indexed.
  synced=("$d"/run-*.wandb.synced)
  if [ "${KAPPA_WANDB_RESYNC:-0}" != "1" ] && [ ${#synced[@]} -gt 0 ]; then
    SKIPPED=$((SKIPPED + 1)); continue
  fi
  echo "--- $d (${bytes} bytes)"
  ATTEMPTED_DIRS+=("$d")
  if ! "$WANDB" sync --legacy "$d" 2>&1; then
    echo "ERROR syncing $d"
    FAILED=$((FAILED + 1))
  fi
done

# Trust the marker on disk, not the exit code of `wandb sync`.
UNMARKED=0
for d in "${ATTEMPTED_DIRS[@]}"; do
  marker=("$d"/run-*.wandb.synced)
  if [ ${#marker[@]} -eq 0 ]; then
    echo "UNMARKED (no run-*.wandb.synced after sync): $d"
    UNMARKED=$((UNMARKED + 1))
  fi
done

echo "summary : ${#runs[@]} offline dir(s), $SKIPPED already synced, ${#ATTEMPTED_DIRS[@]} attempted, $EMPTY_NEW newly lost, $EMPTY_KNOWN previously lost, $INPROGRESS in progress, $FAILED sync error(s), $UNMARKED unmarked after sync"

if [ "$EMPTY_NEW" -gt 0 ]; then
  echo "WARNING: $EMPTY_NEW run(s) were lost SINCE THE LAST PASS and were not synced."
  echo "  A run killed at walltime loses everything unless the job's USR1 trap"
  echo "  fired. Check the job's .out for 'committing the wandb run'."
  echo "  Recorded in $LEDGER so the next pass does not re-report them."
fi
if [ "$EMPTY_KNOWN" -gt 0 ]; then
  echo "note   : $EMPTY_KNOWN previously-reported lost run(s) still on disk (ledger: $LEDGER)"
fi

# Previously reported losses are deliberately left out of the exit code.
if [ "$EMPTY_NEW" -gt 0 ] || [ "$FAILED" -gt 0 ] || [ "$UNMARKED" -gt 0 ]; then
  echo "done, WITH NEW LOSSES -- see summary line above. Exit code is nonzero on purpose."
  exit 1
fi
echo "done"
