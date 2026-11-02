#!/usr/bin/env bash
# Download and stage the competition dataset. Idempotent: returns immediately if
# the data is already present, so any downstream target can depend on it.
#
# Layout, relative to KAPPA_DATA_ROOT (default: <repo>/data):
#   raw/               extracted competition files, treated as immutable
#   raw/.kappa_stamp   completion marker -- its presence means raw/ is whole
#   .stage/            partial archive, kept across runs so a retry resumes
#
# Configuration precedence: flags > environment > .env > defaults.
# On a cluster, point the data root at scratch:
#   echo "KAPPA_DATA_ROOT=/path/to/large/disk/kappa" > .env    # then: make data
#
# Usage: scripts/get_data.sh [--force] [--keep-archive] [--data-root DIR]

set -euo pipefail

readonly STAMP_NAME=".kappa_stamp"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd -- "$script_dir/.." && pwd)"

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
log() { printf '==> %s\n' "$*"; }

# Load .env without clobbering variables already set in the environment.
# Mirrors _load_dotenv in src/kappa/data/paths.py -- keep the two in sync.
load_dotenv() {
  local file="$1" line key val
  [ -f "$file" ] || return 0
  while IFS= read -r line || [ -n "$line" ]; do
    line="${line%$'\r'}"
    line="${line#export }"
    case "$line" in *=*) ;; *) continue ;; esac
    key="${line%%=*}"
    val="${line#*=}"
    case "$key" in ''|*[!A-Za-z0-9_]*) continue ;; esac
    val="${val%\"}"; val="${val#\"}"
    val="${val%\'}"; val="${val#\'}"
    [ -n "${!key-}" ] || export "$key=$val"
  done < "$file"
}
load_dotenv "$REPO_ROOT/.env"

competition="${KAPPA_COMPETITION:-rsna-knee-abnormality-detection}"
data_root="${KAPPA_DATA_ROOT:-$REPO_ROOT/data}"
force=0
keep_archive=0
phase=all

while [ $# -gt 0 ]; do
  case "$1" in
    --force)        force=1 ;;
    --keep-archive) keep_archive=1 ;;
    --fetch-only)   phase=fetch ;;
    --extract-only) phase=extract ;;
    --data-root)    [ $# -ge 2 ] || die "--data-root needs a directory"; data_root="$2"; shift ;;
    -h|--help)      sed -n '2,14p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)              die "unknown argument: $1 (try --help)" ;;
  esac
  shift
done

raw_dir="$data_root/raw"
stage_dir="$data_root/.stage"
stamp="$raw_dir/$STAMP_NAME"

# --- Fast path: already staged -----------------------------------------------
if [ -f "$stamp" ] && [ "$force" -eq 0 ]; then
  log "dataset already staged at $raw_dir"
  sed 's/^/    /' "$stamp"
  exit 0
fi

# --- Preconditions -----------------------------------------------------------
command -v curl   >/dev/null 2>&1 || die "curl not found; it is required to fetch the archive"
command -v unzip  >/dev/null 2>&1 || die "unzip not found; install it, or extract the archive manually"

if [ -z "${KAGGLE_USERNAME:-}" ] || [ -z "${KAGGLE_KEY:-}" ]; then
  cred="${KAGGLE_CONFIG_DIR:-$HOME/.kaggle}/kaggle.json"
  [ -f "$cred" ] || die "no Kaggle credentials. Save kaggle.json to $cred (Kaggle > Settings > Create New Token), or export KAGGLE_USERNAME and KAGGLE_KEY"
  # The CLI refuses to run on a group- or world-readable token.
  if [ "$(stat -f '%OLp' "$cred" 2>/dev/null || stat -c '%a' "$cred")" != "600" ]; then
    log "tightening permissions on $cred to 600"
    chmod 600 "$cred"
  fi
fi

# Never silently discard data this script did not place. An interrupted run
# cannot trigger this: raw/ only ever appears via the atomic rename below.
if [ "$force" -eq 0 ] && [ -d "$raw_dir" ] && [ ! -f "$stamp" ] && [ -n "$(ls -A "$raw_dir" 2>/dev/null)" ]; then
  die "$raw_dir has unstamped contents. If it is already a complete copy, stamp it:
    printf 'competition=%s\npre_staged=true\n' '$competition' > '$stamp'
  Otherwise re-run with --force to replace it."
fi

mkdir -p "$stage_dir"
avail_gib=$(df -Pk "$data_root" | awk 'NR==2 {printf "%d", $4/1048576}')

log "competition : $competition"
log "destination : $raw_dir"
log "free space  : ${avail_gib} GiB"

# Kaggle serves the whole competition as one archive, so the zip and the
# extracted tree coexist until extraction finishes. Measured: the
# archive is 265,018,885,676 bytes (247 GiB) and extracts to 530.6 GiB, so peak
# use is ~778 GiB. 800 leaves headroom. Set KAPPA_MIN_FREE_GIB=0 to skip.
min_free_gib="${KAPPA_MIN_FREE_GIB:-800}"
if [ "$min_free_gib" -gt 0 ] && [ "$avail_gib" -lt "$min_free_gib" ]; then
  die "not enough free space at $data_root: ${avail_gib} GiB available, ~${min_free_gib} GiB needed.
  The archive and the extracted tree must coexist, so peak use is about twice the
  dataset size. Options:
    - stage on cluster scratch instead (KAPPA_DATA_ROOT), see slurm/README.md
    - fetch a subset for local work rather than the full dataset
    - override with KAPPA_MIN_FREE_GIB=<gib> if you know the space is there"
fi

# --- Download ----------------------------------------------------------------
# The kaggle CLI is not used here. On this competition it never begins
# transferring: it pins a core enumerating the 819,640-file paginated listing and
# writes zero bytes (observed 3+ minutes, CLI 1.7.4.5). The download-all endpoint
# streams the archive directly, reports a correct Content-Length, and honours
# Range, so an interrupted transfer resumes with curl -C -.
archive="$stage_dir/${competition}.zip"
endpoint="https://www.kaggle.com/api/v1/competitions/data/download-all/${competition}"

if [ -n "${KAGGLE_USERNAME:-}" ] && [ -n "${KAGGLE_KEY:-}" ]; then
  k_user="$KAGGLE_USERNAME"; k_key="$KAGGLE_KEY"
else
  cred="${KAGGLE_CONFIG_DIR:-$HOME/.kaggle}/kaggle.json"
  k_user=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["username"])' "$cred")
  k_key=$(python3  -c 'import json,sys; print(json.load(open(sys.argv[1]))["key"])'      "$cred")
fi

if [ "$phase" = extract ]; then
  log "skipping download (--extract-only)"
else
log "downloading (resumes an interrupted transfer; the archive is ~247 GiB)"
# Credentials go in via a stdin config file, never in argv, so they cannot leak
# through `ps` or a job's stdout.
printf 'user = "%s:%s"\n' "$k_user" "$k_key" |
  curl -fSL --retry 5 --retry-delay 10 -C - --progress-bar \
       -o "$archive" -K - "$endpoint" || \
  die "download failed. Confirm you have accepted the competition rules at
  https://kaggle.com/competitions/$competition/rules
  and that the API token is valid. Re-run to resume from where it stopped."
fi

if [ "$phase" = fetch ]; then
  log "archive fetched; stopping before extraction (--fetch-only)"
  log "extract on a compute node:  sbatch slurm/get_data.sbatch"
  exit 0
fi

shopt -s nullglob
archives=("$stage_dir"/*.zip)
shopt -u nullglob
[ "${#archives[@]}" -eq 1 ] || \
  die "expected exactly one .zip in $stage_dir, found ${#archives[@]}. Remove stale archives and retry."
archive="${archives[0]}"

# `unzip -t` decompresses every entry to check CRCs -- 530 GiB of work. It must not
# run on a login node: the watchdog kills it, and a killed verifier is not a corrupt
# archive. Distinguish the two, because the wrong message here costs a 247 GiB
# re-download. Exit codes >= 128 mean death by signal.
if [ "${KAPPA_SKIP_VERIFY:-0}" = 1 ]; then
  log "skipping CRC check (KAPPA_SKIP_VERIFY=1)"
else
log "verifying archive integrity (decompresses everything; needs a compute node)"
set +e
unzip -tqq "$archive"
verify_rc=$?
set -e
if [ "$verify_rc" -ge 128 ]; then
  die "integrity check was killed by a signal (exit $verify_rc), which says nothing
  about the archive. Do NOT delete it. Re-run this step inside a batch job:
    sbatch slurm/get_data.sbatch
  To skip the CRC check when size and central directory already match, use
  KAPPA_SKIP_VERIFY=1."
elif [ "$verify_rc" -ne 0 ]; then
  die "$archive is genuinely corrupt (unzip -t exit $verify_rc). Delete it and re-run."
fi
fi

# --- Extract, then publish atomically ----------------------------------------
extract_dir="$data_root/.raw.incoming.$$"
trap 'rm -rf "$extract_dir"' EXIT
mkdir -p "$extract_dir"

log "extracting"
unzip -q "$archive" -d "$extract_dir"

log "writing $STAMP_NAME"
{
  printf 'competition=%s\n' "$competition"
  printf 'completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'files=%s\n' "$(find "$extract_dir" -type f | wc -l | tr -d ' ')"
  printf 'kibibytes=%s\n' "$(du -sk "$extract_dir" | cut -f1)"
} > "$extract_dir/$STAMP_NAME"

# Rename is atomic, so raw/ is never observed half-populated.
retired=""
if [ -e "$raw_dir" ]; then
  retired="$data_root/.raw.retired.$$"
  mv "$raw_dir" "$retired"
fi
mv "$extract_dir" "$raw_dir"
trap - EXIT
if [ -n "$retired" ]; then
  rm -rf "$retired"
fi

if [ "$keep_archive" -eq 1 ]; then
  log "keeping archive at $archive"
else
  log "removing archive (pass --keep-archive to retain it)"
  rm -f "$archive"
  rmdir "$stage_dir" 2>/dev/null || true
fi

log "staged $raw_dir"
sed 's/^/    /' "$raw_dir/$STAMP_NAME"
