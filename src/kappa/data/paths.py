"""Filesystem layout for KAPPA.

Single source of truth for where data lives. Downstream code resolves paths from
here rather than hardcoding them, so identical code runs on a laptop and on
cluster scratch. Set ``KAPPA_DATA_ROOT`` (environment or ``.env``) to relocate
everything large; ``folds_dir()`` deliberately stays in the repository because
fold assignments are tracked artifacts, not bulk data.

    from kappa.data.paths import raw_dir, require_raw

    require_raw()              # fails loudly with the command to run
    train = raw_dir() / "train.csv"

Run ``python -m kappa.data.paths`` to print the resolved layout.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "repo_root",
    "data_root",
    "raw_dir",
    "interim_dir",
    "processed_dir",
    "cache_dir",
    "folds_dir",
    "subset_dir",
    "is_raw_staged",
    "is_subset_staged",
    "require_raw",
    "require_series_root",
]

STAMP_NAME = ".kappa_stamp"

# Written by the script that built the local subset, and cross-checkable against
# disk: it records 105,917 files, the 58 labelled study UIDs and the 500 sampled
# unlabelled ones. Its presence is the test for "the external drive is mounted",
# which is a different question from `is_raw_staged`.
SUBSET_MANIFEST_NAME = "SUBSET_MANIFEST.json"


def _detect_source_root() -> Path | None:
    """Locate the checkout containing this file, or None if installed outside one.

    .../<repo>/src/kappa/data/paths.py -> <repo>. A non-editable install lands in
    site-packages, where that arithmetic is meaningless, so it is verified.
    """
    candidate = Path(__file__).resolve().parents[3]
    return candidate if (candidate / "pyproject.toml").is_file() else None


_SOURCE_ROOT = _detect_source_root()


def _load_dotenv(path: Path) -> None:
    """Populate os.environ from a KEY=VALUE file, never overwriting a set var.

    Mirrors load_dotenv in scripts/get_data.sh -- keep the two in sync.
    """
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if line.startswith("export "):
            line = line[len("export ") :]
        key, sep, value = line.partition("=")
        key = key.strip()
        if not sep or not key.replace("_", "").isalnum():
            continue
        if not os.environ.get(key):  # unset or empty, matching the shell's -n test
            os.environ[key] = value.strip().strip("\"'")


def repo_root() -> Path:
    """Repository root. Set KAPPA_REPO_ROOT when installed outside the checkout."""
    override = os.environ.get("KAPPA_REPO_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    if _SOURCE_ROOT is None:
        raise RuntimeError(
            "kappa is installed outside its source tree, so the repository root "
            "cannot be inferred. Set KAPPA_REPO_ROOT, or set KAPPA_DATA_ROOT to "
            "resolve data paths without it."
        )
    return _SOURCE_ROOT


_dotenv_root = os.environ.get("KAPPA_REPO_ROOT") or _SOURCE_ROOT
if _dotenv_root is not None:
    _load_dotenv(Path(_dotenv_root) / ".env")


def data_root() -> Path:
    """Root of all bulk data. Defaults to ``<repo>/data``, which is gitignored."""
    override = os.environ.get("KAPPA_DATA_ROOT")
    return Path(override).expanduser().resolve() if override else repo_root() / "data"


def raw_dir() -> Path:
    """Extracted competition files. Immutable once staged."""
    return data_root() / "raw"


def interim_dir() -> Path:
    """Intermediate artifacts: manifests, audits, decoded volumes."""
    return data_root() / "interim"


def processed_dir() -> Path:
    """Model-ready caches, keyed by preprocessing version."""
    return data_root() / "processed"


def cache_dir() -> Path:
    """Decoded series volumes: ``<cache>/<study_uid>/<series_uid>.npy``, uint8.

    A subdirectory of :func:`processed_dir` rather than a sibling, because it is
    exactly what that directory is for -- a model-ready artifact keyed by
    preprocessing version. The version is the directory name: a second cache at a
    different size or window goes next to this one instead of overwriting it, so
    a stale cache can never be mistaken for a fresh one.
    """
    return processed_dir() / "cache"


def folds_dir() -> Path:
    """Frozen fold assignments. Tracked in git, so always repo-relative."""
    return repo_root() / "data" / "folds"


def subset_dir() -> Path:
    """The local development subset: 558 train + 3 test studies, 68.5 GiB.

    Deliberately **not** :func:`raw_dir`. The subset is 561 of 4,410 studies, so
    code that treats it as the corpus silently computes a partial answer;
    ``raw_dir`` must keep meaning "all of it", and ``is_raw_staged`` must keep
    meaning "the full extraction finished". It resolves to
    ``$KAPPA_DATA_ROOT/subset`` by default; set ``KAPPA_SUBSET_DIR`` when the subset
    lives somewhere else, for example on an external drive while the data root
    points at cluster storage.

    It contains all 58 label anchors on purpose, so every derivation check and
    every pixel smoke test can run off the cluster.
    """
    override = os.environ.get("KAPPA_SUBSET_DIR")
    return Path(override).expanduser().resolve() if override else data_root() / "subset"


def is_raw_staged() -> bool:
    """True only when scripts/get_data.sh finished a download successfully."""
    return (raw_dir() / STAMP_NAME).is_file()


def is_subset_staged() -> bool:
    """True when the local subset is present, i.e. the external drive is mounted.

    Tests that touch real DICOM bytes gate on this and skip otherwise. Skipping
    is the honest outcome: a test that silently substitutes synthetic data for
    real data reports a pass that means nothing about the real data.
    """
    return (subset_dir() / SUBSET_MANIFEST_NAME).is_file()


def require_series_root(split: str = "train") -> Path:
    """``<root>/<split>_series`` holding ``<study_uid>/<series_uid>/<sop_uid>.dcm``.

    Prefers the full extraction and falls back to the local subset, in that
    order, so the same command works on the cluster and on a laptop. It returns
    the path it chose rather than a boolean, precisely so the caller can print
    it: the difference between the two is 4,410 studies and 561, and a pass that
    does not say which tree it read leaves that ambiguous in its own log.
    """
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    name = f"{split}_series"
    for root in (raw_dir(), subset_dir()):
        candidate = root / name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"no {name}/ under {raw_dir()} or {subset_dir()}. Run `make data` to stage "
        f"the full corpus, or mount the drive holding the local subset."
    )


def require_raw() -> Path:
    """Return the raw data directory, or explain how to create it."""
    if not is_raw_staged():
        raise FileNotFoundError(
            f"competition data is not staged at {raw_dir()}. Run:\n"
            f"    make data      # or: scripts/get_data.sh"
        )
    return raw_dir()


if __name__ == "__main__":
    for _name in (
        "repo_root",
        "data_root",
        "raw_dir",
        "interim_dir",
        "processed_dir",
        "cache_dir",
        "folds_dir",
        "subset_dir",
    ):
        print(f"{_name:14} {globals()[_name]()}")
    print(f"{'raw_staged':14} {is_raw_staged()}")
    print(f"{'subset_staged':14} {is_subset_staged()}")
