"""Guards on the local-subset and cache path helpers, and on what they must not change.

Why this exists: the local subset is 561 of 4,410 studies. Every path helper that
resolves to it is a helper that can make a partial pass look like a full one, and
the specific accident to prevent is ``raw_dir`` or ``is_raw_staged`` quietly
learning to accept the subset. ``is_raw_staged`` is the gate the data pipeline
uses to decide whether the corpus is present; if mounting a 68.5 GiB development
drive flipped it to True, every downstream count -- studies, series, slices,
label coverage -- would be computed over 12.7% of the data and reported as the
answer.

The subset assertions run only when the drive is mounted, and skip loudly
otherwise. The layout assertions run everywhere, against a temporary data root,
because those are contracts about the code rather than about a disk.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from kappa.data import paths
from kappa.data.paths import (
    SUBSET_MANIFEST_NAME,
    cache_dir,
    data_root,
    is_raw_staged,
    is_subset_staged,
    processed_dir,
    raw_dir,
    repo_root,
    require_series_root,
    subset_dir,
)

# Recorded in SUBSET_MANIFEST.json by the script that built the subset. Asserted
# rather than recomputed: a full walk of 105,917 files on an external exFAT
# volume takes minutes, and a test that slow gets skipped by the person running
# it, which is worse than a test that checks the recorded contract.
SUBSET_FILES = 105917
SUBSET_ANCHORS = 58
SUBSET_SAMPLED = 500

requires_subset = pytest.mark.skipif(
    not is_subset_staged(),
    reason="the local DICOM subset is not mounted; cannot check it against disk",
)


@pytest.fixture()
def data_root_at(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point KAPPA_DATA_ROOT at a temporary directory for one test."""
    monkeypatch.setenv("KAPPA_DATA_ROOT", str(tmp_path))
    monkeypatch.delenv("KAPPA_SUBSET_DIR", raising=False)
    return tmp_path


class TestLayout:
    """Where the new directories sit relative to the ones that already existed."""

    def test_cache_lives_under_processed(self, data_root_at: Path) -> None:
        assert cache_dir() == processed_dir() / "cache"
        assert cache_dir() == data_root_at / "processed" / "cache"

    def test_subset_is_a_sibling_of_raw_not_a_child_of_it(self, data_root_at: Path) -> None:
        assert subset_dir() == data_root_at / "subset"
        assert subset_dir().parent == raw_dir().parent

    def test_subset_dir_honours_its_own_environment_override(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The cluster case: data root on shared storage, subset on a mounted drive."""
        elsewhere = tmp_path / "mounted" / "subset"
        elsewhere.mkdir(parents=True)
        monkeypatch.setenv("KAPPA_SUBSET_DIR", str(elsewhere))
        assert subset_dir() == elsewhere.resolve()

    def test_folds_stay_in_the_repository_when_the_data_root_moves(
        self, data_root_at: Path
    ) -> None:
        # Restated here because adding two data-root-relative helpers is exactly
        # when someone makes folds_dir the third one. Fold tables are tracked
        # artifacts and must not follow KAPPA_DATA_ROOT onto cluster storage.
        assert paths.folds_dir() == repo_root() / "data" / "folds"

    def test_every_exported_name_exists(self) -> None:
        for name in paths.__all__:
            assert hasattr(paths, name), name

    def test_the_new_helpers_are_exported(self) -> None:
        for name in ("subset_dir", "cache_dir", "is_subset_staged", "require_series_root"):
            assert name in paths.__all__, name


class TestRawGateIsUnchanged:
    """The assertions that stop the subset from being mistaken for the corpus."""

    def test_raw_staged_is_false_without_the_stamp(self, data_root_at: Path) -> None:
        raw_dir().mkdir(parents=True)
        assert is_raw_staged() is False

    def test_raw_staged_is_true_only_with_the_stamp(self, data_root_at: Path) -> None:
        raw_dir().mkdir(parents=True)
        (raw_dir() / paths.STAMP_NAME).touch()
        assert is_raw_staged() is True

    def test_a_mounted_subset_does_not_make_raw_look_staged(self, data_root_at: Path) -> None:
        """The whole point of keeping the two gates separate.

        Fails if someone widens ``is_raw_staged`` or ``raw_dir`` to fall back to
        the subset. 561 studies would then satisfy a check that means 4,410, and
        every measurement taken after it would be a twelfth of the corpus
        reported as the whole.
        """
        subset_dir().mkdir(parents=True)
        (subset_dir() / SUBSET_MANIFEST_NAME).write_text("{}", encoding="utf-8")
        assert is_subset_staged() is True
        assert is_raw_staged() is False

    def test_subset_staged_is_false_without_its_manifest(self, data_root_at: Path) -> None:
        subset_dir().mkdir(parents=True)
        assert is_subset_staged() is False


class TestRequireSeriesRoot:
    """Which tree a pass actually reads, when both or neither are present."""

    def test_raw_wins_over_the_subset(self, data_root_at: Path) -> None:
        (raw_dir() / "train_series").mkdir(parents=True)
        (subset_dir() / "train_series").mkdir(parents=True)
        assert require_series_root("train") == raw_dir() / "train_series"

    def test_the_subset_is_the_fallback(self, data_root_at: Path) -> None:
        (subset_dir() / "train_series").mkdir(parents=True)
        assert require_series_root("train") == subset_dir() / "train_series"

    def test_the_split_selects_the_directory(self, data_root_at: Path) -> None:
        (subset_dir() / "test_series").mkdir(parents=True)
        assert require_series_root("test") == subset_dir() / "test_series"

    def test_neither_present_raises_with_the_command_to_run(self, data_root_at: Path) -> None:
        with pytest.raises(FileNotFoundError, match="make data"):
            require_series_root("train")

    def test_an_unknown_split_raises_rather_than_building_a_bogus_path(
        self, data_root_at: Path
    ) -> None:
        with pytest.raises(ValueError, match="train"):
            require_series_root("validation")


class TestPrinter:
    """``python -m kappa.data.paths`` is how a fresh session sees the layout."""

    def test_the_printer_reports_the_new_paths(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "kappa.data.paths"],
            capture_output=True,
            text=True,
            check=True,
            cwd=repo_root(),
            env={"PYTHONPATH": str(repo_root() / "src"), "PATH": "/usr/bin:/bin"},
        )
        for label in ("cache_dir", "subset_dir", "raw_staged", "subset_staged"):
            assert label in result.stdout, result.stdout


@requires_subset
class TestSubsetOnDisk:
    """Cross-check SUBSET_MANIFEST.json against the drive it describes."""

    def _manifest(self) -> dict:
        return json.loads((subset_dir() / SUBSET_MANIFEST_NAME).read_text(encoding="utf-8"))

    def test_the_manifest_records_the_expected_shape_of_the_subset(self) -> None:
        manifest = self._manifest()
        assert manifest["files"] == SUBSET_FILES
        assert len(manifest["labelled_studies"]) == SUBSET_ANCHORS
        assert len(manifest["sampled_studies"]) == SUBSET_SAMPLED

    def test_every_one_of_the_58_anchors_is_actually_on_disk(self) -> None:
        """The anchors are the project's only ground truth; a missing one is silent.

        The derivation was validated against these 58 studies. If the subset were
        missing some of them, an image-side check against ground truth would run
        on fewer anchors than it reported.
        """
        root = subset_dir() / "train_series"
        missing = [uid for uid in self._manifest()["labelled_studies"] if not (root / uid).is_dir()]
        assert missing == []

    def test_the_series_root_resolves_inside_the_subset(self) -> None:
        # Local machine only: raw_dir() is not staged here, so the fallback is
        # the path actually exercised by every real-byte test in this suite.
        resolved = require_series_root("train")
        assert resolved.is_dir()
        assert resolved.name == "train_series"
        assert data_root() in resolved.parents

    def test_a_study_directory_holds_series_directories_holding_dicoms(self) -> None:
        root = require_series_root("train")
        study = next(p for p in sorted(root.iterdir()) if p.is_dir() and p.name[0] != ".")
        series = [p for p in sorted(study.iterdir()) if p.is_dir() and p.name[0] != "."]
        assert series, f"{study.name} has no series directories"
        files = [p for p in series[0].glob("*.dcm") if not p.name.startswith("._")]
        assert files, f"{series[0].name} has no DICOMs once sidecars are excluded"
