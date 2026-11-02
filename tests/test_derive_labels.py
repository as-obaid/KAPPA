"""A short label table must be impossible to produce by accident.

The extraction array writes one file per shard, so deriving labels normally
means reading eight inputs rather than one. Both ways of getting that wrong are
silent: pass seven of the eight and the label table is short by roughly 550
studies while looking perfectly healthy, or pass a stale file from a different
shard count and the table is built from two disagreeing partitions. Neither
raises anything on its own, and neither is visible in the output -- a table with
3,850 rows looks exactly as plausible as one with 4,407.

So these tests exercise the resolution and concatenation directly, and assert
that the failure cases are refused rather than absorbed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from derive_labels import load_all_records, resolve_record_paths  # noqa: E402


def _record(uid: str, description: str = "medial meniscus tear") -> dict:
    return {
        "_study_uid": uid,
        "language": "English",
        "knee_side": "left",
        "knee_side_evidence": "left knee",
        "_parsed": True,
        "_valid": True,
        "findings": [
            {
                "span": description,
                "description": description,
                "anatomy": "medial meniscus",
                "assertion": "affirmed",
                "severity_wording": None,
            }
        ],
    }


def _write(path: Path, uids: list[str]) -> Path:
    path.write_text(
        "".join(json.dumps(_record(uid)) + "\n" for uid in uids), encoding="utf-8"
    )
    return path


def _shards(root: Path, groups: dict[str, list[str]]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for name, uids in groups.items():
        _write(root / name, uids)
    return root


class TestResolveRecordPaths:
    def test_a_directory_expands_to_its_shard_files_in_order(self, tmp_path):
        root = _shards(
            tmp_path / "extraction",
            {
                "records.shard01of08.jsonl": ["b"],
                "records.shard00of08.jsonl": ["a"],
            },
        )
        assert [p.name for p in resolve_record_paths([root], tmp_path)] == [
            "records.shard00of08.jsonl",
            "records.shard01of08.jsonl",
        ]

    def test_an_unmatched_glob_raises_instead_of_contributing_nothing(self, tmp_path):
        _shards(tmp_path / "extraction", {"records.shard00of08.jsonl": ["a"]})
        with pytest.raises(FileNotFoundError):
            resolve_record_paths([tmp_path / "extraction" / "records.shard9*.jsonl"], tmp_path)

    def test_the_same_directory_named_twice_yields_each_file_once(self, tmp_path):
        root = _shards(tmp_path / "extraction", {"records.shard00of08.jsonl": ["a"]})
        assert len(resolve_record_paths([root, root], tmp_path)) == 1

    def test_shard_files_are_preferred_over_a_plain_records_file(self, tmp_path):
        root = tmp_path / "extraction"
        _shards(root, {"records.shard00of08.jsonl": ["a"]})
        _write(root / "records.jsonl", ["stale"])
        assert [p.name for p in resolve_record_paths(None, root)] == [
            "records.shard00of08.jsonl"
        ]

    def test_a_plain_records_file_is_the_fallback_when_no_shards_exist(self, tmp_path):
        root = tmp_path / "discovery"
        root.mkdir(parents=True)
        _write(root / "records.jsonl", ["a"])
        assert [p.name for p in resolve_record_paths(None, root)] == ["records.jsonl"]

    def test_an_empty_directory_resolves_to_nothing_rather_than_guessing(self, tmp_path):
        (tmp_path / "extraction").mkdir()
        assert resolve_record_paths(None, tmp_path / "extraction") == []


class TestLoadAllRecords:
    def test_disjoint_shards_concatenate_to_the_sum_of_their_records(self, tmp_path):
        root = _shards(
            tmp_path / "e",
            {
                "records.shard00of08.jsonl": ["a0", "a1", "a2"],
                "records.shard01of08.jsonl": ["b0", "b1"],
            },
        )
        records, duplicates = load_all_records(sorted(root.glob("records.shard*")))
        assert len(records) == 5
        assert duplicates == []

    def test_a_study_in_two_inputs_is_reported_and_not_silently_merged(self, tmp_path):
        # The real scenario: a leftover file from a run at a different shard
        # count, which overlaps the current partition.
        root = _shards(
            tmp_path / "e",
            {"records.shard00of08.jsonl": ["a0", "a1"], "stale.jsonl": ["a0"]},
        )
        records, duplicates = load_all_records(
            [root / "records.shard00of08.jsonl", root / "stale.jsonl"]
        )
        assert len(duplicates) == 1
        assert "a0" in duplicates[0]
        # The first occurrence is kept so the caller can still report counts,
        # but the duplicate list is what makes the run fail.
        assert len(records) == 2

    def test_a_record_with_no_study_uid_is_kept_rather_than_dropped(self, tmp_path):
        # A record can fail extraction and still carry diagnostics worth keeping;
        # dropping it here would lose a counted failure.
        path = tmp_path / "records.shard00of08.jsonl"
        path.write_text(
            json.dumps({"_parsed": False, "_valid": False}) + "\n", encoding="utf-8"
        )
        records, duplicates = load_all_records([path])
        assert len(records) == 1
        assert duplicates == []

    def test_blank_lines_do_not_become_records(self, tmp_path):
        path = tmp_path / "records.shard00of08.jsonl"
        path.write_text(
            json.dumps(_record("a")) + "\n\n   \n" + json.dumps(_record("b")) + "\n",
            encoding="utf-8",
        )
        records, _ = load_all_records([path])
        assert len(records) == 2
