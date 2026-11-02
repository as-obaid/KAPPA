"""A walltime kill must cost one batch, and a resume must re-extract exactly it.

Every test here stands in for a specific way the full-corpus extraction pass has
already failed or can still fail silently. The pass is 4,262 model calls on a
node whose partition is capped at 1 hour, so the interesting cases are all
"finished halfway": a half-written final line, a shard that moves between runs,
a duplicate text extracted twice, a study that no record covers. Each of those
produces a plausible-looking output file rather than an error, which is why they
are tested rather than watched for.

The module under test is deliberately free of vLLM, so all of this runs on a
laptop with no GPU.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from kappa.labels.normalise import content_hash, fold_lookalikes, normalise_report
from kappa.labels.shard import (
    STUDY_KEY, RecordWriter, WorkUnit, describe_units, load_records, missing_studies,
    parse_shard, pending, plan_units, read_done, records_for_unit, select_shard,
    shard_index, shard_path, shard_paths,
)


#: The subprocess-stability test runs a fresh interpreter from here.
_REPO_ROOT = str(Path(__file__).resolve().parents[1])


def _row(uid, report="Normal knee.", script="latin", is_anchor=False, **over):
    row = {
        "study_uid": uid,
        "report": report,
        "script": script,
        "hash": content_hash(normalise_report(report)),
        "is_anchor": is_anchor,
    }
    row.update(over)
    return row


def _unit(**over):
    fields = {
        "content_hash": "deadbeef",
        "report": "Normal knee.",
        "script": "latin",
        "primary_uid": "s1",
        "study_uids": ("s1",),
        "group_size": 1,
    }
    fields.update(over)
    return WorkUnit(**fields)


def _corpus(n, prefix="s"):
    return [_row(f"{prefix}{i}", report=f"Report number {i}.") for i in range(n)]


class TestPlanUnits:
    def test_one_unit_per_distinct_text_not_per_study(self):
        rows = [_row("a", "Same text."), _row("b", "Same text."), _row("c", "Other.")]
        units = plan_units(rows)
        assert len(units) == 2
        assert sum(unit.group_size for unit in units) == 3

    def test_shared_text_names_every_study_that_shares_it(self):
        units = plan_units([_row("b", "Same."), _row("a", "Same.")])
        assert units[0].study_uids == ("a", "b")
        assert units[0].group_size == 2

    def test_primary_uid_is_the_lowest_uid_so_it_survives_a_row_reorder(self):
        forward = plan_units([_row("a", "Same."), _row("b", "Same.")])
        reversed_ = plan_units([_row("b", "Same."), _row("a", "Same.")])
        assert forward[0].primary_uid == reversed_[0].primary_uid == "a"

    def test_plan_is_identical_under_a_row_reorder(self):
        rows = _corpus(20)
        assert plan_units(rows) == plan_units(list(reversed(rows)))

    def test_a_group_holding_an_anchor_is_flagged_whole(self):
        units = plan_units([_row("a", "Same.", is_anchor=True), _row("b", "Same.")])
        assert units[0].is_anchor is True
        assert units[0].study_uids == ("a", "b")

    def test_a_row_with_no_precomputed_hash_is_still_grouped(self):
        rows = [{"study_uid": "a", "report": "Same text."},
                {"study_uid": "b", "report": "Same text."}]
        assert len(plan_units(rows)) == 1

    def test_describe_units_counts_the_calls_the_dedup_saved(self):
        units = plan_units([_row("a", "Same."), _row("b", "Same."), _row("c", "X.")])
        assert describe_units(units) == (
            "2 calls covering 3 studies (1 duplicate groups over 2 studies; "
            "1 calls saved)"
        )


class TestHashingOrder:
    """`fold_lookalikes` belongs before matching and never before hashing."""

    def test_lookalike_folding_is_not_applied_to_the_grouping_key(self):
        # U+2206 INCREMENT against U+0394 GREEK CAPITAL DELTA: the corpus writes
        # the first where it means the second. Folding is right for matching and
        # wrong here -- two texts that differ only by a codepoint the fold
        # erases are still two distinct texts, and merging them would drop one
        # study's extraction on the floor.
        a, b = "∆EΞIOY knee.", "ΔEΞIOY knee."
        assert fold_lookalikes(a) == fold_lookalikes(b)
        assert len(plan_units([_row("a", a), _row("b", b)])) == 2

    def test_whitespace_only_differences_do_group(self):
        # normalise_report collapses whitespace, so this pair is one call. The
        # measured effect on the real corpus: 4,276 distinct raw texts against
        # 4,262 after normalisation.
        assert len(plan_units([_row("a", "Tear  of\tACL."), _row("b", "Tear of ACL.")])) == 1


class TestShardIndex:
    def test_index_is_stable_across_processes(self):
        # The bug this guards: `hash()` of a str is salted per interpreter, so a
        # resumed shard would cover a different set than the run it resumed.
        # Anything not reproducible in a fresh process fails here.
        keys = [f"key{i}" for i in range(50)]
        expected = [shard_index(key, 8) for key in keys]
        code = (
            "import sys; sys.path.insert(0, 'src');"
            "from kappa.labels.shard import shard_index;"
            f"print([shard_index(k, 8) for k in {keys!r}])"
        )
        env = dict(os.environ, PYTHONHASHSEED="random")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                             text=True, env=env, cwd=_REPO_ROOT, check=True)
        assert json.loads(out.stdout) == expected

    def test_index_is_in_range(self):
        assert all(0 <= shard_index(f"k{i}", 5) < 5 for i in range(200))

    def test_a_non_hex_key_is_accepted(self):
        assert 0 <= shard_index("1.2.840.113619.2.1", 8) < 8

    def test_zero_shards_is_refused(self):
        with pytest.raises(ValueError):
            shard_index("k", 0)


class TestSelectShard:
    def test_shards_are_disjoint_and_their_union_is_complete(self):
        units = plan_units(_corpus(300))
        seen: list[str] = []
        for index in range(8):
            seen.extend(unit.content_hash for unit in select_shard(units, index, 8))
        assert len(seen) == len(units)
        assert set(seen) == {unit.content_hash for unit in units}

    def test_a_shared_text_lands_in_exactly_one_shard(self):
        # Sharding after de-duplication is what makes this true. Sharding by
        # study UID instead would put two studies with one text in two shards,
        # and each shard would pay for the same call.
        rows = _corpus(200) + [_row("dup_a", "Shared."), _row("dup_b", "Shared.")]
        units = plan_units(rows)
        holders = [i for i in range(8)
                   if any(u.group_size == 2 for u in select_shard(units, i, 8))]
        assert len(holders) == 1
        shard = select_shard(units, holders[0], 8)
        assert [u.study_uids for u in shard if u.group_size == 2] == [("dup_a", "dup_b")]

    def test_membership_is_unchanged_by_a_resume_narrowing_the_units(self):
        units = plan_units(_corpus(200))
        before = {u.content_hash for u in select_shard(units, 3, 8)}
        narrowed = pending(units, {u.study_uids[0] for u in units[:100]})
        after = {u.content_hash for u in select_shard(narrowed, 3, 8)}
        assert after <= before
        assert after == before & {u.content_hash for u in narrowed}

    def test_every_study_is_covered_once_across_all_shards(self):
        rows = _corpus(200) + [_row("dup_a", "Shared."), _row("dup_b", "Shared.")]
        units = plan_units(rows)
        covered: list[str] = []
        for index in range(8):
            for unit in select_shard(units, index, 8):
                covered.extend(unit.study_uids)
        assert sorted(covered) == sorted(r["study_uid"] for r in rows)

    def test_an_out_of_range_index_is_refused(self):
        with pytest.raises(ValueError):
            select_shard([], 8, 8)
        with pytest.raises(ValueError):
            select_shard([], -1, 8)


class TestParseShard:
    def test_parses_the_documented_form(self):
        assert parse_shard("3/8") == (3, 8)
        assert parse_shard(" 0 / 8 ") == (0, 8)

    def test_the_off_by_one_of_an_array_job_is_refused_loudly(self):
        # `--array=0-7` with `--shard 8/8` is the natural mistake, and silently
        # selecting an empty shard would print "nothing to do" and exit 0.
        with pytest.raises(ValueError, match="out of range"):
            parse_shard("8/8")

    def test_garbage_is_refused(self):
        for spec in ("", "3", "3/", "a/b", "3//8", "-1/8"):
            with pytest.raises(ValueError):
                parse_shard(spec)


class TestShardPath:
    def test_no_shard_means_the_base_path_itself(self, tmp_path):
        assert shard_path(tmp_path / "records.jsonl") == tmp_path / "records.jsonl"

    def test_shard_files_sort_in_shard_order_under_a_glob(self):
        # Unpadded, `shard10of12` sorts before `shard2of12`, and a merge that
        # trusts glob order then silently reorders the corpus.
        paths = shard_paths("records.jsonl", 12)
        assert [p.name for p in paths] == sorted(p.name for p in paths)
        assert paths[0].name == "records.shard00of12.jsonl"

    def test_the_suffix_is_preserved(self):
        assert shard_path("a/records.jsonl", 1, 8).name == "records.shard01of08.jsonl"


class TestReadDone:
    def test_a_missing_file_is_zero_done_not_an_error(self, tmp_path):
        state = read_done(tmp_path / "nope.jsonl")
        assert state.study_uids == frozenset()
        assert state.n_lines == 0

    def test_a_truncated_final_line_is_discarded_and_not_counted_done(self, tmp_path):
        path = tmp_path / "records.jsonl"
        good = json.dumps({STUDY_KEY: "s1", "findings": []})
        path.write_text(good + "\n" + '{"_study_uid": "s2", "find', encoding="utf-8")
        state = read_done(path)
        assert state.study_uids == {"s1"}
        assert "s2" not in state.study_uids
        assert state.truncated_tail is True
        assert any("truncated" in note for note in state.warnings)

    def test_a_final_line_cut_mid_codepoint_does_not_lose_the_good_records(self, tmp_path):
        # The corpus is multilingual, so a kill can land inside a multi-byte
        # UTF-8 sequence. A strict decode would raise over the whole file and
        # throw away every good record before it.
        path = tmp_path / "records.jsonl"
        head = (json.dumps({STUDY_KEY: "s1"}) + "\n").encode("utf-8")
        cut = json.dumps({STUDY_KEY: "s2", "span": "μηνίσκων"}).encode("utf-8")[:-4]
        path.write_bytes(head + cut)
        state = read_done(path)
        assert state.study_uids == {"s1"}
        assert state.truncated_tail is True

    def test_an_unreadable_line_in_the_middle_is_reported_separately(self, tmp_path):
        # A truncated tail is the expected signature of a walltime kill. A bad
        # line in the middle of the file is not something a clean kill produces.
        path = tmp_path / "records.jsonl"
        path.write_text('{"_study_uid": "s1"}\n{oops\n{"_study_uid": "s2"}\n',
                        encoding="utf-8")
        state = read_done(path)
        assert state.study_uids == {"s1", "s2"}
        assert state.truncated_tail is False
        assert any("away from the end" in note for note in state.warnings)

    def test_a_record_with_no_study_key_is_not_counted_done(self, tmp_path):
        path = tmp_path / "records.jsonl"
        path.write_text('{"findings": []}\n{"_study_uid": ""}\n', encoding="utf-8")
        state = read_done(path)
        assert state.study_uids == frozenset()
        assert state.n_missing_key == 2
        assert any(STUDY_KEY in note for note in state.warnings)

    def test_a_clean_file_produces_no_warnings(self, tmp_path):
        path = tmp_path / "records.jsonl"
        path.write_text('{"_study_uid": "s1"}\n{"_study_uid": "s2"}\n', encoding="utf-8")
        assert read_done(path).warnings == ()

    def test_blank_lines_are_neither_done_nor_damage(self, tmp_path):
        path = tmp_path / "records.jsonl"
        path.write_text('\n{"_study_uid": "s1"}\n\n', encoding="utf-8")
        state = read_done(path)
        assert state.study_uids == {"s1"}
        assert state.n_unreadable == 0


class TestPending:
    def test_pending_covers_exactly_the_missing_studies(self):
        units = plan_units(_corpus(10))
        done = {unit.study_uids[0] for unit in units[:4]}
        todo = pending(units, done)
        assert len(todo) == 6
        assert {u.study_uids[0] for u in todo} == {
            u.study_uids[0] for u in units} - done

    def test_nothing_is_pending_once_every_study_is_written(self):
        units = plan_units(_corpus(10))
        assert pending(units, [uid for u in units for uid in u.study_uids]) == []

    def test_a_half_written_duplicate_group_is_narrowed_not_dropped(self):
        # A kill between the first and last record of a 3-study group must not
        # leave the other two permanently unwritten. One redundant call is the
        # price, and it is a price this pays on purpose.
        units = plan_units([_row("a", "Same."), _row("b", "Same."), _row("c", "Same.")])
        todo = pending(units, {"a"})
        assert len(todo) == 1
        assert todo[0].study_uids == ("b", "c")
        assert todo[0].group_size == 3        # the plan-time size is preserved
        assert todo[0].primary_uid == "a"

    def test_the_original_units_are_not_mutated(self):
        units = plan_units([_row("a", "Same."), _row("b", "Same.")])
        pending(units, {"a"})
        assert units[0].study_uids == ("a", "b")


class TestRecordsForUnit:
    def test_one_extraction_becomes_one_record_per_study(self):
        unit = _unit(study_uids=("a", "b", "c"), primary_uid="a", group_size=3)
        records = records_for_unit(unit, {"findings": [{"span": "tear"}]})
        assert [r[STUDY_KEY] for r in records] == ["a", "b", "c"]
        assert all(r["findings"] == [{"span": "tear"}] for r in records)

    def test_a_propagated_record_says_so(self):
        unit = _unit(study_uids=("a", "b"), primary_uid="a", group_size=2)
        records = records_for_unit(unit, {})
        assert [r["_propagated"] for r in records] == [False, True]
        assert all(r["_primary_uid"] == "a" for r in records)
        assert all(r["_group_size"] == 2 for r in records)

    def test_records_written_by_a_resume_are_still_marked_propagated(self):
        # The primary was written before the kill, so this call emits only the
        # remainder -- all of which came from one extraction of one text.
        unit = _unit(study_uids=("b", "c"), primary_uid="a", group_size=3)
        assert all(r["_propagated"] for r in records_for_unit(unit, {}))

    def test_the_payload_is_copied_not_shared(self):
        payload = {"findings": []}
        records = records_for_unit(_unit(study_uids=("a", "b")), payload)
        records[0]["findings"] = ["mutated"]
        assert payload["findings"] == []
        assert records[1]["findings"] == []

    def test_the_shard_label_is_recorded_only_when_given(self):
        assert "_shard" not in records_for_unit(_unit(), {})[0]
        assert records_for_unit(_unit(), {}, shard="3/8")[0]["_shard"] == "3/8"


class TestRecordWriter:
    def test_a_record_is_on_disk_before_the_context_manager_exits(self, tmp_path):
        # This is the whole point of the class. A buffered write that is never
        # flushed loses everything to SIGKILL, which is what a walltime kill
        # sends -- __exit__ does not run.
        path = tmp_path / "records.jsonl"
        with RecordWriter(path) as writer:
            writer.write({STUDY_KEY: "s1"})
            assert read_done(path).study_uids == {"s1"}
            writer.write({STUDY_KEY: "s2"})
            assert read_done(path).study_uids == {"s1", "s2"}
        assert read_done(path).study_uids == {"s1", "s2"}

    def test_appending_after_a_kill_does_not_glue_onto_the_broken_line(self, tmp_path):
        # Append mode writes at the end of the file, not the end of the last
        # line. Without the tail repair the resumed record lands on the same
        # line as the fragment, neither parses, and the resume silently loses
        # the one study it restarted to write.
        path = tmp_path / "records.jsonl"
        path.write_text('{"_study_uid": "s1"}\n{"_study_uid": "s2", "find',
                        encoding="utf-8")
        with RecordWriter(path) as writer:
            writer.write({STUDY_KEY: "s2", "findings": []})
            assert writer.repaired_tail is not None
        state = read_done(path)
        assert state.study_uids == {"s1", "s2"}
        assert state.n_unreadable == 0        # the fragment is gone, not orphaned
        records, warnings = load_records([path])
        assert [r[STUDY_KEY] for r in records] == ["s1", "s2"]
        assert warnings == []

    def test_a_complete_final_record_missing_its_newline_is_kept(self, tmp_path):
        # A kernel write can stop exactly at the end of a record. That record is
        # good and was already counted as done, so deleting it here would lose
        # it permanently -- nothing downstream would ever re-extract it.
        path = tmp_path / "records.jsonl"
        path.write_text('{"_study_uid": "s1"}', encoding="utf-8")
        with RecordWriter(path) as writer:
            writer.write({STUDY_KEY: "s2"})
        assert read_done(path).study_uids == {"s1", "s2"}

    def test_a_clean_file_needs_no_repair(self, tmp_path):
        path = tmp_path / "records.jsonl"
        path.write_text('{"_study_uid": "s1"}\n', encoding="utf-8")
        with RecordWriter(path) as writer:
            assert writer.repaired_tail is None

    def test_a_fresh_file_needs_no_repair(self, tmp_path):
        with RecordWriter(tmp_path / "records.jsonl") as writer:
            assert writer.repaired_tail is None

    def test_reopening_appends_rather_than_truncating(self, tmp_path):
        path = tmp_path / "records.jsonl"
        with RecordWriter(path) as writer:
            writer.write({STUDY_KEY: "s1"})
        with RecordWriter(path) as writer:
            writer.write({STUDY_KEY: "s2"})
        assert read_done(path).study_uids == {"s1", "s2"}

    def test_non_ascii_survives_the_round_trip(self, tmp_path):
        path = tmp_path / "records.jsonl"
        with RecordWriter(path) as writer:
            writer.write({STUDY_KEY: "s1", "span": "μηνίσκων ∆ΕΞΙΟΥ"})
        records, warnings = load_records([path])
        assert records[0]["span"] == "μηνίσκων ∆ΕΞΙΟΥ"
        assert warnings == []

    def test_the_parent_directory_is_created(self, tmp_path):
        path = tmp_path / "deep" / "nested" / "records.jsonl"
        with RecordWriter(path) as writer:
            writer.write({STUDY_KEY: "s1"})
        assert path.is_file()

    def test_writing_outside_the_context_manager_is_an_error(self, tmp_path):
        writer = RecordWriter(tmp_path / "records.jsonl")
        with pytest.raises(RuntimeError):
            writer.write({STUDY_KEY: "s1"})

    def test_n_written_counts_records_not_calls(self, tmp_path):
        unit = _unit(study_uids=("a", "b", "c"), group_size=3)
        with RecordWriter(tmp_path / "r.jsonl") as writer:
            for record in records_for_unit(unit, {}):
                writer.write(record)
            assert writer.n_written == 3


class TestLoadRecords:
    def test_shards_merge_into_one_corpus(self, tmp_path):
        for index in range(2):
            path = shard_path(tmp_path / "r.jsonl", index, 2)
            path.write_text(json.dumps({STUDY_KEY: f"s{index}"}) + "\n", encoding="utf-8")
        records, warnings = load_records(shard_paths(tmp_path / "r.jsonl", 2))
        assert [r[STUDY_KEY] for r in records] == ["s0", "s1"]
        assert warnings == []

    def test_a_missing_shard_is_named_never_silently_skipped(self, tmp_path):
        path = shard_path(tmp_path / "r.jsonl", 0, 2)
        path.write_text(json.dumps({STUDY_KEY: "s0"}) + "\n", encoding="utf-8")
        records, warnings = load_records(shard_paths(tmp_path / "r.jsonl", 2))
        assert len(records) == 1
        assert any("missing" in note for note in warnings)

    def test_a_study_written_twice_is_kept_once_with_the_later_record(self, tmp_path):
        # A resume legitimately appends a second record for a study whose first
        # line was unreadable. Keeping both would double-count it in every
        # aggregate -- invisible in a rate, wrong in a count.
        path = tmp_path / "r.jsonl"
        path.write_text(
            json.dumps({STUDY_KEY: "s1", "_valid": False}) + "\n"
            + json.dumps({STUDY_KEY: "s1", "_valid": True}) + "\n",
            encoding="utf-8",
        )
        records, _ = load_records([path])
        assert len(records) == 1
        assert records[0]["_valid"] is True

    def test_records_come_back_sorted_by_study(self, tmp_path):
        path = tmp_path / "r.jsonl"
        path.write_text("".join(json.dumps({STUDY_KEY: uid}) + "\n"
                                for uid in ("s3", "s1", "s2")), encoding="utf-8")
        records, _ = load_records([path])
        assert [r[STUDY_KEY] for r in records] == ["s1", "s2", "s3"]

    def test_a_truncated_tail_is_dropped_and_reported(self, tmp_path):
        path = tmp_path / "r.jsonl"
        path.write_text(json.dumps({STUDY_KEY: "s1"}) + "\n" + '{"_study_uid": "s2',
                        encoding="utf-8")
        records, warnings = load_records([path])
        assert [r[STUDY_KEY] for r in records] == ["s1"]
        assert any("truncated" in note for note in warnings)


class TestMissingStudies:
    def test_the_studies_no_record_covers_are_named(self):
        records = [{STUDY_KEY: "s1"}, {STUDY_KEY: "s2"}]
        assert missing_studies(records, ["s1", "s2", "s3"]) == ["s3"]

    def test_a_complete_merge_reports_nothing_missing(self):
        assert missing_studies([{STUDY_KEY: "s1"}], ["s1"]) == []

    def test_a_record_with_a_non_string_key_does_not_cover_anything(self):
        assert missing_studies([{STUDY_KEY: None}], ["s1"]) == ["s1"]
