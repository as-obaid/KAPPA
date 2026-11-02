"""The generation loop must survive being killed at minute 59 of a 59-minute run.

Two halves, both about the same incident. The reconnaissance script called
`generate()` once over the whole sample and wrote its records after it returned,
so a Slurm walltime SIGKILL left an empty output directory and a full GPU-hour
bought nothing. `generate_incremental` is the fix, and the first half of this
file pins the contract it has to keep: results come out per batch, in the order
the caller asked for them, and whatever was already yielded stays yielded when a
later batch fails.

The second half is the resume cycle end to end -- plan, extract, get killed
mid-batch with a half-written line on disk, resume, and finish -- because the
individual pieces can each be right while the composition still loses or
duplicates studies. It runs on a laptop: the backend is a fake, and nothing here
imports vLLM.

The logprob tests are here rather than with the schema because the number they
protect is a *calibration* claim. Verbalized confidence is measured to be poorly
calibrated, and the reason to persist logprobs instead is that they are not --
which stops being true if a distribution that lists one literal is allowed to
report certainty.
"""

from __future__ import annotations

import json

import pytest

from kappa.labels.engine import (
    ASSERTION_LITERALS, DEFAULT_BATCH_SIZE, MAX_OUTPUT_TOKENS, Generation,
    align_assertion_confidence, assertion_confidence, assertion_positions,
    generate_incremental,
)
from kappa.labels.normalise import content_hash, normalise_report
from kappa.labels.shard import (
    STUDY_KEY, RecordWriter, load_records, missing_studies, pending, plan_units,
    read_done, records_for_unit, select_shard,
)


def _row(uid, report="Normal knee.", script="latin", is_anchor=False):
    return {
        "study_uid": uid,
        "report": report,
        "script": script,
        "hash": content_hash(normalise_report(report)),
        "is_anchor": is_anchor,
    }


def _extraction(**over):
    payload = {"language": "English", "knee_side": "left",
               "findings": [{"span": "ACL tear", "assertion": "affirmed"}]}
    payload.update(over)
    return payload


class _Killed(Exception):
    """Stands in for the walltime SIGKILL, which cannot be caught for real."""


class _FakeBackend:
    """Answers with the report echoed back, and can die on a chosen batch.

    ``batches`` records the size of every call, which is what proves the results
    were streamed rather than accumulated, and ``prompts`` records what was
    asked for, which is what proves a resume did not re-extract finished work.
    """

    def __init__(self, die_on_batch: int | None = None) -> None:
        self.batches: list[int] = []
        self.prompts: list[str] = []
        self._die_on_batch = die_on_batch

    def generate(self, prompts: list[str], max_tokens: int = 2048) -> list[str]:
        return [g.text for g in self.generate_verbose(prompts, max_tokens)]

    def generate_verbose(self, prompts: list[str], max_tokens: int = 2048):
        if self._die_on_batch is not None and len(self.batches) == self._die_on_batch:
            raise _Killed(f"walltime during batch {self._die_on_batch}")
        self.batches.append(len(prompts))
        self.prompts.extend(prompts)
        return [
            Generation(text=json.dumps(_extraction(notes=prompt)),
                       finish_reason="stop", n_output_tokens=42)
            for prompt in prompts
        ]


def _extract(units, path, backend, batch_size=2, shard="0/1"):
    """The script's loop, reduced to the three pieces this file is about.

    Deliberately a copy of the shape in ``scripts/extract_reports.py`` rather
    than an import of it: what is under test is that streaming, propagation and
    per-record flushing compose correctly, and the script adds only argparse to
    that.
    """
    prompts = [unit.report for unit in units]
    with RecordWriter(path) as writer:
        for position, generation in generate_incremental(
            backend, prompts, batch_size=batch_size
        ):
            payload = {"_parsed": True, "_valid": True,
                       "_finish_reason": generation.finish_reason}
            for record in records_for_unit(units[position], payload, shard=shard):
                writer.write(record)
        return writer.n_written


class TestGenerateIncremental:
    def test_results_arrive_per_batch_not_all_at_the_end(self):
        backend = _FakeBackend()
        seen = []
        for position, _ in generate_incremental(backend, ["a"] * 5, batch_size=2):
            # The backend has been called for at most the batch containing this
            # position. If the whole list had been generated up front, the batch
            # count would already be 3 on the first iteration.
            assert len(backend.batches) == position // 2 + 1
            seen.append(position)
        assert seen == [0, 1, 2, 3, 4]
        assert backend.batches == [2, 2, 1]

    def test_the_index_is_the_position_in_the_prompt_list(self):
        prompts = [f"p{i}" for i in range(7)]
        pairs = list(generate_incremental(_FakeBackend(), prompts, batch_size=3))
        assert [i for i, _ in pairs] == list(range(7))
        for index, generation in pairs:
            assert prompts[index] in generation.text

    def test_a_kill_mid_run_keeps_everything_already_yielded(self):
        backend = _FakeBackend(die_on_batch=2)
        yielded = []
        with pytest.raises(_Killed):
            for position, _ in generate_incremental(backend, ["a"] * 6, batch_size=2):
                yielded.append(position)
        assert yielded == [0, 1, 2, 3]      # the two completed batches, no more

    def test_one_batch_is_the_whole_list(self):
        backend = _FakeBackend()
        assert len(list(generate_incremental(backend, ["a"] * 5, batch_size=100))) == 5
        assert backend.batches == [5]

    def test_an_empty_prompt_list_calls_the_backend_not_at_all(self):
        backend = _FakeBackend()
        assert list(generate_incremental(backend, [])) == []
        assert backend.batches == []

    def test_a_zero_batch_size_is_refused_rather_than_looping_forever(self):
        with pytest.raises(ValueError):
            list(generate_incremental(_FakeBackend(), ["a"], batch_size=0))

    def test_the_default_batch_is_smaller_than_a_shard(self):
        # 8 shards of the measured 4,262-unit corpus are 483 to 576 calls each,
        # so the default has to divide a shard for the loss window to mean
        # anything.
        assert DEFAULT_BATCH_SIZE < 483

    def test_the_token_cap_is_the_default_ceiling(self):
        assert MAX_OUTPUT_TOKENS == 4096


class TestAssertionPositions:
    def test_the_distribution_after_the_assertion_key_is_captured(self):
        tokens = [
            ('{"span":"ACL tear","assertion"', {}),
            (': "', {}),
            ('affirmed', {'aff': -0.02, 'neg': -4.0, 'unc': -6.0}),
            ('"}', {}),
        ]
        positions = assertion_positions(tokens)
        assert len(positions) == 1
        assert set(positions[0]) == set(ASSERTION_LITERALS)

    def test_a_description_containing_unremarkable_is_not_an_assertion(self):
        # `un` prefixes `uncertain` and half the English language. Matching
        # tokens that look like a literal would put a confidence on every
        # `unremarkable` in a description and shift every finding's number onto
        # the wrong finding.
        tokens = [('"description"', {}), (': "', {}),
                  ('unremarkable', {'unremarkable': -0.1}), ('"', {})]
        assert assertion_positions(tokens) == []

    def test_one_position_per_finding_in_generation_order(self):
        stream = []
        for literal in ("affirmed", "negated", "uncertain"):
            stream += [('"assertion"', {}), (': "', {}),
                       (literal, {literal[:3]: -0.01, 'zzz': -9.0})]
        positions = assertion_positions(stream)
        assert len(positions) == 3
        assert [assertion_confidence(p)[0] for p in positions] == [
            "affirmed", "negated", "uncertain"]

    def test_compact_json_without_a_space_is_still_matched(self):
        # disable_any_whitespace stops the model spending tokens on indentation,
        # so the real stream has no space after the colon.
        tokens = [('"assertion":"', {}), ('negated', {'neg': -0.1, 'aff': -3.0})]
        assert len(assertion_positions(tokens)) == 1

    def test_a_position_with_no_assertion_literal_at_all_is_empty(self):
        tokens = [('"assertion": "', {}), ('???', {'???': -0.1, '!!!': -2.0})]
        assert assertion_positions(tokens) == [{}]


class TestAssertionConfidence:
    def test_a_literal_missing_from_the_top_k_does_not_yield_certainty(self):
        # The cheap wrong fix is to renormalise over whatever literals survived
        # the fold, which reports exactly 1.0 whenever the top-k happens to hold
        # one of them -- a certainty that comes from the tokeniser, not the
        # model. Every such finding would look maximally confident downstream.
        only_one = assertion_positions(
            [('"assertion": "', {}), ('affirmed', {'aff': -0.7, 'x': -1.0, 'y': -1.2})]
        )[0]
        literal, probability = assertion_confidence(only_one)
        assert literal == "affirmed"
        assert probability < 1.0

    def test_the_substituted_bound_can_only_understate_confidence(self):
        # The bound for an unseen literal is the smallest logprob that WAS seen,
        # which is an upper bound on the truth, so the reported probability is a
        # lower bound. Erring downward is the safe direction for a soft target.
        seen = {'aff': -0.7, 'x': -1.0}
        bounded = assertion_positions([('"assertion": "', {}), ('affirmed', seen)])[0]
        assert bounded["negated"] == -1.0
        assert bounded["uncertain"] == -1.0

    def test_a_torn_model_reports_a_third_not_a_half(self):
        dist = dict.fromkeys(ASSERTION_LITERALS, -1.0986)
        literal, probability = assertion_confidence(dist)
        assert literal in ASSERTION_LITERALS
        assert probability == pytest.approx(1 / 3, abs=1e-3)

    def test_a_confident_model_reports_close_to_one(self):
        dist = {"affirmed": -0.001, "negated": -12.0, "uncertain": -13.0}
        assert assertion_confidence(dist)[1] == pytest.approx(1.0, abs=1e-4)

    def test_no_distribution_is_no_number_rather_than_a_default(self):
        assert assertion_confidence({}) == (None, None)

    def test_masked_tokens_do_not_put_nan_in_the_softmax(self):
        # Under structured output vLLM masks unreachable tokens to -inf. Using
        # -inf as the bound for a missing literal would make the softmax nan,
        # and nan compares false against every threshold without ever raising.
        masked = assertion_positions(
            [('"assertion": "', {}),
             ('affirmed', {'aff': -0.05, 'neg': float("-inf"), 'unc': float("-inf")})]
        )[0]
        literal, probability = assertion_confidence(masked)
        assert literal == "affirmed"
        assert probability == probability     # not nan
        assert 0.0 < probability <= 1.0


class TestAlignAssertionConfidence:
    def test_one_confidence_per_finding_when_the_counts_agree(self):
        dists = [{"affirmed": -0.1, "negated": -3.0, "uncertain": -4.0}] * 3
        values, warnings = align_assertion_confidence(3, dists)
        assert len(values) == 3
        assert warnings == []
        assert all(0.0 < v <= 1.0 for v in values)

    def test_a_length_mismatch_drops_every_confidence_and_says_both_counts(self):
        # Zipping 12 findings against 11 distributions produces 11 plausible
        # numbers attached to the wrong findings, and nothing downstream can
        # notice: a misplaced confidence does not look wrong the way a missing
        # one does.
        values, warnings = align_assertion_confidence(12, [{"affirmed": -0.1}] * 11)
        assert values == [None] * 12
        assert len(warnings) == 1
        assert "12 findings" in warnings[0] and "11 assertion positions" in warnings[0]

    def test_a_report_with_no_findings_is_not_a_mismatch(self):
        assert align_assertion_confidence(0, []) == ([], [])


class TestResumeCycle:
    """Plan, get killed mid-batch, resume, and land on exactly one record each."""

    def _units(self, n=6):
        rows = [_row(f"s{i}", f"Report {i}.") for i in range(n)]
        return plan_units(rows), rows

    def test_a_kill_leaves_the_completed_batches_on_disk(self, tmp_path):
        units, _ = self._units()
        path = tmp_path / "records.jsonl"
        with pytest.raises(_Killed):
            _extract(units, path, _FakeBackend(die_on_batch=2), batch_size=2)
        # Two batches of two finished before the kill, so four studies are safe.
        assert len(read_done(path).study_uids) == 4

    def test_a_resume_covers_exactly_the_missing_studies_and_no_others(self, tmp_path):
        units, rows = self._units()
        path = tmp_path / "records.jsonl"
        with pytest.raises(_Killed):
            _extract(units, path, _FakeBackend(die_on_batch=2), batch_size=2)
        done = read_done(path).study_uids

        second = _FakeBackend()
        todo = pending(units, done)
        _extract(todo, path, second, batch_size=2)

        # The second run asked the model for the two reports that were missing,
        # and for nothing that was already written.
        assert sorted(second.prompts) == sorted(
            unit.report for unit in units if unit.study_uids[0] not in done)
        records, warnings = load_records([path])
        assert warnings == []
        assert missing_studies(records, [r["study_uid"] for r in rows]) == []
        assert len(records) == len(rows)

    def test_a_truncated_final_line_is_re_extracted_not_skipped(self, tmp_path):
        units, rows = self._units()
        path = tmp_path / "records.jsonl"
        with pytest.raises(_Killed):
            _extract(units, path, _FakeBackend(die_on_batch=2), batch_size=2)

        # A SIGKILL during a flush leaves a half-written object on the last
        # line. Simulate exactly that for the next study in the plan.
        victim = pending(units, read_done(path).study_uids)[0]
        whole = json.dumps({STUDY_KEY: victim.study_uids[0],
                            "findings": [{"span": "ACL tear"}]})
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(whole[: len(whole) // 2])

        state = read_done(path)
        assert state.truncated_tail is True
        assert victim.study_uids[0] not in state.study_uids

        second = _FakeBackend()
        _extract(pending(units, state.study_uids), path, second, batch_size=2)
        assert victim.report in second.prompts

        records, _ = load_records([path])
        assert missing_studies(records, [r["study_uid"] for r in rows]) == []
        assert len(records) == len(rows)          # the fragment did not become a row

    def test_resuming_a_finished_shard_calls_the_model_zero_times(self, tmp_path):
        units, _ = self._units()
        path = tmp_path / "records.jsonl"
        _extract(units, path, _FakeBackend(), batch_size=2)
        second = _FakeBackend()
        _extract(pending(units, read_done(path).study_uids), path, second, batch_size=2)
        assert second.prompts == []

    def test_a_duplicate_group_is_extracted_once_and_written_to_every_study(self, tmp_path):
        rows = [_row("a", "Shared."), _row("b", "Shared."), _row("c", "Other.")]
        units = plan_units(rows)
        path = tmp_path / "records.jsonl"
        backend = _FakeBackend()
        n_written = _extract(units, path, backend, batch_size=8)

        assert len(backend.prompts) == 2      # two calls
        assert n_written == 3                 # three records
        records, _ = load_records([path])
        assert missing_studies(records, ["a", "b", "c"]) == []
        propagated = [r for r in records if r["_propagated"]]
        assert [r[STUDY_KEY] for r in propagated] == ["b"]
        assert propagated[0]["_primary_uid"] == "a"

    def test_a_kill_inside_a_duplicate_group_still_finishes_it(self, tmp_path):
        # The wrong fix is to treat a group as done once its first record
        # exists, which leaves the rest permanently unwritten. One redundant
        # call is the price of correctness here.
        rows = [_row(uid, "Shared.") for uid in ("a", "b", "c")]
        units = plan_units(rows)
        path = tmp_path / "records.jsonl"
        with RecordWriter(path) as writer:
            writer.write(records_for_unit(units[0], {"_valid": True})[0])

        second = _FakeBackend()
        todo = pending(units, read_done(path).study_uids)
        assert todo[0].study_uids == ("b", "c")
        _extract(todo, path, second, batch_size=2)

        records, _ = load_records([path])
        assert missing_studies(records, ["a", "b", "c"]) == []
        assert len(records) == 3

    def test_every_shard_resumes_independently_into_its_own_file(self, tmp_path):
        rows = [_row(f"s{i}", f"Report {i}.") for i in range(40)]
        units = plan_units(rows)
        paths = []
        for index in range(4):
            path = tmp_path / f"records.shard{index}of4.jsonl"
            paths.append(path)
            mine = select_shard(units, index, 4)
            # Kill the first shard part-way and resume it; leave the rest alone.
            if index == 0:
                with pytest.raises(_Killed):
                    _extract(mine, path, _FakeBackend(die_on_batch=1), batch_size=2)
                mine = pending(mine, read_done(path).study_uids)
            _extract(mine, path, _FakeBackend(), batch_size=8)

        records, warnings = load_records(paths)
        assert warnings == []
        assert missing_studies(records, [r["study_uid"] for r in rows]) == []
        assert len(records) == 40
