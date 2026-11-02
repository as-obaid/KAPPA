"""A blend is only as good as the row it thinks it is blending.

Two failure modes are worth a test file of this size, and neither one raises on its
own.

**Blending by position.** Folds run as separate SLURM jobs, so two members' rows are
in whatever order each job produced. A positional combination of two members whose
study order differs returns a full ``(n, 12)`` matrix of finite numbers in which one
study's rank has been averaged with another's. ``TestAlignmentIsByIdentifier``
therefore blends a member against a *shuffled copy of itself* and demands the same
answer -- and, so that the demand is not vacuous, asserts in the same test that the
positional blend of those two arrays is a different matrix.

**Blending probabilities.** Macro ROC-AUC reads only the order of the scores inside
a column, so a member with a wider spread wins a probability average without being
any better. ``TestRankSpaceOnly`` multiplies one member by 1000 and asserts the
blend does not move, next to the assertion that the probability average of the same
two inputs does.

A third family is a composition check rather than a behaviour check.
``kappa.evaluation.metrics.rank_average`` is the repository's one rank transform, and
:func:`kappa.ensemble.blend` calls it a column at a time on ``(n, 1)`` slices so that
per-column weights mean something. That is only legitimate if the uniform case comes
out bit-identical to a single whole-matrix ``rank_average`` call, which is asserted
with ``array_equal`` and not with a tolerance.

The last family is the competition-legality boundary. Submissions are image-only
(constraint A1), so :mod:`kappa.ensemble` must not reach ``kappa.labels`` at import
time or at call time; the check is the same clean-interpreter one
``tests/test_loop.py`` runs against the inference path. Alongside it sits the
``torch`` check: ``pyproject.toml`` confines module-scope ``torch`` imports to
``kappa.models`` and ``kappa.training``, and blending saved artifacts is exactly the
step that has to run where the ``train`` extra was never installed.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kappa.constants import ID_COLUMN, LABELS, N_LABELS, SUBMISSION_COLUMNS
from kappa.data.paths import repo_root
from kappa.ensemble import (
    MASK_SUFFIX,
    OOF_COLUMNS,
    TRUE_SUFFIX,
    Member,
    align_members,
    blend,
    column_weights,
    read_oof,
    score_blend,
)
from kappa.evaluation.metrics import average_ranks, rank_average, rank_normalise
from kappa.inference.submission import expected_header_bytes, validate_submission

#: The prefix every report-reading module shares. Held here as a literal so the two
#: A1 tests below run on a machine with no ``torch`` -- reading it from
#: ``kappa.inference.predict`` would drag the whole modelling stack into a check
#: whose entire point is that it must never be skipped. The last test in this file
#: is what stops the literal drifting from the project's own definition.
FORBIDDEN_PREFIX = "kappa.labels"


def _uids(n: int) -> list[str]:
    """``n`` distinct 64-character study identifiers, the length the real ones are."""
    return [f"{index:064d}" for index in range(n)]


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _member(name: str, uids, rng: np.random.Generator) -> Member:
    return Member(name=name, study_uids=list(uids), scores=rng.random((len(uids), N_LABELS)))


def _binary_truth(rng: np.random.Generator, n_rows: int) -> np.ndarray:
    """Binary truth with both classes forced into every column.

    Forced deliberately: a single-class column is a *skip* in
    :func:`kappa.evaluation.metrics.macro_roc_auc`, and a test that did not ask for a
    skip should not quietly be testing one.
    """
    truth = (rng.random((n_rows, N_LABELS)) < 0.4).astype(np.float64)
    truth[0, :] = 1.0
    truth[1, :] = 0.0
    return truth


def _oof_frame(
    uids,
    scores: np.ndarray,
    truth: np.ndarray,
    mask: np.ndarray,
    *,
    fold: int = 0,
    seed: int = 7,
) -> pd.DataFrame:
    """An OOF table in the training loop's own schema, built column by name."""
    data: dict[str, object] = {
        ID_COLUMN: list(uids),
        "fold": np.full(len(uids), fold, dtype=np.int64),
        "seed": np.full(len(uids), seed, dtype=np.int64),
    }
    for position, label in enumerate(LABELS):
        data[label] = scores[:, position]
        data[f"{label}{TRUE_SUFFIX}"] = truth[:, position]
        data[f"{label}{MASK_SUFFIX}"] = mask[:, position]
    return pd.DataFrame(data, columns=list(OOF_COLUMNS))


def _write_oof(path: Path, frame: pd.DataFrame) -> Path:
    frame.to_parquet(path, index=False)
    return path


class TestMemberContract:
    def test_a_member_carrying_eleven_columns_raises(self):
        with pytest.raises(ValueError, match="12 labels"):
            Member("a", _uids(4), np.zeros((4, N_LABELS - 1)))

    def test_uid_count_and_row_count_must_agree(self):
        with pytest.raises(ValueError, match="3 study UIDs but 4 score rows"):
            Member("a", _uids(3), np.zeros((4, N_LABELS)))

    def test_a_repeated_study_uid_raises_because_the_lookup_becomes_ambiguous(self):
        uids = _uids(3)
        with pytest.raises(ValueError, match="more than once"):
            Member("a", [*uids, uids[0]], np.zeros((4, N_LABELS)))

    def test_a_nan_score_raises_and_names_the_study_and_the_label(self):
        scores = np.zeros((3, N_LABELS))
        scores[2, 5] = np.nan
        with pytest.raises(ValueError, match=f"{LABELS[5]!r}"):
            Member("a", _uids(3), scores)

    def test_an_infinity_raises_because_it_takes_the_top_rank_for_free(self):
        scores = np.zeros((3, N_LABELS))
        scores[1, 0] = np.inf
        with pytest.raises(ValueError, match="non-finite"):
            Member("a", _uids(3), scores)

    def test_from_frame_selects_labels_by_name_not_by_column_position(self):
        rng = _rng(1)
        uids = _uids(5)
        scores = rng.random((5, N_LABELS))
        frame = pd.DataFrame({ID_COLUMN: uids})
        # Reversed on purpose: a positional read would transpose ACL with Fracture
        # and still hand back twelve plausible columns.
        for position, label in reversed(list(enumerate(LABELS))):
            frame[label] = scores[:, position]
        member = Member.from_frame(frame, name="a")
        assert np.array_equal(member.scores, scores)

    def test_from_frame_missing_a_label_raises_and_names_it(self):
        uids = _uids(3)
        frame = pd.DataFrame({ID_COLUMN: uids, **{label: 0.0 for label in LABELS[:-1]}})
        with pytest.raises(ValueError, match=f"{LABELS[-1]!r}"):
            Member.from_frame(frame, name="a")

    def test_reindex_drops_studies_the_caller_did_not_ask_for(self):
        rng = _rng(2)
        uids = _uids(6)
        member = _member("a", uids, rng)
        taken = member.reindex(uids[1:3])
        assert taken.shape == (2, N_LABELS)
        assert np.array_equal(taken, member.scores[1:3])

    def test_reindex_raises_for_a_study_the_member_never_scored(self):
        member = _member("a", _uids(4), _rng(3))
        with pytest.raises(ValueError, match="no score for 1 requested"):
            member.reindex([*_uids(4), "missing"])


class TestAlignmentIsByIdentifier:
    def test_a_shuffled_member_blends_to_the_same_matrix_as_an_ordered_one(self):
        rng = _rng(4)
        uids = _uids(30)
        a = _member("a", uids, rng)
        b = _member("b", uids, rng)

        permutation = rng.permutation(len(uids))
        shuffled = Member(
            name="b",
            study_uids=[uids[index] for index in permutation],
            scores=b.scores[permutation],
        )

        assert np.array_equal(blend([a, b]).scores, blend([a, shuffled]).scores)

        # And the demand is not vacuous: combining the same two arrays by position
        # -- which is what a zip over `.scores` would do -- gives a different matrix.
        positional = rank_average([a.scores, shuffled.scores])
        assert not np.array_equal(positional, blend([a, b]).scores)

    def test_a_member_missing_a_study_raises_rather_than_inner_joining(self):
        rng = _rng(5)
        uids = _uids(10)
        a = _member("a", uids, rng)
        b = _member("b", uids[:-2], rng)
        with pytest.raises(ValueError) as excinfo:
            blend([a, b])
        message = str(excinfo.value)
        assert "2 only in 'a'" in message
        assert "intersection" in message

    def test_a_member_with_a_study_nobody_else_has_raises(self):
        rng = _rng(6)
        uids = _uids(8)
        a = _member("a", uids, rng)
        b = _member("b", [*uids, "extra"], rng)
        with pytest.raises(ValueError, match="1 only in 'b'"):
            blend([a, b])

    def test_duplicate_member_names_raise_because_a_recipe_is_keyed_by_name(self):
        rng = _rng(7)
        uids = _uids(5)
        with pytest.raises(ValueError, match="appear more than once"):
            blend([_member("a", uids, rng), _member("a", uids, rng)])

    def test_order_pins_the_output_row_order(self):
        rng = _rng(8)
        uids = _uids(9)
        a = _member("a", uids, rng)
        b = _member("b", uids, rng)
        wanted = uids[::-1]
        result = blend([a, b], order=wanted)
        assert list(result.study_uids) == wanted
        assert np.array_equal(result.scores, blend([a, b]).scores[::-1])

    def test_order_naming_a_study_nobody_scored_raises(self):
        rng = _rng(9)
        uids = _uids(5)
        a = _member("a", uids, rng)
        with pytest.raises(ValueError, match="1 study/studies no member scored"):
            blend([a], order=[*uids, "ghost"])

    def test_order_that_omits_a_scored_study_raises_rather_than_trimming(self):
        rng = _rng(10)
        uids = _uids(5)
        a = _member("a", uids, rng)
        with pytest.raises(ValueError, match="omits 1"):
            blend([a], order=uids[:-1])

    def test_a_blend_of_no_members_raises(self):
        with pytest.raises(ValueError, match="at least one member"):
            blend([])

    def test_align_members_returns_every_member_on_the_shared_order(self):
        rng = _rng(11)
        uids = _uids(7)
        a = _member("a", uids, rng)
        b = Member("b", uids[::-1], rng.random((7, N_LABELS)))
        order, aligned = align_members([a, b])
        assert order == tuple(uids)
        assert np.array_equal(aligned[0], a.scores)
        assert np.array_equal(aligned[1], b.scores[::-1])


class TestRankSpaceOnly:
    def test_the_uniform_blend_is_bit_identical_to_a_single_rank_average_call(self):
        rng = _rng(12)
        uids = _uids(25)
        members = [_member(name, uids, rng) for name in ("a", "b", "c")]
        _, aligned = align_members(members)
        assert np.array_equal(blend(members).scores, rank_average(aligned))

    def test_a_monotone_rescaling_of_one_member_cannot_move_the_blend(self):
        rng = _rng(13)
        uids = _uids(20)
        a = _member("a", uids, rng)
        b = _member("b", uids, rng)
        # exp is strictly increasing, so it is exactly the class of transform a
        # calibration step applies -- and it must be invisible here.
        rescaled = Member("b", uids, np.exp(b.scores * 3.0) + 11.0)
        assert np.array_equal(blend([a, b]).scores, blend([a, rescaled]).scores)

    def test_the_wider_member_cannot_buy_influence_although_a_probability_average_lets_it(self):
        rng = _rng(14)
        uids = _uids(20)
        a = _member("a", uids, rng)
        b = _member("b", uids, rng)
        loud = Member("b", uids, b.scores * 1000.0)

        assert np.array_equal(blend([a, b]).scores, blend([a, loud]).scores)

        # The contrast that makes the assertion above worth making: the arithmetic
        # mean of the same two pairs orders the studies differently.
        quiet_mean = (a.scores + b.scores) / 2.0
        loud_mean = (a.scores + loud.scores) / 2.0
        assert not np.array_equal(
            average_ranks(quiet_mean[:, 0]), average_ranks(loud_mean[:, 0])
        )

    def test_the_blend_stays_strictly_inside_zero_and_one(self):
        rng = _rng(15)
        uids = _uids(16)
        result = blend([_member("a", uids, rng), _member("b", uids, rng)])
        assert result.scores.min() > 0.0
        assert result.scores.max() < 1.0

    def test_a_single_member_is_exactly_its_own_rank_normalisation(self):
        rng = _rng(16)
        uids = _uids(12)
        a = _member("a", uids, rng)
        assert np.array_equal(blend([a]).scores, rank_normalise(a.scores))

    def test_weights_shift_the_blend_toward_the_heavier_member(self):
        rng = _rng(17)
        uids = _uids(18)
        a = _member("a", uids, rng)
        b = _member("b", uids, rng)
        heavy = blend([a, b], [9.0, 1.0]).scores
        even = blend([a, b]).scores
        alone = rank_normalise(a.scores)
        assert np.abs(heavy - alone).mean() < np.abs(even - alone).mean()

    def test_weights_are_shares_so_scaling_them_all_changes_nothing(self):
        # allclose rather than array_equal: the two are the same quantity computed
        # with different multiplicands and land a few ULP apart. Dropping the
        # normalisation entirely would put them a factor of ten apart, which is
        # what this catches.
        rng = _rng(18)
        uids = _uids(10)
        members = [_member("a", uids, rng), _member("b", uids, rng)]
        small = blend(members, [1.0, 3.0])
        large = blend(members, [10.0, 30.0])
        assert np.allclose(small.scores, large.scores)
        assert np.allclose(small.weights[:, 0], [0.25, 0.75])
        assert np.allclose(large.weights, small.weights)


class TestPerColumnWeights:
    def test_a_zero_weight_drops_a_member_from_that_column_only(self):
        rng = _rng(19)
        uids = _uids(22)
        a = _member("a", uids, rng)
        b = _member("b", uids, rng)
        weights = np.ones((2, N_LABELS))
        weights[1, 0] = 0.0
        result = blend([a, b], weights)
        assert np.array_equal(result.scores[:, 0], rank_normalise(a.scores)[:, 0])
        assert np.array_equal(result.scores[:, 1:], blend([a, b]).scores[:, 1:])

    def test_a_negative_weight_raises_instead_of_reversing_a_member(self):
        rng = _rng(20)
        uids = _uids(6)
        members = [_member("a", uids, rng), _member("b", uids, rng)]
        with pytest.raises(ValueError, match="negative weight reverses"):
            blend(members, [2.0, -1.0])

    def test_a_column_no_member_has_weight_on_raises_and_names_the_label(self):
        rng = _rng(21)
        uids = _uids(6)
        members = [_member("a", uids, rng), _member("b", uids, rng)]
        weights = np.ones((2, N_LABELS))
        weights[:, 3] = 0.0
        with pytest.raises(ValueError, match=f"{LABELS[3]!r}"):
            blend(members, weights)

    def test_the_wrong_number_of_weights_raises(self):
        rng = _rng(22)
        uids = _uids(6)
        members = [_member("a", uids, rng), _member("b", uids, rng)]
        with pytest.raises(ValueError, match="one per member"):
            blend(members, [1.0, 1.0, 1.0])

    def test_column_weights_turns_a_selection_into_the_matrix_blend_takes(self):
        selection = {label: ["a", "b"] for label in LABELS}
        selection[LABELS[0]] = ["b"]
        matrix = column_weights(["a", "b"], selection)
        assert matrix.shape == (2, N_LABELS)
        assert list(matrix[:, 0]) == [0.0, 1.0]
        assert list(matrix[:, 1]) == [1.0, 1.0]

    def test_column_weights_accepts_members_as_well_as_names(self):
        rng = _rng(23)
        uids = _uids(4)
        members = [_member("a", uids, rng), _member("b", uids, rng)]
        matrix = column_weights(members, {label: ["a"] for label in LABELS})
        assert np.array_equal(matrix, np.array([[1.0] * N_LABELS, [0.0] * N_LABELS]))

    def test_column_weights_refuses_a_selection_that_forgets_a_label(self):
        selection = {label: ["a"] for label in LABELS[:-1]}
        with pytest.raises(ValueError, match=f"{LABELS[-1]!r}"):
            column_weights(["a"], selection)

    def test_column_weights_refuses_an_empty_column(self):
        selection = {label: ["a"] for label in LABELS}
        selection[LABELS[2]] = []
        with pytest.raises(ValueError, match="needs a member"):
            column_weights(["a"], selection)

    def test_column_weights_refuses_a_member_that_is_not_in_the_blend(self):
        selection = {label: ["a"] for label in LABELS}
        selection[LABELS[1]] = ["ghost"]
        with pytest.raises(ValueError, match="ghost"):
            column_weights(["a"], selection)

    def test_a_selection_blend_reports_the_per_column_table(self):
        rng = _rng(24)
        uids = _uids(8)
        members = [_member("a", uids, rng), _member("b", uids, rng)]
        selection = {label: ["a", "b"] for label in LABELS}
        selection[LABELS[0]] = ["a"]
        result = blend(members, column_weights(members, selection))
        assert not result.is_uniform
        report = result.report()
        assert "per-column weights" in report
        for label in LABELS:
            assert label in report

    def test_a_uniform_blend_says_so_and_reports_one_line_per_member(self):
        rng = _rng(25)
        uids = _uids(8)
        result = blend([_member("a", uids, rng), _member("b", uids, rng)])
        assert result.is_uniform
        assert "per-column weights" not in result.report()
        assert "0.5000" in result.report()


class TestSubmissionOutput:
    def test_the_frame_carries_exactly_the_submission_columns(self):
        rng = _rng(26)
        uids = _uids(5)
        frame = blend([_member("a", uids, rng)]).frame()
        assert tuple(frame.columns) == SUBMISSION_COLUMNS

    def test_a_written_submission_passes_all_seven_contract_checks(self, tmp_path):
        rng = _rng(27)
        uids = _uids(6)
        result = blend([_member("a", uids, rng), _member("b", uids, rng)])
        path = result.to_submission(tmp_path / "submission.csv", expected_uids=uids)
        check = validate_submission(path, expected_uids=uids)
        assert check.n_passed == 7, check.summary()

    def test_the_header_is_the_shared_writers_header_byte_for_byte(self, tmp_path):
        rng = _rng(28)
        uids = _uids(4)
        path = blend([_member("a", uids, rng)]).to_submission(tmp_path / "s.csv")
        with open(path, "rb") as handle:
            assert handle.readline() == expected_header_bytes()

    def test_expected_uids_reorder_the_rows(self, tmp_path):
        rng = _rng(29)
        uids = _uids(5)
        result = blend([_member("a", uids, rng)])
        wanted = uids[::-1]
        path = result.to_submission(tmp_path / "s.csv", expected_uids=wanted)
        assert validate_submission(path, expected_uids=wanted).n_passed == 7

    def test_a_study_the_blend_never_scored_raises_rather_than_shortening_the_file(
        self, tmp_path
    ):
        rng = _rng(30)
        uids = _uids(5)
        result = blend([_member("a", uids, rng)])
        with pytest.raises(ValueError, match="Every graded study needs a row"):
            result.to_submission(tmp_path / "s.csv", expected_uids=[*uids, "unscored"])

    def test_expected_uids_restrict_the_rows_to_the_graded_set(self, tmp_path):
        rng = _rng(31)
        uids = _uids(9)
        result = blend([_member("a", uids, rng)])
        wanted = uids[2:5]
        path = result.to_submission(tmp_path / "s.csv", expected_uids=wanted)
        check = validate_submission(path, expected_uids=wanted)
        assert check.n_rows == 3
        assert check.n_passed == 7


class TestOofArtifacts:
    def _table(self, tmp_path: Path, *, seed: int = 32, n_rows: int = 24, name: str = "oof"):
        rng = _rng(seed)
        uids = _uids(n_rows)
        scores = rng.random((n_rows, N_LABELS))
        truth = _binary_truth(rng, n_rows)
        mask = np.ones((n_rows, N_LABELS), dtype=bool)
        frame = _oof_frame(uids, scores, truth, mask)
        path = _write_oof(tmp_path / f"{name}.parquet", frame)
        return path, uids, scores, truth, mask

    def test_the_mirrored_schema_is_the_training_loops_schema(self):
        # The copy in kappa.ensemble exists so blending does not need torch. This is
        # the test that stops it drifting from the writer it claims to mirror.
        pytest.importorskip("torch")
        from kappa.training.loop import MASK_SUFFIX as loop_mask
        from kappa.training.loop import OOF_COLUMNS as loop_columns
        from kappa.training.loop import TRUE_SUFFIX as loop_true

        assert OOF_COLUMNS == loop_columns
        assert (TRUE_SUFFIX, MASK_SUFFIX) == (loop_true, loop_mask)

    def test_an_oof_table_round_trips_through_parquet(self, tmp_path):
        path, uids, scores, truth, mask = self._table(tmp_path)
        table = read_oof(path)
        assert list(table.study_uids) == uids
        assert np.allclose(table.scores, scores)
        assert np.array_equal(table.truth, truth)
        assert np.array_equal(table.mask, mask)
        assert table.member.name == "oof"
        assert np.allclose(table.member.scores, scores)

    def test_the_default_name_is_the_file_stem(self, tmp_path):
        path, *_ = self._table(tmp_path, name="oof_seed13")
        assert read_oof(path).name == "oof_seed13"
        assert read_oof(path, name="fold0").name == "fold0"

    def test_a_column_reordering_in_the_file_is_harmless(self, tmp_path):
        path, uids, scores, truth, mask = self._table(tmp_path, seed=33)
        frame = pd.read_parquet(path)
        shuffled = tmp_path / "shuffled.parquet"
        frame[list(reversed(OOF_COLUMNS))].to_parquet(shuffled, index=False)
        table = read_oof(shuffled)
        assert np.allclose(table.scores, scores)
        assert np.array_equal(table.truth, truth)

    def test_a_missing_mask_column_raises_and_names_it(self, tmp_path):
        path, *_ = self._table(tmp_path, seed=34)
        frame = pd.read_parquet(path)
        dropped = f"{LABELS[0]}{MASK_SUFFIX}"
        stripped = tmp_path / "stripped.parquet"
        frame.drop(columns=[dropped]).to_parquet(stripped, index=False)
        with pytest.raises(ValueError, match=dropped):
            read_oof(stripped)

    def test_an_unexpected_column_raises(self, tmp_path):
        path, *_ = self._table(tmp_path, seed=35)
        frame = pd.read_parquet(path)
        frame["surprise"] = 1.0
        widened = tmp_path / "widened.parquet"
        frame.to_parquet(widened, index=False)
        with pytest.raises(ValueError, match="surprise"):
            read_oof(widened)

    def test_two_runs_artifacts_concatenated_raise_instead_of_doubling_the_sample(
        self, tmp_path
    ):
        path, *_ = self._table(tmp_path, seed=36)
        frame = pd.read_parquet(path)
        doubled = tmp_path / "doubled.parquet"
        pd.concat([frame, frame], ignore_index=True).to_parquet(doubled, index=False)
        with pytest.raises(ValueError, match="more than once"):
            read_oof(doubled)


class TestScoringABlend:
    def _perfect(self, n_rows: int = 30, seed: int = 40):
        rng = _rng(seed)
        uids = _uids(n_rows)
        truth = _binary_truth(rng, n_rows)
        # Positives land in [10, 10.1), negatives in [0, 0.1): every column is
        # perfectly ordered, so a correct pipeline scores exactly 1.0 and a
        # mis-aligned one cannot.
        scores = truth * 10.0 + rng.random((n_rows, N_LABELS)) * 0.1
        other = truth * 10.0 + rng.random((n_rows, N_LABELS)) * 0.1
        return uids, truth, scores, other

    def test_two_perfectly_ordered_members_blend_to_a_macro_of_one(self, tmp_path):
        uids, truth, scores, other = self._perfect()
        mask = np.ones_like(truth, dtype=bool)
        path = _write_oof(tmp_path / "oof.parquet", _oof_frame(uids, scores, truth, mask))
        table = read_oof(path)
        result = blend([table.member, Member("b", uids, other)])
        auc = score_blend(result, table)
        assert auc.is_complete
        assert auc.macro == pytest.approx(1.0)

    def test_the_mask_decides_which_cells_are_scored(self, tmp_path):
        uids, truth, scores, _ = self._perfect(n_rows=40, seed=41)
        mask = np.ones_like(truth, dtype=bool)
        mask[5:, 0] = False  # only five cells survive on the first label
        mask[0:2, 0] = True  # keep one positive and one negative in the survivors
        path = _write_oof(tmp_path / "oof.parquet", _oof_frame(uids, scores, truth, mask))
        table = read_oof(path)
        auc = score_blend(blend([table.member]), table)
        assert auc.n_used[LABELS[0]] == 5
        assert auc.n_used[LABELS[1]] == 40

    def test_scoring_is_unmoved_by_the_blends_row_order(self, tmp_path):
        uids, truth, scores, other = self._perfect(seed=42)
        mask = np.ones_like(truth, dtype=bool)
        path = _write_oof(tmp_path / "oof.parquet", _oof_frame(uids, scores, truth, mask))
        table = read_oof(path)
        members = [table.member, Member("b", uids, other)]
        straight = score_blend(blend(members), table)
        shuffled_order = list(_rng(43).permutation(uids))
        shuffled = score_blend(blend(members, order=shuffled_order), table)
        assert shuffled.macro == pytest.approx(straight.macro)
        assert shuffled.per_label == pytest.approx(straight.per_label)

    def test_a_blend_holding_a_study_the_table_does_not_cannot_be_scored(self, tmp_path):
        uids, truth, scores, _ = self._perfect(n_rows=20, seed=44)
        mask = np.ones_like(truth, dtype=bool)
        short = _oof_frame(uids[:-1], scores[:-1], truth[:-1], mask[:-1])
        path = _write_oof(tmp_path / "oof.parquet", short)
        table = read_oof(path)
        stranger = blend([Member("a", uids, scores)])
        with pytest.raises(ValueError, match="no truth for 1"):
            score_blend(stranger, table)

    def test_a_single_class_label_is_reported_as_skipped_not_averaged_away(self, tmp_path):
        uids, truth, scores, _ = self._perfect(n_rows=20, seed=45)
        truth[:, 4] = 1.0
        mask = np.ones_like(truth, dtype=bool)
        path = _write_oof(tmp_path / "oof.parquet", _oof_frame(uids, scores, truth, mask))
        table = read_oof(path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            auc = score_blend(blend([table.member]), table)
        assert auc.skipped == (LABELS[4],)
        assert not auc.is_complete


class TestTheImportBoundary:
    """Submissions are image-only, and blending must not need the modelling extra."""

    def _module_path(self) -> Path:
        return repo_root() / "src" / "kappa" / "ensemble" / "__init__.py"

    def test_the_module_names_no_report_reader_in_any_import(self):
        # ast.walk, not tree.body: constraint A1 forbids reaching kappa.labels at
        # call time as well, so a deferred import inside a function is no better
        # than a module-scope one here.
        tree = ast.parse(self._module_path().read_text(encoding="utf-8"))
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        bad = [
            name
            for name in imported
            if name == FORBIDDEN_PREFIX or name.startswith(FORBIDDEN_PREFIX + ".")
        ]
        assert not bad, f"kappa/ensemble/__init__.py imports {bad}"

    def test_importing_the_module_in_a_clean_interpreter_loads_no_report_reader(self):
        # A clean interpreter, because this session has already imported half the
        # project and sys.modules would answer for all of it.
        code = (
            "import sys, json;"
            "import kappa.ensemble;"
            "print(json.dumps(sorted("
            f"m for m in sys.modules if m == {FORBIDDEN_PREFIX!r}"
            f" or m.startswith({FORBIDDEN_PREFIX!r} + '.')"
            ")))"
        )
        done = subprocess.run(
            [sys.executable, "-c", code],
            cwd=repo_root(),
            capture_output=True,
            text=True,
            env={"PYTHONPATH": str(repo_root() / "src"), "PATH": "/usr/bin:/bin"},
        )
        assert done.returncode == 0, done.stderr
        assert done.stdout.strip() == "[]", (
            f"kappa.ensemble loaded report-reading module(s): {done.stdout.strip()}. "
            "Submissions are image-only (constraint A1)."
        )

    def test_importing_the_module_does_not_require_the_modelling_extra(self):
        # pyproject confines module-scope torch to kappa.models and kappa.training.
        # Blending saved artifacts is exactly the step that has to run on a machine
        # where the train extra was never installed, so this is not style.
        code = "import sys; import kappa.ensemble; print('torch' in sys.modules)"
        done = subprocess.run(
            [sys.executable, "-c", code],
            cwd=repo_root(),
            capture_output=True,
            text=True,
            env={"PYTHONPATH": str(repo_root() / "src"), "PATH": "/usr/bin:/bin"},
        )
        assert done.returncode == 0, done.stderr
        assert done.stdout.strip() == "False", "importing kappa.ensemble pulled in torch"

    def test_the_forbidden_prefix_is_the_one_the_project_pins(self):
        # The literal above exists so the two tests before this one never skip. This
        # is what stops it drifting away from the project's own definition.
        pytest.importorskip("torch")
        from kappa.inference.predict import FORBIDDEN_PREFIX as pinned

        assert FORBIDDEN_PREFIX == pinned
