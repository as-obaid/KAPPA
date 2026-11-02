"""The macro must be macro-12, or say out loud that it is not.

Two numbers in this project are easy to fake and impossible to notice being faked.
The first is the macro's denominator: ``nanmean`` over twelve labels where one has a
single class present returns the mean of eleven and still calls itself macro. The
second is coverage: 98.7% of raw label cells are NaN, so a per-label positive rate
printed against the row count overstates support -- a prior validation table in this
project did exactly that on ten of twelve rows. These tests fail if either number
goes back to being implicit.

The AUC itself is checked the only way worth checking it: against
``sklearn.metrics.roc_auc_score`` on synthetic input, including the tied-score case
where a hand-rolled Mann-Whitney silently diverges. ``scikit-learn`` is a declared
dependency of this package, so the comparison is not optional and is never skipped.

A third invariant lives here because it has nowhere better to be: ROC-AUC reads only
the order of the scores, so the only correct way to blend two members is per-column
rank normalisation. The test that matters is that two monotonically related score
vectors blend to the same thing -- probability averaging cannot pass it.

A fourth family covers :func:`diagnostic_metrics`: average precision against
``sklearn.metrics.average_precision_score``, and fixed-threshold precision, recall
and F1 against ``sklearn.metrics.precision_recall_fscore_support``, both including
the heavily-tied-score case. A zero-positive label must be ``nan`` in every one of
these -- ``ap``, ``precision``, ``recall``, ``f1``, ``f1_best``, ``threshold_best``
-- and named in ``skipped``, the same discipline :class:`AucResult` already
enforces, not ``sklearn``'s own ``0.0``-with-a-warning for the same input. The best
achievable F1 is checked against a brute-force sweep over every unique score,
because that is the only independent way to know a claimed maximum is not a local
one.
"""

from __future__ import annotations

import re
import warnings

import numpy as np
import pytest
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from kappa.constants import LABELS, N_LABELS
from kappa.evaluation.metrics import (
    FIXED_THRESHOLD,
    NOISE_FLOOR_MACRO,
    AucResult,
    DiagnosticResult,
    average_ranks,
    diagnostic_metrics,
    fold_summary,
    is_evidence,
    macro_roc_auc,
    pr_curve_points,
    rank_average,
    rank_normalise,
    roc_curve_points,
)


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _truth(rng: np.random.Generator, n_rows: int = 40, rate: float = 0.35) -> np.ndarray:
    """Binary truth with both classes forced into every column.

    Forced on purpose: a column that happens to be single-class is a *skip*, and a
    test that did not ask for a skip should not silently be testing one.
    """
    y = (rng.random((n_rows, N_LABELS)) < rate).astype(np.float64)
    y[0, :] = 1.0
    y[1, :] = 0.0
    return y


def _scores(rng: np.random.Generator, n_rows: int = 40) -> np.ndarray:
    return rng.random((n_rows, N_LABELS))


def _result(**over) -> AucResult:
    """A hand-built AucResult, for exercising the reporting without an AUC."""
    base = {
        "macro": 0.8,
        "per_label": {name: 0.8 for name in LABELS},
        "n_used": {name: 40 for name in LABELS},
        "n_pos": {name: 10 for name in LABELS},
        "skipped": (),
    }
    base.update(over)
    return AucResult(**base)  # type: ignore[arg-type]


def _quiet(y_true, y_score, mask=None, **kw) -> AucResult:
    """Score without asserting on the skip warning, for tests about something else."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return macro_roc_auc(y_true, y_score, mask, **kw)


def _quiet_diag(y_true, y_score, mask=None, **kw) -> DiagnosticResult:
    """:func:`diagnostic_metrics` without asserting on the skip warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return diagnostic_metrics(y_true, y_score, mask, **kw)


class TestSklearnEquivalence:
    def test_per_label_auc_reproduces_sklearn_on_continuous_scores(self):
        rng = _rng(1)
        for _ in range(20):
            y, s = _truth(rng), _scores(rng)
            result = macro_roc_auc(y, s)
            for column, name in enumerate(LABELS):
                assert result.per_label[name] == pytest.approx(
                    roc_auc_score(y[:, column], s[:, column]), abs=1e-12
                )

    def test_per_label_auc_reproduces_sklearn_when_scores_are_heavily_tied(self):
        # Ties are where a hand-rolled AUC diverges from sklearn: they must take
        # their mid-rank, which is worth half credit, not first or last place.
        rng = _rng(2)
        for _ in range(20):
            y = _truth(rng)
            s = rng.integers(0, 3, size=y.shape).astype(np.float64)
            result = macro_roc_auc(y, s)
            for column, name in enumerate(LABELS):
                assert result.per_label[name] == pytest.approx(
                    roc_auc_score(y[:, column], s[:, column]), abs=1e-12
                )

    def test_macro_reproduces_sklearn_macro_average(self):
        rng = _rng(3)
        y, s = _truth(rng), _scores(rng)
        expected = roc_auc_score(y, s, average="macro")
        assert macro_roc_auc(y, s).macro == pytest.approx(expected, abs=1e-12)

    def test_masked_columns_reproduce_sklearn_on_the_surviving_rows(self):
        rng = _rng(4)
        y, s = _truth(rng, n_rows=60), _scores(rng, n_rows=60)
        mask = rng.random(y.shape) > 0.3
        mask[0, :] = True
        mask[1, :] = True
        result = macro_roc_auc(y, s, mask)
        for column, name in enumerate(LABELS):
            keep = mask[:, column]
            assert result.per_label[name] == pytest.approx(
                roc_auc_score(y[keep, column], s[keep, column]), abs=1e-12
            )

    def test_perfect_separation_is_one_and_inversion_is_zero(self):
        y = np.tile(np.array([[0.0], [0.0], [1.0], [1.0]]), (1, N_LABELS))
        s = np.tile(np.array([[0.1], [0.2], [0.8], [0.9]]), (1, N_LABELS))
        assert macro_roc_auc(y, s).macro == pytest.approx(1.0)
        assert macro_roc_auc(y, -s).macro == pytest.approx(0.0)

    def test_a_constant_prediction_is_exactly_one_half(self):
        y = np.tile(np.array([[0.0], [1.0], [1.0], [0.0]]), (1, N_LABELS))
        s = np.full_like(y, 0.5)
        result = macro_roc_auc(y, s)
        assert result.macro == pytest.approx(0.5)
        assert result.is_complete


class TestSingleClassLabels:
    def _one_single_class_column(self):
        rng = _rng(5)
        y, s = _truth(rng), _scores(rng)
        y[:, 3] = 0.0  # Lateral Meniscus: every study negative, so no AUC exists.
        return y, s

    def test_a_single_class_label_is_nan_and_named_in_skipped(self):
        y, s = self._one_single_class_column()
        result = _quiet(y, s)
        assert result.skipped == ("Lateral Meniscus",)
        assert np.isnan(result.per_label["Lateral Meniscus"])

    def test_a_skipped_label_keeps_its_row_so_the_table_still_has_twelve(self):
        y, s = self._one_single_class_column()
        result = _quiet(y, s)
        assert list(result.per_label) == list(LABELS)
        assert len(result.per_label) == 12

    def test_a_skipped_label_is_excluded_from_the_macro_and_the_macro_says_so(self):
        y, s = self._one_single_class_column()
        result = _quiet(y, s)
        scored = [v for k, v in result.per_label.items() if k != "Lateral Meniscus"]
        assert result.macro == pytest.approx(float(np.mean(scored)))
        assert result.n_scored == 11
        assert not result.is_complete

    def test_eleven_labels_averaged_is_never_reported_as_complete(self):
        # The cheap wrong fix is np.nanmean, which returns this very number with
        # is_complete still True and nothing named. That must not be reachable.
        y, s = self._one_single_class_column()
        result = _quiet(y, s)
        assert result.macro == pytest.approx(float(np.nanmean(list(result.per_label.values()))))
        assert not result.is_complete
        assert result.skipped

    def test_skipping_a_label_warns_that_this_is_not_macro_twelve(self):
        y, s = self._one_single_class_column()
        with pytest.warns(RuntimeWarning, match="not the competition's macro-12"):
            macro_roc_auc(y, s)

    def test_scoring_all_twelve_labels_does_not_warn(self):
        rng = _rng(6)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = macro_roc_auc(_truth(rng), _scores(rng))
        assert result.is_complete

    def test_every_label_single_class_gives_a_nan_macro_not_a_zero(self):
        y = np.zeros((10, N_LABELS))
        s = _scores(_rng(7), n_rows=10)
        result = _quiet(y, s)
        assert np.isnan(result.macro)
        assert len(result.skipped) == 12
        assert result.n_scored == 0

    def test_the_table_names_the_skipped_labels(self):
        y, s = self._one_single_class_column()
        table = _quiet(y, s).table()
        assert "skipped (one class present): Lateral Meniscus" in table
        assert "macro" in table and "over 11 of 12 labels" in table


class TestCoverageDenominator:
    def test_n_used_is_per_label_and_is_not_the_row_count(self):
        rng = _rng(8)
        y, s = _truth(rng, n_rows=58), _scores(rng, n_rows=58)
        y[10:, 0] = np.nan  # ACL known for ten studies only.
        result = _quiet(y, s)
        assert result.n_used["ACL"] == 10
        assert result.n_used["MCL"] == 58

    def test_n_pos_counts_only_the_cells_that_were_used(self):
        y = np.zeros((6, N_LABELS))
        y[0, :] = 1.0
        y[1, :] = 1.0
        s = _scores(_rng(9), n_rows=6)
        mask = np.ones_like(y, dtype=bool)
        mask[1, 0] = False  # hide one ACL positive
        result = _quiet(y, s, mask)
        assert result.n_pos["ACL"] == 1
        assert result.n_used["ACL"] == 5
        assert result.n_pos["MCL"] == 2
        assert result.n_used["MCL"] == 6

    def test_an_entirely_unknown_label_reports_zero_support_not_a_score(self):
        rng = _rng(10)
        y, s = _truth(rng), _scores(rng)
        y[:, 8] = np.nan  # Synovitis never adjudicated.
        result = _quiet(y, s)
        assert result.n_used["Synovitis"] == 0
        assert result.n_pos["Synovitis"] == 0
        assert "Synovitis" in result.skipped

    def test_the_table_prints_every_rate_next_to_its_own_denominator(self):
        # Blocker B8: a positive rate over the row count when the row actually
        # covered fewer studies. The rate is only allowed out with its n.
        result = _result(
            n_used={name: (10 if name == "ACL" else 58) for name in LABELS},
            n_pos={name: (5 if name == "ACL" else 29) for name in LABELS},
        )
        line = next(row for row in result.table().splitlines() if row.startswith("ACL"))
        assert re.search(r"\b10\b", line), line
        assert "50.0" in line, line

    def test_the_table_prints_a_dash_rather_than_dividing_by_zero_support(self):
        result = _result(n_used={name: 0 for name in LABELS}, n_pos={name: 0 for name in LABELS})
        line = next(row for row in result.table().splitlines() if row.startswith("ACL"))
        assert line.endswith("-")


class TestNanHandling:
    def test_nan_truth_cells_are_dropped_and_never_reach_the_macro(self):
        rng = _rng(11)
        y, s = _truth(rng, n_rows=80), _scores(rng, n_rows=80)
        y[rng.random(y.shape) < 0.6] = np.nan
        y[0, :], y[1, :] = 1.0, 0.0
        result = _quiet(y, s)
        assert np.isfinite(result.macro)
        for column, name in enumerate(LABELS):
            keep = ~np.isnan(y[:, column])
            assert result.per_label[name] == pytest.approx(
                roc_auc_score(y[keep, column], s[keep, column]), abs=1e-12
            )

    def test_a_column_that_is_nan_except_for_one_class_is_skipped_not_scored(self):
        y = np.full((20, N_LABELS), np.nan)
        y[0, :], y[1, :] = 1.0, 0.0
        y[1, 5] = 1.0  # Lateral OA: two positives and nothing else known.
        s = _scores(_rng(12), n_rows=20)
        result = _quiet(y, s)
        assert result.n_used["Lateral OA"] == 2
        assert "Lateral OA" in result.skipped
        assert np.isfinite(result.per_label["ACL"])

    def test_nan_truth_is_dropped_even_where_the_mask_says_keep(self):
        rng = _rng(13)
        y, s = _truth(rng), _scores(rng)
        y[5, 0] = np.nan
        mask = np.ones_like(y, dtype=bool)
        assert _quiet(y, s, mask).n_used["ACL"] == y.shape[0] - 1

    def test_a_nan_score_in_a_kept_cell_raises_rather_than_being_dropped(self):
        rng = _rng(14)
        y, s = _truth(rng), _scores(rng)
        s[3, 2] = np.nan
        with pytest.raises(ValueError, match="NaN prediction"):
            macro_roc_auc(y, s)

    def test_a_nan_score_the_mask_already_drops_is_tolerated(self):
        rng = _rng(15)
        y, s = _truth(rng), _scores(rng)
        s[3, 2] = np.nan
        mask = np.ones_like(y, dtype=bool)
        mask[3, 2] = False
        assert np.isfinite(macro_roc_auc(y, s, mask).macro)


class TestInputValidation:
    def test_one_dimensional_truth_raises(self):
        with pytest.raises(ValueError, match="n_studies, n_labels"):
            macro_roc_auc(np.zeros(12), np.zeros(12))

    def test_disagreeing_shapes_raise(self):
        with pytest.raises(ValueError, match="disagree"):
            macro_roc_auc(np.zeros((4, N_LABELS)), np.zeros((5, N_LABELS)))

    def test_the_wrong_number_of_columns_raises(self):
        with pytest.raises(ValueError, match="expected 12"):
            macro_roc_auc(np.zeros((4, 11)), np.zeros((4, 11)))

    def test_a_mask_of_the_wrong_shape_raises(self):
        y = _truth(_rng(16), n_rows=4)
        with pytest.raises(ValueError, match="does not match"):
            macro_roc_auc(y, y, np.ones((4, 3), dtype=bool))

    def test_non_binary_truth_raises_rather_than_scoring_a_probability(self):
        rng = _rng(17)
        y, s = _truth(rng), _scores(rng)
        y[2, 4] = 0.5
        with pytest.raises(ValueError, match="outside"):
            macro_roc_auc(y, s)

    def test_a_duplicate_label_name_raises_instead_of_shrinking_the_denominator(self):
        # Keying per_label by name means a duplicate collapses two columns into one
        # entry: the macro is then over eleven with is_complete still True.
        rng = _rng(18)
        y, s = _truth(rng), _scores(rng)
        names = list(LABELS)
        names[1] = names[0]
        with pytest.raises(ValueError, match="duplicate"):
            macro_roc_auc(y, s, labels=names)


class TestFoldSpread:
    def _folds(self, n_folds: int = 4) -> list[AucResult]:
        rng = _rng(19)
        out = []
        for _ in range(n_folds):
            y, s = _truth(rng, n_rows=30), _scores(rng, n_rows=30)
            out.append(macro_roc_auc(y, s))
        return out

    def test_the_summary_reports_a_spread_for_the_macro_and_for_all_twelve_labels(self):
        summary = fold_summary(self._folds())
        assert np.isfinite(summary.macro_std)
        assert list(summary.per_label_mean) == list(LABELS)
        assert list(summary.per_label_std) == list(LABELS)
        assert all(np.isfinite(v) for v in summary.per_label_std.values())

    def test_the_spread_is_the_sample_standard_deviation_not_the_population_one(self):
        folds = self._folds()
        macros = np.asarray([f.macro for f in folds])
        assert fold_summary(folds).macro_std == pytest.approx(float(macros.std(ddof=1)))
        assert fold_summary(folds).macro_std != pytest.approx(float(macros.std()))

    def test_one_fold_has_a_nan_spread_rather_than_a_confident_zero(self):
        summary = fold_summary(self._folds(n_folds=1))
        assert summary.n_folds == 1
        assert np.isnan(summary.macro_std)
        assert all(np.isnan(v) for v in summary.per_label_std.values())

    def test_a_label_skipped_in_one_fold_is_counted_and_averaged_over_the_rest(self):
        rng = _rng(20)
        folds = []
        for fold in range(3):
            y, s = _truth(rng, n_rows=30), _scores(rng, n_rows=30)
            if fold == 0:
                y[:, 11] = 0.0  # Fracture absent from this fold entirely.
            folds.append(_quiet(y, s))
        summary = fold_summary(folds)
        assert summary.fold_skips["Fracture"] == 1
        assert summary.fold_skips["ACL"] == 0
        others = [f.per_label["Fracture"] for f in folds[1:]]
        assert summary.per_label_mean["Fracture"] == pytest.approx(float(np.mean(others)))

    def test_a_label_skipped_in_every_fold_is_nan_not_absent(self):
        rng = _rng(21)
        folds = []
        for _ in range(3):
            y, s = _truth(rng, n_rows=30), _scores(rng, n_rows=30)
            y[:, 11] = 0.0
            folds.append(_quiet(y, s))
        summary = fold_summary(folds)
        assert "Fracture" in summary.per_label_mean
        assert np.isnan(summary.per_label_mean["Fracture"])
        assert summary.fold_skips["Fracture"] == 3

    def test_folds_that_disagree_on_their_label_set_cannot_be_averaged(self):
        folds = self._folds(n_folds=2)
        broken = _result(per_label={"ACL": 0.7}, n_used={"ACL": 1}, n_pos={"ACL": 1})
        with pytest.raises(ValueError, match="label set"):
            fold_summary([folds[0], broken])

    def test_the_summary_needs_at_least_one_fold(self):
        with pytest.raises(ValueError, match="at least one"):
            fold_summary([])

    def test_the_fold_table_prints_the_spread_and_the_noise_floor_together(self):
        table = fold_summary(self._folds()).table()
        assert "+/-" in table
        assert f"noise floor {NOISE_FLOOR_MACRO:.4f}" in table
        assert "over 4 fold(s)" in table


class TestNoiseFloor:
    def test_the_floor_is_the_recorded_figure(self):
        assert NOISE_FLOOR_MACRO == 0.0020

    def test_a_delta_smaller_than_the_floor_is_not_evidence(self):
        assert not is_evidence(0.0011)
        assert not is_evidence(-0.0011)
        assert not is_evidence(NOISE_FLOOR_MACRO)

    def test_a_delta_the_size_of_the_leaderboard_gap_is_evidence(self):
        assert is_evidence(0.006)
        assert is_evidence(-0.006)

    def test_a_nan_delta_is_not_evidence(self):
        assert not is_evidence(float("nan"))

    def test_the_floor_can_be_overridden_for_a_run_that_measured_its_own(self):
        assert is_evidence(0.003, noise_floor=0.001)
        assert not is_evidence(0.003, noise_floor=0.01)


class TestAverageRanks:
    def test_ties_share_their_mid_rank(self):
        got = average_ranks(np.array([5.0, 1.0, 5.0, 3.0]))
        assert got.tolist() == [3.5, 1.0, 3.5, 2.0]

    def test_a_single_value_ranks_one(self):
        assert average_ranks(np.array([9.0])).tolist() == [1.0]

    def test_an_all_tied_vector_shares_one_mid_rank(self):
        assert average_ranks(np.full(4, 2.0)).tolist() == [2.5] * 4

    def test_an_empty_vector_ranks_empty(self):
        assert average_ranks(np.array([])).size == 0


class TestRankNormalisation:
    def test_it_is_invariant_to_any_monotone_rescaling(self):
        rng = _rng(22)
        scores = rng.random((25, N_LABELS))
        rescaled = np.exp(3.0 * scores) * 17.0 - 4.0
        assert np.array_equal(rank_normalise(scores), rank_normalise(rescaled))

    def test_it_leaves_every_per_label_auc_bit_identical(self):
        rng = _rng(23)
        y, s = _truth(rng), _scores(rng)
        before = macro_roc_auc(y, s)
        after = macro_roc_auc(y, rank_normalise(s))
        assert after.macro == before.macro
        assert after.per_label == before.per_label

    def test_each_column_is_normalised_on_its_own_scale(self):
        # A column measured in a different unit must not drag its neighbours.
        scores = np.column_stack([np.array([1.0, 2.0, 3.0]), np.array([1e6, 2e6, 3e6])])
        out = rank_normalise(scores)
        assert np.array_equal(out[:, 0], out[:, 1])

    def test_permuting_one_column_leaves_the_others_untouched(self):
        rng = _rng(24)
        scores = rng.random((12, N_LABELS))
        shuffled = scores.copy()
        shuffled[:, 0] = shuffled[::-1, 0]
        out, out_shuffled = rank_normalise(scores), rank_normalise(shuffled)
        assert np.array_equal(out[:, 1:], out_shuffled[:, 1:])

    def test_the_output_stays_strictly_inside_zero_and_one(self):
        rng = _rng(25)
        out = rank_normalise(rng.random((30, N_LABELS)))
        assert out.min() > 0.0
        assert out.max() < 1.0

    def test_a_single_row_maps_to_one_half_rather_than_dividing_by_zero(self):
        out = rank_normalise(np.array([[0.3] * N_LABELS]))
        assert out.tolist() == [[0.5] * N_LABELS]

    def test_a_nan_score_has_no_rank_and_raises(self):
        scores = np.zeros((3, N_LABELS))
        scores[1, 1] = np.nan
        with pytest.raises(ValueError, match="no rank"):
            rank_normalise(scores)

    def test_one_dimensional_input_raises(self):
        with pytest.raises(ValueError, match="n_studies, n_labels"):
            rank_normalise(np.zeros(N_LABELS))


class TestRankAverage:
    def test_two_monotonically_related_members_blend_to_either_one_of_them(self):
        # The definition of rank-based: a member cannot buy influence by being
        # spread wider, because only its order survives.
        rng = _rng(26)
        member = rng.random((20, N_LABELS))
        stretched = np.exp(5.0 * member) * 1000.0
        blended = rank_average([member, stretched])
        assert np.allclose(blended, rank_normalise(member))

    def test_probability_averaging_would_change_the_ranking_and_this_does_not(self):
        # Member B is over-confident: averaging probabilities lets it decide the
        # order on its own. Rank averaging gives the two members equal say.
        a = np.array([[0.40], [0.60], [0.45]])  # ranks the second study first
        b = np.array([[0.99], [0.55], [0.01]])  # ranks the first study first, loudly
        naive = (a + b) / 2.0
        assert np.argmax(naive[:, 0]) == 0  # b's spread decided it alone
        blended = rank_average([a, b])
        assert np.argmax(blended[:, 0]) == 1  # two votes each, and a's first place wins

    def test_a_single_member_is_just_its_own_rank_normalisation(self):
        rng = _rng(27)
        member = rng.random((10, N_LABELS))
        assert np.array_equal(rank_average([member]), rank_normalise(member))

    def test_weights_shift_the_blend_towards_the_heavier_member(self):
        rng = _rng(28)
        a, b = rng.random((15, N_LABELS)), rng.random((15, N_LABELS))
        heavy = rank_average([a, b], weights=[3.0, 1.0])
        expected = 0.75 * rank_normalise(a) + 0.25 * rank_normalise(b)
        assert np.allclose(heavy, expected)

    def test_weights_are_normalised_so_the_output_stays_in_zero_one(self):
        rng = _rng(29)
        a, b = rng.random((15, N_LABELS)), rng.random((15, N_LABELS))
        out = rank_average([a, b], weights=[10.0, 10.0])
        assert out.min() > 0.0
        assert out.max() < 1.0

    def test_no_members_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            rank_average([])

    def test_members_of_different_shapes_raise(self):
        rng = _rng(30)
        with pytest.raises(ValueError, match="shape"):
            rank_average([rng.random((5, N_LABELS)), rng.random((6, N_LABELS))])

    def test_the_wrong_number_of_weights_raises(self):
        rng = _rng(31)
        members = [rng.random((5, N_LABELS)), rng.random((5, N_LABELS))]
        with pytest.raises(ValueError, match="weights"):
            rank_average(members, weights=[1.0])

    def test_weights_that_do_not_sum_to_something_positive_raise(self):
        rng = _rng(32)
        members = [rng.random((5, N_LABELS)), rng.random((5, N_LABELS))]
        with pytest.raises(ValueError, match="positive"):
            rank_average(members, weights=[1.0, -1.0])

    def test_a_blend_is_scored_by_the_same_metric_without_calibration(self):
        # End to end: two members, rank-blended, scored. Nothing here fits a
        # calibrator, and the metric would be identical if anything did.
        rng = _rng(33)
        y = _truth(rng, n_rows=50)
        a = y * 0.6 + rng.random(y.shape) * 0.4
        b = y * 0.5 + rng.random(y.shape) * 0.5
        blended = macro_roc_auc(y, rank_average([a, b]))
        assert blended.is_complete
        assert 0.5 < blended.macro <= 1.0


class TestDiagnosticSklearnEquivalence:
    def test_average_precision_reproduces_sklearn_on_continuous_scores(self):
        rng = _rng(40)
        for _ in range(20):
            y, s = _truth(rng), _scores(rng)
            result = diagnostic_metrics(y, s)
            for column, name in enumerate(LABELS):
                assert result.ap[name] == pytest.approx(
                    average_precision_score(y[:, column], s[:, column]), abs=1e-9
                )

    def test_average_precision_reproduces_sklearn_when_scores_are_heavily_tied(self):
        # Ties are where the step-function integral is easiest to get wrong: a
        # tied group is one operating point, not one per row.
        rng = _rng(41)
        for _ in range(20):
            y = _truth(rng)
            s = rng.integers(0, 3, size=y.shape).astype(np.float64)
            result = diagnostic_metrics(y, s)
            for column, name in enumerate(LABELS):
                assert result.ap[name] == pytest.approx(
                    average_precision_score(y[:, column], s[:, column]), abs=1e-9
                )

    def test_fixed_threshold_precision_recall_f1_reproduce_sklearn(self):
        rng = _rng(42)
        for _ in range(20):
            y, s = _truth(rng), _scores(rng)
            result = diagnostic_metrics(y, s)
            for column, name in enumerate(LABELS):
                pred = (s[:, column] >= FIXED_THRESHOLD).astype(int)
                p, r, f, _ = precision_recall_fscore_support(
                    y[:, column], pred, average="binary", zero_division=0
                )
                assert result.precision[name] == pytest.approx(p, abs=1e-12)
                assert result.recall[name] == pytest.approx(r, abs=1e-12)
                assert result.f1[name] == pytest.approx(f, abs=1e-12)

    def test_fixed_threshold_precision_recall_f1_reproduce_sklearn_on_tied_scores(self):
        rng = _rng(43)
        for _ in range(20):
            y = _truth(rng)
            s = rng.integers(0, 2, size=y.shape).astype(np.float64)  # ties on both sides of 0.5
            result = diagnostic_metrics(y, s)
            for column, name in enumerate(LABELS):
                pred = (s[:, column] >= FIXED_THRESHOLD).astype(int)
                p, r, f, _ = precision_recall_fscore_support(
                    y[:, column], pred, average="binary", zero_division=0
                )
                assert result.precision[name] == pytest.approx(p, abs=1e-12)
                assert result.recall[name] == pytest.approx(r, abs=1e-12)
                assert result.f1[name] == pytest.approx(f, abs=1e-12)

    def test_masked_columns_reproduce_sklearn_on_the_surviving_rows(self):
        rng = _rng(44)
        y, s = _truth(rng, n_rows=60), _scores(rng, n_rows=60)
        mask = rng.random(y.shape) > 0.3
        mask[0, :] = True
        mask[1, :] = True
        result = diagnostic_metrics(y, s, mask)
        for column, name in enumerate(LABELS):
            keep = mask[:, column]
            assert result.ap[name] == pytest.approx(
                average_precision_score(y[keep, column], s[keep, column]), abs=1e-9
            )


class TestFixedThreshold:
    def test_the_threshold_is_a_named_constant_of_one_half(self):
        assert FIXED_THRESHOLD == 0.5

    def test_diagnostic_metrics_defaults_to_the_named_threshold(self):
        rng = _rng(45)
        y, s = _truth(rng), _scores(rng)
        assert diagnostic_metrics(y, s).threshold == FIXED_THRESHOLD

    def test_a_different_threshold_can_be_passed_and_is_recorded(self):
        rng = _rng(46)
        y, s = _truth(rng), _scores(rng)
        result = diagnostic_metrics(y, s, threshold=0.9)
        assert result.threshold == 0.9


class TestBestAchievableF1:
    def test_the_reported_threshold_actually_achieves_the_reported_f1(self):
        rng = _rng(47)
        for _ in range(15):
            y, s = _truth(rng, n_rows=30), _scores(rng, n_rows=30)
            result = diagnostic_metrics(y, s)
            for column, name in enumerate(LABELS):
                pred = (s[:, column] >= result.threshold_best[name]).astype(int)
                achieved = f1_score(y[:, column], pred, zero_division=0)
                assert achieved == pytest.approx(result.f1_best[name], abs=1e-9)

    def test_best_f1_matches_a_brute_force_search_over_every_unique_score(self):
        # The independent check: sweep every candidate threshold by hand and
        # confirm the claimed maximum is not a local one.
        rng = _rng(48)
        for _ in range(10):
            y, s = _truth(rng, n_rows=25), _scores(rng, n_rows=25)
            result = diagnostic_metrics(y, s)
            for column, name in enumerate(LABELS):
                truth, scores = y[:, column], s[:, column]
                best = max(
                    f1_score(truth, (scores >= t).astype(int), zero_division=0)
                    for t in np.unique(scores)
                )
                assert result.f1_best[name] == pytest.approx(best, abs=1e-9)

    def test_best_f1_is_never_worse_than_the_fixed_threshold_f1(self):
        # The search covers every achievable point, so it cannot do worse than
        # the one arbitrary point diagnostic_metrics also reports.
        rng = _rng(49)
        y, s = _truth(rng, n_rows=50), _scores(rng, n_rows=50)
        result = diagnostic_metrics(y, s)
        for name in LABELS:
            assert result.f1_best[name] >= result.f1[name] - 1e-12


class TestDiagnosticZeroPositiveLabels:
    def _zero_positive_column(self):
        rng = _rng(50)
        y, s = _truth(rng), _scores(rng)
        y[:, 3] = 0.0  # Lateral Meniscus: nobody positive, so nothing here is defined.
        return y, s

    def test_a_zero_positive_label_is_nan_in_every_diagnostic_and_named_skipped(self):
        y, s = self._zero_positive_column()
        result = _quiet_diag(y, s)
        assert result.skipped == ("Lateral Meniscus",)
        for values in (
            result.ap, result.precision, result.recall, result.f1,
            result.f1_best, result.threshold_best,
        ):
            assert np.isnan(values["Lateral Meniscus"])

    def test_a_skipped_label_keeps_its_row_so_every_dict_still_has_twelve(self):
        y, s = self._zero_positive_column()
        result = _quiet_diag(y, s)
        for values in (result.ap, result.precision, result.recall, result.f1):
            assert list(values) == list(LABELS)
            assert len(values) == 12

    def test_skipped_labels_are_excluded_from_every_macro_and_the_macro_says_so(self):
        y, s = self._zero_positive_column()
        result = _quiet_diag(y, s)
        for values, macro in (
            (result.ap, result.ap_macro),
            (result.precision, result.precision_macro),
            (result.recall, result.recall_macro),
            (result.f1, result.f1_macro),
        ):
            scored = [v for k, v in values.items() if k != "Lateral Meniscus"]
            assert macro == pytest.approx(float(np.mean(scored)))
        assert result.n_scored == 11
        assert not result.is_complete

    def test_eleven_labels_averaged_is_never_mistaken_for_complete(self):
        # The cheap wrong fix is np.nanmean, which returns this very number
        # with is_complete still True and nothing named -- must not be reachable.
        y, s = self._zero_positive_column()
        result = _quiet_diag(y, s)
        assert result.ap_macro == pytest.approx(float(np.nanmean(list(result.ap.values()))))
        assert not result.is_complete
        assert result.skipped

    def test_skipping_a_label_warns_that_its_diagnostics_are_not_defined(self):
        y, s = self._zero_positive_column()
        with pytest.warns(RuntimeWarning, match="zero positives"):
            diagnostic_metrics(y, s)

    def test_scoring_every_label_does_not_warn(self):
        rng = _rng(51)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = diagnostic_metrics(_truth(rng), _scores(rng))
        assert result.is_complete

    def test_every_label_zero_positive_gives_nan_macros_not_zero(self):
        y = np.zeros((10, N_LABELS))
        s = _scores(_rng(52), n_rows=10)
        result = _quiet_diag(y, s)
        assert np.isnan(result.ap_macro)
        assert np.isnan(result.f1_macro)
        assert len(result.skipped) == 12
        assert result.n_scored == 0

    def test_an_all_positive_label_is_defined_and_not_skipped(self):
        # Unlike AucResult, which needs both classes present: average precision,
        # precision, recall and F1 are all well-defined when every used cell is
        # positive, so this is not the same failure as a zero-positive column.
        rng = _rng(53)
        y, s = _truth(rng), _scores(rng)
        column = LABELS.index("PF OA")
        y[:, column] = 1.0  # PF OA: every study positive.
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = diagnostic_metrics(y, s)
        assert "PF OA" not in result.skipped
        assert result.ap["PF OA"] == pytest.approx(1.0)
        assert result.is_complete


class TestDiagnosticCoverageDenominator:
    def test_n_used_is_per_label_and_is_not_the_row_count(self):
        rng = _rng(54)
        y, s = _truth(rng, n_rows=58), _scores(rng, n_rows=58)
        y[10:, 0] = np.nan  # ACL known for ten studies only.
        result = _quiet_diag(y, s)
        assert result.n_used["ACL"] == 10
        assert result.n_used["MCL"] == 58

    def test_n_pos_counts_only_the_cells_that_were_used(self):
        y = np.zeros((6, N_LABELS))
        y[0, :] = 1.0
        y[1, :] = 1.0
        s = _scores(_rng(55), n_rows=6)
        mask = np.ones_like(y, dtype=bool)
        mask[1, 0] = False  # hide one ACL positive
        result = _quiet_diag(y, s, mask)
        assert result.n_pos["ACL"] == 1
        assert result.n_used["ACL"] == 5
        assert result.n_pos["MCL"] == 2
        assert result.n_used["MCL"] == 6


class TestDiagnosticNanHandling:
    def test_nan_truth_cells_are_dropped_and_never_reach_a_denominator(self):
        rng = _rng(56)
        y, s = _truth(rng, n_rows=80), _scores(rng, n_rows=80)
        y[rng.random(y.shape) < 0.6] = np.nan
        y[0, :], y[1, :] = 1.0, 0.0
        result = _quiet_diag(y, s)
        for column, name in enumerate(LABELS):
            keep = ~np.isnan(y[:, column])
            assert result.ap[name] == pytest.approx(
                average_precision_score(y[keep, column], s[keep, column]), abs=1e-9
            )

    def test_nan_truth_is_dropped_even_where_the_mask_says_keep(self):
        rng = _rng(57)
        y, s = _truth(rng), _scores(rng)
        y[5, 0] = np.nan
        mask = np.ones_like(y, dtype=bool)
        assert _quiet_diag(y, s, mask).n_used["ACL"] == y.shape[0] - 1

    def test_a_nan_score_in_a_kept_cell_raises_rather_than_being_dropped(self):
        rng = _rng(58)
        y, s = _truth(rng), _scores(rng)
        s[3, 2] = np.nan
        with pytest.raises(ValueError, match="NaN prediction"):
            diagnostic_metrics(y, s)

    def test_a_nan_score_the_mask_already_drops_is_tolerated(self):
        rng = _rng(59)
        y, s = _truth(rng), _scores(rng)
        s[3, 2] = np.nan
        mask = np.ones_like(y, dtype=bool)
        mask[3, 2] = False
        assert np.isfinite(diagnostic_metrics(y, s, mask).ap_macro)


class TestDiagnosticInputValidation:
    def test_one_dimensional_truth_raises(self):
        with pytest.raises(ValueError, match="n_studies, n_labels"):
            diagnostic_metrics(np.zeros(12), np.zeros(12))

    def test_disagreeing_shapes_raise(self):
        with pytest.raises(ValueError, match="disagree"):
            diagnostic_metrics(np.zeros((4, N_LABELS)), np.zeros((5, N_LABELS)))

    def test_the_wrong_number_of_columns_raises(self):
        with pytest.raises(ValueError, match="expected 12"):
            diagnostic_metrics(np.zeros((4, 11)), np.zeros((4, 11)))

    def test_a_mask_of_the_wrong_shape_raises(self):
        y = _truth(_rng(60), n_rows=4)
        with pytest.raises(ValueError, match="does not match"):
            diagnostic_metrics(y, y, np.ones((4, 3), dtype=bool))

    def test_non_binary_truth_raises_rather_than_scoring_a_probability(self):
        rng = _rng(61)
        y, s = _truth(rng), _scores(rng)
        y[2, 4] = 0.5
        with pytest.raises(ValueError, match="outside"):
            diagnostic_metrics(y, s)

    def test_a_duplicate_label_name_raises_instead_of_shrinking_the_denominator(self):
        rng = _rng(62)
        y, s = _truth(rng), _scores(rng)
        names = list(LABELS)
        names[1] = names[0]
        with pytest.raises(ValueError, match="duplicate"):
            diagnostic_metrics(y, s, labels=names)


class TestCurvePoints:
    """roc_curve_points and pr_curve_points feed the W&B panels in tracking.py.

    Checked by integrating them back into the scalar they illustrate: the ROC
    curve's trapezoidal area must equal macro_roc_auc's per-label AUC, and the
    PR curve's step-function area must equal diagnostic_metrics' ap -- because a
    curve that disagrees with the number quoted next to it is worse than no
    curve at all.
    """

    def test_roc_curve_points_area_matches_the_per_label_auc(self):
        rng = _rng(63)
        for _ in range(15):
            y, s = _truth(rng, n_rows=30), _scores(rng, n_rows=30)
            for column, name in enumerate(LABELS):
                fpr, tpr = roc_curve_points(y[:, column], s[:, column])
                area = np.trapezoid(tpr, fpr)
                expected = roc_auc_score(y[:, column], s[:, column])
                assert area == pytest.approx(expected, abs=1e-9)

    def test_roc_curve_points_starts_and_ends_at_the_corners(self):
        rng = _rng(64)
        y, s = _truth(rng, n_rows=20), _scores(rng, n_rows=20)
        fpr, tpr = roc_curve_points(y[:, 0], s[:, 0])
        assert fpr[0] == 0.0 and tpr[0] == 0.0
        assert fpr[-1] == pytest.approx(1.0) and tpr[-1] == pytest.approx(1.0)

    def test_roc_curve_points_is_empty_when_a_class_is_absent(self):
        y = np.zeros(10)
        s = _scores(_rng(65), n_rows=10)[:, 0]
        fpr, tpr = roc_curve_points(y, s)
        assert fpr.size == 0 and tpr.size == 0

    def test_pr_curve_points_area_matches_average_precision(self):
        rng = _rng(66)
        for _ in range(15):
            y, s = _truth(rng, n_rows=30), _scores(rng, n_rows=30)
            for column, name in enumerate(LABELS):
                recall, precision = pr_curve_points(y[:, column], s[:, column])
                # Ascending recall, so the step-function integral uses the
                # precision reached *after* each step, not before it.
                area = float(np.sum(np.diff(recall) * precision[1:]))
                expected = average_precision_score(y[:, column], s[:, column])
                assert area == pytest.approx(expected, abs=1e-9)

    def test_pr_curve_points_starts_at_recall_zero_precision_one(self):
        rng = _rng(67)
        y, s = _truth(rng, n_rows=20), _scores(rng, n_rows=20)
        recall, precision = pr_curve_points(y[:, 0], s[:, 0])
        assert recall[0] == 0.0
        assert precision[0] == 1.0

    def test_pr_curve_points_is_empty_when_there_are_no_positives(self):
        y = np.zeros(10)
        s = _scores(_rng(68), n_rows=10)[:, 0]
        recall, precision = pr_curve_points(y, s)
        assert recall.size == 0 and precision.size == 0
