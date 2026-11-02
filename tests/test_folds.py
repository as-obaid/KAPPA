"""Folds are the one artifact whose bugs never show up until the private leaderboard.

Every test here is written against a specific way of being wrong:

* a fold with **zero positives** for one of the twelve labels makes that label's
  AUC ``nan``, and a macro over the surviving eleven is not the competition
  metric. Asserted per label, per fold, not claimed in prose.
* a **patient in two folds** inflates every cross-validation number. Asserted for
  every fold pair, and for the transitive grouping too: grouping on patient and
  scanner independently still leaks when one patient used two magnets.
* a **parquet fold file** is written successfully and then never tracked, because
  ``.gitignore`` globs ``*.parquet`` through the ``!data/folds/`` negation. The
  cheap wrong fix is to trust the negation; :func:`write_folds` must refuse.
* stratifying **one label at a time** looks fine on the common labels and starves
  the rare one. The synthetic tables below carry the measured prevalence range,
  starting at 15.5%, plus a language-correlated shift as large as the measured
  Effusion spread of 16% (Turkish) to 72% (Bulgarian).

The v2 tables add a frozen held-out test set, and it brings four more:

* a **group straddling dev and test** contaminates the test set. 52 groups link
  197 studies by identical report text, so this is not hypothetical.
* an **unequal split silently stops stratifying by language**. Before the fix,
  an 85/15 holdout put 0 of 220 cyrillic and 4 of 321 greek studies in
  the test set, because the raw label quota is never tied between an 0.85 bin and
  an 0.15 bin, so the size quota never got to break a tie. Every label gate
  passed on a test set that was 98% Latin.
* an **anchor in the test set** is a spent study in a set that is supposed to be
  unseen. All 58 have already been used to tune the label derivation.
* the **58 anchors and folds_v1.csv are both frozen artifacts**. A v2 schema that
  makes either unreadable makes the numbers taken against them unreproducible.

The label table is another agent's artifact, so nothing here reads it: the
structure being tested is the split, and synthetic tables with known structure
test it harder than real data would.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kappa.constants import ID_COLUMN, LABELS
from kappa.data.folds import (
    DEFAULT_GROUP_KEYS,
    FOLD_COLUMNS,
    FOLD_COLUMNS_V2,
    FoldReport,
    MASK_SUFFIX,
    N_FOLDS,
    TEST_FOLD,
    TEST_FRACTION,
    assign_folds,
    assign_holdout,
    build_folds,
    build_folds_v2,
    check_fold_columns,
    dev_mask,
    fold_report,
    group_components,
    holdout_report,
    positive_matrix,
    read_folds,
    SCANNER_GROUP_KEYS,
    SPLIT_VALUES,
    write_folds,
)

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))

# The script, for its anchor reader alone. anchors.csv is read as *data*, never
# imported, and it is not part of the published repository.
from build_folds import _anchor_ids  # noqa: E402

#: The private anchor list. It is derived from competition data and is not
#: published, so the test that reads it skips when it is absent.
_ANCHORS_CSV = _REPO / "src" / "kappa" / "labels" / "_recovered" / "anchors.csv"

#: Measured prevalence floor for the commonest-to-rarest sweep. MCL sits at
#: 15.5%, so nothing below that is realistic for this corpus -- but the tests
#: also probe a deliberately rarer label, because the gate has to hold there.
_PREVALENCE = np.linspace(0.155, 0.7, len(LABELS))

#: Measured language-correlated shift. Effusion runs 16% to 72% across
#: languages, which this reproduces in miniature so that a split blind to
#: language shows up as a widened prevalence spread.
_LANGUAGE_SHIFT = {"latin": 0.0, "turkish": -0.10, "greek": 0.10, "cyrillic": 0.25}

#: Measured bucket proportions over all 4,407 reports: latin 3,320, turkish 546,
#: greek 321, cyrillic 220.
_LANGUAGE_WEIGHTS = (3320 / 4407, 546 / 4407, 321 / 4407, 220 / 4407)


def _study_table(
    n: int = 800,
    *,
    n_scanners: int = 40,
    repeat_patients: int = 0,
    bridge_scanners: bool = False,
    mask_rate: float = 0.1,
    rare_label: str | None = None,
    rare_count: int = 7,
    seed: int = 11,
) -> pd.DataFrame:
    """One row per study: twelve masked labels, a language bucket, two group keys.

    ``repeat_patients`` makes that many patients own two studies, which is the
    structure the grouping has to survive. Both studies land on the same
    scanner, as a returning patient at one site would; ``bridge_scanners`` puts
    them on different ones instead, which is the case that collapses a
    patient-and-scanner grouping into a single component. ``rare_label`` forces
    one label down to exactly ``rare_count`` positives, which is where a
    one-label-at-a-time split leaves a fold empty.
    """
    rng = np.random.default_rng(seed)
    buckets = list(_LANGUAGE_SHIFT)
    language = rng.choice(buckets, size=n, p=_LANGUAGE_WEIGHTS)
    data: dict[str, object] = {ID_COLUMN: [f"1.2.826.0.1.{index:07d}" for index in range(n)]}
    for column, label in enumerate(LABELS):
        shift = np.array([_LANGUAGE_SHIFT[bucket] for bucket in language])
        probability = np.clip(_PREVALENCE[column] + shift, 0.02, 0.95)
        data[label] = (rng.random(n) < probability).astype(float)
        data[f"{label}{MASK_SUFFIX}"] = rng.random(n) >= mask_rate
    if rare_label is not None:
        values = np.zeros(n)
        values[rng.choice(n, size=rare_count, replace=False)] = 1.0
        data[rare_label] = values
        data[f"{rare_label}{MASK_SUFFIX}"] = np.ones(n, dtype=bool)
    data["language"] = language
    patients = [f"pat-{index:05d}" for index in range(n)]
    scanners = [f"fp-{rng.integers(0, n_scanners):03d}" for _ in range(n)]
    for index in range(repeat_patients):
        patients[n - 1 - index] = patients[index]
        if not bridge_scanners:
            scanners[n - 1 - index] = scanners[index]
    data["patient_id"] = patients
    data["scanner_fingerprint"] = scanners
    # No two synthetic studies share report text, so every report_group is a
    # singleton and the grouping under test is still driven by patient_id --
    # which keeps these cases measuring what they were written to measure.
    data["report_group"] = [f"rg-{index:05d}" for index in range(n)]
    return pd.DataFrame(data)


def _reports(folds: pd.DataFrame, table: pd.DataFrame):
    """Both arms' reports, each gated on the grouping it was actually built with."""
    return (
        fold_report(folds, table, fold_column="fold", group_column="patient_group",
                    extra_group_columns=("scanner_group",)),
        fold_report(folds, table, fold_column="fold_scanner", group_column="scanner_group",
                    extra_group_columns=("patient_group",)),
    )


class TestGroupComponents:
    def test_one_key_gives_one_component_per_distinct_value(self):
        table = pd.DataFrame({"patient_id": ["a", "b", "a", "c"]})
        components = group_components(table, ["patient_id"])
        assert components[0] == components[2]
        assert len({int(value) for value in components}) == 3

    def test_two_keys_link_transitively(self):
        # A patient scanned on two magnets bridges those magnets. Checking the
        # two keys independently would still let that patient sit in two folds,
        # and a per-key overlap check would report zero leakage.
        table = pd.DataFrame(
            {
                "patient_id": ["p1", "p1", "p2", "p3"],
                "scanner_fingerprint": ["s1", "s2", "s2", "s3"],
            }
        )
        components = group_components(table, ["patient_id", "scanner_fingerprint"])
        assert components[0] == components[1] == components[2]
        assert components[3] != components[0]

    def test_blank_keys_do_not_merge_into_one_giant_group(self):
        # Merging every blank PatientID would build a single component holding
        # most of the corpus, make folds impossible to balance, and still pass
        # an overlap check -- one group in one fold overlaps nothing.
        table = pd.DataFrame({"patient_id": ["", "", None, "nan", "p1", "p1"]})
        components = group_components(table, ["patient_id"])
        assert len({int(value) for value in components[:4]}) == 4
        assert components[4] == components[5]

    def test_a_missing_grouping_column_raises(self):
        with pytest.raises(KeyError, match="not in the table"):
            group_components(pd.DataFrame({"a": [1]}), ["patient_id"])


class TestPositiveMatrix:
    def test_a_masked_cell_is_neither_a_positive_nor_a_negative(self):
        # Counting masked cells as negatives makes a label look commoner than it
        # is and lets a fold pass the "has a positive" gate on rows that never
        # reach the loss.
        table = pd.DataFrame({label: [1.0, 1.0, 0.0] for label in LABELS})
        for label in LABELS:
            table[f"{label}{MASK_SUFFIX}"] = [True, False, True]
        positives, usable = positive_matrix(table)
        assert positives[:, 0].tolist() == [True, False, False]
        assert usable[:, 0].tolist() == [True, False, True]

    def test_a_table_with_no_mask_columns_is_fully_usable(self):
        table = pd.DataFrame({label: [1.0, 0.0] for label in LABELS})
        _, usable = positive_matrix(table)
        assert usable.all()

    def test_a_nan_label_is_unusable_even_with_the_mask_set(self):
        # 98.7% of raw label cells are NaN. A NaN that counts as usable is how
        # NaN reaches the loss.
        table = pd.DataFrame({label: [float("nan"), 1.0] for label in LABELS})
        for label in LABELS:
            table[f"{label}{MASK_SUFFIX}"] = [True, True]
        positives, usable = positive_matrix(table)
        assert usable[0, 0] == np.False_
        assert positives[0, 0] == np.False_

    def test_a_missing_label_column_names_the_label(self):
        with pytest.raises(KeyError, match="ACL"):
            positive_matrix(pd.DataFrame({"MCL": [1.0]}))


class TestTheGate:
    def test_every_label_has_a_positive_in_every_fold(self):
        table = _study_table()
        folds = build_folds(table)
        for report in _reports(folds, table):
            assert report.empty_labels == (), report.table()
            for label in LABELS:
                assert min(report.positives[label]) >= 1

    def test_the_gate_holds_when_one_label_has_only_seven_positives(self):
        # Seven positives over five folds is the edge: one per fold with two to
        # spare. A split that stratifies on the common labels first strands them.
        table = _study_table(rare_label="ACL", rare_count=7)
        folds = build_folds(table)
        report, _ = _reports(folds, table)
        assert report.positives["ACL"] == (2, 2, 1, 1, 1) or min(report.positives["ACL"]) >= 1
        assert sum(report.positives["ACL"]) == 7
        report.raise_for_gate()

    def test_no_patient_appears_in_two_folds_for_any_fold_pair(self):
        table = _study_table(repeat_patients=120)
        folds = build_folds(table)
        merged = folds.merge(table, on=ID_COLUMN)
        for left, right in itertools.combinations(range(N_FOLDS), 2):
            left_patients = set(merged.loc[merged["fold"] == left, "patient_id"])
            right_patients = set(merged.loc[merged["fold"] == right, "patient_id"])
            assert not (left_patients & right_patients), f"folds {left} and {right} share a patient"

    def test_no_scanner_appears_in_two_folds_in_the_scanner_arm(self):
        table = _study_table()
        folds = build_folds(table)
        merged = folds.merge(table, on=ID_COLUMN)
        for left, right in itertools.combinations(range(N_FOLDS), 2):
            left_scanners = set(merged.loc[merged["fold_scanner"] == left, "scanner_fingerprint"])
            right_scanners = set(merged.loc[merged["fold_scanner"] == right, "scanner_fingerprint"])
            assert not (left_scanners & right_scanners)

    def test_the_patient_arm_deliberately_spreads_scanners_across_folds(self):
        # Not a bug: it is what makes `fold` the random-fold arm. Gating it on
        # scanner overlap would reject the split it exists to provide.
        table = _study_table()
        folds = build_folds(table)
        report, _ = _reports(folds, table)
        assert report.overlaps["patient_group"] == 0
        assert report.overlaps["scanner_group"] > 0
        assert report.ok

    def test_the_report_names_the_label_and_fold_when_a_fold_is_empty(self):
        table = _study_table(n=60)
        table["ACL"] = 0.0
        folds = build_folds(table)
        report, _ = _reports(folds, table)
        assert not report.ok
        assert {label for label, _ in report.empty_labels} == {"ACL"}
        with pytest.raises(ValueError, match="zero usable positives"):
            report.raise_for_gate()

    def test_a_leaking_fold_column_fails_the_gate(self):
        # The cheap wrong fix: assign folds by row, ignoring the grouping.
        table = _study_table(repeat_patients=200)
        folds = build_folds(table)
        folds["fold"] = np.arange(len(folds)) % N_FOLDS
        report, _ = _reports(folds, table)
        assert report.overlaps["patient_group"] > 0
        with pytest.raises(ValueError, match="leakage"):
            report.raise_for_gate()


class TestPrevalenceSpread:
    def test_the_per_label_prevalence_spread_is_reported_and_small(self):
        # Printed, not captured-and-dropped: run `pytest -s -k prevalence_spread`
        # to read the per-fold table. Plain print rather than capsys.disabled()
        # so a default run of the whole suite stays quiet for everyone else.
        table = _study_table()
        folds = build_folds(table)
        patient_arm, scanner_arm = _reports(folds, table)
        print()
        print(patient_arm.table())
        print()
        print(scanner_arm.table())
        spread = patient_arm.spread
        assert set(spread) == set(LABELS)
        # The rare labels are the ones a split can starve, and they are held to
        # within two studies per fold. The four rarest here are unambiguously
        # minority-positive, so their positive count is what the stratifier is
        # steering directly.
        rarest = sorted(LABELS, key=lambda label: sum(patient_arm.positives[label]))[:4]
        for label in rarest:
            counts = patient_arm.positives[label]
            assert max(counts) - min(counts) <= 2, (label, counts)
        # Prevalence spread is looser than that, and it is not a stratification
        # failure: the denominator is the number of *usable* cells and the mask
        # is per cell, so a fold's usable count moves even when its balanced
        # count does not. Measured up to 0.13 at a 10% mask rate over three
        # seeds; 0.15 is the ceiling this is asserted under.
        assert max(spread.values()) < 0.15, spread

    def test_language_buckets_are_split_across_folds_not_concentrated(self):
        table = _study_table()
        folds = build_folds(table)
        report, _ = _reports(folds, table)
        for bucket, counts in report.languages.items():
            assert min(counts) > 0, f"{bucket} is missing from a fold"
            # The smallest measured bucket is cyrillic at 220 of 4,407, so an
            # even five-way split is ~44. Allowing a factor of two catches
            # concentration without failing on integer rounding.
            assert max(counts) <= 2 * max(1, min(counts)), f"{bucket} {counts}"


class TestDeterminism:
    def test_the_same_seed_gives_the_same_split(self):
        table = _study_table()
        first = build_folds(table, seed=3)
        second = build_folds(table, seed=3)
        pd.testing.assert_frame_equal(first, second)

    def test_a_different_seed_gives_a_different_split(self):
        # If it did not, the seed would be decoration and a fold table would not
        # need it recorded to be reproducible.
        table = _study_table()
        assert not build_folds(table, seed=0)["fold"].equals(build_folds(table, seed=1)["fold"])

    def test_row_order_in_the_input_does_not_change_the_split(self):
        # build_folds sorts by study UID first, so a manifest read back in a
        # different order still produces the same folds.
        table = _study_table()
        shuffled = table.sample(frac=1.0, random_state=5).reset_index(drop=True)
        pd.testing.assert_frame_equal(build_folds(table), build_folds(shuffled))


class TestConfigurableGrouping:
    def test_the_grouping_key_can_be_changed_without_a_rewrite(self):
        # Q2 is unresolved, so which column identifies a patient may change.
        table = _study_table().rename(columns={"patient_id": "subject"})
        folds = build_folds(table, group_keys=("subject",),
                            scanner_group_keys=("subject", "scanner_fingerprint"))
        assert tuple(folds.columns) == FOLD_COLUMNS

    def test_stratification_can_be_turned_off_to_measure_what_it_buys(self):
        table = _study_table()
        folds = build_folds(table, strata_key=None)
        assert set(folds["language"]) == {"all"}

    def test_a_missing_stratification_column_raises(self):
        table = _study_table().drop(columns=["language"])
        with pytest.raises(KeyError, match="language"):
            build_folds(table)

    def test_a_duplicated_study_uid_raises_before_a_study_lands_in_two_folds(self):
        table = _study_table(n=40)
        table.loc[1, ID_COLUMN] = table.loc[0, ID_COLUMN]
        with pytest.raises(ValueError, match="not unique"):
            build_folds(table)

    def test_a_grouping_that_collapses_into_one_component_warns(self):
        # Union over patient and scanner collapses when patients bridge
        # scanners. Scanner-grouped folds are then infeasible, and silence here
        # would hand back five folds where one holds everything.
        table = _study_table(n=200, n_scanners=5, repeat_patients=80,
                             bridge_scanners=True)
        with pytest.warns(UserWarning, match="cannot be balanced"):
            assign_folds(table, group_keys=("patient_id", "scanner_fingerprint"))

    def test_fewer_studies_than_folds_raises(self):
        table = _study_table(n=40).head(3)
        with pytest.raises(ValueError, match="cannot cut 3 studies"):
            build_folds(table)


class TestWriteRead:
    def test_the_column_list_is_exactly_the_contract(self):
        folds = build_folds(_study_table(n=200))
        assert tuple(folds.columns) == FOLD_COLUMNS
        assert FOLD_COLUMNS[0] == ID_COLUMN

    def test_writing_a_parquet_fold_table_is_refused(self, tmp_path):
        # .gitignore globs *.parquet even under data/folds/, so the file would be
        # written, look saved, and never be tracked. Verified out of band:
        # `git check-ignore -v data/folds/folds_v1.parquet` matches
        # `.gitignore:25:*.parquet`, while the .csv exits 1.
        folds = build_folds(_study_table(n=200))
        with pytest.raises(ValueError, match="must be .csv"):
            write_folds(folds, tmp_path / "folds_v1.parquet")

    def test_a_round_trip_keeps_the_study_uid_a_string(self, tmp_path):
        # StudyInstanceUID is a long dotted numeric. Read back as a float it
        # silently stops joining against anything.
        folds = build_folds(_study_table(n=200))
        path = write_folds(folds, tmp_path / "folds_v1.csv")
        back = read_folds(path)
        assert back[ID_COLUMN].dtype == object
        assert list(back[ID_COLUMN]) == list(folds[ID_COLUMN])

    def test_reading_a_fold_table_with_the_wrong_columns_raises(self, tmp_path):
        path = tmp_path / "folds_v1.csv"
        pd.DataFrame({ID_COLUMN: ["a"], "fold": [0]}).to_csv(path, index=False)
        with pytest.raises(ValueError, match="contract requires"):
            read_folds(path)

    def test_writing_a_reordered_fold_table_raises(self, tmp_path):
        folds = build_folds(_study_table(n=200))[list(reversed(FOLD_COLUMNS))]
        with pytest.raises(ValueError, match="contract requires"):
            write_folds(folds, tmp_path / "folds_v1.csv")


class TestFoldReport:
    def test_a_fold_table_that_does_not_cover_the_labels_is_rejected(self):
        # A silent inner join would validate a fold table against a subset of
        # itself and report a pass.
        table = _study_table(n=200)
        folds = build_folds(table)
        with pytest.raises(ValueError, match="have no label row"):
            fold_report(folds, table.head(100))

    def test_prevalence_is_positives_over_usable_cells(self):
        table = _study_table(n=200)
        folds = build_folds(table)
        report, _ = _reports(folds, table)
        for label in LABELS:
            for fold in range(report.n_folds):
                expected = report.positives[label][fold] / report.usable[label][fold]
                assert report.prevalence[label][fold] == pytest.approx(expected)

    def test_the_fold_sizes_sum_to_the_number_of_studies(self):
        table = _study_table(n=200)
        folds = build_folds(table)
        report, _ = _reports(folds, table)
        assert sum(report.sizes) == len(table)
        assert report.n_folds == N_FOLDS


class TestQ2ClosedAgainstPatientId:
    """Folds may not be grouped on ``patient_id``, because it groups nothing.

    Measured over the full manifest, all 4,407 studies: 4,407 studies carry a
    PatientID and all 4,407 are **distinct**. Of the 52
    identical-report groups, **0 share a PatientID and all 52 span several** -- and
    5 of those are the discriminating kind, a report over 400 characters appearing
    in exactly two studies. Two studies with the same long report under two
    different PatientIDs is proof the IDs are re-keyed per study.

    The trap is that this looks like good news. "All patient IDs distinct" reads as
    "no patient recurs", so grouping by patient looks satisfied when in fact the
    key is inert: every group is a singleton, the constraint binds nothing, and
    ``fold`` is a random split wearing a grouped split's name. Every CV number
    would be optimistic and the private leaderboard in late October is where you
    would find out.
    """

    def test_report_group_is_a_default_grouping_key(self):
        assert "report_group" in DEFAULT_GROUP_KEYS
        assert "report_group" in SCANNER_GROUP_KEYS

    def test_studies_sharing_report_text_never_split_across_folds(self):
        # The link patient_id fails to make. 197 studies sit in 52 such groups in
        # the real corpus, and splitting one puts the same report on both sides of
        # a fold boundary.
        table = _study_table(n=180, n_scanners=6)
        shared = table["report_group"].to_numpy().copy()
        for index in range(0, 40, 2):
            shared[index + 1] = shared[index]
        table["report_group"] = shared

        folds = build_folds(table, n_folds=5, strata_key=None, seed=0)
        merged = folds.merge(table[[ID_COLUMN, "report_group"]], on=ID_COLUMN)
        for column in ("fold", "fold_scanner"):
            spread = merged.groupby("report_group")[column].nunique()
            offenders = spread[spread > 1]
            assert offenders.empty, (
                f"{len(offenders)} report group(s) span two folds in {column!r}"
            )

    def test_grouping_on_patient_id_alone_would_have_split_them(self):
        # The control that makes the test above mean something: with the real
        # corpus's all-distinct PatientIDs, the old default cannot hold the pairs
        # together, so this is a leak the previous configuration allowed.
        table = _study_table(n=180, n_scanners=6)
        shared = table["report_group"].to_numpy().copy()
        for index in range(0, 40, 2):
            shared[index + 1] = shared[index]
        table["report_group"] = shared
        table["patient_id"] = [f"unique-{i:05d}" for i in range(len(table))]

        folds = build_folds(
            table, n_folds=5, strata_key=None, seed=0,
            group_keys=("patient_id",), scanner_group_keys=("patient_id",),
        )
        merged = folds.merge(table[[ID_COLUMN, "report_group"]], on=ID_COLUMN)
        spread = merged.groupby("report_group")["fold"].nunique()
        assert (spread > 1).any(), (
            "patient_id-only grouping kept every duplicated report together, so this "
            "control proves nothing -- check the fixture"
        )

    def test_a_missing_report_group_column_is_refused_not_ignored(self):
        # Silently dropping an absent grouping key would reinstate the leak while
        # every fold gate still passed.
        table = _study_table(n=60).drop(columns=["report_group"])
        with pytest.raises(KeyError, match="report_group"):
            build_folds(table, n_folds=5, strata_key=None, seed=0)


def _link_reports(table: pd.DataFrame, n_pairs: int) -> pd.DataFrame:
    """Give ``n_pairs`` of adjacent studies identical report text, on one scanner.

    52 such groups link 197 studies in the real corpus, and that is the link
    ``patient_id`` fails to make. Both studies of a pair land on the same scanner,
    as a returning patient at one site would: putting them on two scanners instead
    bridges those scanners, collapses the scanner grouping into one component, and
    is a different failure with its own test.
    """
    linked = table.copy()
    report = linked["report_group"].to_numpy().copy()
    scanner = linked["scanner_fingerprint"].to_numpy().copy()
    for index in range(0, 2 * n_pairs, 2):
        report[index + 1] = report[index]
        scanner[index + 1] = scanner[index]
    linked["report_group"] = report
    linked["scanner_fingerprint"] = scanner
    return linked


def _v2_reports(folds: pd.DataFrame, table: pd.DataFrame):
    """The three v2 reports: dev-against-test, then both fold arms over dev alone.

    The fold arms are measured over ``dev`` and not over the whole table on
    purpose. A scanner legitimately spans dev and test -- the test set is
    scanner-shared -- so gating ``fold_scanner`` over every row would fail the
    split this function exists to check.
    """
    dev = folds.loc[dev_mask(folds)].reset_index(drop=True)
    return (
        holdout_report(folds, table),
        fold_report(dev, table, fold_column="fold", group_column="report_group",
                    extra_group_columns=("scanner_group",)),
        fold_report(dev, table, fold_column="fold_scanner", group_column="scanner_group",
                    extra_group_columns=("report_group",)),
    )


class TestDevMask:
    """``train = uids[fold != k]`` absorbs the whole test set and never says so."""

    def _v2(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                ID_COLUMN: ["a", "b", "c", "d"],
                "split": ["dev", "dev", "test", "test"],
                "fold": [0, 1, TEST_FOLD, TEST_FOLD],
                "fold_scanner": [0, 1, TEST_FOLD, TEST_FOLD],
                "report_group": [0, 1, 2, 3],
                "scanner_group": [0, 1, 2, 3],
                "language": ["latin"] * 4,
            },
            columns=list(FOLD_COLUMNS_V2),
        )

    def test_a_negative_fold_index_does_not_participate(self):
        assert dev_mask(self._v2()).tolist() == [True, True, False, False]

    def test_a_test_split_does_not_participate_even_with_a_real_fold_index(self):
        # The pair that a hand edit produces: someone sets split=test and leaves
        # the fold index alone. Either condition alone has to be disqualifying.
        folds = self._v2()
        folds.loc[2, ["fold", "fold_scanner"]] = [1, 1]
        assert dev_mask(folds).tolist() == [True, True, False, False]

    def test_a_dev_split_with_a_negative_fold_index_does_not_participate(self):
        folds = self._v2()
        folds.loc[1, "split"] = "dev"
        folds.loc[1, "fold"] = TEST_FOLD
        assert dev_mask(folds).tolist() == [True, False, False, False]

    def test_every_row_of_a_v1_table_participates(self):
        # v1 has no `split` column and no negative index, and it must keep
        # behaving exactly as it did: five folds over every study.
        folds = build_folds(_study_table(n=200))
        assert dev_mask(folds).all()

    def test_the_mask_is_taken_per_column_so_the_scanner_arm_is_checked_too(self):
        folds = self._v2()
        folds.loc[1, "fold_scanner"] = TEST_FOLD
        assert dev_mask(folds, column="fold").tolist() == [True, True, False, False]
        assert dev_mask(folds, column="fold_scanner").tolist() == [True, False, False, False]

    def test_an_absent_fold_column_is_refused_rather_than_ignored(self):
        with pytest.raises(KeyError, match="fold_patient"):
            dev_mask(self._v2(), column="fold_patient")


class TestTheFrozenHoldout:
    def test_the_test_set_is_the_requested_share_of_the_corpus(self):
        table = _study_table(n=1600)
        folds = build_folds_v2(table)
        n_test = int((folds["split"] == "test").sum())
        # Grouped and stratified, so the size lands near the target rather than
        # on it. 15% of 4,407 is 661; a 10% relative tolerance is the honest claim.
        assert n_test == pytest.approx(TEST_FRACTION * len(table), rel=0.1)
        assert set(folds["split"]) == set(SPLIT_VALUES)

    def test_no_group_spans_dev_and_test(self):
        # The contamination gate, and the only one checkable from the written file
        # alone. 52 groups link 197 studies by identical report text in the real
        # corpus; one straddling the boundary means the model has read the report
        # it is being scored on.
        table = _link_reports(_study_table(n=1200, n_scanners=40), n_pairs=100)
        folds = build_folds_v2(table, seed=0)
        merged = folds.merge(table[[ID_COLUMN, "report_group"]], on=ID_COLUMN,
                             suffixes=("", "_hash"))
        spread = merged.groupby("report_group_hash")["split"].nunique()
        assert spread.max() == 1, f"{int((spread > 1).sum())} groups span dev and test"
        assert holdout_report(folds, table).overlaps["report_group"] == 0

    def test_every_label_has_a_usable_positive_in_the_test_set(self):
        table = _study_table(n=1600)
        folds = build_folds_v2(table)
        holdout, patient_arm, scanner_arm = _v2_reports(folds, table)
        for report in (holdout, patient_arm, scanner_arm):
            assert report.empty_labels == (), report.table()
            report.raise_for_gate()
        for label in LABELS:
            assert holdout.positives[label][1] >= 1, label

    def test_the_gate_holds_over_six_bins_when_one_label_is_nearly_starved(self):
        # A test set plus five folds is six bins, not five, so a rare label has
        # one more place to run out. This is the shape the real Fracture column
        # sits in, scaled down.
        table = _study_table(n=1200, rare_label="ACL", rare_count=18)
        folds = build_folds_v2(table)
        holdout, patient_arm, _ = _v2_reports(folds, table)
        assert sum(holdout.positives["ACL"]) == 18
        assert holdout.positives["ACL"][1] >= 1
        assert min(patient_arm.positives["ACL"]) >= 1, patient_arm.positives["ACL"]

    def test_every_language_bucket_reaches_the_test_set(self):
        # Before the share-relative quota fix, 0 of 220 cyrillic and 4 of
        # 321 greek studies reached an 85/15 test set, and every label gate passed
        # on a test set that was 98% Latin. Language implies institution implies
        # scanner, so that test set measures a different corpus.
        table = _study_table(n=2400)
        folds = build_folds_v2(table)
        holdout = holdout_report(folds, table)
        for bucket, (in_dev, in_test) in holdout.languages.items():
            assert in_test >= 1, f"{bucket} is absent from the test set"
            share = in_test / (in_dev + in_test)
            assert abs(share - TEST_FRACTION) < 0.06, f"{bucket} {share:.3f}"

    def test_a_test_row_is_in_no_fold_and_a_dev_row_is_in_one(self):
        table = _study_table(n=800)
        folds = build_folds_v2(table)
        held = folds["split"] == "test"
        for column in ("fold", "fold_scanner"):
            assert (folds.loc[held, column] == TEST_FOLD).all()
            assert (folds.loc[~held, column] >= 0).all()
            assert (folds.loc[~held, column] < N_FOLDS).all()

    def test_the_folds_partition_the_dev_pool_and_nothing_more(self):
        table = _study_table(n=1200)
        folds = build_folds_v2(table)
        _, patient_arm, scanner_arm = _v2_reports(folds, table)
        n_dev = int(dev_mask(folds).sum())
        assert sum(patient_arm.sizes) == n_dev
        assert sum(scanner_arm.sizes) == n_dev
        assert n_dev < len(table)

    def test_a_scanner_spanning_dev_and_test_is_the_design_not_a_leak(self):
        # The test set is scanner-SHARED. The hidden Kaggle test set is drawn from
        # the same 19 sites, so a scanner-disjoint local test would be pessimistic
        # and would answer a question nobody asked. Gating it would reject the
        # split on the grounds of the property it was built to have.
        table = _study_table(n=1200, n_scanners=20)
        folds = build_folds_v2(table)
        holdout, _, scanner_arm = _v2_reports(folds, table)
        assert holdout.overlaps["scanner_group"] > 0
        assert holdout.gate_columns == ("report_group",)
        assert holdout.ok
        # ...and inside the dev pool the scanner arm still keeps each scanner whole.
        assert scanner_arm.overlaps["scanner_group"] == 0
        assert scanner_arm.ok

    def test_the_same_seed_gives_the_same_holdout_and_a_different_one_does_not(self):
        table = _study_table(n=800)
        pd.testing.assert_frame_equal(build_folds_v2(table, seed=3),
                                      build_folds_v2(table, seed=3))
        assert not build_folds_v2(table, seed=0)["split"].equals(
            build_folds_v2(table, seed=1)["split"]
        )

    def test_row_order_in_the_input_does_not_move_the_boundary(self):
        table = _study_table(n=600)
        shuffled = table.sample(frac=1.0, random_state=5).reset_index(drop=True)
        pd.testing.assert_frame_equal(build_folds_v2(table), build_folds_v2(shuffled))

    def test_the_component_ids_cover_the_whole_table_not_one_side(self):
        # They are the only way a reader with no access to the corpus can verify
        # from the file that no group spans dev and test.
        table = _study_table(n=400)
        folds = build_folds_v2(table)
        ordered = table.sort_values(ID_COLUMN).reset_index(drop=True)
        expected = group_components(ordered, DEFAULT_GROUP_KEYS)
        assert folds["report_group"].tolist() == expected.tolist()

    def test_a_test_fraction_outside_zero_to_one_is_refused(self):
        table = _study_table(n=200)
        for bad in (0.0, 1.0, -0.1, 1.5):
            with pytest.raises(ValueError, match="test_frac must be strictly between"):
                build_folds_v2(table, test_frac=bad)

    def test_a_holdout_with_nothing_on_one_side_is_refused(self):
        # Reachable from a small table and a small fraction, and it raises nowhere
        # downstream: the folds still build and every gate still passes on a
        # "held-out" table that holds nothing out.
        table = _study_table(n=40)
        for label in LABELS:
            table[label] = 0.0
        # The size-only fallback runs when no label has a positive left to place,
        # and it fills the big bin first: at 0.1% of 40 studies the small bin never
        # comes up. The balance warning is the same one a collapsed grouping raises.
        with pytest.warns(UserWarning, match="cannot be balanced"):
            with pytest.raises(ValueError, match="studies on both sides"):
                build_folds_v2(table, test_frac=0.001)

    def test_a_duplicated_study_uid_is_refused_before_it_lands_on_both_sides(self):
        table = _study_table(n=60)
        table.loc[1, ID_COLUMN] = table.loc[0, ID_COLUMN]
        with pytest.raises(ValueError, match="not unique"):
            build_folds_v2(table)

    def test_assign_holdout_returns_a_boolean_mask_parallel_to_the_table(self):
        table = _study_table(n=300)
        held = assign_holdout(table)
        assert held.dtype == bool
        assert held.shape == (len(table),)
        assert 0 < int(held.sum()) < len(table)


class TestAnchorsStayInDev:
    """The 58 anchors are spent. A test set containing them is not clean."""

    def test_no_pinned_study_lands_in_the_test_set(self):
        table = _study_table(n=1200)
        anchors = tuple(table[ID_COLUMN].iloc[:58])
        folds = build_folds_v2(table, keep_ids=anchors)
        held = set(folds.loc[folds["split"] == "test", ID_COLUMN])
        assert not held & set(anchors)

    def test_pinning_an_anchor_pins_its_whole_group_not_just_its_row(self):
        # Pinning the row and releasing its group would put the same report text
        # on both sides of the boundary -- which is the contamination the grouping
        # exists to prevent, reintroduced by the anchor rule.
        table = _study_table(n=800)
        shared = table["report_group"].to_numpy().copy()
        shared[1] = shared[0]
        table["report_group"] = shared
        folds = build_folds_v2(table, keep_ids=(table[ID_COLUMN].iloc[0],))
        both = folds.loc[folds[ID_COLUMN].isin(table[ID_COLUMN].iloc[:2]), "split"]
        assert set(both) == {"dev"}

    def test_the_absolute_size_of_the_test_set_survives_pinning(self):
        # The share offered to the eligible pool is scaled up, so test_frac keeps
        # describing a fraction of the whole table rather than of what is left.
        table = _study_table(n=1600)
        anchors = tuple(table[ID_COLUMN].iloc[:200])
        folds = build_folds_v2(table, keep_ids=anchors)
        n_test = int((folds["split"] == "test").sum())
        assert n_test == pytest.approx(TEST_FRACTION * len(table), rel=0.12)

    def test_keep_ids_that_match_nothing_is_refused_not_silently_ignored(self):
        # The failure this catches: anchors.csv joined against a table whose ids
        # are formatted differently. The pin becomes a no-op, no error is raised,
        # and spent studies go into the frozen test set.
        table = _study_table(n=200)
        with pytest.raises(ValueError, match="none of the .* keep_ids appear"):
            build_folds_v2(table, keep_ids=("not-a-study-uid",))

    def test_keep_ids_only_partly_present_warns_with_both_counts(self):
        table = _study_table(n=400)
        anchors = (*table[ID_COLUMN].iloc[:10], "missing-1", "missing-2")
        with pytest.warns(UserWarning, match="2 of 12 keep_ids are not"):
            build_folds_v2(table, keep_ids=anchors)

    def test_pinning_more_than_the_pool_can_spare_is_refused(self):
        table = _study_table(n=200)
        with pytest.raises(ValueError, match="cannot be drawn from"):
            build_folds_v2(table, keep_ids=tuple(table[ID_COLUMN].iloc[:190]))

    @pytest.mark.skipif(
        not _ANCHORS_CSV.is_file(),
        reason="private anchor list (src/kappa/labels/_recovered/anchors.csv) is not published",
    )
    def test_the_quarantined_anchor_file_still_lists_exactly_58_studies(self):
        # Read as data, never imported. If this count moves, the pin silently covers a
        # different set.
        anchors = _anchor_ids(_ANCHORS_CSV)
        assert len(anchors) == 58
        assert len(set(anchors)) == 58
        assert all(uid.startswith("1.2.826.0.1.") for uid in anchors)


class TestHoldoutReport:
    def test_it_names_the_two_bins_rather_than_calling_them_folds(self):
        # A two-column table headed `fold0  fold1` is read as two folds by whoever
        # pastes it into a log, and the column under fold1 is the one nobody may
        # look at yet.
        table = _study_table(n=400)
        report = holdout_report(build_folds_v2(table), table)
        assert report.fold_column == "split"
        assert report.names == SPLIT_VALUES
        text = report.table()
        assert "  dev" in text and " test" in text and "fold0" not in text

    def test_a_default_fold_report_still_says_fold0(self):
        table = _study_table(n=200)
        report, _ = _reports(build_folds(table), table)
        assert report.names == tuple(f"fold{index}" for index in range(N_FOLDS))
        assert "fold0" in report.table()

    def test_a_table_with_no_split_column_is_refused(self):
        table = _study_table(n=200)
        with pytest.raises(KeyError, match="no 'split' column"):
            holdout_report(build_folds(table), table)

    def test_a_split_column_holding_anything_else_is_refused(self):
        table = _study_table(n=200)
        folds = build_folds_v2(table)
        folds.loc[0, "split"] = "val"
        with pytest.raises(ValueError, match="contract allows"):
            holdout_report(folds, table)

    def test_an_all_dev_table_is_refused_rather_than_reported_as_a_pass(self):
        # A one-bin report never looks at the test side, so the gate would pass on
        # a holdout that does not exist.
        table = _study_table(n=200)
        folds = build_folds_v2(table)
        folds["split"] = "dev"
        with pytest.raises(ValueError, match="no \\['test'\\] rows"):
            holdout_report(folds, table)

    def test_a_group_straddling_dev_and_test_fails_the_gate_by_name(self):
        # The cheap wrong fix: draw the holdout by row instead of by group.
        table = _link_reports(_study_table(n=600, n_scanners=30), n_pairs=60)
        folds = build_folds_v2(table)
        folds["split"] = np.where(np.arange(len(folds)) % 7 == 0, "test", "dev")
        report = holdout_report(folds, table)
        assert report.overlaps["report_group"] > 0
        with pytest.raises(ValueError, match="leakage"):
            report.raise_for_gate()


class TestUnequalRatios:
    def test_a_ratio_per_fold_is_required(self):
        table = _study_table(n=200)
        with pytest.raises(ValueError, match="one entry per fold"):
            assign_folds(table, n_folds=5, ratios=(0.5, 0.5))

    def test_a_zero_or_negative_ratio_is_refused(self):
        table = _study_table(n=200)
        for bad in ((0.0, 1.0), (-0.5, 1.5), (float("nan"), 1.0)):
            with pytest.raises(ValueError, match="finite and positive"):
                assign_folds(table, n_folds=2, ratios=bad)

    def test_ratios_are_normalised_so_they_need_not_sum_to_one(self):
        table = _study_table(n=400)
        first = assign_folds(table, n_folds=2, ratios=(0.85, 0.15))
        second = assign_folds(table, n_folds=2, ratios=(85.0, 15.0))
        assert first.tolist() == second.tolist()

    def test_equal_ratios_reproduce_the_default_exactly(self):
        # folds_v1.csv is frozen, so the equal-share path must stay on the exact
        # expression that produced it: passing ratios explicitly may not perturb it.
        table = _study_table(n=800, repeat_patients=60)
        default = assign_folds(table, n_folds=N_FOLDS)
        explicit = assign_folds(table, n_folds=N_FOLDS, ratios=(0.2,) * N_FOLDS)
        assert default.tolist() == explicit.tolist()

    def test_an_unequal_split_lands_near_its_requested_shares(self):
        table = _study_table(n=1200)
        bins = assign_folds(table, n_folds=2, ratios=(0.7, 0.3))
        share = float((bins == 1).mean())
        assert abs(share - 0.3) < 0.05, share


class TestSchemaV2:
    def test_the_v2_column_list_is_exactly_the_contract(self):
        folds = build_folds_v2(_study_table(n=300))
        assert tuple(folds.columns) == FOLD_COLUMNS_V2
        assert FOLD_COLUMNS_V2[0] == ID_COLUMN
        assert FOLD_COLUMNS_V2[1] == "split"
        assert "patient_group" not in FOLD_COLUMNS_V2

    def test_both_schemas_are_accepted_by_the_writer_and_the_reader(self, tmp_path):
        table = _study_table(n=300)
        for folds, name in ((build_folds(table), "folds_v1.csv"),
                            (build_folds_v2(table), "folds_v2.csv")):
            path = write_folds(folds, tmp_path / name)
            back = read_folds(path)
            assert tuple(back.columns) == tuple(folds.columns)
            assert back[ID_COLUMN].dtype == object
            assert list(back[ID_COLUMN]) == list(folds[ID_COLUMN])

    def test_a_v2_round_trip_keeps_the_negative_fold_index_an_integer(self):
        # Read back as a float, `fold == -1` still works and `astype(int)` still
        # works, so nothing complains -- but a NaN from a blank cell would become
        # a large negative integer and quietly move rows out of their fold.
        table = _study_table(n=300)
        folds = build_folds_v2(table)
        assert folds["fold"].min() == TEST_FOLD
        assert str(folds["fold"].dtype).startswith("int")

    def test_a_v2_parquet_fold_table_is_still_refused(self, tmp_path):
        # .gitignore:25 globs *.parquet even under data/folds/, so the file would
        # be written, look saved, and never be tracked.
        folds = build_folds_v2(_study_table(n=200))
        with pytest.raises(ValueError, match="must be .csv"):
            write_folds(folds, tmp_path / "folds_v2.parquet")

    def test_a_third_column_list_is_refused_naming_both_schemas(self):
        # Naming both matters: the reader who hits this needs to know which schema
        # their file is close to, not just that it is wrong.
        with pytest.raises(ValueError, match=r"contract requires.*\(v1\).*\(v2\)"):
            check_fold_columns((ID_COLUMN, "fold"))

    def test_a_reordered_v2_header_is_refused(self):
        folds = build_folds_v2(_study_table(n=200))[list(reversed(FOLD_COLUMNS_V2))]
        with pytest.raises(ValueError, match="contract requires"):
            check_fold_columns(folds.columns)

    def test_the_matched_schema_is_returned_so_a_caller_can_tell_them_apart(self):
        table = _study_table(n=200)
        assert check_fold_columns(build_folds(table).columns) == FOLD_COLUMNS
        assert check_fold_columns(build_folds_v2(table).columns) == FOLD_COLUMNS_V2


class TestFoldsV1StaysReadable:
    """``data/folds/folds_v1.csv`` is a frozen artifact, and v2 must not break it.

    Numbers already taken against it stop being reproducible the moment a reader
    refuses it, so v2 extends the accepted schemas rather than replacing them.

    What is checkable here is limited by design, and the limit is worth naming:
    the label table is cluster-only, so the label gates cannot run against the
    real file on a laptop. The properties below are exactly the ones the file
    carries on its own -- which is why the integer component ids are in it.
    """

    PATH = _REPO / "data" / "folds" / "folds_v1.csv"

    def test_it_still_reads_under_the_v1_schema(self):
        folds = read_folds(self.PATH)
        assert tuple(folds.columns) == FOLD_COLUMNS
        assert len(folds) == 4407
        assert folds[ID_COLUMN].dtype == object
        assert not folds[ID_COLUMN].duplicated().any()

    def test_no_group_spans_two_folds_in_the_arm_that_groups_on_it(self):
        folds = read_folds(self.PATH)
        for column, group in (("fold", "patient_group"), ("fold_scanner", "scanner_group")):
            spread = folds.groupby(group)[column].nunique()
            assert int((spread > 1).sum()) == 0, f"{group} spans folds in {column}"

    def test_every_fold_is_populated_and_no_row_is_held_out(self):
        # v1 has no holdout: every one of the 4,407 studies is in a fold, and
        # dev_mask must therefore keep all of them.
        folds = read_folds(self.PATH)
        for column in ("fold", "fold_scanner"):
            counts = folds[column].value_counts()
            assert sorted(counts.index) == list(range(N_FOLDS))
            assert counts.min() > 0
        assert dev_mask(folds).all()

    def test_every_language_bucket_is_present_in_every_fold(self):
        folds = read_folds(self.PATH)
        table = folds.pivot_table(index="language", columns="fold",
                                  values=ID_COLUMN, aggfunc="count").fillna(0)
        assert table.to_numpy().min() > 0, table

    def test_a_synthetic_v1_table_still_passes_the_full_label_gate(self):
        # The label gates the real file cannot be held to on a laptop. Synthetic,
        # with the measured prevalence range, so the gate itself is exercised.
        table = _study_table(n=1200)
        patient_arm, scanner_arm = _reports(build_folds(table), table)
        for report in (patient_arm, scanner_arm):
            report.raise_for_gate()
            assert report.empty_labels == ()
        assert sum(patient_arm.sizes) == len(table)


def _bridged_table(n: int = 300, *, n_scanners: int = 30, n_bridges: int = 15) -> pd.DataFrame:
    """Scanner clusters joined in pairs by a shared report group.

    This is the structure that breaks when connectivity is re-derived on a subset.
    Cluster ``2k`` and cluster ``2k+1`` are one ``scanner_group`` **only** because
    two studies bridge them by sharing report text. The holdout groups on
    ``report_group``, so it moves both bridge rows together -- and when it moves
    them to *test*, the two clusters are no longer connected in what is left.
    """
    table = _study_table(n=n, n_scanners=n_scanners, seed=5)
    scanners = [f"fp-{index % n_scanners:03d}" for index in range(n)]
    reports = [f"rg-{index:05d}" for index in range(n)]
    for bridge in range(n_bridges):
        left, right = 2 * bridge, 2 * bridge + 1
        # One row in each of two different scanner clusters, sharing report text.
        scanners[left] = f"fp-{(2 * bridge) % n_scanners:03d}"
        scanners[right] = f"fp-{(2 * bridge + 1) % n_scanners:03d}"
        reports[left] = reports[right] = f"bridge-{bridge:03d}"
    table["scanner_fingerprint"] = scanners
    table["report_group"] = reports
    return table


class TestGroupingSurvivesTheHoldout:
    """A component cut by the holdout must not come back as two components.

    Removing rows can disconnect a graph. ``scanner_group`` is the arm this bites,
    because the holdout groups on ``report_group`` and is *supposed* to cut scanner
    groups across dev and test -- so a scanner chain whose only bridge is held out
    is two groups on the dev pool and one group in the file that records it.
    Nothing raises; the scanner-grouped CV that exists to bound site memorisation
    is simply contaminated for those groups, and it reads as a smaller gap, which
    is the direction that flatters the model.
    """

    def _spanning(self, folds: pd.DataFrame, column: str, group: str) -> int:
        dev = folds[folds["split"] == "dev"]
        return int((dev.groupby(group)[column].nunique() > 1).sum())

    def test_no_scanner_group_spans_two_folds_when_the_holdout_cuts_it(self):
        table = _bridged_table()
        folds = build_folds_v2(table, seed=0)
        # The premise: the holdout really does cut scanner groups here, otherwise
        # this test would pass for the wrong reason.
        cut = int(
            (folds.groupby("scanner_group")["split"].nunique() > 1).sum()
        )
        assert cut > 0, "the fixture no longer exercises a holdout that cuts a scanner group"
        assert self._spanning(folds, "fold_scanner", "scanner_group") == 0
        assert self._spanning(folds, "fold", "report_group") == 0

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_it_holds_across_seeds_because_which_bridge_is_held_out_is_luck(self, seed):
        # The count of shattered components is fixed by the holdout; whether the
        # pieces then land in different folds is chance. A single seed passing
        # proves nothing, which is why the real corpus showed 4 under one setting
        # and 1 under another.
        folds = build_folds_v2(_bridged_table(), seed=seed)
        assert self._spanning(folds, "fold_scanner", "scanner_group") == 0

    def test_the_emitted_group_column_is_the_one_the_assignment_used(self):
        # The file is meant to be verifiable on its own, with no access to the
        # corpus. That only holds if the scanner_group column it carries is the
        # same partition that constrained fold_scanner.
        table = _bridged_table()
        folds = build_folds_v2(table, seed=0)
        ordered = table.sort_values(ID_COLUMN).reset_index(drop=True)
        expected = group_components(ordered, SCANNER_GROUP_KEYS)
        assert folds["scanner_group"].to_numpy().tolist() == expected.tolist()


class TestAssignFoldsAcceptsAPrecomputedPartition:
    """`components=` overrides `group_keys`, which is what makes the above fixable."""

    def test_a_passed_partition_is_honoured_over_the_keys(self):
        table = _study_table(n=120, n_scanners=12, seed=3)
        # Every row its own group by the keys, but two groups by the partition.
        forced = (np.arange(len(table)) % 2).astype(np.int64)
        assignment = assign_folds(table, n_folds=2, components=forced, strata_key=None)
        for group in (0, 1):
            assert len(set(assignment[forced == group])) == 1, "a group cannot span folds"

    def test_a_partition_with_gaps_is_densified_rather_than_refused(self):
        # A partition sliced out of a bigger table has holes in its ids. That is
        # the normal case here, not an error.
        table = _study_table(n=60, n_scanners=6, seed=4)
        sparse = (np.arange(len(table)) % 3 * 1000).astype(np.int64)
        assignment = assign_folds(table, n_folds=3, components=sparse, strata_key=None)
        assert set(assignment.tolist()) <= {0, 1, 2}
        for group in np.unique(sparse):
            assert len(set(assignment[sparse == group])) == 1

    def test_a_wrong_length_partition_is_refused_by_name(self):
        table = _study_table(n=40, n_scanners=4, seed=6)
        with pytest.raises(ValueError, match="one entry per row"):
            assign_folds(table, n_folds=2, components=np.zeros(7, dtype=np.int64))


def _report(languages: dict[str, tuple[int, ...]], sizes: tuple[int, ...], **kwargs):
    """A FoldReport carrying only what the language gate reads."""
    n = len(sizes)
    return FoldReport(
        fold_column=kwargs.pop("fold_column", "split"),
        n_folds=n,
        sizes=sizes,
        n_groups=sum(sizes),
        largest_group=1,
        positives={label: (1,) * n for label in LABELS},
        usable={label: (10,) * n for label in LABELS},
        languages=languages,
        empty_labels=(),
        overlaps={},
        gate_columns=(),
        **kwargs,
    )


class TestTheLanguageGate:
    """The second door on the 98%-Latin failure, which passed every gate there was.

    The first holdout built for this project produced a test set that was 98%
    Latin -- cyrillic 0 of 220, greek 4 of 321 -- and every label gate passed on
    it, because the gates asked only whether each label had a positive somewhere.
    Language was measured, printed, and never checked. Reports are multilingual
    and a test set that is one language measures the model on one language.
    """

    #: The corpus, as measured: latin 3320, turkish 546, greek 321, cyrillic 220.
    BALANCED = {
        "latin": (2823, 497),
        "turkish": (440, 106),
        "greek": (272, 49),
        "cyrillic": (187, 33),
    }

    def test_the_measured_holdout_passes(self):
        # These are the real numbers from the cluster build. If the tolerance ever
        # rejects them, the tolerance is wrong, not the split.
        report = _report(self.BALANCED, (3722, 685), gate_languages=True)
        assert report.starved_languages == ()
        assert report.ok

    def test_the_historical_98_percent_latin_test_set_is_refused(self):
        report = _report(
            {"latin": (3100, 220), "turkish": (546, 0), "greek": (317, 4), "cyrillic": (220, 0)},
            (4183, 224),
            gate_languages=True,
        )
        starved = dict(report.starved_languages)
        assert set(starved) == {"turkish", "greek", "cyrillic"}
        assert not report.ok
        with pytest.raises(ValueError, match="one language"):
            report.raise_for_gate()

    def test_an_empty_bucket_is_named_in_the_message_with_its_count(self):
        # Named with the holdout's own bin names, so the message reads "in test"
        # rather than "in fold1" -- the bin nobody may look at yet.
        report = _report(
            {"latin": (3320, 0), "cyrillic": (0, 220)}, (3320, 220),
            gate_languages=True, fold_names=SPLIT_VALUES,
        )
        with pytest.raises(ValueError) as caught:
            report.raise_for_gate()
        message = str(caught.value)
        assert "latin in test (0)" in message
        assert "cyrillic in dev (0)" in message

    def test_the_gate_is_off_unless_the_caller_asks_for_it(self):
        # Default off, so measuring a split never silently starts rejecting it.
        report = _report({"latin": (3320, 220), "cyrillic": (220, 0)}, (3540, 220))
        assert report.starved_languages != ()
        assert report.ok, "measured and reported, but not gated"

    def test_the_degenerate_fold_scanner_arm_is_not_gated_on_language(self):
        # Measured: cyrillic 164/13/5/2/3 across fold_scanner. Language collapses
        # across scanner groups because of where the scanners are, and gating it
        # would reject the pessimistic arm the leakage read depends on.
        report = _report(
            {"cyrillic": (164, 13, 5, 2, 3), "latin": (453, 549, 367, 737, 717)},
            (651, 676, 718, 855, 822),
            fold_column="fold_scanner",
        )
        assert report.starved_languages != ()
        assert report.ok

    def test_a_bucket_too_small_to_be_proportional_only_has_to_be_present(self):
        # Three expected in a bin is a number rounding moves; demanding half of it
        # would reject correct splits over a single study.
        present = _report({"tiny": (9, 3), "latin": (3000, 1000)}, (3009, 1003),
                          gate_languages=True)
        assert present.starved_languages == ()
        absent = _report({"tiny": (12, 0), "latin": (3000, 1000)}, (3012, 1000),
                         gate_languages=True)
        assert ("tiny", 1) in absent.starved_languages

    def test_the_gate_judges_a_bucket_against_the_bin_size_not_the_other_buckets(self):
        # latin 3320 vs cyrillic 220 is the corpus, not an imbalance to fix. A gate
        # that compared buckets to each other would reject every correct split.
        report = _report(self.BALANCED, (3722, 685), gate_languages=True)
        assert report.ok
        assert report.languages["latin"][1] > 10 * report.languages["cyrillic"][1]
