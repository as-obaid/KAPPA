"""The residue feature table, checked as the leak guard it is rather than as a schema.

The teacher's whole legitimacy rests on one claim: the channel it reads carries report
information that the twelve labels DISCARDED, and not the labels themselves. That claim
is not self-evident and it has already been wrong four times -- ``build_residue_features``
records four candidate rules that scored 0.7643 to 0.8928 text-only against a 0.765 bar
before the fifth cleared it at 0.6910.

What makes the failure worth a test file of its own is that it is SILENT. A teacher
handed the full 156-channel table converges, prints a clean loss curve, posts a large
margin, and transfers nothing to the student -- and that table, probed out of fold,
reproduces the twelve labels at 0.9538 macro, with ACL, MCL, Medial Meniscus
and Lateral Meniscus all at 1.0000. Nothing in the training loop can tell the two tables
apart, because the residue is deliberately kept 156 columns wide with the identical name
order so that ``load_features`` and ``load_frozen_teacher`` still accept it. The only
things that distinguish them are the VALUES, and these tests.

So the assertions here are about the exclusion rule and not about the file format: every
cluster the label map ties to a label must be absent from the residue, in both polarity
channels, derived from ``CLUSTER_LABELS`` rather than from a list anybody maintains by
hand.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from kappa.constants import LABELS, N_LABELS
from kappa.labels.features import (
    FEATURE_TABLE_COLUMNS,
    PRESENT_COLUMN,
    REPORT_CLUSTERS,
    REPORT_FEATURE_DIM,
    REPORT_FEATURE_NAMES,
    UID_COLUMN,
    load_features,
)
from kappa.labels.label_map import CLUSTER_LABELS, UNMAPPED_CLUSTERS

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_residue_features import (  # noqa: E402
    EXPECTED_KEPT_CHANNELS,
    residue_channels,
    residue_clusters,
)

pd = pytest.importorskip("pandas")

REPO = Path(__file__).resolve().parents[1]


def _frame(uids: list[str], *, fire: dict[str, list[str]] | None = None) -> "pd.DataFrame":
    """A well-formed feature table: every column present, in order, zeros by default.

    ``fire`` names the channels set to 1.0 for a given uid, which is how these tests put
    a value into a channel and then ask whether the residue kept or zeroed it.
    """
    data: dict[str, object] = {UID_COLUMN: list(uids)}
    for name in REPORT_FEATURE_NAMES:
        data[name] = np.zeros(len(uids), dtype=np.float32)
    data[PRESENT_COLUMN] = np.ones(len(uids), dtype=bool)
    frame = pd.DataFrame(data)
    for uid, channels in (fire or {}).items():
        row = frame.index[frame[UID_COLUMN] == uid][0]
        for channel in channels:
            frame.loc[row, channel] = 1.0
    return frame[list(FEATURE_TABLE_COLUMNS)]


class TestTheVocabularyIsTheShapeEverythingElseAssumes:
    def test_the_table_is_78_clusters_times_2_polarity_channels(self):
        assert len(REPORT_CLUSTERS) == 78
        assert REPORT_FEATURE_DIM == 156 == len(REPORT_FEATURE_NAMES)
        # `TeacherClassifier.__init__` defaults `report_dim` to this and its forward
        # raises on a mismatch, so a change here silently renames every teacher arm.
        assert len(FEATURE_TABLE_COLUMNS) == 1 + 156 + 1

    def test_the_cluster_order_is_sorted_and_not_set_iteration_order(self):
        # CPython randomises frozenset iteration per process. If the column order came
        # from `CLUSTER_NAMES` directly, two builds in two processes would produce two
        # different tables under one name -- and `load_features` would accept either,
        # because it checks the names it was GIVEN against `REPORT_FEATURE_NAMES`
        # computed in the SAME process, so both sides would drift together and agree.
        assert REPORT_CLUSTERS == tuple(sorted(REPORT_CLUSTERS))
        assert REPORT_FEATURE_NAMES[0] == f"{REPORT_CLUSTERS[0]}:pos"
        assert REPORT_FEATURE_NAMES[1] == f"{REPORT_CLUSTERS[0]}:neg"

    def test_every_channel_is_a_known_cluster_and_a_known_polarity(self):
        for name in REPORT_FEATURE_NAMES:
            cluster, _, polarity = name.rpartition(":")
            assert cluster in REPORT_CLUSTERS
            assert polarity in {"pos", "neg"}


class TestNoDirectLabelChannelReachesTheTeacher:
    """The one thing this file exists for. Derived from CLUSTER_LABELS, never listed."""

    def test_no_cluster_with_any_label_mapping_survives_as_a_channel(self):
        kept = set(residue_clusters())
        offenders = {
            cluster: mapping
            for cluster, mapping in CLUSTER_LABELS.items()
            if mapping and cluster in kept
        }
        assert not offenders, (
            "these clusters map to a label AND survive into the residue, so the teacher "
            f"can read the label off its own input: {sorted(offenders)}"
        )

    def test_both_polarity_channels_of_a_mapped_cluster_are_excluded(self):
        # Exclusion is per CLUSTER, and it has to be: a negative statement about the ACL
        # ("no ACL tear") determines the ACL label just as surely as a positive one. A
        # rule that dropped only `:pos` would leave the label recoverable by flipping a
        # sign, which is not a harder problem for a linear model.
        channels = set(residue_channels())
        for cluster, mapping in CLUSTER_LABELS.items():
            if not mapping:
                continue
            assert f"{cluster}:pos" not in channels, cluster
            assert f"{cluster}:neg" not in channels, cluster

    @pytest.mark.parametrize(
        "cluster",
        [
            # One per label family, named so a reader can see what is being excluded
            # rather than trusting the comprehension above. Each of these is a cluster
            # whose whole job is to determine one of the twelve columns.
            "ACL",
            "MCL",
            "MENISCUS_MEDIAL",
            "MENISCUS_LATERAL",
            "JOINT_SPACE_EFFUSION",
            "CYST_POPLITEAL_BAKER",
            "SYNOVIUM_PLICA",
            "CARTILAGE_PATELLOFEMORAL",
            "CARTILAGE_MEDIAL_TIBIOFEMORAL",
            "CARTILAGE_LATERAL_TIBIOFEMORAL",
        ],
    )
    def test_the_obvious_shortcuts_are_gone_by_name(self, cluster):
        assert cluster in CLUSTER_LABELS and CLUSTER_LABELS[cluster], (
            f"{cluster} no longer maps to any label, so this test is asserting "
            "something other than what it was written to assert"
        )
        assert cluster not in set(residue_clusters())

    def test_an_ambiguous_only_mapping_is_still_excluded(self):
        # `MENISCUS_SIDE_UNSPECIFIED` maps to {MM: A, LM: A} -- ambiguous, never direct.
        # It is still excluded, because `not {"MM": "A"}` is False. The candidate rule
        # that kept the DIRECT-only exclusions measured 0.7809 text-only, above the bar.
        ambiguous = {
            cluster
            for cluster, mapping in CLUSTER_LABELS.items()
            if mapping and all(directness == "A" for directness in mapping.values())
        }
        assert ambiguous, "no ambiguous-only cluster found; the fixture assumption moved"
        assert not (ambiguous & set(residue_clusters()))

    def test_what_survives_is_exactly_the_unmapped_and_the_empty_mapped(self):
        # The positive statement of the rule, so this file says what the residue IS and
        # not only what it is not.
        expected = {
            cluster for cluster in REPORT_CLUSTERS if not CLUSTER_LABELS.get(cluster)
        }
        assert set(residue_clusters()) == expected
        assert UNMAPPED_CLUSTERS <= expected

    def test_the_kept_count_is_the_one_the_leak_number_was_measured_on(self):
        # 29 clusters, 58 channels, macro 0.6910. If the label map gains or loses a
        # mapping the residue MOVES and the 0.6910 stops describing this table -- so the
        # count is pinned here, and `--expect-kept` refuses to write when it drifts.
        assert len(residue_clusters()) == 29
        assert len(residue_channels()) == EXPECTED_KEPT_CHANNELS == 58
        assert len(residue_channels()) == 2 * len(residue_clusters())

    def test_the_residue_is_a_minority_of_the_table(self):
        assert REPORT_FEATURE_DIM - len(residue_channels()) == 98


class TestTheConstructionIsDeterministic:
    def test_repeated_calls_agree(self):
        assert residue_clusters() == residue_clusters()
        assert residue_channels() == residue_channels()

    def test_the_channels_are_in_table_order_not_arbitrary_order(self):
        # The builder indexes columns by `REPORT_FEATURE_NAMES.index(...)`. If these came
        # back in a different order the zeroing would still be correct, but a reader
        # diffing two runs' printed kept-sets would see spurious churn.
        positions = [REPORT_FEATURE_NAMES.index(name) for name in residue_channels()]
        assert positions == sorted(positions)

    def test_a_fresh_interpreter_produces_the_same_kept_set(self):
        # The real determinism risk is PYTHONHASHSEED, which only differs BETWEEN
        # processes -- so this has to run some. Three subprocesses with three seeds must
        # agree, which is what `sorted(CLUSTER_NAMES)` buys and what a bare set
        # comprehension over `CLUSTER_NAMES` would have silently cost.
        code = (
            "import sys; sys.path.insert(0, 'scripts');"
            "from build_residue_features import residue_channels;"
            "print(','.join(residue_channels()))"
        )
        outputs = []
        for seed in ("0", "1", "12345"):
            done = subprocess.run(
                [sys.executable, "-c", code],
                cwd=REPO,
                capture_output=True,
                text=True,
                env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"},
                check=True,
            )
            outputs.append(done.stdout.strip())
        assert len(set(outputs)) == 1, "the kept set depends on PYTHONHASHSEED"
        assert outputs[0].split(",") == list(residue_channels())


class TestTheBuilderZeroesRatherThanDrops:
    """Width and name order must survive, or `load_features` refuses the output."""

    def _build(self, tmp_path, frame, *, expect_kept=None):
        from build_residue_features import main

        source = tmp_path / "report_features.parquet"
        out = tmp_path / "residue.parquet"
        frame.to_parquet(source, index=False)
        argv = ["--features", str(source), "--out", str(out)]
        if expect_kept is not None:
            argv += ["--expect-kept", str(expect_kept)]
        return main(argv), out

    def test_the_output_keeps_the_full_width_and_the_full_name_order(self, tmp_path):
        rc, out = self._build(tmp_path, _frame(["a", "b"]))
        assert rc == 0
        uids, values, present = load_features(out)
        assert uids == ["a", "b"]
        assert values.shape == (2, REPORT_FEATURE_DIM)
        assert present.tolist() == [True, True]

    def test_a_kept_channel_survives_and_a_mapped_channel_is_zeroed(self, tmp_path):
        keep = residue_channels()[0]
        drop = "ACL:pos"
        assert drop not in set(residue_channels())
        rc, out = self._build(tmp_path, _frame(["a"], fire={"a": [keep, drop]}))
        assert rc == 0
        _, values, _ = load_features(out)
        assert values[0, REPORT_FEATURE_NAMES.index(keep)] == pytest.approx(1.0)
        assert values[0, REPORT_FEATURE_NAMES.index(drop)] == 0.0

    def test_every_excluded_channel_is_exactly_zero_for_every_row(self, tmp_path):
        # Fire EVERY channel, then assert the complement of the kept set is all zero.
        # "Mostly zero" is not the property: one surviving cell in a label-determining
        # channel is one study the teacher can read the answer off.
        rc, out = self._build(
            tmp_path, _frame(["a", "b"], fire={"a": list(REPORT_FEATURE_NAMES)})
        )
        assert rc == 0
        _, values, _ = load_features(out)
        kept = set(residue_channels())
        for index, name in enumerate(REPORT_FEATURE_NAMES):
            if name in kept:
                assert values[0, index] == pytest.approx(1.0), name
            else:
                assert not values[:, index].any(), f"{name} survived into the residue"

    def test_the_presence_flag_is_carried_through_untouched(self, tmp_path):
        frame = _frame(["a", "b"])
        frame.loc[1, PRESENT_COLUMN] = False
        rc, out = self._build(tmp_path, frame)
        assert rc == 0
        _, _, present = load_features(out)
        # An extraction failure must stay distinguishable from an empty report after the
        # residue pass; the teacher gates its whole branch on this flag, and the residue
        # ablation arm works by clearing it.
        assert present.tolist() == [True, False]

    def test_the_expect_kept_tripwire_refuses_to_write_when_the_count_moves(self, tmp_path):
        rc, out = self._build(
            tmp_path, _frame(["a"]), expect_kept=EXPECTED_KEPT_CHANNELS + 1
        )
        assert rc != 0, "a drifted kept-count must fail the build, not rename the table"
        assert not out.exists(), "a refused build must leave no output to train on"

    def test_a_permuted_source_is_refused_rather_than_silently_scrambled(self, tmp_path):
        from build_residue_features import main

        frame = _frame(["a"])
        permuted = frame[[UID_COLUMN, *reversed(REPORT_FEATURE_NAMES), PRESENT_COLUMN]]
        source = tmp_path / "permuted.parquet"
        permuted.to_parquet(source, index=False)
        # A permuted table trains, converges and reports a number. This is the guard
        # that the 156-wide output design exists to preserve.
        with pytest.raises(Exception):
            main(["--features", str(source), "--out", str(tmp_path / "o.parquet")])


class TestRowAndFoldAlignment:
    """A table short by studies looks perfectly healthy; only a count catches it."""

    def test_the_uid_column_is_what_every_other_table_joins_on(self):
        assert UID_COLUMN == "study_uid"
        folds = pd.read_csv(REPO / "data/folds/folds_v2.csv")
        # The feature table keys on `study_uid` and the fold table on
        # `StudyInstanceUID`; `report_frame` and `probe_report_leak` both RENAME on
        # load, and nothing validates the values, so the two names are pinned here.
        assert "StudyInstanceUID" in folds.columns
        assert {"fold", "split"} <= set(folds.columns)

    def test_the_fold_table_has_five_dev_folds_and_a_held_out_test_split(self):
        folds = pd.read_csv(REPO / "data/folds/folds_v2.csv")
        dev = folds[folds["split"] == "dev"]
        assert sorted(dev["fold"].unique().tolist()) == [0, 1, 2, 3, 4]
        # A test row carries fold -1, so a naive groupby would invent a sixth fold.
        assert (folds[folds["split"] == "test"]["fold"] == -1).all()

    def test_a_residue_row_set_that_misses_dev_studies_is_detectable(self, tmp_path):
        # The check the builders do NOT do, written as the assertion a caller can make.
        # `build_report_features`'s own docstring warns that pointing `--records` at the
        # wrong directory yields a table "quietly short by 4,107 studies that looks
        # perfectly healthy" -- healthy because every column is present and in order.
        folds = pd.read_csv(REPO / "data/folds/folds_v2.csv")
        dev = set(folds[folds["split"] == "dev"]["StudyInstanceUID"])
        source = tmp_path / "short.parquet"
        _frame(sorted(dev)[:10]).to_parquet(source, index=False)
        uids, _, _ = load_features(source)
        assert set(uids) < dev
        assert len(dev - set(uids)) == len(dev) - 10


def _consumed(uids: list[str], **kwargs) -> "pd.DataFrame":
    """A feature frame as the DATASET sees it, not as the parquet stores it.

    The boundary is easy to trip over and worth naming: the parquet keys on
    ``study_uid`` (`features.UID_COLUMN`), and `report_frame` RENAMES that to
    ``StudyInstanceUID`` (`constants.ID_COLUMN`) on load so the frame joins like every
    other table in the project. `_index_features` therefore requires the renamed form,
    and a test that hands it the on-disk form fails with a missing-column KeyError that
    reads like a broken fixture rather than like the rename it actually is.
    """
    from kappa.constants import ID_COLUMN

    return _frame(uids, **kwargs).rename(columns={UID_COLUMN: ID_COLUMN})


class TestMissingnessIsAZeroVectorAndAClearedFlag:
    def test_a_study_absent_from_the_table_is_zeros_and_not_an_error(self):
        pytest.importorskip("torch")
        from kappa.training.report_dataset import _index_features

        index, values, present = _index_features(_consumed(["present-study"]))
        assert values.shape == (1, REPORT_FEATURE_DIM)
        assert present.tolist() == [True]
        # `ReportStudyDataset.__getitem__` does `self._report_index.get(uid)` and emits
        # zeros with present=False on a miss. That is deliberate, not an oversight: the
        # fold tables carry every study and the label pipeline can fail on a report, so
        # the alternative is a fabricated all-negative report the teacher would read as
        # twelve confident negatives.
        assert "absent-study" not in index
        assert set(index) == {"present-study"}

    def test_a_nonfinite_feature_cell_is_refused_rather_than_trained_on(self):
        pytest.importorskip("torch")
        from kappa.training.report_dataset import _index_features

        frame = _consumed(["a"])
        frame.loc[0, REPORT_FEATURE_NAMES[0]] = float("nan")
        # Fail closed. A NaN here would reach the teacher's report MLP, and
        # `_check_distill_inputs` guards the teacher's LOGITS rather than its inputs --
        # by the time a NaN shows up there, the soft targets for the whole batch are
        # already poisoned, because `0.0 * nan` is `nan`.
        with pytest.raises(ValueError):
            _index_features(frame)


class TestTheLeakProbeIsUsableAsAGate:
    def test_the_probe_reports_every_one_of_the_twelve_labels_not_just_macro(self):
        # The residue is clean on macro (0.6910) and NOT clean on two columns: ACL
        # 0.7841 and MCL 0.7518 text-only, both above the 0.765 pipeline figure. That is
        # why the standing obligation is to report macro, ACL and MCL separately -- and
        # it is only possible because the probe prints all twelve rows.
        source = (REPO / "scripts/probe_report_leak.py").read_text()
        assert "LABELS" in source and "MACRO" in source
        assert "ACL" in LABELS and "MCL" in LABELS
        assert len(LABELS) == N_LABELS == 12
