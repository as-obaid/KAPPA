"""The submission header is graded mechanically, and the labels are hostile to strings.

Why this exists: a misspelled or reordered column is not a degraded score, it is a
rejected file, and it burns one of a limited number of submission slots. Two label
names make the obvious shortcuts wrong.

* ``Baker's`` carries an apostrophe. Type the header by hand and it becomes
  ``Bakers``, which looks identical at a glance and is a different column name.
* **Five** of the twelve labels contain a space -- ``Medial Meniscus``,
  ``Lateral Meniscus``, ``Medial OA``, ``Lateral OA``, ``PF OA`` -- measured here
  rather than assumed. So ``header.split()`` recovers 6 fields where there are 13,
  and any reconstruction that goes through whitespace is silently broken.

So the header comes from ``kappa.constants.SUBMISSION_COLUMNS`` and from nowhere
else. :class:`TestTheHeaderIsDerivedNotTyped` is the class that fails if someone
replaces that derivation with a literal: it monkeypatches the column tuple and
asserts the written header follows it. A hardcoded string would not move.

The byte arithmetic is a measurement, not a preference. A constant-0.5 file over 3
studies with 64-character UIDs measured **470 bytes** and passed its contract
checks: 131 header bytes + 3 rows of (64 + 12x4 + 1). With ``\\r\\n`` the same file
is 474, so the line terminator is part of the artifact and is asserted.

The last class is about the ensemble. Macro ROC-AUC reads only the order of the
scores inside a column, so calibration cannot move it and probability averaging is
decided by whichever member has the widest spread. ``rank_blend`` is tested for the
property that matters: rescaling one member by any positive factor cannot change the
blend.
"""

from __future__ import annotations

import csv
import io

import numpy as np
import pytest

from kappa.constants import ID_COLUMN, LABELS, N_LABELS, SUBMISSION_COLUMNS
from kappa.evaluation.metrics import macro_roc_auc, rank_normalise
from kappa.inference import submission as sub
from kappa.inference.submission import (
    LINE_TERMINATOR,
    N_FIELDS,
    constant_submission,
    expected_header_bytes,
    rank_blend,
    read_submission,
    validate_submission,
    write_submission,
)

#: Three 64-character study UIDs, the length the real ones are, so a byte count
#: measured here is the byte count of a real file of the same shape.
UIDS = ("1" * 64, "2" * 64, "3" * 64)

#: The header the constant-0.5 file measured at, in bytes, including its newline.
HEADER_BYTES = 131

#: The whole measured artifact: 131 + 3 * (64 + 48 + 1).
CONSTANT_BYTES = 470


def _scores(n: int = 3, value: float = 0.5) -> np.ndarray:
    return np.full((n, N_LABELS), float(value))


def _written(tmp_path, scores=None, uids=UIDS, **over):
    path = tmp_path / "submission.csv"
    write_submission(path, uids, scores if scores is not None else _scores(len(uids)), **over)
    return path


class TestTheLabelNamesThemselves:
    """Measured properties of the twelve names, which every other claim rests on."""

    def test_exactly_five_labels_contain_a_space(self):
        with_space = [name for name in LABELS if " " in name]
        assert len(with_space) == 5
        assert with_space == [
            "Medial Meniscus",
            "Lateral Meniscus",
            "Medial OA",
            "Lateral OA",
            "PF OA",
        ]

    def test_exactly_one_label_contains_an_apostrophe(self):
        assert [name for name in LABELS if "'" in name] == ["Baker's"]

    def test_bakers_without_the_apostrophe_is_not_a_label(self):
        assert "Bakers" not in LABELS

    def test_no_label_contains_a_comma_so_quote_minimal_quotes_nothing(self):
        # Worth stating: this is why a comma split *does* happen to work and a
        # whitespace split does not. The hazard is the apostrophe and the spaces.
        assert not any("," in name for name in LABELS)

    def test_there_are_twelve_labels_and_thirteen_columns(self):
        assert len(LABELS) == N_LABELS == 12
        assert len(SUBMISSION_COLUMNS) == N_FIELDS == 13
        assert SUBMISSION_COLUMNS[0] == ID_COLUMN


class TestTheHeaderIsDerivedNotTyped:
    """The class that fails if the header becomes a literal instead of a derivation."""

    def test_the_header_follows_submission_columns_rather_than_a_literal(self, monkeypatch):
        # The decisive test. Swap the constant and the header must swap with it. A
        # hardcoded string -- or one rebuilt from a hand-typed line -- would not move,
        # and this is the only assertion here that a literal cannot satisfy.
        monkeypatch.setattr(sub, "SUBMISSION_COLUMNS", ("Id", "A", "B"))
        assert expected_header_bytes() == b"Id,A,B\n"

    def test_the_written_header_follows_submission_columns_too(self, monkeypatch, tmp_path):
        # Not just the expectation -- the writer. Otherwise the two could be derived
        # from different sources and only the comparison between them would be tested.
        monkeypatch.setattr(sub, "SUBMISSION_COLUMNS", ("Id", "A", "B"))
        monkeypatch.setattr(sub, "N_LABELS", 2)
        path = tmp_path / "s.csv"
        write_submission(path, ["x"], np.zeros((1, 2)))
        assert path.read_bytes().split(b"\n")[0] == b"Id,A,B"

    def test_a_whitespace_split_of_the_header_does_not_recover_the_columns(self):
        # The concrete failure mode: five labels contain a space, so any
        # reconstruction through whitespace yields 6 fields for 13 columns.
        line = expected_header_bytes().decode().rstrip("\n")
        assert len(line.split()) == 6
        assert len(line.split()) != len(SUBMISSION_COLUMNS)

    def test_a_hand_typed_header_that_drops_the_apostrophe_fails_validation(self, tmp_path):
        # Exactly the file someone produces by typing the header out, and exactly what
        # the validator has to catch before it is uploaded.
        path = tmp_path / "typed.csv"
        typed = ",".join(["StudyInstanceUID", *[n.replace("'", "") for n in LABELS]])
        path.write_text(typed + "\n" + "\n".join(f"{u}," + ",".join(["0.5"] * 12) for u in UIDS))
        check = validate_submission(path)
        assert check.results["header exact"] is False
        assert check.results["Baker's apostrophe intact"] is False
        assert not check.ok

    def test_a_header_with_underscores_for_spaces_fails_validation(self, tmp_path):
        path = tmp_path / "underscored.csv"
        typed = ",".join(["StudyInstanceUID", *[n.replace(" ", "_") for n in LABELS]])
        path.write_text(typed + "\n" + f"{UIDS[0]}," + ",".join(["0.5"] * 12) + "\n")
        assert validate_submission(path).results["header exact"] is False

    def test_a_reordered_header_fails_validation(self, tmp_path):
        path = tmp_path / "reordered.csv"
        reordered = [ID_COLUMN, *reversed(LABELS)]
        buffer = io.StringIO()
        csv.writer(buffer, lineterminator="\n").writerow(reordered)
        path.write_text(buffer.getvalue() + f"{UIDS[0]}," + ",".join(["0.5"] * 12) + "\n")
        check = validate_submission(path)
        assert check.results["header exact"] is False
        # And it is caught even though the apostrophe survived and the field count is
        # right, which is why "header exact" is a byte comparison and not a set test.
        assert check.results["Baker's apostrophe intact"] is True
        assert check.results["13 fields per row"] is True


class TestTheWrittenBytes:
    """The measured artifact: 131 header bytes, 470 total, LF terminators."""

    def test_the_header_is_the_measured_131_bytes(self):
        assert len(expected_header_bytes()) == HEADER_BYTES

    def test_the_constant_baseline_is_the_measured_470_bytes(self, tmp_path):
        path = constant_submission(tmp_path / "s.csv", UIDS)
        assert path.stat().st_size == CONSTANT_BYTES

    def test_the_byte_count_decomposes_exactly(self, tmp_path):
        # 470 = 131 + 3 * (64 UID + 48 of ",0.5" + 1 newline). Stated as arithmetic so
        # a change to the float format or the terminator cannot pass unnoticed.
        path = constant_submission(tmp_path / "s.csv", UIDS)
        assert path.stat().st_size == HEADER_BYTES + len(UIDS) * (64 + 12 * 4 + 1)

    def test_the_line_terminator_is_lf_not_crlf(self):
        assert LINE_TERMINATOR == "\n"

    def test_no_carriage_return_reaches_the_file(self, tmp_path):
        # csv.writer defaults to \r\n. With CRLF the known-good 470-byte file is 474,
        # so this is a silent change to a measured artifact.
        assert b"\r" not in _written(tmp_path).read_bytes()

    def test_the_file_has_no_byte_order_mark(self, tmp_path):
        assert not _written(tmp_path).read_bytes().startswith(b"\xef\xbb\xbf")

    def test_the_written_header_is_byte_for_byte_the_expectation(self, tmp_path):
        first = _written(tmp_path).read_bytes().split(b"\n", 1)[0] + b"\n"
        assert first == expected_header_bytes()

    def test_the_apostrophe_survives_the_round_trip_unquoted(self, tmp_path):
        raw = _written(tmp_path).read_bytes()
        assert b"Baker's" in raw
        assert b'"Baker\'s"' not in raw

    def test_the_writer_verifies_the_header_it_wrote(self, tmp_path, monkeypatch):
        # The re-read is the guarantee, so removing it must break a test. Make the
        # expectation disagree with the write and the writer has to raise.
        path = tmp_path / "s.csv"
        monkeypatch.setattr(sub, "expected_header_bytes", lambda: b"nope\n")
        with pytest.raises(AssertionError, match="byte for byte"):
            write_submission(path, UIDS, _scores())


class TestScoresThatMustNotBeWritten:
    """A rejected file is worse than a bad score, so these fail at write time."""

    def test_a_nan_score_is_refused_and_the_label_is_named(self, tmp_path):
        scores = _scores()
        scores[1, LABELS.index("Baker's")] = np.nan
        with pytest.raises(ValueError, match="Baker's"):
            write_submission(tmp_path / "s.csv", UIDS, scores)

    def test_the_nan_error_says_masking_applies_to_the_loss_not_the_output(self, tmp_path):
        scores = _scores()
        scores[0, 0] = np.nan
        with pytest.raises(ValueError, match="masking applies to the loss"):
            write_submission(tmp_path / "s.csv", UIDS, scores)

    def test_an_infinite_score_is_refused(self, tmp_path):
        scores = _scores()
        scores[2, 3] = np.inf
        with pytest.raises(ValueError, match="non-finite"):
            write_submission(tmp_path / "s.csv", UIDS, scores)

    def test_the_wrong_number_of_label_columns_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="twelve labels"):
            write_submission(tmp_path / "s.csv", UIDS, np.zeros((3, 11)))

    def test_a_uid_count_mismatch_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="score rows"):
            write_submission(tmp_path / "s.csv", UIDS[:2], _scores(3))

    def test_a_duplicate_uid_is_refused_because_the_leaderboard_joins_on_it(self, tmp_path):
        with pytest.raises(ValueError, match="duplicate StudyInstanceUID"):
            write_submission(tmp_path / "s.csv", (UIDS[0], UIDS[0]), _scores(2))

    def test_a_single_row_of_scores_is_reshaped_rather_than_refused(self, tmp_path):
        path = write_submission(tmp_path / "s.csv", [UIDS[0]], np.full(N_LABELS, 0.25))
        _, uids, values = read_submission(path)
        assert uids == [UIDS[0]] and values.shape == (1, N_LABELS)


class TestTheSevenChecks:
    """"7/7" only means something if an unrun check is not counted as a pass."""

    def test_a_good_file_with_expected_uids_passes_all_seven(self, tmp_path):
        check = validate_submission(_written(tmp_path), expected_uids=UIDS)
        assert check.n_checks == 7
        assert check.n_passed == 7
        assert check.ok

    def test_without_expected_uids_two_checks_report_unchecked_not_passed(self, tmp_path):
        check = validate_submission(_written(tmp_path))
        assert check.n_checks == 7
        assert check.n_passed == 5
        assert check.results["row count matches test.csv"] is None
        assert check.results["UID order matches test.csv"] is None

    def test_an_unchecked_entry_still_leaves_ok_true(self, tmp_path):
        # Unchecked is not a failure either -- it is an absence of evidence, and the
        # count is what says so.
        assert validate_submission(_written(tmp_path)).ok

    def test_a_reordered_row_fails_the_uid_order_check_only(self, tmp_path):
        path = _written(tmp_path, uids=(UIDS[1], UIDS[0], UIDS[2]))
        check = validate_submission(path, expected_uids=UIDS)
        assert check.results["UID order matches test.csv"] is False
        assert check.results["row count matches test.csv"] is True
        assert not check.ok

    def test_a_short_submission_fails_the_row_count_check(self, tmp_path):
        check = validate_submission(_written(tmp_path, uids=UIDS[:2]), expected_uids=UIDS)
        assert check.results["row count matches test.csv"] is False

    def test_an_empty_body_fails_the_finite_values_check(self, tmp_path):
        path = tmp_path / "s.csv"
        path.write_bytes(expected_header_bytes())
        assert validate_submission(path).results["values finite"] is False

    def test_a_short_row_fails_the_field_count_check(self, tmp_path):
        path = tmp_path / "s.csv"
        path.write_bytes(expected_header_bytes() + f"{UIDS[0]},0.5\n".encode())
        assert validate_submission(path).results["13 fields per row"] is False

    def test_the_summary_reports_the_ratio_and_the_byte_count(self, tmp_path):
        text = validate_submission(_written(tmp_path), expected_uids=UIDS).summary()
        assert "7/7 checks passed" in text
        assert "bytes" in text

    def test_the_summary_marks_an_unrun_check_distinctly_from_a_pass(self, tmp_path):
        text = validate_submission(_written(tmp_path)).summary()
        assert "[----]" in text
        assert "[PASS]" in text


class TestRoundTrip:
    def test_the_header_read_back_equals_submission_columns(self, tmp_path):
        header, _, _ = read_submission(_written(tmp_path))
        assert tuple(header) == SUBMISSION_COLUMNS

    def test_the_values_survive_the_float_format(self, tmp_path):
        scores = np.linspace(0.0, 1.0, 3 * N_LABELS).reshape(3, N_LABELS)
        _, uids, values = read_submission(_written(tmp_path, scores=scores))
        assert uids == list(UIDS)
        np.testing.assert_allclose(values, scores, atol=1e-6)

    def test_a_long_dotted_uid_is_not_turned_into_a_float(self, tmp_path):
        dotted = ("1.2.840.10008.9.0.000001", "1.2.840.10008.9.0.000002")
        _, uids, _ = read_submission(_written(tmp_path, uids=dotted, scores=_scores(2)))
        assert uids == list(dotted)

    def test_an_empty_file_is_refused_on_read(self, tmp_path):
        path = tmp_path / "empty.csv"
        path.write_text("")
        with pytest.raises(ValueError, match="empty"):
            read_submission(path)


class TestRankBlendIsTheOnlyEnsemble:
    """No calibration and no probability averaging, and the tests that pin that down."""

    def test_rescaling_one_member_cannot_change_the_blend(self, ):
        # The defining property. A probability average would move here, because the
        # rescaled member's spread would dominate it; a rank blend cannot, because
        # ranks are invariant to any positive monotone rescaling.
        rng = np.random.default_rng(0)
        a = rng.normal(size=(40, N_LABELS))
        b = rng.normal(size=(40, N_LABELS))
        np.testing.assert_allclose(rank_blend([a, b]), rank_blend([a * 1000.0, b]))

    def test_shifting_one_member_cannot_change_the_blend(self):
        rng = np.random.default_rng(1)
        a = rng.normal(size=(40, N_LABELS))
        b = rng.normal(size=(40, N_LABELS))
        np.testing.assert_allclose(rank_blend([a, b]), rank_blend([a + 50.0, b]))

    def test_a_probability_average_would_have_moved(self):
        # The contrast that makes the two tests above meaningful rather than trivially
        # true of any blend. Without this, "invariant" could just mean "constant".
        rng = np.random.default_rng(2)
        a = rng.normal(size=(40, N_LABELS))
        b = rng.normal(size=(40, N_LABELS))
        plain = (a + b) / 2.0
        scaled = (a * 1000.0 + b) / 2.0
        assert not np.allclose(plain, scaled)

    def test_a_single_member_blend_is_its_own_rank_normalisation(self):
        rng = np.random.default_rng(3)
        a = rng.normal(size=(20, N_LABELS))
        np.testing.assert_allclose(rank_blend([a]), rank_normalise(a))

    def test_the_blend_is_per_column_not_across_labels(self):
        # A shared normalisation across the twelve labels would mix twelve unrelated
        # orderings. Column 0 of the blend must depend only on column 0 of the inputs.
        rng = np.random.default_rng(4)
        a = rng.normal(size=(20, N_LABELS))
        b = a.copy()
        b[:, 1] += 100.0
        np.testing.assert_allclose(rank_blend([a])[:, 0], rank_blend([b])[:, 0])

    def test_rank_normalisation_leaves_the_auc_bit_identical(self):
        # Why the rank transform is safe to apply before writing: it is monotone
        # within a column, so the metric cannot see it.
        rng = np.random.default_rng(5)
        truth = (rng.random((60, N_LABELS)) < 0.3).astype(float)
        scores = rng.normal(size=(60, N_LABELS))
        assert macro_roc_auc(truth, scores).macro == macro_roc_auc(
            truth, rank_normalise(scores)
        ).macro

    def test_a_sigmoid_also_leaves_the_auc_identical_which_is_why_it_is_pointless(self):
        rng = np.random.default_rng(6)
        truth = (rng.random((60, N_LABELS)) < 0.3).astype(float)
        scores = rng.normal(size=(60, N_LABELS))
        sigmoid = 1.0 / (1.0 + np.exp(-scores))
        assert macro_roc_auc(truth, scores).macro == macro_roc_auc(truth, sigmoid).macro

    def test_blending_disagreeing_shapes_is_refused(self):
        with pytest.raises(ValueError, match="shape"):
            rank_blend([np.zeros((3, N_LABELS)), np.zeros((4, N_LABELS))])

    def test_blending_nothing_is_refused(self):
        with pytest.raises(ValueError, match="at least one"):
            rank_blend([])

    def test_the_blend_lands_inside_the_unit_interval(self):
        rng = np.random.default_rng(7)
        blended = rank_blend([rng.normal(size=(30, N_LABELS)) for _ in range(3)])
        assert blended.min() > 0.0 and blended.max() < 1.0

    def test_there_is_no_calibration_function_to_reach_for(self):
        # A named absence. The one way calibration gets added is someone importing a
        # helper that was quietly left available.
        for forbidden in ("calibrate", "platt", "isotonic", "temperature", "sigmoid"):
            assert not any(forbidden in name.lower() for name in sub.__all__)
