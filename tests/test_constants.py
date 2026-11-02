"""Lock the submission contract.

These are not aspirational tests. The submission header is graded mechanically:
a misspelled, reordered or requoted column scores as an ERROR and burns one of a
limited number of submission slots. `Baker's` carries an apostrophe and four
labels contain a space, so the header is exactly the kind of constant that a
well-meaning refactor breaks silently.

The expected values here are written out literally rather than derived from the
module under test -- a test that imports its expectation from the code it is
testing verifies nothing.
"""

from __future__ import annotations

import csv
import io

from kappa.constants import ID_COLUMN, LABEL_GROUPS, LABELS, N_LABELS, SUBMISSION_COLUMNS

# Transcribed from docs/dataset.md (Submission format) by hand. Do not generate this.
EXPECTED_HEADER = (
    "StudyInstanceUID,ACL,MCL,Medial Meniscus,Lateral Meniscus,Medial OA,"
    "Lateral OA,PF OA,Effusion,Synovitis,Baker's,Contusion,Fracture"
)


def test_label_count():
    assert N_LABELS == 12
    assert len(LABELS) == 12
    assert len(set(LABELS)) == 12, "duplicate label name"


def test_header_matches_competition_spec_exactly():
    assert ",".join(SUBMISSION_COLUMNS) == EXPECTED_HEADER


def test_id_column_is_first_and_labels_follow_in_order():
    assert SUBMISSION_COLUMNS[0] == ID_COLUMN == "StudyInstanceUID"
    assert tuple(SUBMISSION_COLUMNS[1:]) == LABELS


def test_apostrophe_and_spaces_survive():
    assert "Baker's" in LABELS, "apostrophe stripped from Baker's"
    # Five labels contain a space. Not to be confused with the *four*
    # laterality-dependent columns (Medial/Lateral Meniscus, Medial/Lateral OA);
    # PF OA has a space but no medial/lateral distinction.
    spaced = [label for label in LABELS if " " in label]
    assert spaced == [
        "Medial Meniscus",
        "Lateral Meniscus",
        "Medial OA",
        "Lateral OA",
        "PF OA",
    ], spaced


def test_the_four_laterality_columns_are_present():
    """A third of the score depends on medial-versus-lateral being right, and
    the keyword floor scored zero true positives on two of these."""
    assert {"Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA"} <= set(LABELS)


def test_header_survives_a_csv_roundtrip():
    """The apostrophe must not induce quoting that changes the header bytes."""
    buf = io.StringIO()
    csv.writer(buf).writerow(SUBMISSION_COLUMNS)
    written = buf.getvalue().strip()
    assert written == EXPECTED_HEADER, f"csv module altered the header: {written!r}"
    assert next(csv.reader(io.StringIO(written))) == list(SUBMISSION_COLUMNS)


def test_labels_are_immutable():
    """A list would let a caller reorder the header in place."""
    assert isinstance(LABELS, tuple)
    assert isinstance(SUBMISSION_COLUMNS, tuple)


def test_label_groups_cover_every_label_exactly_once():
    grouped = [l for group in LABEL_GROUPS.values() for l in group]
    assert sorted(grouped) == sorted(LABELS)
    assert len(grouped) == len(set(grouped)), "a label appears in two groups"


def test_the_docstring_s_count_of_spaced_labels_is_right():
    """The module docstring names the two hostile spellings; both must be accurate.

    A wrong line in a docstring is worse than a missing one, because it gets
    trusted. This one had said "four" since the module
    was written, and the figure had already been copied into two other files and a
    task brief before anyone counted.
    """
    from kappa import constants

    spaced = [label for label in LABELS if " " in label]
    assert len(spaced) == 5, spaced
    assert "five labels contain a" in constants.__doc__
    assert len([label for label in LABELS if "'" in label]) == 1
