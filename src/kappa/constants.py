"""Competition constants for KAPPA.

Single source of truth for the twelve target labels and the submission header.
Every write of a submission file, every fold table and every metrics call takes
its column order from here rather than rebuilding it, because the header is
graded mechanically: a misspelled or reordered column scores as an error and
burns one of a limited number of submission slots.

    from kappa.constants import LABELS, SUBMISSION_COLUMNS

    assert list(submission.columns) == list(SUBMISSION_COLUMNS)

Two names are hostile to naive string handling and are the reason this module
exists at all: ``Baker's`` contains an apostrophe, and five labels contain a
space. Never build this list by splitting a string on punctuation, and always
quote on write.
"""

from __future__ import annotations

__all__ = [
    "ID_COLUMN",
    "LABELS",
    "N_LABELS",
    "SUBMISSION_COLUMNS",
    "LABEL_GROUPS",
]

ID_COLUMN = "StudyInstanceUID"

#: The twelve targets, in the exact order the submission header requires.
#: Scored as macro ROC-AUC, so each contributes exactly 1/12 regardless of how
#: rare it is, and each column is scored independently of the others.
LABELS = (
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
)

N_LABELS = len(LABELS)

#: Full submission header: the study identifier followed by the twelve labels.
SUBMISSION_COLUMNS = (ID_COLUMN, *LABELS)

#: Clinical grouping. Descriptive only -- the metric scores all twelve columns
#: independently, so this must never drive model selection or fold construction.
#: Useful for reading per-label result tables, and nothing else.
LABEL_GROUPS = {
    "ligament": ("ACL", "MCL"),
    "meniscus": ("Medial Meniscus", "Lateral Meniscus"),
    "osteoarthritis": ("Medial OA", "Lateral OA", "PF OA"),
    "synovium": ("Effusion", "Synovitis", "Baker's"),
    "bone": ("Contusion", "Fracture"),
}
