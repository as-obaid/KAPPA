"""Write and validate ``submission.csv``.

Why this exists: **the submission header is graded mechanically and the labels
are hostile to string handling.** ``Baker's`` carries an apostrophe and five of
the twelve labels contain a space, so the two obvious shortcuts -- splitting a
hard-coded header string on commas, or letting a CSV writer decide when to quote
-- each produce a file that looks right in a terminal and scores as an error.
The header therefore comes from ``kappa.constants.SUBMISSION_COLUMNS`` and from
nowhere else, is written with ``csv`` at ``QUOTE_MINIMAL``, and is read back off
disk and compared byte for byte before the writer returns.

    from kappa.inference.submission import write_submission, validate_submission

    write_submission(out_dir / "submission.csv", uids, scores)
    check = validate_submission(out_dir / "submission.csv", expected_uids=uids)
    assert check.ok, check.summary()

The line terminator is ``\\n``, not the ``\\r\\n`` that ``csv.writer`` defaults
to. That is not a preference: the constant-0.5 file that passed 7/7 contract
checks measured **470 bytes**, and 470 = 130 header bytes + 1 + 3 rows of (64
byte UID + 48 bytes of ``,0.5`` + 1). With CRLF the same file is 474 bytes, so
``\\r\\n`` would have been a silent change to a known-good artifact.

**No calibration and no probability averaging.** Macro ROC-AUC is rank-based and
threshold-free, so squeezing scores toward calibrated probabilities cannot move
it and averaging two models' probabilities is dominated by whichever model
happens to be more confident. ``rank_blend`` is the only supported ensemble and
it works per column.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from kappa.constants import ID_COLUMN, LABELS, N_LABELS, SUBMISSION_COLUMNS

__all__ = [
    "LINE_TERMINATOR",
    "N_FIELDS",
    "SubmissionCheck",
    "expected_header_bytes",
    "write_submission",
    "read_submission",
    "validate_submission",
    "rank_blend",
    "constant_submission",
]

#: Newline used on every row. See the module docstring for the byte arithmetic.
LINE_TERMINATOR = "\n"

#: One identifier column plus twelve labels. A row with any other field count is
#: malformed, and an unquoted apostrophe or space is exactly how that happens.
N_FIELDS = len(SUBMISSION_COLUMNS)


def expected_header_bytes() -> bytes:
    """The header line, encoded, derived from ``SUBMISSION_COLUMNS`` alone.

    Built through ``csv`` rather than ``",".join`` so that the expectation and
    the write share one quoting implementation: if a future label needs quoting,
    both change together instead of disagreeing.
    """
    import io

    buffer = io.StringIO()
    writer = csv.writer(buffer, quoting=csv.QUOTE_MINIMAL, lineterminator=LINE_TERMINATOR)
    writer.writerow(SUBMISSION_COLUMNS)
    return buffer.getvalue().encode("utf-8")


def _as_scores(scores) -> np.ndarray:
    """Coerce predictions to a finite ``(N, 12)`` float array, or explain why not."""
    array = np.asarray(scores, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] != N_LABELS:
        raise ValueError(
            f"scores must be (N, {N_LABELS}) to match the twelve labels, got {array.shape}"
        )
    if not np.isfinite(array).all():
        # NaN in a submission is not a degraded score, it is a rejected file, and
        # a masked label that never got a prediction is the way it arrives.
        bad = np.argwhere(~np.isfinite(array))
        first = tuple(bad[0])
        raise ValueError(
            f"{len(bad)} non-finite score(s); first at row {first[0]}, "
            f"label {LABELS[first[1]]!r}. Every study needs a finite score for all "
            f"{N_LABELS} labels -- masking applies to the loss, never to the output."
        )
    return array


def write_submission(
    path: str | Path,
    study_uids: Sequence[str],
    scores,
    *,
    float_format: str = "%.6f",
) -> Path:
    """Write ``submission.csv`` and verify its header byte for byte.

    ``scores`` is ``(N, 12)`` in ``LABELS`` order. The header check re-reads the
    file rather than trusting the writer, because the failure mode being guarded
    against -- a BOM, a quoted header, CRLF from a different writer -- is
    invisible on the object that was passed in.
    """
    path = Path(path)
    array = _as_scores(scores)
    uids = [str(uid) for uid in study_uids]
    if len(uids) != len(array):
        raise ValueError(f"{len(uids)} study UIDs but {len(array)} score rows")
    if len(set(uids)) != len(uids):
        raise ValueError("duplicate StudyInstanceUID; the leaderboard joins on this column")

    path.parent.mkdir(parents=True, exist_ok=True)
    # newline="" is required: without it csv's terminator is translated again by
    # the text layer on some platforms and the byte count stops being stable.
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, quoting=csv.QUOTE_MINIMAL, lineterminator=LINE_TERMINATOR)
        writer.writerow(SUBMISSION_COLUMNS)
        for uid, row in zip(uids, array):
            writer.writerow([uid, *(float_format % value for value in row)])

    with open(path, "rb") as handle:
        written = handle.readline()
    expected = expected_header_bytes()
    if written != expected:
        raise AssertionError(
            "submission header does not match SUBMISSION_COLUMNS byte for byte:\n"
            f"  wrote    {written!r}\n  expected {expected!r}"
        )
    return path


def read_submission(path: str | Path) -> tuple[list[str], list[str], np.ndarray]:
    """Return ``(header, uids, scores)`` exactly as the file states them."""
    path = Path(path)
    with open(path, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise ValueError(f"{path} is empty")
    header, *body = rows
    uids = [row[0] for row in body if row]
    values = np.array(
        [[float(cell) for cell in row[1:]] for row in body if row and len(row) == N_FIELDS],
        dtype=np.float64,
    ) if body else np.zeros((0, N_LABELS))
    return header, uids, values


@dataclass(frozen=True)
class SubmissionCheck:
    """Outcome of the seven contract checks, one entry each.

    Kept as named results rather than a bool so "7/7" means something. A check
    whose reference data was not supplied reports ``None`` -- not checked -- and
    is counted separately, because a validator that scores an unrun check as a
    pass is worse than no validator.
    """

    path: Path
    results: dict[str, bool | None]
    notes: dict[str, str]
    n_rows: int
    n_bytes: int

    @property
    def ok(self) -> bool:
        """True only if no check failed. Unchecked entries do not count as passes."""
        return all(value is not False for value in self.results.values())

    @property
    def n_passed(self) -> int:
        return sum(1 for value in self.results.values() if value is True)

    @property
    def n_checks(self) -> int:
        return len(self.results)

    def summary(self) -> str:
        lines = [f"{self.path}  {self.n_bytes} bytes, {self.n_rows} rows"]
        for name, value in self.results.items():
            mark = {True: "PASS", False: "FAIL", None: "----"}[value]
            note = self.notes.get(name, "")
            lines.append(f"  [{mark}] {name}" + (f" -- {note}" if note else ""))
        lines.append(f"  {self.n_passed}/{self.n_checks} checks passed")
        return "\n".join(lines)


def validate_submission(
    path: str | Path, *, expected_uids: Sequence[str] | None = None
) -> SubmissionCheck:
    """Run the seven submission contract checks against a file on disk.

    The same seven that a constant-0.5 file passed at 470 bytes: header exact,
    apostrophe intact, row count, UID order, field count, values finite, UIDs
    unique. ``expected_uids`` comes from ``test.csv``; without it the two
    order-and-count checks report as not checked rather than as passes.
    """
    path = Path(path)
    raw = path.read_bytes()
    header, uids, values = read_submission(path)
    results: dict[str, bool | None] = {}
    notes: dict[str, str] = {}

    header_line = raw.split(b"\n", 1)[0] + b"\n"
    results["header exact"] = header_line == expected_header_bytes()
    if not results["header exact"]:
        notes["header exact"] = f"{header_line!r}"

    results["Baker's apostrophe intact"] = "Baker's" in header
    if not results["Baker's apostrophe intact"]:
        notes["Baker's apostrophe intact"] = f"header field 10 is {header[10:11]}"

    if expected_uids is None:
        results["row count matches test.csv"] = None
        results["UID order matches test.csv"] = None
        notes["row count matches test.csv"] = "no expected_uids supplied"
        notes["UID order matches test.csv"] = "no expected_uids supplied"
    else:
        wanted = [str(uid) for uid in expected_uids]
        results["row count matches test.csv"] = len(uids) == len(wanted)
        notes["row count matches test.csv"] = f"{len(uids)} rows vs {len(wanted)} expected"
        results["UID order matches test.csv"] = uids == wanted

    with open(path, "r", encoding="utf-8", newline="") as handle:
        widths = {len(row) for row in csv.reader(handle) if row}
    results["13 fields per row"] = widths == {N_FIELDS}
    notes["13 fields per row"] = f"field counts seen: {sorted(widths)}"

    results["values finite"] = bool(values.size) and bool(np.isfinite(values).all())
    results["UIDs unique"] = len(set(uids)) == len(uids)

    return SubmissionCheck(
        path=path, results=results, notes=notes, n_rows=len(uids), n_bytes=len(raw)
    )


def rank_blend(score_sets: Iterable[np.ndarray]) -> np.ndarray:
    """Combine several models' predictions by per-column rank normalisation.

    The only supported ensemble. Averaging probabilities lets the more confident
    model dominate a metric that only reads order, and calibration cannot move a
    rank-based metric at all. Ranks are normalised to ``[0, 1]`` per label
    column, so each of the twelve -- which contribute 1/12 each -- is blended on
    its own scale.

    Delegates to ``kappa.evaluation.metrics.rank_normalise`` so that one
    implementation of the rank transform exists in the repository.
    """
    from kappa.evaluation.metrics import rank_normalise

    stack = [rank_normalise(np.asarray(scores, dtype=np.float64)) for scores in score_sets]
    if not stack:
        raise ValueError("rank_blend needs at least one set of scores")
    shapes = {array.shape for array in stack}
    if len(shapes) != 1:
        raise ValueError(f"score sets disagree on shape: {sorted(shapes)}")
    return np.mean(stack, axis=0)


def constant_submission(path: str | Path, study_uids: Sequence[str], value: float = 0.5) -> Path:
    """Write the all-``value`` baseline. Useful as a plumbing test end to end."""
    scores = np.full((len(study_uids), N_LABELS), float(value))
    return write_submission(path, study_uids, scores, float_format="%.1f")


if __name__ == "__main__":
    import sys

    for _path in sys.argv[1:]:
        print(validate_submission(_path).summary())
        print(f"{ID_COLUMN} column and {N_LABELS} labels expected")
