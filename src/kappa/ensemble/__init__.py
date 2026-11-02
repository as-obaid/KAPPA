"""Per-column rank fusion over saved predictions, aligned on ``StudyInstanceUID``.

Why this exists: **combining members is two decisions, and the obvious answer to
each is the wrong one.**

**Decision one: what to average.** Averaging probabilities is dominated by
whichever member happens to have the widest spread, so it weights members by their
confidence calibration -- a quantity macro ROC-AUC never reads. Calibrating first
does not repair that; ROC-AUC depends only on the order of the scores inside a
column, so Platt scaling, isotonic regression and a temperature all leave every
number here bit-identical while consuming a fold to fit. The one supported blend is
per-column rank normalisation, and it is not re-implemented here: every path goes
through :func:`kappa.evaluation.metrics.rank_average`, so one rank transform exists
in this repository and cannot disagree with itself.

**Decision two: how to line members up.** Folds run as parallel SLURM jobs, so each
member's rows arrive in whatever order its own dataset iteration produced. Two
members whose study order differs by one row blend one study's rank with another's,
return a full ``(n, 12)`` matrix, and score plausibly -- there is no shape error and
no NaN to notice. So nothing here blends by position. :func:`align_members`
reindexes every member onto a shared ``StudyInstanceUID`` order and **raises on a
study-set mismatch rather than inner-joining**: a member missing forty studies is a
run that failed, and an inner join would silently score the blend on the subset it
survived, which is this project's canonical shape of wrong answer.

    from kappa.ensemble import Member, blend, read_oof, score_blend

    tables = [read_oof(p) for p in sorted(run_dir.glob("oof_seed*.parquet"))]
    result = blend([t.member for t in tables])
    print(score_blend(result, tables[0]).table())
    result.to_submission(out_dir / "submission.csv", expected_uids=sample_uids)

**All twelve columns, always, in ``LABELS`` order.** Each label is 1/12 of the
score whether or not any member is good at it, so there is no interface here that
carries a subset. Weights may be given per column -- phase 09's deliverable is a
column-to-member-set mapping, not one global blend, and a weight of zero is how a
member is dropped from a column -- but the matrix is always ``(n_members, 12)`` and
:func:`column_weights` refuses a selection that forgets a label.

**The submission writer is not re-implemented.** ``Baker's`` carries an apostrophe
and five of the twelve labels contain a space, so the header comes from
``kappa.constants.SUBMISSION_COLUMNS`` through
:func:`kappa.inference.submission.write_submission` and from nowhere else.

Two import constraints hold and both are load-bearing. **Nothing here may reach
``kappa.labels``** at import time or at call time: submissions are image-only
(test studies carry no report), so a report-derived import is not a slow path but a notebook that
cannot run. And nothing here imports ``torch`` at module scope, so blending and
scoring still work where the ``train`` extra is not installed -- which is why the
submission writer is imported inside :meth:`Blend.to_submission` rather than at the
top of this file. See the note there.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from kappa.constants import ID_COLUMN, LABELS, N_LABELS, SUBMISSION_COLUMNS
from kappa.evaluation.metrics import AucResult, macro_roc_auc, rank_average

__all__ = [
    "MASK_SUFFIX",
    "OOF_COLUMNS",
    "TRUE_SUFFIX",
    "Blend",
    "Member",
    "OofTable",
    "align_members",
    "blend",
    "column_weights",
    "read_oof",
    "score_blend",
]

#: Suffixes in the out-of-fold table. Mirrored from ``kappa.training.loop`` rather
#: than imported from it, exactly as ``kappa.inference.predict`` mirrors
#: ``NOTEBOOK_LIMIT_SECONDS``: importing the training loop pulls ``torch``, and
#: blending saved artifacts must work on a machine where the ``train`` extra was
#: never installed. ``tests/test_ensemble.py`` pins these against the loop's own
#: definition, so the copy cannot drift without a test failing.
TRUE_SUFFIX = "__true"
MASK_SUFFIX = "__mask"

#: The out-of-fold table's schema: ids, then scores, then truth, then the mask the
#: truth was scored under. The mask travels with the scores because a score column
#: re-joined to a *different* mask later re-scores to a different macro, and
#: neither number announces which mask produced it.
OOF_COLUMNS: tuple[str, ...] = (
    ID_COLUMN,
    "fold",
    "seed",
    *LABELS,
    *(f"{name}{TRUE_SUFFIX}" for name in LABELS),
    *(f"{name}{MASK_SUFFIX}" for name in LABELS),
)


def _duplicates(values: Sequence[str]) -> list[str]:
    """The values that appear more than once, in first-seen order."""
    counts = Counter(values)
    seen: list[str] = []
    for value in values:
        if counts[value] > 1 and value not in seen:
            seen.append(value)
    return seen


@dataclass(frozen=True)
class Member:
    """One ensemble member: a name and a ``(n_studies, 12)`` score matrix by study.

    The identifiers are carried *with* the scores rather than alongside them,
    because every mis-alignment this module exists to prevent begins with a bare
    array whose row order is remembered by a human. A ``Member`` cannot be built
    without saying which study each row is.

    Scores are required to be finite. A NaN has no rank at all, and an infinity
    takes the top rank of its column for free, so either one produces a blend that
    is wrong only in the ordering -- the single property this metric reads.
    """

    name: str
    study_uids: tuple[str, ...]
    scores: np.ndarray

    def __post_init__(self) -> None:
        uids = tuple(str(uid) for uid in self.study_uids)
        scores = np.asarray(self.scores, dtype=np.float64)
        if scores.ndim != 2 or scores.shape[1] != N_LABELS:
            raise ValueError(
                f"member {self.name!r} has scores of shape {scores.shape}; every interface "
                f"here carries all {N_LABELS} labels in LABELS order, because each one is "
                "1/12 of the macro whether or not this member is any good at it"
            )
        if len(uids) != scores.shape[0]:
            raise ValueError(
                f"member {self.name!r}: {len(uids)} study UIDs but {scores.shape[0]} score rows"
            )
        repeated = _duplicates(uids)
        if repeated:
            raise ValueError(
                f"member {self.name!r} lists {len(repeated)} study UID(s) more than once, "
                f"e.g. {repeated[:3]}. Alignment is a lookup by identifier, so a repeated "
                "UID makes 'the row for this study' ambiguous and one of the two rows is "
                "silently discarded."
            )
        if not np.isfinite(scores).all():
            bad = np.argwhere(~np.isfinite(scores))
            row, column = (int(bad[0][0]), int(bad[0][1]))
            raise ValueError(
                f"member {self.name!r} has {len(bad)} non-finite score(s); first at study "
                f"{uids[row]!r}, label {LABELS[column]!r}. A NaN cannot be ranked and an "
                "infinity ranks first for free, so neither degrades the blend -- both "
                "reorder it."
            )
        object.__setattr__(self, "study_uids", uids)
        object.__setattr__(self, "scores", scores)

    @property
    def n_studies(self) -> int:
        return len(self.study_uids)

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, *, name: str) -> Member:
        """Read a member from a frame holding ``StudyInstanceUID`` and the 12 labels.

        Columns are selected **by name into ``LABELS`` order**, never by position.
        A frame whose label columns are in a different order -- a parquet round
        trip, a manual ``reindex``, a merge -- would otherwise transpose two labels
        against each other and still produce twelve plausible AUCs.
        """
        missing = [column for column in (ID_COLUMN, *LABELS) if column not in frame.columns]
        if missing:
            raise ValueError(
                f"member {name!r} frame is missing column(s) {missing}; a member carries all "
                f"{N_LABELS} labels plus {ID_COLUMN}"
            )
        return cls(
            name=name,
            study_uids=tuple(str(uid) for uid in frame[ID_COLUMN]),
            scores=frame[list(LABELS)].to_numpy(dtype=np.float64),
        )

    def reindex(self, study_uids: Sequence[str]) -> np.ndarray:
        """This member's scores in ``study_uids`` order, or a ``ValueError`` naming the gap.

        Studies this member holds but ``study_uids`` does not are dropped, which is
        what makes it usable for restricting a blend to a submission's row set. A
        study ``study_uids`` asks for and this member does not hold raises: filling
        it would fabricate a prediction, and dropping it would shorten the output
        against its caller's expectation.
        """
        wanted = [str(uid) for uid in study_uids]
        position = {uid: index for index, uid in enumerate(self.study_uids)}
        missing = [uid for uid in wanted if uid not in position]
        if missing:
            raise ValueError(
                f"member {self.name!r} has no score for {len(missing)} requested study/studies, "
                f"e.g. {missing[:3]}"
            )
        return self.scores[[position[uid] for uid in wanted]]


@dataclass(frozen=True)
class Blend:
    """The result of a blend: rank-space scores, and the recipe that produced them.

    ``weights`` is kept at full ``(n_members, 12)`` shape and column-normalised
    even when the caller passed one number per member, so that reading a blend back
    never requires knowing which of the two calling conventions was used. Per-column
    selection is legible in the same object: a zero in a column is a member that
    column does not use.
    """

    study_uids: tuple[str, ...]
    scores: np.ndarray
    members: tuple[str, ...]
    weights: np.ndarray

    @property
    def n_studies(self) -> int:
        return len(self.study_uids)

    @property
    def is_uniform(self) -> bool:
        """True when every column uses every member with the same weight.

        The uniform rank-average is phase 09's null: any weighting or per-column
        selection has to beat it by more than that column's fold spread before it
        has earned the degrees of freedom it spends. Worth being able to ask a
        blend which one it is.
        """
        return bool(np.allclose(self.weights, 1.0 / len(self.members)))

    def frame(self) -> pd.DataFrame:
        """The blend as a ``SUBMISSION_COLUMNS`` frame, for joining and inspection.

        Not a submission: this goes through ``pandas`` and therefore through
        ``pandas``' own quoting rules. Use :meth:`to_submission` for anything that
        will be uploaded.
        """
        data: dict[str, object] = {ID_COLUMN: list(self.study_uids)}
        for position, label in enumerate(LABELS):
            data[label] = self.scores[:, position]
        return pd.DataFrame(data, columns=list(SUBMISSION_COLUMNS))

    def to_submission(
        self, path: str | Path, *, expected_uids: Sequence[str] | None = None
    ) -> Path:
        """Write ``submission.csv`` through the one writer that exists.

        Delegates to :func:`kappa.inference.submission.write_submission`, which
        derives the header from ``SUBMISSION_COLUMNS``, quotes it, and reads the
        first line back off disk to check it byte for byte. Reconstructing that
        here would be a second header implementation, and the label set --
        ``Baker's`` with its apostrophe, five names with a space -- is precisely
        the input on which two implementations diverge without either one erroring.

        ``expected_uids`` is the sample submission's row order. Rows not named in
        it are dropped, which is what makes it safe to blend a table that also
        carries dev studies; a study named in it that the blend does not hold
        raises, because a short submission scores as an error rather than as a
        lower score.

        The import is deliberately inside the function. ``kappa/inference/
        __init__.py`` imports ``predict``, which imports ``torch``, so importing
        the writer at module scope would make ``import kappa.ensemble`` require the
        ``train`` extra -- and blending saved artifacts is exactly the step that has
        to run on a machine that never installed it.
        """
        from kappa.inference.submission import write_submission

        uids = list(self.study_uids)
        scores = self.scores
        if expected_uids is not None:
            wanted = [str(uid) for uid in expected_uids]
            position = {uid: index for index, uid in enumerate(uids)}
            missing = [uid for uid in wanted if uid not in position]
            if missing:
                raise ValueError(
                    f"{len(missing)} expected study UID(s) are not in the blend, e.g. "
                    f"{missing[:3]}. Every graded study needs a row."
                )
            scores = scores[[position[uid] for uid in wanted]]
            uids = wanted
        return write_submission(path, uids, scores)

    def report(self) -> str:
        """One block naming the members, their weights and the row count.

        Prints the per-label weight table whenever the weights are not the same in
        every column, because a per-column recipe summarised as a single number per
        member is a recipe nobody can reproduce.
        """
        lines = [f"blend of {len(self.members)} member(s) over {self.n_studies} studies"]
        column_constant = bool(np.allclose(self.weights, self.weights[:, :1]))
        if column_constant:
            for name, weight in zip(self.members, self.weights[:, 0]):
                lines.append(f"  {name:<28}{weight:>8.4f}")
        else:
            lines.append("  per-column weights (a zero is a member this column does not use)")
            header = "  " + f"{'label':<18}" + "".join(f"{name[:11]:>12}" for name in self.members)
            lines.append(header)
            for position, label in enumerate(LABELS):
                row = "".join(f"{weight:>12.4f}" for weight in self.weights[:, position])
                lines.append(f"  {label:<18}{row}")
        return "\n".join(lines)


def align_members(
    members: Sequence[Member], *, order: Sequence[str] | None = None
) -> tuple[tuple[str, ...], list[np.ndarray]]:
    """A shared study order, and every member's scores reindexed onto it.

    **This is the function that makes a blend mean what it says.** Members are
    written by separate jobs; nothing guarantees their row order agrees, and a
    positional combination of two members whose order differs is a full-shape,
    finite, plausible matrix in which one study's rank has been averaged with
    another's. So the scores are looked up by identifier even when the orders
    happen to match, and the caller pays nothing for the case where they do.

    A **study-set mismatch raises**. The tempting alternative -- intersect the sets
    and blend what both members have -- turns a member that crashed on forty
    studies into a blend that scores well on the studies it survived, and reports
    no error at all. The exception names both directions, because "member B is
    missing rows" and "member B has rows nobody else does" need different fixes.

    ``order`` pins the output order, e.g. to a fold table or a sample submission.
    It must name exactly the shared study set: a missing or unknown UID raises
    rather than quietly trimming the blend.
    """
    if not members:
        raise ValueError("align_members needs at least one member")
    repeated = _duplicates([member.name for member in members])
    if repeated:
        raise ValueError(
            f"member name(s) {repeated} appear more than once. Weights and per-column "
            "selections are keyed by name, so a duplicate makes a recipe ambiguous and "
            "silently applies one member's weight to the other."
        )

    reference = members[0]
    reference_set = set(reference.study_uids)
    for member in members[1:]:
        other = set(member.study_uids)
        if other != reference_set:
            only_reference = sorted(reference_set - other)
            only_other = sorted(other - reference_set)
            raise ValueError(
                f"members {reference.name!r} and {member.name!r} disagree on their study set: "
                f"{len(only_reference)} only in {reference.name!r} (e.g. {only_reference[:3]}), "
                f"{len(only_other)} only in {member.name!r} (e.g. {only_other[:3]}). "
                "Blending the intersection would score the ensemble on the studies every "
                "member happened to survive, so this raises instead."
            )

    if order is None:
        resolved = reference.study_uids
    else:
        resolved = tuple(str(uid) for uid in order)
        repeated_order = _duplicates(resolved)
        if repeated_order:
            raise ValueError(f"order repeats study UID(s), e.g. {repeated_order[:3]}")
        wanted = set(resolved)
        if wanted != reference_set:
            unknown = sorted(wanted - reference_set)
            absent = sorted(reference_set - wanted)
            raise ValueError(
                f"order names {len(unknown)} study/studies no member scored (e.g. "
                f"{unknown[:3]}) and omits {len(absent)} that every member scored (e.g. "
                f"{absent[:3]}); it must name the shared study set exactly"
            )

    return resolved, [member.reindex(resolved) for member in members]


def _resolve_weights(weights, n_members: int) -> np.ndarray:
    """Coerce a weight argument to a validated ``(n_members, 12)``, still unnormalised.

    Both calling conventions land in the same shape so that everything downstream
    -- the blend loop, :attr:`Blend.weights`, :meth:`Blend.report` -- has one case
    to handle.

    Deliberately **not** normalised here. ``rank_average`` divides by the sum of the
    weights it is handed, so pre-dividing would make ``blend`` accumulate
    ``(w/s)*x`` and then divide by ``sum(w/s)``, which is a different sequence of
    floating-point operations from ``sum(w*x)/sum(w)`` and lands a few ULP away.
    The uniform blend is asserted to be *bit-identical* to a single
    ``rank_average`` call, and that assertion is only worth making if it is exact.
    :func:`blend` normalises once, at the end, for the copy it stores on the result.
    """
    if weights is None:
        matrix = np.ones((n_members, N_LABELS), dtype=np.float64)
    else:
        matrix = np.asarray(weights, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = np.repeat(matrix.reshape(-1, 1), N_LABELS, axis=1)
        if matrix.shape != (n_members, N_LABELS):
            raise ValueError(
                f"weights must be one per member ({n_members},) or one per member and label "
                f"({n_members}, {N_LABELS}), got {np.shape(weights)}"
            )
    if not np.isfinite(matrix).all():
        raise ValueError("weights contain a non-finite value")
    if (matrix < 0.0).any():
        # rank_average only checks that the weights sum to something positive, so a
        # sign typo cancels against a larger positive and returns a blend in which
        # one member's column is ordered backwards -- finite, in range, and wrong.
        raise ValueError(
            "weights contain a negative value. A negative weight reverses that member's "
            "ranking inside the column rather than removing it; use 0.0 to drop a member "
            "from a column."
        )
    totals = matrix.sum(axis=0)
    empty = [LABELS[position] for position in np.flatnonzero(totals <= 0.0)]
    if empty:
        raise ValueError(
            f"no member has positive weight on label(s) {empty}. Every label is 1/12 of the "
            "macro, so a column with no members is a twelfth of the score left unscored, "
            "not a column to fill with a constant."
        )
    return matrix


def blend(
    members: Sequence[Member],
    weights: Sequence[float] | np.ndarray | None = None,
    *,
    order: Sequence[str] | None = None,
) -> Blend:
    """Fuse members by per-column rank normalisation, aligned on study identifier.

    The whole supported ensemble, in one call. ``weights`` is either one number per
    member or a ``(n_members, 12)`` matrix; ``None`` is the uniform rank-average,
    which phase 09 fixes as the null that any weighting has to beat by more than a
    column's fold standard deviation before it is worth its degrees of freedom.

    The blend is performed **one column at a time**, on ``(n, 1)`` slices, through
    :func:`kappa.evaluation.metrics.rank_average`. That is not an optimisation and
    not a second implementation: the twelve labels are twelve independent rankings,
    and per-column weights only mean anything if the blend is actually carried out a
    column at a time. On uniform weights it is bit-identical to one ``rank_average``
    call over the whole matrices, which ``tests/test_ensemble.py`` asserts.

    The output is in rank space, in ``(0, 1)``. That is a valid submission -- the
    metric reads only order -- and it is deliberately not mapped back to anything
    resembling a probability, because no such mapping exists and inventing one is
    calibration by another name.
    """
    resolved_order, aligned = align_members(members, order=order)
    resolved_weights = _resolve_weights(weights, len(members))

    blended = np.empty((len(resolved_order), N_LABELS), dtype=np.float64)
    for position in range(N_LABELS):
        column = rank_average(
            [scores[:, position : position + 1] for scores in aligned],
            weights=resolved_weights[:, position],
        )
        blended[:, position] = column[:, 0]

    return Blend(
        study_uids=resolved_order,
        scores=blended,
        members=tuple(member.name for member in members),
        # Normalised only for the stored copy: a recipe is read as shares of a
        # column, and the raw numbers a caller passed carry a scale that means
        # nothing once ``rank_average`` has divided by their sum.
        weights=resolved_weights / resolved_weights.sum(axis=0),
    )


def column_weights(
    members: Sequence[Member | str],
    selection: Mapping[str, Sequence[str]],
) -> np.ndarray:
    """Turn a column-to-member-set mapping into the weight matrix :func:`blend` takes.

    Phase 09's deliverable is a mapping from label to member set, not one global
    blend, because the twelve columns are scored independently and a model that is
    useless on eleven of them can still win the twelfth. This is the translation
    from that recipe into weights: 1.0 for a selected member, 0.0 otherwise,
    normalised per column by :func:`blend`.

    **Every one of the twelve labels must appear in ``selection``.** Defaulting an
    absent label to "all members" would let a recipe that forgot a column produce a
    complete-looking twelve-column blend and a plausible macro, which is the same
    class of failure as a macro averaged over eleven labels and still called macro.
    """
    names = [member if isinstance(member, str) else member.name for member in members]
    repeated = _duplicates(names)
    if repeated:
        raise ValueError(f"member name(s) {repeated} appear more than once")
    position = {name: index for index, name in enumerate(names)}

    missing = [label for label in LABELS if label not in selection]
    if missing:
        raise ValueError(
            f"selection has no entry for label(s) {missing}. Every label is 1/12 of the "
            "macro, so a column with no recipe is an omission rather than a default."
        )
    unknown_labels = [label for label in selection if label not in LABELS]
    if unknown_labels:
        raise ValueError(f"selection names label(s) {unknown_labels} that are not in LABELS")

    matrix = np.zeros((len(names), N_LABELS), dtype=np.float64)
    for index, label in enumerate(LABELS):
        chosen = list(selection[label])
        if not chosen:
            raise ValueError(f"selection for label {label!r} is empty; a column needs a member")
        unknown = [name for name in chosen if name not in position]
        if unknown:
            raise ValueError(
                f"selection for label {label!r} names member(s) {unknown}, which are not "
                f"among {names}"
            )
        for name in chosen:
            matrix[position[name], index] = 1.0
    return matrix


@dataclass(frozen=True)
class OofTable:
    """One run's out-of-fold artifact: its scores, and the truth and mask it was scored under.

    The truth and the mask are carried here rather than re-joined from a label table
    at blend time, for the reason the training loop writes them: a score column
    re-scored against a *different* mask returns a different macro, and neither
    number says which mask produced it. Reading them from the same file the scores
    came from makes that impossible by construction.
    """

    name: str
    study_uids: tuple[str, ...]
    scores: np.ndarray
    truth: np.ndarray
    mask: np.ndarray
    fold: np.ndarray
    seed: np.ndarray

    @property
    def n_studies(self) -> int:
        return len(self.study_uids)

    @property
    def member(self) -> Member:
        """This run as an ensemble member, named after the artifact it came from."""
        return Member(name=self.name, study_uids=self.study_uids, scores=self.scores)

    def aligned_truth(self, study_uids: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        """``(truth, mask)`` reindexed onto ``study_uids``, by identifier.

        By identifier, not by position, for the same reason :func:`align_members`
        exists: a blend is in the aligned order and this table is in its own file's
        order, and lining the two up positionally re-introduces the mis-alignment
        one step after it was removed -- this time between a prediction and the
        label it is scored against, where it looks like a mediocre model.
        """
        wanted = [str(uid) for uid in study_uids]
        position = {uid: index for index, uid in enumerate(self.study_uids)}
        missing = [uid for uid in wanted if uid not in position]
        if missing:
            raise ValueError(
                f"OOF table {self.name!r} has no truth for {len(missing)} study/studies, "
                f"e.g. {missing[:3]}; it cannot score this blend"
            )
        rows = [position[uid] for uid in wanted]
        return self.truth[rows], self.mask[rows]


def read_oof(path: str | Path, *, name: str | None = None) -> OofTable:
    """Read an ``oof_seed*.parquet`` written by :func:`kappa.training.run_experiment`.

    The schema is checked against :data:`OOF_COLUMNS` by *name*, and columns are
    then pulled in ``LABELS`` order. Reading by name is what makes a column
    reordering harmless; a column that is genuinely absent is not harmless, and a
    missing ``__mask`` column in particular would leave the truth looking like
    4,407 studies of real labels when 98.7% of the cells are unknown.

    A repeated ``StudyInstanceUID`` raises. The training loop already refuses to
    emit one, so seeing one here means two runs' artifacts were concatenated -- a
    file in which every study appears twice blends and scores without complaint and
    reports a macro over a doubled, correlated sample.
    """
    path = Path(path)
    frame = pd.read_parquet(path)
    found = tuple(str(column) for column in frame.columns)
    missing = [column for column in OOF_COLUMNS if column not in found]
    extra = [column for column in found if column not in OOF_COLUMNS]
    if missing or extra:
        raise ValueError(
            f"{path} is not an OOF table: missing {missing}, unexpected {extra}. The "
            f"contract is {OOF_COLUMNS}."
        )

    uids = tuple(str(uid) for uid in frame[ID_COLUMN])
    repeated = _duplicates(uids)
    if repeated:
        raise ValueError(
            f"{path} lists {len(repeated)} study UID(s) more than once, e.g. {repeated[:3]}. "
            "Out-of-fold rows are one per study by construction, so this is two runs' "
            "artifacts concatenated rather than one run's."
        )

    return OofTable(
        name=name if name is not None else path.stem,
        study_uids=uids,
        scores=frame[list(LABELS)].to_numpy(dtype=np.float64),
        truth=frame[[f"{label}{TRUE_SUFFIX}" for label in LABELS]].to_numpy(dtype=np.float64),
        mask=frame[[f"{label}{MASK_SUFFIX}" for label in LABELS]].to_numpy(dtype=bool),
        fold=frame["fold"].to_numpy(dtype=np.int64),
        seed=frame["seed"].to_numpy(dtype=np.int64),
    )


def score_blend(result: Blend, table: OofTable) -> AucResult:
    """Macro ROC-AUC for a blend, against the truth and mask of an OOF artifact.

    Takes the table rather than two bare arrays on purpose: the truth is in the
    file's row order and the blend is in the aligned order, so handing
    :func:`kappa.evaluation.metrics.macro_roc_auc` a truth matrix and a score matrix
    that were never lined up by identifier is the one remaining way to get a
    finite, plausible, wrong number out of this module. :meth:`OofTable.aligned_truth`
    closes it.

    Read ``skipped`` before quoting ``macro``: a label with one class present in
    this split has no AUC, and the returned mean is over the labels that did.
    """
    truth, mask = table.aligned_truth(result.study_uids)
    return macro_roc_auc(truth, result.scores, mask)
