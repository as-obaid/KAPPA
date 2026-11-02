"""Five folds that do not leak, over twelve correlated labels in four languages.

Why this exists: the obvious fold split is wrong in three independent ways on
this dataset, and every one of them inflates cross-validation while leaving the
private leaderboard untouched. The failure is only visible in late October, when
nothing can be changed.

**1. Twelve labels, stratified one at a time, is not stratification.** Label
prevalences here run from 15.5% up and the labels are correlated, so splitting on
one and hoping is how a fold ends up with zero positives for a rare label. A fold
with no positives contributes ``nan`` to a macro-AUC over twelve columns, and the
mean of the remaining eleven is not the metric. This module runs **iterative
multi-label stratification**: it repeatedly takes the rarest label still
unplaced, and gives its next group to whichever fold is furthest behind on that
label. The rarest label is served first, every time, because it is the one that
runs out.

**2. Grouping.** No patient may appear in two folds. Whether patients
recur at all is **not settled**: over the 558 locally staged studies every
``PatientID`` is distinct, but the corpus de-identifier plausibly re-keys per
study, which erases recurrence rather than measuring it (see
``kappa.data.manifest.PatientLinkage``). So the grouping key is a **parameter**,
not a constant. The default groups on ``patient_id``, which is the safe reading
whichever way that question lands, and ``group_keys`` takes any number of columns: groups
are the connected components over all of them, so grouping on two keys at once
cannot leak through either.

**3. Language.** This is the under-appreciated one. Language implies institution
implies scanner implies image appearance. The derived labels are strongly
language-correlated -- measured spreads: Effusion **16% (Turkish) to 72%
(Bulgarian)**, Medial Meniscus 33-82%, PF OA 16-65%, Lateral OA 2-29% -- so a
CNN can satisfy a language-correlated label from scanner signature instead of
pathology, and score well doing it. **No robust loss repairs this**: a systematic
per-language polarity flip is not i.i.d. noise, which is the assumption every
robust-loss method makes. It has to be fixed at the label source. What folds can
do is make it *visible*, so this module stratifies on the language bucket as well
and reports per-fold prevalence per label, per fold.

Two fold assignments are emitted, not one, because which is right is genuinely
undecided. ``fold`` groups by patient only, leaving scanners spread across folds;
``fold_scanner`` additionally keeps every scanner inside one fold. A published
header-only probe on this data scored macro AUC 0.6516 with random folds against
0.5981 with scanner-grouped folds, so a random-fold score carries roughly 0.05 of
site memorisation -- but the hidden test set is drawn from the same sites, so
some of that signal may legitimately transfer. Report both. Do not quietly pick.

    from kappa.data.folds import build_folds, fold_report, write_folds

    folds = build_folds(study_table)          # study_uid, labels, language, groups
    print(fold_report(folds, study_table).table())
    write_folds(folds, folds_dir() / "folds_v1.csv")

**folds_v2 adds a frozen held-out test set, and folds_v1 stays as it is.** Five
folds over all 4,407 studies give model selection and no honest final number:
every backbone, every augmentation, every threshold and every "that fold looked
unlucky, try seed 2" is chosen against those same five validation sets, so by
late October the CV mean is a number the choices were fitted to.
:func:`build_folds_v2` cuts **15% -- 661 of 4,407 studies -- into ``split ==
"test"`` carrying ``fold == -1``**, and splits the remaining 85% (3,746) into the
five folds.

    folds = build_folds_v2(study_table, keep_ids=anchor_uids)
    print(holdout_report(folds, study_table).table())        # dev against test
    dev = folds.loc[dev_mask(folds)]                         # the CV pool alone
    print(fold_report(dev, study_table, group_column="report_group").table())
    write_folds(folds, folds_dir() / "folds_v2.csv")

**The test set is not looked at until the pipeline is finished.** Not for a
sanity check, not to see whether a fold was unlucky, not once. This project has
already spent its only other held-out set on exactly that mistake: the 58
annotated anchor studies were read to measure the label derivation
at macro balanced accuracy 0.765, which demoted them from an unseen test set to a
regression check, permanently. There is no third set. ``keep_ids`` exists to keep
those 58 in ``dev`` for the same reason -- a test set holding studies that have
already been tuned against is not clean.

**What the holdout costs, stated rather than glossed:** 661 studies stop being
training data, so a fold trains on ~2,997 studies where folds_v1 gave it ~3,526.
That is the price of one trustworthy number, and it is worth paying once. Once
the pipeline is frozen, the *submission* models should be retrained on dev+test
together: the split is a measurement device, not a property of the final model.

**The test set is scanner-shared, deliberately.** The holdout groups on
``group_keys`` -- report content -- and never on ``scanner_fingerprint``, so the
same sites appear on both sides of it. The hidden Kaggle test set is presumably
drawn from the same 19 sites, so a scanner-disjoint local test would be
pessimistic and would answer a question nobody asked. The pessimistic bound is
reported separately, as the ``fold`` against ``fold_scanner`` CV gap -- and that
gap is an **upper bound** on site memorisation, not a measurement of it, because
``fold_scanner`` is degenerate on this corpus: 337 groups, the largest holding
327 studies, and the language buckets collapse across it.

**The output must be CSV.** ``.gitignore`` globs ``*.parquet`` under a ``data/*``
rule with a ``!data/folds/`` negation, so a parquet fold table is silently
untracked while looking saved. Fold assignments are the artifact reproducibility
depends on. :func:`write_folds` refuses any other suffix.
"""

from __future__ import annotations

import csv
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from kappa.constants import ID_COLUMN, LABELS

__all__ = [
    "N_FOLDS",
    "TEST_FRACTION",
    "TEST_FOLD",
    "SPLIT_VALUES",
    "FOLD_COLUMNS",
    "FOLD_COLUMNS_V2",
    "FOLD_SCHEMAS",
    "MASK_SUFFIX",
    "DEFAULT_GROUP_KEYS",
    "SCANNER_GROUP_KEYS",
    "check_fold_columns",
    "dev_mask",
    "group_components",
    "positive_matrix",
    "assign_folds",
    "assign_holdout",
    "build_folds",
    "build_folds_v2",
    "FoldReport",
    "fold_report",
    "holdout_report",
    "write_folds",
    "read_folds",
]

#: Five. Enough that each fold holds ~880 of 4,407 studies, which at the lowest
#: measured prevalence still leaves a usable number of positives per fold.
N_FOLDS = 5

#: Suffix of the per-label usability mask in the derived label table. Pinned
#: with the label table's own contract -- a cell whose mask is False has an
#: undefined value and must be counted as neither positive nor negative.
MASK_SUFFIX = "__mask"

#: Fold table schema. ``patient_group`` and ``scanner_group`` are integer
#: component ids, not the raw ``PatientID``: the fold table is tracked in git
#: while the competition data is not redistributable, and an integer is all a
#: reader needs to verify from the file alone that no group spans two folds.
FOLD_COLUMNS: tuple[str, ...] = (
    ID_COLUMN,
    "fold",
    "fold_scanner",
    "patient_group",
    "scanner_group",
    "language",
)

#: Fraction of the corpus held out as the frozen test set. 15% of 4,407 is 661
#: studies, which at the binding label -- Fracture, 7.5% -- is about 50 positives,
#: enough for an AUC that is not dominated by which single positive landed where.
TEST_FRACTION = 0.15

#: Fold index written for a row that is in no fold at all. Negative on purpose:
#: ``fold != k`` is the natural way to spell "the training side" and it silently
#: swallows a held-out row, so the sentinel has to be one that an explicit
#: ``>= 0`` check can reject. See :func:`dev_mask`.
#: How far below its proportional share a language bucket may fall in one bin
#: before the gate refuses the split. A bucket is compared against the share of
#: the corpus that bin holds, never against the other buckets: the buckets are
#: 3320/546/321/220, so equality between them is not the goal and testing for it
#: would reject every correct split.
LANGUAGE_TOLERANCE = 0.5

#: Below this expected count a bucket is only required to be non-empty, not
#: proportional. Three studies expected in a bin is a number rounding moves.
LANGUAGE_MIN_EXPECTED = 10.0

TEST_FOLD = -1

#: The only two values the v2 ``split`` column may take.
SPLIT_VALUES: tuple[str, ...] = ("dev", "test")

#: Fold table schema, v2: ``split`` and a frozen held-out test set on top of v1.
#:
#: ``patient_group`` becomes ``report_group``, which is a rename toward honesty
#: rather than cosmetics. The component ids in that column were *already* driven
#: entirely by report content: all 4,407 studies carry a distinct ``PatientID``
#: (measured over the full manifest), so ``patient_id`` contributes zero linking
#: edges to the connected components and every non-singleton component is one that
#: identical report text created. Calling the column ``patient_group`` invited the
#: reader to believe a patient grouping was in force. The grouping *keys* are
#: unchanged -- see :data:`DEFAULT_GROUP_KEYS` -- so ``patient_id`` still
#: strengthens the grouping by itself if the IDs ever turn out to mean something.
FOLD_COLUMNS_V2: tuple[str, ...] = (
    ID_COLUMN,
    "split",
    "fold",
    "fold_scanner",
    "report_group",
    "scanner_group",
    "language",
)

#: Every accepted on-disk schema, oldest first. Two, not one, and v1 is not
#: deprecated: ``data/folds/folds_v1.csv`` is a frozen artifact that published
#: numbers were produced against, so a reader that refuses it makes those numbers
#: unreproducible. Extend this tuple to add a schema; never edit an entry.
FOLD_SCHEMAS: tuple[tuple[str, ...], ...] = (FOLD_COLUMNS, FOLD_COLUMNS_V2)

#: Grouping for the ``fold`` column.
#:
#: **``patient_id`` does not link studies.** Measured over the full manifest (all
#: 4,407 studies): 4,407 studies carry a
#: ``PatientID`` and all 4,407 are distinct, but of the 52 identical-report groups,
#: **0 share a PatientID and all 52 span several** -- and 5 of those are the
#: discriminating kind (a report over 400 characters appearing in exactly two
#: studies, far too specific to be a coincidence of templating). Two studies with
#: the same long report and two different PatientIDs is proof the IDs are re-keyed
#: per study. So the IDs do not identify patients, recurrence is **not** ruled out,
#: and grouping on ``patient_id`` alone is a random split wearing a grouped split's
#: name: every group is a singleton, so the constraint binds nothing.
#:
#: ``report_group`` is what replaces it -- the content hash of the normalised report,
#: which links the 197 studies in those 52 groups that ``patient_id`` fails to link.
#: It is a floor on patient identity, not a measurement of it: it catches recurrence
#: that left identical text and misses recurrence that did not. That is strictly
#: better than a key measured to link nothing. ``patient_id`` is retained so the
#: grouping strengthens by itself if the IDs ever turn out to mean something.
DEFAULT_GROUP_KEYS: tuple[str, ...] = ("patient_id", "report_group")

#: Grouping for the ``fold_scanner`` column. All three keys, because a patient
#: scanned twice on two magnets would otherwise leak across a scanner boundary.
SCANNER_GROUP_KEYS: tuple[str, ...] = (
    "patient_id",
    "report_group",
    "scanner_fingerprint",
)

# Values that mean "this row carries no key", and so must not be linked to any
# other row. Merging every blank PatientID into one component would build a
# single giant group and make folds impossible to balance -- while still passing
# a no-overlap check, because one group in one fold never overlaps anything.
_BLANK = ("", "nan", "none", "unknown")


def _is_blank(value: object) -> bool:
    """True for the values that must not link two studies together."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return True
    return str(value).strip().casefold() in _BLANK


def check_fold_columns(
    columns: Iterable[str], *, what: str = "fold table"
) -> tuple[str, ...]:
    """The pinned schema ``columns`` matches, or a ``ValueError`` naming both.

    One function so that every reader, writer and consumer of a fold table agrees
    on what a fold table is. The alternative -- each site holding its own literal
    -- is how a v2 file passes :func:`write_folds` and is then rejected by the
    training loop three weeks later, on a cluster, in a job that has already
    queued for an hour.

    Column *order* is part of the contract, not just the set of names: the file is
    read back by column position in places, and a reordered header is a silent
    mismatch rather than a loud one.
    """
    found = tuple(str(column) for column in columns)
    for schema in FOLD_SCHEMAS:
        if found == schema:
            return schema
    raise ValueError(
        f"{what} has columns {found}, contract requires {FOLD_COLUMNS} (v1) "
        f"or {FOLD_COLUMNS_V2} (v2)"
    )


def dev_mask(folds: pd.DataFrame, *, column: str = "fold") -> np.ndarray:
    """Boolean per row: True where the row takes part in cross-validation.

    **This is the guard the frozen test set depends on.** ``train = uids[fold !=
    k]`` is the natural way to spell a fold's training side, and it is wrong the
    instant one row carries ``fold == -1``: every fold's training set then absorbs
    the entire held-out test set. Nothing raises, no shape changes, and the only
    symptom is a cross-validation score that is quietly too high -- discovered
    from the private leaderboard, when nothing can be changed. A row that does not
    participate must land on **neither** side, which needs a positive test for
    participation rather than an inequality against one fold index.

    Two conditions are checked and either one alone is disqualifying, because the
    schemas disagree about which exists: v1 has no ``split`` column and no
    negative index, v2 has both, and a table edited by hand may carry one without
    the other. A ``split`` of ``"test"`` with ``fold == 3`` is still out.
    """
    if column not in folds.columns:
        raise KeyError(
            f"fold table has no {column!r} column; it has {tuple(folds.columns)}"
        )
    mask = folds[column].astype(int).to_numpy() >= 0
    if "split" in folds.columns:
        mask &= folds["split"].astype(str).to_numpy() == "dev"
    return mask


def group_components(table: pd.DataFrame, keys: Sequence[str]) -> np.ndarray:
    """Connected components over one or more grouping columns, as integer ids.

    Two rows land in the same component when they share a non-blank value in
    *any* of ``keys``, transitively. That transitivity is the point: grouping on
    ``patient_id`` and ``scanner_fingerprint`` independently would still let a
    patient scanned on two magnets sit in two folds, and the leak would be
    invisible to a per-key overlap check.

    Ids are assigned in order of first appearance, so the numbering is stable
    for a stable row order and a re-run produces a byte-identical fold table.
    """
    missing = [key for key in keys if key not in table.columns]
    if missing:
        raise KeyError(f"grouping columns {missing} are not in the table")

    n = len(table)
    parent = list(range(n))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for key in keys:
        first_seen: dict[str, int] = {}
        for position, value in enumerate(table[key].to_numpy()):
            if _is_blank(value):
                continue
            token = str(value)
            if token in first_seen:
                union(first_seen[token], position)
            else:
                first_seen[token] = position

    labelled: dict[int, int] = {}
    out = np.empty(n, dtype=np.int64)
    for position in range(n):
        root = find(position)
        if root not in labelled:
            labelled[root] = len(labelled)
        out[position] = labelled[root]
    return out


def positive_matrix(
    table: pd.DataFrame,
    labels: Sequence[str] = LABELS,
    mask_suffix: str = MASK_SUFFIX,
) -> tuple[np.ndarray, np.ndarray]:
    """Split a derived label table into ``(positives, usable)`` boolean matrices.

    A masked cell is neither a positive nor a negative -- it is a cell nobody
    can score. Counting masked cells as negatives would make a label look
    commoner than it is and let a fold pass the "has a positive" gate on rows
    that never reach the loss. Labels with no mask column are treated as fully
    usable, which is what a synthetic or fully-observed table means.
    """
    n = len(table)
    positives = np.zeros((n, len(labels)), dtype=bool)
    usable = np.ones((n, len(labels)), dtype=bool)
    for column, label in enumerate(labels):
        if label not in table.columns:
            raise KeyError(f"label table is missing the {label!r} column")
        values = pd.to_numeric(table[label], errors="coerce").to_numpy(dtype=float)
        mask_column = f"{label}{mask_suffix}"
        if mask_column in table.columns:
            usable[:, column] = table[mask_column].to_numpy(dtype=bool)
        usable[:, column] &= ~np.isnan(values)
        positives[:, column] = usable[:, column] & (np.nan_to_num(values) > 0.5)
    return positives, usable


def _stratify_one_bucket(
    order: list[int],
    sizes: np.ndarray,
    counts: np.ndarray,
    desired_labels: np.ndarray,
    ratios: np.ndarray,
) -> np.ndarray:
    """Iterative stratification of one language bucket, in place on the desires.

    ``order`` lists global group ids in the sequence they are offered, and the
    returned array is parallel to ``order``, not indexed by group id.
    ``desired_labels`` is the *global* remaining label quota per fold and is
    carried across buckets, while the size quota is local to the bucket so that
    each bucket is itself split ``ratios``-wise. That is what makes the language
    stratification real rather than incidental.

    The label quota is compared **relative to each bin's share** whenever the
    shares are unequal. This was measured, and it is why this branch exists: an 85/15
    holdout over the four language buckets put **0 of 220 cyrillic and 4 of 321
    greek studies** in the test set. The raw quota is never tied between an 0.85
    bin and an 0.15 bin, so the size quota below never got to break a tie, the
    per-bucket size split silently stopped happening, and the dev bin won every
    group in both small buckets before the big one had run its global quota down.
    A test set that is 98% Latin passes every label gate and is not the corpus.
    Equal shares divide by a constant and so are left on the exact expression that
    produced ``folds_v1.csv`` -- a rescaling here is a rescaling of the 1e-9
    tolerance too, and this must stay bit-reproducible.
    """
    n_folds = len(ratios)
    equal_shares = bool(np.ptp(np.asarray(ratios, dtype=float)) == 0.0)
    assigned = np.full(len(order), -1, dtype=np.int64)
    desired_size = ratios * float(sizes[order].sum())
    remaining = counts[order].sum(axis=0).astype(float)
    unplaced = len(order)
    # One cursor per label into `order`, advanced monotonically, so finding the
    # next unassigned group carrying a label stays linear overall.
    cursor = [0] * counts.shape[1]

    while unplaced:
        candidates = np.flatnonzero(remaining > 0)
        if candidates.size == 0:
            for position in sorted(
                (p for p in range(len(order)) if assigned[p] < 0),
                key=lambda p: -sizes[order[p]],
            ):
                fold = int(np.argmax(desired_size))
                assigned[position] = fold
                desired_size[fold] -= sizes[order[position]]
            break

        label = int(candidates[np.argmin(remaining[candidates])])
        position = cursor[label]
        while position < len(order) and (
            assigned[position] >= 0 or counts[order[position], label] == 0
        ):
            position += 1
        cursor[label] = position
        if position >= len(order):  # pragma: no cover - remaining[] would be 0
            remaining[label] = 0.0
            continue

        want = desired_labels[:, label]
        if not equal_shares:
            want = want / ratios
        best = np.flatnonzero(want >= want.max() - 1e-9)
        if best.size > 1:
            sizes_wanted = desired_size[best]
            best = best[sizes_wanted >= sizes_wanted.max() - 1e-9]
        fold = int(best[0])

        group = order[position]
        assigned[position] = fold
        desired_labels[fold] -= counts[group]
        desired_size[fold] -= sizes[group]
        remaining -= counts[group]
        np.clip(remaining, 0.0, None, out=remaining)
        unplaced -= 1

    if (assigned < 0).any() or (assigned >= n_folds).any():  # pragma: no cover - guard
        raise RuntimeError("stratification left a group unassigned")
    return assigned


def assign_folds(
    table: pd.DataFrame,
    *,
    n_folds: int = N_FOLDS,
    group_keys: Sequence[str] = DEFAULT_GROUP_KEYS,
    strata_key: str | None = "language",
    labels: Sequence[str] = LABELS,
    mask_suffix: str = MASK_SUFFIX,
    seed: int = 0,
    ratios: Sequence[float] | None = None,
    components: np.ndarray | None = None,
) -> np.ndarray:
    """Fold index per row: iterative multi-label stratification, grouped.

    ``table`` is one row per study, carrying ``group_keys``, ``strata_key`` and
    the label columns. Returns an ``int64`` array parallel to ``table``.

    ``seed`` shuffles the order groups are offered in. It is not cosmetic: the
    order decides the outcome, so a fold table is only reproducible with the
    seed recorded. Groups are then sorted large-first within the shuffle, because
    a large group is the constraint that cannot be repaired later.

    ``ratios`` gives the target share of each bin and defaults to equal fifths.
    Unequal shares are what makes an 85/15 train/test holdout the *same*
    algorithm as a 5-fold split rather than a second implementation of it -- see
    :func:`assign_holdout`. A holdout cut by a different code path is a holdout
    whose stratification and grouping guarantees have to be argued twice.

    ``components`` supplies the grouping directly and **overrides**
    ``group_keys``. Pass it whenever ``table`` is a *subset* of the table whose
    grouping the result will be judged against, because grouping is not a
    per-row property: removing rows can cut a connected component in half.
    ``A -- shared scanner -- T -- shared report -- B`` is one scanner group over
    the whole corpus, and with ``T`` held out, ``A`` and ``B`` are two groups
    over the remainder and land in two different folds. Nothing raises, the
    file's own ``scanner_group`` column still says they are one group, and the
    scanner-grouped CV that exists to bound site memorisation is quietly
    contaminated for them. **Measured on the full 4,407-study corpus: 337 scanner
    components, 106 cut by the 15% holdout, 4 of them shattered into more than
    one component on the dev pool.** The ``fold`` arm is immune only because the
    holdout is grouped on those same keys, so it never cuts one of them -- 0
    measured, and that is a property of the holdout, not a property of the
    algorithm here.
    """
    if n_folds < 2:
        raise ValueError(f"n_folds must be at least 2, got {n_folds}")
    if len(table) < n_folds:
        raise ValueError(f"cannot cut {len(table)} studies into {n_folds} folds")
    if ratios is None:
        weights = np.full(n_folds, 1.0 / n_folds)
    else:
        weights = np.asarray(list(ratios), dtype=float)
        if weights.shape != (n_folds,):
            raise ValueError(
                f"ratios needs one entry per fold: got {weights.size} for {n_folds} folds"
            )
        if not np.isfinite(weights).all() or (weights <= 0.0).any():
            raise ValueError(f"every ratio must be finite and positive, got {list(weights)}")
        weights = weights / weights.sum()

    if components is None:
        components = group_components(table, group_keys)
    else:
        components = np.asarray(components)
        if components.shape != (len(table),):
            raise ValueError(
                f"components must be one entry per row: got {components.shape} for "
                f"{len(table)} rows"
            )
        # Densified because a partition sliced out of a larger table has gaps, and
        # the group bookkeeping below indexes by component id. `np.unique` sorts,
        # so ids keep their relative order and the seeded shuffle stays a function
        # of the partition rather than of how it was numbered.
        components = np.unique(components, return_inverse=True)[1].astype(np.int64)
    positives, usable = positive_matrix(table, labels, mask_suffix)

    # Stratify on each label's **minority class**, not on its positives. The
    # algorithm serves the scarcest target first, and for a label at 70%
    # prevalence the scarce class is the negative -- so counting positives there
    # asks it to balance the plentiful side and lets the scarce one fall where
    # it may. Measured on an 800-study synthetic table with prevalences from 15%
    # to 70%, worst per-label prevalence spread across five folds over three
    # seeds: **0.118 / 0.134 / 0.152 balancing positives, against 0.057 / 0.045
    # / 0.130 balancing the minority class.** It does not tighten the commonest
    # labels' positive *counts* -- those move with each fold's usable-cell count
    # -- and prevalence is the quantity that decides whether a fold is
    # representative, so that is the one traded for.
    target = positives.copy()
    for column in range(len(labels)):
        if positives[:, column].sum() * 2 > usable[:, column].sum():
            target[:, column] = usable[:, column] & ~positives[:, column]

    n_groups = int(components.max()) + 1
    sizes = np.bincount(components, minlength=n_groups).astype(np.int64)
    counts = np.zeros((n_groups, len(labels)), dtype=np.int64)
    np.add.at(counts, components, target.astype(np.int64))

    if strata_key is None:
        bucket_of = np.zeros(len(table), dtype=object)
        bucket_of[:] = "all"
    else:
        if strata_key not in table.columns:
            raise KeyError(f"stratification column {strata_key!r} is not in the table")
        bucket_of = table[strata_key].astype(str).to_numpy()

    # A group must sit in exactly one stratum. Where a group spans languages the
    # group wins, because splitting it would leak; the stratum is taken from the
    # group's first row so the choice is deterministic rather than arbitrary.
    group_bucket: dict[int, str] = {}
    for row, component in enumerate(components):
        group_bucket.setdefault(int(component), str(bucket_of[row]))

    largest = int(sizes.max())
    # Against the *smallest* target share, not the average one: with unequal
    # ``ratios`` the bin that cannot hold a large group is the small one, and
    # comparing against 1/n_folds would stay silent exactly where an 85/15
    # holdout is infeasible. Identical to the old check when shares are equal.
    smallest_share = len(table) * float(weights.min())
    if largest > smallest_share:
        warnings.warn(
            f"grouping on {list(group_keys)} produced a group of {largest} studies, more "
            f"than one fold's share of {len(table)} ({smallest_share:.0f}). Folds "
            f"cannot be balanced and a scanner-grouped split may be infeasible: check "
            f"whether the grouping keys collapsed into one component.",
            stacklevel=2,
        )

    desired_labels = weights[:, None] * counts.sum(axis=0).astype(float)[None, :]

    rng = np.random.default_rng(seed)
    fold_of_group = np.full(n_groups, -1, dtype=np.int64)
    # Smallest bucket first: it is the one that runs out of groups, so it gets
    # to place its rare labels while every fold is still empty.
    buckets = sorted(
        {group_bucket[g] for g in range(n_groups)},
        key=lambda b: (sum(1 for g in range(n_groups) if group_bucket[g] == b), b),
    )
    for bucket in buckets:
        members = [g for g in range(n_groups) if group_bucket[g] == bucket]
        rng.shuffle(members)
        members.sort(key=lambda g: -sizes[g])
        placed = _stratify_one_bucket(members, sizes, counts, desired_labels, weights)
        for position, group in enumerate(members):
            fold_of_group[group] = placed[position]

    return fold_of_group[components]


def assign_holdout(
    table: pd.DataFrame,
    *,
    test_frac: float = TEST_FRACTION,
    group_keys: Sequence[str] = DEFAULT_GROUP_KEYS,
    strata_key: str | None = "language",
    labels: Sequence[str] = LABELS,
    mask_suffix: str = MASK_SUFFIX,
    seed: int = 0,
    keep_ids: Iterable[str] = (),
    id_column: str = ID_COLUMN,
) -> np.ndarray:
    """Boolean per row: True where the study is held out as the frozen test set.

    The same iterative multi-label stratification as :func:`assign_folds`, run as
    a two-bin split with ``ratios`` of ``(1 - f, f)``, so the test set is
    stratified over all twelve labels and over the language bucket and is grouped
    exactly as the folds are. Reusing the algorithm is the point: a holdout cut by
    ``rng.choice`` would need its own argument for why no group straddles it, and
    the answer would be that none does.

    **Grouped on ``group_keys``, which does not include the scanner fingerprint,
    and that is deliberate.** A group straddling dev and test contaminates the
    test set -- 52 groups link 197 studies by identical report text, and
    ``PatientID`` links nothing at all -- so report content has to be a grouping
    key. Scanner does not: the hidden Kaggle test set is drawn from the same
    sites, so a scanner-disjoint local test would measure a harder problem than
    the one being scored. The test set is scanner-**shared** on purpose.

    ``keep_ids`` names studies that must stay in ``dev``. Their whole group is
    pinned, not just the row, since pinning a row and releasing its group would
    put the same report text on both sides. The absolute size of the test set is
    preserved across pinning: the share offered to the eligible pool is scaled up
    so that ``test_frac`` still describes a fraction of the *whole* table.
    """
    if not 0.0 < test_frac < 1.0:
        raise ValueError(f"test_frac must be strictly between 0 and 1, got {test_frac}")
    if id_column not in table.columns:
        raise KeyError(f"the study table is missing its identifier column {id_column!r}")

    components = group_components(table, group_keys)
    uids = table[id_column].astype(str).to_numpy()
    wanted_ids = {str(uid) for uid in keep_ids}
    if wanted_ids:
        matched = {uid for uid in uids.tolist() if uid in wanted_ids}
        if not matched:
            # The 58 anchors are named by StudyInstanceUID; if not one of them is
            # in the table, the two id spaces do not match and the pin is a no-op
            # that would look like it worked. Silence here puts spent studies into
            # the test set.
            raise ValueError(
                f"none of the {len(wanted_ids)} keep_ids appear in {id_column!r}, so "
                f"nothing would be pinned to dev. The id spaces do not match: e.g. "
                f"keep_ids has {sorted(wanted_ids)[0]!r}, the table has {uids[0]!r}"
            )
        if len(matched) < len(wanted_ids):
            warnings.warn(
                f"{len(wanted_ids) - len(matched)} of {len(wanted_ids)} keep_ids are not "
                f"in the study table, so only {len(matched)} studies are pinned to dev. "
                f"Check the join before trusting the holdout.",
                stacklevel=2,
            )
        pinned = np.asarray(
            sorted({int(c) for c, uid in zip(components, uids.tolist()) if uid in wanted_ids}),
            dtype=np.int64,
        )
    else:
        pinned = np.empty(0, dtype=np.int64)

    eligible = ~np.isin(components, pinned)
    n_eligible = int(eligible.sum())
    if n_eligible < 2:
        raise ValueError(
            f"{len(table) - n_eligible} of {len(table)} studies are pinned to dev by "
            f"keep_ids, leaving {n_eligible} to draw a test set from"
        )
    share = test_frac * len(table) / n_eligible
    if share >= 1.0:
        raise ValueError(
            f"a test set of {test_frac:.0%} of {len(table)} studies cannot be drawn from "
            f"the {n_eligible} not pinned to dev by keep_ids"
        )

    bins = assign_folds(
        table.loc[eligible].reset_index(drop=True),
        n_folds=2,
        group_keys=group_keys,
        strata_key=strata_key,
        labels=labels,
        mask_suffix=mask_suffix,
        seed=seed,
        ratios=(1.0 - share, share),
    )
    out = np.zeros(len(table), dtype=bool)
    out[eligible] = bins == 1
    # A "held-out" table with nothing held out, or with nothing left to train on,
    # is the failure this whole module exists to make impossible. Both are
    # reachable from a small table and a small fraction, and neither raises
    # anywhere downstream: the folds still build, the gates still pass.
    if not out.any() or out.all():
        raise ValueError(
            f"test_frac={test_frac} over {len(table)} studies put {int(out.sum())} of them "
            f"in the test set; a holdout needs studies on both sides"
        )
    return out


def build_folds(
    table: pd.DataFrame,
    *,
    n_folds: int = N_FOLDS,
    group_keys: Sequence[str] = DEFAULT_GROUP_KEYS,
    scanner_group_keys: Sequence[str] = SCANNER_GROUP_KEYS,
    strata_key: str | None = "language",
    labels: Sequence[str] = LABELS,
    mask_suffix: str = MASK_SUFFIX,
    seed: int = 0,
    id_column: str = ID_COLUMN,
) -> pd.DataFrame:
    """Both fold assignments, as a :data:`FOLD_COLUMNS` frame.

    ``fold`` is grouped by ``group_keys`` only; ``fold_scanner`` is grouped by
    ``scanner_group_keys`` as well, so no scanner spans two folds. Both use the
    same stratification and the same seed, so the pair differs only in the
    grouping constraint -- which is what makes comparing a score across them
    informative about site memorisation rather than about fold noise.
    """
    if id_column not in table.columns:
        raise KeyError(f"the study table is missing its identifier column {id_column!r}")
    if table[id_column].duplicated().any():
        raise ValueError(f"{id_column} is not unique: build_folds needs one row per study")

    ordered = table.sort_values(id_column).reset_index(drop=True)
    fold = assign_folds(
        ordered,
        n_folds=n_folds,
        group_keys=group_keys,
        strata_key=strata_key,
        labels=labels,
        mask_suffix=mask_suffix,
        seed=seed,
    )
    fold_scanner = assign_folds(
        ordered,
        n_folds=n_folds,
        group_keys=scanner_group_keys,
        strata_key=strata_key,
        labels=labels,
        mask_suffix=mask_suffix,
        seed=seed,
    )
    return pd.DataFrame(
        {
            id_column: ordered[id_column].astype(str),
            "fold": fold,
            "fold_scanner": fold_scanner,
            "patient_group": group_components(ordered, group_keys),
            "scanner_group": group_components(ordered, scanner_group_keys),
            "language": (
                ordered[strata_key].astype(str)
                if strata_key is not None and strata_key in ordered.columns
                else "all"
            ),
        },
        columns=list(FOLD_COLUMNS),
    )


def build_folds_v2(
    table: pd.DataFrame,
    *,
    n_folds: int = N_FOLDS,
    test_frac: float = TEST_FRACTION,
    group_keys: Sequence[str] = DEFAULT_GROUP_KEYS,
    scanner_group_keys: Sequence[str] = SCANNER_GROUP_KEYS,
    strata_key: str | None = "language",
    labels: Sequence[str] = LABELS,
    mask_suffix: str = MASK_SUFFIX,
    seed: int = 0,
    id_column: str = ID_COLUMN,
    keep_ids: Iterable[str] = (),
) -> pd.DataFrame:
    """A frozen test set plus five folds over the rest, as a :data:`FOLD_COLUMNS_V2` frame.

    ``split`` is ``"test"`` for the held-out ``test_frac`` and ``"dev"`` for the
    rest; a test row carries ``fold == fold_scanner == -1``
    (:data:`TEST_FOLD`). Both fold columns are cut over the **dev pool only**, so
    fold *k*'s training side is dev minus fold *k* and the test set is in no fold
    at all. Reading it back needs :func:`dev_mask`, not ``fold != k``.

    ``report_group`` and ``scanner_group`` are the integer component ids over
    ``group_keys`` and ``scanner_group_keys``, computed over the **whole** table
    rather than per side. That is what lets a reader verify from the file alone --
    with no access to the non-redistributable corpus -- that no group spans two
    folds *and* that no group spans dev and test. Note the input table's
    ``report_group`` column holds the report content *hash*; the emitted column of
    that name holds the component id derived from it.

    The test set is scanner-shared, the holdout costs 661 of 4,407 studies as
    training data, and it is never looked at until the pipeline is finished. All
    three are explained in this module's docstring, which is the thing to read
    before using this function.
    """
    if id_column not in table.columns:
        raise KeyError(f"the study table is missing its identifier column {id_column!r}")
    if table[id_column].duplicated().any():
        raise ValueError(
            f"{id_column} is not unique: build_folds_v2 needs one row per study"
        )

    ordered = table.sort_values(id_column).reset_index(drop=True)
    # Computed ONCE over the whole table and reused for the assignment, the gate and
    # the emitted column. Recomputing on the dev pool is not a refactor away from
    # this: the holdout cuts scanner components, and a cut component comes back as
    # two, which is how 4 scanner groups ended up spanning two folds while the
    # file's own scanner_group column still called them one. See assign_folds.
    report_components = group_components(ordered, group_keys)
    scanner_components = group_components(ordered, scanner_group_keys)
    is_test = assign_holdout(
        ordered,
        test_frac=test_frac,
        group_keys=group_keys,
        strata_key=strata_key,
        labels=labels,
        mask_suffix=mask_suffix,
        seed=seed,
        keep_ids=keep_ids,
        id_column=id_column,
    )
    dev = ordered.loc[~is_test].reset_index(drop=True)

    fold = np.full(len(ordered), TEST_FOLD, dtype=np.int64)
    fold_scanner = np.full(len(ordered), TEST_FOLD, dtype=np.int64)
    keep = ~np.asarray(is_test)
    fold[~is_test] = assign_folds(
        dev,
        n_folds=n_folds,
        strata_key=strata_key,
        labels=labels,
        mask_suffix=mask_suffix,
        seed=seed,
        components=report_components[keep],
    )
    fold_scanner[~is_test] = assign_folds(
        dev,
        n_folds=n_folds,
        strata_key=strata_key,
        labels=labels,
        mask_suffix=mask_suffix,
        seed=seed,
        components=scanner_components[keep],
    )
    return pd.DataFrame(
        {
            id_column: ordered[id_column].astype(str),
            "split": np.where(is_test, "test", "dev"),
            "fold": fold,
            "fold_scanner": fold_scanner,
            "report_group": report_components,
            "scanner_group": scanner_components,
            "language": (
                ordered[strata_key].astype(str)
                if strata_key is not None and strata_key in ordered.columns
                else "all"
            ),
        },
        columns=list(FOLD_COLUMNS_V2),
    )


@dataclass(frozen=True)
class FoldReport:
    """Per-fold evidence for one fold column, and the gate it either passes or not.

    ``empty_labels`` is the one that matters most: a ``(label, fold)`` pair with
    zero usable positives makes that label's AUC ``nan`` on that fold, and the
    macro over the remaining eleven is silently a different metric.

    ``overlaps`` counts, per grouping column, the groups that span more than one
    fold. Only the columns named in ``gate_columns`` have to be zero. The
    distinction is not pedantry: the patient-grouped ``fold`` arm is *expected*
    to spread each scanner across folds -- that is what makes it the random-fold
    arm -- so failing it for a non-zero ``scanner_group`` overlap would reject
    the very split it is there to provide, and the honest thing is to measure the
    overlap and report it.
    """

    fold_column: str
    n_folds: int
    sizes: tuple[int, ...]
    n_groups: int
    largest_group: int
    positives: dict[str, tuple[int, ...]]
    usable: dict[str, tuple[int, ...]]
    languages: dict[str, tuple[int, ...]]
    empty_labels: tuple[tuple[str, int], ...]
    overlaps: dict[str, int]
    gate_columns: tuple[str, ...]
    #: Whether ``starved_languages`` is part of the gate. Off by default and off
    #: for the ``fold_scanner`` arm on purpose: language collapses across scanner
    #: groups here (cyrillic 164/13/5/2/3 as measured), which is a property of the
    #: corpus, not a defect in the split, and gating it would reject the
    #: pessimistic arm the project needs.
    gate_languages: bool = False
    #: Display name per bin. Empty means ``fold0..foldN``. Set by
    #: :func:`holdout_report` to ``("dev", "test")``, because a two-column table
    #: headed ``fold0  fold1`` is read as two folds by whoever pastes it into a
    #: log, and the number under ``fold1`` is the one nobody may look at yet.
    fold_names: tuple[str, ...] = ()

    @property
    def names(self) -> tuple[str, ...]:
        """One display name per bin, defaulting to ``fold0``, ``fold1``, ..."""
        return self.fold_names or tuple(f"fold{index}" for index in range(self.n_folds))

    @property
    def prevalence(self) -> dict[str, tuple[float, ...]]:
        """Usable positives over usable cells, per fold. NaN where nothing is usable."""
        return {
            label: tuple(
                (p / u if u else float("nan"))
                for p, u in zip(self.positives[label], self.usable[label])
            )
            for label in self.positives
        }

    @property
    def spread(self) -> dict[str, float]:
        """Max minus min prevalence across folds, per label. Small is the goal."""
        out = {}
        for label, values in self.prevalence.items():
            finite = [v for v in values if v == v]
            out[label] = (max(finite) - min(finite)) if finite else float("nan")
        return out

    @property
    def starved_languages(self) -> tuple[tuple[str, int], ...]:
        """``(bucket, bin)`` pairs holding far less of a language than their size implies.

        **This is the second door on the failure that has already happened once.**
        A holdout once produced a test set that was 98% Latin -- cyrillic 0 of 220,
        greek 4 of 321 -- and passed every gate there was, because the gates asked
        only whether each *label* had a positive. Language was measured, printed,
        and never checked. Reports are multilingual (9 buckets, 39.4% English) and
        a test set that is one language measures the model on one language.

        A bucket is judged against the share of the corpus its bin holds, not
        against the other buckets, and a bin expected to hold fewer than
        :data:`LANGUAGE_MIN_EXPECTED` of a bucket only has to hold *some*.
        """
        total = sum(self.sizes)
        if not total:
            return ()
        starved = []
        for bucket, counts in sorted(self.languages.items()):
            present = sum(counts)
            for index, count in enumerate(counts):
                expected = present * (self.sizes[index] / total)
                if expected < 1.0:
                    continue
                floor = (
                    LANGUAGE_TOLERANCE * expected
                    if expected >= LANGUAGE_MIN_EXPECTED
                    else 1.0
                )
                if count < floor:
                    starved.append((bucket, index))
        return tuple(starved)

    @property
    def ok(self) -> bool:
        """True when no label is empty in a fold and no gated group spans folds."""
        return (
            not self.empty_labels
            and not any(self.overlaps.get(column, 0) for column in self.gate_columns)
            and not (self.gate_languages and self.starved_languages)
        )

    def raise_for_gate(self) -> FoldReport:
        """Raise unless the fold table passes the gate. Returns self when it does."""
        problems = []
        if self.empty_labels:
            names = self.names
            listed = ", ".join(f"{label} in {names[fold]}" for label, fold in self.empty_labels)
            problems.append(
                f"{len(self.empty_labels)} label/fold pairs have zero usable positives "
                f"({listed}); macro AUC over that fold is not the competition metric"
            )
        for column in self.gate_columns:
            count = self.overlaps.get(column, 0)
            if count:
                problems.append(f"{count} {column} values span more than one fold: leakage")
        if self.gate_languages and self.starved_languages:
            names = self.names
            listed = ", ".join(
                f"{bucket} in {names[index]} ({self.languages[bucket][index]})"
                for bucket, index in self.starved_languages
            )
            problems.append(
                f"{len(self.starved_languages)} language bucket/bin pairs hold far less "
                f"than their share ({listed}); a bin that is one language measures the "
                f"model on one language, and the label gates do not see it"
            )
        if problems:
            raise ValueError(
                f"fold column {self.fold_column!r} fails the gate: " + "; ".join(problems)
            )
        return self

    def table(self) -> str:
        """Per-label prevalence per fold and its spread, as fixed-width text."""
        names = self.names
        header = f"{'label':18}" + "".join(f"  {name:>5}" for name in names) + "   spread"
        lines = [
            f"fold column {self.fold_column!r}: sizes {list(self.sizes)}, "
            f"{self.n_groups} groups, largest {self.largest_group} studies",
            "languages: "
            + ", ".join(f"{k} {list(v)}" for k, v in sorted(self.languages.items())),
            header,
        ]
        prevalence = self.prevalence
        spread = self.spread
        for label in self.positives:
            cells = "".join(f"  {value:.3f}" for value in prevalence[label])
            lines.append(f"{label:18}{cells}   {spread[label]:.3f}")
        for label in self.positives:
            lines.append(
                f"{label:18}" + "".join(f"  {n:5d}" for n in self.positives[label]) + "   positives"
            )
        if self.empty_labels:
            lines.append(
                "GATE FAILED: zero positives for "
                + ", ".join(f"{label}/{names[fold]}" for label, fold in self.empty_labels)
            )
        for column, count in self.overlaps.items():
            gated = column in self.gate_columns
            note = "  LEAK" if count and gated else ("  (not gated)" if not gated else "")
            lines.append(f"{column} spanning folds: {count}{note}")
        starved = self.starved_languages
        if starved:
            names = self.names
            listed = ", ".join(f"{bucket}/{names[index]}" for bucket, index in starved)
            lines.append(
                f"language buckets below their share: {listed}"
                + ("  STARVED" if self.gate_languages else "  (not gated)")
            )
        return "\n".join(lines)


def fold_report(
    folds: pd.DataFrame,
    table: pd.DataFrame,
    *,
    fold_column: str = "fold",
    group_column: str = "patient_group",
    labels: Sequence[str] = LABELS,
    mask_suffix: str = MASK_SUFFIX,
    id_column: str = ID_COLUMN,
    extra_group_columns: Iterable[str] = (),
    gate_languages: bool = False,
) -> FoldReport:
    """Measure one fold column against the label table it was built from.

    ``group_column`` is the grouping the fold column was built with, and the only
    one the gate enforces. ``extra_group_columns`` are measured and printed but
    not gated -- see :class:`FoldReport`.

    ``folds`` and ``table`` are joined on ``id_column`` rather than assumed to be
    in the same order, because the fold table is written once and read back much
    later by code that has no reason to preserve row order.
    """
    for frame, name in ((folds, "fold table"), (table, "label table")):
        if id_column not in frame.columns:
            raise KeyError(f"the {name} is missing its identifier column {id_column!r}")
    if fold_column not in folds.columns:
        raise KeyError(f"the fold table has no {fold_column!r} column")

    joined = folds.merge(table, on=id_column, how="inner", suffixes=("", "_label"))
    if len(joined) != len(folds):
        raise ValueError(
            f"{len(folds) - len(joined)} of {len(folds)} fold rows have no label row; "
            f"a fold table that does not cover the label table cannot be validated"
        )

    assignment = joined[fold_column].to_numpy(dtype=np.int64)
    n_folds = int(assignment.max()) + 1
    positives, usable = positive_matrix(joined, labels, mask_suffix)

    per_label_pos: dict[str, tuple[int, ...]] = {}
    per_label_usable: dict[str, tuple[int, ...]] = {}
    empty: list[tuple[str, int]] = []
    for column, label in enumerate(labels):
        pos = tuple(int(positives[assignment == f, column].sum()) for f in range(n_folds))
        use = tuple(int(usable[assignment == f, column].sum()) for f in range(n_folds))
        per_label_pos[label] = pos
        per_label_usable[label] = use
        empty.extend((label, f) for f in range(n_folds) if pos[f] == 0)

    languages: dict[str, tuple[int, ...]] = {}
    if "language" in joined.columns:
        for bucket in sorted(joined["language"].astype(str).unique()):
            selected = joined["language"].astype(str).to_numpy() == bucket
            languages[bucket] = tuple(
                int((assignment[selected] == f).sum()) for f in range(n_folds)
            )

    overlaps: dict[str, int] = {}
    for column in (group_column, *extra_group_columns):
        if column in joined.columns and column not in overlaps:
            spans = joined.groupby(column)[fold_column].nunique()
            overlaps[column] = int((spans > 1).sum())

    groups = joined[group_column] if group_column in joined.columns else joined[id_column]
    return FoldReport(
        fold_column=fold_column,
        n_folds=n_folds,
        sizes=tuple(int((assignment == f).sum()) for f in range(n_folds)),
        n_groups=int(groups.nunique()),
        largest_group=int(groups.value_counts().max()),
        positives=per_label_pos,
        usable=per_label_usable,
        languages=languages,
        empty_labels=tuple(empty),
        overlaps=overlaps,
        gate_columns=(group_column,) if group_column in overlaps else (),
        gate_languages=gate_languages,
    )


def holdout_report(
    folds: pd.DataFrame,
    table: pd.DataFrame,
    *,
    group_column: str = "report_group",
    labels: Sequence[str] = LABELS,
    mask_suffix: str = MASK_SUFFIX,
    id_column: str = ID_COLUMN,
    extra_group_columns: Iterable[str] = ("scanner_group",),
    gate_languages: bool = True,
) -> FoldReport:
    """Dev against test as a two-bin :class:`FoldReport`: index 0 dev, index 1 test.

    Measured with the fold machinery rather than beside it, so the holdout is held
    to the same two gates the folds are, and one ``raise_for_gate`` covers both:

    * **every one of the twelve labels has a usable positive in the test set.** A
      label with none has no test AUC, so the one honest number at the end would
      be a macro over eleven columns -- a different metric, quietly.
    * **no group spans dev and test.** This is the gate that matters most and the
      only one that can be checked from the written file alone. 52 groups link 197
      studies by identical report text; one of them straddling the boundary means
      the model has seen the report -- and, for a repeat patient, the knee -- it is
      being scored on.

    ``scanner_group`` is measured but **not** gated, and a non-zero overlap there
    is the intended outcome, not a leak: the test set is scanner-shared because
    the hidden Kaggle test set is drawn from the same sites.
    """
    if "split" not in folds.columns:
        raise KeyError(
            f"the fold table has no 'split' column, so it holds nothing out; its "
            f"columns are {tuple(folds.columns)}"
        )
    values = folds["split"].astype(str)
    unknown = sorted(set(values) - set(SPLIT_VALUES))
    if unknown:
        raise ValueError(f"the 'split' column holds {unknown}, contract allows {SPLIT_VALUES}")
    missing = [name for name in SPLIT_VALUES if not bool((values == name).any())]
    if missing:
        # An all-dev table reports one bin, and a one-bin report never looks at
        # the test side at all -- so the gate would pass on a holdout that does
        # not exist. Refuse rather than report a pass.
        raise ValueError(
            f"the 'split' column has no {missing} rows, so there is no holdout to report"
        )

    probe = folds.copy()
    probe["_split_bin"] = (values == "test").astype(np.int64).to_numpy()
    report = fold_report(
        probe,
        table,
        fold_column="_split_bin",
        group_column=group_column,
        labels=labels,
        mask_suffix=mask_suffix,
        id_column=id_column,
        extra_group_columns=extra_group_columns,
        gate_languages=gate_languages,
    )
    return replace(report, fold_column="split", fold_names=SPLIT_VALUES)


def write_folds(folds: pd.DataFrame, path: Path | str) -> Path:
    """Write the fold table as CSV, refusing any other suffix.

    The suffix check is the whole point of this function existing. ``data/*`` is
    gitignored with a ``!data/folds/`` negation, but ``*.parquet`` is globbed
    unconditionally, so ``folds_v1.parquet`` under ``data/folds/`` is written
    successfully and then never tracked -- and the loss is only noticed when
    someone tries to reproduce a score from a fresh clone. Verify with
    ``git check-ignore -v <path>``: a CSV under ``data/folds/`` exits 1, not
    ignored.

    Either pinned schema is accepted -- :data:`FOLD_COLUMNS` or
    :data:`FOLD_COLUMNS_V2`. v1 is not deprecated by v2: ``folds_v1.csv`` is the
    table earlier numbers were produced against, and a writer that refuses it
    makes them unreproducible.
    """
    out = Path(path)
    if out.suffix.lower() != ".csv":
        raise ValueError(
            f"fold tables must be .csv, got {out.suffix!r}. .gitignore globs *.parquet "
            f"even under data/folds/, so the file would be written and never tracked."
        )
    check_fold_columns(folds.columns, what="the fold table being written")
    out.parent.mkdir(parents=True, exist_ok=True)
    folds.to_csv(out, index=False, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
    return out


def read_folds(path: Path | str) -> pd.DataFrame:
    """Read a fold table back, asserting the column list and the id dtype.

    ``StudyInstanceUID`` is read as a string on purpose. It is a long dotted
    numeric, and pandas will happily turn a column of them into floats on some
    inputs, which silently breaks every join against it.

    Both pinned schemas are accepted, so this reads ``folds_v1.csv`` and
    ``folds_v2.csv`` alike. What differs is what the caller must then do with the
    result: a v2 table holds rows that are in no fold, and splitting it with
    ``fold != k`` puts every one of them in the training set. Use
    :func:`dev_mask`.
    """
    folds = pd.read_csv(path, dtype={ID_COLUMN: str})
    check_fold_columns(folds.columns, what=str(path))
    return folds
