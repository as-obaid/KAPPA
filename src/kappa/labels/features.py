"""The per-study privileged feature vector: 78 anatomy clusters x 2 polarities.

Why this exists: the twelve competition labels are a **lossy projection** of what a
report actually says. ``kappa.labels.label_map`` throws most of the report away by
construction -- 17 of the 78 clusters map to none of the 12 labels (PCL, LCL, the
patellar and quadriceps tendons, the MPFL, the IT band, the retinaculum, the
popliteus, the fat pad), and every one of them is a real structure visible in the
same pixels. This module keeps that discarded evidence in the one form a model can
read: for each cluster, was it mentioned positively, and was it mentioned negatively.

That vector is **privileged information**. Reports exist for the 4,407 train studies
and for none of the test studies, so nothing that reads one can run at
submission time. The whole point of :mod:`kappa.models.teacher` is to convert this
train-only signal into an image-only student via distillation, and this module is what
the teacher is given. Without it the teacher sees exactly what the student sees and
distillation degenerates into self-distillation -- which still trains, still prints a
loss curve, and measures nothing about whether report information transfers.

It lives in ``kappa.labels`` deliberately. That package is training-only by the repo's
import boundary (``tests/test_loop.py::TestTheInferenceBoundary``), so putting the
report featuriser anywhere reachable from ``kappa.inference`` is a notebook that cannot
run rather than one that is slow. This module imports **no torch**, for the same reason
every other ``kappa.labels`` module does not.

Three decisions that are load-bearing rather than stylistic.

* **The feature order is ``sorted(CLUSTER_NAMES)``, never the frozenset's own order.**
  :data:`~kappa.labels.anatomy_map.CLUSTER_NAMES` is a ``frozenset``, and CPython
  randomises string hashing per process, so its iteration order genuinely differs
  between runs. A features table written by one process and read by another would then
  be silently permuted: the teacher would train on a column labelled ``ACL:pos`` that
  actually held ``SOFT_TISSUE:neg``, converge to something, and report a number.
  :func:`load_features` re-checks the order on read for exactly this reason.
* **A cluster mentioned both ways gets both channels.** ``medial and lateral meniscus``
  is one finding and two sites, a study images one knee but a bilateral report names
  two, and reports routinely say what is preserved alongside what is torn ("intact,
  with periligamentous edema"). Two channels firing is information about the report,
  not a conflict to arbitrate, and collapsing it to one would throw away the hedge.
* **No finding-type gates.** ``label_map.GATES`` exists to stop a *site* from inventing
  a *finding-type* label -- a medial femoral condyle with an osteophyte is Medial OA,
  not a Contusion. That problem does not arise here, because these features name a
  site and a direction and claim nothing about which of the twelve labels they imply.
  Gating them would remove evidence the teacher is allowed to have.

``mask`` polarity contributes nothing to either channel (159 of 3,962 survey findings):
measurements, normal variants and benign incidentals genuinely carry no information
about the site, and ``kappa.labels.polarity`` documents why collapsing them to
``negative`` is worse than dropping them.

    from kappa.labels.features import study_features, write_features

    vector = study_features(record)              # (156,) float32, values in {0, 1}
    write_features(records, out_path)             # one row per study, joinable on uid
    uids, matrix, present = load_features(out_path)
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np

from kappa.labels.anatomy_map import CLUSTER_NAMES, clusters_of
from kappa.labels.masking import report_mask
from kappa.labels.polarity import NEGATIVE, POSITIVE, polarity

__all__ = [
    "PRESENT_COLUMN",
    "REPORT_CLUSTERS",
    "REPORT_FEATURE_DIM",
    "REPORT_FEATURE_NAMES",
    "UID_COLUMN",
    "FEATURE_TABLE_COLUMNS",
    "load_features",
    "study_features",
    "study_features_present",
    "write_features",
]

#: The 78 clusters, in a **stable** order. ``sorted``, not ``tuple(CLUSTER_NAMES)``:
#: a frozenset has no defined iteration order and CPython randomises string hashing
#: per process, so the unsorted spelling permutes the feature vector between runs
#: without changing anything a caller can see.
REPORT_CLUSTERS: tuple[str, ...] = tuple(sorted(CLUSTER_NAMES))

#: Two channels per cluster, positive then negative, in :data:`REPORT_CLUSTERS`
#: order. The names are the parquet column names, so they are part of the contract.
REPORT_FEATURE_NAMES: tuple[str, ...] = tuple(
    f"{cluster}:{channel}" for cluster in REPORT_CLUSTERS for channel in ("pos", "neg")
)

#: Width of the vector the teacher's report branch consumes: 78 clusters x 2.
REPORT_FEATURE_DIM: int = len(REPORT_FEATURE_NAMES)

#: Join key of the features table. Matches ``record["_study_uid"]``, which is the
#: same field ``kappa.labels.label_map.derive_study`` keys its output on.
UID_COLUMN = "study_uid"

#: Whether this study has a report at all, as distinct from a report that said
#: nothing. See :func:`study_features_present`.
PRESENT_COLUMN = "report_present"

#: Full column order of the parquet :func:`write_features` writes.
FEATURE_TABLE_COLUMNS: tuple[str, ...] = (UID_COLUMN, *REPORT_FEATURE_NAMES, PRESENT_COLUMN)

_CHANNEL_INDEX: dict[str, int] = {name: i for i, name in enumerate(REPORT_FEATURE_NAMES)}


def study_features(record: dict) -> np.ndarray:
    """Featurise one extraction record into ``(REPORT_FEATURE_DIM,)`` float32.

    Every finding's ``anatomy`` is mapped through
    :func:`~kappa.labels.anatomy_map.clusters_of` -- which returns *more than one*
    cluster whenever the string names more than one site -- and its polarity through
    :func:`~kappa.labels.polarity.polarity`, which needs both ``assertion`` and
    ``description`` or it is language-biased.

    A record whose extraction never parsed returns an all-zero vector, mirroring
    ``derive_study``: there is no evidence of any kind, which is a different thing
    from a report that parsed and said little. Use
    :func:`study_features_present` to tell the two apart -- an all-zero vector on its
    own cannot, and the teacher must not read "no report" as "a report full of
    negatives".
    """
    vector = np.zeros(REPORT_FEATURE_DIM, dtype=np.float32)
    if "extraction_failed" in report_mask(record).reasons:
        return vector

    for finding in record.get("findings") or []:
        if not isinstance(finding, dict):
            continue
        description = finding.get("description") or ""
        found = polarity(finding.get("assertion"), description)
        if found == POSITIVE:
            channel = "pos"
        elif found == NEGATIVE:
            channel = "neg"
        else:
            # `mask` -- a measurement, a normal variant, a benign incidental. No
            # information about the site, so neither channel fires.
            continue
        for cluster in clusters_of(finding.get("anatomy") or ""):
            index = _CHANNEL_INDEX.get(f"{cluster}:{channel}")
            if index is None:
                # The frozen vocabulary has drifted from the clusterer. Dropping the
                # cluster instead would remove signal from every study that names it
                # and nothing downstream would report a shorter vector.
                raise ValueError(
                    f"clusters_of returned {cluster!r}, which is not in CLUSTER_NAMES. "
                    "The frozen cluster vocabulary and the clustering cascade have "
                    "drifted apart; fix anatomy_map.CLUSTER_NAMES rather than this."
                )
            vector[index] = 1.0
    return vector


def study_features_present(record: dict) -> bool:
    """Does this study have a usable report at all?

    ``False`` only for ``extraction_failed`` -- the model produced nothing
    parseable. An empty findings section is ``True``: the report exists and is
    genuinely silent, and the teacher's gating must not treat the two the same way.
    """
    return "extraction_failed" not in report_mask(record).reasons


def write_features(records: Iterable[dict], out_path: Path) -> Path:
    """Write one parquet row per study: uid, 156 float32 channels, presence flag.

    Columns are exactly :data:`FEATURE_TABLE_COLUMNS`, in that order, because
    :func:`load_features` refuses anything else.
    """
    # pandas is imported here rather than at module scope so that the pure
    # featurising half of this module -- the half the derivation scripts use -- does
    # not pull in pandas and pyarrow to build a 156-element array.
    import pandas as pd

    uids: list[str] = []
    rows: list[np.ndarray] = []
    present: list[bool] = []
    for position, record in enumerate(records):
        uid = record.get("_study_uid")
        if not isinstance(uid, str) or not uid:
            # A row keyed by None cannot be joined to the manifest, and the join
            # would drop it silently rather than fail.
            raise ValueError(
                f"record at position {position} has _study_uid={uid!r}; the features "
                "table is joined on that field, so it must be a non-empty string"
            )
        uids.append(uid)
        rows.append(study_features(record))
        present.append(study_features_present(record))

    matrix = (
        np.stack(rows) if rows else np.zeros((0, REPORT_FEATURE_DIM), dtype=np.float32)
    )
    frame = pd.DataFrame(matrix, columns=list(REPORT_FEATURE_NAMES))
    frame.insert(0, UID_COLUMN, pd.Series(uids, dtype=object))
    frame[PRESENT_COLUMN] = np.asarray(present, dtype=bool)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out_path, index=False)
    return out_path


def load_features(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Read a features parquet back as ``(study_uids, features, present)``.

    ``features`` is ``(n, REPORT_FEATURE_DIM)`` float32 in
    :data:`REPORT_FEATURE_NAMES` order, ``present`` is ``(n,)`` bool.

    Raises ``ValueError`` if the feature columns are not exactly
    :data:`REPORT_FEATURE_NAMES`, in order. This check is the point of the function:
    a reordered or truncated feature table trains the teacher on permuted inputs, and
    a permuted 156-wide input produces a model that converges, prints a loss curve
    and is evidence about nothing. Nothing else downstream would notice.
    """
    import pandas as pd

    frame = pd.read_parquet(path)
    for column in (UID_COLUMN, PRESENT_COLUMN):
        if column not in frame.columns:
            raise ValueError(
                f"{path} has no {column!r} column; expected the table written by "
                f"write_features, whose columns are {FEATURE_TABLE_COLUMNS[:2]}...  "
                f"got {list(frame.columns)[:4]}..."
            )
    found = tuple(c for c in frame.columns if c not in (UID_COLUMN, PRESENT_COLUMN))
    if found != REPORT_FEATURE_NAMES:
        where = _first_mismatch(found)
        raise ValueError(
            f"{path} has {len(found)} feature columns, expected {REPORT_FEATURE_DIM} in "
            f"REPORT_FEATURE_NAMES order. First difference at index {where}: got "
            f"{_at(found, where)} where {_at(REPORT_FEATURE_NAMES, where)} was expected. "
            "A permuted or truncated feature table trains a teacher on permuted inputs "
            "and fails silently."
        )
    features = frame.loc[:, list(REPORT_FEATURE_NAMES)].to_numpy(dtype=np.float32)
    present = frame[PRESENT_COLUMN].to_numpy(dtype=bool)
    return [str(uid) for uid in frame[UID_COLUMN]], features, present


def _first_mismatch(found: tuple[str, ...]) -> int:
    """Index of the first column that differs from the pinned order.

    For a table that matches on its whole (shorter or longer) prefix, that index is
    the point one of the two ran out, which is off the end of one of them -- hence
    :func:`_at` rather than plain subscripting in the error message.
    """
    for index, (got, want) in enumerate(zip(found, REPORT_FEATURE_NAMES)):
        if got != want:
            return index
    return min(len(found), REPORT_FEATURE_DIM)


def _at(names: tuple[str, ...], index: int) -> str:
    """``names[index]`` for an error message, or ``<end of table>`` past the end."""
    return repr(names[index]) if index < len(names) else "<end of table>"
