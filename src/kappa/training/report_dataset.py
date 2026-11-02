"""The teacher's half of the dataset: a study, plus what its report said.

Why this is a separate module rather than two more arguments on
:class:`kappa.data.dataset.StudyDataset`. ``kappa/data/dataset.py`` is on the
**inference path** -- the Kaggle notebook reaches it through
``kappa.inference.predict`` -- and the competition's test studies carry no report
at all. Putting report handling there would make the privileged input
*expressible* by the code that ships, and ``tests/test_loop.py`` enforces the
stricter rule that no module on that path may even contain the string
``"report"``. That rule is why this file exists: the boundary is structural, not a
convention someone has to remember.

The report vector is therefore added by a subclass living in ``kappa.training``,
which is training-only under the same boundary. An inference run cannot construct
it, cannot pass it, and cannot name it.

    from kappa.training.report_dataset import build_study_dataset, build_report_collate

    dataset = build_study_dataset(manifest, cache_dir, report_features=frame, **common)
    loader = DataLoader(dataset, collate_fn=build_report_collate(cfg, seed=0))

The two extra keys::

    report          FloatTensor (report_dim,)  78 anatomy clusters x 2 polarity channels
    report_present  BoolTensor  ()             False = no usable extraction

``report_present`` is not redundant with an all-zero vector. A report that parsed
and said nothing is an observation; an extraction that failed is an absence of
evidence. The teacher gates its report branch on this flag, so the two cannot
collapse into the same input.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
import torch

from kappa.config import AugmentConfig, TrainConfig
from kappa.constants import ID_COLUMN
from kappa.data.augment import build_collate
from kappa.data.dataset import StudyDataset, collate_studies

__all__ = [
    "ReportStudyDataset",
    "build_study_dataset",
    "build_report_collate",
    "report_collate",
]

PRESENT_COLUMN = "report_present"


def _index_features(frame: pd.DataFrame) -> tuple[dict[str, int], np.ndarray, np.ndarray]:
    """Flatten the feature table, checking only the shape contract.

    Deliberately name-agnostic. The feature vocabulary and its order are owned and
    validated by :func:`kappa.labels.features.load_features`; re-checking them here
    would give two places to disagree about what the 156 columns are.
    """
    for column in (ID_COLUMN, PRESENT_COLUMN):
        if column not in frame.columns:
            raise KeyError(f"report feature table is missing the {column!r} column")
    columns = [c for c in frame.columns if c not in (ID_COLUMN, PRESENT_COLUMN)]
    if not columns:
        raise ValueError("report feature table has no feature columns")

    values = frame[columns].to_numpy(dtype=np.float32, copy=True)
    if not np.isfinite(values).all():
        raise ValueError(
            "report feature table contains a non-finite cell. It is fed straight into "
            "the teacher's report branch, so a NaN here becomes a NaN gradient with no "
            "exception raised and no diverging loss curve to notice."
        )
    present = np.asarray(
        np.where(pd.isna(frame[PRESENT_COLUMN]), False, frame[PRESENT_COLUMN]), dtype=bool
    )
    uids = [str(uid) for uid in frame[ID_COLUMN]]
    return {uid: position for position, uid in enumerate(uids)}, values, present


class ReportStudyDataset(StudyDataset):
    """:class:`StudyDataset` plus the per-study report vector. Training only."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        cache_dir: Path | str,
        *,
        report_features: pd.DataFrame,
        **kwargs: Any,
    ) -> None:
        super().__init__(manifest, cache_dir, **kwargs)
        self._report_index, self._report_values, self._report_present = _index_features(
            report_features
        )

    @property
    def report_dim(self) -> int:
        """Width of the report vector. Non-zero by construction."""
        return int(self._report_values.shape[1])

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = super().__getitem__(index)
        position = self._report_index.get(item["study_uid"])
        if position is None:
            # Not an error. The fold tables carry every study and the label
            # pipeline can fail on a report, so this is emitted as zeros with
            # present=False and the teacher gates the branch off rather than
            # reading a fabricated all-negative report.
            item["report"] = torch.zeros(self.report_dim, dtype=torch.float32)
            item["report_present"] = torch.zeros((), dtype=torch.bool)
        else:
            item["report"] = torch.from_numpy(self._report_values[position].copy())
            item["report_present"] = torch.tensor(bool(self._report_present[position]))
        return item


def build_study_dataset(
    manifest: pd.DataFrame,
    cache_dir: Path | str,
    *,
    report_features: pd.DataFrame | None = None,
    **kwargs: Any,
) -> StudyDataset:
    """The dataset this run needs: with the report vector, or without it.

    Two real implementations, so this selects rather than abstracts. An image-only
    run gets the exact :class:`StudyDataset` it got before any of this existed,
    which is what keeps the baseline a baseline.
    """
    if report_features is None:
        return StudyDataset(manifest, cache_dir, **kwargs)
    return ReportStudyDataset(manifest, cache_dir, report_features=report_features, **kwargs)


def _stack_report(samples: Sequence[Any], batch: dict[str, Any]) -> dict[str, Any]:
    if "report" in samples[0]:
        batch["report"] = torch.stack([s["report"] for s in samples])
        batch["report_present"] = torch.stack([s["report_present"] for s in samples])
    return batch


def build_report_collate(
    cfg: TrainConfig | AugmentConfig, *, seed: int = 0, training: bool = True
) -> Callable[[Any], dict[str, Any]]:
    """``build_collate``, extended to carry the report keys through.

    The pixel half is delegated, never reimplemented, so CutMix and the batch
    finiteness guard behave identically to an image-only run. CutMix is refused
    alongside report features by ``kappa.config``: it blends two studies' pixels
    and labels, and there is no honest way to blend their reports.
    """
    inner = build_collate(cfg, seed=seed, training=training)

    def collate(samples: Sequence[Any]) -> dict[str, Any]:
        return _stack_report(samples, inner(samples))

    return collate


def report_collate(samples: Sequence[Any]) -> dict[str, Any]:
    """Plain collate plus the report keys, for a scoring pass with no augmentation."""
    return _stack_report(samples, collate_studies(samples))
