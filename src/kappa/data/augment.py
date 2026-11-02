"""Training-time augmentation: per-study geometry and intensity, plus batch CutMix.

Why this module is separate from the dataset: the two hooks it fills run in
different places for a reason. ``StudyDataset(augment=...)`` sees one study's
pixels and can rotate them; CutMix mixes *two* studies and therefore has to run
where both are in scope, which is ``collate_studies(batch_augment=...)``. Wiring
CutMix per-sample is the standard way to get it wrong -- the labels of the study
it borrowed from are not in scope, so the pixels mix and the labels do not.

    from kappa.data.augment import build_augment, build_batch_augment

    train = StudyDataset(..., training=True, augment=build_augment(cfg))
    loader = DataLoader(train, collate_fn=build_collate(cfg, seed=0))

**Everything here is train-only and off unless a config asks for it.** Published
write-ups put rotation at +0.025 CV and CutMix ahead of flip, rotate, brightness,
contrast, blur and mixup on a controlled table -- but both are measured on other
datasets, so they arrive as ablation arms rather than as defaults.

Three things in here are load-bearing and easy to get subtly wrong.

**The label and the mask must move together.** CutMix mixes ``labels`` toward the
borrowed study; if ``mask`` does not move with them, the loss trains a cell that
is only partly known as though it were fully known, and nothing raises. Where the
two studies disagree about whether a cell is known, the mixed cell is masked
*out* -- the conservative direction, since the alternative applies one study's
target to an image the other study partly painted. On the label table as it
stands this is a no-op: all 4,407 studies are present with all twelve cells
known, so the intersection retains 100% of known cells. It is written for a
label table that leaves some cells unknown, not the one that exists.

**An absent slot is not a black slot.** Pasting a patch from a study whose slot is
absent would paint a black rectangle into a slot still flagged ``present``, which
is precisely the confusion ``series_dropout`` exists to prevent. So a patch is
only ever pasted into slots that *both* studies have, and a pair with no shared
slot is left unmixed.

**Rotation fills with zero, which is the mean pixel, not black.** The hook runs
after the backbone's normalisation, so 0.0 is the dataset mean rather than an
intensity the scanner could produce. Rotating before normalisation would push the
corners to a value the model reads as signal.
"""

from __future__ import annotations

import functools
import hashlib
from typing import Any, Callable, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from kappa.config import AugmentConfig, TrainConfig

__all__ = [
    "StudyAugment",
    "CutMix",
    "PerBatchStream",
    "rotate",
    "build_augment",
    "build_batch_augment",
    "build_collate",
    "batch_seed_from_uids",
]


def rotate(pixels: Tensor, degrees: float) -> Tensor:
    """Rotate ``(..., H, W)`` about its centre, preserving the shape exactly.

    ``StudyDataset`` raises if the augment hook changes the crop, and it is right
    to: the crop is a physical measurement (``physical_crop_mm / size`` mm/px), so
    a transform that resized it would silently change the scale the run reports.
    Rotation by ``grid_sample`` maps the square onto itself and cannot.

    Corners vacated by the rotation are filled with 0.0. Downstream of the
    backbone's normalisation that is the mean pixel value, which is the least
    informative fill available -- and notably not black, which would be a real
    intensity.
    """
    if degrees == 0.0:
        return pixels
    lead, height, width = pixels.shape[:-2], pixels.shape[-2], pixels.shape[-1]
    flat = pixels.reshape(-1, 1, height, width)
    theta = torch.tensor(float(np.deg2rad(degrees)), dtype=pixels.dtype)
    cos, sin = torch.cos(theta), torch.sin(theta)
    # affine_grid maps output coords to input coords, so this is the inverse
    # rotation; the translation column is zero because the centre is the origin
    # in the normalised (-1, 1) coordinates grid_sample uses.
    matrix = torch.tensor(
        [[cos, -sin, 0.0], [sin, cos, 0.0]], dtype=pixels.dtype, device=pixels.device
    ).expand(flat.shape[0], 2, 3)
    grid = F.affine_grid(matrix, list(flat.shape), align_corners=False)
    turned = F.grid_sample(
        flat, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    return turned.reshape(*lead, height, width)


class StudyAugment:
    """The per-study hook: rotation, then brightness, then contrast.

    Matches the signature ``StudyDataset`` expects, ``(pixels, rng) -> pixels``,
    and never changes the crop. Called once per *slot*, sharing the study's
    generator, so two slots of one study get independent draws -- they are
    separate acquisitions and there is nothing to keep consistent between them.

    Intensity is adjusted in normalised space, where 0.0 is the mean and 1.0 is
    one standard deviation. ``brightness`` is therefore an additive shift in
    units of sigma and ``contrast`` a gain about the mean, which is what makes
    both independent of whatever normalisation the backbone happens to resolve.
    """

    def __init__(
        self, rotate_deg: float = 0.0, brightness: float = 0.0, contrast: float = 0.0
    ) -> None:
        for name, value in (("rotate_deg", rotate_deg), ("brightness", brightness), ("contrast", contrast)):
            if value < 0.0:
                raise ValueError(f"{name}={value} is a half-width and must be non-negative")
        self.rotate_deg = float(rotate_deg)
        self.brightness = float(brightness)
        self.contrast = float(contrast)

    @property
    def enabled(self) -> bool:
        return self.rotate_deg > 0.0 or self.brightness > 0.0 or self.contrast > 0.0

    def __call__(self, pixels: Tensor, rng: np.random.Generator) -> Tensor:
        if self.rotate_deg > 0.0:
            pixels = rotate(pixels, float(rng.uniform(-self.rotate_deg, self.rotate_deg)))
        if self.contrast > 0.0:
            pixels = pixels * float(1.0 + rng.uniform(-self.contrast, self.contrast))
        if self.brightness > 0.0:
            pixels = pixels + float(rng.uniform(-self.brightness, self.brightness))
        return pixels

    def __repr__(self) -> str:
        return (
            f"StudyAugment(rotate_deg={self.rotate_deg}, "
            f"brightness={self.brightness}, contrast={self.contrast})"
        )


def _cut_box(
    height: int, width: int, lam: float, rng: np.random.Generator
) -> tuple[int, int, int, int]:
    """Corners of a patch covering ``1 - lam`` of the frame, clipped to the frame.

    Clipping is why the caller must recompute ``lam`` from the box that came back
    rather than trusting the one it asked for: a box centred near an edge loses
    part of its area, and using the requested ratio to mix the labels would
    describe a mix that did not happen.
    """
    ratio = float(np.sqrt(max(0.0, 1.0 - lam)))
    box_h, box_w = int(round(height * ratio)), int(round(width * ratio))
    centre_y, centre_x = int(rng.integers(height)), int(rng.integers(width))
    y0 = int(np.clip(centre_y - box_h // 2, 0, height))
    y1 = int(np.clip(centre_y + box_h - box_h // 2, 0, height))
    x0 = int(np.clip(centre_x - box_w // 2, 0, width))
    x1 = int(np.clip(centre_x + box_w - box_w // 2, 0, width))
    return y0, y1, x0, x1


class CutMix:
    """Paste a rectangle from another study, and move the labels with it.

    Runs in ``collate_studies``' ``batch_augment`` hook. The batch it receives has
    ``pixels (B, N_SLOTS, n_slices, 3, H, W)``, ``present (B, N_SLOTS)``,
    ``labels (B, 12)`` and ``mask (B, 12)``; the keys are ``labels`` and ``mask``,
    which are *not* the label table's ``label`` / ``label_mask`` column names.

    One box is drawn per batch and applied to every study, which is how the
    measured recipe does it and keeps the mixing weight a single number per pair.
    """

    def __init__(self, prob: float = 1.0, alpha: float = 1.0) -> None:
        if not 0.0 <= prob <= 1.0:
            raise ValueError(f"prob={prob} is a probability")
        if alpha <= 0.0:
            raise ValueError(f"alpha={alpha} must be positive; it parameterises a Beta")
        self.prob = float(prob)
        self.alpha = float(alpha)

    @property
    def enabled(self) -> bool:
        return self.prob > 0.0

    def __call__(
        self, batch: dict[str, Any], rng: np.random.Generator
    ) -> dict[str, Any]:
        if self.prob <= 0.0 or float(rng.random()) >= self.prob:
            return batch
        pixels, present = batch["pixels"], batch["present"]
        labels, mask = batch["labels"], batch["mask"]
        size = pixels.shape[0]
        if size < 2:
            return batch  # nothing to mix with; mixing a study into itself is a no-op

        order = torch.from_numpy(rng.permutation(size)).to(pixels.device)
        height, width = int(pixels.shape[-2]), int(pixels.shape[-1])
        y0, y1, x0, x1 = _cut_box(height, width, float(rng.beta(self.alpha, self.alpha)), rng)
        if y1 <= y0 or x1 <= x0:
            return batch  # a degenerate box mixes nothing; do not pretend otherwise

        # An absent slot holds exact zeros. Pasting from one would paint a black
        # rectangle into a slot still flagged present, which is the exact confusion
        # series_dropout exists to prevent -- so only shared slots are mixed.
        shared = present & present[order]
        mixed_study = shared.any(dim=1)
        if not bool(mixed_study.any()):
            return batch

        pixels = pixels.clone()
        donor = pixels[order]
        rows = torch.nonzero(shared, as_tuple=False)
        for study, slot in rows.tolist():
            pixels[study, slot, ..., y0:y1, x0:x1] = donor[study, slot, ..., y0:y1, x0:x1]

        # The weight the *pixels* actually got, recomputed from the clipped box.
        area = float((y1 - y0) * (x1 - x0)) / float(height * width)
        lam = 1.0 - area
        # (B, 1) so it broadcasts across the twelve labels. Studies with no shared
        # slot kept their pixels, so their weight is exactly 1 and their labels are
        # left alone -- the weight has to agree with what the pixels actually did.
        per_study = mixed_study.unsqueeze(1)
        weight = torch.where(
            per_study, torch.full_like(labels[:, :1], lam), torch.ones_like(labels[:, :1])
        )

        # Labels move with the pixels, and the mask moves with the labels. A cell
        # only one of the pair knows is masked out: the alternative trains that
        # study's target against an image the other study partly painted.
        mixed_labels = weight * labels + (1.0 - weight) * labels[order]
        keep = mask & mask[order]
        mixed_mask = torch.where(per_study, keep, mask)
        # A masked-out cell must carry 0.0, never a stale or mixed value: the batch
        # contract is that `labels` is finite everywhere and `mask` alone decides.
        mixed_labels = torch.where(mixed_mask, mixed_labels, torch.zeros_like(mixed_labels))

        out = dict(batch)
        out["pixels"] = pixels
        out["labels"] = mixed_labels
        out["mask"] = mixed_mask
        return out

    def __repr__(self) -> str:
        return f"CutMix(prob={self.prob}, alpha={self.alpha})"


def batch_seed_from_uids(uids: Any, seed: int) -> np.random.Generator:
    """A generator that differs per batch and is still reproducible.

    ``collate_studies`` builds ``np.random.default_rng(seed)`` from a *constant*
    seed on every call, so a hook that used it directly would draw the identical
    box and the identical permutation for every batch of every epoch -- CutMix
    reduced to one fixed pattern, with no error and a plausible loss curve. The
    batch's own study ids are the entropy that fixes it: they differ per batch,
    they are reshuffled every epoch, and they are identical across DataLoader
    workers, so the stream is a deterministic function of the run's seed and the
    batch's contents rather than of the process it landed in.
    """
    # blake2b, not hash(): Python salts str.__hash__ per process from PYTHONHASHSEED,
    # so the builtin would give a different stream on every run -- and a different one
    # per DataLoader worker under spawn -- while this docstring promised reproducible.
    joined = "\x00".join(str(u) for u in uids).encode("utf-8")
    digest = int.from_bytes(hashlib.blake2b(joined, digest_size=8).digest(), "big")
    return np.random.default_rng([int(seed), digest])


def build_augment(cfg: TrainConfig | AugmentConfig) -> Callable[[Tensor, np.random.Generator], Tensor] | None:
    """The per-study hook a config asks for, or ``None`` when it asks for nothing.

    ``None`` rather than an identity callable, so that ``StudyDataset`` skips the
    hook and its shape check entirely and an unaugmented run is bit-identical to
    one built before this module existed.
    """
    augment = cfg.augment if isinstance(cfg, TrainConfig) else cfg
    if not augment.per_sample_enabled:
        return None
    return StudyAugment(
        rotate_deg=augment.rotate_deg,
        brightness=augment.brightness,
        contrast=augment.contrast,
    )


class PerBatchStream:
    """Adapter giving the wrapped hook a generator that actually varies per batch.

    ``collate_studies`` constructs ``np.random.default_rng(seed)`` from a constant
    on every call, which is correct for a hook that wants one fixed stream and
    silently wrong for CutMix: every batch of every epoch would draw the same box
    and the same permutation, and the run would look exactly like a working one.

    So the generator handed in is **deliberately ignored** in favour of one keyed
    on the batch's own study ids -- see :func:`batch_seed_from_uids`. This is
    stated loudly rather than done quietly because "ignores an argument it was
    passed" is otherwise indistinguishable from a bug.
    """

    def __init__(
        self,
        hook: Callable[[dict[str, Any], np.random.Generator], dict[str, Any]],
        seed: int = 0,
    ) -> None:
        self.hook = hook
        self.seed = int(seed)

    def __call__(self, batch: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
        del rng  # see the class docstring; the batch's ids are the entropy
        return self.hook(batch, batch_seed_from_uids(batch["study_uid"], self.seed))

    def __repr__(self) -> str:
        return f"PerBatchStream({self.hook!r}, seed={self.seed})"


def build_batch_augment(
    cfg: TrainConfig | AugmentConfig, *, seed: int = 0
) -> Callable[[dict[str, Any], np.random.Generator], dict[str, Any]] | None:
    """The batch hook a config asks for, or ``None``. See :func:`build_augment`."""
    augment = cfg.augment if isinstance(cfg, TrainConfig) else cfg
    if not augment.batch_enabled:
        return None
    return PerBatchStream(CutMix(prob=augment.cutmix_prob, alpha=augment.cutmix_alpha), seed=seed)


def build_collate(
    cfg: TrainConfig | AugmentConfig, *, seed: int = 0, training: bool = True
) -> Callable[[Any], dict[str, Any]]:
    """The ``collate_fn`` for a loader, with CutMix bound only when ``training``.

    ``training=False`` returns bare ``collate_studies``. Validation and inference
    must see the pixels the model will be judged on, and a batch hook that leaked
    into the validation loader would mix the labels the score is computed against.
    """
    from kappa.data.dataset import collate_studies

    hook = build_batch_augment(cfg, seed=seed) if training else None
    if hook is None:
        return collate_studies
    return functools.partial(collate_studies, batch_augment=hook)
