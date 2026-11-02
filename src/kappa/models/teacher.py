"""The multimodal teacher: image plus report features, for distillation only.

**This model is TRAINING-ONLY and ``kappa.inference`` must never import it.** It reads
a per-study report feature vector from :mod:`kappa.labels.features`, and reports exist
for the 4,407 train studies and for none of the test studies. A
submission notebook that reached this class would not be slow, it would be a notebook
that cannot run. That is why this module is deliberately *not* re-exported from
``kappa.models.__init__``: that file is on the inference path and
``tests/test_loop.py::TestTheInferenceBoundary`` asserts no module on that path pulls
in ``kappa.labels`` at module scope. Import it by its full path, from the training
side, or not at all.

Why it exists: the research question is whether privileged report-derived information
available only at training time can be transferred into an image-only model. This
class is the source of the soft targets that carry it. It exists to be distilled from
and for no other purpose -- it is never evaluated as a submission candidate, because
it cannot be one.

The design is minimal on purpose, and each restraint has a reason.

* **The image branch is a whole :class:`~kappa.models.study_classifier.StudyClassifier`,
  reused unmodified.** ``image_logits`` is literally that model's own ``logits``, so
  the teacher's image branch is *architecturally identical to the student*. That is
  what makes the later comparison honest: a teacher that also had a bigger encoder
  would confound "the report helped" with "the encoder was bigger", and the encoder
  scaling question is already a measured null on this dataset.
* **Fusion is one concatenation and one Linear.** Concat plus a single
  ``Linear(embed_dim + report_hidden, 12)`` is the smallest fusion that can express
  "text overrides image on this label", because each label's weight row sees both
  halves at once. There is no cross-attention, no fusion-kind switch and no factory:
  those are arms of an ablation nobody has asked for yet, and every one of them would
  have to be paid for in runs.
* **The report branch is gated by presence, and the gating zeroes the *embedding*.**
  ``report_present=False`` means the extraction never parsed, so there is no evidence
  of any kind. Zeroing the *input* would not be enough: the first Linear's bias would
  survive it and the MLP would hand the fusion a fabricated report -- the same
  constant one for every study that has none. Zeroing the embedding after the MLP is
  what makes an absent report contribute exactly the fusion Linear's own bias.

One consequence worth stating. Under ``head="query"``, ``StudyClassifier`` documents
that its two pooling modules receive no gradient, because the query head reads the
slice tokens directly and nothing consumes the pooled study vector. Inside this class
the fusion *does* consume it, so those modules are trained here even with the query
head. The teacher and the student therefore do not have identical *gradient* paths
through the pool, only identical architecture.

    from kappa.models.teacher import TeacherClassifier

    teacher = TeacherClassifier(backbone="convnext_small", n_slices=16)
    out = teacher(batch["pixels"], batch["present"], batch["slot_ids"],
                  batch["report"], batch["report_present"])
    soft = out.logits          # what the student is distilled against
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from kappa.constants import N_LABELS
from kappa.labels.features import REPORT_FEATURE_DIM
from kappa.models.study_classifier import StudyClassifier

__all__ = ["TeacherOutput", "TeacherClassifier"]


@dataclass(frozen=True)
class TeacherOutput:
    """The fused prediction and the image branch's own, side by side.

    ``logits`` is what gets distilled -- the fused head, the only one that has seen
    the report. ``image_logits`` is the same tensor the student's architecture would
    produce from the same pixels, kept because the gap between the two is the
    measurement this whole model exists to produce: if fusing the report does not move
    the teacher above its own image branch, there is nothing to distil.
    """

    logits: Tensor
    image_logits: Tensor


class TeacherClassifier(nn.Module):
    """The student's architecture, plus a report branch and a concat fusion.

    Args:
        report_dim: width of the report vector, and the contract with
            :data:`kappa.labels.features.REPORT_FEATURE_DIM`. Defaults to it rather
            than restating 156, so the two cannot drift.
        backbone: passed straight to :class:`StudyClassifier`; see its docstring for
            why there is no size knob.
        n_slices: slices sampled per series, as the dataset emits them.
        pretrained: load the backbone's published weights. Stated in the startup
            print either way.
        drop_path_rate: stochastic depth in the backbone.
        head: ``"linear"`` or ``"query"`` for the image branch's own head. It sets
            what ``image_logits`` is; the fused head is a Linear regardless.
        head_num_heads: attention heads for ``head="query"``.
        report_hidden: width of the report MLP, and of its half of the fusion input.
        report_dropout: dropout inside the report MLP. Non-zero by default: this is
            the one branch that sees a feature vector derived from a *model's* reading
            of a report, and the teacher must not be able to memorise a study by its
            156-bit signature.
        freeze_backbone_bn: forwarded to the image branch's :class:`StudyClassifier`
            unchanged; see its docstring. Not part of :meth:`arch`, for the same
            reason ``pretrained`` is not: a teacher reloaded via ``load_frozen_teacher``
            is immediately put into eval mode and never trained again, so whether its
            BatchNorm layers were held frozen during ITS OWN training has no effect on
            reconstructing it for distillation.

    ``aux_dim`` is deliberately not exposed. The auxiliary head in
    :class:`StudyClassifier` exists to supervise the student with report-derived
    targets; here the report is an *input*, and wiring both would make the teacher
    predict what it was just told.
    """

    def __init__(
        self,
        *,
        report_dim: int = REPORT_FEATURE_DIM,
        backbone: str = "convnext_small",
        n_slices: int = 16,
        pretrained: bool = True,
        drop_path_rate: float = 0.2,
        head: str = "linear",
        head_num_heads: int = 8,
        report_hidden: int = 256,
        report_dropout: float = 0.1,
        freeze_backbone_bn: bool = False,
        img_size: int | None = None,
    ) -> None:
        super().__init__()
        if report_dim <= 0:
            raise ValueError(f"report_dim={report_dim} must be positive")
        if report_hidden <= 0:
            raise ValueError(f"report_hidden={report_hidden} must be positive")
        self.report_dim = int(report_dim)
        self.report_hidden = int(report_hidden)
        # Every constructor argument, kept so :meth:`arch` can hand a checkpoint
        # what it needs to rebuild this exact model. Reconstructing a teacher from
        # the *student's* config instead is a shape mismatch that only surfaces
        # once the fold is already running.
        self._arch = {
            "report_dim": int(report_dim),
            "backbone": backbone,
            "n_slices": int(n_slices),
            "drop_path_rate": float(drop_path_rate),
            "head": head,
            "head_num_heads": int(head_num_heads),
            "report_hidden": int(report_hidden),
            "report_dropout": float(report_dropout),
            # In `arch` and not merely on the constructor, because `load_teacher`
            # rebuilds from this dict alone. A ViT teacher trained at 392 and rebuilt
            # without the key is rebuilt at its native 518 and asserts on the first
            # forward of the distillation run -- one job after the mistake was made,
            # which is the worst place to find it.
            "img_size": None if img_size is None else int(img_size),
        }
        self.image = StudyClassifier(
            backbone=backbone,
            n_slices=n_slices,
            pretrained=pretrained,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            aux_dim=0,
            head=head,
            head_num_heads=head_num_heads,
            freeze_backbone_bn=freeze_backbone_bn,
        )
        self.report = nn.Sequential(
            nn.Linear(self.report_dim, self.report_hidden),
            nn.GELU(),
            nn.Dropout(report_dropout),
            nn.Linear(self.report_hidden, self.report_hidden),
            nn.GELU(),
        )
        embed_dim = self.image.encoder.embed_dim
        self.fuse = nn.Linear(embed_dim + self.report_hidden, N_LABELS)
        # Same initialisation argument as StudyClassifier's linear heads: small
        # weights, zero bias, and no prevalence priming -- macro ROC-AUC ranks within
        # a column, so a constant offset cannot move the score and would only make
        # the printed logits look calibrated.
        nn.init.trunc_normal_(self.fuse.weight, std=0.01)
        nn.init.zeros_(self.fuse.bias)

        print(self.describe())

    def arch(self) -> dict[str, Any]:
        """The constructor arguments that rebuild this model, for the checkpoint.

        ``pretrained`` is deliberately absent: a reload takes every weight from the
        file, so recording it would only invite asking a hub for weights that are
        already on disk.
        """
        return dict(self._arch)

    @property
    def data_config(self) -> dict[str, Any]:
        """The image branch's preprocessing config; the only sanctioned source.

        Delegated rather than re-resolved, so the teacher and the student are
        guaranteed to be fed identically normalised pixels.
        """
        return self.image.data_config

    @property
    def n_parameters(self) -> int:
        """Total parameter count, for the run log and for memory budgeting."""
        return sum(p.numel() for p in self.parameters())

    def describe(self) -> str:
        """Everything a reader needs to know what this run is evidence about.

        Includes the image branch's own description verbatim. That repeats what
        :class:`StudyClassifier` already printed from its constructor, and the
        repetition is the point: the teacher's block in the log has to be readable on
        its own, and a run that does not say what it ran is not evidence.
        """
        return "\n".join(
            [
                "TeacherClassifier  -- TRAINING ONLY (report features are train-only)",
                self.image.describe(),
                f"report     : {self.report_dim} features -> MLP {self.report_hidden} "
                f"(zeroed where report_present is False)",
                f"fusion     : concat({self.image.encoder.embed_dim} study + "
                f"{self.report_hidden} report) -> Linear -> {N_LABELS} logits",
                f"parameters : {self.n_parameters:,}",
            ]
        )

    def set_grad_checkpointing(self, enable: bool = True) -> None:
        """Trade backbone compute for activation memory. See :class:`SliceEncoder`."""
        self.image.set_grad_checkpointing(enable)

    def forward(
        self,
        pixels: Tensor,
        present: Tensor,
        slot_ids: Tensor,
        report: Tensor,
        report_present: Tensor,
    ) -> TeacherOutput:
        """Score one batch of studies with their reports.

        Args:
            pixels: ``(B, N_SLOTS, n_slices, 3, H, W)``.
            present: ``(B, N_SLOTS)`` bool slot presence.
            slot_ids: ``(B, N_SLOTS)`` long, global slot index.
            report: ``(B, report_dim)`` float, from
                :func:`kappa.labels.features.study_features`.
            report_present: ``(B,)`` bool. ``False`` means this study has no report
                at all, and its report embedding is zeroed before fusion.

        Returns:
            :class:`TeacherOutput` with ``logits`` and ``image_logits``, both ``(B, 12)``.
        """
        batch = pixels.shape[0]
        if report.ndim != 2 or report.shape[1] != self.report_dim:
            raise ValueError(
                f"report has shape {tuple(report.shape)}, expected (B, {self.report_dim}); "
                "the width is kappa.labels.features.REPORT_FEATURE_DIM"
            )
        if report.shape[0] != batch:
            raise ValueError(
                f"report has batch {report.shape[0]} but pixels has {batch}"
            )
        if report_present.ndim != 1 or report_present.shape[0] != batch:
            raise ValueError(
                f"report_present has shape {tuple(report_present.shape)}, expected {(batch,)}"
            )

        image = self.image(pixels, present, slot_ids)
        study = image.study_vector
        if study is None:  # pragma: no cover - both StudyClassifier paths populate it
            raise ValueError(
                "StudyClassifier returned study_vector=None; the teacher's fusion has "
                "nothing to fuse onto"
            )

        embedding = self.report(report)
        # Zero the *embedding*, not the input: the first Linear's bias would survive
        # an all-zero input, so a study with no report would be handed the same
        # fabricated report as every other study with no report.
        embedding = embedding * report_present.bool().unsqueeze(-1).to(embedding.dtype)
        fused = self.fuse(torch.cat([study, embedding], dim=-1))
        return TeacherOutput(logits=fused, image_logits=image.logits)
