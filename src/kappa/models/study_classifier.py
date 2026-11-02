"""The study-level model: 12 masked-loss logits, plus an optional report-supervised head.

Why this exists: this is the one object the training loop, the evaluation loop and the
Kaggle inference notebook all hold, so it is where the whole-study shape is pinned. It
takes the batch exactly as ``kappa.data.dataset.collate_studies`` emits it --
``pixels (B, N_SLOTS, n_slices, 3, H, W)``, ``present (B, N_SLOTS)``,
``slot_ids (B, N_SLOTS)`` -- and returns :class:`ModelOutput`. Nothing else about the
architecture is visible to its consumers.

**It prints which backbone it actually loaded, at construction.** That is a rule
learned the hard way rather than a nicety: a run that does not state its backbone,
whether the weights were pretrained or random, and what preprocessing that backbone
expects is not evidence about that backbone. The single most expensive version of this
mistake is normalisation: a model whose weights want ``mean=0.5`` fed ImageNet
constants still trains, still prints a loss curve, and still reports a number -- one
that supports a false conclusion. So the print carries the preprocessing config, and
:attr:`data_config` is the value the dataset must be constructed from.

Three deliberate non-features.

* **No size knob.** Scaling the encoder is a measured null here: DINOv2-S to DINOv2-B
  was worth +0.0011 against a 0.0020 noise floor and improved 5 of 12 labels. Four
  RSNA competitions agree; RSNA 2024's winner measured convnext-large < convnext-base
  < convnext-small. The default is ``convnext_small`` and the way to spend compute is
  input geometry -- the physical-millimetre crop -- not parameters. ``img_size`` is
  that sanctioned axis and not a breach of this rule: it changes what the model is
  shown, not how much model there is, and for a ViT it is not a knob at all but a
  requirement -- ``vit_base_patch14_dinov2`` is published at 518 px and this project
  feeds 392, so the position grid has to be built at 392 or the forward pass asserts.
* **No sigmoid, no calibration.** The metric is macro ROC-AUC, which is rank-based and
  threshold-free, so logits are the natural output and probability calibration cannot
  change the score. Ensembling is per-column rank normalisation, done downstream.
* **No loss.** The head emits all 12 columns always, including the ones masked out for
  this study; 98.7% of raw label cells are unknown, and it is
  ``kappa.training.losses.AsymmetricLoss`` that consumes the mask.

``aux_dim`` wires an auxiliary head onto the study vector for report supervision.
Reports are train-only (test studies carry none), so they can supervise what the 12 columns
cannot: the anatomy clusters and their polarity, including the 17 clusters that map to
none of the 12 labels -- PCL, patellar and quadriceps tendon, LCL, MPFL, IT band,
retinaculum, popliteus, fat pad. Those structures are visible in the same pixels and
their presence is evidence about the neighbouring joint, so predicting them is free
signal on 4,407 studies rather than the 58 that carry labels. The head is wired here
and left unsupervised; a later phase supplies the targets.

    from kappa.models.study_classifier import StudyClassifier

    model = StudyClassifier(backbone="convnext_small", n_slices=16, aux_dim=0)
    out = model(batch["pixels"], batch["present"], batch["slot_ids"])
    loss = criterion(out.logits, batch["labels"], batch["mask"])
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import Tensor, nn

from kappa.constants import LABELS, N_LABELS
from kappa.models.aggregator import LabelQueryHead, SeriesAggregator, StudyAggregator
from kappa.models.encoder import N_SLOTS, VIDEO_BACKBONES, SliceEncoder

__all__ = ["ModelOutput", "StudyClassifier"]


@dataclass(frozen=True)
class ModelOutput:
    """Logits for the 12 scored labels, plus the optional auxiliary head.

    Logits, not probabilities: see the module docstring. ``aux_logits`` is ``None``
    whenever ``aux_dim=0``, so a consumer that forgets to check gets a ``TypeError``
    rather than a silently ignored tensor of zeros.

    ``study_vector`` is the pooled ``(B, dim)`` study vector, exposed so that
    :class:`kappa.models.teacher.TeacherClassifier` can fuse report features onto it;
    it is ``None`` for nothing currently -- both ``forward`` paths always populate it.
    """

    logits: Tensor
    aux_logits: Tensor | None = None
    study_vector: Tensor | None = None


class StudyClassifier(nn.Module):
    """One shared encoder, two attention pools, twelve masked logits.

    Args:
        backbone: any timm name; see ``RECOMMENDED_BACKBONES``. Asking for pretrained
            weights that do not exist raises rather than returning random ones.
        n_slices: slices sampled per series. Sizes the positional table; a batch with
            a different count is handled by interpolation, not by failing.
        pretrained: load the backbone's published weights. ``False`` is stated
            explicitly in the startup print so a random-init run cannot be mistaken
            for a pretrained one.
        drop_path_rate: stochastic depth in the backbone. The only regulariser this
            model exposes, matching ``ModelConfig``.
        aux_dim: width of the report-supervision head; 0 disables it.
        head: ``"linear"`` -- one ``nn.Linear(dim, N_LABELS)`` over the pooled study
            vector, where the twelve labels share every parameter but a final weight
            row. ``"query"`` -- :class:`LabelQueryHead`, one learned query per label
            cross-attending the slice tokens, which is the only one of the two where
            "per-label" means anything.

            **With ``head="query"`` and ``aux_dim=0`` the two pooling modules receive
            no gradient**: the query head reads the encoder's slice tokens directly,
            so nothing downstream consumes the pooled study vector. They are still
            built and still run -- their cost is negligible beside the encoder, and
            keeping them makes the ``aux_dim > 0`` path identical across both heads --
            but they sit in the checkpoint at their initial values. That is expected,
            not a load bug.
        head_num_heads: attention heads for ``head="query"``; ignored by ``linear``.
        freeze_backbone_bn: hold the encoder's ``BatchNorm`` layers in eval mode
            throughout training. Forwarded to the encoder only when it is
            :class:`kappa.models.volume_encoder.VolumeEncoder`, which is the sole
            encoder wired to honour it (see its ``train()``); ``kappa.config``
            refuses this flag on any other backbone before a run ever reaches here,
            so ``SliceEncoder`` never has to know the argument exists.
        img_size: build the backbone for this input size instead of its native one.
            ``None`` means "do not pass the argument at all", which is the only correct
            value for a convolutional backbone and for every arm measured so far. A ViT
            needs it, and the value must be the crop the dataset emits -- ``data.size``
            in the config, which ``TrainConfig`` already refuses to let disagree with
            the backbone family: 392 for the patch-14 DINOv2 weights, published at 518.
            What was actually applied is printed at construction by
            :class:`kappa.models.encoder.BackboneInfo`, the refused case included,
            because a family that does not accept the argument builds at its native
            geometry and no other line of the log would say so.
        per_label_slots: give each of the twelve labels its own softmax over the
            present slots, ``n_labels x n_slots`` extra weights in
            :class:`StudyAggregator`, **added to** the shared pooled vector rather
            than replacing it. The shared pool stays exactly where it was: the
            published ablation puts class-agnostic pooling alone at 82.1,
            class-specific alone at 84.2 and both together at 85.3, so replacing is
            the losing arrangement and the residual is not a detail of this
            implementation. ``False`` leaves the module structurally identical --
            same submodules, same parameter names, same arithmetic -- so every
            checkpoint written before this argument existed still loads strictly.
        head_residual: add the pooled study vector into what :class:`LabelQueryHead`
            reads, and therefore **requires** ``head="query"``; passing it with the
            linear head raises here rather than being ignored. It exists because of
            the paragraph above about ``head="query"`` with ``aux_dim=0``: without it
            the two pools receive no gradient at all and sit in the checkpoint at
            their initial values. See :meth:`forward` for where the addition happens
            and :class:`kappa.config.PoolConfig` for what it is worth.
    """

    def __init__(
        self,
        backbone: str = "convnext_small",
        n_slices: int = 16,
        pretrained: bool = True,
        drop_path_rate: float = 0.2,
        aux_dim: int = 0,
        head: str = "linear",
        head_num_heads: int = 8,
        freeze_backbone_bn: bool = False,
        img_size: int | None = None,
        per_label_slots: bool = False,
        head_residual: bool = False,
    ) -> None:
        super().__init__()
        if aux_dim < 0:
            raise ValueError(f"aux_dim={aux_dim} must be non-negative")
        if head not in ("linear", "query"):
            raise ValueError(f"unknown head {head!r}; known: 'linear', 'query'")
        if head_residual and head != "query":
            # `kappa.config` refuses this pair before a job is ever submitted; this is
            # the same refusal for the direct constructor, because the linear head
            # reads the pooled study vector and nothing else -- there is nothing to add
            # it back to, and silently ignoring the flag would make a run that claims
            # the arm indistinguishable from the baseline.
            raise ValueError(
                f"head_residual=True with head={head!r}; the residual adds the pooled "
                "study vector into LabelQueryHead, which only exists under head='query'"
            )
        # The backbone NAME is the whole switch between the 2.5D and volumetric arms.
        # There is deliberately no second `mode` argument: two knobs that can disagree
        # is how a config comes to describe a run that did not happen. A timm name
        # takes the path that has always existed, unchanged.
        self.volumetric = backbone.split(".", 1)[0] in VIDEO_BACKBONES
        if self.volumetric:
            # Imported here rather than at module scope so that `kappa.models` still
            # imports on a machine with timm but without torchvision -- the 2.5D arms,
            # the tests that build them and the inference path must not acquire a
            # dependency because a 3D ablation exists.
            from kappa.models.volume_encoder import VolumeEncoder

            self.encoder: nn.Module = VolumeEncoder(
                backbone,
                pretrained=pretrained,
                img_size=img_size,
                freeze_backbone_bn=freeze_backbone_bn,
            )
        else:
            self.encoder = SliceEncoder(
                backbone,
                pretrained=pretrained,
                drop_path_rate=drop_path_rate,
                img_size=img_size,
            )
        dim = self.encoder.embed_dim
        self.n_slices = int(n_slices)
        self.aux_dim = int(aux_dim)
        # The volumetric encoder has ALREADY consumed the slice axis, so it emits one
        # token per slot and the series pool sees a length-1 sequence. Sizing the
        # positional table at n_slices there would build 32 entries of which exactly
        # one is ever read, and `SeriesAggregator.positional` would interpolate 32 -> 1
        # on every forward to reach it. Both are wrong the same way: they describe a
        # slice axis this arm does not have.
        self.series_pool = SeriesAggregator(
            dim=dim, n_slices=1 if self.volumetric else self.n_slices
        )
        self.per_label_slots = bool(per_label_slots)
        self.head_residual = bool(head_residual)
        self.study_pool = StudyAggregator(
            dim=dim, n_slots=N_SLOTS, per_label=N_LABELS if self.per_label_slots else 0
        )
        self.head_kind = str(head)
        if self.head_kind == "query":
            self.head = LabelQueryHead(dim=dim, n_labels=N_LABELS, num_heads=head_num_heads)
        else:
            self.head = nn.Linear(dim, N_LABELS)
        self.aux_head = nn.Linear(dim, self.aux_dim) if self.aux_dim else None

        # Small weights and a zero bias on the fresh linear heads. The bias is
        # deliberately not primed with per-label prevalence: a constant offset shifts
        # an entire column equally, and macro ROC-AUC ranks within a column, so
        # priming it cannot move the score -- it would only make the printed logits
        # look calibrated, which is the appearance this project does not want.
        # LabelQueryHead initialises its own parameters; re-initialising them here
        # would silently overwrite the query table with a head-shaped assumption.
        linear_heads = [self.aux_head]
        if self.head_kind != "query":
            linear_heads.append(self.head)
        for linear in linear_heads:
            if linear is not None:
                nn.init.trunc_normal_(linear.weight, std=0.01)
                nn.init.zeros_(linear.bias)

        print(self.describe())

    @property
    def data_config(self) -> dict[str, Any]:
        """The backbone's own preprocessing config.

        Construct the dataset from this, via
        ``Normalisation.from_data_config(model.data_config)``. Reading it from here
        is the only way the pixels and the weights are guaranteed to agree.
        """
        return self.encoder.data_config

    @property
    def n_parameters(self) -> int:
        """Total parameter count, for the run log and for memory budgeting."""
        return sum(p.numel() for p in self.parameters())

    def describe(self) -> str:
        """Everything a reader needs to know what this run is evidence about."""
        aux = "disabled" if not self.aux_dim else f"{self.aux_dim} report-supervised units"
        if self.head_kind == "query":
            shape = (
                f"{N_LABELS} learned queries x {self.head.num_heads} attn heads over slice tokens"
            )
        else:
            shape = f"one shared Linear({self.encoder.embed_dim}, {N_LABELS})"
        return "\n".join(
            [
                "StudyClassifier",
                self.encoder.describe(),
                (
                    f"pooling    : {self.n_slices}-slice VOLUME -> 1 token/series -> "
                    f"{N_SLOTS} slots (presence-masked) -> study"
                    if self.volumetric
                    else f"pooling    : {self.n_slices} slices/series -> series -> "
                    f"{N_SLOTS} slots (presence-masked) -> study"
                ),
                f"slot pool  : {self._pooling_description()}",
                f"head       : {self.head_kind} -- {shape}",
                f"heads      : {N_LABELS} labels ({', '.join(LABELS[:3])}, ...)  aux={aux}",
                f"parameters : {self.n_parameters:,}",
            ]
        )

    def _pooling_description(self) -> str:
        """Which slot pooling ran and who reads it, for the run log.

        Printed at every setting, default included. The default -- one shared pool,
        twelve labels reading one set of plane weights -- is an assumption rather than
        the absence of one, and a log that only mentions pooling when it is unusual
        cannot be read backwards into what the run was evidence about.
        """
        parts = [
            f"{N_LABELS} per-label softmaxes over {N_SLOTS} slots "
            f"({N_LABELS * N_SLOTS} weights), residual on the shared pool"
            if self.per_label_slots
            else f"1 shared pool over {N_SLOTS} slots -- all {N_LABELS} labels "
            "read the same plane weights"
        ]
        if self.head_residual:
            parts.append("study vector added into the query head (pools trained)")
        elif self.head_kind == "query" and not self.aux_dim:
            parts.append("NOT read by the head: series_pool/study_pool get no gradient")
        return "  |  ".join(parts)

    def set_grad_checkpointing(self, enable: bool = True) -> None:
        """Trade backbone compute for activation memory. See :class:`SliceEncoder`."""
        self.encoder.set_grad_checkpointing(enable)

    def forward(self, pixels: Tensor, present: Tensor, slot_ids: Tensor) -> ModelOutput:
        """Score one batch of studies.

        Args:
            pixels: ``(B, N_SLOTS, n_slices, 3, H, W)``.
            present: ``(B, N_SLOTS)`` bool. ``False`` means absent, and an absent slot
                provably cannot influence the output.
            slot_ids: ``(B, N_SLOTS)`` long, global slot index.

        Returns:
            :class:`ModelOutput` with ``logits (B, 12)``.
        """
        embeddings = self.encoder(pixels, slot_ids, present)
        batch, slots, n_slices, dim = embeddings.shape
        # One flat call rather than a loop over slots: the series pool never attends
        # across slots, so folding them into the batch axis is exact and lets the
        # whole study go through in one shot.
        series = self.series_pool(embeddings.reshape(batch * slots, n_slices, dim))
        study, _, per_label = self.study_pool.forward_with_per_label(
            series.view(batch, slots, dim), present, slot_ids
        )
        aux = self.aux_head(study) if self.aux_head is not None else None
        if self.head_kind == "query":
            # The slice tokens, not the pooled vector: a per-label query that could
            # only read one summary would have nothing extra to attend to. Presence
            # is per slot, so it repeats across that slot's slices.
            tokens = embeddings.reshape(batch, slots * n_slices, dim)
            token_mask = (
                present.bool().unsqueeze(-1).expand(batch, slots, n_slices).reshape(batch, -1)
            )
            # The residual is what the pools contribute to a head that would otherwise
            # never read them. `per_label` is the increment ALONE here, not
            # `study + per_label`: the query head never read `study`, so the two flags
            # add two independent terms and setting both does not count `study` twice.
            residual = per_label
            if self.head_residual:
                shared = study.unsqueeze(1)
                residual = shared if residual is None else residual + shared
            return ModelOutput(
                logits=self.head(tokens, token_mask, residual), aux_logits=aux, study_vector=study
            )
        if per_label is not None:
            # A per-label readout of a per-label vector, written out because
            # `nn.Linear` cannot express it. `self.head` holds `weight (12, D)` and
            # `bias (12,)`; the linear head computes `logits[b, l] = study[b] . W[l] +
            # bias[l]`, one shared vector dotted against twelve rows. Here each label
            # brings its own vector:
            #
            #     v[b, l] = study[b] + per_label[b, l]          (B, 12, D)
            #     logits[b, l] = v[b, l] . W[l] + bias[l]       row-wise, not shared
            #
            # which is `(v * W).sum(-1) + bias` and NOT `self.head(v)` -- that would
            # score every label's vector against every label's row and return
            # (B, 12, 12). The parameters are the same `self.head`, so this arm adds
            # no readout capacity at all; the only new weights are the 72 in the slot
            # table. The degenerate case is algebraically exact but NOT bit-exact:
            # with `per_label` identically zero, `v[b, l] = study[b]` for all l and
            # this reduces to `F.linear(study, W, bias)` as an expression, while the
            # two contract the `dim` axis in different orders. Measured on random
            # inputs: max |difference| 3.0e-8 at dim 64 and 6.6e-7 at dim 768, which
            # is float32 reduction order and not a different computation. It is worth
            # stating rather than rounding away, because the `per_label_slots=False`
            # path below does NOT go through this branch at all and IS bit-identical
            # to what every existing checkpoint was trained under.
            vector = study.unsqueeze(1) + per_label
            logits = (vector * self.head.weight).sum(dim=-1) + self.head.bias
            return ModelOutput(logits=logits, aux_logits=aux, study_vector=study)
        return ModelOutput(logits=self.head(study), aux_logits=aux, study_vector=study)
