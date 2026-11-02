"""One shared 2D encoder for all six plane x contrast slots, conditioned on the slot.

Why this exists: the obvious build is six encoders, one per ``plane x contrast`` slot.
That choice silently divides the training set. There are **4,407** train studies
carrying roughly 5.5 series each, so a shared encoder is fed on the order of 24,000
slice stacks while six per-slot encoders each see at most 4,407 -- and the
``Axial_fs0`` one sees 19.4% of that, about 855 studies. Sharing the weights and
passing the slot in as *conditioning* keeps every sample available to every
parameter, which is the whole argument for this module existing rather than a
``nn.ModuleDict`` of six backbones.

Two silent failures it is built to prevent.

**1. Normalisation invented rather than read.** The per-channel mean and standard
deviation are taken from timm's own resolved data config and are never written down
in this file. There is no ``IMAGENET_MEAN`` here on purpose. Measured this session
with ``timm.data.resolve_data_config``: ``convnext_small`` resolves to
``mean=(0.485, 0.456, 0.406)`` at 224 px, while ``maxvit_tiny_tf_384`` resolves to
``mean=(0.5, 0.5, 0.5)`` at 384 px. Hardcoding either one gives the other backbone a
run that trains, prints a loss curve and reports a number that means nothing.
:func:`backbone_data_config` is the only supported source, and
``kappa.data.dataset.Normalisation.from_data_config`` consumes it directly.

**2. Absent-slot pixels reaching the backbone.** Only ``Axial_fs1`` is present in
every study; ``Axial_fs0`` is present in 19.4%, and there are 12 distinct
availability patterns. An absent slot still occupies its place in the rectangular
batch, filled with zeros. The cheap build runs those zeros through the encoder and
masks them later, which is *not* equivalent: with a BatchNorm backbone in training
mode the batch statistics are pooled across the flattened slice axis, so the content
of an absent slot moves the embedding of every present slot in the same batch.
Measured this session: ``tf_efficientnetv2_s`` contains **110** BatchNorm layers
(``convnext_small`` contains 0) and is a listed diversity member, and a mask-later
variant of this encoder shifted the logits by up to **0.364** when an absent slot was
filled with garbage, through a BatchNorm backbone in training mode. The gathering
version stayed bit-identical. :class:`SliceEncoder` therefore *gathers* the present
slots, runs the backbone on those rows only, and scatters exact zeros back into the
absent ones. The absent slot cannot influence the output because it is never
computed, not because a later multiplication happened to be by zero.

Scaling the encoder is a measured null on this dataset: DINOv2-S to DINOv2-B was
worth +0.0011 against a 0.0020 noise floor and improved 5 of 12 labels. There is
deliberately no width or depth knob here. Scale the input geometry instead -- the
physical-millimetre crop in ``kappa.data.dataset`` -- and keep the backbone small.

``img_size`` is the construction-time half of that same lever, and it is the one
argument a ViT cannot be built without: ``vit_base_patch14_dinov2`` is published at
518 px with patch 14, this project's data config feeds **392** (= 14 x 28; 384 / 14 =
27.43 is not an integer, which is why the usual 384 is not an option), and a ViT built
at 518 does not adapt to 392 on its own. Measured this session on the small sibling,
``vit_small_patch14_dinov2``: built at its native size and handed a 392 px batch, timm
raises ``AssertionError: Input height (392) doesn't match model (518)``. Passing
``img_size`` at construction is what lets timm interpolate the pretrained position
embeddings once, into a 28 x 28 grid, instead of per forward pass or not at all.

    from kappa.models.encoder import SliceEncoder

    enc = SliceEncoder(backbone="convnext_small", pretrained=True)
    print(enc.describe())                       # states what was actually loaded
    emb = enc(pixels, slot_ids, present)        # (B, N_SLOTS, n_slices, embed_dim)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Mapping

import timm
import torch
from torch import Tensor, nn

__all__ = [
    "SLOT_ORDER",
    "N_SLOTS",
    "PLANES",
    "RECOMMENDED_BACKBONES",
    "VIDEO_BACKBONES",
    "BackboneInfo",
    "backbone_data_config",
    "create_backbone",
    "SliceEncoder",
]

try:  # The manifest module owns SLOT_ORDER; this package must not require it to import.
    from kappa.data.manifest import N_SLOTS, SLOT_ORDER  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - exercised only when kappa.data is unavailable
    #: Mirrors the pinned cross-agent contract. A slot's index in this tuple is its
    #: ``slot_ids`` value in the batch, so the order is load-bearing, not cosmetic.
    SLOT_ORDER = (
        ("Axial", 0),
        ("Axial", 1),
        ("Coronal", 0),
        ("Coronal", 1),
        ("Sagittal", 0),
        ("Sagittal", 1),
    )
    N_SLOTS = len(SLOT_ORDER)

#: Acquisition planes, in the order their embedding table is indexed.
PLANES: tuple[str, ...] = ("Axial", "Coronal", "Sagittal")

#: Backbones this project supports, with the role each one plays. Every name was
#: checked against ``timm.list_pretrained`` this session, which is why the
#: EfficientNetV2-S entry is ``tf_efficientnetv2_s``: ``timm.list_pretrained
#: ("efficientnetv2_s")`` returns an empty list, so that spelling would hand back
#: random weights while reporting success.
RECOMMENDED_BACKBONES: Mapping[str, str] = {
    "convnext_small": "primary; RSNA 2024's winner measured large < base < small",
    "tf_efficientnetv2_s": "diversity member; BatchNorm backbone, different error modes",
    "caformer_s18": "diversity member; hybrid conv/attention",
    "maxvit_tiny_tf_384": "diversity member; native 384 px, and mean/std of 0.5 not ImageNet",
}

#: The 3D backbones, which are ``torchvision.models.video`` names and NOT timm names.
#: Naming one of these in ``model.backbone`` is the entire switch from
#: :class:`SliceEncoder` onto :class:`kappa.models.volume_encoder.VolumeEncoder`.
#:
#: **This is the second copy, and the duplication is deliberate.** ``kappa.config``
#: holds the first, because a config must validate on a machine with no torch -- the
#: same reason it writes ``SLOT_NAMES`` out rather than importing ``SLOT_ORDER`` from
#: the manifest. The copy lives HERE rather than being imported from ``kappa.config``
#: because ``tests/test_tta.py`` pins that importing the scoring path must not pull in
#: the training-side config module, and ``StudyClassifier`` needs this tuple at import
#: time. ``tests/test_models.py`` cross-checks the two, which is what stops them
#: drifting.
VIDEO_BACKBONES: tuple[str, ...] = (
    "r3d_18",
    "mc3_18",
    "r2plus1d_18",
)

#: Name suffixes that mark a scaled-up variant of a supported family. Warned about
#: rather than refused, because the decision belongs to whoever reads the warning.
_SCALED_UP_SUFFIXES = ("_base", "_large", "_huge", "_giant", "_xl", "_l", "_m", "_b36", "_m36")

#: Standard deviation of the FiLM conditioning at initialisation. Non-zero so the
#: conditioning is live from step 0 and therefore testable -- a zero init makes a
#: fresh model provably ignore the slot, which reads identically to the wiring being
#: broken. Small enough that the pretrained activations move by a couple of percent.
_FILM_INIT_STD = 0.02


@dataclass(frozen=True)
class BackboneInfo:
    """What the encoder actually loaded, as opposed to what was asked for.

    A run that does not state this is not evidence about the backbone it claims. The
    fields that catch real mistakes are ``weights_loaded`` (asking for pretrained and
    silently getting random initialisation), ``drop_path_applied`` (asking for
    stochastic depth and silently getting none, because not every timm family accepts
    the argument) and ``img_size_applied`` (asking for a non-native input geometry and
    getting the native one, because the same is true of ``img_size``).

    ``img_size_applied`` is *stored*, not recovered from ``input_size``, and reading it
    back from there is the obvious-and-wrong implementation: ``input_size`` comes from
    timm's ``pretrained_cfg``, which describes the checkpoint rather than the object
    that was built. Measured this session, a ``vit_small_patch14_dinov2`` created with
    ``img_size=392`` still resolves to ``input_size=(3, 518, 518)``. A summary that
    trusted that field would print 518 for a model whose patch grid is 28 x 28 and
    whose forward pass asserts on anything but 392.
    """

    name: str
    pretrained_requested: bool
    weights_loaded: bool
    pretrained_tag: str | None
    hf_hub_id: str | None
    num_features: int
    input_size: tuple[int, int, int]
    mean: tuple[float, ...]
    std: tuple[float, ...]
    drop_path_applied: float | None
    n_batchnorm: int
    #: What was asked for, and what the family took. Two fields for the same reason
    #: ``pretrained_requested`` sits beside ``weights_loaded``: one nullable field
    #: cannot tell "never asked" apart from "asked and refused", and the second of
    #: those is a config claiming a geometry the model is not using. Defaulted so that
    #: the pair reads truthfully -- as "no size was requested" -- for every caller
    #: written before this argument existed.
    img_size_requested: int | None = None
    img_size_applied: int | None = None

    def render(self) -> str:
        """Multi-line rendering for the top of a run log."""
        weights = "pretrained" if self.weights_loaded else "RANDOM INIT (pretrained=False)"
        # Naming a checkpoint that was not loaded is how a random-init run gets read
        # as a pretrained one six weeks later, so the source is withheld unless real.
        if self.weights_loaded:
            source = self.hf_hub_id or self.pretrained_tag or "unknown"
        else:
            source = "n/a"
        drop_path = "not supported by this family" if self.drop_path_applied is None else (
            f"{self.drop_path_applied:g}"
        )
        return "\n".join(
            [
                f"backbone   : {self.name}  weights={weights}  source={source}",
                f"features   : dim={self.num_features}  batchnorm_layers={self.n_batchnorm}",
                f"preprocess : input_size={self.input_size}  mean={self.mean}  std={self.std}"
                "   <- from timm, never hardcoded",
                f"img_size   : {self._render_img_size()}",
                f"drop_path  : {drop_path}",
            ]
        )

    def _render_img_size(self) -> str:
        """The one line that can contradict ``input_size`` on the line above it.

        It has to be able to, because ``input_size`` is the checkpoint's geometry and
        this is the built model's. The two disagree in exactly the case this argument
        exists for -- a ViT built at 392 from 518-px weights -- and printing only the
        first would file a 392-px run as a 518-px one.
        """
        native = self.input_size[-1]
        if self.img_size_requested is None:
            return f"not requested; built at the family's native {native}"
        if self.img_size_applied is None:
            # Shouted, because this is the state where the config and the model
            # disagree and no other line in the summary would say so.
            return (
                f"{self.img_size_requested} REQUESTED BUT DROPPED -- this family does not "
                "accept img_size, so the model was built at its native geometry"
            )
        if self.img_size_applied != native:
            return (
                f"{self.img_size_applied}  <- built at this size; position embeddings "
                f"interpolated from the native {native}, which is the size input_size "
                "above still reports"
            )
        return f"{self.img_size_applied} (the family's native size)"


def backbone_data_config(model: nn.Module) -> dict[str, Any]:
    """Resolve the backbone's own preprocessing config.

    The single supported way to learn what this backbone expects. Feed the result to
    ``kappa.data.dataset.Normalisation.from_data_config`` so the pixels the encoder
    sees are normalised the way its weights were trained, whatever the family.
    """
    return dict(timm.data.resolve_data_config(model=model))


def create_backbone(
    name: str = "convnext_small",
    *,
    pretrained: bool = True,
    drop_path_rate: float = 0.2,
    img_size: int | None = None,
) -> tuple[nn.Module, BackboneInfo]:
    """Build a pooled-feature timm backbone and report exactly what came back.

    Refuses ``pretrained=True`` for a name that has no published weights. timm only
    warns in that case and returns a randomly initialised model, which is how an
    eleven-hour run ends up being evidence about nothing.

    ``img_size`` builds the model *at the size the data config will feed it*, which
    only matters for the families where geometry is baked into the parameters. A
    ``convnext_small`` is fully convolutional and takes any size, so leaving this
    ``None`` is correct for it and for the measured 0.7280 baseline; a ViT is not, and
    ``vit_base_patch14_dinov2`` at its native 518 cannot consume the 392 px this
    project feeds. Interpolating the position embeddings once here beats resizing the
    pixels to 518 (2.7x the tokens for no new information) and beats interpolating per
    forward pass.

    ``None`` is not "the default size": it is "do not pass the argument at all", and
    the distinction is load-bearing. Every existing arm was built by a call with no
    ``img_size`` in it, so that call is reproduced byte-for-byte rather than routed
    through a new code path with the native size filled in -- a resolved-then-passed
    size would change construction for those arms and put the 0.7280 reference point
    beyond comparison.
    """
    if name.endswith(_SCALED_UP_SUFFIXES):
        warnings.warn(
            f"backbone {name!r} looks like a scaled-up variant. Scaling the encoder is a "
            "measured null on this dataset: DINOv2-S to DINOv2-B was worth +0.0011 against "
            "a 0.0020 noise floor, improving 5 of 12 labels. Scale the input geometry "
            f"instead. Supported: {sorted(RECOMMENDED_BACKBONES)}.",
            stacklevel=2,
        )
    if pretrained and not timm.list_pretrained(name):
        hint = ""
        if timm.list_pretrained(f"tf_{name}"):
            hint = f" Did you mean {'tf_' + name!r}?"
        raise ValueError(
            f"backbone {name!r} has no pretrained weights in timm, so pretrained=True would "
            f"return a randomly initialised model without failing.{hint} "
            f"Supported: {sorted(RECOMMENDED_BACKBONES)}."
        )

    kwargs: dict[str, Any] = {"pretrained": pretrained, "num_classes": 0, "global_pool": "avg"}

    # The optional kwargs, i.e. the ones some timm families refuse. Each is tried and,
    # if refused, dropped and recorded -- building without them is fine, silently
    # building without them is not, because the config still claims 0.2 and 392.
    optional: dict[str, Any] = {"drop_path_rate": drop_path_rate}
    if img_size is not None:
        optional["img_size"] = int(img_size)

    # Retried with one kwarg removed at a time, not with the whole set removed on the
    # first TypeError. The single-fallback version looks equivalent and is not: a
    # family that refused only ``drop_path_rate`` would lose the ``img_size`` it does
    # honour, and the run would then be a ViT holding a 518-px position grid while the
    # loader hands it 392 px -- a crash at step 0 if the family asserts, and a silently
    # wrong grid if it interpolates. Two optional kwargs make this enumeration
    # complete; a third would need the power set, not another entry here.
    attempts: list[dict[str, Any]] = [optional]
    if len(optional) > 1:
        attempts += [{k: v for k, v in optional.items() if k != drop} for drop in optional]

    # HEAD-NORM PLACEMENT, and why this is a fallback rather than a default.
    # ``global_pool="avg"`` makes timm build ``fc_norm`` and leave ``norm`` as
    # Identity -- ``use_fc_norm = global_pool == "avg"`` whenever ``fc_norm`` is
    # unset. DINOv2's published weights were trained with TOKEN pooling and ship
    # ``norm.*``, so the module tree and the checkpoint disagree and
    # ``load_state_dict`` raises inside ``create_model``: missing ``fc_norm.*``,
    # unexpected ``norm.*``. Under timm 1.0.29 this failed every DINOv2 run within
    # ~42 s of starting.
    #
    # ``fc_norm=False`` restores the checkpoint's ``norm`` AND keeps mean pooling, so
    # the backbone axis stays comparable with convnext and maxvit, which also pool
    # "avg". ``global_pool="token"`` also loads and is deliberately NOT used: it
    # would swap mean pooling for the CLS token on one arm only, and the backbone
    # column would then measure pooling and architecture at the same time.
    #
    # Tried only AFTER the default fails, because most timm ViT checkpoints match the
    # default and forcing ``fc_norm=False`` on those would break the ones that
    # genuinely ship ``fc_norm.*`` -- trading this failure for its mirror image.
    # RuntimeError only: a TypeError means the family has no such argument at all,
    # which is what the enumeration above already handles.
    def _build(attempt: dict[str, Any]) -> nn.Module:
        try:
            return timm.create_model(name, **attempt, **kwargs)
        except RuntimeError as exc:
            if "fc_norm" not in str(exc):
                raise
            rebuilt = timm.create_model(name, **attempt, **kwargs, fc_norm=False)
            warnings.warn(
                f"backbone {name!r} ships a token-pooling checkpoint whose head norm is "
                f"'norm', which does not match the 'fc_norm' that global_pool='avg' "
                f"builds. Rebuilt with fc_norm=False, so the pretrained norm IS loaded "
                f"and pooling stays 'avg', comparable with the other backbones. "
                f"Recorded rather than silent: the alternative fix, global_pool='token', "
                f"would change pooling on this arm alone.",
                stacklevel=2,
            )
            return rebuilt

    model: nn.Module | None = None
    used: dict[str, Any] = {}
    for attempt in attempts:
        try:
            model = _build(attempt)
        except TypeError:
            continue
        used = attempt
        break
    if model is None:
        # Unguarded on purpose. If the bare call raises TypeError too then nothing here
        # was ever about these kwargs, and catching it would report a phantom
        # "unsupported argument" for a defect that lives inside the family.
        # Still routed through _build: a family that refused BOTH optional kwargs
        # would otherwise skip the head-norm fallback and fail on the state dict.
        model = _build({})

    applied: float | None = drop_path_rate if "drop_path_rate" in used else None
    config = backbone_data_config(model)
    pretrained_cfg = dict(getattr(model, "pretrained_cfg", {}) or {})
    info = BackboneInfo(
        name=name,
        pretrained_requested=pretrained,
        weights_loaded=bool(pretrained),
        pretrained_tag=pretrained_cfg.get("tag") or None,
        hf_hub_id=pretrained_cfg.get("hf_hub_id") or None,
        num_features=int(model.num_features),
        input_size=tuple(int(v) for v in config["input_size"]),  # type: ignore[arg-type]
        mean=tuple(float(v) for v in config["mean"]),
        std=tuple(float(v) for v in config["std"]),
        drop_path_applied=applied,
        n_batchnorm=sum(
            1 for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)
        ),
        img_size_requested=None if img_size is None else int(img_size),
        img_size_applied=used.get("img_size"),
    )
    return model, info


def _slot_tables() -> tuple[Tensor, Tensor]:
    """Plane index and contrast bit for each slot, derived from ``SLOT_ORDER``."""
    planes = torch.tensor([PLANES.index(plane) for plane, _ in SLOT_ORDER], dtype=torch.long)
    contrast = torch.tensor([int(fs) for _, fs in SLOT_ORDER], dtype=torch.long)
    return planes, contrast


class SliceEncoder(nn.Module):
    """Shared 2D backbone over 2.5D three-slice stacks, conditioned on the slot.

    The three input channels are *adjacent slices*, not colour. Reusing an RGB stem
    for that is the standard 2.5D trick and it is why the backbone's own
    normalisation must be applied identically to all three channels -- see
    ``Normalisation`` in ``kappa.data.dataset``, which broadcasts a single-element
    mean across the stack for exactly this reason.

    Conditioning is FiLM on the pooled embedding, factorised into a plane term and a
    contrast term rather than a single six-way lookup. Factorising means a gradient
    from a ``Sagittal_fs1`` sample also updates the ``Sagittal`` and ``fs1`` terms,
    which is the same sharing argument that justifies one encoder over six -- and it
    matters most for ``Axial_fs0`` at 19.4% coverage, whose ``Axial`` and ``fs0``
    terms are learned largely from the other slots.

    ``img_size`` is forwarded to :func:`create_backbone` and must equal the crop the
    dataset emits, because for a ViT the two are one number in two files: the patch
    grid is fixed when the model is built, and ``kappa.data.dataset`` decides what it
    is fed. Leave it ``None`` for a convolutional backbone, which has no such coupling.
    """

    def __init__(
        self,
        backbone: str = "convnext_small",
        *,
        pretrained: bool = True,
        drop_path_rate: float = 0.2,
        n_slots: int = N_SLOTS,
        img_size: int | None = None,
    ) -> None:
        super().__init__()
        if not 1 <= n_slots <= len(SLOT_ORDER):
            # A job may select fewer slots, but slot_ids stay global, so the
            # conditioning tables can never be shorter than the ids they are indexed
            # by. Silently truncating them would index the wrong plane instead.
            raise ValueError(f"n_slots={n_slots} outside 1..{len(SLOT_ORDER)}")
        self.backbone, self.info = create_backbone(
            backbone,
            pretrained=pretrained,
            drop_path_rate=drop_path_rate,
            img_size=img_size,
        )
        self.n_slots = int(n_slots)
        self.embed_dim = self.info.num_features

        planes, contrast = _slot_tables()
        if self.n_slots != len(SLOT_ORDER):
            planes, contrast = planes[: self.n_slots], contrast[: self.n_slots]
        # Buffers rather than plain tensors so ``.to(device)`` moves them with the model.
        self.register_buffer("slot_plane", planes, persistent=False)
        self.register_buffer("slot_contrast", contrast, persistent=False)

        self.plane_film = nn.Embedding(len(PLANES), 2 * self.embed_dim)
        self.contrast_film = nn.Embedding(2, 2 * self.embed_dim)
        nn.init.normal_(self.plane_film.weight, std=_FILM_INIT_STD)
        nn.init.normal_(self.contrast_film.weight, std=_FILM_INIT_STD)

    @property
    def data_config(self) -> dict[str, Any]:
        """The backbone's own preprocessing config. The only sanctioned source."""
        return backbone_data_config(self.backbone)

    def describe(self) -> str:
        """What was actually loaded, for the run log."""
        return self.info.render()

    def set_grad_checkpointing(self, enable: bool = True) -> None:
        """Trade compute for activation memory in the backbone.

        Needed because the batch is ``(B, 6, n_slices, 3, H, W)``: one step pushes up
        to ``B * 6 * n_slices`` images through the backbone, so activations rather
        than weights are what exhausts the GPU -- and training runs on one GPU, so
        there is no second card to spill onto.
        """
        setter = getattr(self.backbone, "set_grad_checkpointing", None)
        if setter is None:  # pragma: no cover - every timm family in use supports it
            warnings.warn(
                f"backbone {self.info.name!r} does not support gradient checkpointing; "
                "activation memory is unchanged.",
                stacklevel=2,
            )
            return
        setter(enable=enable)

    def forward(self, pixels: Tensor, slot_ids: Tensor, present: Tensor | None = None) -> Tensor:
        """Embed every slice stack of every present slot.

        Args:
            pixels: ``(B, S, N, 3, H, W)`` float. The 3 is the 2.5D neighbour stack
                and is pinned by the batch contract.
            slot_ids: ``(B, S)`` long, global slot index in ``0..N_SLOTS-1``.
            present: ``(B, S)`` bool. ``False`` means the slot is absent; those rows
                are never passed to the backbone and come back exactly zero.

        Returns:
            ``(B, S, N, embed_dim)``, exactly zero on absent slots.
        """
        if pixels.ndim != 6:
            raise ValueError(
                f"pixels has shape {tuple(pixels.shape)}; the pinned batch is "
                "(B, N_SLOTS, n_slices, 3, H, W)"
            )
        batch, slots, n_slices, channels, height, width = pixels.shape
        if channels != 3:
            raise ValueError(
                f"pixels has {channels} channels; the 2.5D neighbour stack is pinned at 3"
            )
        if tuple(slot_ids.shape) != (batch, slots):
            raise ValueError(
                f"slot_ids has shape {tuple(slot_ids.shape)}, expected {(batch, slots)}"
            )
        if present is None:
            present = pixels.new_ones((batch, slots), dtype=torch.bool)
        elif tuple(present.shape) != (batch, slots):
            raise ValueError(
                f"present has shape {tuple(present.shape)}, expected {(batch, slots)}"
            )
        present = present.bool()

        flat_present = present.unsqueeze(-1).expand(batch, slots, n_slices).reshape(-1)
        flat_pixels = pixels.reshape(-1, channels, height, width)
        flat_slots = slot_ids.unsqueeze(-1).expand(batch, slots, n_slices).reshape(-1)

        out = flat_pixels.new_zeros((flat_pixels.shape[0], self.embed_dim))
        keep = flat_present.nonzero(as_tuple=True)[0]
        if keep.numel():
            # index_select, not a boolean mask, so the gather is a real gather: the
            # absent rows are not merely multiplied by zero downstream, they are
            # never handed to the backbone at all. That is what makes the presence
            # guarantee hold for a BatchNorm family in training mode, and what makes
            # a non-finite fill in an absent slot harmless.
            features = self.backbone(flat_pixels.index_select(0, keep))
            features = self._condition(features, flat_slots.index_select(0, keep))
            out = out.index_copy(0, keep, features.to(out.dtype))
        return out.view(batch, slots, n_slices, self.embed_dim)

    def _condition(self, features: Tensor, slot_ids: Tensor) -> Tensor:
        """Apply factorised plane x contrast FiLM to pooled features."""
        if int(slot_ids.max()) >= self.n_slots or int(slot_ids.min()) < 0:
            raise ValueError(
                f"slot_ids outside [0, {self.n_slots - 1}]: saw "
                f"[{int(slot_ids.min())}, {int(slot_ids.max())}]"
            )
        plane = self.plane_film(self.slot_plane[slot_ids])
        contrast = self.contrast_film(self.slot_contrast[slot_ids])
        gamma, beta = (plane + contrast).chunk(2, dim=-1)
        return features * (1.0 + gamma) + beta
