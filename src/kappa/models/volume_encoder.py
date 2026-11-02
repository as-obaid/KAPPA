"""One shared 3D encoder for all six plane x contrast slots, conditioned on the slot.

Why this exists: :class:`kappa.models.encoder.SliceEncoder` runs a 2D backbone over
each sampled slice independently and lets :class:`kappa.models.aggregator.
SeriesAggregator` -- a transformer over the slice axis -- do all the cross-slice
reasoning. That is the 2.5D design, and ``docs/model.md`` argues for it from
the geometry: with median 30 slices, **~11.7x anisotropy** and 90.5% of series
carrying an inter-slice gap over 0.1 mm, isotropic 3D resampling would be mostly
interpolated data, so "2.5D stacks are the honest baseline and **3D is an ablation**".
This module is that ablation. It exists to make the claim testable, not to replace the
baseline.

**The depth axis is ``data.n_slices``.** There is deliberately no second depth knob:
``n_slices`` already sizes the batch's slice axis, is already in ``config_hash``, is
already logged as a dotted field, and already sizes the positional table. A separate
``volume.depth`` could disagree with it, and a config whose two halves disagree is the
failure this package spends most of its docstrings preventing. What ``volume.mode``
adds is only *which* indices those are -- see
:func:`kappa.data.dataset.contiguous_slice_indices`.

**The batch contract does not move.** ``pixels`` is ``(B, S, N, 3, H, W)`` here exactly
as it is for the 2.5D path, and this encoder returns ``(B, S, 1, embed_dim)`` -- one
token per slot rather than one per slice -- so ``SeriesAggregator``, ``StudyAggregator``
and the head are reached unchanged. Nothing in the loader, the collate, the checkpoint
writer or the inference path had to learn that a volumetric arm exists.

Three things it is built to get right, each a silent failure otherwise.

**1. The redundant channels are dropped, not averaged.** The 3 is the 2.5D neighbour
stack: channel 1 is the sampled slice, channels 0 and 2 are its geometric neighbours.
Under ``contiguous`` sampling those neighbours are *already in the block* -- channel 0
of slice ``i`` is channel 1 of slice ``i-1`` -- so averaging them would blur each plane
with its neighbours and hand the 3D kernels a pre-smoothed volume, destroying exactly
the through-plane detail the arm is testing for. This module takes **channel 1 alone**
and expands it to the three the Kinetics stem expects. The dataset still materialises
all three; at depth 32 that is 56.6 MB per study against the 18.9 MB used, which is
real but small beside a 384 px volume, and it is the price of not touching the batch
contract.

**2. Absent-slot pixels never reach the backbone.** The argument is
:class:`SliceEncoder`'s and it is **stronger** here, not weaker. That class measured a
0.364 logit shift from a garbage-filled absent slot through a BatchNorm backbone in
training mode, and noted ``convnext_small`` contains 0 BatchNorm layers.
``torchvision``'s ``VideoResNet`` carries a ``BatchNorm3d`` after **every** convolution
-- so the whole volumetric arm is the case that gather was built for. Present slots are
gathered, the backbone runs on those rows only, and exact zeros are scattered back.

**3. The weights come from a second store, and it is offline.** ``torchvision`` fetches
from ``download.pytorch.org`` into ``$TORCH_HOME/hub/checkpoints``, which is **not** the
HF cache the timm backbones use and is **not** covered by ``HF_HUB_OFFLINE``. Compute
nodes have no outbound internet, so an unstaged weight is a hang inside urllib, not an
error. :func:`create_video_backbone` therefore refuses to reach the network at all --
it checks the file is on disk first and raises naming both the file and the staging
script. Stage with ``slurm/stage_torch_weights.sh`` on the login node.

    from kappa.models.volume_encoder import VolumeEncoder

    enc = VolumeEncoder(backbone="r3d_18", pretrained=True)
    print(enc.describe())                       # states what was actually loaded
    emb = enc(pixels, slot_ids, present)        # (B, N_SLOTS, 1, embed_dim)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from kappa.models.encoder import _FILM_INIT_STD, N_SLOTS, PLANES, VIDEO_BACKBONES

__all__ = [
    "VIDEO_BACKBONES",
    "KINETICS_MEAN",
    "KINETICS_STD",
    "VideoBackboneInfo",
    "video_backbone_data_config",
    "create_video_backbone",
    "VolumeEncoder",
]

#: Kinetics-400 preprocessing, from
#: ``torchvision.models.video.R3D_18_Weights.KINETICS400_V1.transforms``. Reduced to
#: ONE number per statistic rather than carried as a triple, and that is deliberate:
#: the volume is greyscale MRI expanded to three identical channels, so three
#: different per-channel means would normalise three copies of one image differently
#: and then feed them to a stem that expects them to be an RGB frame.
#: ``kappa.data.dataset.Normalisation`` broadcasts a one-element mean across all three
#: channels, which keeps the expansion exact. The values are the channel means of the
#: published triples ``(0.43216, 0.394666, 0.37645)`` and
#: ``(0.22803, 0.22145, 0.216989)``.
KINETICS_MEAN = 0.401092
KINETICS_STD = 0.22215633333333332

#: Environment variable ``torchvision`` reads for its download cache. Named here so
#: the error message can quote the resolved path rather than a guess.
_TORCH_HOME_ENV = "TORCH_HOME"


@dataclass(frozen=True)
class VideoBackboneInfo:
    """What was actually built, so the run log can state it rather than imply it."""

    name: str
    pretrained: bool
    weights_file: str | None
    embed_dim: int
    n_parameters: int
    #: Count of ``nn.BatchNorm3d`` modules in the backbone. Printed unconditionally,
    #: same reason ``kappa.models.encoder.BackboneInfo`` prints ``batchnorm_layers``
    #: for the 2.5D path: it is what makes "20 BatchNorm3d layers" a measured fact in
    #: every run log rather than a number quoted from a docstring.
    n_batchnorm: int = 0
    #: Whether ``freeze_backbone_bn`` was requested. Printed here, not inferred, so a
    #: reader cannot mistake an unfrozen run for a frozen one from the backbone line
    #: alone -- the exact failure this flag exists to make visible rather than repeat.
    freeze_backbone_bn: bool = False

    def describe(self) -> str:
        weights = "random init" if not self.pretrained else f"Kinetics-400 <- {self.weights_file}"
        bn = f"batchnorm_layers={self.n_batchnorm}"
        if self.freeze_backbone_bn:
            bn += " (FROZEN: eval-mode running stats held throughout training)"
        return (
            f"encoder    : {self.name} (3D, torchvision.models.video) {weights}\n"
            f"             embed {self.embed_dim}, {self.n_parameters:,} backbone params, {bn}"
        )


def video_backbone_data_config(name: str) -> dict[str, Any]:
    """Preprocessing statistics for a video backbone, in timm's ``data_config`` shape.

    The same key names timm uses (``mean``, ``std``, ``input_size``), so that
    :meth:`kappa.data.dataset.Normalisation.from_data_config` consumes it without a
    branch and ``StudyClassifier.data_config`` means one thing on both paths.
    """
    if name.split(".", 1)[0] not in VIDEO_BACKBONES:
        raise ValueError(f"unknown video backbone {name!r}; known: {list(VIDEO_BACKBONES)}")
    return {"mean": (KINETICS_MEAN,), "std": (KINETICS_STD,), "input_size": (3, 112, 112)}


def _weights_enum(name: str) -> Any:
    """The Kinetics-400 weights enum for one family. Imports torchvision lazily."""
    from torchvision.models import video as tvv

    enums = {
        "r3d_18": tvv.R3D_18_Weights.KINETICS400_V1,
        "mc3_18": tvv.MC3_18_Weights.KINETICS400_V1,
        "r2plus1d_18": tvv.R2Plus1D_18_Weights.KINETICS400_V1,
    }
    try:
        return enums[name]
    except KeyError:  # pragma: no cover - guarded by the caller's membership test
        raise ValueError(f"no Kinetics weights known for {name!r}") from None


def staged_weight_path(name: str) -> Path:
    """Where the staging script must have put this backbone's weights.

    Shared by :func:`create_video_backbone` and ``slurm/stage_torch_weights.sh``'s
    verification step, so the two cannot disagree about the filename.
    """
    weights = _weights_enum(name.split(".", 1)[0])
    torch_home = Path(
        os.environ.get(_TORCH_HOME_ENV) or (Path.home() / ".cache" / "torch")
    ).expanduser()
    return torch_home / "hub" / "checkpoints" / Path(weights.url).name


def create_video_backbone(name: str, *, pretrained: bool = True) -> tuple[nn.Module, int]:
    """Build a ``torchvision.models.video`` backbone with its classifier removed.

    Returns ``(module, embed_dim)``. The module maps ``(B, 3, D, H, W)`` to
    ``(B, embed_dim)``: ``VideoResNet`` ends in an ``AdaptiveAvgPool3d(1)`` before its
    ``fc``, so replacing ``fc`` with ``nn.Identity`` leaves exactly the pooled feature
    and no reshape is needed here.

    **Never downloads.** ``pretrained=True`` asserts the weight file is already in
    ``$TORCH_HOME`` and raises naming it otherwise. On a compute node with no route off
    the box, letting ``torchvision`` try would produce a job that hangs inside urllib
    with no output until the walltime kills it -- indistinguishable from a slow first
    epoch, and the exact failure ``HF_HUB_OFFLINE=1`` was added to convert into an
    error for the timm half of the model zoo.
    """
    family = name.split(".", 1)[0]
    if family not in VIDEO_BACKBONES:
        raise ValueError(f"unknown video backbone {name!r}; known: {list(VIDEO_BACKBONES)}")

    from torchvision.models import video as tvv

    weights = None
    weights_file: str | None = None
    if pretrained:
        weights = _weights_enum(family)
        cached = staged_weight_path(family)
        if not cached.is_file():
            raise FileNotFoundError(
                f"{family} pretrained weights are not staged: {cached} does not exist.\n"
                f"  {_TORCH_HOME_ENV}={os.environ.get(_TORCH_HOME_ENV, '<unset>')}\n"
                f"  source={weights.url}\n"
                "Compute nodes have no outbound internet, so torchvision would hang in "
                "urllib rather than fail. Stage it on the LOGIN node first:\n"
                "  slurm/stage_torch_weights.sh"
            )
        weights_file = cached.name

    model = getattr(tvv, family)(weights=weights)
    embed_dim = int(model.fc.in_features)
    model.fc = nn.Identity()
    return model, embed_dim


class VolumeEncoder(nn.Module):
    """A 3D backbone over the slice axis, conditioned on the slot.

    Signature and output contract deliberately mirror
    :class:`kappa.models.encoder.SliceEncoder`, so :class:`kappa.models.StudyClassifier`
    selects between them and changes nothing else. The one difference is the third axis
    of the output: ``1`` here against ``n_slices`` there, because the depth axis has
    been consumed *inside* the encoder rather than left for ``SeriesAggregator`` to
    pool. That is the whole experimental claim of this module, stated as a tensor shape.

    Args:
        backbone: one of ``kappa.config.VIDEO_BACKBONES``.
        pretrained: load the staged Kinetics-400 weights. ``False`` is stated in
            :meth:`describe` so a random-init run cannot be mistaken for a pretrained
            one -- which matters more here than on the 2D path, because a random-init
            3D arm compared against a pretrained 2.5D arm measures initialisation.
        n_slots: size of the slot-conditioning tables.
        img_size: accepted and **ignored**, so ``StudyClassifier`` can pass it
            unconditionally. A convolutional 3D backbone has no position embedding to
            build for a fixed grid, which is the only thing that argument is for.
            Recorded in :meth:`describe` rather than dropped silently.
        freeze_backbone_bn: hold every ``BatchNorm3d`` in the backbone in eval mode
            for the whole of training, so a training step and a validation step read
            the identical running statistics rather than the current minibatch's own
            mean/var. Off by default -- see ``kappa.config.ModelConfig`` for why this
            exists (``r3d_18`` at micro-batch 4) and :meth:`train` for how it survives
            the per-epoch ``model.train()`` reset.
    """

    def __init__(
        self,
        backbone: str = "r3d_18",
        *,
        pretrained: bool = True,
        n_slots: int = N_SLOTS,
        img_size: int | None = None,
        freeze_backbone_bn: bool = False,
    ) -> None:
        super().__init__()
        self.backbone_name = str(backbone)
        self.n_slots = int(n_slots)
        self._img_size = img_size
        self._pretrained = bool(pretrained)
        self.backbone, self.embed_dim = create_video_backbone(backbone, pretrained=pretrained)
        self._grad_checkpointing = False
        self._freeze_backbone_bn = bool(freeze_backbone_bn)

        self.info = VideoBackboneInfo(
            name=self.backbone_name,
            pretrained=self._pretrained,
            weights_file=staged_weight_path(self.backbone_name).name if pretrained else None,
            embed_dim=self.embed_dim,
            n_parameters=sum(p.numel() for p in self.backbone.parameters()),
            n_batchnorm=sum(1 for m in self.backbone.modules() if isinstance(m, nn.BatchNorm3d)),
            freeze_backbone_bn=self._freeze_backbone_bn,
        )

        # Factorised plane x contrast FiLM, identical in form and initialisation to
        # SliceEncoder's, and identical on purpose: the conditioning is not the
        # variable under test, so building it the same way twice keeps the
        # 2.5D-vs-3D comparison about the encoder. Non-zero init so the conditioning
        # is live from step 0 and therefore testable -- a zero init makes a fresh
        # model provably ignore the slot, which reads identically to broken wiring.
        self.plane_film = nn.Parameter(torch.randn(len(PLANES), 2, self.embed_dim) * _FILM_INIT_STD)
        self.contrast_film = nn.Parameter(torch.randn(2, 2, self.embed_dim) * _FILM_INIT_STD)

        # nn.Module.__init__ leaves every submodule at training=True, so a freshly
        # built encoder is not yet frozen even when asked for -- apply it once here so
        # the state is correct before anyone calls train() or forward() for the first
        # time. This is NOT sufficient on its own: see train() below for why it has to
        # be reapplied every epoch too.
        if self._freeze_backbone_bn:
            self._freeze_backbone_batchnorm()

    @property
    def data_config(self) -> dict[str, Any]:
        """Preprocessing the loaded weights expect. The only supported source."""
        return video_backbone_data_config(self.backbone_name)

    def describe(self) -> str:
        """Everything a reader needs to know what this encoder is evidence about."""
        lines = [self.info.describe()]
        if self._img_size is not None:
            lines.append(
                f"             img_size={self._img_size} accepted and ignored "
                "(a 3D conv stem has no position embedding to size)"
            )
        if not self._pretrained:
            lines.append(
                "             *** RANDOM INIT: this arm cannot be read against a "
                "pretrained 2.5D arm ***"
            )
        return "\n".join(lines)

    def set_grad_checkpointing(self, enable: bool = True) -> None:
        """Trade backbone compute for activation memory.

        ``torchvision`` exposes no ``set_grad_checkpointing``, unlike timm, so this
        checkpoints the four residual stages by hand. Off by default. It is the escape
        hatch if a depth turns out not to fit: peak activation is dominated by
        ``layer1``, which runs at ``(64, D, H/2, W/2)`` -- the stem stride is
        ``(1, 2, 2)`` and, unlike a 2D ResNet, ``VideoResNet`` has **no max-pool after
        the stem**, so the largest tensor in the network is 64 channels at half
        resolution and full depth.
        """
        self._grad_checkpointing = bool(enable)

    def _freeze_backbone_batchnorm(self) -> None:
        """Force every ``BatchNorm3d`` in the backbone into eval mode.

        Eval mode is what makes a ``BatchNorm`` layer normalise with its stored
        ``running_mean``/``running_var`` instead of the current batch's own
        statistics, and stop updating them -- exactly what ``model.freeze_backbone_bn``
        promises: the same statistics at train time and at val time. Scoped to
        ``BatchNorm3d`` alone, never the whole backbone, so a family that ever grew a
        ``Dropout`` would still see its normal train-mode behaviour.
        """
        for module in self.backbone.modules():
            if isinstance(module, nn.BatchNorm3d):
                module.eval()

    def train(self, mode: bool = True) -> "VolumeEncoder":
        """``nn.Module.train()``, plus re-freezing the backbone's BatchNorm3d layers.

        THIS OVERRIDE IS LOAD-BEARING, NOT COSMETIC. ``kappa.training.loop`` calls
        ``model.train()`` once at the top of every epoch, and ``nn.Module.train()``
        recurses into every submodule -- including this one -- resetting
        ``training = True`` unconditionally along the way. A ``freeze_backbone_bn``
        implementation that only flips the backbone to eval mode at construction time,
        or once before the training loop starts, is silently undone by that very first
        epoch boundary: the config would still claim ``freeze_backbone_bn: true``, the
        run log would still print it, and the run would nonetheless train with
        ordinary batch statistics throughout -- a wasted GPU run whose config and
        behaviour disagree, and disagree invisibly. Overriding ``train()`` here
        instead of patching the epoch loop means the fix travels with the module: no
        matter how many wrapper classes call ``.train()`` on their children (
        ``StudyClassifier.encoder``, or ``TeacherClassifier.image.encoder``), Python
        dispatches to THIS method for a ``VolumeEncoder``, so the reapplication always
        happens. Do not remove this override in favour of a one-shot flip elsewhere.
        """
        super().train(mode)
        if mode and self._freeze_backbone_bn:
            self._freeze_backbone_batchnorm()
        return self

    def _run_backbone(self, volume: Tensor) -> Tensor:
        if not self._grad_checkpointing or not self.training:
            return self.backbone(volume)
        hidden = self.backbone.stem(volume)
        for stage in (
            self.backbone.layer1,
            self.backbone.layer2,
            self.backbone.layer3,
            self.backbone.layer4,
        ):
            hidden = checkpoint(stage, hidden, use_reentrant=False)
        return torch.flatten(self.backbone.avgpool(hidden), 1)

    def forward(self, pixels: Tensor, slot_ids: Tensor, present: Tensor | None = None) -> Tensor:
        """Embed the volume of every present slot.

        Args:
            pixels: ``(B, S, N, 3, H, W)`` float, the pinned batch. ``N`` is the
                volumetric depth ``D`` on this path, and the 3 is the 2.5D neighbour
                stack of which only the centre channel is read -- see the module
                docstring for why it is dropped rather than averaged.
            slot_ids: ``(B, S)`` long, global slot index in ``0..N_SLOTS-1``.
            present: ``(B, S)`` bool. ``False`` means the slot is absent; those rows
                are never passed to the backbone and come back exactly zero.

        Returns:
            ``(B, S, 1, embed_dim)``, exactly zero on absent slots. The ``1`` is not a
            vestigial axis: it is what keeps this drop-in with ``SliceEncoder``, so
            ``SeriesAggregator`` and the presence-masked ``StudyAggregator`` are
            reached without a branch.
        """
        if pixels.ndim != 6:
            raise ValueError(
                f"pixels has shape {tuple(pixels.shape)}; the pinned batch is "
                "(B, N_SLOTS, n_slices, 3, H, W)"
            )
        batch, slots, depth, channels, height, width = pixels.shape
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

        # (B, S, D, 3, H, W) -> centre channel -> (B*S, 1, D, H, W). This reshape is
        # what makes the tensor a volume rather than a batch of planes: D becomes the
        # dimension the 3D kernels convolve over, instead of being folded into the
        # batch axis the way SliceEncoder folds it.
        volumes = pixels[:, :, :, 1].reshape(batch * slots, 1, depth, height, width)

        flat_present = present.reshape(-1)
        flat_slots = slot_ids.reshape(-1)
        out = volumes.new_zeros((batch * slots, self.embed_dim))
        keep = flat_present.nonzero(as_tuple=True)[0]
        if keep.numel():
            # index_select, not a boolean mask, for SliceEncoder's reason -- and here
            # it is load-bearing rather than defensive: VideoResNet carries a
            # BatchNorm3d after every convolution, so in training mode an absent
            # slot's zeros would otherwise enter the batch statistics of every present
            # slot in the same step. Never computed beats multiplied by zero.
            selected = volumes.index_select(0, keep)
            # Three identical channels, because the Kinetics stem is RGB. `expand` is
            # a view, so this costs no memory, and the normalisation is ONE broadcast
            # number per statistic, so all three channels carry identical values by
            # construction rather than by coincidence.
            selected = selected.expand(-1, 3, -1, -1, -1)
            features = self._run_backbone(selected)
            features = self._condition(features, flat_slots.index_select(0, keep))
            out = out.index_copy(0, keep, features.to(out.dtype))
        return out.view(batch, slots, 1, self.embed_dim)

    def _condition(self, features: Tensor, slot_ids: Tensor) -> Tensor:
        """Apply factorised plane x contrast FiLM to pooled features."""
        if int(slot_ids.max()) >= self.n_slots or int(slot_ids.min()) < 0:
            raise ValueError(
                f"slot id outside [0, {self.n_slots - 1}]: got "
                f"[{int(slot_ids.min())}, {int(slot_ids.max())}]"
            )
        plane = torch.div(slot_ids, 2, rounding_mode="floor")
        contrast = slot_ids % 2
        gamma = self.plane_film[plane, 0] + self.contrast_film[contrast, 0]
        beta = self.plane_film[plane, 1] + self.contrast_film[contrast, 1]
        return features * (1.0 + gamma) + beta
