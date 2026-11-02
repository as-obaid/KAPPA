"""Slices to a series vector, series to a study vector -- with an honest presence mask.

Why this exists: a knee study is a *set of sets*. Each series is an ordered sequence
of slices where position along the slice normal is meaningful, and each study is an
unordered set of up to six ``plane x contrast`` slots where **which slots exist is
itself information**. Collapsing either level with a mean loses the thing that
matters, and collapsing the study level without a mask loses more than that.

**The study-level pool takes the presence mask and the mask is load-bearing.** Only
``Axial_fs1`` is present in every study; ``Sagittal_fs0`` is 96.8%, ``Coronal_fs0``
77.3%, and ``Axial_fs0`` **19.4%**. There are **12 distinct availability patterns**
and only 566 of 4,407 studies carry all six slots. An absent slot has to be
zero-filled to keep the batch rectangular, and a model given a zero-filled slot with
no flag cannot distinguish "this plane was never acquired" from "this plane is black"
-- so it learns to read a black volume as anatomy, and at inference the two cases
score the same. :class:`StudyAggregator` therefore excludes absent slots from the
attention softmax *and* replaces their features with zeros before any nonlinearity
touches them.

That second step is not belt-and-braces. The cheap version relies on the softmax
weight for an absent slot being exactly ``0.0`` and on ``0.0 * x == 0.0``, which is
false when ``x`` is not finite. Measured this session on a six-slot batch with slots
absent: multiplying by a zero weight leaves a ``nan``-filled absent slot visible in
the *forward* pass, whereas substituting with :func:`torch.where` leaves the study
vector, the parameter gradients and even the gradient at the poisoned input all
finite for ``nan`` and ``inf`` alike. The substitution therefore happens at the top of
the forward, before the slot embedding, the LayerNorm and the attention scores.
Ordering is the point: a LayerNorm applied to a poisoned row cannot be un-poisoned
afterwards.

The stronger guarantee lives one level up. :class:`kappa.models.encoder.SliceEncoder`
never hands an absent slot to the backbone at all, so in the assembled model nothing
poisoned reaches this module. Measured against a mask-later variant of that encoder,
garbage in an absent slot shifted the logits by up to **0.364** through a BatchNorm
backbone in training mode while the gathering version stayed bit-identical. This
module's substitution is the second line of defence, and the reason the pool can be
unit-tested with ``nan`` on its own. The test that proves either fills an absent slot
with garbage and asserts the study vector is bit-identical; if that test does not
pass, the mask is decorative.

Nothing in this module normalises across the batch dimension. Every norm here is a
``LayerNorm``, which is per-token, because a BatchNorm anywhere in the pooling stack
would couple studies inside a batch and make the presence guarantee false no matter
how carefully the mask is applied.

    from kappa.models.aggregator import SeriesAggregator, StudyAggregator

    series = SeriesAggregator(dim=768, n_slices=16)(slice_embeddings)   # (B*S, D)
    study, slot_weights = StudyAggregator(dim=768)(series, present, slot_ids)

One shared pool over the slots is the default and it is an assumption, not a neutral
choice: all twelve labels then read a single set of plane weights, and knee MRI's best
plane is not one plane. ``StudyAggregator(per_label=N_LABELS)`` gives each label its own
softmax over the present slots -- ``n_labels x n_slots`` extra numbers, 72 of them --
returned as a third output by :meth:`StudyAggregator.forward_with_per_label` and added
to the shared vector rather than replacing it. ``per_label=0`` is the default and is
structurally identical to this module before the option existed: same submodules, same
parameter names, same arithmetic, so every checkpoint already written still loads
strictly. See :class:`kappa.config.PoolConfig` for what the two shapes are worth.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = [
    "GatedAttentionPool",
    "LabelQueryHead",
    "SeriesAggregator",
    "StudyAggregator",
]


def _head_count(dim: int, preferred: int = 4) -> int:
    """Largest head count not exceeding ``preferred`` that divides ``dim``.

    Embedding width is whatever the backbone reports -- 768 for ``convnext_small``,
    1280 for ``tf_efficientnetv2_s``, 512 for ``caformer_s18``, all measured this
    session. A fixed head count would make a legal backbone raise deep inside
    ``nn.MultiheadAttention`` instead of here.
    """
    for heads in (preferred, 8, 4, 2, 1):
        if heads <= preferred and dim % heads == 0:
            return heads
    return 1


class GatedAttentionPool(nn.Module):
    """Gated attention pooling over a sequence, with an optional exclusion mask.

    Gated rather than plain attention because a single ``tanh`` branch cannot
    suppress an element: it can only weight it. The ``sigmoid`` gate multiplies the
    score down toward zero, which is what a series carrying no relevant anatomy
    needs. Returns the weights alongside the pooled vector so a run can report which
    slot or slice it actually leaned on.
    """

    def __init__(self, dim: int, hidden: int | None = None) -> None:
        super().__init__()
        hidden = hidden or max(32, dim // 4)
        self.value = nn.Linear(dim, hidden)
        self.gate = nn.Linear(dim, hidden)
        self.score = nn.Linear(hidden, 1)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Pool ``(B, T, D)`` to ``(B, D)``, returning ``(pooled, weights (B, T))``.

        ``mask`` is ``True`` where the element participates. Excluded elements are
        substituted with zeros before any linear layer sees them, so a non-finite
        value in an excluded position cannot reach the output or the gradient.
        """
        if x.ndim != 3:
            raise ValueError(f"expected (B, T, D), got {tuple(x.shape)}")
        batch, length, _ = x.shape
        if mask is None:
            mask = x.new_ones((batch, length), dtype=torch.bool)
        elif tuple(mask.shape) != (batch, length):
            raise ValueError(f"mask has shape {tuple(mask.shape)}, expected {(batch, length)}")
        mask = mask.bool()

        keep = mask.unsqueeze(-1)
        x = torch.where(keep, x, torch.zeros_like(x))

        logits = self.score(torch.tanh(self.value(x)) * torch.sigmoid(self.gate(x))).squeeze(-1)
        # finfo.min rather than -inf: under autocast this tensor may be fp16, where
        # -inf survives the softmax but poisons any later arithmetic on the logits.
        excluded = torch.full_like(logits, torch.finfo(logits.dtype).min)
        logits = torch.where(mask, logits, excluded)
        # A row with nothing present would softmax a row of identical minima. That is
        # uniform, not NaN, but it would still average in the zeros; zero the weights
        # so the pooled vector is exactly zero and the caller sees an honest nothing.
        empty = ~mask.any(dim=-1, keepdim=True)
        logits = torch.where(empty, torch.zeros_like(logits), logits)
        weights = logits.softmax(dim=-1)
        weights = torch.where(empty, torch.zeros_like(weights), weights)
        return (weights.unsqueeze(-1) * x).sum(dim=1), weights


class SeriesAggregator(nn.Module):
    """Ordered slice embeddings to one series vector.

    A transformer over the slice axis, then gated attention pooling. Order is
    supplied by a learned positional table because slice position carries anatomy:
    the corpus is **11.7x anisotropic** with a median 3.00 mm slice thickness, so
    adjacent slices are genuinely different structures rather than a smooth volume,
    and a permutation-invariant pool would discard the one axis along which a tear
    propagates.

    Attention is confined to the slice axis. This module is applied per series, so
    there is no path by which one slot's slices can influence another's -- which is
    what lets :class:`StudyAggregator` be the single place the presence mask is
    enforced.
    """

    def __init__(
        self,
        dim: int,
        n_slices: int = 16,
        n_layers: int = 2,
        n_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim <= 0 or n_slices <= 0:
            raise ValueError(f"dim={dim} and n_slices={n_slices} must both be positive")
        self.dim = int(dim)
        self.n_slices = int(n_slices)
        # Dropout defaults off: the config surface exposes only drop_path_rate, so
        # regularisation lives in the backbone's stochastic depth. A non-zero default
        # here would be a second, undeclared regulariser. The feed-forward width is
        # 2x rather than the conventional 4x -- a judgement, not a measurement: this
        # attends over 16 slice tokens on 4,407 studies, and 4x would put a third of
        # the model's parameters in the pool rather than the encoder.
        layer = nn.TransformerEncoderLayer(
            d_model=self.dim,
            nhead=_head_count(self.dim, n_heads),
            dim_feedforward=int(self.dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.position = nn.Parameter(torch.zeros(1, self.n_slices, self.dim))
        nn.init.trunc_normal_(self.position, std=0.02)
        self.norm = nn.LayerNorm(self.dim)
        self.pool = GatedAttentionPool(self.dim)

    def positional(self, length: int) -> Tensor:
        """Positional embedding for a sequence of ``length`` slices.

        Interpolated when ``length`` differs from the configured ``n_slices``. Slices
        are sampled to span the volume, so index ``i`` of ``length`` sits at a
        *fraction* of the way through the knee; absolute table entries would name
        different anatomy at train and inference time if the count changed. The
        equal-length path returns the table untouched, so the common case is exact.
        """
        if length == self.n_slices:
            return self.position
        table = self.position.transpose(1, 2)
        return F.interpolate(table, size=length, mode="linear", align_corners=False).transpose(1, 2)

    def forward(self, slices: Tensor) -> Tensor:
        """Reduce ``(B, T, D)`` slice embeddings to ``(B, D)`` series vectors."""
        if slices.ndim != 3:
            raise ValueError(f"expected (B, n_slices, dim), got {tuple(slices.shape)}")
        if slices.shape[-1] != self.dim:
            raise ValueError(f"slice embedding is {slices.shape[-1]}-dim, expected {self.dim}")
        hidden = self.encoder(slices + self.positional(slices.shape[1]))
        pooled, _ = self.pool(self.norm(hidden))
        return pooled


class StudyAggregator(nn.Module):
    """Series vectors plus a slot embedding to one study vector, honouring presence.

    The slot embedding is added rather than concatenated so the pool sees one width
    regardless of how many slots a job selects -- the baseline job is ``Axial_fs1``
    only, and the six-slot job must share this module. ``present`` is required, not
    optional: a default of "all present" is precisely the bug this module exists to
    prevent, and 3,841 of 4,407 studies would hit it.
    """

    def __init__(
        self, dim: int, n_slots: int = 6, dropout: float = 0.0, per_label: int = 0
    ) -> None:
        super().__init__()
        if dim <= 0 or n_slots <= 0:
            raise ValueError(f"dim={dim} and n_slots={n_slots} must both be positive")
        if per_label < 0:
            raise ValueError(f"per_label={per_label} must be non-negative (0 disables it)")
        self.dim = int(dim)
        self.n_slots = int(n_slots)
        self.per_label = int(per_label)
        self.slot_embedding = nn.Embedding(self.n_slots, self.dim)
        nn.init.normal_(self.slot_embedding.weight, std=0.02)
        self.norm = nn.LayerNorm(self.dim)
        self.dropout = nn.Dropout(dropout)
        self.pool = GatedAttentionPool(self.dim)
        self.out_norm = nn.LayerNorm(self.dim)
        # Built ONLY when asked for, and that is the whole compatibility story: at
        # `per_label=0` this module has the same submodules and the same parameter
        # names it has always had, so the twenty checkpoints already on the cluster
        # still load with `strict=True`. A parameter registered unconditionally and
        # merely left unused would add two unexpected keys to every one of them.
        if self.per_label:
            # `n_labels x n_slots` scores -- 72 numbers at twelve labels and six
            # slots, against ~50 M in the encoder. Deliberately a table and not a
            # projection of the features: the question this arm asks is "which PLANE
            # does this label read", which is a property of the slot, not of the
            # study, and a content-dependent scorer would answer a different one.
            self.label_slot_logits = nn.Parameter(torch.zeros(self.per_label, self.n_slots))
            nn.init.trunc_normal_(self.label_slot_logits, std=0.02)
            # LayerNorm, not BatchNorm, for the reason in the module docstring, and
            # separate from `out_norm` so the per-label residual and the shared vector
            # reach the head on comparable scales without either rescaling the other.
            self.label_norm = nn.LayerNorm(self.dim)

    def forward(
        self, series: Tensor, present: Tensor, slot_ids: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Fuse ``(B, S, D)`` series vectors into ``(B, D)``, returning the weights.

        **The return contract is two elements and stays two elements**, at every
        setting of ``per_label``. The per-label tensor is an additional output of
        :meth:`forward_with_per_label`, which this method delegates to and whose third
        element it drops. A third element that is ``None`` when the arm is off was the
        alternative and was rejected: every existing caller and test unpacks a pair,
        and a return arity that depends on a constructor argument is the kind of thing
        that is discovered by a `ValueError` in a queued job. The arm is opt-in, so it
        is the arm that carries the wider signature.
        """
        pooled, weights, _ = self.forward_with_per_label(series, present, slot_ids)
        return pooled, weights

    def forward_with_per_label(
        self, series: Tensor, present: Tensor, slot_ids: Tensor
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """As :meth:`forward`, plus the ``(B, n_labels, D)`` per-label study vectors.

        The third element is ``None`` when ``per_label=0``, and on that path every
        line below is the arithmetic this module has always performed -- not an
        equivalent rearrangement of it.

        Absent slots are substituted with zeros *first*, before the slot embedding,
        the LayerNorm or the attention scores. Doing it in that order is what makes
        an absent slot unable to change the result even when it holds ``nan``, and
        what keeps the backward pass finite. The per-label softmax inherits the same
        two-step guard as the shared one, and neither step is redundant: the mask
        keeps the result independent of *how many* slots are absent, and the zeroing
        is what survives a value that is not finite.

        Called directly rather than through ``__call__`` by
        :class:`kappa.models.study_classifier.StudyClassifier`, so a forward hook
        registered on this module sees the pair and not the triple. Nothing in this
        repository registers one; it is written down here because the day something
        does, this is the line that explains why it fired once.
        """
        if series.ndim != 3:
            raise ValueError(f"expected (B, n_slots, dim), got {tuple(series.shape)}")
        batch, slots, dim = series.shape
        if dim != self.dim:
            raise ValueError(f"series vector is {dim}-dim, expected {self.dim}")
        if tuple(present.shape) != (batch, slots):
            raise ValueError(
                f"present has shape {tuple(present.shape)}, expected {(batch, slots)}"
            )
        if tuple(slot_ids.shape) != (batch, slots):
            raise ValueError(
                f"slot_ids has shape {tuple(slot_ids.shape)}, expected {(batch, slots)}"
            )
        present = present.bool()

        keep = present.unsqueeze(-1)
        series = torch.where(keep, series, torch.zeros_like(series))
        hidden = self.norm(series + self.slot_embedding(slot_ids))
        hidden = torch.where(keep, hidden, torch.zeros_like(hidden))
        # One dropout call, shared by both readouts. Two calls would draw two masks
        # and make the per-label vectors a view of a different study than the shared
        # one -- and would change this line's RNG consumption at `per_label=0`.
        hidden = self.dropout(hidden)
        pooled, weights = self.pool(hidden, present)
        per_label = self._per_label_pool(hidden, present, slot_ids)
        return self.out_norm(pooled), weights, per_label

    def _per_label_weights(self, present: Tensor, slot_ids: Tensor) -> Tensor:
        """One softmax over the PRESENT slots per label: ``(B, n_labels, n_slots)``.

        Keyed on ``slot_ids`` rather than on column position. A column's index in the
        batch is whatever ``data.slots`` happened to select; the thing a label has a
        preference about is the slot -- ``Sagittal_fs1`` for a meniscus, ``Coronal_*``
        for a collateral ligament -- so a table read by column would mean a different
        plane in a one-slot run than in a six-slot run and could not be compared
        across them.

        Depends on ``present`` and ``slot_ids`` and on **no feature at all**, which is
        what makes it separable from the pooling below and directly assertable: these
        weights cannot be moved by what an absent slot contains, because they never
        see it.
        """
        batch = present.shape[0]
        # (n_slots, n_labels) gathered by slot identity -> (B, S, L) -> (B, L, S).
        logits = self.label_slot_logits.t()[slot_ids].permute(0, 2, 1)
        mask = present.unsqueeze(1)
        # finfo.min rather than -inf, for the reason GatedAttentionPool gives: under
        # autocast a -inf survives the softmax and poisons everything downstream.
        excluded = torch.full_like(logits, torch.finfo(logits.dtype).min)
        logits = torch.where(mask, logits, excluded)
        # A study with nothing present would softmax a row of identical minima --
        # uniform, not NaN, but it would still average in the zero-filled slots.
        # Zero the weights so the study vector is an honest nothing instead.
        empty = (~present.any(dim=-1)).view(batch, 1, 1)
        logits = torch.where(empty, torch.zeros_like(logits), logits)
        weights = logits.softmax(dim=-1)
        return torch.where(empty, torch.zeros_like(weights), weights)

    def _per_label_pool(
        self, hidden: Tensor, present: Tensor, slot_ids: Tensor
    ) -> Tensor | None:
        """Each label's own weighted sum of the present slots, ``(B, n_labels, D)``."""
        if not self.per_label:
            return None
        weights = self._per_label_weights(present.bool(), slot_ids)
        # `hidden` is already exactly zero in every absent column, so an absent slot
        # contributes 0 here whatever it held -- the same second line of defence the
        # shared pool has, and the reason a `nan` cannot reach the output or the
        # gradient through this path either.
        pooled = torch.einsum("bls,bsd->bld", weights.to(hidden.dtype), hidden)
        return self.label_norm(pooled)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, n_slots={self.n_slots}, per_label={self.per_label}"


class LabelQueryHead(nn.Module):
    """One learned query per label, cross-attending the study's slice tokens.

    Why this exists: the twelve "heads" of the default model are a single
    ``nn.Linear(dim, N_LABELS)`` over one pooled 768-d vector. Every label reads
    the *same* summary and differs only in a final weight row, so no label has any
    capacity to look somewhere else -- and the twelve findings live in different
    places. ``Fracture`` turns on cortex and ``Effusion`` on a diffuse joint
    signal; one pooled vector has already decided how much of each survived.

    Here each label owns a query that attends the ``n_slots x n_slices`` slice
    tokens directly, so a label can weight the slices it needs. Worth about 0.006
    on the published ablation -- real but small, and roughly twenty times less than
    where the encoder's weights come from. It is an ablation arm, not
    a default, and ``do not hand-couple the medial/lateral pairs``: they are already
    maximally coupled through the shared encoder and the problem is the opposite one.

    Absent slots are excluded the same way ``StudyAggregator`` excludes them --
    zeroed before any projection *and* removed from the attention softmax. Both
    steps are load-bearing and neither is redundant, which is worth stating because
    it looks redundant:

    * The softmax mask is what keeps the result independent of *how many* absent
      slots there are. Without it the zeroed tokens still take softmax mass, so the
      same study scored with four absent slots and with five would get different
      logits.
    * The zeroing is what survives a non-finite value. Measured on torch 2.14: a
      ``nan`` in a key excluded by ``key_padding_mask`` still reaches the output,
      because the masked weight is exactly 0 and ``0 * nan`` is ``nan``. The mask
      alone does not protect the forward pass.

    A study with nothing present yields the bias row, finitely. On this torch
    version an all-masked ``key_padding_mask`` already returns zeros rather than the
    NaN older versions produced, so the explicit guard here is version-defensive: it
    makes the behaviour this module's contract instead of the library's.
    """

    def __init__(
        self,
        dim: int,
        n_labels: int,
        num_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.n_labels = int(n_labels)
        self.num_heads = _head_count(dim, preferred=num_heads)
        self.queries = nn.Parameter(torch.zeros(self.n_labels, self.dim))
        self.attention = nn.MultiheadAttention(
            self.dim, self.num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(self.dim)
        # A per-label projection to a scalar, rather than a shared Linear: a shared
        # one would put the twelve back on a common readout and undo the point.
        self.weight = nn.Parameter(torch.zeros(self.n_labels, self.dim))
        self.bias = nn.Parameter(torch.zeros(self.n_labels))
        nn.init.trunc_normal_(self.queries, std=0.02)
        nn.init.trunc_normal_(self.weight, std=0.01)

    def forward(
        self, tokens: Tensor, token_mask: Tensor, residual: Tensor | None = None
    ) -> Tensor:
        """Score ``(B, T, D)`` tokens into ``(B, n_labels)`` logits.

        ``token_mask`` is ``True`` where the token participates, matching the
        convention of :class:`GatedAttentionPool` rather than PyTorch's inverted
        ``key_padding_mask`` -- one convention per codebase, inverted once, here.

        ``residual`` is study-level evidence added to what each label reads, either
        ``(B, 1, D)`` to give all twelve the same vector or ``(B, n_labels, D)`` to
        give each its own. It is added **before** :attr:`norm`, not after, for two
        reasons: the LayerNorm then owns the scale of the sum, so a residual whose
        magnitude happens to differ from the attended vector's cannot swamp it; and
        the attended vector is exactly zero for a study with nothing present, so a
        post-norm addition would be the only term left and would decide the logits of
        an empty study. ``None`` -- the default -- is the arithmetic this head has
        always performed, to the bit.
        """
        if tokens.ndim != 3:
            raise ValueError(f"expected (B, T, D), got {tuple(tokens.shape)}")
        batch, length, dim = tokens.shape
        if dim != self.dim:
            raise ValueError(f"tokens are {dim}-dim, expected {self.dim}")
        if tuple(token_mask.shape) != (batch, length):
            raise ValueError(
                f"token_mask has shape {tuple(token_mask.shape)}, expected {(batch, length)}"
            )
        token_mask = token_mask.bool()

        # Zero excluded tokens before the projections inside the attention, so a
        # non-finite value in an absent slot cannot reach the output or the gradient.
        tokens = torch.where(token_mask.unsqueeze(-1), tokens, torch.zeros_like(tokens))

        # A row with no tokens at all would make key_padding_mask all-True and
        # nn.MultiheadAttention return NaN. Attend uniformly instead, then zero the
        # result: the study genuinely has no evidence and must say so, not poison
        # the batch.
        empty = ~token_mask.any(dim=-1)
        keys_valid = token_mask | empty.unsqueeze(-1)

        queries = self.queries.unsqueeze(0).expand(batch, -1, -1)
        attended, _ = self.attention(
            queries, tokens, tokens, key_padding_mask=~keys_valid, need_weights=False
        )
        attended = torch.where(empty.view(batch, 1, 1), torch.zeros_like(attended), attended)
        if residual is not None:
            if residual.ndim != 3 or residual.shape[0] != batch or residual.shape[-1] != dim:
                raise ValueError(
                    f"residual has shape {tuple(residual.shape)}, expected "
                    f"{(batch, 1, dim)} or {(batch, self.n_labels, dim)}"
                )
            if residual.shape[1] not in (1, self.n_labels):
                raise ValueError(
                    f"residual has {residual.shape[1]} label rows, expected 1 or "
                    f"{self.n_labels}"
                )
            attended = attended + residual
        attended = self.norm(attended)
        return (attended * self.weight).sum(dim=-1) + self.bias

    def extra_repr(self) -> str:
        return f"dim={self.dim}, n_labels={self.n_labels}, num_heads={self.num_heads}"
