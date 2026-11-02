"""Input adapters, visual encoders, pooling, fusion, heads, report supervision.

One shared 2D encoder over 2.5D slice stacks, conditioned on the ``plane x contrast``
slot rather than replicated six times, then two attention pools: slices to a series
vector, series to a study vector. The study pool takes the presence mask, which is the
load-bearing part -- see :mod:`kappa.models.aggregator`.

    from kappa.models import StudyClassifier

    model = StudyClassifier(backbone="convnext_small", n_slices=16)
"""

from __future__ import annotations

from kappa.models.aggregator import GatedAttentionPool, SeriesAggregator, StudyAggregator
from kappa.models.encoder import (
    N_SLOTS,
    RECOMMENDED_BACKBONES,
    SLOT_ORDER,
    BackboneInfo,
    SliceEncoder,
    backbone_data_config,
    create_backbone,
)
from kappa.models.study_classifier import ModelOutput, StudyClassifier

# NOT imported eagerly: kappa.models.volume_encoder needs torchvision, and the 2.5D
# arms, the inference path and every test that builds them must not acquire that
# dependency because a 3D ablation exists. StudyClassifier imports it lazily, inside
# the branch that needs it. Import it by module path when you want it directly:
#     from kappa.models.volume_encoder import VolumeEncoder

__all__ = [
    "SLOT_ORDER",
    "N_SLOTS",
    "RECOMMENDED_BACKBONES",
    "BackboneInfo",
    "backbone_data_config",
    "create_backbone",
    "SliceEncoder",
    "GatedAttentionPool",
    "SeriesAggregator",
    "StudyAggregator",
    "ModelOutput",
    "StudyClassifier",
]
