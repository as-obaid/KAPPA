"""The volumetric arm, and the four ways it could be wrong while still reporting a number.

Why this exists. The 3D sweep is an ADDITION -- V1, V2 and V3 must stay the experiments
they were -- and its own result is only readable if four things hold. Each fails
silently:

* **The depth axis is not what the config says.** ``data.n_slices`` IS the depth D on
  this path. A batch that arrives with a different D still trains, still prints a loss
  curve and still reports a macro, filed under a depth it never used.
* **The block is not contiguous, or not in order.** The whole hypothesis is "ordered
  neighbouring slices". A sampler that spread them, reversed them or repeated them
  would measure something else under the same name.
* **A defaulted section moved every hash.** ``canonical_dump`` hashes the full model
  dump, so adding ``volume`` would have re-hashed all nine grid cells and the measured
  baseline. Nothing raises; the old numbers just stop matching their own config files.
* **An absent slot reached the backbone.** ``VideoResNet`` carries a ``BatchNorm3d``
  after every convolution, so in training mode a zero-filled absent slot would enter
  the batch statistics of every present slot. ``SliceEncoder`` measured a 0.364 logit
  shift from exactly this on a BatchNorm backbone; the 3D arm is that case throughout.

The contiguity claims are made END TO END through the real loader, not against the
sampler function, using the synthetic phantom's per-slice index marker: with an
identity crop the value read out of the batch IS the cached slice index. Asserting
against ``contiguous_slice_indices`` would only prove the function agrees with itself.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from kappa.config import (  # noqa: E402
    VIDEO_BACKBONES,
    ConfigError,
    VolumeConfig,
    canonical_dump,
    config_hash,
    load_config,
    loads_config,
)
from kappa.data.dataset import (  # noqa: E402
    SLICE_SAMPLERS,
    CropSpec,
    Normalisation,
    StudyDataset,
    contiguous_slice_indices,
    sample_slice_indices,
)
from kappa.data.paths import repo_root  # noqa: E402
from kappa.training.loop import build_optimiser, build_scheduler  # noqa: E402
from kappa.data.synthetic import make_synthetic_dataset  # noqa: E402

# The same pinned geometry tests/test_dataset.py uses: 0.5 mm/px over a 320 px matrix
# letterboxed into a 32 px cache is 5.0 mm per cached pixel.
GEOMETRY = ((0.5, 320, 320),)
CACHE_MM_PER_PIXEL = 5.0

#: The eight cells of the volumetric sweep, as (name, depth, mode). Written out rather
#: than globbed: a glob would silently pass if a config file were deleted, and the
#: point of this table is that all eight exist and say what they are meant to say.
CELLS = (
    ("3d_ctg_d8", 8, "contiguous"),
    ("3d_ctg_d16", 16, "contiguous"),
    ("3d_ctg_d24", 24, "contiguous"),
    ("3d_ctg_d32", 32, "contiguous"),
    ("3d_spn_d8", 8, "spanning"),
    ("3d_spn_d16", 16, "spanning"),
    ("3d_spn_d24", 24, "spanning"),
    ("3d_spn_d32", 32, "spanning"),
)

#: The path of every volumetric cell, derived from `CELLS` so the two cannot drift.
VOLUMETRIC_PATHS = frozenset(f"configs/{name}.yaml" for name, _, _ in CELLS)

#: The only configs allowed to be MISSING from `ON_DISK_ARTIFACT_HASHES`: configs added
#: after commit `705c262`, the code that produced every recorded artifact, and never
#: trained under it. Such a config has no run directory and therefore no digest to
#: record; fabricating one would claim an artifact that does not exist. Named
#: explicitly, rather than expressed as "whatever `FROZEN_HASHES` has that
#: `ON_DISK_ARTIFACT_HASHES` doesn't", so that a FUTURE config silently missing its
#: on-disk entry is caught as a regression instead of being absorbed into an
#: ever-growing exemption list.
POSTDATE_THE_HASH_MOVE = frozenset(
    {
        "configs/3d3_ctg_d16.yaml",
        "configs/3d3_ctg_d16_bs8.yaml",
        "configs/3d3_ctg_d16_bnfreeze.yaml",
        # The 10-epoch programme, including its learning-rate level probe.
        "configs/b2_v2_convnext_h1e4.yaml",
        "configs/b2_v2_convnext_h1e3_b5e5.yaml",
        "configs/b2_v2_convnext_h2e4_b5e5.yaml",
        "configs/b2_v2_convnext_lr100.yaml",
        "configs/b2_v2_convnext_lr10.yaml",
        "configs/b2_v2_convnext_lr03.yaml",
        "configs/b2_v1_convnext_lr03.yaml",
        "configs/b2_v1_convnext_lr10.yaml",
        "configs/b2_v2_maxvit_lr03.yaml",
        "configs/b2_v2_maxvit_lr10.yaml",
        "configs/b2_v3_convnext_lr03.yaml",
        "configs/b2_v3_convnext_lr10.yaml",
        # The next eight shipped at `optim.backbone_lr_mult: 0.03` and were regenerated
        # at 1.0 once the multiplier probe found 0.03 the worst of the three values
        # tried. Repinning is normally the bug the digest table exists to catch; it was
        # safe here only because none of the eight had ever run, so no result was filed
        # under the old digests. Had one fold existed, the right move would have been a
        # new name, because `name:` is the output directory.
        "configs/b2_aug_rot.yaml",
        "configs/b2_aug_rotbc.yaml",
        "configs/b2_aug_cutmix.yaml",
        "configs/b2_aug_all.yaml",
        "configs/b3_v2_convnext_perlabel.yaml",
        "configs/b3_v2_convnext_query_h1.yaml",
        "configs/b3_v2_coatnet384.yaml",
        "configs/b3_compose.yaml",
        # Five arms on top of the rotation augmentation, each moving one further field.
        "configs/b4_aug_ema.yaml",
        "configs/b4_aug_slices24.yaml",
        "configs/b4_aug_query_h4.yaml",
        "configs/b4_aug_droppath03.yaml",
        "configs/b4_aug_epochs14.yaml",
        # The residual control for `b4_aug_query_h4`: `pool.head_residual` set to false
        # and nothing else moved.
        "configs/b4_aug_query_h4_nores.yaml",
        # Volumetric backbone and plane arms, and MaxViT on the 10-epoch recipe.
        "configs/w5_3d_mc3_d32_ctg.yaml",
        "configs/w5_3d_r2p1_d32_ctg.yaml",
        "configs/w5_3d_r3d_d32_cor.yaml",
        "configs/w5_3d_r3d_d32_sag.yaml",
        "configs/w5_maxvit_plain.yaml",
        "configs/w5_maxvit_rot.yaml",
        "configs/w5_maxvit_query_h1.yaml",
        "configs/w5_maxvit_rot_query.yaml",
        # The ConvNeXt V2 scale ladder. `b5_cnv2_huge` is pinned so the file cannot
        # drift, although one of its folds is estimated at about 12 h, beyond an 8 h
        # job time limit.
        "configs/b5_cnv2_tiny.yaml",
        "configs/b5_cnv2_base.yaml",
        "configs/b5_cnv2_large.yaml",
        "configs/b5_cnv2_huge.yaml",
        # The teacher/student arms. `w6_e2_control` is listed even though its digest is
        # `b4_aug_query_h4`'s and that config has run: the exemption is keyed by PATH,
        # and no artifact has ever been written to `data/runs/w6_e2_control/`.
        "configs/w6_e2_control.yaml",
        "configs/w6_e1_teacher.yaml",
        "configs/w6_e3_kd_all.yaml",
        "configs/w6_e4_kd_labelled.yaml",
        "configs/w6_e5_kd_ablate.yaml",
        # The two older teacher/student prototypes are deliberately NOT listed. Their
        # digests moved when `DistillConfig.teacher_report_ablate` was added, but this
        # set means "in `FROZEN_HASHES` but absent from `ON_DISK_ARTIFACT_HASHES`", and
        # `configs/teacher_axial_fs1.yaml` and `configs/distil_axial_fs1.yaml` both
        # existed at commit `705c262`, so both have entries in that second table.
    }
)

#: Every config on disk, with the hash it has now, read by loading each file.
#: `test_the_pinned_set_is_every_config_on_disk` globs the directory, so a config added
#: without an entry here fails instead of going silently unpinned.
#:
#: **These digests have moved twice, both times deliberately**: once when
#: `optim.backbone_lr_mult` was added to `OptimConfig` so the pretrained trunk can
#: train slower than the fresh head, and again when `model.freeze_backbone_bn` was
#: added to `ModelConfig` so the volumetric arm's `BatchNorm3d` layers can be held in
#: eval mode throughout training. Neither section is default-droppable, so a field on
#: either rehashes every config that exists. Only ONE earlier generation of these
#: digests is kept, in `ON_DISK_ARTIFACT_HASHES` below -- see its own comment for why.
#: This table is not a licence to move them again: adding a config SECTION must still
#: list it in `kappa.config._DEFAULTED_SECTIONS` with a no-op default.
#:
#: The `pool` section (`per_label_slots` and `head_residual`, both defaulting to False)
#: was added by exactly that route, and every digest below is byte-identical across it.
#: `TestThePoolSection` in `tests/test_config.py` holds the other end of it: a
#: NON-default `pool` section must still change the hash, or the arm and the baseline
#: would share one digest.
FROZEN_HASHES = {
    "configs/baseline_axial_fs1.yaml": "894a006e3049",
    "configs/baseline_axial_fs1_480.yaml": "df3dfc2dcc9e",
    "configs/ablate_rotate.yaml": "fc9c798ed899",
    "configs/ablate_cutmix.yaml": "dcd9455293fa",
    # Rehashed by `DistillConfig.teacher_report_ablate`; these are the only two files
    # that field moved. It defaults to False, so `canonical_dump` keeps dropping the
    # whole `distill` section for every image-only config. These two carry a
    # non-default `distill` block, so the section is hashed for them and a new field in
    # it necessarily moves them.
    #
    # Excluding the flag from the hash would be worse: the residue-ablation arm and the
    # full-KD arm are the same architecture, temperature and lambdas, differing only in
    # whether the frozen teacher's report branch is gated, so an unhashed flag would
    # give them one digest and nothing downstream could tell them apart. Neither file
    # has ever run, so no artifact carries the old digests.
    "configs/teacher_axial_fs1.yaml": "7bbb3e885f03",
    "configs/distil_axial_fs1.yaml": "2f263bf58801",
    "configs/v1_axial_fs1_convnext.yaml": "acb51fd6d2bb",
    "configs/v1_axial_fs1_maxvit.yaml": "23751faa0a85",
    "configs/v1_axial_fs1_dinov2.yaml": "acbc6760cb06",
    "configs/v2_fs_three_planes_convnext.yaml": "9e25fdda94da",
    "configs/v2_fs_three_planes_maxvit.yaml": "4bd084770012",
    "configs/v2_fs_three_planes_dinov2.yaml": "f7e3d14abeac",
    "configs/v3_all_slots_convnext.yaml": "c57458c9172e",
    "configs/v3_all_slots_maxvit.yaml": "05aa6887fe28",
    "configs/v3_all_slots_dinov2.yaml": "dae4d9b828fa",
    "configs/3d_ctg_d8.yaml": "a10ffde6718f",
    "configs/3d_ctg_d16.yaml": "6d1b127242ae",
    "configs/3d_ctg_d24.yaml": "7e30b75f80ff",
    "configs/3d_ctg_d32.yaml": "aa750507fcf9",
    "configs/3d_spn_d8.yaml": "fc2adf5cea75",
    "configs/3d_spn_d16.yaml": "35aaf86fa9ba",
    "configs/3d_spn_d24.yaml": "06efdcfe5b5a",
    "configs/3d_spn_d32.yaml": "1ad9092a9a95",
    # The 3-slot volumetric arm (fusion x 3D, neither in CELLS nor in
    # tests/test_configs_on_disk.py's GRID). These are `POSTDATE_THE_HASH_MOVE`, so they
    # have no entry in `ON_DISK_ARTIFACT_HASHES` below.
    "configs/3d3_ctg_d16.yaml": "bff272c0fc46",
    "configs/3d3_ctg_d16_bs8.yaml": "6f2665442d0e",
    "configs/3d3_ctg_d16_bnfreeze.yaml": "e4c6ac6d58e3",
    # The 10-epoch programme. Every file carries `optim.epochs: 10` where the grid
    # carries 8, and `optim.epochs` IS in `config_hash`, so a new budget is a new
    # experiment and needs a new file AND a new `name` -- `name` is the output
    # directory, so reusing one would overwrite the first run's weights and OOF
    # tables. They are new files rather than edits, so adding them moved no hash above.
    "configs/b2_v2_convnext_lr100.yaml": "1feb168fa2c5",
    "configs/b2_v2_convnext_lr10.yaml": "09f1a221b438",
    "configs/b2_v2_convnext_lr03.yaml": "18b61d21cc2a",
    "configs/b2_v1_convnext_lr03.yaml": "9634a3cb5220",
    "configs/b2_v1_convnext_lr10.yaml": "48d3a65c0417",
    "configs/b2_v2_maxvit_lr03.yaml": "a4114928395e",
    "configs/b2_v2_maxvit_lr10.yaml": "12f126b892af",
    "configs/b2_v3_convnext_lr03.yaml": "42c3c0d21d38",
    "configs/b2_v3_convnext_lr10.yaml": "7ba5905513dd",
    "configs/b2_aug_rot.yaml": "ec0be62ad96b",
    "configs/b2_aug_rotbc.yaml": "f9f1cd481687",
    "configs/b2_aug_cutmix.yaml": "5ca1753b3620",
    "configs/b2_aug_all.yaml": "5742dfa40fa4",
    "configs/b3_v2_convnext_perlabel.yaml": "001864a40bcd",
    "configs/b3_v2_convnext_query_h1.yaml": "0163827e4c31",
    "configs/b3_v2_coatnet384.yaml": "2a71e9d05a1e",
    "configs/b3_compose.yaml": "3f9657db9f82",
    # Learning-rate level. These three move `optim.lr`: the multiplier probe held the
    # head at 5e-5 and was still improving at its highest backbone rate, and
    # `backbone_lr_mult` is capped at 1.0, so the backbone cannot go higher without
    # raising `lr`.
    "configs/b2_v2_convnext_h1e4.yaml": "f8f43a33f3b7",
    "configs/b2_v2_convnext_h1e3_b5e5.yaml": "047682c3740f",
    "configs/b2_v2_convnext_h2e4_b5e5.yaml": "9eda169863e9",
    # Five arms, each carrying `augment.rotate_deg: 10.0` and changing exactly ONE
    # further field on top of it: `optim.ema_decay` 0.0 -> 0.999, `data.n_slices`
    # 16 -> 24, `model.head_num_heads` 1 -> 4, `model.drop_path_rate` 0.2 -> 0.3,
    # `optim.epochs` 10 -> 14. Four are copies of `configs/b2_aug_rot.yaml`;
    # `b4_aug_query_h4` is a copy of `configs/b3_v2_convnext_query_h1.yaml` with the
    # augmentation added. All five are new files rather than edits, so adding them
    # moved no digest above.
    #
    # `b4_aug_epochs14` is the only config in the 10-epoch programme that is not at
    # ten epochs, and that is the experiment, not a slip: the augmented arms selected
    # later epochs on average than the unaugmented ones on the same schedule.
    "configs/b4_aug_ema.yaml": "da10ba3c2ee5",
    "configs/b4_aug_slices24.yaml": "13b91a0eb336",
    "configs/b4_aug_query_h4.yaml": "4215a78dca77",
    "configs/b4_aug_droppath03.yaml": "89cb7f3742a9",
    "configs/b4_aug_epochs14.yaml": "63119dd3b239",
    # `b4_aug_query_h4_nores` is `b4_aug_query_h4` with `pool.head_residual`
    # true -> false; the YAML diff is that field and `name:`, and nothing else.
    #
    # The two digests differing is the point. `pool` is a `_DEFAULTED_SECTIONS`
    # section, so a NON-default `pool` must change the hash or the arm and its control
    # would be indistinguishable. Here the flag moves from a non-default true to the
    # DEFAULT false, and 4215a78dca77 -> 40680d40fcf7 confirms the mechanism works in
    # that direction too. Contrast `--select`, which is a CLI argument, never reaches
    # TrainConfig, and so CANNOT be told apart by hash.
    "configs/b4_aug_query_h4_nores.yaml": "40680d40fcf7",
    # Volumetric arms on the two sub-axes the 3D configs above never vary: the backbone
    # (mc3_18, r2plus1d_18) and the slot (Coronal_fs1, Sagittal_fs1). Each moves one
    # field from `3d_ctg_d32`. They are new files, so adding them moved no hash above.
    "configs/w5_3d_mc3_d32_ctg.yaml": "837e3b4e9eec",
    "configs/w5_3d_r2p1_d32_ctg.yaml": "b9baa7705308",
    "configs/w5_3d_r3d_d32_cor.yaml": "9539293579dc",
    "configs/w5_3d_r3d_d32_sag.yaml": "96c14b744f5e",
    # MaxViT on the 10-epoch recipe. The three grid MaxViT configs ran at 8 epochs with
    # no augmentation and no head variant; these move it onto the recipe the convnext
    # arms use. `w5_maxvit_plain` is the reference the other three are read against.
    "configs/w5_maxvit_plain.yaml": "2d5e0a266ff4",
    "configs/w5_maxvit_rot.yaml": "641bac004a04",
    "configs/w5_maxvit_query_h1.yaml": "895f4ec48dcb",
    "configs/w5_maxvit_rot_query.yaml": "d1fe3f1ede88",
    # The ConvNeXt V2 scale ladder. Each is a copy of `b4_aug_query_h4` with
    # `model.backbone` moved, except that the two largest also halve `optim.batch_size`
    # and double `optim.grad_accum` to hold the effective batch at 16, so the optimiser
    # step is not a second moving part. `b5_cnv2_tiny` is smaller than convnext_small,
    # so a win there reflects ConvNeXt V2's FCMAE pretraining rather than scale.
    #
    # drop_path scales with size on the two largest, following facebookresearch/
    # ConvNeXt's 384 fine-tuning recipe (Base 0.2, Large 0.3, XL 0.4). `large` and
    # `huge` therefore move two hashed fields from `b4_aug_query_h4`, while `tiny` and
    # `base` both stay at 0.2, so the contrast between those two is still clean.
    "configs/b5_cnv2_tiny.yaml": "98a4ea02cdaf",
    "configs/b5_cnv2_base.yaml": "e8a9639df805",
    "configs/b5_cnv2_large.yaml": "addbbb8fbe58",
    "configs/b5_cnv2_huge.yaml": "f222815a8877",
    # The teacher/student arms. `w6_e2_control` shares its digest with
    # `b4_aug_query_h4` on purpose, and that duplicate is the exemption named in
    # `DELIBERATE_HASH_TWINS` below. `name` is excluded from `config_hash`, so a
    # byte-identical recipe under a new name hashes the same, and here the identical
    # digest is the machine-checkable proof that the no-distillation control is that
    # arm and not a near-miss of it. The four treatments each differ from it by the
    # `distill` section and nothing else, so each has its own digest.
    "configs/w6_e2_control.yaml": "4215a78dca77",
    "configs/w6_e1_teacher.yaml": "ecff8f5fb065",
    "configs/w6_e3_kd_all.yaml": "c3d00b3d0d60",
    "configs/w6_e4_kd_labelled.yaml": "7531fdf24075",
    "configs/w6_e5_kd_ablate.yaml": "f7a798ec0a61",
}

#: The ONLY pair of config files allowed to share a `config_hash`, and the reason the
#: pair exists. Named explicitly rather than expressed as "however many collisions
#: there happen to be", so that an ACCIDENTAL twin -- two arms that were meant to
#: differ and do not -- still fails instead of being absorbed.
#:
#: A shared digest is normally one experiment wearing two names, and that is a real
#: hazard: the seed-spread reading of a shared hash ("two seeds, one config") would be
#: applied to two different models. Here it is the opposite. `w6_e2_control` is a
#: deliberate re-run of `b4_aug_query_h4`'s exact recipe under a new `name`, so that
#: the distillation arms have a control built by the same code they are. The new
#: `name` keeps the two runs' artifacts in separate directories -- `name` IS the output
#: path -- while the shared digest is what proves the recipe did not drift.
#:
#: If this exemption is ever the only thing keeping the suite green for some OTHER
#: pair, that pair has drifted into being one experiment and the fix is in the configs,
#: not here.
DELIBERATE_HASH_TWINS = frozenset(
    {
        frozenset({"b4_aug_query_h4.yaml", "w6_e2_control.yaml"}),
    }
)

#: The digests carried by every artifact a training run has already written
#: (checkpoints and OOF tables), as produced by `config.py` at commit `705c262`. An
#: analyst holding one of these digests (say `e47ec0e7e863`) needs to be able to find
#: the config file that produced it, and that is the ONLY reason this table exists.
#:
#: NOT EQUAL to `FROZEN_HASHES`'s keys: the `POSTDATE_THE_HASH_MOVE` configs were added
#: after `705c262` and have no artifact under that code, so none is invented here. The
#: two are not simply asserted as a subset either -- an emptied or gutted
#: `ON_DISK_ARTIFACT_HASHES` would still satisfy `subset of FROZEN_HASHES` and pass
#: silently. Instead `TestAddingTheSectionMovedNoExistingHash
#: .test_the_on_disk_artifact_digests_are_recorded_and_no_longer_produced` asserts the
#: COMPLEMENT is exactly `POSTDATE_THE_HASH_MOVE` -- every config missing an on-disk
#: digest is accounted for by name, and nothing else may go missing.
#:
#: A table is warranted here ONLY when a real run has produced artifacts under a
#: digest that later moved, not for every code-only change in between. `config.py` is
#: byte-identical from `705c262` through `25f5bef`, so one table covers every recorded
#: artifact despite two later commits (the `optim.backbone_lr_mult` split and
#: `model.freeze_backbone_bn`) each moving `FROZEN_HASHES` again; nothing was ever
#: trained under either intermediate generation.
#:
#: Reconstructed by extracting `src/` at commit 25f5bef and hashing every config file
#: against that tree. `fc8ee481925e` (v1_axial_fs1_convnext at 8 epochs) and
#: `e296f5db8835` (the 10-epoch baseline) match the digests recorded when those runs
#: were made, and `d83ef4ecd835`, `b1b65ac368f1`, `0e9862d988cc` and `7733c3daf2e4`
#: match the run-log headers of four completed 3D runs, as do all fifteen 2.5D
#: digests. `configs/3d_*_d24` and `_d32` are computed only.
ON_DISK_ARTIFACT_HASHES = {
    "configs/baseline_axial_fs1.yaml": "e296f5db8835",
    "configs/baseline_axial_fs1_480.yaml": "3e9cd27e9604",
    "configs/ablate_rotate.yaml": "e0791297f464",
    "configs/ablate_cutmix.yaml": "8af8ae3af0cd",
    "configs/teacher_axial_fs1.yaml": "7916b39fa8e7",
    "configs/distil_axial_fs1.yaml": "1f2e487a3c24",
    "configs/v1_axial_fs1_convnext.yaml": "fc8ee481925e",
    "configs/v1_axial_fs1_maxvit.yaml": "e47ec0e7e863",
    "configs/v1_axial_fs1_dinov2.yaml": "42ffcdd5a373",
    "configs/v2_fs_three_planes_convnext.yaml": "940b6314cebb",
    "configs/v2_fs_three_planes_maxvit.yaml": "05f0ed4c717a",
    "configs/v2_fs_three_planes_dinov2.yaml": "bb8e88361103",
    "configs/v3_all_slots_convnext.yaml": "3c9b3dcb1a69",
    "configs/v3_all_slots_maxvit.yaml": "9c4b2b050506",
    "configs/v3_all_slots_dinov2.yaml": "b31299c77e8b",
    "configs/3d_ctg_d8.yaml": "d83ef4ecd835",
    "configs/3d_ctg_d16.yaml": "b1b65ac368f1",
    "configs/3d_ctg_d24.yaml": "471385536c33",
    "configs/3d_ctg_d32.yaml": "62cfffcbc482",
    "configs/3d_spn_d8.yaml": "0e9862d988cc",
    "configs/3d_spn_d16.yaml": "7733c3daf2e4",
    "configs/3d_spn_d24.yaml": "2bc8aa490a75",
    "configs/3d_spn_d32.yaml": "9f58f1f402d3",
}


def _corpus(tmp_path, **over):
    options = {
        "n_studies": 3,
        "seed": 0,
        "cache_size": 32,
        "slice_counts": (30, 30),
        "patterns": ((1,),),
        "geometry": GEOMETRY,
        "mask_rate": 0.0,
        "phantom_mm": 100.0,
    }
    options.update(over)
    return make_synthetic_dataset(tmp_path, **options)


def _identity_crop(corpus):
    """A crop that resamples nothing, so the cached slice markers survive to the batch."""
    return CropSpec(fov_mm=CACHE_MM_PER_PIXEL * corpus.cache_size, out_px=corpus.cache_size)


def _dataset(corpus, **over):
    options = {
        "normalisation": Normalisation.identity(),
        "labels": corpus.labels,
        "slots": (1,),
        "n_slices": 8,
        "crop": _identity_crop(corpus),
    }
    options.update(over)
    return StudyDataset(corpus.manifest, corpus.cache_dir, **options)


def _markers(volume, channel=1):
    """The phantom slice indices carried by one slot volume ``(n_slices, 3, H, W)``."""
    return (volume[:, channel, 0, 0] * 255.0).round().to(torch.int64).tolist()


def _load(path):
    return load_config(repo_root() / path)


# ---------------------------------------------------------------------------
# The sampler
# ---------------------------------------------------------------------------
class TestTheContiguousSamplerIsContiguousAndOrdered:
    @pytest.mark.parametrize("depth", [8, 16, 24, 32])
    @pytest.mark.parametrize("available", [40, 64, 192])
    def test_the_block_is_exactly_depth_long_and_strictly_adjacent(self, available, depth):
        idx = contiguous_slice_indices(available, depth)
        assert len(idx) == depth, "D must be the configured depth, always"
        assert np.array_equal(np.diff(idx), np.ones(depth - 1, dtype=np.int64))

    @pytest.mark.parametrize("depth", [8, 16, 24])
    def test_a_long_series_yields_no_duplicates_at_all(self, depth):
        # "no slice duplication occurs accidentally". With more slices available than
        # requested there is no reason for a repeat, so a repeat is a bug.
        idx = contiguous_slice_indices(64, depth)
        assert len(np.unique(idx)) == depth

    def test_the_block_is_centred_so_the_choice_does_not_drift(self):
        # Deterministic centring, stated: a tie-break that varied between runs would
        # make two runs of one config sample different anatomy while reporting the
        # same config hash. Floor division sends the odd slice to the high-index end.
        assert contiguous_slice_indices(32, 8).tolist() == [12, 13, 14, 15, 16, 17, 18, 19]
        assert contiguous_slice_indices(33, 8).tolist() == [12, 13, 14, 15, 16, 17, 18, 19]

    def test_it_never_indexes_outside_the_array(self):
        for available in range(1, 40):
            for depth in (8, 16, 32):
                idx = contiguous_slice_indices(available, depth)
                assert idx.min() >= 0 and idx.max() < available

    @pytest.mark.parametrize("bad", [(0, 8), (10, 0), (-1, 8)])
    def test_a_degenerate_request_raises_rather_than_returning_something(self, bad):
        with pytest.raises(ValueError):
            contiguous_slice_indices(*bad)


class TestAShortSeriesIsHandledExplicitly:
    """A series shorter than the depth must not silently produce an invalid tensor.

    It produces a valid one, of the right length, over the right extent -- and the
    dataset says how often that happened.
    """

    @pytest.mark.parametrize("available,depth", [(12, 16), (12, 32), (20, 24), (31, 32)])
    def test_it_still_returns_exactly_depth_indices_inside_the_array(self, available, depth):
        idx = contiguous_slice_indices(available, depth)
        assert len(idx) == depth
        assert idx.min() >= 0 and idx.max() < available
        assert np.all(np.diff(idx) >= 0), "order must survive the short-series path"

    def test_it_covers_the_whole_short_series_and_invents_nothing(self):
        # The block cannot be a centred sub-block when the series is shorter than the
        # block: the correct answer is the entire series. Every real slice appears,
        # and no index outside the series does.
        idx = contiguous_slice_indices(12, 32)
        assert set(idx.tolist()) == set(range(12))

    def test_it_does_not_pile_up_ten_copies_of_the_end_slices(self):
        # The regression this path exists for. A clipped arange gives
        # [0]*11 + [1..10] + [11]*11 at (12, 32) -- two thirds of the volume is two
        # duplicated end planes, a constant slab no acquisition produces. Delegating to
        # the spanning sampler covers the identical extent with the repeats spread out.
        counts = np.bincount(contiguous_slice_indices(12, 32))
        assert counts.max() <= 3, f"end slices replicated {counts.max()} deep: {counts}"


class TestSpanningIsTheOldBehaviourExactly:
    """The default must be a no-op, or the volumetric arm changed the 2.5D arms."""

    @pytest.mark.parametrize(
        "available,depth", [(32, 16), (11, 16), (320, 16), (1, 4), (30, 5), (64, 32)]
    )
    def test_the_spanning_sampler_is_the_function_it_has_always_been(self, available, depth):
        assert SLICE_SAMPLERS["spanning"] is sample_slice_indices
        assert np.array_equal(
            SLICE_SAMPLERS["spanning"](available, depth),
            sample_slice_indices(available, depth),
        )

    def test_the_dataset_defaults_to_spanning(self, tmp_path):
        assert _dataset(_corpus(tmp_path)).slice_mode == "spanning"

    def test_an_unknown_mode_is_refused_by_name(self, tmp_path):
        with pytest.raises(ValueError, match="slice_mode"):
            _dataset(_corpus(tmp_path), slice_mode="middle")


# ---------------------------------------------------------------------------
# End to end through the loader
# ---------------------------------------------------------------------------
class TestTheBatchCarriesTheConfiguredDepthInOrder:
    @pytest.mark.parametrize("depth", [4, 8, 16])
    def test_the_slice_axis_is_the_configured_depth(self, tmp_path, depth):
        item = _dataset(_corpus(tmp_path), n_slices=depth, slice_mode="contiguous")[0]
        assert item["pixels"].shape[1] == depth, "D == configured_depth, in the batch"

    def test_the_batch_contract_shape_is_unchanged_by_the_volumetric_mode(self, tmp_path):
        # (n_slots, n_slices, 3, H, W) either way. The volumetric arm reads the centre
        # channel inside the encoder; the loader emits the same tensor it always has,
        # which is what lets collate, the checkpoint writer and inference stay put.
        corpus = _corpus(tmp_path)
        size = corpus.cache_size
        spanning = _dataset(corpus, n_slices=8)[0]["pixels"]
        contiguous = _dataset(corpus, n_slices=8, slice_mode="contiguous")[0]["pixels"]
        assert spanning.shape == contiguous.shape == (1, 8, 3, size, size)

    def test_the_slices_that_land_are_an_adjacent_ascending_run(self, tmp_path):
        # END TO END: the marker read out of the batch IS the cached slice index, so
        # this proves the loader delivered adjacent anatomy, not that the sampler
        # agrees with itself. 30 cached slices, depth 8 -> the centred block 11..18.
        dataset = _dataset(_corpus(tmp_path), n_slices=8, slice_mode="contiguous")
        assert _markers(dataset[0]["pixels"][0]) == [11, 12, 13, 14, 15, 16, 17, 18]

    def test_spanning_still_spreads_across_the_whole_extent(self, tmp_path):
        # The contrast that makes the sweep meaningful, asserted rather than assumed:
        # written out, so it cannot agree with a broken sampler by calling it twice.
        dataset = _dataset(_corpus(tmp_path), n_slices=5, slice_mode="spanning")
        assert _markers(dataset[0]["pixels"][0]) == [0, 7, 14, 22, 29]

    def test_the_two_modes_actually_differ_on_a_long_series(self, tmp_path):
        corpus = _corpus(tmp_path)
        span = _markers(_dataset(corpus, n_slices=8)[0]["pixels"][0])
        ctg = _markers(_dataset(corpus, n_slices=8, slice_mode="contiguous")[0]["pixels"][0])
        assert span != ctg, "if these agree the mode is not reaching the loader"

    def test_the_neighbour_channels_are_still_the_geometric_neighbours(self, tmp_path):
        # The 3 is unchanged and still means (previous, centre, next). VolumeEncoder
        # drops channels 0 and 2; this pins that they were never redefined for it.
        dataset = _dataset(_corpus(tmp_path), n_slices=4, slice_mode="contiguous")
        volume = dataset[0]["pixels"][0]
        previous, centre, following = (_markers(volume, channel=c) for c in (0, 1, 2))
        assert centre == [13, 14, 15, 16]
        assert previous == [12, 13, 14, 15]
        assert following == [14, 15, 16, 17]

    def test_a_short_series_is_counted_and_reported_not_hidden(self, tmp_path):
        corpus = _corpus(tmp_path, slice_counts=(6, 6))
        dataset = _dataset(corpus, n_slices=16, slice_mode="contiguous")
        assert dataset.short_series == {}, "the counter fills on read, not on construction"
        for i in range(len(dataset)):
            dataset[i]
        assert len(dataset.short_series) == 3
        assert set(dataset.short_series.values()) == {6}

    def test_a_series_long_enough_is_not_reported_as_short(self, tmp_path):
        dataset = _dataset(_corpus(tmp_path), n_slices=8, slice_mode="contiguous")
        for i in range(len(dataset)):
            dataset[i]
        assert dataset.short_series == {}


class TestNoCrossSeriesOrCrossStudyMixing:
    def test_every_study_reads_only_its_own_cached_series(self, tmp_path):
        corpus = _corpus(tmp_path, n_studies=4)
        dataset = _dataset(corpus, n_slices=8, slice_mode="contiguous")
        seen: dict[object, str] = {}
        for uid in dataset.study_uids:
            ref = dataset._series[uid][1]
            assert uid in str(ref.path), "a study read a series filed under another study"
            seen.setdefault(ref.path, uid)
            assert seen[ref.path] == uid, "two studies resolved to one cache file"
        assert len(seen) == len(dataset.study_uids)

    def test_selecting_a_subset_of_studies_emits_exactly_that_subset(self, tmp_path):
        # train/validation boundaries are the caller's `study_uids` and nothing here
        # may widen them: the fold split is a control for this arm, not a variable.
        corpus = _corpus(tmp_path, n_studies=4)
        wanted = sorted(corpus.labels["StudyInstanceUID"].astype(str))[:2]
        dataset = _dataset(corpus, study_uids=wanted, n_slices=8, slice_mode="contiguous")
        assert list(dataset.study_uids) == wanted
        assert len(dataset) == 2

    def test_the_mode_does_not_change_which_studies_a_split_contains(self, tmp_path):
        corpus = _corpus(tmp_path, n_studies=4)
        wanted = sorted(corpus.labels["StudyInstanceUID"].astype(str))[:3]
        span = _dataset(corpus, study_uids=wanted, n_slices=8)
        ctg = _dataset(corpus, study_uids=wanted, n_slices=8, slice_mode="contiguous")
        assert span.study_uids == ctg.study_uids

    def test_the_labels_a_study_gets_do_not_depend_on_the_mode(self, tmp_path):
        corpus = _corpus(tmp_path, n_studies=4)
        span = _dataset(corpus, n_slices=8)
        ctg = _dataset(corpus, n_slices=8, slice_mode="contiguous")
        for i in range(len(span)):
            assert torch.equal(span[i]["labels"], ctg[i]["labels"])
            assert torch.equal(span[i]["mask"], ctg[i]["mask"])
            assert span[i]["study_uid"] == ctg[i]["study_uid"]


# ---------------------------------------------------------------------------
# The hash, which is the whole "this is an addition" guarantee
# ---------------------------------------------------------------------------
class TestAddingTheSectionMovedNoExistingHash:
    """The `volume` SECTION moved no hash. A FIELD on an existing section is another
    matter: `optim.backbone_lr_mult` moved every digest, and
    `model.freeze_backbone_bn` moved them all again, both on purpose.
    `FROZEN_HASHES` is the current generation; `ON_DISK_ARTIFACT_HASHES` is the one
    earlier generation that a real run ever produced artifacts under, and is the only
    one of the two replaced generations kept as its own table."""

    @pytest.mark.parametrize("path,expected", sorted(FROZEN_HASHES.items()))
    def test_every_pre_existing_config_hashes_to_its_pinned_digest(self, path, expected):
        assert config_hash(_load(path)) == expected, (
            f"{path} re-hashed. Every result filed under {expected} is now untraceable "
            "to the file that produced it. A section added to TrainConfig must be listed "
            "in kappa.config._DEFAULTED_SECTIONS and must default to a no-op; a FIELD "
            "added to an existing section rehashes everything and needs a cohort "
            "boundary declared for it, as `optim.backbone_lr_mult` has."
        )

    def test_the_pinned_set_is_every_config_on_disk(self):
        # Globbed on purpose, in the one place a glob is the right instrument. The
        # explicit table catches a deleted config; only the glob catches an ADDED one,
        # and an added-but-unpinned config is exactly how all eight volumetric cells --
        # the arm under investigation -- came to be missing from a table whose whole
        # job is to make a result traceable to the file that produced it.
        on_disk = {
            f"configs/{path.name}" for path in (repo_root() / "configs").glob("*.yaml")
        }
        assert on_disk == set(FROZEN_HASHES)
        assert VOLUMETRIC_PATHS <= on_disk

    def test_the_on_disk_artifact_digests_are_recorded_and_no_longer_produced(self):
        # Both halves matter. The on-disk digests have to stay findable, because the
        # checkpoints and OOF tables already written are filed under them; and they
        # have to be genuinely stale, because a config that still produced one would
        # mean the code that wrote those artifacts is still what runs today.
        #
        # THE COMPLEMENT, NOT A SUBSET. This was `==` while every pinned config
        # predated `705c262` and therefore had a recorded artifact.
        # `POSTDATE_THE_HASH_MOVE` was added after that commit and has never been
        # trained, so there is no run directory to record a digest for -- fabricating
        # one here would claim an artifact that does not exist. But `subset of
        # FROZEN_HASHES` is too weak to express that: an EMPTIED or gutted
        # `ON_DISK_ARTIFACT_HASHES` is *also* a subset of `FROZEN_HASHES` and would
        # pass silently, destroying the only map from a digest an analyst actually
        # holds (say `e47ec0e7e863`) back to the config that produced it -- which is
        # the entire reason this table exists. Asserting the exact complement instead
        # means the ONLY configs allowed to lack an on-disk digest are the ones named
        # in `POSTDATE_THE_HASH_MOVE`; anything else missing is a regression, and
        # nothing may be silently dropped.
        assert set(FROZEN_HASHES) - set(ON_DISK_ARTIFACT_HASHES) == POSTDATE_THE_HASH_MOVE
        # This line is the companion guard and previously could pass vacuously if
        # `ON_DISK_ARTIFACT_HASHES` were emptied (`live & set()` is always `set()`).
        # It no longer can: the assertion above already fails first in that case,
        # because an emptied table's complement would be all of `FROZEN_HASHES`,
        # not `POSTDATE_THE_HASH_MOVE`. This line still earns its place for the
        # case the complement check cannot see -- a table that keeps every path but
        # corrupts a VALUE into one that now collides with a live hash.
        live = {config_hash(_load(path)) for path in FROZEN_HASHES}
        assert not (live & set(ON_DISK_ARTIFACT_HASHES.values()))

    def test_a_defaulted_volume_section_is_dropped_before_hashing(self):
        assert "volume" not in canonical_dump(loads_config("name: t\n"))

    def test_writing_the_default_out_explicitly_changes_nothing(self):
        # A file that spells `volume: {mode: spanning}` must hash as one that omits it,
        # or the four spanning cells would depend on whether someone wrote the section.
        implicit = loads_config("name: t\n")
        explicit = loads_config("name: t\nvolume:\n  mode: spanning\n")
        assert config_hash(implicit) == config_hash(explicit)

    def test_a_non_default_volume_section_is_hashed(self):
        # The other half. If contiguous were dropped too, 3d_ctg_dN and 3d_spn_dN would
        # share a hash, and the control arm would be indistinguishable from the arm it
        # controls.
        base = "name: t\nmodel:\n  backbone: r3d_18\n  drop_path_rate: 0.0\n"
        span = loads_config(base)
        ctg = loads_config(base + "volume:\n  mode: contiguous\n")
        assert config_hash(span) != config_hash(ctg)

    def test_the_default_is_spanning(self):
        assert VolumeConfig().mode == "spanning"


class TestTheConfigRefusesASilentMismatch:
    def test_a_2d_backbone_with_contiguous_sampling_is_refused(self):
        # Legal input to the loader, and a different experiment from the V1/V2/V3 grid.
        # Refused so that choosing it is deliberate rather than a copy-paste.
        with pytest.raises(ConfigError, match="contiguous"):
            loads_config("name: t\nvolume:\n  mode: contiguous\n")

    def test_a_video_backbone_keeping_drop_path_rate_is_refused(self):
        # torchvision has no stochastic depth, so a non-zero value would be discarded
        # and the file would claim a regulariser the run does not have.
        with pytest.raises(ConfigError, match="drop_path_rate"):
            loads_config("name: t\nmodel:\n  backbone: r3d_18\n  drop_path_rate: 0.2\n")

    def test_a_video_backbone_with_the_query_head_is_refused(self):
        with pytest.raises(ConfigError, match="query"):
            loads_config(
                "name: t\nmodel:\n  backbone: r3d_18\n  drop_path_rate: 0.0\n  head: query\n"
            )

    def test_a_depth_below_eight_is_refused_with_the_arithmetic_named(self):
        # Three temporal stride-2 stages divide D by 8; below that the depth axis is
        # gone before layer4 and the run is a 2D model paying 3D prices.
        with pytest.raises(ConfigError, match="n_slices"):
            loads_config(
                "name: t\nmodel:\n  backbone: r3d_18\n  drop_path_rate: 0.0\n"
                "data:\n  n_slices: 4\n"
            )

    def test_a_legal_volumetric_config_loads(self):
        cfg = loads_config(
            "name: t\nmodel:\n  backbone: r3d_18\n  drop_path_rate: 0.0\n"
            "data:\n  n_slices: 16\nvolume:\n  mode: contiguous\n"
        )
        assert cfg.data.n_slices == 16 and cfg.volume.mode == "contiguous"

    def test_freeze_backbone_bn_on_a_2d_backbone_is_refused(self):
        # Only VolumeEncoder overrides train() to re-freeze its BatchNorm3d layers;
        # SliceEncoder would silently ignore the flag, whatever BatchNorm the chosen
        # timm family happens to carry (tf_efficientnetv2_s carries 110).
        with pytest.raises(ConfigError, match="freeze_backbone_bn"):
            loads_config(
                "name: t\nmodel:\n  backbone: convnext_small\n  freeze_backbone_bn: true\n"
            )

    def test_freeze_backbone_bn_on_a_volumetric_backbone_loads(self):
        cfg = loads_config(
            "name: t\nmodel:\n  backbone: r3d_18\n  drop_path_rate: 0.0\n"
            "  freeze_backbone_bn: true\ndata:\n  n_slices: 16\n"
        )
        assert cfg.model.freeze_backbone_bn is True


class TestTheEightCellsOnDisk:
    @pytest.mark.parametrize("name,depth,mode", CELLS)
    def test_each_cell_says_what_its_filename_says(self, name, depth, mode):
        cfg = _load(f"configs/{name}.yaml")
        assert cfg.name == name, "`name` IS the output directory; it must match the stem"
        assert cfg.data.n_slices == depth
        assert cfg.volume.mode == mode
        assert cfg.model.backbone in VIDEO_BACKBONES

    def test_no_two_cells_write_to_the_same_directory(self):
        homes = {_load(f"configs/{n}.yaml").out_dir / n for n, _, _ in CELLS}
        assert len(homes) == len(CELLS), "two cells sharing a name silently overwrite"

    def test_the_eight_hashes_are_pairwise_distinct(self):
        hashes = {n: config_hash(_load(f"configs/{n}.yaml")) for n, _, _ in CELLS}
        assert len(set(hashes.values())) == len(CELLS), hashes

    def test_no_cell_collides_with_a_2d_config(self):
        # The 2.5D half is now FILTERED out of the pinned table rather than taken to be
        # all of it: FROZEN_HASHES covers every config on disk, including these eight,
        # so an unfiltered intersection would be the cells colliding with themselves.
        volumetric = {config_hash(_load(f"configs/{n}.yaml")) for n, _, _ in CELLS}
        two_and_a_half_d = {
            digest
            for path, digest in FROZEN_HASHES.items()
            if path not in VOLUMETRIC_PATHS
        }
        # The length check is what makes the intersection below mean something: it says
        # the 2.5D half is as many DISTINCT digests as it is files, so a collision
        # cannot hide by two paths folding onto one digest before the comparison runs.
        # `DELIBERATE_HASH_TWINS` is subtracted because each named twin pair
        # contributes one digest for two paths BY DESIGN -- see that set's comment.
        twinned_paths = sum(len(pair) - 1 for pair in DELIBERATE_HASH_TWINS)
        assert len(two_and_a_half_d) == len(FROZEN_HASHES) - len(CELLS) - twinned_paths
        assert not (volumetric & two_and_a_half_d)

    def test_only_depth_and_mode_move_across_the_eight(self):
        # The controls, asserted as a set of size one each. Anything that varies here
        # and is not depth or mode is a second variable in a one-variable sweep.
        cfgs = [_load(f"configs/{n}.yaml") for n, _, _ in CELLS]
        for attr, get in (
            ("slots", lambda c: c.data.slots),
            ("size", lambda c: c.data.size),
            ("physical_crop_mm", lambda c: c.data.physical_crop_mm),
            ("series_dropout", lambda c: c.data.series_dropout),
            ("backbone", lambda c: c.model.backbone),
            ("pretrained", lambda c: c.model.pretrained),
            ("drop_path_rate", lambda c: c.model.drop_path_rate),
            ("head", lambda c: c.model.head),
            ("lr", lambda c: c.optim.lr),
            ("schedule", lambda c: c.optim.schedule),
            ("epochs", lambda c: c.optim.epochs),
            ("batch_size", lambda c: c.optim.batch_size),
            ("grad_accum", lambda c: c.optim.grad_accum),
            ("amp_dtype", lambda c: c.optim.amp_dtype),
            ("loss", lambda c: (c.loss.kind, c.loss.gamma_neg, c.loss.gamma_pos, c.loss.clip)),
            ("folds", lambda c: (c.folds.n_folds, c.folds.file, c.folds.column)),
            ("augment", lambda c: c.augment),
        ):
            assert len({get(c) for c in cfgs}) == 1, f"{attr} varies across the volumetric sweep"

    def test_the_effective_batch_matches_the_2d_arms(self):
        # 4 x 4 = 16, the same optimiser step V1 takes at 8 x 2 and V3 at 4 x 4. If this
        # drifted, a 3D-vs-2.5D delta would be partly a schedule delta.
        baseline = _load("configs/v1_axial_fs1_convnext.yaml")
        want = baseline.optim.batch_size * baseline.optim.grad_accum
        for name, _, _ in CELLS:
            cfg = _load(f"configs/{name}.yaml")
            assert cfg.optim.batch_size * cfg.optim.grad_accum == want, name

    def test_the_folds_are_the_ones_every_other_arm_uses(self):
        baseline = _load("configs/v1_axial_fs1_convnext.yaml")
        for name, _, _ in CELLS:
            cfg = _load(f"configs/{name}.yaml")
            assert cfg.folds.file == baseline.folds.file
            assert cfg.folds.column == baseline.folds.column
            assert cfg.folds.n_folds == baseline.folds.n_folds

    def test_the_depth_ladder_is_complete_in_both_modes(self):
        got = {(mode, depth) for _, depth, mode in CELLS}
        want = {(m, d) for m in ("contiguous", "spanning") for d in (8, 16, 24, 32)}
        assert got == want


# ---------------------------------------------------------------------------
# The encoder
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def volume_encoder():
    pytest.importorskip("torchvision")
    from kappa.models.volume_encoder import VolumeEncoder

    # pretrained=False: the Kinetics weights are a 127 MB download that compute nodes
    # cannot reach, and none of these assertions are about the weight values.
    return VolumeEncoder("r3d_18", pretrained=False)


class TestTheVolumeEncoderContract:
    def test_the_two_copies_of_the_backbone_list_agree(self):
        # kappa.config writes VIDEO_BACKBONES out rather than importing it, because a
        # config must validate with no torch installed -- the same reason it writes
        # SLOT_NAMES out rather than importing SLOT_ORDER. Two copies means this test.
        from kappa.models.encoder import VIDEO_BACKBONES as from_models

        assert tuple(from_models) == tuple(VIDEO_BACKBONES)

    def test_the_kinetics_statistics_are_torchvisions_own(self):
        # Hardcoded so that kappa.data.dataset can normalise without torchvision, and
        # cross-checked here so the copy cannot drift from the weights it describes.
        pytest.importorskip("torchvision")
        from torchvision.models.video import R3D_18_Weights

        from kappa.models.volume_encoder import KINETICS_MEAN, KINETICS_STD

        published = R3D_18_Weights.KINETICS400_V1.transforms()
        assert KINETICS_MEAN == pytest.approx(float(np.mean(published.mean)))
        assert KINETICS_STD == pytest.approx(float(np.mean(published.std)))

    def test_the_staging_scripts_urls_are_torchvisions_urls(self):
        # slurm/stage_torch_weights.sh writes the URLs out because it must run without
        # importing torch: importing torch + torchvision and building r3d_18 can be
        # OOM-killed on a memory-capped login node. A wrong URL there would stage a file
        # the runtime never looks for, and training would fail on a compute node with
        # the weights sitting right there under another name.
        pytest.importorskip("torchvision")

        from kappa.models.volume_encoder import _weights_enum

        script = (repo_root() / "slurm" / "stage_torch_weights.sh").read_text()
        for name in VIDEO_BACKBONES:
            url = _weights_enum(name).url
            assert url in script, f"{name}: {url} is not the URL the staging script uses"

    def test_the_runtime_looks_where_the_staging_script_writes(self, tmp_path, monkeypatch):
        # The other half: same filename, same directory. `staged_weight_path` is what
        # the encoder checks and what the script's own --check reports.
        pytest.importorskip("torchvision")

        from kappa.models.volume_encoder import staged_weight_path

        monkeypatch.setenv("TORCH_HOME", str(tmp_path))
        path = staged_weight_path("r3d_18")
        assert path.parent == tmp_path / "hub" / "checkpoints"
        assert path.name == "r3d_18-b3b3357e.pth"

    def test_one_token_per_slot_comes_back_whatever_the_depth(self, volume_encoder):
        for depth in (8, 16):
            pixels = torch.randn(2, 1, depth, 3, 64, 64)
            out = volume_encoder(pixels, torch.ones(2, 1, dtype=torch.long))
            assert out.shape == (2, 1, 1, volume_encoder.embed_dim)

    def test_the_depth_axis_is_consumed_not_folded_into_the_batch(self, volume_encoder):
        # The claim that makes this a 3D model. SliceEncoder returns (B, S, N, dim) --
        # one embedding per slice, and the slice axis survives. Here it must not: if
        # this returned N tokens the "volume" would just be N independent planes.
        out = volume_encoder(torch.randn(2, 1, 16, 3, 64, 64), torch.ones(2, 1, dtype=torch.long))
        assert out.shape[2] == 1

    def test_permuting_the_depth_axis_changes_the_answer(self, volume_encoder):
        # The other half of "genuinely 3D": a per-slice 2D encoder followed by a
        # permutation-invariant pool would give the SAME embedding for a shuffled
        # volume. A 3D kernel cannot.
        volume_encoder.eval()
        pixels = torch.randn(1, 1, 16, 3, 64, 64)
        slots = torch.ones(1, 1, dtype=torch.long)
        with torch.no_grad():
            straight = volume_encoder(pixels, slots)
            reversed_ = volume_encoder(pixels.flip(2), slots)
        assert not torch.allclose(straight, reversed_, atol=1e-4)

    def test_an_absent_slot_comes_back_exactly_zero(self, volume_encoder):
        pixels = torch.randn(2, 2, 8, 3, 64, 64)
        slots = torch.tensor([[0, 1], [0, 1]])
        present = torch.tensor([[True, False], [True, True]])
        out = volume_encoder(pixels, slots, present)
        assert torch.equal(out[0, 1], torch.zeros_like(out[0, 1]))

    def test_an_absent_slot_cannot_influence_a_present_one(self, volume_encoder):
        # The BatchNorm3d argument, and the reason the encoder gathers rather than
        # masks afterwards. In TRAINING mode a masked-later build pools batch
        # statistics across the flattened slot axis, so garbage in an absent slot moves
        # the embedding of every present slot. Never computed beats multiplied by zero.
        volume_encoder.train()
        pixels = torch.randn(2, 2, 8, 3, 64, 64)
        slots = torch.tensor([[0, 1], [0, 1]])
        present = torch.tensor([[True, False], [True, False]])
        with torch.no_grad():
            clean = volume_encoder(pixels, slots, present)
            poisoned = pixels.clone()
            poisoned[:, 1] = 1e4
            dirty = volume_encoder(poisoned, slots, present)
        assert torch.equal(clean, dirty), "an absent slot reached the backbone"

    def test_a_non_finite_absent_slot_is_harmless(self, volume_encoder):
        volume_encoder.train()
        pixels = torch.randn(1, 2, 8, 3, 64, 64)
        pixels[:, 1] = float("nan")
        out = volume_encoder(pixels, torch.tensor([[0, 1]]), torch.tensor([[True, False]]))
        assert torch.isfinite(out).all()

    def test_the_wrong_channel_count_is_refused_by_name(self, volume_encoder):
        with pytest.raises(ValueError, match="2.5D neighbour stack"):
            volume_encoder(torch.randn(1, 1, 8, 1, 64, 64), torch.ones(1, 1, dtype=torch.long))

    def test_a_five_dim_batch_is_refused_by_name(self, volume_encoder):
        with pytest.raises(ValueError, match="pinned batch"):
            volume_encoder(torch.randn(1, 8, 3, 64, 64), torch.ones(1, 1, dtype=torch.long))

    def test_it_reads_the_centre_channel_and_not_the_neighbours(self, volume_encoder):
        # VolumeEncoder drops channels 0 and 2. Changing them must therefore change
        # nothing; changing channel 1 must change everything.
        volume_encoder.eval()
        slots = torch.ones(1, 1, dtype=torch.long)
        pixels = torch.randn(1, 1, 8, 3, 64, 64)
        with torch.no_grad():
            base = volume_encoder(pixels, slots)
            neighbours_changed = pixels.clone()
            neighbours_changed[:, :, :, 0] = 5.0
            neighbours_changed[:, :, :, 2] = -5.0
            assert torch.equal(base, volume_encoder(neighbours_changed, slots))
            centre_changed = pixels.clone()
            centre_changed[:, :, :, 1] = 5.0
            assert not torch.allclose(base, volume_encoder(centre_changed, slots))

    def test_unstaged_weights_raise_and_name_the_staging_script(self, tmp_path, monkeypatch):
        # The offline guarantee. Without this check torchvision goes into urllib and
        # BLOCKS on a compute node -- no output, no error, until the walltime kill.
        pytest.importorskip("torchvision")
        from kappa.models.volume_encoder import create_video_backbone

        monkeypatch.setenv("TORCH_HOME", str(tmp_path))
        with pytest.raises(FileNotFoundError, match="stage_torch_weights"):
            create_video_backbone("r3d_18", pretrained=True)

    def test_an_unknown_backbone_is_refused_with_the_legal_names(self):
        pytest.importorskip("torchvision")
        from kappa.models.volume_encoder import create_video_backbone

        with pytest.raises(ValueError, match="r3d_18"):
            create_video_backbone("resnet3d_18", pretrained=False)


@pytest.fixture(scope="module")
def volumetric_model():
    pytest.importorskip("torchvision")
    from kappa.models import StudyClassifier

    return StudyClassifier(backbone="r3d_18", n_slices=8, pretrained=False)


class TestTheMultiplierSlowsTheTRANSFERREDWeightsOnly:
    """`optim.backbone_lr_mult` against the arm it was written for.

    `VolumeEncoder` builds its FiLM conditioning as `nn.Parameter(torch.randn(...))`
    rather than as an `nn.Embedding`, so on this arm the tables are named
    `encoder.plane_film` with no `.weight` and are rank 3 rather than rank 2. Both
    differences are invisible to the 2.5D test in `tests/test_loop.py`, and both are
    reasons a prefix that happened to work there could still misfile them here.
    """

    def test_the_trunk_is_slowed_and_the_fresh_film_tables_are_not(self, volumetric_model):
        cfg = load_config(
            repo_root() / "configs/3d_ctg_d8.yaml",
            overrides=["optim.backbone_lr_mult=0.2"],
        )
        optimiser = build_optimiser(cfg, volumetric_model)
        scheduler = build_scheduler(cfg, optimiser, 100)
        with warnings.catch_warnings():
            # Stepping without an optimiser step warns about the order; the rates are
            # the claim here, and they have to be read AFTER a step because a scalar
            # `max_lr` would flatten the split at exactly that point.
            warnings.simplefilter("ignore", UserWarning)
            scheduler.step()
        by_name = {group["name"]: group for group in optimiser.param_groups}

        film = {
            id(parameter)
            for name, parameter in volumetric_model.named_parameters()
            if "film" in name
        }
        assert len(film) == 2
        assert {
            name for name, _ in volumetric_model.named_parameters() if "film" in name
        } == {"encoder.plane_film", "encoder.contrast_film"}
        holders = {
            name
            for name, group in by_name.items()
            if any(id(p) in film for p in group["params"])
        }
        assert holders == {"head-decay"}, "a randomly initialised table is not a trunk"

        trunk = {id(p) for p in volumetric_model.encoder.backbone.parameters()}
        in_backbone = {
            id(p)
            for name, group in by_name.items()
            if name.startswith("backbone")
            for p in group["params"]
        }
        assert in_backbone == trunk

        assert by_name["backbone-decay"]["lr"] == pytest.approx(
            by_name["head-decay"]["lr"] * 0.2
        )
        assert by_name["head-decay"]["lr"] > by_name["backbone-decay"]["lr"] > 0.0


class TestTheStudyClassifierSwitchesOnTheBackboneName:
    def test_it_builds_the_volumetric_encoder(self, volumetric_model):
        from kappa.models.volume_encoder import VolumeEncoder

        assert volumetric_model.volumetric and isinstance(volumetric_model.encoder, VolumeEncoder)

    def test_the_series_pool_is_sized_for_one_token(self, volumetric_model):
        # Not n_slices. A 32-entry positional table of which one entry is ever read
        # would be interpolated 32 -> 1 on every forward, describing a slice axis this
        # arm does not have.
        assert volumetric_model.series_pool.n_slices == 1

    def test_twelve_logits_come_out_of_the_pinned_batch(self, volumetric_model):
        out = volumetric_model(
            torch.randn(2, 1, 8, 3, 64, 64),
            torch.ones(2, 1, dtype=torch.bool),
            torch.ones(2, 1, dtype=torch.long),
        )
        assert out.logits.shape == (2, 12)
        assert torch.isfinite(out.logits).all()

    def test_it_trains_end_to_end_on_one_step(self, volumetric_model):
        # Wiring, not accuracy: a gradient reaching the 3D stem is the claim.
        out = volumetric_model(
            torch.randn(1, 1, 8, 3, 64, 64),
            torch.ones(1, 1, dtype=torch.bool),
            torch.ones(1, 1, dtype=torch.long),
        )
        out.logits.sum().backward()
        stem = volumetric_model.encoder.backbone.stem[0].weight
        assert stem.grad is not None and torch.isfinite(stem.grad).all()
        assert stem.grad.abs().sum() > 0, "no gradient reached the 3D stem"

    def test_a_timm_backbone_still_builds_the_2d_encoder(self):
        # The addition must not have moved the arm it was added beside.
        pytest.importorskip("timm")
        from kappa.models import SliceEncoder, StudyClassifier

        two_d = StudyClassifier(backbone="convnext_small", n_slices=16, pretrained=False)
        assert not two_d.volumetric
        assert isinstance(two_d.encoder, SliceEncoder)
        assert two_d.series_pool.n_slices == 16

    def test_the_normalisation_comes_from_the_video_weights(self, volumetric_model):
        from kappa.models.volume_encoder import KINETICS_MEAN

        config = volumetric_model.data_config
        assert config["mean"] == (KINETICS_MEAN,)
        # One element, broadcast across all three channels, so expanding the greyscale
        # volume to the RGB stem is exact rather than three differently-scaled copies.
        assert len(config["mean"]) == 1 and len(config["std"]) == 1

    def test_describe_states_the_depth_and_that_it_is_a_volume(self, volumetric_model):
        text = volumetric_model.describe()
        assert "VOLUME" in text and "r3d_18" in text


class TestFreezeBackboneBnSurvivesTheEpochReset:
    """`model.freeze_backbone_bn`, read against the failure mode it exists to avoid.

    `kappa.training.loop` calls `model.train()` once per epoch, and `nn.Module.train()`
    recurses into every submodule, resetting `training = True` unconditionally as it
    goes. An implementation that only flips the backbone's `BatchNorm3d` layers to eval
    mode at construction time -- or once before the training loop starts -- would be
    silently undone at the very first epoch boundary: the config would still claim
    `freeze_backbone_bn: true`, the run log would still print it, and the run would
    nonetheless train with ordinary batch statistics throughout. Every assertion below
    therefore calls `model.train()` FIRST and only then inspects state; a test that
    checked immediately after construction would not catch that failure.
    """

    def _model(self, *, freeze_backbone_bn):
        pytest.importorskip("torchvision")
        from kappa.models import StudyClassifier

        return StudyClassifier(
            backbone="r3d_18",
            n_slices=8,
            pretrained=False,
            freeze_backbone_bn=freeze_backbone_bn,
        )

    def _backbone_batchnorms(self, model):
        return [m for m in model.encoder.backbone.modules() if isinstance(m, torch.nn.BatchNorm3d)]

    def test_backbone_batchnorm_is_frozen_after_train_while_head_and_pools_are_not(self):
        model = self._model(freeze_backbone_bn=True)
        bns = self._backbone_batchnorms(model)
        assert len(bns) == 20, "r3d_18 is expected to carry 20 BatchNorm3d layers"

        # THE ORDER THAT MATTERS: train() first, exactly as loop.py calls it once per
        # epoch, and only then read state -- not the other way around.
        model.train()

        assert all(bn.training is False for bn in bns), (
            "BatchNorm3d in the backbone must report training=False AFTER model.train() "
            "-- if this fails, the freeze is being undone by the per-epoch reset"
        )
        assert model.head.training is True
        assert model.series_pool.training is True
        assert model.study_pool.training is True

    def test_backbone_batchnorm_stays_frozen_across_repeated_epoch_boundaries(self):
        # Not a one-off: loop.py calls model.train() again at the top of EVERY epoch,
        # so the re-freeze has to survive being called more than once too.
        model = self._model(freeze_backbone_bn=True)
        bns = self._backbone_batchnorms(model)
        for _ in range(3):
            model.eval()
            model.train()
            assert all(bn.training is False for bn in bns)

    def test_running_mean_is_unchanged_by_a_forward_pass_when_frozen(self):
        model = self._model(freeze_backbone_bn=True)
        model.train()
        bns = self._backbone_batchnorms(model)
        before = [bn.running_mean.clone() for bn in bns]

        model(
            torch.randn(2, 1, 8, 3, 64, 64),
            torch.ones(2, 1, dtype=torch.bool),
            torch.ones(2, 1, dtype=torch.long),
        )

        for bn, mean_before in zip(bns, before, strict=True):
            assert torch.equal(bn.running_mean, mean_before), (
                "running_mean moved during a forward pass in train mode -- the "
                "BatchNorm3d layer was not actually held in eval mode"
            )

    def test_unset_the_flag_behaves_exactly_as_before(self):
        # The default MUST be False and MUST be a no-op: every existing config trains
        # exactly as it does today.
        model = self._model(freeze_backbone_bn=False)
        bns = self._backbone_batchnorms(model)
        model.train()
        assert all(bn.training is True for bn in bns)

        before = [bn.running_mean.clone() for bn in bns]
        model(
            torch.randn(2, 1, 8, 3, 64, 64),
            torch.ones(2, 1, dtype=torch.bool),
            torch.ones(2, 1, dtype=torch.long),
        )
        moved = [
            not torch.equal(bn.running_mean, mean_before)
            for bn, mean_before in zip(bns, before, strict=True)
        ]
        assert any(moved), (
            "running_mean never moved with the flag unset -- ordinary BatchNorm "
            "training behaviour must be unaffected by this feature existing"
        )

    def test_the_default_is_false(self):
        from kappa.config import ModelConfig

        assert ModelConfig().freeze_backbone_bn is False

    def test_it_is_visible_in_the_run_log_header(self):
        cfg = load_config(
            repo_root() / "configs/3d_ctg_d8.yaml",
            overrides=["model.freeze_backbone_bn=true"],
        )
        from kappa.config import describe

        assert "freeze_bn=True" in describe(cfg)

    def test_it_is_visible_in_the_model_describe(self):
        model = self._model(freeze_backbone_bn=True)
        assert "FROZEN" in model.describe()
        assert "20" in model.describe()
