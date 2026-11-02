"""The shipped config files, checked as artifacts rather than as syntax.

An ablation arm is only evidence if it differs from its baseline in exactly one
place. There is no ``extends`` in this config format, so every arm is a full
copy, and a later edit to the baseline would silently turn a one-variable
comparison into a two-variable one. Nothing would raise; the delta would just
stop meaning what the file says it means. These tests are that check.

Two shapes of experiment live here and they need different checks. ``ARMS`` is
the one-section shape: an ablation arm that moves ``augment`` or ``distill`` and
nothing else. ``GRID`` is the 3 x 3 fusion x backbone family, where two sections
move by construction -- a backbone comes with the only input matrix its weights
accept -- so the guarantee is one level up: the axes are the only things that
move, and a delta is read ALONG an axis rather than across the diagonal.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kappa.config import config_hash, load_config
from kappa.data.paths import repo_root

# The one allowed config-hash collision, defined beside the pinned digest table it
# annotates rather than duplicated here -- two copies of an exemption is how one of
# them goes stale.
from test_volume import DELIBERATE_HASH_TWINS

BASELINE = "configs/baseline_axial_fs1.yaml"
#: Arm -> the section it is allowed to change, and the keys within it.
ARMS = {
    "configs/ablate_rotate.yaml": ("augment", {"rotate_deg"}),
    "configs/ablate_cutmix.yaml": ("augment", {"cutmix_prob", "cutmix_alpha"}),
    # The teacher and student arms. Both differ from the baseline in `distill` and
    # nowhere else, which is the whole reason the comparison is readable: the
    # teacher's margin has to be attributable to the report channel, and the
    # student's to the KD term. Registered here so that a later edit to either file
    # cannot quietly turn a one-variable comparison into a two-variable one.
    "configs/teacher_axial_fs1.yaml": (
        "distill",
        {"role", "report_features", "report_hidden", "report_dropout", "report_drop_rate"},
    ),
    "configs/distil_axial_fs1.yaml": (
        "distill",
        {
            "role",
            "teacher_ckpt",
            "report_features",
            "temperature",
            "lambda_supervised",
            "lambda_distill",
            "distill_unlabelled",
        },
    ),
}


def _load(path: str):
    return load_config(repo_root() / path)


@pytest.fixture(scope="module")
def baseline():
    return _load(BASELINE)


class TestTheBaselineIsUnaugmented:
    """Every later delta is read against this file; an augmentation here voids them."""

    def test_the_baseline_asks_for_no_augmentation(self, baseline):
        assert not baseline.augment.enabled
        assert baseline.eval.tta == ()
        assert baseline.optim.ema_decay == 0.0 and baseline.optim.swa_last_n == 0
        assert baseline.model.head == "linear"

    def test_the_480_probe_differs_from_the_baseline_only_in_resolution(self):
        # It exists to time 480 px, so anything else that moved is a confound.
        probe = _load("configs/baseline_axial_fs1_480.yaml")
        base = _load(BASELINE)
        differing = _diff(base.model_dump(mode="json"), probe.model_dump(mode="json"))
        assert differing <= {("name",), ("data", "size")}, differing


class TestEachArmChangesOneThing:
    @pytest.mark.parametrize("path", sorted(ARMS))
    def test_the_arm_differs_from_the_baseline_only_in_its_own_section(self, path, baseline):
        section, keys = ARMS[path]
        differing = _diff(baseline.model_dump(mode="json"), _load(path).model_dump(mode="json"))
        allowed = {("name",)} | {(section, key) for key in keys}
        assert differing <= allowed, f"{path} also changed {sorted(differing - allowed)}"

    @pytest.mark.parametrize("path", sorted(ARMS))
    def test_the_arm_actually_turns_something_on(self, path, baseline):
        # The mirror image: a copy that changed nothing would pass the test above
        # and measure the baseline twice.
        #
        # Stated against the arm's OWN declared section rather than against
        # `augment`. This used to read `assert arm.augment.enabled`, which held for
        # every arm while every arm was an augmentation arm, and became simply wrong
        # when the teacher and distillation arms arrived -- those turn on `distill`
        # and correctly leave `augment` at its defaults. Comparing the section to a
        # freshly defaulted instance of its own type is the general form, and it
        # keeps working for whatever section the next arm moves.
        arm = _load(path)
        section, _ = ARMS[path]
        configured = getattr(arm, section)
        assert configured != type(configured)(), (
            f"{path} declares itself a {section!r} arm but leaves that whole section "
            "at its defaults, so it would measure the baseline twice"
        )
        assert config_hash(arm) != config_hash(baseline)

    @pytest.mark.parametrize("path", sorted(ARMS))
    def test_the_arm_writes_somewhere_else(self, path, baseline):
        # `name` is excluded from config_hash, so two arms sharing one would be
        # different experiments writing to one out_dir. That cost a run already.
        assert _load(path).name != baseline.name

    def test_the_arms_do_not_collide_with_each_other(self):
        names = [_load(path).name for path in ARMS]
        assert len(set(names)) == len(names)
        hashes = [config_hash(_load(path)) for path in ARMS]
        assert len(set(hashes)) == len(hashes)


#: The 3 x 3 fusion x backbone grid: config file -> its slot count.
GRID = {
    "configs/v1_axial_fs1_convnext.yaml": 1,
    "configs/v1_axial_fs1_maxvit.yaml": 1,
    "configs/v1_axial_fs1_dinov2.yaml": 1,
    "configs/v2_fs_three_planes_convnext.yaml": 3,
    "configs/v2_fs_three_planes_maxvit.yaml": 3,
    "configs/v2_fs_three_planes_dinov2.yaml": 3,
    "configs/v3_all_slots_convnext.yaml": 6,
    "configs/v3_all_slots_maxvit.yaml": 6,
    "configs/v3_all_slots_dinov2.yaml": 6,
}

#: The only (backbone, size) pairs the grid may use. Each backbone's pretrained
#: weights accept one matrix: maxvit's ``_tf_384`` window tables are built for 384,
#: and patch14 needs a multiple of 14, which 384 is not. Enforced at load time by
#: ``TrainConfig._backbone_and_size_agree``; restated here so a config that quietly
#: acquires a fourth pair is caught as a *grid* error and not only as a load error.
LEGAL_PAIRS = {
    ("convnext_small", 384),
    ("maxvit_tiny_tf_384", 384),
    ("vit_base_patch14_dinov2.lvd142m", 392),
}

#: There is no longer an anchor cell. V1/convnext once reproduced the baseline's
#: hashed fields exactly (Axial_fs1 alone, convnext_small at 384 px), so the two
#: files were one experiment under two names. The grid now trains 8 epochs and the
#: baseline 10, so every grid config is distinct from it. Kept as None rather than
#: deleted: the collision is what a future edit could silently recreate, and a
#: named constant is where a reader looks for it.
ANCHOR_CELL = None


class TestTheFusionBackboneGrid:
    """The nine 3 x 3 cells, checked as a family rather than as nine arms.

    **They are not in ARMS on purpose.** ``ARMS`` encodes "differs from the baseline
    in exactly ONE section", and every cell off the V1/convnext corner differs in two
    -- ``data`` (slots, series_dropout, size) and ``model`` (backbone) -- with the V3
    row differing in ``optim`` as well. Registering them would fail
    ``test_the_arm_differs_from_the_baseline_only_in_its_own_section`` for configs
    that are correct, and the honest fix is not to loosen that test: an ablation arm
    really must move one section, and this is not an ablation arm. It is a grid, and
    the guarantee a grid needs is different -- that the axes are the only things that
    move, so a delta read ALONG one axis is attributable. That is what
    ``test_no_cell_drifts_outside_the_two_grid_axes`` asserts, and it is the same
    property one level up.
    """

    def test_all_nine_cells_load(self):
        for path in GRID:
            assert _load(path) is not None, path

    def test_every_name_matches_its_filename_stem(self):
        # `name` is the output directory (scripts/train.py: out_dir/cfg.name), so a
        # stem/name mismatch sends a run's artifacts to a path nobody will look in,
        # and two cells agreeing on `name` send two experiments to one path. That
        # collision has cost a run here already.
        for path in GRID:
            assert _load(path).name == Path(path).stem, path

    def test_no_cell_writes_where_another_cell_or_the_baseline_writes(self, baseline):
        homes = {Path(_load(p).out_dir) / _load(p).name for p in GRID}
        assert len(homes) == len(GRID)
        assert Path(baseline.out_dir) / baseline.name not in homes

    def test_the_nine_hashes_are_pairwise_distinct(self):
        # Nine cells, nine experiments. Two sharing a hash would be filed as one, and
        # the seed-spread reading of a shared hash (two seeds, one experiment) would
        # then be applied to two different models.
        hashes = {path: config_hash(_load(path)) for path in GRID}
        assert len(set(hashes.values())) == len(GRID), hashes

    def test_exactly_the_anchor_cell_shares_the_baselines_hash(self, baseline):
        # Nothing may share the baseline's hash. V1 x convnext_small at 384 px once
        # reproduced the baseline exactly; now that the grid trains a different
        # budget, two files with one hash would be one experiment under two names,
        # quietly measuring the baseline twice and reporting it as a grid result.
        base = config_hash(baseline)
        sharing = sorted(path for path in GRID if config_hash(_load(path)) == base)
        assert sharing == [], (
            f"{sharing} share the baseline's config hash, so they are the same "
            "experiment under another name and one of them measures nothing new"
        )

    def test_every_multi_slot_cell_asks_for_series_dropout(self):
        # Zero-fill, the `present` flag and random series dropout are one design.
        # Above one slot, series_dropout=0.0 means the encoder never meets an empty
        # slot in training, so `present=0` carries no information at inference --
        # where 3,841 of 4,407 studies are missing at least one slot. kappa.config
        # refuses it; this is the check that the shipped files never test that refusal.
        for path, n_slots in GRID.items():
            cfg = _load(path)
            assert len(cfg.data.slots) == n_slots, path
            if n_slots > 1:
                assert cfg.data.series_dropout > 0.0, path
            else:
                assert cfg.data.series_dropout == 0.0, path

    def test_every_cell_keeps_the_effective_batch_and_fits_its_backbones_ceiling(
        self, baseline
    ):
        # The memory ceiling is per backbone, not per slot count. Sizing the
        # micro-batch by slot count alone is wrong: V2 x maxvit at 8 x 2 ran an H200
        # NVL (143771 MiB) out of memory about 50 s in, while V2 x convnext, with the
        # same three slots, the same 8 x 2 and the same card, trained normally.
        #
        # The governing quantity is stacks per forward, because the encoder
        # flattens the batch to one slice-stack per (study, slot, slice) before the
        # backbone (`pixels.reshape(-1, channels, height, width)`,
        # src/kappa/models/encoder.py):
        #
        #     stacks = batch_size * n_slots * n_slices
        #
        # Observed on H200 NVL (143771 MiB), n_slices = 16 in every config:
        #
        #     convnext_small       384 stacks  fits           V2 x convnext
        #     maxvit_tiny_tf_384   128 stacks  fits           V1 x maxvit
        #     maxvit_tiny_tf_384   384 stacks  out of memory  V2 x maxvit
        #     vit_base_patch14...  128 stacks  fits           V1 x dinov2
        #
        # so the ceilings below are observations and one interpolation, not a
        # formula. maxvit's 192 is bounded (128 fits, 384 does not) and has not
        # itself been run; DINOv2's 128 is the largest footprint seen to fit for a
        # backbone heavier than the one that ran out of memory (86M parameters
        # against 35M, and full attention over 28 x 28 = 784 tokens rather than
        # windowed attention).
        #
        # The micro-batch is free to move because only maxvit has batch-dependent
        # normalisation (`batchnorm_layers=34`). convnext and DINOv2 use LayerNorm,
        # which is per-sample, so shrinking their micro-batch leaves gradients
        # identical and moves only the activation peak.
        #
        # What must not move is the effective batch. grad_accum rises as batch_size
        # falls, as baseline_axial_fs1_480 already does; an out-of-memory fix that
        # edited batch_size alone would move the optimiser too.
        #
        # V1 x DINOv2 sits at the 128 ceiling itself because it already trained
        # cleanly there, and re-sizing it would re-hash a completed experiment.
        ceilings = {
            "convnext_small": 384,
            "maxvit_tiny_tf_384": 192,
            "vit_base_patch14_dinov2.lvd142m": 128,
        }
        sized = {
            (1, "convnext_small"): (8, 2),
            (1, "maxvit_tiny_tf_384"): (8, 2),
            (1, "vit_base_patch14_dinov2.lvd142m"): (8, 2),
            (3, "convnext_small"): (8, 2),
            (3, "maxvit_tiny_tf_384"): (4, 4),
            (3, "vit_base_patch14_dinov2.lvd142m"): (2, 8),
            (6, "convnext_small"): (4, 4),
            (6, "maxvit_tiny_tf_384"): (2, 8),
            (6, "vit_base_patch14_dinov2.lvd142m"): (1, 16),
        }
        effective = baseline.optim.batch_size * baseline.optim.grad_accum
        for path, n_slots in GRID.items():
            cfg = _load(path)
            backbone = cfg.model.backbone
            assert (cfg.optim.batch_size, cfg.optim.grad_accum) == sized[
                (n_slots, backbone)
            ], path
            # The optimiser step is the baseline's in every cell, so a delta read
            # along either axis is about the axis and not about the schedule.
            assert cfg.optim.batch_size * cfg.optim.grad_accum == effective, path
            stacks = cfg.optim.batch_size * n_slots * cfg.data.n_slices
            assert stacks <= ceilings[backbone], (
                f"{path} asks {stacks} stacks per forward against a measured "
                f"ceiling of {ceilings[backbone]} for {backbone}"
            )

    def test_every_cell_uses_one_of_the_three_legal_backbone_size_pairs(self):
        for path in GRID:
            cfg = _load(path)
            assert (cfg.model.backbone, cfg.data.size) in LEGAL_PAIRS, path

    def test_the_grid_is_complete_and_the_field_of_view_is_constant_across_it(self):
        # Three fusion levels x three backbones, every cell present exactly once -- a
        # missing cell turns a grid into a set of unpaired runs. And the crop stays 150 mm
        # everywhere, which is why the 392 px cells are comparable to the 384 px ones:
        # only sampling density moves (0.383 vs 0.391 mm/px), not the anatomy in frame.
        cells = {(GRID[path], _load(path).model.backbone) for path in GRID}
        assert len(cells) == 9
        assert {n for n, _ in cells} == {1, 3, 6}
        assert {b for _, b in cells} == {b for b, _ in LEGAL_PAIRS}
        assert {_load(path).data.physical_crop_mm for path in GRID} == {150.0}

    @pytest.mark.parametrize("path", sorted(GRID))
    def test_no_cell_drifts_outside_the_two_grid_axes(self, path, baseline):
        # The ARMS guarantee, generalised to a grid: there is no `extends` in this
        # format, so each cell is a full copy of the baseline and a later edit to
        # either file could silently add a third variable. `data.size` is listed
        # because it is not free -- it is forced by the backbone axis (patch14 cannot
        # take 384) -- and `optim.batch_size`/`grad_accum` because V3 must fit on one
        # GPU at a fixed effective batch. Anything else that moved is a confound.
        allowed = {
            ("name",),
            ("data", "slots"),
            ("data", "series_dropout"),
            ("data", "size"),
            ("model", "backbone"),
            ("optim", "batch_size"),
            ("optim", "grad_accum"),
            # The grid trains 8 epochs where the baseline trains 10. On the first
            # grid run, validation macro peaked at epoch 4 and fell by epoch 9 while
            # the training loss kept dropping, so the budget is 8 and selection takes
            # the best epoch rather than the last.
            ("optim", "epochs"),
        }
        differing = _diff(baseline.model_dump(mode="json"), _load(path).model_dump(mode="json"))
        assert differing <= allowed, f"{path} also changed {sorted(differing - allowed)}"


#: The 10-epoch programme: every config in it, and the config it is a verbatim copy
#: of. There is no ``extends`` in this format, so a config is its base plus the
#: fields in ``PROGRAMME_AXES`` and nothing else, and a later edit to either file
#: could silently add a variable. That is the same exposure ``ARMS`` and ``GRID``
#: exist for, one programme along.
PROGRAMME_BASE = {
    # Learning-rate structure: one base, three backbone LR multipliers.
    "b2_v2_convnext_lr100": "configs/v2_fs_three_planes_convnext.yaml",
    "b2_v2_convnext_lr10": "configs/v2_fs_three_planes_convnext.yaml",
    "b2_v2_convnext_lr03": "configs/v2_fs_three_planes_convnext.yaml",
    # The multiplier carried to the other three arms. Both candidate values ship;
    # only one of each pair is run.
    "b2_v1_convnext_lr03": "configs/v1_axial_fs1_convnext.yaml",
    "b2_v1_convnext_lr10": "configs/v1_axial_fs1_convnext.yaml",
    "b2_v2_maxvit_lr03": "configs/v2_fs_three_planes_maxvit.yaml",
    "b2_v2_maxvit_lr10": "configs/v2_fs_three_planes_maxvit.yaml",
    "b2_v3_convnext_lr03": "configs/v3_all_slots_convnext.yaml",
    "b2_v3_convnext_lr10": "configs/v3_all_slots_convnext.yaml",
    # Learning-rate level. The multiplier probe was still improving at the highest
    # backbone rate sampled, and `backbone_lr_mult` is capped at 1.0, so the
    # backbone cannot be raised without raising `lr`. These three move `lr`.
    "b2_v2_convnext_h1e4": "configs/v2_fs_three_planes_convnext.yaml",
    "b2_v2_convnext_h1e3_b5e5": "configs/v2_fs_three_planes_convnext.yaml",
    "b2_v2_convnext_h2e4_b5e5": "configs/v2_fs_three_planes_convnext.yaml",
    # Augmentation on the V2 x convnext arm.
    "b2_aug_rot": "configs/v2_fs_three_planes_convnext.yaml",
    "b2_aug_rotbc": "configs/v2_fs_three_planes_convnext.yaml",
    "b2_aug_cutmix": "configs/v2_fs_three_planes_convnext.yaml",
    "b2_aug_all": "configs/v2_fs_three_planes_convnext.yaml",
    # Architecture.
    "b3_v2_convnext_perlabel": "configs/v2_fs_three_planes_convnext.yaml",
    "b3_v2_convnext_query_h1": "configs/v2_fs_three_planes_convnext.yaml",
    "b3_v2_coatnet384": "configs/v2_fs_three_planes_convnext.yaml",
    "b3_compose": "configs/v2_fs_three_planes_convnext.yaml",
}

#: What each cell is allowed to move against its base, ON TOP of the two fields every
#: cell in the programme moves (``optim.epochs`` and ``optim.backbone_lr_mult``).
#: Anything else that differs is a confound, and the point of listing it here is that
#: adding an axis has to be a deliberate edit to this dict rather than a diff nobody
#: reads.
_AUG4 = {
    ("augment", "rotate_deg"),
    ("augment", "brightness"),
    ("augment", "contrast"),
    ("augment", "cutmix_prob"),
}
PROGRAMME_AXES = {
    # The LR LEVEL cells move `optim.lr`, which the implicit allowance below does not
    # cover -- it grants only `name`, `optim.epochs` and `optim.backbone_lr_mult`.
    "b2_v2_convnext_h1e4": {("optim", "lr")},
    "b2_v2_convnext_h1e3_b5e5": {("optim", "lr")},
    "b2_v2_convnext_h2e4_b5e5": {("optim", "lr")},
    "b2_aug_rot": {("augment", "rotate_deg")},
    "b2_aug_rotbc": {("augment", "rotate_deg"), ("augment", "brightness"),
                     ("augment", "contrast")},
    "b2_aug_cutmix": {("augment", "cutmix_prob")},
    "b2_aug_all": set(_AUG4),
    "b3_v2_convnext_perlabel": {("pool", "per_label_slots")},
    # ONE DESIGN, NOT TWO ARMS. With head="query" and aux_dim=0 the pooling modules
    # receive no gradient at all -- the query head cross-attends the encoder's slice
    # tokens directly and nothing downstream consumes the pooled vector -- so
    # pool.head_residual is what makes the arm measure anything. TrainConfig refuses
    # head_residual on any head but query, and ModelConfig refuses head_num_heads on
    # any head but query, so these three fields move together or not at all.
    "b3_v2_convnext_query_h1": {("model", "head"), ("model", "head_num_heads"),
                                ("pool", "head_residual")},
    # The backbone axis carries its own micro-batch, exactly as it does in GRID: the
    # effective batch is held at 16 so the optimiser step is not a third moving part.
    "b3_v2_coatnet384": {("model", "backbone"), ("optim", "batch_size"),
                         ("optim", "grad_accum")},
    "b3_compose": set(_AUG4) | {("pool", "per_label_slots")},
}


class TestTheTenEpochProgramme:
    """The 10-epoch programme, checked as a family rather than as separate arms.

    **These are not in ARMS on purpose, and for a new reason.** ``ARMS`` encodes
    "differs from the baseline in exactly ONE section"; every cell here moves ``optim``
    twice over (``epochs`` AND ``backbone_lr_mult``) and most of them move a second
    section with it. ``GRID`` does not fit either: these are not cells of the fusion x
    backbone grid and they are not at its 8-epoch budget. So the guarantee is the same
    one level up -- each cell's OWN declared axes are the only things that move against
    its OWN base, which is what ``PROGRAMME_AXES`` spells out.

    **Why a new budget needs new files.** ``optim.epochs`` IS in ``config_hash``
    (v1_axial_fs1_convnext hashes fc8ee481925e at 8 and e296f5db8835 at 10), so editing
    a shipped config from 8 to 10 would re-hash an arm whose folds are already on disk.
    ``name`` is NOT in the hash but IS the output directory, so the new budget also
    needs a new name or the two cohorts share one ``data/runs/<name>/`` and the second
    overwrites the first. Both halves are asserted below.

    **Why the whole programme is unreadable against the 8-epoch grid.** Under
    ``onecycle`` the LR anneals to zero at the end of the schedule, so the metric peak
    tracks schedule length -- maxvit peaked at e9 of 10 and at e6 of 8. That is why
    ``b2_v2_convnext_lr100`` exists: it is the epoch-only control, this base at 10
    epochs and at the default single rate, so the LR cells have a reference inside
    their own budget.
    """

    @pytest.mark.parametrize("name", sorted(PROGRAMME_BASE))
    def test_every_cell_loads(self, name):
        assert _load(f"configs/{name}.yaml") is not None

    @pytest.mark.parametrize("name", sorted(PROGRAMME_BASE))
    def test_every_name_matches_its_filename_stem(self, name):
        # `name` is the output directory (scripts/train.py: out_dir/cfg.name), so a
        # stem/name mismatch sends a run's artifacts to a path nobody will look in.
        assert _load(f"configs/{name}.yaml").name == name

    def test_no_two_configs_anywhere_in_the_directory_share_a_name(self):
        # Stated over the WHOLE directory, not just this programme: `name` is the
        # output path, so a collision with any other shipped config -- a grid cell, an
        # ablation arm, the baseline -- is two experiments writing into one directory
        # with no warning. That has already cost a run.
        paths = sorted((repo_root() / "configs").glob("*.yaml"))
        names = [_load(f"configs/{p.name}").name for p in paths]
        duplicated = {n for n in names if names.count(n) > 1}
        assert not duplicated, f"{sorted(duplicated)} are declared by more than one file"
        assert len(names) == len(paths)

    @pytest.mark.parametrize("name", sorted(PROGRAMME_BASE))
    def test_every_cell_is_ten_epochs_and_five_folds(self, name):
        # The two things that define the programme. `optim.epochs` is hashed, so a cell
        # that quietly stayed at 8 is a different experiment filed under this one's
        # name; `folds.n_folds` is not hashed as an epoch budget is, but a 3-fold cell
        # among 5-fold siblings makes the fold-mean it contributes to incomparable.
        cfg = _load(f"configs/{name}.yaml")
        assert cfg.optim.epochs == 10, name
        assert cfg.folds.n_folds == 5, name

    @pytest.mark.parametrize("name", sorted(PROGRAMME_BASE))
    def test_no_cell_drifts_outside_its_declared_axes(self, name):
        # The ARMS guarantee, restated per cell against its own base. Every cell moves
        # `name` and the two optim fields; anything beyond that has to be declared in
        # PROGRAMME_AXES, so an accidental third variable fails here rather than
        # becoming an uninterpretable delta in a table.
        base = _load(PROGRAMME_BASE[name])
        cell = _load(f"configs/{name}.yaml")
        allowed = {("name",), ("optim", "epochs"), ("optim", "backbone_lr_mult")}
        allowed |= PROGRAMME_AXES.get(name, set())
        differing = _diff(base.model_dump(mode="json"), cell.model_dump(mode="json"))
        assert differing <= allowed, f"{name} also changed {sorted(differing - allowed)}"

    @pytest.mark.parametrize("name", sorted(PROGRAMME_BASE))
    def test_every_cell_is_a_new_experiment_and_not_a_rename_of_its_base(self, name):
        # The mirror image of the test above: a copy that changed nothing would pass it
        # and measure its base twice under a second name.
        assert config_hash(_load(f"configs/{name}.yaml")) != config_hash(
            _load(PROGRAMME_BASE[name])
        ), f"{name} hashes identically to {PROGRAMME_BASE[name]}"

    def test_no_cell_shares_a_hash_with_any_other_config_on_disk(self):
        # Seventeen cells, seventeen experiments, and none of them a duplicate of
        # anything already shipped. Two files with one hash are ONE experiment: the
        # seed-spread reading of a shared hash ("two seeds, one config") would then be
        # applied to two different models. The live near-miss is b2_v1_convnext_*:
        # configs/baseline_axial_fs1.yaml is already Axial_fs1 / convnext_small / 384 /
        # 10 epochs / 8 x 2, so `optim.backbone_lr_mult` is the ONLY hashed field
        # separating them -- which is why there is deliberately no b2_v1_convnext_lr100.
        paths = sorted((repo_root() / "configs").glob("*.yaml"))
        hashes = {}
        for path in paths:
            digest = config_hash(_load(f"configs/{path.name}"))
            hashes.setdefault(digest, []).append(path.name)
        collisions = {h: f for h, f in hashes.items() if len(f) > 1}
        # One pair is allowed, and it is named in tests/test_volume.py rather than here
        # so there is a single place to read why. `w6_e2_control` is a deliberate re-run
        # of `b4_aug_query_h4`'s exact recipe under a new `name`, so that the
        # distillation arms have a no-distillation control built by the same commit
        # they are; the shared digest is the proof the recipe did not drift, and the
        # new `name` keeps the two runs' artifacts in separate directories. Anything
        # else sharing a digest is still two names for one experiment and still fails.
        allowed = {frozenset(pair) for pair in DELIBERATE_HASH_TWINS}
        unexpected = {h: f for h, f in collisions.items() if frozenset(f) not in allowed}
        assert not unexpected, f"configs sharing a hash: {unexpected}"

    def test_the_control_changed_neither_learning_rate_field(self):
        # b2_v2_convnext_lr100 is the epoch-only control: 1.0 is OptimConfig's
        # default single-rate optimiser, so this config isolates the budget change and
        # every other config in the programme is read against it. It is written out
        # rather than omitted so the file says so.
        #
        # This used to assert the control was the only config at both LR defaults.
        # That stopped holding once the multiplier probe found nothing better than
        # 1.0 and the downstream configs were regenerated at 1.0. What still holds is
        # that the control changed neither LR field. Distinctness from every other
        # config is enforced, more strictly, by
        # test_no_cell_shares_a_hash_with_any_other_config_on_disk above.
        control = _load("configs/b2_v2_convnext_lr100.yaml")
        assert control.optim.backbone_lr_mult == 1.0
        assert control.optim.lr == 5.0e-5
        assert control.optim.epochs == 10

    @pytest.mark.parametrize("name", sorted(PROGRAMME_BASE))
    def test_no_cell_asks_for_a_horizontal_flip(self, name):
        # AugmentConfig refuses hflip=True outright, so this cannot be true by
        # accident -- it is asserted because the refusal is the thing worth restating.
        # Flipping a knee swaps MEDIAL and LATERAL. Medial/Lateral Meniscus and
        # Medial/Lateral OA could be swapped back, but MCL has no LCL column and PF OA
        # is not lateralised, so no relabelling makes a flipped study correct for all
        # twelve. Four of the twelve labels are defined on that axis, and the failure
        # is a plausible wrong number on them rather than a crash.
        assert _load(f"configs/{name}.yaml").augment.hflip is False


def _diff(left: dict, right: dict, prefix: tuple[str, ...] = ()) -> set[tuple[str, ...]]:
    """Paths at which two resolved configs disagree."""
    paths: set[tuple[str, ...]] = set()
    for key in set(left) | set(right):
        here = (*prefix, key)
        a, b = left.get(key), right.get(key)
        if isinstance(a, dict) and isinstance(b, dict):
            paths |= _diff(a, b, here)
        elif a != b:
            paths.add(here)
    return paths
