"""A config that accepts a misspelled key files a run under the wrong hypothesis.

Why this exists: ``droppath_rate: 0.3`` in a YAML file looks like it configured
stochastic depth. With a plain ``dict`` it configures nothing, the run uses the
default 0.2, and the result is recorded as evidence about a drop-path rate that
was never used. Nothing in the training log would say so, and nothing in the
scores would look wrong. So every section here sets ``extra="forbid"`` and the
first class below proves it at *every* level, not just the top one -- a nested
section that quietly accepts extras is the same bug one indent down.

The second half of this file is about the hash. ``config_hash`` has to be stable
across the fields that name a run (``name``, ``out_dir``, ``seed``, ``seeds``) and
sensitive to every field that defines the experiment, because that is what makes a
seed-to-seed spread readable as a noise floor rather than as two different
experiments. Both halves of that claim are tested, and the sensitivity half is
tested field by field rather than on one example: a knob left out of the hash is
invisible until two experiments collide under one identifier.

``SLOT_NAMES`` is cross-checked against the manifest's ``SLOT_ORDER`` here because
``kappa.config`` deliberately writes the slot names out rather than importing them
-- a config must validate on a machine where the manifest module is absent -- and
two copies of a slot ordering is exactly the drift that mislabels a whole run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kappa.config import (
    _DEFAULTED_SECTIONS,
    NOTEBOOK_LIMIT_SECONDS,
    SLOT_NAMES,
    TTA_TRANSFORMS,
    AugmentConfig,
    ConfigError,
    DataConfig,
    DistillConfig,
    EvalConfig,
    FoldsConfig,
    LossConfig,
    ModelConfig,
    OptimConfig,
    PoolConfig,
    TrainConfig,
    apply_overrides,
    canonical_dump,
    config_hash,
    describe,
    load_config,
    loads_config,
)
from kappa.constants import LABELS
from kappa.data.paths import repo_root

BASELINE = "configs/baseline_axial_fs1.yaml"
VARIANT_480 = "configs/baseline_axial_fs1_480.yaml"


def _yaml(**over: str) -> str:
    """A minimal valid config as YAML text, with sections appended verbatim."""
    body = ["name: t"]
    body.extend(over.values())
    return "\n".join(body) + "\n"


def _config(**over) -> TrainConfig:
    """A valid TrainConfig with keyword overrides applied to the top level."""
    options = {"name": "t"}
    options.update(over)
    return TrainConfig(**options)


class TestUnknownKeysAreRefused:
    """extra="forbid" at every level, and the error has to name the offending key."""

    def test_an_unknown_top_level_key_is_refused(self):
        with pytest.raises(ConfigError, match="epocs"):
            loads_config("name: t\nepocs: 10\n")

    def test_a_misspelled_drop_path_rate_is_refused_by_name(self):
        # The motivating case. Accepting this would run at drop_path_rate=0.2 while
        # the file, and therefore the experiment log, claims 0.3.
        with pytest.raises(ConfigError, match="droppath_rate"):
            loads_config("name: t\nmodel:\n  droppath_rate: 0.3\n")

    @pytest.mark.parametrize(
        "section", ["data", "model", "loss", "optim", "folds"]
    )
    def test_every_nested_section_forbids_extras(self, section):
        with pytest.raises(ConfigError, match="nonsense"):
            loads_config(f"name: t\n{section}:\n  nonsense: 1\n")

    @pytest.mark.parametrize(
        "model",
        [DataConfig, ModelConfig, LossConfig, OptimConfig, FoldsConfig, TrainConfig],
    )
    def test_the_pydantic_models_themselves_forbid_extras(self, model):
        # Asserted on the class rather than only through the YAML door, so a section
        # reachable only from Python cannot quietly differ.
        assert model.model_config["extra"] == "forbid"

    def test_a_config_with_no_name_is_refused(self):
        with pytest.raises(ConfigError, match="name"):
            loads_config("seed: 0\n")

    def test_a_yaml_list_at_the_top_level_is_refused(self):
        with pytest.raises(ConfigError, match="must be a mapping"):
            loads_config("- a\n- b\n")

    def test_an_empty_file_is_refused_because_it_has_no_name(self):
        with pytest.raises(ConfigError):
            loads_config("")


class TestValuesThatWouldSilentlyMisconfigure:
    """The validators that exist because the loader would otherwise ignore the value."""

    def test_a_stack_other_than_three_is_refused_with_the_contract_named(self):
        # The pinned batch is (B, N_SLOTS, n_slices, 3, H, W). A config asking for 5
        # would be accepted and then ignored by the loader.
        with pytest.raises(ConfigError, match="2.5D neighbour stack"):
            loads_config("name: t\ndata:\n  stack: 5\n")

    def test_the_pinned_stack_of_three_is_accepted(self):
        assert loads_config("name: t\ndata:\n  stack: 3\n").data.stack == 3

    def test_a_size_off_the_32_grid_is_refused(self):
        # Backbones downsample by 2 at least five times, so an off-grid size is
        # rounded inside the model and the effective mm/px stops matching the config.
        with pytest.raises(ConfigError, match="multiple of 32"):
            loads_config("name: t\ndata:\n  size: 300\n")

    def test_an_unknown_slot_name_is_refused_and_the_legal_names_are_listed(self):
        with pytest.raises(ConfigError, match="Axial_fs1"):
            loads_config('name: t\ndata:\n  slots: ["Axial_FS1"]\n')

    def test_a_duplicated_slot_is_refused(self):
        with pytest.raises(ConfigError, match="duplicate slot"):
            loads_config('name: t\ndata:\n  slots: ["Axial_fs1", "Axial_fs1"]\n')

    def test_an_empty_slot_list_is_refused(self):
        with pytest.raises(ConfigError, match="at least one slot"):
            loads_config("name: t\ndata:\n  slots: []\n")

    def test_a_parquet_fold_file_is_refused_because_gitignore_globs_it(self):
        with pytest.raises(ConfigError, match="gitignore"):
            loads_config("name: t\nfolds:\n  file: data/folds/f.parquet\n")

    def test_a_csv_fold_file_is_accepted(self):
        cfg = loads_config("name: t\nfolds:\n  file: data/folds/f.csv\n")
        assert cfg.folds.file is not None and cfg.folds.file.suffix == ".csv"

    def test_the_fold_file_in_the_baseline_config_is_not_gitignored(self):
        # The suffix check is only half the guarantee; this is the other half, taken
        # from git itself rather than from a rule restated in Python.
        import subprocess

        cfg = load_config(repo_root() / BASELINE)
        assert cfg.folds.file is not None
        done = subprocess.run(
            ["git", "check-ignore", "-q", str(cfg.folds.file)],
            cwd=repo_root(),
            capture_output=True,
        )
        assert done.returncode == 1, f"{cfg.folds.file} is gitignored and would never be tracked"

    def test_two_slots_without_series_dropout_are_refused(self):
        # Zero-filling an absent slot without random series dropout teaches the model
        # that "absent" and "black" are the same input.
        with pytest.raises(ConfigError, match="series_dropout"):
            loads_config(
                'name: t\ndata:\n  slots: ["Axial_fs1", "Coronal_fs1"]\n  series_dropout: 0.0\n'
            )

    def test_two_slots_with_series_dropout_are_accepted(self):
        cfg = loads_config(
            'name: t\ndata:\n  slots: ["Axial_fs1", "Coronal_fs1"]\n  series_dropout: 0.1\n'
        )
        assert cfg.data.series_dropout == pytest.approx(0.1)

    def test_one_slot_with_no_dropout_is_accepted_because_nothing_is_ever_absent(self):
        assert loads_config('name: t\ndata:\n  slots: ["Axial_fs1"]\n').data.series_dropout == 0.0

    def test_a_fold_outside_the_range_is_refused(self):
        with pytest.raises(ConfigError, match="outside 0..4"):
            loads_config("name: t\nfolds:\n  n_folds: 5\n  only: [7]\n")

    @pytest.mark.parametrize("lr", ["0.0", "-1.0", "0.5"])
    def test_a_learning_rate_outside_the_published_range_is_refused(self, lr):
        with pytest.raises(ConfigError):
            loads_config(f"name: t\noptim:\n  lr: {lr}\n")

    def test_a_thirty_epoch_schedule_loads_but_the_default_is_short(self):
        # "Long epochs" sits in the didn't-work list of two RSNA 2024 top-3 teams, so
        # the burden of proof is on lengthening it -- but the bound is a sanity limit,
        # not a prohibition.
        assert OptimConfig().epochs == 10
        assert loads_config("name: t\noptim:\n  epochs: 30\n").optim.epochs == 30
        with pytest.raises(ConfigError):
            loads_config("name: t\noptim:\n  epochs: 500\n")

    @pytest.mark.parametrize("mult", ["0.0", "-0.5", "1.5", "10"])
    def test_a_backbone_multiplier_outside_zero_to_one_is_refused(self, mult):
        # Above 1.0 would run the pretrained trunk FASTER than the fresh head, which
        # is the opposite of the hypothesis and is what a mistyped `10` looks like.
        # Zero would freeze the trunk by arithmetic rather than by saying so, leaving
        # a run whose config claims to train a backbone it never updates.
        with pytest.raises(ConfigError):
            loads_config(f"name: t\noptim:\n  backbone_lr_mult: {mult}\n")

    def test_the_backbone_multiplier_defaults_to_one_which_is_the_old_optimiser(self):
        # Every config on disk omits this field, so its default is the promise that
        # they train exactly as they did: one rate for the trunk and the head.
        assert OptimConfig().backbone_lr_mult == 1.0
        assert loads_config("name: t\n").optim.backbone_lr_mult == 1.0
        slowed = loads_config("name: t\noptim:\n  backbone_lr_mult: 0.2\n")
        assert slowed.optim.backbone_lr_mult == 0.2

    def test_an_unknown_schedule_is_refused(self):
        with pytest.raises(ConfigError, match="schedule"):
            loads_config("name: t\noptim:\n  schedule: linear\n")

    def test_an_unknown_fold_column_is_refused(self):
        with pytest.raises(ConfigError, match="column"):
            loads_config("name: t\nfolds:\n  column: fold_patient\n")

    def test_both_legal_fold_columns_load(self):
        assert loads_config("name: t\nfolds:\n  column: fold\n").folds.column == "fold"
        assert (
            loads_config("name: t\nfolds:\n  column: fold_scanner\n").folds.column
            == "fold_scanner"
        )


class TestTheBackboneAndTheInputMatrixMustAgree:
    """A backbone/size mismatch is a NUMBER, not a crash, so it is refused at load.

    ``maxvit_*_tf_384`` weights carry window partitioning and relative-position tables
    built for a 384 px input; a patch14 ViT interpolates its pretrained position
    embedding onto a ``size/14`` token grid. Off-spec, timm either raises deep inside
    that interpolation -- an hour into a queued job, with the GPU already spent -- or
    interpolates onto a grid the weights never trained on, finishes, and reports a
    macro AUC. The second outcome is the one this class exists for.

    The check has to see ``model.backbone`` and ``data.size`` together, so it is a
    ``TrainConfig`` model validator; a field validator on either section sees one half
    of a pair whose *combination* is the error.
    """

    def _yaml_for(self, backbone: str, size: int) -> str:
        return f"name: t\nmodel:\n  backbone: {backbone}\ndata:\n  size: {size}\n"

    @pytest.mark.parametrize("size", [256, 392, 480])
    def test_a_maxvit_tf_384_away_from_384_is_refused(self, size):
        with pytest.raises(ConfigError) as excinfo:
            loads_config(self._yaml_for("maxvit_tiny_tf_384", size))
        message = str(excinfo.value)
        # Both halves of the offending pair are named: "not a multiple of 32" alone
        # would send the reader to data.size when the fixable field may be either.
        assert "maxvit_tiny_tf_384" in message and str(size) in message

    @pytest.mark.parametrize("size", [384, 480])
    def test_a_patch14_backbone_off_the_14_grid_is_refused(self, size):
        with pytest.raises(ConfigError) as excinfo:
            loads_config(self._yaml_for("vit_base_patch14_dinov2.lvd142m", size))
        message = str(excinfo.value)
        assert "vit_base_patch14_dinov2" in message and str(size) in message
        assert "14" in message

    def test_the_patch14_message_states_the_fractional_token_grid(self):
        # 384/14 = 27.43. The arithmetic is in the message because the alternative
        # error a reader would otherwise see comes from inside timm.
        with pytest.raises(ConfigError, match="27.43"):
            loads_config(self._yaml_for("vit_base_patch14_dinov2.lvd142m", 384))

    def test_the_14_grid_is_not_a_licence_for_conv_backbones(self):
        # The mirror of the case above, and the reason DataConfig's own size check was
        # loosened rather than deleted: 392 became loadable so DINOv2 could use it, and
        # convnext at 392 must still be refused or that loosening lost a guarantee.
        with pytest.raises(ConfigError) as excinfo:
            loads_config(self._yaml_for("convnext_small", 392))
        message = str(excinfo.value)
        assert "convnext_small" in message and "392" in message
        assert "multiple of 32" in message

    def test_a_size_on_neither_grid_is_refused_before_the_backbone_is_consulted(self):
        # 300 is off both moduli, so no backbone choice can rescue it and the error
        # belongs to data.size alone.
        with pytest.raises(ConfigError, match="multiple of 32"):
            loads_config("name: t\ndata:\n  size: 300\n")

    @pytest.mark.parametrize(
        "backbone,size",
        [
            ("convnext_small", 384),
            ("maxvit_tiny_tf_384", 384),
            ("vit_base_patch14_dinov2.lvd142m", 392),
        ],
    )
    def test_the_three_legal_pairs_of_the_grid_load(self, backbone, size):
        # The three (backbone, size) pairs the nine grid configs are built from. A
        # validator that refused one of these would take out a third of the grid.
        cfg = loads_config(self._yaml_for(backbone, size))
        assert (cfg.model.backbone, cfg.data.size) == (backbone, size)

    def test_the_pretrained_tag_does_not_smuggle_a_constraint_past_the_check(self):
        # timm names are `family.pretrained_tag`, and the constraint belongs to the
        # family. Matching the whole string would let `maxvit_tiny_tf_384.in1k` -- the
        # same weights under their explicit tag -- through at any size.
        with pytest.raises(ConfigError, match="maxvit_tiny_tf_384"):
            loads_config(self._yaml_for("maxvit_tiny_tf_384.in1k", 480))

    def test_the_field_of_view_is_the_same_150_mm_at_both_legal_sizes(self):
        # Why the backbone comparison stays readable across 384 and 392: the crop is
        # physical, so only sampling density moves, by 2%.
        at_384 = DataConfig(size=384)
        at_392 = DataConfig(size=392)
        assert at_384.physical_crop_mm == at_392.physical_crop_mm == pytest.approx(150.0)
        assert at_392.mm_per_px == pytest.approx(150.0 / 392.0)
        assert abs(at_392.mm_per_px / at_384.mm_per_px - 1.0) < 0.021


class TestPhysicalScale:
    """mm/px is arithmetic on two config fields, so it can be asserted exactly."""

    def test_the_default_geometry_is_150_mm_over_384_px(self):
        cfg = DataConfig()
        assert cfg.physical_crop_mm == pytest.approx(150.0)
        assert cfg.size == 384
        assert cfg.mm_per_px == pytest.approx(150.0 / 384.0)
        assert cfg.mm_per_px == pytest.approx(0.390625)

    def test_halving_the_pixel_count_doubles_the_pixel_size(self):
        # The point of a fixed physical crop: the field of view is held and the
        # resolution floats, never the other way around.
        coarse = DataConfig(size=192)
        fine = DataConfig(size=384)
        assert coarse.mm_per_px == pytest.approx(2.0 * fine.mm_per_px)

    def test_a_crop_of_zero_millimetres_is_refused(self):
        with pytest.raises(ConfigError):
            loads_config("name: t\ndata:\n  physical_crop_mm: 0.0\n")


class TestSlotNames:
    """The config's slot names must equal the manifest's, or a run trains the wrong slot."""

    def test_slot_names_match_the_manifest_slot_order_exactly(self):
        from kappa.data.manifest import SLOT_ORDER

        derived = tuple(f"{plane}_fs{fs}" for plane, fs in SLOT_ORDER)
        assert SLOT_NAMES == derived

    def test_there_are_six_slots(self):
        from kappa.data.manifest import N_SLOTS

        assert len(SLOT_NAMES) == N_SLOTS == 6

    def test_a_slot_name_index_is_its_global_slot_id(self):
        # slot_ids in the batch are positions in this tuple. Axial_fs1 is slot 1 and
        # is the one slot at 100% coverage, which is why job one selects it.
        assert SLOT_NAMES.index("Axial_fs1") == 1


class TestConfigHash:
    """Stable over what names a run, sensitive to everything that defines one."""

    @pytest.mark.parametrize(
        "field,value",
        [
            ("name", "other"),
            ("seed", 99),
            ("seeds", (3, 4)),
            ("out_dir", "elsewhere"),
        ],
    )
    def test_the_hash_ignores_the_fields_that_only_name_a_run(self, field, value):
        base = _config()
        assert config_hash(base) == config_hash(_config(**{field: value}))

    def test_two_seeds_of_one_config_share_a_hash(self):
        # This is what makes a seed-to-seed spread readable as a noise floor: if the
        # seed entered the hash, the two runs would be two experiments.
        assert config_hash(_config(seed=0)) == config_hash(_config(seed=1))

    @pytest.mark.parametrize(
        "section,field,value",
        [
            ("data", "n_slices", 8),
            ("data", "size", 256),
            ("data", "physical_crop_mm", 120.0),
            ("data", "slots", ("Coronal_fs1",)),
            ("data", "series_dropout", 0.1),
            ("data", "anchors_override", False),
            ("model", "backbone", "caformer_s18"),
            ("model", "pretrained", False),
            ("model", "drop_path_rate", 0.3),
            ("model", "aux_dim", 17),
            ("loss", "gamma_neg", 2.0),
            ("loss", "gamma_pos", 1.0),
            ("loss", "clip", 0.0),
            ("optim", "lr", 1e-4),
            ("optim", "backbone_lr_mult", 0.2),
            ("optim", "weight_decay", 0.05),
            ("optim", "schedule", "cosine"),
            ("optim", "warmup_pct", 0.2),
            ("optim", "epochs", 20),
            ("optim", "batch_size", 32),
            ("optim", "grad_accum", 4),
            ("optim", "amp", False),
            ("optim", "amp_dtype", "fp16"),
            ("folds", "n_folds", 10),
            ("folds", "column", "fold_scanner"),
            # `pool` is default-DROPPED, not unhashed. The two are easy to confuse and
            # the difference is the whole point: at its default the section vanishes so
            # that adding it moved no existing digest, and away from its default it
            # must separate the arm from the baseline like any other knob.
            ("pool", "per_label_slots", True),
        ],
    )
    def test_every_experiment_defining_field_changes_the_hash(self, section, field, value):
        # Field by field rather than by example. A knob left out of the hash is
        # invisible until two experiments collide under one identifier, and by then
        # the results table is already wrong.
        base = _config()
        changed = base.model_copy(
            update={section: getattr(base, section).model_copy(update={field: value})}
        )
        assert config_hash(base) != config_hash(changed), f"{section}.{field} is not hashed"

    def test_the_hash_is_stable_across_processes(self):
        # sha256 of a sorted json dump, not Python's salted hash(): a digest that
        # changed between interpreters could not identify an experiment at all.
        import subprocess
        import sys

        expected = config_hash(load_config(repo_root() / BASELINE))
        out = subprocess.run(
            [
                sys.executable,
                "-c",
                "from kappa.config import config_hash, load_config;"
                f"print(config_hash(load_config({str(repo_root() / BASELINE)!r})))",
            ],
            cwd=repo_root(),
            capture_output=True,
            text=True,
            env={"PYTHONPATH": str(repo_root() / "src"), "PATH": "/usr/bin:/bin"},
        )
        assert out.stdout.strip() == expected, out.stderr
        json.loads(canonical_dump(load_config(repo_root() / BASELINE)))

    def test_the_canonical_dump_excludes_exactly_the_run_naming_fields(self):
        # Parsed, not grepped. A substring check over the JSON blob cannot tell the
        # top-level `name` from a section field that happens to be spelled the same,
        # so it would pass or fail for reasons that have nothing to do with the hash.
        cfg = _config()
        payload = json.loads(canonical_dump(cfg))
        expected = set(TrainConfig.model_fields) - {"name", "out_dir", "seed", "seeds"}
        # A section left at its DEFAULT is dropped too, so that adding one did not
        # move every hash filed before it existed -- including the first measured
        # baseline's. That is deliberate and documented on `canonical_dump`; the test
        # has to know about it or it reads as a bug. The property that matters is
        # asserted separately, below.
        #
        # Driven from `_DEFAULTED_SECTIONS` rather than naming `distill` by hand. The
        # hand-written version failed the moment `volume` was added, which is the
        # right alarm in the wrong place: the invariant this test owns is "exactly the
        # run-naming fields are excluded", and *which* sections are default-dropped is
        # `canonical_dump`'s business. `TestADefaultedSectionMovesNoHash` is where
        # that list is actually held to account.
        for section, factory in _DEFAULTED_SECTIONS:
            if getattr(cfg, section) == factory():
                expected -= {section}
        assert set(payload) == expected
        assert payload["model"]["backbone"] == cfg.model.backbone
        assert "n_slices" in payload["data"]
        assert "gamma_neg" in payload["loss"]

    def test_a_configured_distill_section_is_hashed(self):
        # The other half of the exemption above, and the half that keeps an ablation
        # readable: a teacher, a distilling student and an image-only baseline must
        # be three different experiments. If a non-default `distill` section were
        # dropped as well, every arm of the E1-E4 ladder would share one hash, write
        # to one run name, and silently overwrite each other's results.
        baseline = _config()
        teacher = _config(distill={"role": "teacher", "report_features": "f.parquet"})
        assert teacher.distill != DistillConfig()
        assert "distill" in json.loads(canonical_dump(teacher))
        assert config_hash(teacher) != config_hash(baseline)

    def test_the_hash_ignores_only_the_run_naming_fields(self):
        # The property the substring check was reaching for: renaming a run, moving
        # its output or reseeding it must not change which experiment it is, and
        # changing anything else must.
        base = _config()
        digest = config_hash(base)
        for field, value in (
            ("name", "some-other-run"),
            ("out_dir", Path("data/elsewhere")),
            ("seed", base.seed + 1),
            ("seeds", (7, 8)),
        ):
            assert config_hash(base.model_copy(update={field: value})) == digest, field
        moved = base.model_copy(update={"model": base.model.model_copy(update={"head": "query"})})
        assert config_hash(moved) != digest

    def test_the_hash_is_twelve_hex_characters_by_default(self):
        digest = config_hash(_config())
        assert len(digest) == 12
        assert set(digest) <= set("0123456789abcdef")


class TestOverrides:
    """Overrides are applied before validation, so a misspelled one is still refused."""

    def test_a_dotted_override_reaches_a_nested_section(self):
        merged = apply_overrides({"name": "t"}, ["optim.epochs=2"])
        assert merged["optim"]["epochs"] == 2

    def test_a_boolean_override_is_parsed_as_a_bool_not_a_string(self):
        # "false" as a string is truthy, which would quietly download weights inside
        # an offline job that explicitly asked not to.
        cfg = TrainConfig(**apply_overrides({"name": "t"}, ["model.pretrained=false"]))
        assert cfg.model.pretrained is False

    def test_a_numeric_override_is_parsed_as_a_number(self):
        cfg = TrainConfig(**apply_overrides({"name": "t"}, ["optim.lr=1e-4"]))
        assert cfg.optim.lr == pytest.approx(1e-4)

    def test_a_list_override_is_parsed_as_a_list(self):
        cfg = TrainConfig(**apply_overrides({"name": "t"}, ['data.slots=["Coronal_fs1"]']))
        assert cfg.data.slots == ("Coronal_fs1",)

    def test_an_override_of_an_unknown_key_is_still_refused_by_extra_forbid(self):
        with pytest.raises(ConfigError, match="droppath_rate"):
            load_config(repo_root() / BASELINE, overrides=["model.droppath_rate=0.3"])

    def test_a_malformed_override_is_refused_with_the_expected_form(self):
        with pytest.raises(ConfigError, match="key.path=value"):
            apply_overrides({"name": "t"}, ["optim.epochs"])

    def test_an_override_does_not_mutate_the_input_mapping(self):
        raw = {"name": "t", "optim": {"epochs": 10}}
        apply_overrides(raw, ["optim.epochs=2"])
        assert raw["optim"]["epochs"] == 10


class TestTheBaselineConfigOnDisk:
    """The file that describes the first training job has to load and mean what it says."""

    def test_the_baseline_config_loads(self):
        cfg = load_config(repo_root() / BASELINE)
        assert cfg.name == "baseline_axial_fs1"

    def test_the_baseline_is_axial_fs1_only_with_no_series_dropout(self):
        # Axial_fs1 is the one slot at 100% coverage, so nothing is ever absent and
        # there is nothing for dropout to protect against on day one.
        cfg = load_config(repo_root() / BASELINE)
        assert cfg.data.slots == ("Axial_fs1",)
        assert cfg.data.series_dropout == 0.0

    def test_the_baseline_geometry_is_the_documented_16_slices_at_384_over_150_mm(self):
        cfg = load_config(repo_root() / BASELINE)
        assert (cfg.data.n_slices, cfg.data.stack, cfg.data.size) == (16, 3, 384)
        assert cfg.data.physical_crop_mm == pytest.approx(150.0)
        assert cfg.data.mm_per_px == pytest.approx(0.390625, abs=5e-4)

    def test_the_baseline_names_all_three_of_its_inputs(self):
        # A run whose log does not say where its pixels and labels came from cannot be
        # reproduced, so none of these may be left to a default.
        cfg = load_config(repo_root() / BASELINE)
        assert cfg.data.manifest is not None
        assert cfg.data.labels is not None
        assert cfg.data.cache_dir is not None
        assert cfg.folds.file is not None

    def test_the_baseline_uses_a_small_backbone_and_the_asl_defaults(self):
        cfg = load_config(repo_root() / BASELINE)
        assert cfg.model.backbone == "convnext_small"
        assert (cfg.loss.gamma_neg, cfg.loss.gamma_pos, cfg.loss.clip) == (4.0, 0.0, 0.05)

    def test_the_baseline_learning_rate_is_inside_the_published_rsna_range(self):
        cfg = load_config(repo_root() / BASELINE)
        assert 2.5e-5 <= cfg.optim.lr <= 1e-4

    def test_the_baseline_schedule_is_short(self):
        assert load_config(repo_root() / BASELINE).optim.epochs <= 20

    def test_the_baseline_carries_at_least_two_seeds_for_the_noise_floor(self):
        # --noise-floor needs a spread and one seed has none, so `seeds` has to be
        # populated in the file. `seed` stays the single seed a normal run uses --
        # run_experiment reads cfg.seed, measure_noise_floor reads cfg.seed_list() --
        # and neither field enters config_hash, so all of them share one experiment id.
        cfg = load_config(repo_root() / BASELINE)
        assert len(cfg.seeds) >= 2
        assert cfg.seed == 0
        assert config_hash(cfg) == config_hash(cfg.model_copy(update={"seeds": ()}))


class TestTheResolutionVariantOnDisk:
    """`baseline_axial_fs1_480.yaml` is the baseline with one value changed.

    A variant that quietly drifts on anything but `size` stops being a readable delta,
    so the diff is asserted rather than trusted.
    """

    def test_the_variant_loads_and_is_480_px(self):
        assert load_config(repo_root() / VARIANT_480).data.size == 480

    def test_the_variant_differs_from_the_baseline_in_size_and_nothing_else(self):
        base = load_config(repo_root() / BASELINE)
        variant = load_config(repo_root() / VARIANT_480)
        assert base.data.size == 384
        # `name` is excluded here and asserted DIFFERENT below: it is not an
        # experimental variable (config_hash drops it) but it is the output path,
        # and the two runs must not write to one directory.
        assert base.model_dump(exclude={"data", "name"}) == variant.model_dump(
            exclude={"data", "name"}
        )
        drop = {"size"}
        assert {k: v for k, v in base.data.model_dump().items() if k not in drop} == {
            k: v for k, v in variant.data.model_dump().items() if k not in drop
        }

    def test_the_variant_does_not_write_into_the_baselines_output_directory(self):
        # scripts/train.py resolves `out_dir = args.out_dir or Path(cfg.out_dir)/cfg.name`.
        # Identical names meant probe-480 overwrote the baseline's checkpoints and OOF
        # table with no warning -- and the comparison the run exists for reads the
        # artifacts that were just destroyed.
        base = load_config(repo_root() / BASELINE)
        variant = load_config(repo_root() / VARIANT_480)
        assert base.name != variant.name
        assert Path(base.out_dir) / base.name != Path(variant.out_dir) / variant.name

    def test_the_two_configs_hash_differently(self):
        # `size` is a real experimental variable, so the variant must not share the
        # baseline's experiment id.
        assert config_hash(load_config(repo_root() / BASELINE)) != config_hash(
            load_config(repo_root() / VARIANT_480)
        )


class TestDescribe:
    """The startup banner has to name the backbone, the hash and the scale."""

    def test_describe_names_the_backbone_and_whether_the_weights_are_pretrained(self):
        text = describe(_config())
        assert "convnext_small" in text
        assert "pretrained=True" in text

    def test_describe_carries_the_config_hash(self):
        cfg = _config()
        assert config_hash(cfg) in describe(cfg)

    def test_describe_states_the_physical_scale_not_only_the_pixel_size(self):
        text = describe(_config())
        assert "mm/px" in text
        assert "384" in text

    def test_describe_distinguishes_a_differential_run_from_a_uniform_one(self):
        # Two runs at the same `lr` and different `backbone_lr_mult` are two different
        # experiments, and the Slurm log is where a reader finds out which one ran.
        # Both rates are printed, so the trunk's rate does not have to be multiplied
        # out by hand at the point of reading.
        uniform = describe(_config())
        assert "one rate: backbone and head" in uniform
        split = describe(_config(optim={"lr": 1e-4, "backbone_lr_mult": 0.1}))
        assert "lr=0.0001 head / 1e-05 backbone" in split
        assert "backbone x0.1" in split

    def test_describe_states_the_fold_column(self):
        # Which grouping a run used is part of what its number means: 408 per-study
        # scanner fingerprints exist and site memorisation is worth ~0.05 of a random-fold
        # score, so `fold` and `fold_scanner` are two different results.
        assert "column=fold" in describe(_config())

    def test_describe_states_the_seeds_and_the_out_dir(self):
        text = describe(_config(seeds=(0, 1)))
        assert "[0, 1]" in text
        assert "out_dir" in text


class TestNotebookLimit:
    """The runtime term's denominator is a constant, not a number retyped per call site."""

    def test_the_notebook_limit_is_nine_hours_in_seconds(self):
        assert NOTEBOOK_LIMIT_SECONDS == 32_400 == 9 * 3600

    def test_the_inference_module_agrees_with_the_config_module(self):
        # predict.py restates it so a notebook need not import the config module. Two
        # copies of a denominator is how a runtime term silently changes scale.
        from kappa.inference.predict import NOTEBOOK_LIMIT_SECONDS as inference_limit

        assert inference_limit == NOTEBOOK_LIMIT_SECONDS


class TestSelectedFoldsAndSeeds:
    """``only`` and ``seeds`` are the two fields a Slurm array element reads."""

    def test_no_only_means_every_fold(self):
        assert FoldsConfig(n_folds=5).selected() == (0, 1, 2, 3, 4)

    def test_only_restricts_the_selection_and_keeps_its_order(self):
        assert FoldsConfig(n_folds=5, only=(3, 1)).selected() == (3, 1)

    def test_no_seeds_means_the_single_seed(self):
        assert _config(seed=7).seed_list() == (7,)

    def test_seeds_override_the_single_seed(self):
        assert _config(seed=7, seeds=(1, 2)).seed_list() == (1, 2)


def test_loading_a_file_and_loading_its_text_agree():
    path = repo_root() / BASELINE
    assert config_hash(load_config(path)) == config_hash(
        loads_config(path.read_text(encoding="utf-8"))
    )


def test_a_config_section_is_frozen_so_a_run_cannot_edit_its_own_config():
    cfg = _config()
    with pytest.raises(Exception):
        cfg.optim.lr = 1.0  # type: ignore[misc]


class TestAugmentConfig:
    """Augmentation is expressible, and every field is off until a file says otherwise."""

    def test_every_augmentation_defaults_to_off(self):
        aug = AugmentConfig()
        assert (aug.rotate_deg, aug.brightness, aug.contrast, aug.cutmix_prob) == (
            0.0,
            0.0,
            0.0,
            0.0,
        )
        assert aug.hflip is False
        assert not aug.enabled and not aug.per_sample_enabled and not aug.batch_enabled

    def test_the_shipped_configs_ask_for_no_augmentation(self):
        # The load-bearing one. The baseline and noise-floor runs measure the specified
        # baseline, and every later change is read as a delta against them.
        # If a shipped config ever acquires an augmentation, that comparison is void.
        for path in ("configs/baseline_axial_fs1.yaml", "configs/baseline_axial_fs1_480.yaml"):
            cfg = load_config(repo_root() / path)
            assert not cfg.augment.enabled, path
            assert cfg.model.head == "linear", path
            assert cfg.loss.kind == "asl" and cfg.loss.gamma_neg_by_label is None, path
            assert cfg.optim.ema_decay == 0.0 and cfg.optim.swa_last_n == 0, path
            assert cfg.eval.tta == (), path

    def test_horizontal_flip_is_refused_with_the_anatomy(self):
        with pytest.raises(ConfigError) as excinfo:
            loads_config("name: t\naugment: {hflip: true}\n")
        message = str(excinfo.value)
        # The message has to carry the reason: "extra inputs are not permitted" reads
        # like a typo and invites someone to add the field for real.
        assert "medial" in message.lower() and "lateral" in message.lower()
        assert "MCL" in message and "PF OA" in message

    def test_horizontal_flip_off_is_accepted_so_the_tombstone_is_not_a_trap(self):
        assert loads_config("name: t\naugment: {hflip: false}\n").augment.hflip is False

    @pytest.mark.parametrize(
        "text",
        [
            "name: t\naugment: {rotate_deg: 90}\n",
            "name: t\naugment: {rotate_deg: -1}\n",
            "name: t\naugment: {brightness: 2.0}\n",
            "name: t\naugment: {contrast: -0.1}\n",
            "name: t\naugment: {cutmix_prob: 1.5}\n",
            "name: t\naugment: {cutmix_alpha: 0.0}\n",
            "name: t\naugment: {rotation: 5}\n",
        ],
    )
    def test_out_of_range_and_misspelled_augmentations_are_refused(self, text):
        with pytest.raises(ConfigError):
            loads_config(text)

    def test_the_two_hooks_are_reported_separately(self):
        # They run in different places -- per-sample in the dataset, CutMix in the
        # collate -- so a caller wiring one must be able to ask about it alone.
        rotate_only = AugmentConfig(rotate_deg=10.0)
        assert rotate_only.per_sample_enabled and not rotate_only.batch_enabled
        cutmix_only = AugmentConfig(cutmix_prob=1.0)
        assert cutmix_only.batch_enabled and not cutmix_only.per_sample_enabled
        assert rotate_only.enabled and cutmix_only.enabled


class TestHeadConfig:
    """The head is selectable, and a setting the run would ignore is an error."""

    def test_the_head_defaults_to_the_shared_linear_one(self):
        assert ModelConfig().head == "linear"

    def test_a_query_head_option_on_a_linear_head_is_refused(self):
        with pytest.raises(ConfigError) as excinfo:
            loads_config("name: t\nmodel: {head_num_heads: 4}\n")
        assert "head_num_heads" in str(excinfo.value)

    def test_a_query_head_accepts_its_own_option(self):
        cfg = loads_config("name: t\nmodel: {head: query, head_num_heads: 4}\n")
        assert (cfg.model.head, cfg.model.head_num_heads) == ("query", 4)

    def test_an_unknown_head_is_refused(self):
        with pytest.raises(ConfigError):
            loads_config("name: t\nmodel: {head: transformer}\n")


class TestPerLabelGamma:
    """``gamma_neg`` may vary per label, because the derived table's rarity does."""

    def test_the_vector_is_the_scalar_when_nothing_is_overridden(self):
        assert LossConfig().gamma_neg_vector() == (4.0,) * len(LABELS)

    def test_an_override_lands_on_the_named_label_only(self):
        # Effusion is 58.8% positive in the derived table (measured, n=4,407), so
        # gamma_neg=4 focuses down the rarer class there -- the opposite of ASL's intent.
        cfg = LossConfig(gamma_neg_by_label={"Effusion": 0.0})
        vector = cfg.gamma_neg_vector()
        assert vector[LABELS.index("Effusion")] == 0.0
        assert all(v == 4.0 for i, v in enumerate(vector) if i != LABELS.index("Effusion"))

    def test_the_vector_is_in_labels_order_not_mapping_order(self):
        cfg = LossConfig(gamma_neg_by_label={"Fracture": 1.0, "ACL": 2.0})
        vector = cfg.gamma_neg_vector()
        assert vector[LABELS.index("ACL")] == 2.0
        assert vector[LABELS.index("Fracture")] == 1.0

    def test_the_apostrophe_label_survives_yaml(self):
        # "Baker's" is the name that breaks naive string handling; if the schema can
        # only be configured for eleven labels, the twelfth silently keeps the default.
        cfg = loads_config("name: t\nloss:\n  gamma_neg_by_label:\n    \"Baker's\": 1.5\n")
        assert cfg.loss.gamma_neg_vector()[LABELS.index("Baker's")] == 1.5

    def test_a_misspelled_label_is_refused_rather_than_ignored(self):
        with pytest.raises(ConfigError) as excinfo:
            loads_config("name: t\nloss: {gamma_neg_by_label: {Bakers: 1.0}}\n")
        assert "Bakers" in str(excinfo.value)

    def test_a_gamma_outside_the_scalar_range_is_refused(self):
        with pytest.raises(ConfigError):
            loads_config("name: t\nloss: {gamma_neg_by_label: {ACL: 99.0}}\n")

    def test_the_loss_is_selectable_for_the_asl_versus_bce_comparison(self):
        assert LossConfig().kind == "asl"
        assert loads_config("name: t\nloss: {kind: bce}\n").loss.kind == "bce"


class TestWeightAveraging:
    """EMA and last-N averaging are two answers to one question, so never both."""

    def test_both_default_to_off(self):
        assert OptimConfig().ema_decay == 0.0 and OptimConfig().swa_last_n == 0

    def test_asking_for_both_is_refused(self):
        with pytest.raises(ConfigError) as excinfo:
            loads_config("name: t\noptim: {ema_decay: 0.99, swa_last_n: 3}\n")
        assert "swa_last_n" in str(excinfo.value)

    def test_averaging_more_checkpoints_than_there_are_epochs_is_refused(self):
        with pytest.raises(ConfigError):
            loads_config("name: t\noptim: {swa_last_n: 11, epochs: 10}\n")

    def test_each_alone_is_accepted(self):
        assert loads_config("name: t\noptim: {ema_decay: 0.999}\n").optim.ema_decay == 0.999
        assert loads_config("name: t\noptim: {swa_last_n: 3}\n").optim.swa_last_n == 3


class TestEvalConfig:
    """TTA is label-safe by construction, and there is no flip in the legal set."""

    def test_tta_defaults_to_a_single_plain_pass(self):
        assert EvalConfig().tta == ()

    def test_no_flip_is_reachable_through_the_legal_set(self):
        assert not [name for name in TTA_TRANSFORMS if "flip" in name or "mirror" in name]

    def test_an_unknown_or_repeated_transform_is_refused(self):
        with pytest.raises(ConfigError):
            loads_config("name: t\neval: {tta: [hflip]}\n")
        with pytest.raises(ConfigError):
            loads_config("name: t\neval: {tta: [identity, identity]}\n")

    def test_the_legal_transforms_are_accepted(self):
        cfg = loads_config(f"name: t\neval: {{tta: {list(TTA_TRANSFORMS)}}}\n")
        assert cfg.eval.tta == TTA_TRANSFORMS


class TestNewSectionsReachTheHash:
    """A default-off knob still has to be part of the experiment's identity."""

    def test_turning_an_augmentation_on_changes_the_hash(self):
        base = _config()
        for section, update in (
            ("augment", {"rotate_deg": 10.0}),
            ("augment", {"cutmix_prob": 1.0}),
            ("model", {"head": "query"}),
            ("loss", {"kind": "bce"}),
            ("optim", {"ema_decay": 0.999}),
            ("eval", {"tta": ("identity", "rotate_cw")}),
        ):
            current = getattr(base, section)
            moved = base.model_copy(
                update={section: current.model_copy(update=update)}
            )
            assert config_hash(moved) != config_hash(base), (section, update)


class TestTheLogStatesWhatTheRunDoes:
    """``describe`` is the only record in the Slurm log; a silent knob is a lost run."""

    def test_an_enabled_augmentation_appears_in_the_description(self):
        text = describe(loads_config("name: t\naugment: {rotate_deg: 10, cutmix_prob: 1.0}\n"))
        assert "rotate" in text and "cutmix" in text

    def test_a_run_with_no_augmentation_says_so_rather_than_omitting_the_line(self):
        # An absent line is ambiguous between "off" and "this build predates the field".
        text = describe(_config())
        assert "augment    : off" in text
        assert "averaging  : off" in text
        assert "tta        : off" in text

    def test_the_head_the_ema_and_the_tta_all_reach_the_description(self):
        text = describe(
            loads_config(
                "name: t\nmodel: {head: query, head_num_heads: 4}\n"
                "optim: {ema_decay: 0.999}\neval: {tta: [identity, rotate_cw]}\n"
            )
        )
        assert "query" in text and "0.999" in text and "rotate_cw" in text

    def test_the_loss_line_names_the_loss_that_will_run(self):
        assert "ASL" in describe(_config())
        assert "BCE" in describe(loads_config("name: t\nloss: {kind: bce}\n"))

    def test_the_loss_line_does_not_advertise_parameters_bce_ignores(self):
        # Printing gamma_neg next to BCE describes an asymmetry that never applies.
        text = describe(loads_config("name: t\nloss: {kind: bce}\n"))
        assert "gamma_neg" not in text and "clip" not in text

    def test_a_per_label_gamma_is_named_in_the_log_not_hidden_in_the_hash(self):
        text = describe(loads_config("name: t\nloss: {gamma_neg_by_label: {Effusion: 0.0}}\n"))
        assert "Effusion=0.0" in text


class TestAslKnobsRequireAsl:
    """A focusing exponent set next to ``kind: bce`` documents an experiment that will not run."""

    @pytest.mark.parametrize(
        "knob", ["gamma_neg: 2.0", "gamma_pos: 1.0", "clip: 0.1", "gamma_neg_by_label: {ACL: 1.0}"]
    )
    def test_an_asl_knob_on_a_bce_run_is_refused(self, knob):
        with pytest.raises(ConfigError) as excinfo:
            loads_config(f"name: t\nloss: {{kind: bce, {knob}}}\n")
        assert "asl" in str(excinfo.value).lower()

    def test_bce_alone_is_accepted(self):
        assert loads_config("name: t\nloss: {kind: bce}\n").loss.kind == "bce"

    def test_the_same_knob_on_an_asl_run_is_accepted(self):
        assert loads_config("name: t\nloss: {kind: asl, gamma_neg: 2.0}\n").loss.gamma_neg == 2.0


class TestThePoolSection:
    """A section, so it moved no hash -- and a knob, so it still identifies its arm.

    Both halves are load-bearing and they pull in opposite directions. ``pool`` had to
    be a default-dropped SECTION because a field on ``ModelConfig`` would have
    re-hashed all twenty-six configs on disk, as ``optim.backbone_lr_mult`` and
    ``model.freeze_backbone_bn`` each did; and a section that were dropped
    unconditionally would let a per-label run and the baseline share one digest, one
    run name and one set of OOF rows.
    """

    def test_both_flags_default_to_off(self):
        # The promise `_DEFAULTED_SECTIONS` is: this section's default is a no-op.
        assert PoolConfig().per_label_slots is False
        assert PoolConfig().head_residual is False
        assert loads_config("name: t\n").pool == PoolConfig()

    def test_a_defaulted_pool_section_is_dropped_before_hashing(self):
        assert "pool" not in json.loads(canonical_dump(loads_config("name: t\n")))

    def test_writing_the_default_out_explicitly_changes_nothing(self):
        # A file that spells the section out at its default must hash as one that omits
        # it, or a digest would depend on whether someone felt like being explicit.
        implicit = loads_config("name: t\n")
        explicit = loads_config(
            "name: t\npool:\n  per_label_slots: false\n  head_residual: false\n"
        )
        assert config_hash(implicit) == config_hash(explicit)

    def test_a_non_default_pool_section_is_hashed(self):
        base = loads_config("name: t\n")
        arm = loads_config("name: t\npool:\n  per_label_slots: true\n")
        assert "pool" in json.loads(canonical_dump(arm))
        assert config_hash(arm) != config_hash(base)

    def test_the_two_flags_are_two_different_experiments(self):
        # Not one "pooling changed" bit. A per-label run and a residual run differ in
        # what they are evidence about, so they must not collide under one digest.
        per_label = loads_config("name: t\npool:\n  per_label_slots: true\n")
        residual = loads_config(
            "name: t\nmodel: {head: query}\npool:\n  head_residual: true\n"
        )
        both = loads_config(
            "name: t\nmodel: {head: query}\n"
            "pool:\n  per_label_slots: true\n  head_residual: true\n"
        )
        digests = {config_hash(c) for c in (per_label, residual, both)}
        assert len(digests) == 3

    def test_the_head_residual_on_a_linear_head_is_refused(self):
        # The same refusal as `model.head_num_heads` on a linear head, one level up:
        # the linear head reads the pooled study vector and nothing else, so "add it
        # back in" is a setting the run would ignore. A config that claims one is
        # filed as an experiment and runs as the baseline.
        with pytest.raises(ConfigError) as excinfo:
            loads_config("name: t\npool:\n  head_residual: true\n")
        assert "head_residual" in str(excinfo.value)
        assert "query" in str(excinfo.value)

    def test_the_head_residual_is_accepted_on_the_query_head(self):
        cfg = loads_config("name: t\nmodel: {head: query}\npool: {head_residual: true}\n")
        assert cfg.pool.head_residual is True

    def test_per_label_slots_is_legal_on_either_head(self):
        # Unlike the residual: the per-label vector has a readout under both heads --
        # a row-wise dot against the linear head's own weight rows, or a residual into
        # LabelQueryHead -- so neither head would ignore it.
        for head in ("linear", "query"):
            cfg = loads_config(
                f"name: t\nmodel: {{head: {head}}}\npool: {{per_label_slots: true}}\n"
            )
            assert cfg.pool.per_label_slots is True

    def test_an_unknown_pool_key_is_refused_like_every_other_section(self):
        with pytest.raises(ConfigError, match="per_label_slot"):
            loads_config("name: t\npool:\n  per_label_slot: true\n")

    def test_the_run_log_states_which_pooling_ran_even_at_the_default(self):
        # `pool` is dropped from the hash at its default, so the log line is the only
        # place a reader learns that all twelve labels shared one set of plane weights.
        assert "shared pool" in describe(loads_config("name: t\n"))
        assert "per-label" in describe(loads_config("name: t\npool: {per_label_slots: true}\n"))
        assert "query head" in describe(
            loads_config("name: t\nmodel: {head: query}\npool: {head_residual: true}\n")
        )
