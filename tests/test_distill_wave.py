"""The teacher/student machinery, checked at the five places it can fail quietly.

Every failure guarded here produces a NUMBER. That is the whole reason the file exists:
a distillation run that is wired wrongly does not crash, it converges, prints a loss
curve, writes an OOF table with the right schema, and reports a macro AUC that somebody
then puts in a table.

The five:

1. **Fold matching.** A teacher trained on fold 0 saw fold 1's validation studies in its
   own training set. Distilling fold 1 from a fold-0 teacher leaks validation labels
   through the soft targets and INFLATES the score -- the direction that looks like
   success. Nothing about the resulting run looks wrong.
2. **Residue width and vocabulary.** The residue table is deliberately the same 156
   columns in the same order as the full table, so a repointed path is accepted by every
   loader and only the VALUES differ. tests/test_residue_features.py guards the values;
   this file guards the width and the name list the checkpoint carries.
3. **Checkpoint roles.** A teacher reads report text and can never be submitted, because
   test studies carry no report (constraint A1).
4. **The inference boundary.** The submission path must refuse a teacher file outright
   rather than shape-checking its way into a confusing error.
5. **Teacher freezing.** A teacher that keeps training is a second model being fitted,
   and the student's gain stops being attributable to fixed knowledge.

Plus the arithmetic of the wave itself: 5 configs x 5 folds x seed 0 = 25 cells, each a
distinct (config, fold, seed) identity.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("timm")

import torch  # noqa: E402

from kappa.config import config_hash, load_config  # noqa: E402
from kappa.constants import N_LABELS  # noqa: E402
from kappa.inference.predict import load_checkpoint, save_inference_checkpoint  # noqa: E402
from kappa.labels.features import REPORT_FEATURE_DIM, REPORT_FEATURE_NAMES  # noqa: E402
from kappa.models.encoder import N_SLOTS  # noqa: E402
from kappa.models.teacher import TeacherClassifier, TeacherOutput  # noqa: E402
from kappa.training.distill import BinaryKLDistillation, DistillationLoss  # noqa: E402
from kappa.training.loop import (  # noqa: E402
    load_frozen_teacher,
    resolve_teacher_ckpt,
    save_teacher_checkpoint,
)

REPO = Path(__file__).resolve().parents[1]

#: The same tiny real timm model tests/test_models.py uses: 64-dim features, no
#: BatchNorm, and a resolved mean/std that is deliberately NOT ImageNet.
TINY = "test_convnext"
SIZE = 32
N_SLICES = 4

#: The five teacher/student arms, in reading order.
W6_ARMS = (
    "w6_e2_control",
    "w6_e1_teacher",
    "w6_e3_kd_all",
    "w6_e4_kd_labelled",
    "w6_e5_kd_ablate",
)
W6_STUDENTS = ("w6_e3_kd_all", "w6_e4_kd_labelled", "w6_e5_kd_ablate")
RESIDUE = "data/interim/labels/report_features_residue.parquet"


def _teacher(**over) -> TeacherClassifier:
    kwargs = dict(
        report_dim=REPORT_FEATURE_DIM,
        backbone=TINY,
        n_slices=N_SLICES,
        pretrained=False,
        drop_path_rate=0.0,
        report_hidden=8,
        report_dropout=0.0,
        img_size=SIZE,
    )
    kwargs.update(over)
    return TeacherClassifier(**kwargs)


def _batch(batch: int = 2, *, report_dim: int = REPORT_FEATURE_DIM):
    generator = torch.Generator().manual_seed(0)
    pixels = torch.randn(batch, N_SLOTS, N_SLICES, 3, SIZE, SIZE, generator=generator)
    slots = torch.ones(batch, N_SLOTS, dtype=torch.bool)
    slot_ids = torch.arange(N_SLOTS, dtype=torch.long).expand(batch, N_SLOTS).contiguous()
    report = torch.rand(batch, report_dim, generator=generator)
    flag = torch.ones(batch, dtype=torch.bool)
    return pixels, slots, slot_ids, report, flag


def _load(name: str):
    return load_config(REPO / f"configs/{name}.yaml")


def _saved(tmp_path: Path, *, fold: int = 0, seed: int = 0, **over) -> Path:
    path = tmp_path / f"fold{fold}_seed{seed}.pt"
    save_teacher_checkpoint(path, _teacher(**over), meta={"fold": fold, "seed": seed})
    return path


# --------------------------------------------------------------------------- 1. folds


class TestTheTeacherFoldMustMatchTheStudentFold:
    def test_a_mismatched_fold_is_refused(self, tmp_path):
        # The failure this prevents INFLATES the score, so it can never be a warning: a
        # fold-0 teacher has fold 1's validation studies in its training set, and its
        # soft targets carry their labels.
        with pytest.raises(ValueError, match="fold"):
            load_frozen_teacher(_saved(tmp_path, fold=0), fold=1)

    def test_the_matching_fold_loads(self, tmp_path):
        loaded = load_frozen_teacher(_saved(tmp_path, fold=3), fold=3)
        assert isinstance(loaded, TeacherClassifier)

    @pytest.mark.parametrize("fold", [0, 1, 2, 3, 4])
    def test_every_fold_resolves_to_its_own_file(self, fold):
        template = "data/runs/w6_e1_teacher/fold{fold}_seed{seed}.pt"
        assert resolve_teacher_ckpt(template, fold=fold, seed=0) == Path(
            f"data/runs/w6_e1_teacher/fold{fold}_seed0.pt"
        )

    def test_the_five_folds_resolve_to_five_distinct_files(self):
        template = "data/runs/w6_e1_teacher/fold{fold}_seed{seed}.pt"
        assert len({resolve_teacher_ckpt(template, fold=k, seed=0) for k in range(5)}) == 5

    def test_the_seed_is_substituted_too_not_just_the_fold(self):
        # Both placeholders, because a seed-1 student distilling from a seed-0 teacher
        # is a different experiment that would be filed as the same one.
        template = "data/runs/w6_e1_teacher/fold{fold}_seed{seed}.pt"
        assert resolve_teacher_ckpt(template, fold=2, seed=1) == Path(
            "data/runs/w6_e1_teacher/fold2_seed1.pt"
        )

    def test_every_student_config_carries_the_fold_placeholder(self):
        # A hardcoded fold in the template would make all five array elements distil
        # from ONE teacher -- four of them leaking. The placeholder is what makes a
        # five-element array correct by default.
        for name in W6_STUDENTS:
            template = str(_load(name).distill.teacher_ckpt)
            assert "{fold}" in template, f"{name} would distil every fold from one teacher"
            assert "{seed}" in template, name
            assert "w6_e1_teacher" in template, name


# ------------------------------------------------------------------ 2. residue width


class TestTheResidueWidthAndVocabularyAreTheContract:
    def test_the_teacher_refuses_a_report_vector_of_the_wrong_width(self):
        model = _teacher().eval()
        pixels, slots, slot_ids, _, flag = _batch()
        with pytest.raises(ValueError):
            model(pixels, slots, slot_ids, torch.rand(2, REPORT_FEATURE_DIM - 1), flag)

    def test_the_checkpoint_records_the_vocabulary_it_was_trained_on(self, tmp_path):
        payload = torch.load(_saved(tmp_path), map_location="cpu", weights_only=True)
        assert payload["report_features"] == list(REPORT_FEATURE_NAMES)
        assert len(payload["report_features"]) == REPORT_FEATURE_DIM == 156

    def test_a_checkpoint_from_a_different_vocabulary_is_refused(self, tmp_path):
        path = _saved(tmp_path)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        # Permuted, not truncated: same width, same names, different order. This is the
        # failure that trains, converges and reports a number, and it is the reason the
        # residue table keeps the full 156 columns instead of narrowing to 58.
        payload["report_features"] = list(reversed(REPORT_FEATURE_NAMES))
        torch.save(payload, path)
        with pytest.raises(ValueError):
            load_frozen_teacher(path, fold=0)

    def test_the_teacher_arm_reads_the_residue_and_not_the_full_table(self):
        assert str(_load("w6_e1_teacher").distill.report_features) == RESIDUE

    def test_every_student_arm_reads_the_residue_too(self):
        # The frozen teacher still reads its input at student-training time. Pointing a
        # student at the FULL table would hand that frozen teacher a lookup table for
        # the labels -- the same leak in a different costume, and on the arm whose
        # number actually gets reported.
        for name in W6_STUDENTS:
            assert str(_load(name).distill.report_features) == RESIDUE, name


# ---------------------------------------------------------------- 3. checkpoint roles


class TestATeacherFileIsADifferentKindOfFile:
    def test_a_teacher_checkpoint_is_marked(self, tmp_path):
        payload = torch.load(_saved(tmp_path), map_location="cpu", weights_only=True)
        assert payload["kappa_teacher"] is True

    def test_a_teacher_checkpoint_carries_the_arch_that_rebuilds_it(self, tmp_path):
        payload = torch.load(_saved(tmp_path), map_location="cpu", weights_only=True)
        assert payload["arch"]["report_dim"] == REPORT_FEATURE_DIM
        # `pretrained` is deliberately absent: a reload takes every weight from the
        # file, and recording it would invite asking a hub for weights already on disk.
        assert "pretrained" not in payload["arch"]

    def test_a_student_checkpoint_is_not_marked_as_a_teacher(self, tmp_path):
        from kappa.data.dataset import CropSpec, Normalisation
        from kappa.models.study_classifier import StudyClassifier

        model = StudyClassifier(
            backbone=TINY, n_slices=N_SLICES, pretrained=False, drop_path_rate=0.0, aux_dim=0
        )
        path = tmp_path / "student.pt"
        save_inference_checkpoint(
            path,
            model,
            normalisation=Normalisation.from_data_config(model.data_config),
            crop=CropSpec(fov_mm=150.0, out_px=SIZE),
            slots=tuple(range(N_SLOTS)),
            n_slices=N_SLICES,
            meta={"fold": 0, "seed": 0, "role": "student"},
        )
        payload = torch.load(path, map_location="cpu", weights_only=True)
        # There is no way for a student file to accidentally acquire the marker:
        # `save_inference_checkpoint` never writes the key at all.
        assert "kappa_teacher" not in payload

    def test_the_role_is_recorded_in_the_artifact_not_only_in_the_config(self, tmp_path):
        path = tmp_path / "fold0_seed0.pt"
        save_teacher_checkpoint(
            path, _teacher(), meta={"fold": 0, "seed": 0, "role": "teacher"}
        )
        payload = torch.load(path, map_location="cpu", weights_only=True)
        # `--select` is the standing lesson: a policy that is not in the artifact is a
        # policy nothing downstream can tell apart.
        assert payload["meta"]["role"] == "teacher"


# ------------------------------------------------------------- 4. inference boundary


class TestATeacherCanNeverReachInference:
    def test_the_submission_loader_refuses_a_teacher_file(self, tmp_path):
        # Refused on the MARKER, before any version or shape check, so the error names
        # the real problem instead of surfacing as a state_dict mismatch.
        with pytest.raises(ValueError, match="teacher"):
            load_checkpoint(_saved(tmp_path))

    def test_the_inference_module_does_not_import_the_teacher(self):
        # The boundary is structural, not conventional: if `kappa.inference.predict`
        # imported the teacher, report text would be one import away from the path that
        # ships. tests/test_loop.py::TestTheInferenceBoundary holds the other end.
        source = (REPO / "src/kappa/inference/predict.py").read_text()
        assert "models.teacher" not in source
        assert "report_dataset" not in source

    def test_the_teacher_is_not_exported_from_the_models_package(self):
        import kappa.models as models

        assert not hasattr(models, "TeacherClassifier")

    def test_exactly_one_arm_of_the_wave_writes_a_teacher_file(self):
        roles = {name: _load(name).distill.role for name in W6_ARMS}
        assert [n for n, r in roles.items() if r == "teacher"] == ["w6_e1_teacher"]
        assert sum(1 for r in roles.values() if r == "student") == 4


# ---------------------------------------------------------------- 5. teacher freezing


class TestTheTeacherIsFrozen:
    def test_it_loads_in_eval_mode_with_gradients_off(self, tmp_path):
        teacher = load_frozen_teacher(_saved(tmp_path), fold=0)
        assert not teacher.training, "a teacher in train mode would move its BN/dropout"
        assert not any(p.requires_grad for p in teacher.parameters())

    def test_its_soft_targets_carry_no_gradient(self, tmp_path):
        teacher = load_frozen_teacher(_saved(tmp_path), fold=0)
        with torch.no_grad():
            out = teacher(*_batch())
        assert isinstance(out, TeacherOutput)
        assert not out.logits.requires_grad

    def test_two_forwards_of_a_frozen_teacher_agree(self, tmp_path):
        # Eval mode is what makes this true. In train mode the report branch's dropout
        # would make the soft targets differ between the two passes, and the student
        # would be chasing a moving target that still looks like a frozen teacher.
        teacher = load_frozen_teacher(_saved(tmp_path, report_dropout=0.5), fold=0)
        args = _batch()
        with torch.no_grad():
            assert torch.allclose(teacher(*args).logits, teacher(*args).logits)


# ------------------------------------------------------- the KD term and the ablation


class TestTheDistillationTerm:
    def test_the_temperature_is_applied_rather_than_ignored(self):
        # Without the T*T factor the gradient magnitude moves ~56x across T=1..8, so
        # sweeping the temperature would silently rescale the effective learning rate
        # and every temperature arm would also be a learning-rate arm.
        student = torch.zeros(4, N_LABELS)
        teacher = torch.full((4, N_LABELS), 2.0)
        at_one = BinaryKLDistillation(temperature=1.0)(student, teacher)
        at_two = BinaryKLDistillation(temperature=2.0)(student, teacher)
        assert at_one.item() > 0.0 and at_two.item() > 0.0
        assert not torch.isclose(at_one, at_two)

    def test_each_label_is_independent_of_the_other_eleven(self):
        # The twelve columns CO-OCCUR -- an effusion and a meniscal tear are not
        # competing hypotheses, and a knee with both ACL and MCL findings is ordinary.
        # A softmax over the label axis answers "WHICH ONE of the twelve is it", which
        # is not this problem: it would couple every column's loss to the other eleven,
        # so moving the teacher's confidence on label 3 would silently change the
        # gradient on label 0.
        #
        # Asserted behaviourally rather than by grepping for "softmax", because the
        # module's docstring quotes the softmax formulation in order to reject it -- a
        # source-text check passes or fails on the commentary rather than on the code.
        isolate = torch.zeros(2, N_LABELS)
        isolate[:, 0] = 1.0
        student = torch.zeros(2, N_LABELS)
        base = torch.zeros(2, N_LABELS)
        base[:, 0] = 2.0
        moved = base.clone()
        moved[:, 1:] = 5.0  # every OTHER column's teacher logit changes
        loss = BinaryKLDistillation(temperature=2.0)
        assert torch.allclose(
            loss(student, base, isolate), loss(student, moved, isolate)
        ), "label 0's KD term moved when only labels 1-11 changed: the axis is coupled"

    def test_identical_logits_give_zero_divergence(self):
        logits = torch.randn(4, N_LABELS, generator=torch.Generator().manual_seed(1))
        loss = BinaryKLDistillation(temperature=2.0)(logits.clone(), logits.clone())
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_a_nonfinite_teacher_logit_is_refused(self):
        # `0.0 * nan` is `nan`, so a masked-out non-finite cell still poisons the whole
        # batch. The guard is therefore unconditional over the tensor, not masked.
        teacher = torch.zeros(2, N_LABELS)
        teacher[0, 0] = float("nan")
        with pytest.raises(ValueError):
            BinaryKLDistillation(temperature=2.0)(torch.zeros(2, N_LABELS), teacher)

    def test_unlabelled_distillation_sees_more_cells_than_labelled_only(self):
        # The arithmetic behind arms 3 and 4. 98.7% of raw label cells are unknown, so
        # restricting KD to the labelled ones discards almost all of the teacher's
        # opinion -- which is exactly what e4 measures.
        from kappa.training.losses import MaskedBCE

        student = torch.randn(8, N_LABELS, generator=torch.Generator().manual_seed(2))
        teacher = torch.randn(8, N_LABELS, generator=torch.Generator().manual_seed(3))
        targets = torch.zeros(8, N_LABELS)
        mask = torch.zeros(8, N_LABELS)
        mask[0, 0] = 1.0
        wide = DistillationLoss(MaskedBCE(), temperature=2.0, distill_unlabelled=True)
        narrow = DistillationLoss(MaskedBCE(), temperature=2.0, distill_unlabelled=False)
        assert not torch.isclose(
            wide(student, targets, mask, teacher).distillation,
            narrow(student, targets, mask, teacher).distillation,
        )

    def test_no_teacher_means_a_zero_kd_term_still_attached_to_the_graph(self):
        from kappa.training.losses import MaskedBCE

        student = torch.randn(4, N_LABELS, requires_grad=True)
        out = DistillationLoss(MaskedBCE(), temperature=2.0)(
            student, torch.zeros(4, N_LABELS), torch.ones(4, N_LABELS), None
        )
        assert out.distillation.item() == pytest.approx(0.0)
        # Attached, not a bare Python 0.0: the control path must run the same backward
        # shape as the treatment path or the two are not comparable.
        assert out.total.requires_grad


class TestTheResidueAblationRemovesTheResidueAndNothingElse:
    def test_clearing_the_presence_flag_changes_the_fused_logits(self, tmp_path):
        teacher = load_frozen_teacher(_saved(tmp_path), fold=0)
        pixels, slots, slot_ids, report, _ = _batch()
        with torch.no_grad():
            on = teacher(pixels, slots, slot_ids, report, torch.ones(2, dtype=torch.bool))
            off = teacher(pixels, slots, slot_ids, report, torch.zeros(2, dtype=torch.bool))
        # If these agreed, the ablation arm would be the full-KD arm under another name
        # and the wave would report two measurements of one thing.
        assert not torch.allclose(on.logits, off.logits)

    def test_the_gate_is_on_the_flag_not_on_the_feature_tensor(self, tmp_path):
        # Zeroing `report` instead of the flag pushes a genuine all-zero vector through
        # the report MLP and picks up its BIASES, so the branch still contributes. The
        # two are different ablations and only one of them is the one the config names.
        teacher = load_frozen_teacher(_saved(tmp_path), fold=0)
        pixels, slots, slot_ids, report, _ = _batch()
        with torch.no_grad():
            flag_off = teacher(
                pixels, slots, slot_ids, report, torch.zeros(2, dtype=torch.bool)
            ).logits
            tensor_zeroed = teacher(
                pixels,
                slots,
                slot_ids,
                torch.zeros_like(report),
                torch.ones(2, dtype=torch.bool),
            ).logits
        assert not torch.allclose(flag_off, tensor_zeroed)

    def test_the_image_branch_is_untouched_by_the_gate(self, tmp_path):
        teacher = load_frozen_teacher(_saved(tmp_path), fold=0)
        pixels, slots, slot_ids, report, _ = _batch()
        with torch.no_grad():
            on = teacher(pixels, slots, slot_ids, report, torch.ones(2, dtype=torch.bool))
            off = teacher(pixels, slots, slot_ids, report, torch.zeros(2, dtype=torch.bool))
        # Same pixels, same weights: the residue can only enter through the fusion, so
        # the ablation must leave the image branch bit-identical.
        assert torch.allclose(on.image_logits, off.image_logits)

    def test_only_the_ablation_arm_sets_the_flag(self):
        assert {name: _load(name).distill.teacher_report_ablate for name in W6_ARMS} == {
            "w6_e2_control": False,
            "w6_e1_teacher": False,
            "w6_e3_kd_all": False,
            "w6_e4_kd_labelled": False,
            "w6_e5_kd_ablate": True,
        }

    def test_the_ablation_differs_from_the_treatment_in_exactly_that_field(self):
        treatment = _load("w6_e3_kd_all").model_dump(mode="json")
        ablation = _load("w6_e5_kd_ablate").model_dump(mode="json")
        differing = {
            key for key in set(treatment) | set(ablation)
            if treatment.get(key) != ablation.get(key)
        }
        assert differing == {"name", "distill"}
        moved = {
            key for key in treatment["distill"]
            if treatment["distill"][key] != ablation["distill"][key]
        }
        assert moved == {"teacher_report_ablate"}

    def test_the_flag_is_refused_where_it_would_do_nothing(self):
        from kappa.config import loads_config

        # Both of these would hash as the ablation and train as something else, which
        # is the `--select` failure mode with a different field.
        with pytest.raises(Exception):
            loads_config("name: t\ndistill:\n  teacher_report_ablate: true\n")
        with pytest.raises(Exception):
            loads_config(
                "name: t\ndistill:\n  role: teacher\n"
                "  report_features: x.parquet\n  teacher_report_ablate: true\n"
            )


# ------------------------------------------------------------------- the wave's shape


class TestTheWaveIsTwentyFiveCells:
    def test_five_arms_five_folds_one_seed(self):
        cells = {(name, fold, 0) for name in W6_ARMS for fold in range(5)}
        assert len(cells) == 25
        assert len({name for name, _, _ in cells}) == 5
        assert {fold for _, fold, _ in cells} == {0, 1, 2, 3, 4}
        assert {seed for _, _, seed in cells} == {0}

    def test_every_arm_declares_its_own_name_and_so_its_own_directory(self):
        # `name` is excluded from config_hash but IS the output path, so two arms
        # sharing one are ONE directory and silently overwrite each other's weights and
        # OOF rows. That has already cost a run in this project.
        for name in W6_ARMS:
            assert _load(name).name == name
        assert len({(_load(n).out_dir, _load(n).name) for n in W6_ARMS}) == 5

    def test_the_control_hashes_as_the_measured_base_and_the_treatments_do_not(self):
        base = config_hash(load_config(REPO / "configs/b4_aug_query_h4.yaml"))
        assert config_hash(_load("w6_e2_control")) == base
        for name in W6_ARMS:
            if name != "w6_e2_control":
                assert config_hash(_load(name)) != base, name

    def test_the_five_arms_hash_distinctly_from_each_other(self):
        digests = {name: config_hash(_load(name)) for name in W6_ARMS}
        assert len(set(digests.values())) == 5, digests

    def test_every_arm_shares_the_base_recipe_outside_the_distill_section(self):
        # The wave is readable only if the five arms differ in `distill` and `name` and
        # nowhere else. A stray edit to one arm's `optim` or `augment` would turn the
        # comparison into two variables and nothing would raise.
        control = _load("w6_e2_control").model_dump(mode="json")
        for name in W6_ARMS:
            arm = _load(name).model_dump(mode="json")
            differing = {
                key for key in set(control) | set(arm) if control.get(key) != arm.get(key)
            }
            assert differing <= {"name", "distill"}, f"{name} also moved {differing}"

    @pytest.mark.parametrize("name", W6_ARMS)
    def test_every_arm_runs_the_same_budget_and_the_same_architecture(self, name):
        cfg = _load(name)
        # `optim.epochs` IS in config_hash, and under onecycle the metric peak tracks
        # SCHEDULE LENGTH -- so a budget difference anywhere in the wave would price the
        # schedule rather than the distillation.
        assert cfg.optim.epochs == 10
        assert cfg.optim.schedule == "onecycle"
        assert cfg.optim.lr == pytest.approx(5.0e-5)
        assert cfg.optim.backbone_lr_mult == pytest.approx(1.0)
        assert cfg.optim.amp_dtype == "bf16"
        assert cfg.model.backbone == "convnext_small"
        assert cfg.model.head == "query" and cfg.model.head_num_heads == 4
        assert cfg.pool.head_residual is True
        assert cfg.augment.rotate_deg == pytest.approx(10.0)
        assert tuple(cfg.data.slots) == ("Axial_fs1", "Coronal_fs1", "Sagittal_fs1")
        assert cfg.seed == 0

    @pytest.mark.parametrize("name", W6_STUDENTS)
    def test_every_student_uses_the_declared_kd_hyperparameters(self, name):
        cfg = _load(name)
        assert cfg.distill.temperature == pytest.approx(2.0)
        assert cfg.distill.lambda_supervised == pytest.approx(1.0)
        assert cfg.distill.lambda_distill == pytest.approx(1.0)
        assert cfg.distill.distilling is True

    def test_the_cutmix_guard_holds_for_every_report_reading_arm(self):
        # `_report_text_cannot_be_cutmixed` refuses it, and the reason is worth stating:
        # cutmix blends two studies' PIXELS, and there is no meaningful way to blend two
        # studies' reports, so the teacher would read study A's report over a mixture.
        for name in W6_ARMS:
            assert _load(name).augment.cutmix_prob == pytest.approx(0.0), name
