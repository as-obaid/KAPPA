#!/usr/bin/env python3
"""Prove the teacher/student pipeline runs, without the competition dataset.

Argparse plus calls into the library, per the repo contract. This is a **wiring
check, not an experiment.** It trains on ~12 synthetic studies from random
initialisation for one epoch, so the macro figures it prints are noise and must
never be quoted, compared, or logged as results.

    python scripts/smoke_distill.py --root /tmp/kappa-smoke

What it establishes, in one pass, is that these six things hold together:

1. the teacher trains on image + report features and writes a checkpoint
2. that checkpoint rebuilds itself, frozen, at its own report width
3. the student trains image-only against the teacher's soft targets
4. the no-teacher control takes the same code path with the KD term at zero
5. the student's checkpoint scores through the image-only inference path
6. the two guards fire: inference refuses a teacher, and a fold-N teacher is
   refused for fold M

Run it after any change to the model, the loss, the config schema or the loop.
It costs seconds on a laptop and it is the only check that crosses all four.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kappa.config import TrainConfig  # noqa: E402
from kappa.constants import ID_COLUMN  # noqa: E402
from kappa.data.folds import build_folds  # noqa: E402
from kappa.data.synthetic import make_synthetic_dataset  # noqa: E402
from kappa.inference.predict import load_checkpoint, predict_studies  # noqa: E402
from kappa.labels.features import REPORT_FEATURE_DIM, write_features  # noqa: E402
from kappa.training.loop import DataBundle, load_frozen_teacher, train_fold  # noqa: E402

#: Anatomy strings that map onto real clusters, so the feature vector is not all
#: zeros. Fed through the production featuriser rather than a stand-in.
ANATOMY = (
    "anterior cruciate ligament",
    "medial meniscus",
    "joint effusion",
    "medial collateral ligament",
)


def synthetic_records(study_uids) -> list[dict]:
    """Extraction-record-shaped dicts, one per study, with a positive and a negative."""
    return [
        {
            "_study_uid": uid,
            "findings": [
                {
                    "anatomy": ANATOMY[i % len(ANATOMY)],
                    "assertion": "present",
                    "description": "tear with surrounding oedema",
                },
                {
                    "anatomy": ANATOMY[(i + 1) % len(ANATOMY)],
                    "assertion": "absent",
                    "description": "intact, no tear",
                },
            ],
        }
        for i, uid in enumerate(study_uids)
    ]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--root",
        type=Path,
        default=None,
        help="where to write the synthetic corpus and the checkpoints "
        "(default: a temporary directory, left in place for inspection)",
    )
    ap.add_argument("--studies", type=int, default=12, help="synthetic studies (default: 12)")
    ap.add_argument("--folds", type=int, default=2, help="fold count (default: 2)")
    args = ap.parse_args()

    root = args.root or Path(tempfile.mkdtemp(prefix="kappa-smoke-"))
    root.mkdir(parents=True, exist_ok=True)
    # The synthetic corpus warns that its studies carry phantom geometry. That is
    # the point of a phantom, and the warning would otherwise bury the output.
    warnings.filterwarnings("ignore")

    print(f"root       : {root}")
    corpus = make_synthetic_dataset(
        root / "corpus",
        n_studies=args.studies,
        seed=0,
        patterns=((1,),),
        cache_size=32,
        slice_counts=(4, 6),
        # mask_rate=0.0 and prevalence=0.5: at the measured 98.7% unknown rate a
        # 12-study fold would carry no positives and every label would be skipped,
        # so the run would assert nothing. The sparsity is real and the loss is
        # checked against it elsewhere; this script checks the wiring.
        mask_rate=0.0,
        prevalence=0.5,
    )
    study_table = corpus.labels.merge(
        corpus.manifest.drop_duplicates("study_uid")[
            ["study_uid", "patient_id", "scanner_fingerprint"]
        ].rename(columns={"study_uid": ID_COLUMN}),
        on=ID_COLUMN,
        how="left",
    )
    # No report text in the synthetic corpus, so each study is its own group.
    study_table["report_group"] = study_table[ID_COLUMN]
    bundle = DataBundle(
        manifest=corpus.manifest,
        labels=corpus.labels,
        folds=build_folds(study_table, n_folds=args.folds, strata_key=None, seed=0),
        cache_dir=corpus.cache_dir,
    )

    features = write_features(
        synthetic_records(corpus.study_uids), root / "report_features.parquet"
    )
    print(f"features   : {args.studies} studies x {REPORT_FEATURE_DIM} -> {features.name}")

    # Tiny, CPU, random init, one epoch. `pretrained=False` matters: this has to
    # run with no network, the same as a compute node.
    base = dict(
        seed=0,
        out_dir=root / "runs",
        data={
            "slots": ("Axial_fs1",),
            "n_slices": 2,
            "size": 64,
            "physical_crop_mm": 100.0,
            "num_workers": 0,
            "pin_memory": False,
        },
        model={"backbone": "resnet18", "pretrained": False, "drop_path_rate": 0.0},
        optim={"epochs": 1, "batch_size": 2, "grad_accum": 1, "amp": False, "lr": 1e-4},
        folds={"n_folds": args.folds},
    )

    print("\n1. teacher: image + report -> checkpoint")
    teacher_cfg = TrainConfig(
        name="smoke_teacher",
        **base,
        distill={
            "role": "teacher",
            "report_features": features,
            # Deliberately not the 256 default. A teacher rebuilt from the
            # student's config would come back at 256 and fail to load, so this
            # is what pins the checkpoint as self-describing.
            "report_hidden": 32,
            "report_drop_rate": 0.25,
        },
    )
    teacher = train_fold(teacher_cfg, bundle, 0, save_checkpoint=True, verbose=False)
    print(f"   macro {teacher.auc.macro:.4f} (noise)  -> {Path(teacher.checkpoint).name}")

    print("\n2. student: image only, distilling from the frozen teacher")
    student_cfg = TrainConfig(
        name="smoke_student",
        **base,
        distill={
            "role": "student",
            "teacher_ckpt": teacher.checkpoint,
            "report_features": features,
            "temperature": 2.0,
            "lambda_supervised": 1.0,
            "lambda_distill": 1.0,
        },
    )
    student = train_fold(student_cfg, bundle, 0, save_checkpoint=True, verbose=False)
    print(f"   macro {student.auc.macro:.4f} (noise)  -> {Path(student.checkpoint).name}")

    print("\n3. control: same code path, no teacher, KD term absent")
    control = train_fold(
        TrainConfig(name="smoke_control", **base), bundle, 0, save_checkpoint=False, verbose=False
    )
    print(f"   macro {control.auc.macro:.4f} (noise)")

    print("\n4. image-only inference from the student checkpoint")
    loaded = load_checkpoint(student.checkpoint, device="cpu")
    prediction = predict_studies(
        loaded.model,
        loaded.dataset(corpus.manifest, corpus.cache_dir),
        device="cpu",
        batch_size=2,
    )
    print(f"   {prediction.scores.shape} logits, {len(prediction.study_uids)} studies, no report")

    print("\n5. inference must refuse a teacher checkpoint")
    try:
        load_checkpoint(teacher.checkpoint, device="cpu")
    except ValueError:
        print("   refused, as it must: the teacher reads report features")
    else:
        print("   FAILED: a teacher checkpoint loaded into the image-only path")
        return 1

    print("\n6. a fold-0 teacher must be refused for fold 1")
    try:
        load_frozen_teacher(teacher.checkpoint, fold=1, device="cpu")
    except ValueError:
        print("   refused, as it must: that teacher trained on fold 1's validation studies")
    else:
        print("   FAILED: a mismatched fold's teacher loaded, which leaks validation labels")
        return 1

    print("\nSMOKE OK -- wiring holds. The macro figures above are noise, not results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
