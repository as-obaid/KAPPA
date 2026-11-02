# data/

Bulk data for KAPPA. It lives here on a laptop and under `KAPPA_DATA_ROOT` on a cluster. The
competition data is distributed only through Kaggle, so only this file and `folds/` are tracked.

## Directory structure

```
data/
├── .stage/                      # downloaded archive, kept so a fetch resumes
├── raw/                         # extracted competition files, read-only
│   ├── .kappa_stamp             # written after a complete extraction
│   ├── train.csv
│   └── train_series/            # <study_uid>/<series_uid>/<sop_uid>.dcm
├── interim/
│   ├── manifest.parquet         # one row per series (scripts/build_manifest.py)
│   └── labels/derived.parquet   # derived labels (scripts/derive_labels.py)
├── processed/cache/             # uint8 volumes (scripts/build_cache.py)
├── folds/                       # tracked (scripts/build_folds.py)
│   ├── folds_v1.csv             # 5 folds over 4,407 studies
│   └── folds_v2.csv             # 15% frozen test split + 5 folds
└── runs/<name>/                 # checkpoints and out-of-fold predictions
```

## Paths

```bash
make paths                       # print the resolved layout
```

- **Resolution.** `kappa.data.paths` owns every location. `data_root()` is `KAPPA_DATA_ROOT`
  when set, otherwise `<repo>/data`; `raw_dir()`, `interim_dir()` and `processed_dir()` sit under
  it. `folds_dir()` always resolves inside the repository.
- **Configuration.** The module and `scripts/get_data.sh` read `.env` at the repository root
  (template: `.env.example`) without overriding exported variables.
- **Training configs.** Configs name inputs relative to the working directory, for example
  `data/interim/manifest.parquet`. When the data root is elsewhere, link `data/raw`,
  `data/interim` and `data/processed` to it; `slurm/train.sbatch` does this itself.

## Folds

- **Tracked.** The split travels with the code. Never edit a fold file; a new split is a new
  version.
- **CSV only.** `.gitignore` excludes `*.parquet`, `*.npy` and similar extensions everywhere,
  including here.
- **Columns.** `fold` keeps studies sharing a patient ID or report text together;
  `fold_scanner` also keeps each scanner in one fold. In `folds_v2.csv`, held-out studies have
  `split == "test"` and `fold == -1`.

## Fetching the data

1. Accept the rules at
   <https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules>.
2. Save a Kaggle API token to `~/.kaggle/kaggle.json`, or export `KAGGLE_USERNAME` and
   `KAGGLE_KEY`.
3. Fetch and extract:

```bash
make data            # download the 247 GiB archive to .stage/ (login node, resumable)
make extract         # sbatch slurm/extract.sbatch: parallel extraction, writes the stamp
scripts/get_data.sh  # without Slurm: fetch, verify and extract in one run
```

- **Disk.** The archive and the 530 GiB tree coexist, so `get_data.sh` requires 800 GiB free
  (`KAPPA_MIN_FREE_GIB`).
- **Idempotent.** `make data` exits immediately when `raw/.kappa_stamp` exists.
