# slurm/locks/

The exact package versions of each cluster environment, as `uv pip freeze` output.
`slurm/env_build.sh` rewrites the file after every build.

```
locks/
└── kappa-train.lock   # training env: torch 2.14.0+cu126, timm 1.0.29, wandb 0.29.0
```

`slurm/env_build.sh vllm` writes `kappa-vllm.lock` beside it.

- **Why it is tracked.** The environments live on purgeable scratch and are rebuilt, not backed
  up. The lock file is the record of what a working build contained. Every tracked training run
  logs `kappa-train.lock` to W&B (`kappa.tracking.LOCKFILE`), so keep the path and format.
- **Using it.** `env_build.sh` resolves versions fresh and does not read this file. Diff a new
  lock against the committed one to find version drift.
