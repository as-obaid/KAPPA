# slurm/

Slurm scripts for running [KAPPA](../README.md) on a GPU cluster. Partition names, the GPU type
and the `module load` line are specific to the cluster they were written for.

## Configuration

Every script reads `KAPPA_*` settings from the shell first, then from the repository's `.env`:

```bash
KAPPA_SCRATCH=/scratch/<user>                 # required: envs, caches, logs, W&B
KAPPA_DATA_ROOT=/scratch/<user>/kappa-data    # optional; this is the default
```

Optional overrides: `KAPPA_VENV` (vLLM env, default `$KAPPA_SCRATCH/envs/kappa-vllm`),
`KAPPA_TRAIN_VENV` (default `$KAPPA_SCRATCH/envs/kappa-train`), `KAPPA_WEIGHTS_DIR` (default
`$HOME/kappa-weights`). Scripts stop with a message if `KAPPA_SCRATCH` is unset.

`sbatch` reads its own options before any script runs, so set these in the shell, not `.env`:

```bash
export SBATCH_ACCOUNT=<allocation>                                  # or: sbatch -A <allocation>
export SBATCH_OUTPUT=/scratch/<user>/kappa-work/logs/%x_%A_%a.out   # optional
export SBATCH_ERROR=/scratch/<user>/kappa-work/logs/%x_%A_%a.err    # optional
```

Without them, logs go to `slurm-<jobid>.out`. Slurm does not create a log directory. Submit
every job from the repository root.

## Run order

```bash
# login node
slurm/env_build.sh all
scripts/get_data.sh --fetch-only
slurm/stage_weights.sh                       # LLM weights; also takes timm/<model> repos
slurm/stage_torch_weights.sh                 # 3D configs only
# compute nodes
sbatch slurm/get_data.sbatch                 # or slurm/extract.sbatch
sbatch slurm/manifest.sbatch                 # then: python scripts/build_manifest.py --merge
sbatch slurm/extract_reports.sbatch
sbatch slurm/cache.sbatch
sbatch slurm/train.sbatch --config configs/baseline_axial_fs1.yaml --track
# login node
slurm/sync_wandb.sh
```

Labels come from `scripts/derive_labels.py` after report extraction; `train.sbatch` checks
that every input its config names exists before it starts.

## Files

| File | Runs on | Purpose |
|---|---|---|
| `env_common.sh` | sourced | Loads settings, requires `KAPPA_SCRATCH`, moves caches off `$HOME` |
| `env_build.sh` | login | Builds the training (cu126) and vLLM (cu130) environments with uv |
| `locks/` | | `uv pip freeze` of each environment |
| `stage_weights.sh` | login | Downloads Hugging Face weights into `$HF_HOME` |
| `stage_torch_weights.sh` | login | Downloads torchvision 3D weights into `$TORCH_HOME` |
| `get_data.sbatch` | CPU | Verifies and extracts the archive through `scripts/get_data.sh` |
| `extract.sbatch` | CPU | Parallel, resumable archive extraction |
| `manifest.sbatch` | CPU array | Series manifest, one header read per series |
| `extract_reports.sbatch` | GPU array | Structured findings from reports with vLLM |
| `cache.sbatch` | CPU array | uint8 image cache for the dataloader |
| `train.sbatch` | GPU array | One config, one fold per task |
| `sync_wandb.sh` | login | Uploads offline W&B runs |

## Offline compute nodes

- **No outbound internet.** Builds, downloads and W&B sync run on the login node, under `tmux`.
- **Fail fast.** Jobs set `HF_HUB_OFFLINE=1`, so a missing weight is an immediate error rather
  than a hang. For 3D configs, export `TORCH_HOME=$KAPPA_SCRATCH/torch_home` before `sbatch`.
- **W&B.** Runs log offline. Pass `--track` or no run is written. `--signal=B:USR1@300` lets a
  run killed at walltime commit its log first.
- **Login-node memory.** Do not import torch or build models there.

## Site-specific settings

- **GPU.** `--gres=gpu:h200:1` names the card because mixed partitions include V100 and T4,
  which lack bf16. Keep one GPU per job.
- **Partitions.** `short`, `gpu`, `gpu-short` and `sharing` are local names. A shorter partition
  needs an explicit `--time` within its limit.
- **Drivers.** The vLLM environment needs NVIDIA driver 580 or newer; pin a node with
  `--nodelist` if the partition mixes drivers. Training (cu126) runs on 570-series drivers.
