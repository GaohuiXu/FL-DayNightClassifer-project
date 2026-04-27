# Scripts Usage Guide

This guide explains every shell script in the project: what it does, what it
depends on, and when to use it. Read the **Environment Activation** section
first — every other script assumes that environment is set up (SLURM jobs
set it up themselves inside the job; anything you run on a login node needs
you to set it up first).

> **Current workflow.** Every experiment — smoke test or full run — is
> submitted to Alvis via `submit_experiment.sh`. ResNet18 + Flower + Ray
> won't run in reasonable time on a login node, and the login node has no
> A40 GPU. Login-node Python is for analysis / plotting only. The earlier
> "two-terminal local SuperLink" workflow (`start_superlink.sh`,
> `run_flwr_local.sh`, `run_experiment.sh`, `stop_superlink.sh`,
> `flwr_local_env.sh`) was retired with the Cephyr→Mimer migration on
> 2026-04-27 — those scripts have been removed.

---

## 1. Environment Activation — `activate_env.sh`

**Path:** [../../activate_env.sh](../../activate_env.sh) (project root, one
level above `fl_v2/`).

This is the **single entry point** for activating the Python environment on
Alvis. Run it once per shell session before doing anything that calls
`python`, `flwr`, or any analysis script on a login/compute node:

```bash
source /mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/thesis_workspace/fl_weather_project/activate_env.sh
```

Note: use `source` (or `.`), not `bash activate_env.sh` — the `module load`
and `venv activate` must modify your current shell.

### What it does — two steps

```bash
module purge
module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0

source /mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/thesis_workspace/fl_weather_project/.venv/bin/activate
```

**Step 1 — the HPC module.** `PyTorch/2.7.1-foss-2024a-CUDA-12.6.0` is
Alvis's curated PyTorch module. It provides:
- Python 3.12 interpreter
- CUDA 12.6 runtime + cuDNN
- `torch 2.7.1`, `torchvision`, `numpy`, `scipy`, `scikit-learn`
- the foss-2024a toolchain (GCC, OpenBLAS, MPI)

`module purge` first clears anything inherited from login defaults so the
stack is reproducible.

**Step 2 — the project venv.** The virtualenv at
`fl_weather_project/.venv/` was created **on top of** the module's Python.
Activating it prepends project-specific packages that the module doesn't
carry: `flwr`, `ray`, `matplotlib`, `seaborn`, `umap-learn`, `pandas`, etc.

### Why both steps are required

The venv's `python` is a symlink to the module's Python 3.12 binary. If the
module is not loaded, that symlink resolves to nothing and `python` either
fails outright or (worse) silently picks a system Python that can't import
`torch`. **Always load the module first, then activate the venv.** Doing
only one of the two will produce cryptic import errors like "torch not
found" or "libpython3.12.so missing".

### When you actually source it

| Context | What activates the env |
|---|---|
| Running **local analysis** on a login node (`python -m analysis.xxx`, quick plotting, reading `.npz` files) | **You must `source activate_env.sh`** |
| Submitting an experiment via `submit_experiment.sh` | Handled inside `run_alvis.sh` — nothing to do |
| Any `analysis/run_phase*.sh` submitted via `sbatch` | Script detects `SLURM_JOB_ID` and activates the env itself — nothing to do |
| Running `analysis/run_phase*.sh` **locally** (without sbatch) | **You must `source activate_env.sh`** first |

In practice, the only time you source it interactively is to run analysis /
plotting on a login node after experiments have finished on Alvis.

---

## 2. Experiment Execution

> **We always submit to Alvis.** ResNet18 + Flower + Ray won't run in
> reasonable time on a login node, and the login node has no A40 GPU.

### 2.1 `run_alvis.sh` — the SLURM job template

**Path:** [../run_alvis.sh](../run_alvis.sh).

This is the job script that actually runs a federated experiment on an
Alvis compute node. It is **not** meant to be invoked directly; use
`submit_experiment.sh` (see 2.2) instead, which passes the YAML via the
`EXPERIMENT_YAML` environment variable.

SLURM directives at the top control the resource request:

```bash
#SBATCH -A NAISS2025-22-1113     # project account
#SBATCH -p alvis                 # partition
#SBATCH --gpus-per-node=A40:1    # one A40 GPU
#SBATCH -t 0-03:00:00            # 3 hours
#SBATCH -J flwr_gtsrb            # default job name (overridden by --job-name)
```

Inside the job it:
1. Loads the PyTorch module and activates the venv (same two steps as
   `activate_env.sh`).
2. Creates a **per-job `FLWR_HOME`** at `/tmp/flwr_${SLURM_JOB_ID}` and
   copies [`configs/flwr_config.toml`](../configs/flwr_config.toml) (the
   repo-checked federation config — 50 supernodes, 0.10 GPU each) into
   it. This isolates concurrent jobs so two experiments running at once
   don't trample each other's SuperLink state.
3. Parses `EXPERIMENT_YAML` into `--run-config` overrides via an inline
   awk parser.
4. Starts a private `flower-superlink` in the background on ports
   39093/39094.
5. Calls `flwr run . local-simulation-gpu --stream --run-config ...`.
6. Cleans up the SuperLink and `/tmp/flwr_${SLURM_JOB_ID}` on exit.

**If you need a different resource profile** (longer time, T4 instead of
A40, CPU-only), copy this file and edit the `#SBATCH` lines rather than
editing the original — the analysis pipeline scripts still expect the
standard profile.

### 2.2 `submit_experiment.sh` — the standard entry point

**Path:** [../submit_experiment.sh](../submit_experiment.sh).

**This is how we run every experiment — smoke test or full run.**

```bash
./submit_experiment.sh configs/experiments/phaseC_v2/1_clean.yaml
./submit_experiment.sh configs/experiments/phaseD/1_modelrep_5mal_nodefense.yaml
```

What it does:
- Uses the YAML filename (without `.yaml`) as the SLURM job name, so
  `squeue` output is readable.
- Calls `sbatch --job-name=<name> --export=ALL,EXPERIMENT_YAML=<path> run_alvis.sh`.

Running with no argument prints a menu of available experiments grouped by
phase subdirectory.

For a smoke test, keep a small YAML under
`configs/experiments/<phase>/` with `num-server-rounds: 2` or similar — it
still runs on Alvis, still uses a real GPU, but finishes in a couple of
minutes.

**Dependency:** `submit_experiment.sh` → `run_alvis.sh`. The `-t` time
limit and `--gpus-per-node` GPU request live in `run_alvis.sh`. If a YAML
has more rounds than fit in 3 hours, edit `run_alvis.sh`'s `#SBATCH -t`
line before submitting.

### 2.3 Wandb (live monitoring)

Server-side wandb logging is wired into `ExperimentLogger` so every per-round
metric automatically lands in a wandb run alongside `rounds.csv`. There is no
separate "wandb mode" — set `wandb-enabled: true` in the YAML and submit
normally.

What gets logged:
- **Per-round, server namespace:** `test_loss`, `test_accuracy`,
  `target_class_clean_accuracy`, `asr` (when an attack is active), plus the
  `n=` sample counts.
- **Per-round, client namespace:** `train_loss`, `train_accuracy`,
  `val_loss`, `val_accuracy`, `num-examples` (weighted average across
  selected clients, captured by the strategy).
- **One-shot artifacts:** client × class label-histogram heatmap PNG (rendered
  at startup), final summary scalars (best test_accuracy round, best ASR
  round).
- **Run config:** the entire merged `run_config` is uploaded as wandb
  config so any hyperparameter shows up in side-by-side comparisons.

Naming and grouping (so dozens of runs stay readable):
- **Project:** `gtsrb-{cycle}` (with `_` → `-`). E.g. `gtsrb-cycle-02`.
- **Group:** auto-derived from `experiment-name` by stripping trailing
  defense / `<n>mal` tokens. E.g. `phaseD-modelrep-15mal-nodefense` →
  group `phaseD-modelrep`. Override with `wandb-group` in the YAML.
- **Run name:** `<experiment-name>_seed<seed>`.
- **Tags:** `cycle`, `phase`, `model-type`, `attack:<type>`,
  `defense:<type>`, `<n>mal` (when attack), `seed<n>`. Add more via
  `wandb-tags: "tag1,tag2"`.

Online vs offline:
- Default: `wandb-mode: online`. Compute-node egress to api.wandb.ai is
  verified working on Alvis (job 6512727, 2026-04-27).
- Offline: set `WANDB_MODE=offline` in the submit shell (or
  `wandb-mode: offline` in the YAML). The run writes to
  `<exp_dir>/wandb/` and you upload later with `wandb sync <exp_dir>/wandb/`
  from a login node after `source activate_env.sh`.
- Disable entirely: `wandb-enabled: false` (no run is created; nothing
  uploads).

First-time setup: `wandb login` once on alvis1; the API key lands in
`~/.netrc` and SLURM jobs pick it up via `--export=ALL`. See
[`wandb_setup.md`](wandb_setup.md) for the full setup walkthrough.

### 2.4 `monitor.sh` — check on a submitted job

**Path:** [../monitor.sh](../monitor.sh).

```bash
./monitor.sh                 # list your queued/running jobs
./monitor.sh 12345           # status + last 40 lines of output
./monitor.sh 12345 follow    # tail -f the output log
```

Reads the SLURM log at
`/mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/fl_outputs/slurm/flwr_gtsrb_<jobid>.out`.

---

## 3. Analysis Pipeline (`analysis/run_phase*.sh`)

All scripts under [../analysis/](../analysis/) are SLURM-submittable
**and** locally runnable. They check `SLURM_JOB_ID` and only do `module
load + venv activate` when they detect they are inside a SLURM job —
running them locally requires you to `source activate_env.sh` first.

Analysis scripts are light enough to run on a login node (no FL
simulation, no GPU needed — most are `NOGPU` feature-extraction or
plotting). For the bigger ones (`run_phaseC_extract.sh` does 12k forward
passes per checkpoint × multiple rounds × multiple experiments),
`sbatch` them.

The standard pipeline for a phase is:

```
Stage 1: extract features from checkpoints   →  run_phaseX_extract.sh
Stage 2: compute 4-axis framework metrics    →  run_framework.sh (or run_phaseD_framework.sh)
Stage 3: visualize (t-SNE, cluster plots)    →  run_phaseC_analyze.sh / run_phaseD_viz.sh
```

(Training-curves / defense-comparison plots are no longer produced offline —
wandb covers them. See section 2.3.)

| Script | Stage | GPU? | Typical time |
|---|---|---|---|
| `run_phaseC_extract.sh` | Stage 1 (Phase C v2) | NOGPU (CPU forward passes on test set) | ~2-4 h |
| `run_phaseD_extract.sh` | Stage 1 (Phase D) | NOGPU | ~1-2 h |
| `run_framework.sh` | Stage 2 (Phase C v2) | NOGPU | ~30 min |
| `run_phaseD_framework.sh` | Stage 2 (Phase D) | NOGPU | ~15 min |
| `run_phaseC_analyze.sh` | Stage 3 | NOGPU | ~30 min |
| `run_phaseD_viz.sh` | Stage 3 | NOGPU | ~15 min |

All stages are **idempotent**: they skip outputs that already exist, so
re-running after a partial failure is safe.

Submit with `sbatch`:

```bash
sbatch analysis/run_phaseC_extract.sh
sbatch --dependency=afterany:<extract_jobid> analysis/run_framework.sh
```

Or run locally after activating the env (fine for the short Stage 2–3
scripts, too slow for Stage 1):

```bash
source ../activate_env.sh
./analysis/run_framework.sh
```

---

## 4. Dependency Diagram

```
activate_env.sh  (module load + venv activate)
      │
      ├── used interactively ──────► python -m analysis.xxx  (login-node analysis)
      │
      └── embedded inside SLURM ───► run_alvis.sh  ◄── submit_experiment.sh  ◄── ← how every experiment is launched
                                     │
                                     └─► starts its own SuperLink on /tmp
                                     └─► flwr run . local-simulation-gpu

analysis/run_phaseX_extract.sh      (saves .npz features per round)
            │
            ▼
analysis/run_framework.sh            (computes 4-axis profile JSON/CSV)
            │
            ▼
analysis/run_phaseX_analyze.sh       (t-SNE / cluster plots)

# Training curves / defense comparisons live in wandb (see section 2.3)
# rather than as offline analysis-pipeline outputs.
```

---

## 5. Common Gotchas

1. **`python` doesn't find `torch` on the login node.** You forgot
   `source activate_env.sh`. Module must come before venv activation.
2. **Inline YAML comments break parsing.** The awk parser in
   `run_alvis.sh` treats `key: 0.0  # comment` as the string
   `0.0  # comment`. Strip inline comments from YAML values.
3. **SLURM job times out at 3 h.** Edit `#SBATCH -t` in `run_alvis.sh`
   (or the specific `analysis/run_phaseX_*.sh`) before submitting.
4. **Analysis script fails with "no such file: features_test.npz".**
   Stage 1 (extract) hasn't run yet, or was run before a new checkpoint
   was saved. Re-run the extract script — it's idempotent.

---

## 6. Conventions When Adding a New Script

- **SLURM vs local:** if it might run either way (like the `analysis/`
  scripts), guard the `module load` block with `if [[ "${SLURM_JOB_ID:-}"
  != "" ]]; then ... fi`. Locally the user is responsible for `source
  activate_env.sh`.
- **Always:** `set -euo pipefail` at the top, `export OPENBLAS_NUM_THREADS=16`
  and `export PYTHONUNBUFFERED=1` before any Python invocation.
- **Output paths:** put SLURM logs under
  `/mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/fl_outputs/slurm/` and
  experiment artifacts under `.../fl_outputs/gtsrb_v2/<phase>/`.
- **Idempotency:** check if outputs exist before recomputing; `[skip]`
  messages should explain which file already existed.
