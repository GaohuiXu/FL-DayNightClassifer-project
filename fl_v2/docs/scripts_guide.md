# Scripts Usage Guide

This guide explains every shell script in the project: what it does, what it
depends on, and when to use it. Read the **Environment Activation** section
first — every other script assumes that environment is set up (SLURM jobs
set it up themselves inside the job; anything you run on a login node needs
you to set it up first).

> **Current workflow note.** Since ResNet18 + Flower + Ray is too heavy for
> login-node execution, we **do not smoke-test locally anymore**. Both
> smoke tests (short runs) and full experiments are submitted to Alvis via
> `submit_experiment.sh`. The "local" scripts in section 2 are kept for
> legacy / debugging use — you likely won't use them day-to-day.

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
| Legacy local experiment runs (`run_flwr_local.sh`, `run_experiment.sh`) | **You must `source activate_env.sh`** — but we don't use these anymore, see section 2 |

In practice, the only time you source it interactively is to run analysis /
plotting on a login node after experiments have finished on Alvis.

---

## 2. Experiment Execution

> **We always submit to Alvis.** ResNet18 + Flower + Ray won't run in
> reasonable time on a login node, and the login node has no GPU. The
> legacy local workflow (sections 2.3–2.6) is kept for reference and the
> occasional dry-run check, but not used in day-to-day experiments.

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
2. Creates a **per-job `FLWR_HOME`** at `/tmp/flwr_${SLURM_JOB_ID}`. This
   isolates concurrent jobs so two experiments running at once don't
   trample each other's SuperLink state.
3. Parses `EXPERIMENT_YAML` into `--run-config` overrides (same awk parser
   as `run_experiment.sh`).
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

### 2.3 `run_experiment.sh` — local run from a YAML *(legacy)*

**Path:** [../run_experiment.sh](../run_experiment.sh).

Runs the same experiment on the current machine without SLURM. Needed the
environment to be activated already and a local SuperLink running
(section 2.5). **Not used day-to-day** — ResNet18 makes the forward pass
slow on a login node and the login node has no GPU. Kept because a
compute-node interactive session (`srun --pty bash`) can still use it.

```bash
source ../activate_env.sh
./run_experiment.sh configs/experiments/phaseC_v2/1_clean.yaml
```

### 2.4 `run_flwr_local.sh` — raw local run (no YAML) *(legacy)*

**Path:** [../run_flwr_local.sh](../run_flwr_local.sh).

Minimal wrapper: sources `flwr_local_env.sh` and runs
`flwr run . local-simulation-gpu --stream "$@"`. Was useful when iterating
on code with hand-passed `--run-config`. Same "only works if you're on a
GPU node with env activated and a SuperLink up" caveat as 2.3. **Not used
day-to-day.**

### 2.5 `start_superlink.sh` / `stop_superlink.sh` — local SuperLink *(legacy)*

**Paths:** [../start_superlink.sh](../start_superlink.sh),
[../stop_superlink.sh](../stop_superlink.sh).

The historical two-terminal local workflow, rarely used now:

```bash
# Terminal 1
source ../activate_env.sh
./start_superlink.sh               # runs foreground, Ctrl+C to stop

# Terminal 2
source ../activate_env.sh
./run_flwr_local.sh                # or ./run_experiment.sh <yaml>

# When done (or if ports 39093/39094 are stuck):
./stop_superlink.sh
```

`stop_superlink.sh` is still genuinely useful if a SLURM job died mid-run
on a login-node shared state or if you used the legacy workflow: it kills
anything listening on 39093/39094, pkills leftover
`flower-superlink` / `flwr-simulation` processes, and removes
`$FLWR_HOME/local-superlink/` so the next start gets a clean DB.

### 2.6 `flwr_local_env.sh` — shared env snippet

**Path:** [../flwr_local_env.sh](../flwr_local_env.sh).

One-line helper that local scripts `source`:
`export FLWR_HOME="${HOME}/.flwr"`. Ceph home is used instead of `/tmp`
because `/tmp` is node-local and vanishes between sessions — do **not**
change this to `/tmp`. SLURM jobs (`run_alvis.sh`) use per-job
`/tmp/flwr_${JOBID}` internally and that's fine because they clean up on
exit.

### 2.7 `monitor.sh` — check on a submitted job

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
Stage 4: training curves + defense compare   →  run_phaseC_curves.sh / run_phaseD_curves.sh
```

| Script | Stage | GPU? | Typical time |
|---|---|---|---|
| `run_phaseC_extract.sh` | Stage 1 (Phase C v2) | NOGPU (CPU forward passes on test set) | ~2-4 h |
| `run_phaseD_extract.sh` | Stage 1 (Phase D) | NOGPU | ~1-2 h |
| `run_framework.sh` | Stage 2 (Phase C v2) | NOGPU | ~30 min |
| `run_phaseD_framework.sh` | Stage 2 (Phase D) | NOGPU | ~15 min |
| `run_phaseC_analyze.sh` | Stage 3 | NOGPU | ~30 min |
| `run_phaseD_viz.sh` | Stage 3 | NOGPU | ~15 min |
| `run_phaseC_curves.sh` | Stage 4 | NOGPU | ~5 min |
| `run_phaseD_curves.sh` | Stage 4 | NOGPU | ~5 min |

All stages are **idempotent**: they skip outputs that already exist, so
re-running after a partial failure is safe.

Submit with `sbatch`:

```bash
sbatch analysis/run_phaseC_extract.sh
sbatch --dependency=afterany:<extract_jobid> analysis/run_framework.sh
```

Or run locally after activating the env (fine for the short Stage 2–4
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
analysis/run_phaseX_curves.sh        (training curves + defense compare)
```

(Legacy path — rarely used: `start_superlink.sh` + `run_flwr_local.sh` /
`run_experiment.sh` on a GPU node with `activate_env.sh` sourced.)

---

## 5. Common Gotchas

1. **`python` doesn't find `torch` on the login node.** You forgot
   `source activate_env.sh`. Module must come before venv activation.
2. **Inline YAML comments break parsing.** The awk parser in
   `run_experiment.sh` / `run_alvis.sh` treats `key: 0.0  # comment` as
   the string `0.0  # comment`. Strip inline comments from YAML values.
3. **`FLWR_HOME` on `/tmp`.** Don't — for local scripts. `/tmp` is
   node-local and disappears between sessions; use `$HOME/.flwr` on Ceph
   (default in `flwr_local_env.sh`). SLURM jobs using per-job
   `/tmp/flwr_${JOBID}` are fine because they clean up on exit.
4. **Port 39093/39094 already in use.** Left-over SuperLink from a crashed
   local run (or an interrupted interactive compute-node session). Run
   `./stop_superlink.sh`.
5. **SLURM job times out at 3 h.** Edit `#SBATCH -t` in `run_alvis.sh`
   (or the specific `analysis/run_phaseX_*.sh`) before submitting.
6. **Analysis script fails with "no such file: features_test.npz".**
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
