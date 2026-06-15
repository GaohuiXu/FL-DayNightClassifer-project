# fl_v3 environment — Alvis x86 build + Arrhenius (ARM) rebuild note

The venv is a **throwaway**; the **manifest is the durable artifact**. Rebuild
anywhere from the manifest. This is the Cycle-04 portability posture: we never
depend on a frozen binary venv, only on a clean pinned manifest that rebuilds on
a new machine.

## What's in the manifest (and what is deliberately NOT)

- **Pure-PyTorch stack only.** `flwr[simulation]`, `ray`, `scikit-learn`
  (HDBSCAN for FLAME), `nuscenes-devkit`, `matplotlib`, `wandb`, `pytest`, plus
  the nuScenes runtime deps (`opencv-python-headless`, `pyquaternion`, `shapely`,
  `cachetools`, `tqdm`, `fire`).
- **NO `mmdet3d` / `mmcv` / `spconv`.** These fail to build on 2026-era CUDA and
  on ARM, and ship `atomicAdd` voxel/scatter ops that break bit-determinism. The
  whole BEVFusion-class model (T2) is reimplemented atomic-free from Apache-2.0
  references instead. **This "no mmdet3d" decision IS the Arrhenius-portability
  decision.**
- **`torch` / `numpy` / `scipy` / `Pillow` come from the module**, not pip — see
  the `--system-site-packages` step below. `constraints.txt` pins numpy/scipy to
  the module versions so pip never shadows the CUDA-matched build.
- **HDBSCAN = `sklearn.cluster.HDBSCAN`**, not the standalone `hdbscan` PyPI
  package. sklearn's implementation ships in the wheel (no Cython compile on the
  target), so it rebuilds painlessly on ARM. fl_v2 also used sklearn's HDBSCAN,
  so FLAME parity is exact.

### nuscenes-devkit footgun (resolved)

`nuscenes-devkit==1.1.11` publishes `matplotlib<3.6.0` (via the abandoned
`descartes` map-rendering dep, which has **no Python-3.12 wheel**). We don't use
map rendering, so the build installs **`nuscenes-devkit --no-deps`** and supplies
its real runtime deps (above) at modern, py3.12-compatible, numpy-1.x versions.
The data + `DetectionEval` APIs we need import without `descartes` (asserted in
the build's import-sanity check). Routing around an abandoned transitive is
itself the portability posture.

## Build on Alvis (x86)

```bash
# one command (idempotent — recreates the venv):
bash fl_v3/scripts/build_venv.sh
```

Equivalent manual steps:

```bash
module purge
module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0     # provides torch/numpy/scipy/Pillow
python3 -m venv --system-site-packages /mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/thesis_workspace/fl_weather_project/.venv_v3
source .../.venv_v3/bin/activate
pip install --upgrade pip
pip install --no-cache-dir -c fl_v3/constraints.txt -r fl_v3/requirements.txt
pip install --no-cache-dir --no-deps -c fl_v3/constraints.txt nuscenes-devkit==1.1.11
pip install --no-cache-dir -e fl_v3 --no-deps
pip freeze > fl_v3/requirements.lock.txt
```

The build verifies (and the import-sanity step asserts): `numpy == 1.26.4`
(module build NOT shadowed), `torch` CUDA available, `sklearn.cluster.HDBSCAN`
importable, nuScenes data + `DetectionEval` APIs import with no `descartes`, and
`import fl_v3`.

**Verified build (this T0):** torch 2.7.1 (CUDA), numpy 1.26.4, scipy 1.13.1,
flwr 1.27.0, ray 2.51.1, scikit-learn 1.8.0, matplotlib 3.10.8,
nuscenes-devkit 1.1.11.

## Rebuild on Arrhenius (ARM, H200) — later

1. `module load` that machine's PyTorch/CUDA module (provides ARM torch + numpy).
2. Update `constraints.txt` numpy/scipy pins to that module's versions.
3. Re-run `build_venv.sh` (point `PROJ_ROOT` / `PY_MODULE` at the ARM machine).
4. Re-point data paths (nuScenes ZIPs) — see T1's data module.

Nothing x86-specific is baked into the **manifest**; only the module name + the
numpy/scipy constraint versions change. No package in the manifest requires
compilation against a specific CUDA/arch (the `mmdet3d`/`spconv` hazard is
absent by design).

## Files

- `pyproject.toml` — project metadata + dependency rationale.
- `requirements.txt` — curated direct pins (**the durable manifest**).
- `constraints.txt` — numpy/scipy floor=ceiling at the module versions.
- `requirements.lock.txt` — a `pip freeze` **audit/freeze SNAPSHOT**, NOT a
  portable reinstall file: because the venv is `--system-site-packages`, the
  module-provided deps (torch, numpy, scipy, Pillow, pytest, …) are captured as
  machine-local `@ file:///dev/shm/...` / `@ file:///apps/...` URLs, and the
  editable self-entry encodes a worktree-local path. It records exactly what was
  installed on the Alvis build node; it does NOT `pip install -r` cleanly on a
  fresh checkout or on ARM. **The canonical, portable rebuild is
  `build_venv.sh` + `pyproject.toml`/`requirements.txt`/`constraints.txt`** — not
  the lockfile.
- `scripts/build_venv.sh` — the canonical build.
- `scripts/run_in_venv.sh` — run a command inside the venv (tests, fixtures).
