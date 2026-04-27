# Wandb Setup & Usage

How wandb is wired into the project, what it logs, and the one-time setup
needed on Alvis. For the day-to-day flow see section 2.3 of
[`scripts_guide.md`](scripts_guide.md); this doc is the deeper reference.

---

## 1. One-time setup

### 1.1 Account
Create a free academic account at <https://wandb.ai/site>. The free tier
covers all our use (per-run scalar logs are tiny; we don't upload model
checkpoints).

### 1.2 Login on alvis1
On alvis1 (login node), once per machine:

```bash
source /mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/thesis_workspace/fl_weather_project/activate_env.sh
wandb login
```

Paste your API key from <https://wandb.ai/authorize>. The key lands in
`~/.netrc`, which is on Cephyr `$HOME` and persists across sessions. SLURM
jobs inherit it via `--export=ALL` from `submit_experiment.sh`.

### 1.3 Verify (optional)
```bash
python -c "import wandb; print(wandb.api.api_key[:6] + '...')"
```
Should print the first six characters of your API key.

Compute-node egress to `api.wandb.ai` is verified working on Alvis (job
6512727, 2026-04-27). If it ever stops working, fall back to offline mode
(section 4 below).

---

## 2. What gets logged

`ExperimentLogger.log_round()` is the single choke point. Every server-side
eval forwards through it, so adding wandb required no edits to any
strategy or client code.

### 2.1 Per-round metrics

| wandb key | source | when |
|---|---|---|
| `server-round` | step axis | every round |
| `server/test_loss` | server-side global eval | every round |
| `server/test_accuracy` | global eval | every round |
| `server/target_class_clean_accuracy` | global eval | every round |
| `server/target_class_num_samples` | global eval | every round |
| `server/asr` | global eval (only when `attack-type` ∈ pixel_backdoor / model_replacement) | every round with attack |
| `server/asr_num_samples` | global eval | every round with attack |
| `client/train_loss` | weighted avg across selected clients | every train round |
| `client/train_accuracy` | weighted avg | every train round |
| `client/val_loss` | weighted avg | every train round |
| `client/val_accuracy` | weighted avg | every train round |
| `client/num-examples` | weighted avg | every train round |

The `client/*` group is captured by `_last_train_metrics` on every custom
strategy plus the `CapturedKrum` / `CapturedMultiKrum` wrappers. If a
custom strategy is added later, it must populate
`self._last_train_metrics` at the end of `aggregate_train` to participate.

### 2.2 One-shot artifacts

- **`client_label_histograms`** — 50 × 43 (clients × classes) heatmap PNG
  rendered once at server startup. Useful for confirming the non-IID
  partition matches expectations.
- **Run config** — every key from the merged `run_config` (Flower's
  `Context.run_config`) is uploaded as wandb config. Filterable in the UI.
- **Final summary** — `summary.json` contents (best-test-accuracy round,
  best-asr round, final metrics) are pushed to `wandb.run.summary` at
  finalize.

### 2.3 What is NOT logged
- Per-client norms (`*_norm_log.json` is the source of truth for those;
  not in wandb scope).
- Model checkpoints (`.pt` files are too heavy and live on Mimer).
- Per-client raw metrics (only the aggregated weighted average).
- Pixel-trigger comparison images (we're abandoning pixel-trigger attacks
  in Phase D.2 onward, so the file isn't always produced).

---

## 3. Organization conventions

Project / group / tag structure is auto-derived from existing config keys:

| wandb field | derivation | example |
|---|---|---|
| project | `gtsrb-{cycle.replace('_','-')}` | `gtsrb-cycle-02` |
| group | `experiment-name` with trailing `<n>mal` and defense tokens stripped | `phaseD-modelrep` |
| name | `{experiment-name}_seed{seed}` | `phaseD-modelrep-15mal-fedmedian_seed42` |
| tags | `cycle`, `phase`, `model-type`, `attack:<type>`, `defense:<type>`, `<n>mal`, `seed<n>` | `[cycle_02, phaseD, resnet18, attack:model_replacement, defense:fedmedian, 15mal, seed42]` |

Override any of these via the YAML keys `wandb-project`, `wandb-group`,
`wandb-tags` (comma-separated). YAMLs that leave them blank get the
auto-derived values, which is the recommended path for consistency.

### 3.1 Useful queries

The conventions are designed so that filter queries scale across cycles:

- "All Phase D.2 runs": filter by tag `phaseD2`.
- "All model-replacement runs across cycles": filter by tag
  `attack:model_replacement` (works in any project).
- "All 15-malicious runs in Cycle 02": filter by tags `cycle_02` AND
  `15mal`.
- "Compare defenses for the same attack": pick a group like
  `phaseD-modelrep`; the wandb workspace shows side-by-side curves with
  defense as the natural color axis.

---

## 4. Online vs offline mode

### 4.1 Online (default)
Set `wandb-enabled: true` and `wandb-mode: online` (both default in
`pyproject.toml`). The job streams scalars to wandb.ai during training.

### 4.2 Offline
Two ways to switch:

```bash
# Per-submit (recommended — doesn't require editing the YAML):
WANDB_MODE=offline ./submit_experiment.sh configs/experiments/...

# Or in the YAML:
wandb-mode: offline
```

The job writes to `<exp_dir>/wandb/` instead of streaming. To upload after
the job finishes:

```bash
source activate_env.sh
wandb sync /mimer/.../fl_outputs/gtsrb/experiments/cycle_02/phaseD2/<exp>/wandb/
```

`wandb sync` is idempotent — safe to re-run if the upload was interrupted.

### 4.3 Disabled
`wandb-enabled: false` (or `WANDB_MODE=disabled`) skips wandb entirely. No
run is created, no network call. Useful for very fast smoke runs where the
wandb startup cost (~2-3s) matters.

---

## 5. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `wandb: ERROR Network error (ConnectionError)` in the SLURM log | Compute node can't reach api.wandb.ai (rare on Alvis) | Re-submit with `WANDB_MODE=offline`, then `wandb sync` from a login node |
| Run shows in wandb but with no client metrics | Strategy didn't populate `_last_train_metrics` (only happens for non-wrapped Flower built-ins; we wrap Krum/MultiKrum already) | If you added a new custom strategy, set `self._last_train_metrics = self._capture_metrics(metrics)` before returning from `aggregate_train` |
| "API key not configured" | First-run case; the venv was recreated and `~/.netrc` is empty | `wandb login` on alvis1 |
| Wandb run name has weird characters | `experiment-name` in the YAML has unusual punctuation | Stick to `[a-zA-Z0-9_-]` for `experiment-name`; the wandb run name is `<experiment-name>_seed<seed>` |
| Two runs in wandb for the same experiment | You re-submitted; each SLURM job creates a fresh wandb run id | Either delete the old run from the wandb UI or use `wandb-group` to keep them grouped |

---

## 6. Adding wandb logging to a new metric

Per-round metric: it must already flow through `ExperimentLogger.log_round`,
which means it must be in either:
- the dict returned by `server_evaluate(...)` in
  `fl_v2/training/server_eval.py` (server-side eval), or
- the aggregated client `MetricRecord` returned by
  `train_metrics_aggr_fn(...)` in the strategy (client-aggregated).

Both flow into wandb automatically — no wandb code change needed.

For one-shot artifacts (a new image / table / config item), add a call to
`logger.log_image(name, path)` or `logger.wandb.log_summary(...)` in
`server_app.py` near where the artifact is generated. The wandb logger is
a no-op when wandb is disabled, so the call is safe unconditionally.
