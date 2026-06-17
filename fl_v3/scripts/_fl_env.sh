#!/bin/bash
# Sourced helper (NOT a SLURM script): the per-job FL/Ray/determinism environment for the
# flwr-1.27 launchers (run_alvis.sh, run_fedavg_a40.sh). Carries ONLY the fl_v2 hardening
# that still applies under flwr 1.27 (T3_SPEC §0.1): per-job auto-SuperLink ports + per-job
# Ray ports/tmp. flwr 1.27 `flwr run` AUTO-MANAGES the local SuperLink — there is NO manual
# `flower-superlink` launch and NO SuperExec grep/sleep wait (those collide on default ports
# and re-create the silent race). Determinism env (CUBLAS) is set BEFORE any CUDA context.

# fl_setup_env: export the determinism + per-job-isolation environment. Idempotent.
fl_setup_env() {
    export CUBLAS_WORKSPACE_CONFIG=":4096:8"          # cuBLAS GEMM determinism (pre-CUDA)
    export TORCH_HOME="/cephyr/users/gaohui/Alvis/.cache/torch"  # offline ImageNet weights
    export PYTHONHASHSEED=0                            # cheap defense-in-depth
    export RAY_DEDUP_LOGS=0                            # full Ray logs (cheap)
    export WANDB_MODE=offline                          # offline compute node — never egress

    local jid=${SLURM_JOB_ID:-0}
    # flwr-1.27 binds its AUTO-managed local SuperLink to these (defaults 39093/39094
    # collide across concurrent same-node jobs). The ONLY SuperLink hardening still needed.
    export FLWR_LOCAL_CONTROL_API_PORT=$(( 39000 + (jid % 1000) * 2 ))
    export FLWR_LOCAL_SIMULATIONIO_API_PORT=$(( FLWR_LOCAL_CONTROL_API_PORT + 1 ))
    # Per-job Ray ports — even with unique SuperLink ports, Ray's default GCS/dashboard/
    # object-manager ports cause a silent 2nd-on-node failure (fl_v2 finding).
    local ray_base=$(( 10000 + (jid % 1000) * 10 ))
    export RAY_GCS_SERVER_PORT=$(( ray_base + 1 ))
    export RAY_DASHBOARD_PORT=$(( ray_base + 2 ))
    export RAY_NODE_MANAGER_PORT=$(( ray_base + 3 ))
    export RAY_OBJECT_MANAGER_PORT=$(( ray_base + 4 ))
    export RAY_RUNTIME_ENV_AGENT_PORT=$(( ray_base + 5 ))
    # Per-job Ray + FLWR temp dirs (isolate the raylet/plasma sockets + the auto-SuperLink
    # state from concurrent jobs sharing /tmp).
    export RAY_TMPDIR="/tmp/ray_${jid}"
    export FLWR_HOME="/tmp/flwr_${jid}"
    mkdir -p "$RAY_TMPDIR" "$FLWR_HOME"
    echo "[_fl_env] FLWR_HOME=$FLWR_HOME ctl=$FLWR_LOCAL_CONTROL_API_PORT sio=$FLWR_LOCAL_SIMULATIONIO_API_PORT ray_base=$ray_base RAY_TMPDIR=$RAY_TMPDIR"
}

# fl_stamp_supernodes N SRC_TOML DST_TOML [BLOCK]
# Write DST_TOML = SRC_TOML with `options.num-supernodes = N` ONLY inside the given federation
# BLOCK header (default the single-GPU `[superlink.local-simulation-gpu]`; pass
# `[superlink.local-simulation-gpu-4x]` for the D9 Path-A multi-GPU federation, whose default
# num-supernodes=8 is hardcoded to the gate config). Exact-string block match (so
# `…-gpu` does NOT also match `…-gpu-4x`). The launcher stamps the DERIVED client count N.
fl_stamp_supernodes() {
    local n="$1" src="$2" dst="$3" block="${4:-[superlink.local-simulation-gpu]}"
    awk -v n="$n" -v block="$block" '
        /^\[/ { inblk = ($0 == block) }
        inblk && /^options\.num-supernodes[[:space:]]*=/ { print "options.num-supernodes = " n; next }
        { print }
    ' "$src" > "$dst"
    if ! grep -q "^options.num-supernodes = ${n}$" "$dst"; then
        echo "[_fl_env] ERROR: failed to stamp num-supernodes=${n} into $dst (block ${block})" >&2
        return 3
    fi
    echo "[_fl_env] stamped num-supernodes=${n} -> $dst (block ${block})"
}

# fl_preflight_offline: assert the run will make ZERO network calls — ImageNet weights
# cached + nuScenes info-cache built. A missing one would silently try to egress (and hang)
# on the offline compute node, so fail LOUD here on the (online-capable) entry instead.
fl_preflight_offline() {
    local cache_dir="${1:-./fl_outputs/nuscenes/info_cache}"
    local ok=1
    [ -f "${TORCH_HOME}/hub/checkpoints/resnet18-f37072fd.pth" ] || { echo "[_fl_env] MISSING resnet18 ImageNet weights under TORCH_HOME"; ok=0; }
    [ -f "${TORCH_HOME}/hub/checkpoints/swin_t-704ceda3.pth" ]   || { echo "[_fl_env] MISSING swin_t ImageNet weights under TORCH_HOME"; ok=0; }
    if ! ls "${cache_dir}"/*.json >/dev/null 2>&1 && ! ls "${cache_dir}"/* >/dev/null 2>&1; then
        echo "[_fl_env] MISSING nuScenes info-cache under ${cache_dir} — build it on the login node:"
        echo "          bash fl_v3/scripts/run_in_venv.sh python fl_v3/scripts/build_nuscenes_cache.py"
        ok=0
    fi
    [ "$ok" = 1 ] || { echo "[_fl_env] offline-preflight FAILED (would egress on the compute node)"; return 3; }
    echo "[_fl_env] offline-preflight OK (weights cached + info-cache present)"
}

# fl_silent_exit_guard RUN_LOG
# `flwr run` returns exit 0 even on a ZERO-round run (the documented Ray-under-SLURM silent
# race: ServerApp books the run but trains nothing). A real, fully-aggregated run prints
# `FL_TRAINABLE_CHECKSUM = ...` exactly once at completion (server_app). Its ABSENCE => the
# run silently did nothing => return non-zero so SLURM marks the job FAILED (visible), not a
# dead run masquerading as a clean baseline.
fl_silent_exit_guard() {
    local run_log="$1"
    if grep -q "FL_TRAINABLE_CHECKSUM = " "$run_log" 2>/dev/null; then
        return 0
    fi
    echo "[_fl_env] ERROR: '$run_log' has NO 'FL_TRAINABLE_CHECKSUM' line — silent zero-round run." >&2
    echo "[_fl_env]        Forcing FAILED so it is not mistaken for a real baseline. Resubmit." >&2
    return 1
}
