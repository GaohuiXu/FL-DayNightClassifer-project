"""TF32 determinism re-confirmation gate — run on an A40 (or A100), NOT the login T4.

WHY THIS EXISTS: the login node is a Tesla T4 (Turing, cc 7.5) with NO TF32 tensor
cores, so on the login node ``numeric_mode='tf32'`` silently falls back to true FP32
and `high`==`highest` byte-for-byte (a FALSE PASS). THIS gate refuses to run on cc<8
hardware and re-confirms, on hardware where TF32 actually engages, that the REAL
production path (``enforce_determinism(numeric_mode='tf32')`` — D14 mechanics) is:
  (a) no-raise under the strict deterministic harness,
  (b) run-to-run BYTE-IDENTICAL (TF32 is deterministic same-seed on one GPU), and
  (c) DIFFERENT from FP32 (=> TF32 genuinely engaged → adopting it is a NEW reference,
      a re-baseline, not run-to-run drift).

It exercises GEMM + conv (the matmul/conv-heavy ops that dominate the frozen Swin-T
backbone). Full-model byte-identity is established separately by the D/E paired FL runs
(the new TF32 reference checksum); this gate is the fast regime-mechanics prerequisite.

Run:  sbatch fl_v3/scripts/run_tf32_det_gate_a40.sh
(or:  CUBLAS_WORKSPACE_CONFIG=:4096:8 bash fl_v3/scripts/run_in_venv.sh \
          python fl_v3/scripts/tf32_det_gate_a40.py [OUT.json] )
"""
import hashlib
import json
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

sys.path.insert(0, "fl_v3/src")
from fl_v3.utils.runtime import enforce_determinism, precision_state, seed_everything


def fatal(msg):
    print(f"[tf32_gate] FATAL: {msg}", file=sys.stderr)
    sys.exit(2)


def checksum(t):
    h = hashlib.sha256()
    h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


def gemm(seed=1234, dev="cuda"):
    seed_everything(seed)
    a = torch.randn(4096, 4096, device=dev)
    b = torch.randn(4096, 4096, device=dev)
    return a @ b


def conv(seed=1234, dev="cuda"):
    seed_everything(seed)
    x = torch.randn(8, 64, 128, 128, device=dev)
    w = torch.randn(128, 64, 3, 3, device=dev)
    return torch.nn.functional.conv2d(x, w, padding=1)


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else None
    if not torch.cuda.is_available():
        fatal("no CUDA — TF32 is a hardware feature; run on an A40/A100 node.")
    name = torch.cuda.get_device_name(0)
    cc = torch.cuda.get_device_capability(0)
    print(f"[tf32_gate] device={name} cc={cc} torch={torch.__version__}")
    # HARD REFUSAL on cc<8: Turing/T4 has no TF32 tensor cores -> tf32 silently == fp32 ->
    # the gate would FALSE-PASS. The whole reason this gate exists is to run on real TF32 HW.
    if cc[0] < 8:
        fatal(
            f"compute capability {cc} < 8.0 — NO TF32 tensor cores (Turing/T4 falls back to "
            "FP32). This gate MUST run on an A40/A100 (cc>=8); a 'pass' here is meaningless."
        )
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        fatal("CUBLAS_WORKSPACE_CONFIG != ':4096:8' (set BEFORE first CUDA context).")

    # --- TF32 ON via the REAL production path (D14 mechanics) ---
    enforce_determinism(strict=True, numeric_mode="tf32")
    ps_tf32 = precision_state()
    print(f"[tf32_gate] tf32 precision_state={ps_tf32}")
    if not (ps_tf32["cuda_matmul_allow_tf32"] and ps_tf32["cudnn_allow_tf32"]
            and ps_tf32["float32_matmul_precision"] == "high"):
        fatal("(0) enforce_determinism(numeric_mode='tf32') did NOT set the TF32 flags.")
    if not ps_tf32["tf32_engaged"]:
        fatal("(0) tf32_engaged=False on cc>=8 — flags not taking effect.")

    # (a) raise-vs-run under strict + tf32
    try:
        g_tf32 = gemm(); c_tf32 = conv(); torch.cuda.synchronize()
        print("[tf32_gate] (a) matmul+conv RAN under strict+tf32 (no raise) — OK")
    except Exception as e:
        fatal(f"(a) RAISED under strict+tf32: {type(e).__name__}: {e}")

    # (b) run-to-run byte-identity (same enforce_determinism state)
    ok_m = torch.equal(g_tf32, gemm())
    ok_c = torch.equal(c_tf32, conv())
    print(f"[tf32_gate] (b) run-to-run byte-identical: gemm={ok_m} conv={ok_c}")
    if not (ok_m and ok_c):
        fatal("(b) TF32 NOT run-to-run byte-identical — determinism BROKEN.")

    # --- FP32 reference for contrast, via the REAL production path ---
    enforce_determinism(strict=True, numeric_mode="fp32")
    ps_fp32 = precision_state()
    print(f"[tf32_gate] fp32 precision_state={ps_fp32}")
    g_fp32 = gemm(); c_fp32 = conv()
    differ_g = not torch.equal(g_fp32, g_tf32)
    differ_c = not torch.equal(c_fp32, c_tf32)
    print(f"[tf32_gate] (c) TF32 differs from FP32: gemm={differ_g} conv={differ_c}  "
          f"max|dg|={(g_fp32 - g_tf32).abs().max().item():.3e} "
          f"max|dc|={(c_fp32 - c_tf32).abs().max().item():.3e}")
    if not (differ_g or differ_c):
        fatal("(c) TF32 did NOT change GEMM/conv on cc>=8 hardware — flags not engaging.")

    result = {
        "device": name,
        "compute_capability": f"{cc[0]}.{cc[1]}",
        "torch": torch.__version__,
        "raise_vs_run_ok": True,
        "tf32_run_to_run_byte_identical": bool(ok_m and ok_c),
        "tf32_differs_from_fp32": bool(differ_g or differ_c),
        "gemm_tf32": checksum(g_tf32),
        "gemm_fp32": checksum(g_fp32),
        "conv_tf32": checksum(c_tf32),
        "conv_fp32": checksum(c_fp32),
        "precision_state_tf32": ps_tf32,
        "precision_state_fp32": ps_fp32,
        "verdict": "PASS",
    }
    print(f"[tf32_gate] CHECKSUMS gemm_tf32={result['gemm_tf32']} gemm_fp32={result['gemm_fp32']} "
          f"conv_tf32={result['conv_tf32']} conv_fp32={result['conv_fp32']}")
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, sort_keys=True)
        print(f"[tf32_gate] wrote {out_path}")
    print("[tf32_gate] VERDICT: PASS — enforce_determinism(numeric_mode='tf32') is RUN-TO-RUN "
          "DETERMINISTIC on real TF32 hardware and DIFFERS from FP32; adopting TF32 = a NEW "
          "reference checksum (re-baseline), not run-to-run drift.")


if __name__ == "__main__":
    main()
