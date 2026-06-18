"""TF32 determinism re-confirmation gate — run on an A40 (or A100), NOT the login T4.

WHY THIS EXISTS: the login node is a Tesla T4 (Turing, cc 7.5) with NO TF32 tensor
cores, so on the login node `allow_tf32=True` silently falls back to true FP32 and
`high`==`highest` byte-for-byte. The DETERMINISM mechanics (raise-vs-run, run-to-run
byte-identity under the strict harness) were already confirmed on the T4; THIS gate
re-confirms them on hardware where TF32 actually engages, and proves that engaging
TF32 produces a DIFFERENT-but-internally-DETERMINISTIC result (a new reference, not
drift).

Run:  CUBLAS_WORKSPACE_CONFIG=:4096:8 bash fl_v3/scripts/run_in_venv.sh \
          python /path/to/tf32_det_gate_a40.py
(or wrap in an sbatch like run_det_gate_a40.sh and require an A40/A100 node)
"""
import hashlib
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

sys.path.insert(0, "fl_v3/src")
from fl_v3.utils.runtime import enforce_determinism, seed_everything


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
    if not torch.cuda.is_available():
        fatal("no CUDA — TF32 is a hardware feature; run on an A40/A100 node.")
    name = torch.cuda.get_device_name(0)
    cc = torch.cuda.get_device_capability(0)
    has_tf32_hw = cc[0] >= 8
    print(f"[tf32_gate] device={name} cc={cc} torch={torch.__version__} "
          f"tf32_tensor_cores={'YES' if has_tf32_hw else 'NO (Turing/T4 falls back to FP32)'}")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        fatal("CUBLAS_WORKSPACE_CONFIG != ':4096:8' (set BEFORE first CUDA context).")

    # --- TF32 ON, strict harness ---
    enforce_determinism(strict=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # (a) raise-vs-run
    try:
        g_tf32 = gemm(); c_tf32 = conv(); torch.cuda.synchronize()
        print("[tf32_gate] (a) matmul+conv RAN under strict+allow_tf32 (no raise) — OK")
    except Exception as e:
        fatal(f"(a) RAISED under strict+allow_tf32: {type(e).__name__}: {e}")

    # (b) run-to-run byte-identity
    ok_m = torch.equal(g_tf32, gemm())
    ok_c = torch.equal(c_tf32, conv())
    print(f"[tf32_gate] (b) run-to-run byte-identical: gemm={ok_m} conv={ok_c}")
    if not (ok_m and ok_c):
        fatal("(b) TF32 NOT run-to-run byte-identical — determinism BROKEN.")

    # --- FP32 reference for contrast ---
    enforce_determinism(strict=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    g_fp32 = gemm()
    differ = not torch.equal(g_fp32, g_tf32)
    print(f"[tf32_gate] (c) TF32 differs from FP32 (=> TF32 engaged): {differ}  "
          f"max|d|={(g_fp32 - g_tf32).abs().max().item():.3e}")
    if has_tf32_hw and not differ:
        fatal("(c) TF32 did NOT change the GEMM on cc>=8 hardware — flags not taking effect.")

    print(f"[tf32_gate] CHECKSUMS  gemm_fp32={checksum(g_fp32)}  gemm_tf32={checksum(g_tf32)}")
    print("[tf32_gate] VERDICT: TF32 is RUN-TO-RUN DETERMINISTIC under the strict harness; "
          "adopting it = a NEW reference checksum (re-baseline), not run-to-run drift.")


if __name__ == "__main__":
    main()
