"""Build the nuScenes info-cache once without extracting sensor blobs.

The detection task's ``client_data`` loads this cache and **raises if it is absent**
(it must never build the devkit during a training/compute run). Run this once where
the devkit + dataset are reachable. On Arrhenius, submit through a GH200 job after
activating ``fl_v3/scripts/arrhenius_env.sh``:

    python fl_v3/scripts/build_nuscenes_cache.py \
        --dataroot "$ARRHENIUS_NUSCENES_DATAROOT"
    # trainval (heavier; metadata + geometry build):
    python fl_v3/scripts/build_nuscenes_cache.py \
        --dataroot "$ARRHENIUS_NUSCENES_DATAROOT" \
        --version v1.0-trainval --splits train val
    # MULTI-SWEEP cache (MCR P1 lever). t1.v2 binds depth in filename/meta/records;
    # a dedicated directory remains recommended for immutable run provenance:
    python fl_v3/scripts/build_nuscenes_cache.py \
        --dataroot "$ARRHENIUS_NUSCENES_DATAROOT" \
        --version v1.0-trainval --splits train val --n-sweeps 10 \
        --cache-dir ./fl_outputs/nuscenes/info_cache_msweep10

Mini is engineering smoke; trainval is the scientific scale (T3+).
"""
from __future__ import annotations

import argparse

from fl_v3.data.nuscenes import info_cache as IC
from fl_v3.data.nuscenes import paths as P


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--splits", nargs="+", default=["mini_train", "mini_val"])
    ap.add_argument("--cache-dir", default="./fl_outputs/nuscenes/info_cache")
    ap.add_argument("--dataroot", default="",
                    help="Extracted or Arrhenius module nuScenes root. Defaults to config-like "
                         "environment resolution, including NUSCENES_DATA_DIR.")
    ap.add_argument("--n-sweeps", type=int, default=1,
                    help="MCR P1: >1 accumulates prev LIDAR_TOP sweeps (+dt). Cache t1.v2 "
                         "binds this depth in its filename, metadata, and every record.")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    dataroot = args.dataroot or P.get_dataroot()
    dataset_report = P.verify_dataset(args.version, dataroot)
    print(
        f"[cache] source backend={dataset_report['blob_backend']} "
        f"table_dir={dataset_report['table_dir']} dataroot={dataroot}"
    )
    # The Arrhenius module uses trainval/ and test/ table directories instead of
    # the devkit's v1.0-trainval/ and v1.0-test/ names.  The factory overrides
    # only table_root; sensor filenames remain DATAROOT-relative and no blob is
    # opened or extracted while the info cache is built.
    nusc = P.create_nuscenes(args.version, dataroot, verbose=False)
    scale = "mini-smoke" if "mini" in args.version else "trainval-scientific"
    for split in args.splits:
        info_list, meta = IC.get_or_build_cache(
            nusc, args.cache_dir, args.version, split, scale, dataroot,
            rebuild=args.rebuild, n_sweeps=args.n_sweeps,
        )
        n_sw = sum(len(i.get("lidar_sweeps", [])) for i in info_list)
        print(f"[cache] {args.version}/{split}: n_samples={meta['n_samples']} "
              f"n_boxes={meta['n_boxes']} n_sweeps={args.n_sweeps} "
              f"(total prev-sweep records={n_sw}) hash={meta['cache_hash'][:12]}…")


if __name__ == "__main__":
    main()
