"""Build the nuScenes info-cache on the LOGIN node (T2 setup; devkit-using).

The detection task's ``client_data`` loads this cache and **raises if it is absent**
(it must never build the devkit during a training/compute run). Run this once on the
login node (where the devkit + dataset are reachable):

    bash fl_v3/scripts/run_in_venv.sh python fl_v3/scripts/build_nuscenes_cache.py
    # trainval (heavier; metadata + geometry build):
    bash fl_v3/scripts/run_in_venv.sh python fl_v3/scripts/build_nuscenes_cache.py \
        --version v1.0-trainval --splits train val

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
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    P.verify_dataset(args.version)
    from nuscenes import NuScenes

    nusc = NuScenes(version=args.version, dataroot=P.DATAROOT, verbose=False)
    scale = "mini-smoke" if "mini" in args.version else "trainval-scientific"
    for split in args.splits:
        info_list, meta = IC.get_or_build_cache(
            nusc, args.cache_dir, args.version, split, scale, P.DATAROOT, rebuild=args.rebuild
        )
        print(f"[cache] {args.version}/{split}: n_samples={meta['n_samples']} "
              f"n_boxes={meta['n_boxes']} hash={meta['cache_hash'][:12]}…")


if __name__ == "__main__":
    main()
