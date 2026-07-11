#!/usr/bin/env python3
"""Build the S01 read-only member-to-archive manifest.

Scanning the ten shared trainval archives is material compute and requires an
owner-approved S01 RUN_REQUEST.  Synthetic/local invocations are suitable for
focused tests without that full-data authorization.
"""
from __future__ import annotations

import argparse
import json

from fl_v3.data.nuscenes import paths as P
from fl_v3.data.nuscenes.zip_backend import TRAINVAL_ARCHIVE_NAMES, build_zip_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", default="", help="Defaults through paths.get_dataroot().")
    parser.add_argument("--manifest", required=True, help="Output SQLite path outside DATAROOT.")
    parser.add_argument(
        "--archives",
        nargs="+",
        default=list(TRAINVAL_ARCHIVE_NAMES),
        help="Exact root-level archives to scan. Full gate defaults to trainval01..10; "
        "a bounded smoke may name one archive explicitly.",
    )
    parser.add_argument("--force", action="store_true", help="Atomically replace an existing manifest.")
    args = parser.parse_args()

    dataroot = args.dataroot or P.get_dataroot()
    P.resolve_writable(args.manifest, dataroot)
    report = build_zip_manifest(
        dataroot, args.manifest, archive_names=args.archives, force=args.force
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
