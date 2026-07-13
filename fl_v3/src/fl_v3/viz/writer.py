"""Deterministic visualization artifact writer for clean perception runs.

Lays down the V1–V4 viz tree under a run directory and records a stable
manifest. The actual renderers (calibrated projections, feature heatmaps, BEV
overlays, and clean detections) live beside their owning components; this writer
provides the shared tree and manifest contract:

    <run_dir>/viz/{calibration,encoder,fusion,detection}/

Mapping to the plan's viz stages:
    calibration → V1   encoder → V2   fusion → V3
    detection   → V4

**Deterministic by construction:** stage dirs are fixed; artifact names are
caller-supplied (no timestamps / RNG in names); the manifest is written with
sorted keys and entries sorted by (stage, name); JSON dumps use ``sort_keys=True``.
Two runs that register the same artifacts produce an identical ``manifest.json``.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

# Clean visualization stages, in pipeline order.
VIZ_STAGES = ("calibration", "encoder", "fusion", "detection")
_STAGE_TO_V = {
    "calibration": "V1",
    "encoder": "V2",
    "fusion": "V3",
    "detection": "V4",
}


class VizWriter:
    """Creates and tracks the ``<run_dir>/viz/<stage>/`` tree + a manifest."""

    def __init__(self, run_dir: str) -> None:
        self.run_dir = run_dir
        self.viz_root = os.path.join(run_dir, "viz")
        # Track registered artifacts: {stage: [relative paths]}.
        self._artifacts: Dict[str, List[str]] = {s: [] for s in VIZ_STAGES}
        for stage in VIZ_STAGES:
            os.makedirs(os.path.join(self.viz_root, stage), exist_ok=True)

    def stage_dir(self, stage: str) -> str:
        self._check_stage(stage)
        return os.path.join(self.viz_root, stage)

    def _check_stage(self, stage: str) -> None:
        if stage not in VIZ_STAGES:
            raise ValueError(f"Unknown viz stage {stage!r}. Valid: {VIZ_STAGES}")

    def _register(self, stage: str, filename: str) -> str:
        rel = os.path.join("viz", stage, filename)
        if rel not in self._artifacts[stage]:
            self._artifacts[stage].append(rel)
        return os.path.join(self.run_dir, rel)

    def write_json(self, stage: str, name: str, obj) -> str:
        """Write a deterministic JSON artifact (sorted keys).

        Determinism holds for JSON-serializable, order-stable payloads (dicts,
        scalars, and lists whose order is intentional). ``sort_keys=True`` orders
        dict keys but NOT list elements; callers must pre-sort lists whose order
        is not semantic, and must not pass ``set``s (not JSON-serializable /
        iteration-order-unstable).
        """
        self._check_stage(stage)
        filename = name if name.endswith(".json") else f"{name}.json"
        path = self._register(stage, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, sort_keys=True)
        return path

    def figure_path(self, stage: str, name: str, ext: str = "png") -> str:
        """Reserve a deterministic path for a figure (renderer saves to it).

        The renderer (e.g. ``fig.savefig(path)``) lives in the owning task; this
        only fixes the filename + registers it in the manifest. No timestamp in
        the name, so the path is run-stable.
        """
        self._check_stage(stage)
        filename = name if name.endswith(f".{ext}") else f"{name}.{ext}"
        return self._register(stage, filename)

    def write_manifest(self) -> str:
        """Write ``<run_dir>/viz/manifest.json`` — stable, sorted."""
        manifest = {
            "stages": {
                stage: {
                    "viz_id": _STAGE_TO_V[stage],
                    "artifacts": sorted(self._artifacts[stage]),
                }
                for stage in VIZ_STAGES
            }
        }
        path = os.path.join(self.viz_root, "manifest.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
        return path
