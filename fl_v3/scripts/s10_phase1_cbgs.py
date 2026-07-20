#!/usr/bin/env python3
"""Derive and optionally materialize the exact D_fit Phase-I CBGS artifact.

This is a data-identity tool, not an experiment.  It consumes only the accepted
train cache, STOP-A split manifest, and nuScenes annotation metadata bound by the
resolved Phase-I config.  It implements MIT's ``use_valid_flag=True`` semantics
before invoking the pinned ``CBGSDataset`` sampling algorithm.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
from typing import Any

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fl_v3.config import load_resolved_config, verify_physical_data_identities
from fl_v3.phase1_sampling import official_cbgs_artifact


_UNFROZEN_SHA256 = "0" * 64
_SPLIT_MANIFEST_SCHEMA = "fl_v3.s10.split_manifest.v1"
_SPLIT_ROLES = {"D_fit", "D_select", "D_audit", "D_low", "D_mid"}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _canonical_artifact_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _token_vector_sha256(tokens: list[str]) -> str:
    return hashlib.sha256(_canonical_artifact_bytes(tokens)).hexdigest()


def _ordered_unique_tokens(value: Any, where: str) -> list[str]:
    _require(isinstance(value, list), f"{where} must be a JSON list")
    _require(
        all(isinstance(token, str) and token for token in value),
        f"{where} contains invalid tokens",
    )
    tokens = list(value)
    _require(tokens == sorted(tokens), f"{where} must be sorted")
    _require(len(tokens) == len(set(tokens)), f"{where} contains duplicate tokens")
    return tokens


def _load_fit_role(config) -> dict[str, Any]:
    raw = config.as_dict()
    data = raw["data"]
    path = Path(data["split_manifest"]["path"])
    manifest = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(manifest, dict), "split manifest root must be an object")
    _require(
        manifest.get("schema") == _SPLIT_MANIFEST_SCHEMA,
        "split manifest schema drift",
    )
    _require(
        manifest.get("parent_version") == data["version"],
        "split parent version drift",
    )
    _require(manifest.get("parent_split") == "train", "split parent must remain train")
    sources = manifest.get("source_identities")
    _require(isinstance(sources, dict), "split source identities are missing")
    expected_sources = {
        "version": data["version"],
        "n_sweeps": data["cache_capacity_sweeps"],
        "train_cache_logical_sha256": data["cache"]["logical_sha256"],
        "train_cache_pickle_sha256": data["cache"]["pickle_sha256"],
        "train_cache_sidecar_sha256": data["cache"]["sidecar_sha256"],
        "zip_manifest_logical_sha256": data["zip_manifest"]["logical_sha256"],
        "zip_manifest_file_sha256": data["zip_manifest"]["file_sha256"],
    }
    for key, value in expected_sources.items():
        _require(sources.get(key) == value, f"split source identity {key!r} drift")
    roles = manifest.get("roles")
    _require(isinstance(roles, dict), "split roles are missing")
    _require(set(roles) == _SPLIT_ROLES, "split role registry drift")
    fit = roles.get("D_fit")
    _require(isinstance(fit, dict), "D_fit role is missing")
    _require(
        set(fit) == {"log_tokens", "scene_tokens", "sample_tokens"},
        "D_fit fields drift",
    )
    vectors = {
        kind: _ordered_unique_tokens(fit[f"{kind}_tokens"], f"D_fit.{kind}_tokens")
        for kind in ("log", "scene", "sample")
    }
    expected_role = data["roles"]["fit"]
    for kind in ("log", "scene", "sample"):
        tokens = vectors[kind]
        _require(len(tokens) == expected_role[f"{kind}s"], f"D_fit {kind} count drift")
        _require(
            _token_vector_sha256(tokens) == expected_role[f"{kind}_tokens_sha256"],
            f"D_fit {kind}-token identity drift",
        )
    return {
        "sample_tokens": vectors["sample"],
        "sample_tokens_sha256": _token_vector_sha256(vectors["sample"]),
    }


def _load_exact_fit_infos(config) -> tuple[dict[str, Any], list[dict]]:
    raw = config.as_dict()
    data = raw["data"]
    role = _load_fit_role(config)
    cache_path = Path(data["cache"]["path"])
    with cache_path.open("rb") as stream:
        blob = pickle.load(stream)
    _require(isinstance(blob, dict) and set(blob) == {"info_list", "meta"},
             "train cache pickle envelope drift")
    infos, meta = blob["info_list"], blob["meta"]
    _require(isinstance(infos, list) and isinstance(meta, dict),
             "train cache pickle types drift")
    sidecar = json.loads(Path(data["cache"]["sidecar_path"]).read_text(encoding="utf-8"))
    _require(sidecar == meta, "train cache sidecar differs from pickle metadata")
    expected_meta = {
        "format_version": data["cache"]["format"],
        "version": data["version"],
        "split": "train",
        "n_sweeps": int(data["cache_capacity_sweeps"]),
        "cache_hash": data["cache"]["logical_sha256"],
    }
    for key, value in expected_meta.items():
        _require(meta.get(key) == value, f"train cache metadata {key!r} drift")
    for index, info in enumerate(infos):
        _require(isinstance(info, dict), f"train cache record {index} is not an object")
        _require(
            info.get("_cache_n_sweeps") == data["cache_capacity_sweeps"],
            f"train cache record {index} sweep-capacity drift",
        )
        sweeps = info.get("lidar_sweeps")
        _require(isinstance(sweeps, list), f"train cache record {index} has no sweeps")
        _require(
            len(sweeps) <= data["cache_capacity_sweeps"] - 1,
            f"train cache record {index} exceeds sweep capacity",
        )
    by_token = {str(info["sample_token"]): info for info in infos}
    _require(len(by_token) == len(infos), "train cache contains duplicate sample tokens")
    missing = [token for token in role["sample_tokens"] if token not in by_token]
    _require(not missing, f"D_fit tokens are missing from the frozen train cache: {missing[:3]}")
    return role, [by_token[token] for token in role["sample_tokens"]]


def _reference_classes_and_validity(config, infos: list[dict]) -> tuple[list[list[int]], str]:
    raw = config.as_dict()
    sampling = raw["sampling"]
    eligibility = sampling["eligibility"]
    metadata_path = Path(eligibility["metadata_path"])
    _require(metadata_path.is_file(), f"annotation metadata is missing: {metadata_path}")
    _require(_sha256_file(metadata_path) == eligibility["metadata_sha256"],
             "sample_annotation.json physical identity drift")
    annotations = json.loads(metadata_path.read_text(encoding="utf-8"))
    _require(isinstance(annotations, list), "sample_annotation.json root must be a list")
    by_ann: dict[str, tuple[int, int]] = {}
    for annotation in annotations:
        token = str(annotation["token"])
        _require(token not in by_ann, "sample_annotation.json has duplicate tokens")
        by_ann[token] = (
            int(annotation["num_lidar_pts"]),
            int(annotation["num_radar_pts"]),
        )

    reference_names = tuple(raw["taxonomy"]["reference_object_classes"])
    reference_id = {name: index for index, name in enumerate(reference_names)}
    per_sample: list[list[int]] = []
    validity_vector: list[list[bool]] = []
    for info in infos:
        ann_tokens = [str(token) for token in info["gt_ann_tokens"]]
        names = [str(name) for name in info["gt_names"]]
        lidar_counts = np.asarray(info["gt_num_lidar_pts"], dtype=np.int64).tolist()
        _require(len(ann_tokens) == len(names) == len(lidar_counts),
                 "cache annotation arrays are not row-aligned")
        valid: list[bool] = []
        present: set[int] = set()
        for ann_token, name, cached_lidar in zip(ann_tokens, names, lidar_counts):
            _require(ann_token in by_ann, f"cache annotation token is absent: {ann_token}")
            lidar_points, radar_points = by_ann[ann_token]
            _require(int(cached_lidar) == lidar_points,
                     f"cache/metadata num_lidar_pts drift for {ann_token}")
            is_valid = lidar_points + radar_points > 0
            valid.append(is_valid)
            if is_valid and name in reference_id:
                present.add(reference_id[name])
        validity_vector.append(valid)
        per_sample.append(sorted(present))
    validity_sha = hashlib.sha256(_canonical_artifact_bytes(validity_vector)).hexdigest()
    _require(
        validity_sha == eligibility["ordered_D_fit_annotation_validity_sha256"],
        "D_fit ordered annotation-validity identity drift",
    )
    return per_sample, validity_sha


def derive(config_path: str) -> tuple[dict[str, Any], bytes, str]:
    config = load_resolved_config(config_path)
    _require(config.is_phase1, "CBGS derivation requires schema s10.phase1.v1")
    verify_physical_data_identities(config)
    raw = config.as_dict()
    role, infos = _load_exact_fit_infos(config)
    per_sample, validity_sha = _reference_classes_and_validity(config, infos)
    sampling = raw["sampling"]
    artifact = official_cbgs_artifact(
        contract={
            "plan_sha": raw["contract"]["plan_sha"],
            "request_commit": raw["contract"]["request_commit"],
            "split_manifest_sha256": raw["data"]["split_manifest"]["sha256"],
            "train_cache_logical_sha256": raw["data"]["cache"]["logical_sha256"],
            "D_fit_sample_tokens_sha256": role["sample_tokens_sha256"],
        },
        sample_tokens=role["sample_tokens"],
        per_sample_reference_classes=per_sample,
        class_names=raw["taxonomy"]["reference_object_classes"],
        seed=int(raw["contract"]["seed"]),
        epochs=int(raw["training"]["epochs"]),
        effective_batch=int(raw["training"]["effective_global_batch"]),
        eligibility={**sampling["eligibility"], "observed_validity_sha256": validity_sha},
    )
    expected = {
        key: sampling[key]
        for key in (
            "class_pool_sizes", "duplicated_class_memberships", "target_class_fraction",
            "segment_sizes", "expanded_length", "source_sample_order_sha256",
            "expanded_indices_sha256", "expanded_tokens_sha256", "class_segments_sha256",
            "twenty_epoch_order_sha256", "twenty_epoch_remainder_sha256",
        )
    }
    expected.update({
        key: raw["training"][key]
        for key in (
            "consumed_samples_per_epoch", "dropped_samples_per_epoch",
            "optimizer_updates_per_epoch", "max_optimizer_updates",
        )
    })
    observed = {key: artifact[key] for key in expected}
    _require(observed == expected, f"derived official-CBGS identity drift: {observed}")
    encoded = _canonical_artifact_bytes(artifact)
    digest = hashlib.sha256(encoded).hexdigest()
    frozen_digest = sampling["artifact"]["sha256"]
    if frozen_digest != _UNFROZEN_SHA256:
        _require(digest == frozen_digest, "official-CBGS artifact physical identity drift")
    return artifact, encoded, digest


def _materialize(path: Path, encoded: bytes) -> None:
    path = path.resolve()
    if path.exists():
        raise FileExistsError(f"CBGS artifact output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--materialize", action="store_true")
    args = parser.parse_args()
    artifact, encoded, digest = derive(args.config)
    config = load_resolved_config(args.config).as_dict()
    output = Path(config["sampling"]["artifact"]["path"])
    if args.materialize:
        _require(
            config["sampling"]["artifact"]["sha256"] != _UNFROZEN_SHA256,
            "refuse to materialize before the artifact SHA-256 is frozen",
        )
        _materialize(output, encoded)
    report = {
        "artifact_path": str(output),
        "artifact_sha256": digest,
        "materialized": bool(args.materialize),
        "expanded_length": artifact["expanded_length"],
        "consumed_samples_per_epoch": artifact["consumed_samples_per_epoch"],
        "dropped_samples_per_epoch": artifact["dropped_samples_per_epoch"],
        "optimizer_updates_per_epoch": artifact["optimizer_updates_per_epoch"],
        "max_optimizer_updates": artifact["max_optimizer_updates"],
    }
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
