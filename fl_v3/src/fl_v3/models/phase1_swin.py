"""Strict original-Swin -> torchvision mapping for S10 Phase I.

The owner-approved artifact is the Microsoft Swin release checkpoint used by
the pinned MIT BEVFusion YAML, not torchvision's separately trained object.
This module maps every feature-extractor parameter by name and shape, verifies
architecture-derived relative-position indices, and leaves only the three
MMDetection-style output LayerNorms at their specified identity initialization.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping

import torch
import torch.nn as nn


SWIN_SOURCE_URL = (
    "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/"
    "swin_tiny_patch4_window7_224.pth"
)
SWIN_LICENSE = "MIT"
SWIN_EXPECTED_PHYSICAL_SHA256 = (
    "9f71c168d837d1b99dd1dc29e14990a7a9e8bdc5f673d46b04fe36fe15590ad3"
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def tensor_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Content-address an ordered tensor state without pickle serialization."""
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        if not torch.is_tensor(value):
            raise TypeError(f"state value {name!r} is not a tensor")
        tensor = value.detach().contiguous().cpu()
        header = {
            "name": name,
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
        }
        encoded = _canonical_bytes(header)
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        # ``state_dict`` contains scalar integer buffers such as BatchNorm's
        # ``num_batches_tracked``.  PyTorch 2.11 refuses a 0-D cross-element-size
        # ``view(torch.uint8)``; NumPy's raw contiguous byte export covers both
        # scalar and non-scalar tensors with the same bytes and no value cast.
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


_BLOCK = re.compile(r"^layers\.(\d+)\.blocks\.(\d+)\.(.+)$")
_DOWNSAMPLE = re.compile(r"^layers\.(\d+)\.downsample\.(.+)$")


def original_swin_destination_key(source_key: str) -> str | None:
    """Return the CameraBackbone destination key or ``None`` if non-feature."""
    if source_key.startswith("patch_embed.proj."):
        return "_swin_features.0.0." + source_key.removeprefix("patch_embed.proj.")
    if source_key.startswith("patch_embed.norm."):
        return "_swin_features.0.2." + source_key.removeprefix("patch_embed.norm.")
    block = _BLOCK.match(source_key)
    if block:
        stage, block_index, suffix = block.groups()
        feature_index = 2 * int(stage) + 1
        if suffix.startswith("mlp.fc1."):
            suffix = "mlp.0." + suffix.removeprefix("mlp.fc1.")
        elif suffix.startswith("mlp.fc2."):
            suffix = "mlp.3." + suffix.removeprefix("mlp.fc2.")
        elif suffix.endswith("attn_mask"):
            return None
        return f"_swin_features.{feature_index}.{block_index}.{suffix}"
    downsample = _DOWNSAMPLE.match(source_key)
    if downsample:
        stage, suffix = downsample.groups()
        if int(stage) >= 3:
            raise ValueError("original Swin checkpoint has an impossible stage-3 downsample")
        return f"_swin_features.{2 * int(stage) + 2}.{suffix}"
    if source_key.startswith("head.") or source_key.startswith("norm."):
        return None
    return None


def _allowed_ignored_source(key: str) -> bool:
    return (
        key.startswith("head.")
        or key.startswith("norm.")
        or bool(re.match(r"^layers\.\d+\.blocks\.\d+\.attn_mask$", key))
    )


def _extract_model_state(payload: Any) -> tuple[dict[str, torch.Tensor], list[str]]:
    if not isinstance(payload, Mapping):
        raise RuntimeError("Swin checkpoint top level must be a mapping")
    top_keys = sorted(str(key) for key in payload)
    if "model" not in payload:
        raise RuntimeError("Swin release checkpoint must contain top-level 'model'")
    state = payload["model"]
    if not isinstance(state, Mapping) or not state:
        raise RuntimeError("Swin checkpoint model state must be a non-empty mapping")
    result: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if not isinstance(key, str) or not torch.is_tensor(value):
            raise RuntimeError("Swin model state must map string names to tensors")
        normalized = key.removeprefix("module.")
        if normalized in result:
            raise RuntimeError(f"duplicate normalized Swin source key {normalized!r}")
        result[normalized] = value
    return result, top_keys


def map_original_swin_state(
    backbone: nn.Module,
    source_state: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Build a strict load state and a complete mapping report payload."""
    destination_state = backbone.state_dict()
    destination_parameters = {name for name, _ in backbone.named_parameters()}
    expected_fresh = {
        "stage_output_norms.1.weight",
        "stage_output_norms.1.bias",
        "stage_output_norms.2.weight",
        "stage_output_norms.2.bias",
        "stage_output_norms.3.weight",
        "stage_output_norms.3.bias",
    }
    if not expected_fresh.issubset(destination_parameters):
        raise RuntimeError("Camera backbone does not expose the three reference output norms")

    mapped: dict[str, torch.Tensor] = {}
    rows: list[dict[str, Any]] = []
    ignored: list[str] = []
    unknown: list[str] = []
    for source_key in sorted(source_state):
        value = source_state[source_key]
        destination_key = original_swin_destination_key(source_key)
        if destination_key is None:
            if _allowed_ignored_source(source_key):
                ignored.append(source_key)
            else:
                unknown.append(source_key)
            continue
        if destination_key not in destination_state:
            unknown.append(source_key)
            continue
        expected = destination_state[destination_key]
        candidate = value.detach().cpu()
        if source_key.endswith("relative_position_index"):
            if candidate.numel() != expected.numel():
                raise RuntimeError(
                    f"relative-position index size mismatch for {source_key!r}"
                )
            candidate = candidate.reshape(expected.shape)
            if not torch.equal(candidate.to(expected.dtype), expected.detach().cpu()):
                raise RuntimeError(
                    f"relative-position index values differ for {source_key!r}"
                )
        if tuple(candidate.shape) != tuple(expected.shape):
            raise RuntimeError(
                f"Swin tensor shape mismatch {source_key!r}->{destination_key!r}: "
                f"{tuple(candidate.shape)} != {tuple(expected.shape)}"
            )
        if candidate.dtype != expected.dtype:
            raise RuntimeError(
                f"Swin tensor dtype mismatch {source_key!r}->{destination_key!r}: "
                f"{candidate.dtype} != {expected.dtype}"
            )
        if destination_key in mapped:
            raise RuntimeError(f"two Swin source tensors map to {destination_key!r}")
        mapped[destination_key] = candidate
        rows.append(
            {
                "source": source_key,
                "destination": destination_key,
                "shape": list(candidate.shape),
                "dtype": str(candidate.dtype),
                "sha256": tensor_state_sha256({destination_key: candidate}),
            }
        )

    if unknown:
        raise RuntimeError(f"unrecognized Swin source tensor keys: {unknown}")
    expected_loaded = destination_parameters - expected_fresh
    missing_parameters = sorted(expected_loaded - set(mapped))
    unexpected_destinations = sorted(set(mapped) - set(destination_state))
    if missing_parameters or unexpected_destinations:
        raise RuntimeError(
            "strict Swin destination coverage failed: "
            f"missing={missing_parameters}, unexpected={unexpected_destinations}"
        )
    for name in sorted(expected_fresh):
        tensor = destination_state[name]
        expected = torch.ones_like(tensor) if name.endswith("weight") else torch.zeros_like(tensor)
        if not torch.equal(tensor, expected):
            raise RuntimeError(f"reference output norm {name!r} is not identity initialized")

    report = {
        "source_tensor_count": len(source_state),
        "mapped_tensor_count": len(mapped),
        "mapped_parameter_count": len(expected_loaded),
        "ignored_source_keys": ignored,
        "fresh_destination_parameters": sorted(expected_fresh),
        "missing_destination_parameters": missing_parameters,
        "unexpected_destination_keys": unexpected_destinations,
        "mapping": rows,
        "mapped_state_sha256": tensor_state_sha256(mapped),
    }
    return mapped, report


def validate_and_load_original_swin(
    model: nn.Module,
    checkpoint_path: str | Path,
    *,
    expected_physical_sha256: str = SWIN_EXPECTED_PHYSICAL_SHA256,
) -> dict[str, Any]:
    """Validate release bytes and initialize ``model.camera_backbone`` strictly."""
    path = Path(checkpoint_path).resolve()
    physical_sha = sha256_file(path)
    if physical_sha != expected_physical_sha256:
        raise RuntimeError(
            f"Swin physical SHA mismatch: {physical_sha} != {expected_physical_sha256}"
        )
    if not hasattr(model, "camera_backbone"):
        raise TypeError("Phase-I Camera model must expose camera_backbone")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    source_state, top_keys = _extract_model_state(payload)
    mapped, mapping = map_original_swin_state(model.camera_backbone, source_state)
    incompatible = model.camera_backbone.load_state_dict(mapped, strict=False)
    expected_missing = mapping["fresh_destination_parameters"]
    if sorted(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "Swin load_state_dict report differs from strict mapping: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    return {
        "schema": "s10.phase1.swin-initialization.v1",
        "source_url": SWIN_SOURCE_URL,
        "license": SWIN_LICENSE,
        "physical_path": str(path),
        "physical_bytes": path.stat().st_size,
        "physical_sha256": physical_sha,
        "checkpoint_top_level_keys": top_keys,
        "loaded_missing_keys": list(incompatible.missing_keys),
        "loaded_unexpected_keys": list(incompatible.unexpected_keys),
        **mapping,
        "initialization_state_sha256": tensor_state_sha256(model.state_dict()),
    }


def seal_validated_swin_checkpoint(
    model: nn.Module,
    *,
    quarantine_path: str | Path,
    final_path: str | Path,
    mapping_report_path: str | Path,
    acquisition_redirect_host: str,
    acquisition_redirect_path: str,
) -> dict[str, Any]:
    """Validate quarantine, write immutable report, then atomically promote bytes."""
    quarantine = Path(quarantine_path).resolve()
    final = Path(final_path).resolve()
    report_path = Path(mapping_report_path).resolve()
    if final.exists():
        raise FileExistsError(f"final Swin checkpoint already exists: {final}")
    if not quarantine.is_file():
        raise FileNotFoundError(f"quarantined Swin checkpoint is missing: {quarantine}")
    if acquisition_redirect_host != "release-assets.githubusercontent.com":
        raise RuntimeError("Swin acquisition redirect host is outside the allowlist")
    report = validate_and_load_original_swin(model, quarantine)
    report["validated_quarantine_path"] = str(quarantine)
    report["physical_path"] = str(final)
    report["acquisition"] = {
        "count": 1,
        "http_status": 200,
        "redirect_host": acquisition_redirect_host,
        "redirect_path": acquisition_redirect_path,
        "query_recording": "redacted_ephemeral_signed_query",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_name(report_path.name + ".partial")
    if report_path.exists() or temporary.exists():
        raise FileExistsError("Swin mapping report path is not fresh")
    encoded = _canonical_bytes(report) + b"\n"
    with temporary.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(temporary, 0o400)
    os.replace(temporary, report_path)
    report_sha = sha256_file(report_path)
    # Only now are the downloaded bytes accepted for model use.
    os.replace(quarantine, final)
    os.chmod(final, 0o400)
    return {
        **report,
        "mapping_report_path": str(report_path),
        "mapping_report_sha256": report_sha,
    }


__all__ = [
    "SWIN_EXPECTED_PHYSICAL_SHA256",
    "SWIN_LICENSE",
    "SWIN_SOURCE_URL",
    "map_original_swin_state",
    "original_swin_destination_key",
    "seal_validated_swin_checkpoint",
    "sha256_file",
    "tensor_state_sha256",
    "validate_and_load_original_swin",
]
