"""Pure-NumPy S10 Phase-I official-CBGS identity implementation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


OFFICIAL_CBGS_SCHEMA = "s10.phase1.official_cbgs.v1"


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    """SHA-256 of the canonical JSON representation used by S10 identities."""
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def load_official_cbgs_artifact(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_sampling: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load one sealed official-CBGS artifact and verify its frozen identities."""
    artifact_path = Path(path)
    encoded = artifact_path.read_bytes()
    actual_sha256 = hashlib.sha256(encoded).hexdigest()
    if actual_sha256 != str(expected_sha256):
        raise ValueError(
            "official-CBGS artifact physical identity drift: "
            f"expected={expected_sha256}, actual={actual_sha256}"
        )
    try:
        artifact = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("official-CBGS artifact is not canonical JSON") from exc
    if not isinstance(artifact, dict) or artifact.get("schema") != OFFICIAL_CBGS_SCHEMA:
        raise ValueError("official-CBGS artifact schema drift")
    if _canonical_json_bytes(artifact) != encoded:
        raise ValueError("official-CBGS artifact is not canonically encoded")
    expanded = artifact.get("expanded_indices")
    if not isinstance(expanded, list) or not all(
        isinstance(value, int) and not isinstance(value, bool) for value in expanded
    ):
        raise ValueError("official-CBGS expanded_indices is invalid")
    if len(expanded) != artifact.get("expanded_length"):
        raise ValueError("official-CBGS expanded length drift")
    if canonical_json_sha256(expanded) != artifact.get("expanded_indices_sha256"):
        raise ValueError("official-CBGS expanded-index identity drift")
    if expected_sampling is not None:
        for key in (
            "source_sample_order_sha256",
            "class_pool_sizes",
            "duplicated_class_memberships",
            "target_class_fraction",
            "segment_sizes",
            "expanded_length",
            "expanded_indices_sha256",
            "expanded_tokens_sha256",
            "class_segments_sha256",
            "twenty_epoch_order_sha256",
            "twenty_epoch_remainder_sha256",
        ):
            if artifact.get(key) != expected_sampling.get(key):
                raise ValueError(f"official-CBGS artifact field {key!r} drift")
    return artifact


def build_official_cbgs_indices(
    per_sample_reference_classes: Sequence[Sequence[int] | np.ndarray],
    *,
    class_names: Sequence[str],
    seed: int,
) -> tuple[np.ndarray, tuple[np.ndarray, ...], dict[str, Any]]:
    """Reproduce the pinned MIT ``CBGSDataset._get_sample_indices`` exactly.

    Input classes already use the reference ``object_classes`` label space and
    ``use_valid_flag=True`` eligibility.  Sampling uses one MT19937
    ``RandomState`` and replacement in reference class order.
    """
    names = tuple(str(name) for name in class_names)
    if not names or len(names) != len(set(names)):
        raise ValueError("official CBGS class_names must be non-empty and unique")
    pools: list[list[int]] = [[] for _ in names]
    for sample_index, raw_classes in enumerate(per_sample_reference_classes):
        values = np.asarray(raw_classes, dtype=np.int64).reshape(-1)
        if values.size and (
            int(values.min()) < 0 or int(values.max()) >= len(names)
        ):
            raise ValueError("official CBGS sample contains an out-of-range class id")
        if values.size != np.unique(values).size:
            raise ValueError("official CBGS per-sample class ids must be unique")
        for class_id in values.tolist():
            pools[int(class_id)].append(int(sample_index))

    pool_sizes = [len(pool) for pool in pools]
    if any(size == 0 for size in pool_sizes):
        empty = [names[index] for index, size in enumerate(pool_sizes) if size == 0]
        raise ValueError(f"official CBGS cannot balance empty class pools: {empty}")
    duplicated = int(sum(pool_sizes))
    target_fraction = 1.0 / float(len(names))
    rng = np.random.RandomState(int(seed))
    segments: list[np.ndarray] = []
    ratios: list[float] = []
    for pool, size in zip(pools, pool_sizes):
        ratio = target_fraction / (float(size) / float(duplicated))
        # Preserve the archived expression and int truncation exactly.
        count = int(float(size) * ratio)
        segments.append(np.asarray(rng.choice(pool, count), dtype=np.int64))
        ratios.append(float(ratio))
    expanded = np.ascontiguousarray(np.concatenate(segments), dtype=np.int64)
    stats = {
        "class_pool_sizes": pool_sizes,
        "duplicated_class_memberships": duplicated,
        "target_class_fraction": target_fraction,
        "sampling_ratios": ratios,
        "segment_sizes": [int(segment.size) for segment in segments],
        "expanded_length": int(expanded.size),
    }
    return expanded, tuple(segments), stats


def official_cbgs_identities(
    *,
    sample_tokens: Sequence[str],
    segments: Sequence[np.ndarray],
    epochs: int,
    seed: int,
    effective_batch: int,
) -> dict[str, Any]:
    """Bind the expansion and the separate epoch order/remainder policy."""
    tokens = [str(token) for token in sample_tokens]
    if not tokens or len(tokens) != len(set(tokens)):
        raise ValueError("official CBGS source tokens must be non-empty and unique")
    normalized_segments = [
        np.asarray(segment, dtype=np.int64).reshape(-1) for segment in segments
    ]
    expanded = np.concatenate(normalized_segments)
    if expanded.size and (
        int(expanded.min()) < 0 or int(expanded.max()) >= len(tokens)
    ):
        raise ValueError("official CBGS expanded index leaves the D_fit token vector")
    if int(epochs) < 1 or int(effective_batch) < 1:
        raise ValueError("epochs and effective_batch must be positive")
    consumed = (int(expanded.size) // int(effective_batch)) * int(effective_batch)
    epoch_positions: list[list[int]] = []
    epoch_remainders: list[list[int]] = []
    for epoch in range(int(epochs)):
        permutation = np.random.RandomState(int(seed) + epoch).permutation(
            int(expanded.size)
        )
        epoch_positions.append(permutation[:consumed].tolist())
        epoch_remainders.append(permutation[consumed:].tolist())
    return {
        "source_sample_order_sha256": canonical_json_sha256(tokens),
        "expanded_indices_sha256": canonical_json_sha256(expanded.tolist()),
        "expanded_tokens_sha256": canonical_json_sha256(
            [tokens[int(index)] for index in expanded]
        ),
        "class_segments_sha256": canonical_json_sha256(
            [segment.tolist() for segment in normalized_segments]
        ),
        "twenty_epoch_order_sha256": canonical_json_sha256(epoch_positions),
        "twenty_epoch_remainder_sha256": canonical_json_sha256(epoch_remainders),
        "consumed_samples_per_epoch": consumed,
        "dropped_samples_per_epoch": int(expanded.size) - consumed,
        "optimizer_updates_per_epoch": consumed // int(effective_batch),
        "max_optimizer_updates": consumed // int(effective_batch) * int(epochs),
    }


def official_cbgs_artifact(
    *,
    contract: Mapping[str, Any],
    sample_tokens: Sequence[str],
    per_sample_reference_classes: Sequence[Sequence[int] | np.ndarray],
    class_names: Sequence[str],
    seed: int,
    epochs: int,
    effective_batch: int,
    eligibility: Mapping[str, Any],
) -> dict[str, Any]:
    """Construct the compact deterministic artifact consumed by Phase-I runs."""
    expanded, segments, stats = build_official_cbgs_indices(
        per_sample_reference_classes,
        class_names=class_names,
        seed=seed,
    )
    identities = official_cbgs_identities(
        sample_tokens=sample_tokens,
        segments=segments,
        epochs=epochs,
        seed=seed,
        effective_batch=effective_batch,
    )
    return {
        "schema": OFFICIAL_CBGS_SCHEMA,
        "contract": dict(contract),
        "algorithm": {
            "reference": "mit-han-lab/bevfusion:CBGSDataset._get_sample_indices",
            "rng": "numpy.random.RandomState(MT19937)",
            "seed": int(seed),
            "choice_replace": True,
            "class_order": [str(name) for name in class_names],
        },
        "eligibility": dict(eligibility),
        "source_sample_count": len(sample_tokens),
        "source_sample_order_sha256": identities["source_sample_order_sha256"],
        **stats,
        **{
            key: value
            for key, value in identities.items()
            if key != "source_sample_order_sha256"
        },
        "expanded_indices": expanded.tolist(),
    }
