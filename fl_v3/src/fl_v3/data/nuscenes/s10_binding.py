"""Immutable S10 split-role and observation-panel bindings.

This module consumes the accepted STOP-A manifest; it never constructs or solves
another split.  Downstream stages bind the physical manifest SHA-256, its source
data identities, one declared role, and the exact ordered token vector.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from fl_v3.models.fusion.head import NUSCENES_CENTERHEAD_TASKS


SPLIT_MANIFEST_SCHEMA = "fl_v3.s10.split_manifest.v1"
OBSERVATION_PANEL_SCHEMA = "fl_v3.s10.observation_panel.v1"
SPLIT_ROLES = ("D_fit", "D_select", "D_audit", "D_low", "D_mid")
PANEL_DOMAIN = "fl_v3.s10.stop_b.panel.v1"


class S10BindingError(ValueError):
    """An immutable split or panel binding is absent, ambiguous, or changed."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def token_vector_sha256(tokens: Sequence[str]) -> str:
    """Hash an ordered token vector with an unambiguous canonical encoding."""
    return hashlib.sha256(canonical_json_bytes(list(tokens))).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise S10BindingError(message)


def _ordered_unique_strings(value: Any, where: str) -> tuple[str, ...]:
    _require(isinstance(value, list), f"{where} must be a JSON list")
    _require(all(isinstance(item, str) and item for item in value), f"{where} has invalid tokens")
    tokens = tuple(value)
    _require(len(tokens) == len(set(tokens)), f"{where} contains duplicate tokens")
    _require(tokens == tuple(sorted(tokens)), f"{where} must be sorted")
    return tokens


@dataclass(frozen=True)
class FrozenSplitRole:
    manifest_path: str
    manifest_sha256: str
    role: str
    parent_version: str
    parent_split: str
    source_identities: Mapping[str, Any]
    log_tokens: tuple[str, ...]
    scene_tokens: tuple[str, ...]
    sample_tokens: tuple[str, ...]
    log_tokens_sha256: str
    scene_tokens_sha256: str
    sample_tokens_sha256: str

    def identity(self) -> dict[str, Any]:
        return {
            "manifest_path": self.manifest_path,
            "manifest_sha256": self.manifest_sha256,
            "role": self.role,
            "parent_version": self.parent_version,
            "parent_split": self.parent_split,
            "source_identities": dict(self.source_identities),
            "log_count": len(self.log_tokens),
            "scene_count": len(self.scene_tokens),
            "sample_count": len(self.sample_tokens),
            "log_tokens_sha256": self.log_tokens_sha256,
            "scene_tokens_sha256": self.scene_tokens_sha256,
            "sample_tokens_sha256": self.sample_tokens_sha256,
        }


def load_frozen_split_role(
    manifest_path: str | Path,
    *,
    expected_manifest_sha256: str,
    role: str,
    expected_source_identities: Mapping[str, Any] | None = None,
) -> FrozenSplitRole:
    """Load one accepted role without permitting split reconstruction."""
    path = Path(manifest_path).resolve()
    _require(path.is_file(), f"split manifest is missing: {path}")
    actual_sha = sha256_file(path)
    _require(actual_sha == expected_manifest_sha256, "split manifest physical SHA-256 drift")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise S10BindingError(f"cannot read split manifest: {exc}") from exc
    _require(isinstance(manifest, dict), "split manifest root must be an object")
    _require(manifest.get("schema") == SPLIT_MANIFEST_SCHEMA, "split manifest schema drift")
    _require(manifest.get("parent_version") == "v1.0-trainval", "split parent version drift")
    _require(manifest.get("parent_split") == "train", "split parent must remain train")
    sources = manifest.get("source_identities")
    _require(isinstance(sources, dict), "split source identities are missing")
    if expected_source_identities is not None:
        for key, expected in expected_source_identities.items():
            _require(key in sources, f"split source identity {key!r} is missing")
            _require(sources[key] == expected, f"split source identity {key!r} drift")

    roles = manifest.get("roles")
    _require(isinstance(roles, dict), "split roles are missing")
    _require(set(roles) == set(SPLIT_ROLES), "split role registry drift")
    _require(role in SPLIT_ROLES, f"unknown S10 role {role!r}")
    vectors: dict[str, dict[str, tuple[str, ...]]] = {}
    for role_name in SPLIT_ROLES:
        record = roles[role_name]
        _require(isinstance(record, dict), f"roles.{role_name} must be an object")
        _require(
            set(record) == {"log_tokens", "scene_tokens", "sample_tokens"},
            f"roles.{role_name} fields drift",
        )
        vectors[role_name] = {
            kind: _ordered_unique_strings(record[f"{kind}_tokens"], f"roles.{role_name}.{kind}_tokens")
            for kind in ("log", "scene", "sample")
        }

    for kind in ("log", "scene", "sample"):
        fit = set(vectors["D_fit"][kind])
        select = set(vectors["D_select"][kind])
        audit = set(vectors["D_audit"][kind])
        low = set(vectors["D_low"][kind])
        mid = set(vectors["D_mid"][kind])
        _require(not (fit & select or fit & audit or select & audit), f"base-role {kind} overlap")
        _require(low <= mid <= fit, f"D_low/D_mid/D_fit {kind} nesting drift")

    selected = vectors[role]
    return FrozenSplitRole(
        manifest_path=str(path),
        manifest_sha256=actual_sha,
        role=role,
        parent_version=str(manifest["parent_version"]),
        parent_split=str(manifest["parent_split"]),
        source_identities=dict(sources),
        log_tokens=selected["log"],
        scene_tokens=selected["scene"],
        sample_tokens=selected["sample"],
        log_tokens_sha256=token_vector_sha256(selected["log"]),
        scene_tokens_sha256=token_vector_sha256(selected["scene"]),
        sample_tokens_sha256=token_vector_sha256(selected["sample"]),
    )


def _rank(token: str, suffix: str) -> str:
    payload = f"{PANEL_DOMAIN}\0{suffix}\0{token}".encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _largest_remainder_quotas(sizes: Mapping[str, int], total: int, minimum: int) -> dict[str, int]:
    logs = sorted(sizes)
    _require(logs, "panel quota input is empty")
    _require(all(sizes[log] >= minimum for log in logs), "a D_low log is smaller than the panel minimum")
    base = minimum * len(logs)
    _require(total >= base, "panel total is smaller than per-log minima")
    remaining = total - base
    weight_sum = sum(int(sizes[log]) for log in logs)
    raw = {log: Fraction(remaining * int(sizes[log]), weight_sum) for log in logs}
    quotas = {log: minimum + int(raw[log].numerator // raw[log].denominator) for log in logs}
    left = total - sum(quotas.values())
    order = sorted(logs, key=lambda log: (-(raw[log] - int(raw[log])), log))
    for log in order[:left]:
        quotas[log] += 1
    _require(sum(quotas.values()) == total, "largest-remainder quota arithmetic drift")
    _require(all(quotas[log] <= sizes[log] for log in logs), "panel quota exceeds a log population")
    return quotas


def _target_support(info: Mapping[str, Any], *, x_min: float, x_max: float, y_min: float, y_max: float) -> dict[str, Any]:
    boxes = np.asarray(info["gt_boxes"])
    labels = np.asarray(info["gt_labels"], dtype=np.int64)
    _require(boxes.ndim == 2 and boxes.shape[1] >= 2, "cache GT boxes have invalid shape")
    _require(labels.shape == (boxes.shape[0],), "cache GT labels have invalid shape")
    valid = (
        np.isfinite(boxes[:, :2]).all(axis=1)
        & (boxes[:, 0] >= x_min)
        & (boxes[:, 0] < x_max)
        & (boxes[:, 1] >= y_min)
        & (boxes[:, 1] < y_max)
        & (labels >= 0)
        & (labels < 10)
    )
    labels_valid = labels[valid]
    task_ids = []
    from fl_v3.models.fusion.head import NUSCENES_DETECTION_NAMES

    by_name = {name: index for index, name in enumerate(NUSCENES_DETECTION_NAMES)}
    counts = []
    for task in NUSCENES_CENTERHEAD_TASKS:
        ids = np.asarray([by_name[name] for name in task], dtype=np.int64)
        count = int(np.isin(labels_valid, ids).sum())
        counts.append(count)
        task_ids.append([int(value) for value in ids.tolist()])
    return {
        "input_gt": int(labels.shape[0]),
        "in_range_gt": int(valid.sum()),
        "task_counts": counts,
        "task_positive": [count > 0 for count in counts],
        "task_global_ids": task_ids,
    }


def build_stop_b_panel(
    binding: FrozenSplitRole,
    info_list: Sequence[Mapping[str, Any]],
    *,
    x_min: float = -54.0,
    x_max: float = 54.0,
    y_min: float = -54.0,
    y_max: float = 54.0,
) -> dict[str, Any]:
    """Create the one-shot 48-core/16-term panel from accepted ``D_low``."""
    _require(binding.role == "D_low", "STOP-B panel construction requires role D_low")
    by_token = {str(info["sample_token"]): info for info in info_list}
    _require(len(by_token) == len(info_list), "train cache contains duplicate sample tokens")
    missing = [token for token in binding.sample_tokens if token not in by_token]
    _require(not missing, f"D_low has {len(missing)} tokens absent from the train cache")

    by_log: dict[str, list[str]] = {token: [] for token in binding.log_tokens}
    support: dict[str, dict[str, Any]] = {}
    for token in binding.sample_tokens:
        info = by_token[token]
        log = str(info["log_token"])
        _require(log in by_log, f"D_low sample {token} has out-of-role log {log}")
        by_log[log].append(token)
        support[token] = _target_support(
            info, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max
        )
    _require(sum(len(tokens) for tokens in by_log.values()) == len(binding.sample_tokens), "D_low log partition drift")

    quotas = _largest_remainder_quotas({log: len(tokens) for log, tokens in by_log.items()}, 48, 4)
    core: list[str] = []
    for log in sorted(by_log):
        ranked = sorted(by_log[log], key=lambda token: (_rank(token, "core"), token))
        core.extend(ranked[: quotas[log]])
    core = sorted(core, key=lambda token: (_rank(token, "core-order"), token))
    _require(len(core) == 48 and len(set(core)) == 48, "P_core cardinality drift")

    available = [token for token in binding.sample_tokens if token not in set(core)]
    task_frame_counts = [0] * len(NUSCENES_CENTERHEAD_TASKS)
    term: list[str] = []
    while len(term) < 16:
        candidates = [token for token in available if token not in set(term)]
        _require(candidates, "P_term candidate population exhausted")

        def candidate_key(token: str):
            positive = support[token]["task_positive"]
            gain = sum(
                int(bool(positive[index]) and task_frame_counts[index] < 4)
                for index in range(len(task_frame_counts))
            )
            return (-gain, _rank(token, "term"), token)

        chosen = min(candidates, key=candidate_key)
        term.append(chosen)
        for index, positive in enumerate(support[chosen]["task_positive"]):
            task_frame_counts[index] += int(bool(positive))
    _require(all(value >= 4 for value in task_frame_counts), "P_term cannot satisfy four positive frames per task")
    _require(not set(core) & set(term), "P_core/P_term overlap")

    batches = {
        "P_core": [core[index : index + 4] for index in range(0, len(core), 4)],
        "P_term": [term[index : index + 4] for index in range(0, len(term), 4)],
    }
    record = {
        "schema": OBSERVATION_PANEL_SCHEMA,
        "selection_policy": {
            "model_output_observed_before_freeze": False,
            "reroll_allowed": False,
            "seed": None,
            "rank_domain": PANEL_DOMAIN,
            "P_core": "48 samples; minimum four per D_low log; remaining quotas by exact largest remainder over log populations; fixed hash rank",
            "P_term": "16 samples disjoint from P_core; deterministic greedy to at least four positive frames per CenterHead task; fixed hash tie-break",
        },
        "split_binding": binding.identity(),
        "bev_target_range_xyxy": [float(x_min), float(y_min), float(x_max), float(y_max)],
        "centerhead_tasks": [list(task) for task in NUSCENES_CENTERHEAD_TASKS],
        "core_log_quotas": quotas,
        "task_positive_frames_in_P_term": task_frame_counts,
        "tokens": {"P_core": core, "P_term": term},
        "token_sha256": {
            "P_core": token_vector_sha256(core),
            "P_term": token_vector_sha256(term),
            "P_broad": token_vector_sha256([*core, *term]),
        },
        "batches_b4": batches,
        "sample_support": {
            token: {
                "log_token": str(by_token[token]["log_token"]),
                **support[token],
            }
            for token in [*core, *term]
        },
    }
    record["panel_sha256"] = hashlib.sha256(canonical_json_bytes(record)).hexdigest()
    return record


def validate_stop_b_panel(panel: Mapping[str, Any], binding: FrozenSplitRole) -> dict[str, Any]:
    """Fail-closed reload check for a frozen STOP-B panel."""
    _require(panel.get("schema") == OBSERVATION_PANEL_SCHEMA, "observation panel schema drift")
    stored_sha = panel.get("panel_sha256")
    payload = dict(panel)
    payload.pop("panel_sha256", None)
    _require(stored_sha == hashlib.sha256(canonical_json_bytes(payload)).hexdigest(), "panel content SHA drift")
    split = panel.get("split_binding")
    _require(isinstance(split, dict), "panel split binding missing")
    _require(split == binding.identity(), "panel/split binding drift")
    tokens = panel.get("tokens")
    _require(isinstance(tokens, dict) and set(tokens) == {"P_core", "P_term"}, "panel token registry drift")
    core = tuple(tokens["P_core"])
    term = tuple(tokens["P_term"])
    _require(len(core) == 48 and len(term) == 16, "panel cardinality drift")
    _require(len(set(core)) == 48 and len(set(term)) == 16, "panel duplicate token")
    _require(not set(core) & set(term), "panel strata overlap")
    _require(set(core) | set(term) <= set(binding.sample_tokens), "panel token lies outside D_low")
    expected_hashes = {
        "P_core": token_vector_sha256(core),
        "P_term": token_vector_sha256(term),
        "P_broad": token_vector_sha256([*core, *term]),
    }
    _require(panel.get("token_sha256") == expected_hashes, "panel token-vector identity drift")
    batches = panel.get("batches_b4")
    _require(isinstance(batches, dict), "panel B4 batches missing")
    _require(batches.get("P_core") == [list(core[index:index + 4]) for index in range(0, 48, 4)], "P_core B4 order drift")
    _require(batches.get("P_term") == [list(term[index:index + 4]) for index in range(0, 16, 4)], "P_term B4 order drift")
    return {
        "status": "PASS",
        "panel_sha256": stored_sha,
        "P_core_samples": 48,
        "P_term_samples": 16,
        "P_broad_samples": 64,
        "P_core_batches_b4": 12,
        "P_term_batches_b4": 4,
    }


def write_canonical_json(path: str | Path, value: Any) -> None:
    Path(path).write_bytes(canonical_json_bytes(value) + b"\n")


__all__ = [
    "FrozenSplitRole",
    "OBSERVATION_PANEL_SCHEMA",
    "S10BindingError",
    "build_stop_b_panel",
    "canonical_json_bytes",
    "load_frozen_split_role",
    "sha256_file",
    "token_vector_sha256",
    "validate_stop_b_panel",
    "write_canonical_json",
]
