"""fl_v3 — federated multimodal AD-perception platform (Cycle 04).

A fresh, bit-deterministic rewrite. ``fl_v2`` is frozen and used only as an
*implementation oracle* (parity on fixtures, not scientific validity). See
``fl_v3/collab/T0/SPEC.md`` and the durable plan
``fl_v2/docs/roadmap/cycle_04_fusion_layer_backdoors.md``.

Design pillars carried into T0:
  * **Bit-determinism is sacred** — all RNG via ``fl_v3.utils.runtime.derive_seed``;
    ``enforce_determinism()`` pins the global determinism state; banned ops
    (atomic scatter, ``grid_sample`` backward, non-stable sort/topk, flash-attn)
    are absent.
  * **Task-agnostic FL** — no hardcoded loss / num-classes / single-logits
    assumption; the criterion, model, data, and eval metrics are supplied by a
    ``Task`` selected from ``fl_v3.training.tasks.TASK_REGISTRY``.
  * **Defenses as pure-numpy cores** — each defense's numerical decision is a
    framework-free function over update vectors, validated against the ``fl_v2``
    oracle on saved fixtures, then wrapped thinly for Flower.
"""
from __future__ import annotations

__version__ = "0.1.0"
