from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Trigger geometry
# ---------------------------------------------------------------------------
# A trigger region is an axis-aligned rectangle (row0, col0, row1, col1) with
# row1/col1 exclusive (slice-ready). The plain pixel-backdoor stamps a single
# corner square; DBA (Xie et al. 2020) splits that square into a grid of
# disjoint sub-rectangles, one per malicious client. Both go through the same
# `_stamp_rects` primitive so train-time and eval-time triggers cannot drift.


def _corner_rect(
    h: int, w: int, trigger_size: int, position: str
) -> tuple[int, int, int, int]:
    """Return the (row0, col0, row1, col1) rectangle for a corner trigger.

    row1/col1 are exclusive. Matches the legacy four-corner placement, so a
    single-rect stamp is bit-identical to the pre-Cycle-02 behaviour.
    """
    s = trigger_size
    if position == "bottom-right":
        return (h - s, w - s, h, w)
    if position == "top-left":
        return (0, 0, s, s)
    if position == "bottom-left":
        return (h - s, 0, h, s)
    if position == "top-right":
        return (0, w - s, s, w)
    raise ValueError(f"Unknown trigger_position: {position}")


def _stamp_rects(
    image: torch.Tensor,
    rects: list[tuple[int, int, int, int]],
    trigger_value: float,
) -> torch.Tensor:
    """Stamp one or more solid rectangles onto a (C, H, W) image tensor."""
    triggered = image.clone()
    for (r0, c0, r1, c1) in rects:
        triggered[:, r0:r1, c0:c1] = trigger_value
    return triggered


def _stamp_trigger(
    image: torch.Tensor,
    trigger_size: int,
    trigger_value: float,
    trigger_position: str,
) -> torch.Tensor:
    """Stamp a solid-square corner trigger onto a (C, H, W) image tensor.

    Backwards-compatible shim over `_stamp_rects` — a single corner
    rectangle. Bit-identical to the original implementation.
    """
    h, w = image.shape[1], image.shape[2]
    rect = _corner_rect(h, w, trigger_size, trigger_position)
    return _stamp_rects(image, [rect], trigger_value)


def _parse_grid(grid: str) -> tuple[int, int]:
    """Parse a 'RxC' grid string (e.g. '2x2', '1x1') into (rows, cols)."""
    try:
        parts = str(grid).lower().split("x")
        gr, gc = int(parts[0]), int(parts[1])
    except (ValueError, IndexError) as exc:
        raise ValueError(
            f"Invalid dba-grid {grid!r}; expected 'RxC' e.g. '2x2'"
        ) from exc
    if gr < 1 or gc < 1:
        raise ValueError(f"dba-grid {grid!r} must have positive dimensions")
    return gr, gc


def dba_subregions(
    trigger_size: int,
    trigger_position: str,
    grid: str,
    image_size: int,
    pattern: str = "corner-grid",
) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int, int, int]]]:
    """Sub-trigger geometry for the DBA distributed-backdoor attack.

    Returns ``(per_client_rects, union_rects)``:
      - ``per_client_rects``: a list of single rectangles -- one
        sub-trigger per slot. Each malicious client stamps the rectangle
        assigned to its index in (sorted) malicious-client-ids modulo
        ``len(per_client_rects)``.
      - ``union_rects``: the list of rectangles the server stamps at
        eval time for ASR (the union of all sub-triggers the malicious
        clients are stamping). Always a list, possibly of one element.

    Two patterns:

    - ``"corner-grid"`` (legacy / null-config-compatible): splits the
      ``trigger_size x trigger_size`` square at ``trigger_position`` into
      a ``grid`` (e.g. '2x2' -> 4) of disjoint sub-rectangles tiling the
      square. The union is the single corner square the sub-rects tile.
      ``grid='1x1'`` returns ``([corner_square], [corner_square])`` --
      bit-identical to the plain pixel backdoor.

    - ``"scattered-bars"`` (paper-faithful, Xie et al. 2020): four
      non-contiguous 1x6 horizontal bars with 3-px gaps in the top
      region of a 32x32 image -- the CIFAR pattern from the DBA reference
      ``cifar_params.yaml``. Each malicious client stamps ONE bar; the
      server measures ASR with all 4 bars stamped together (the gaps
      stay clean, NOT filled by a bounding box). ``grid``,
      ``trigger_size``, and ``trigger_position`` are IGNORED in this
      mode -- the pattern is hardcoded paper-faithful for image_size=32.
    """
    if pattern == "scattered-bars":
        bars: list[tuple[int, int, int, int]] = [
            (0, 0, 1, 6),    # row 0, cols 0..5
            (0, 9, 1, 15),   # row 0, cols 9..14
            (4, 0, 5, 6),    # row 4, cols 0..5
            (4, 9, 5, 15),   # row 4, cols 9..14
        ]
        return bars, bars

    if pattern != "corner-grid":
        raise ValueError(
            f"Unknown dba-pattern {pattern!r}; expected 'corner-grid' or 'scattered-bars'."
        )

    gr, gc = _parse_grid(grid)
    r0, c0, r1, c1 = _corner_rect(
        image_size, image_size, trigger_size, trigger_position
    )
    # Even-as-possible integer tiling of [r0, r1) into gr parts (cols: gc).
    row_edges = [r0 + round(i * (r1 - r0) / gr) for i in range(gr + 1)]
    col_edges = [c0 + round(j * (c1 - c0) / gc) for j in range(gc + 1)]
    per_client_rects: list[tuple[int, int, int, int]] = []
    for i in range(gr):
        for j in range(gc):
            per_client_rects.append(
                (row_edges[i], col_edges[j], row_edges[i + 1], col_edges[j + 1])
            )
    union_rect = (r0, c0, r1, c1)
    return per_client_rects, [union_rect]


def make_pixel_trigger_fn(
    trigger_size: int = 4,
    trigger_value: float = 1.0,
    trigger_position: str = "bottom-right",
    trigger_rects: list[tuple[int, int, int, int]] | None = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return a function that stamps the trigger on a batch (B, C, H, W).

    When ``trigger_rects`` is given (DBA: the union rectangle from
    ``dba_subregions``) those exact rectangles are stamped; otherwise the
    four-corner square is used (plain pixel backdoor).
    """

    def trigger_fn(images: torch.Tensor) -> torch.Tensor:
        triggered = images.clone()
        h, w = triggered.shape[2], triggered.shape[3]
        if trigger_rects is not None:
            rects = trigger_rects
        else:
            rects = [_corner_rect(h, w, trigger_size, trigger_position)]
        for (r0, c0, r1, c1) in rects:
            triggered[:, :, r0:r1, c0:c1] = trigger_value
        return triggered

    return trigger_fn


class PixelBackdoorDataset(Dataset):
    """Wrap a dataset: stamp a pixel trigger on a fraction of samples and relabel to target.

    The trigger is a solid square of ``trigger_size x trigger_size`` pixels
    placed at ``trigger_position`` with value ``trigger_value``. Applied
    AFTER transforms (the base dataset already returns tensors).

    Cycle-02 / DBA: when ``trigger_rects`` is given the dataset stamps those
    exact rectangles instead of the corner square — this is how a malicious
    client stamps only its assigned DBA sub-trigger. ``trigger_rects=None``
    keeps the original corner behaviour bit-for-bit.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        target_label: int,
        poison_fraction: float = 0.1,
        trigger_size: int = 4,
        trigger_value: float = 1.0,
        trigger_position: str = "bottom-right",
        seed: int = 42,
        source_classes: tuple[int, ...] | None = None,
        trigger_rects: list[tuple[int, int, int, int]] | None = None,
    ) -> None:
        self.base_dataset = base_dataset
        self.target_label = int(target_label)
        self.poison_fraction = float(poison_fraction)
        self.trigger_size = int(trigger_size)
        self.trigger_value = float(trigger_value)
        self.trigger_position = str(trigger_position)
        self.seed = int(seed)
        # Cycle-02 base/edge regime: when set, only samples whose clean
        # label is in this set are eligible to be poisoned. None => every
        # non-target class is eligible (the original behaviour).
        self.source_classes = (
            tuple(int(c) for c in source_classes)
            if source_classes is not None else None
        )
        # Cycle-02 DBA: explicit sub-trigger rectangle(s) for this client.
        # None => stamp the corner square (plain pixel backdoor).
        self.trigger_rects = (
            [tuple(int(x) for x in r) for r in trigger_rects]
            if trigger_rects is not None else None
        )

        if not 0.0 <= self.poison_fraction <= 1.0:
            raise ValueError(
                f"poison_fraction must be in [0, 1], got {self.poison_fraction}"
            )

        self.poison_mask = self._build_poison_mask()

    def _build_poison_mask(self) -> np.ndarray:
        """Select non-target samples to poison (deterministic via seed)."""
        rng = np.random.default_rng(self.seed)
        n = len(self.base_dataset)

        # Only poison samples not already in the target class.
        # Use get_labels() when available to avoid loading images / running transforms.
        if hasattr(self.base_dataset, "get_labels"):
            labels = self.base_dataset.get_labels()
        else:
            labels = [int(self.base_dataset[i][1]) for i in range(n)]
        # Eligible sources: non-target classes, optionally restricted to
        # the configured base/edge source-class pool.
        allowed = self.source_classes  # None => all non-target classes
        non_target_indices = np.array(
            [
                i for i, lbl in enumerate(labels)
                if int(lbl) != self.target_label
                and (allowed is None or int(lbl) in allowed)
            ],
            dtype=np.intp,
        )

        num_poison = int(len(non_target_indices) * self.poison_fraction)
        mask = np.zeros(n, dtype=bool)

        if num_poison == 0 or len(non_target_indices) == 0:
            return mask

        chosen = rng.choice(non_target_indices, size=num_poison, replace=False)
        mask[chosen] = True
        return mask

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        image, label = self.base_dataset[idx]
        if self.poison_mask[idx]:
            if self.trigger_rects is not None:
                image = _stamp_rects(
                    image, self.trigger_rects, self.trigger_value
                )
            else:
                image = _stamp_trigger(
                    image,
                    self.trigger_size,
                    self.trigger_value,
                    self.trigger_position,
                )
            label = self.target_label
        return image, label

    def num_poisoned(self) -> int:
        return int(self.poison_mask.sum())
