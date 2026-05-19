"""
Hard-macro 12 μm spacing metrics and optional conservative spacing repair.

Clearance definition (edge-to-edge, L∞ approximation):

    gap_x = |cx_i - cx_j| - (hw_i + hw_j)   # horizontal edge-to-edge gap
    gap_y = |cy_i - cy_j| - (hh_i + hh_j)   # vertical edge-to-edge gap

    clearance = 0                if gap_x <= 0 and gap_y <= 0  (overlapping)
    clearance = gap_y            if gap_x <= 0 and gap_y > 0   (vertical neighbors)
    clearance = gap_x            if gap_y <= 0 and gap_x > 0   (horizontal neighbors)
    clearance = min(gap_x, gap_y) if gap_x > 0 and gap_y > 0  (diagonal, conservative)

Using min() for the diagonal case is conservative: it may over-count violations
relative to Euclidean corner distance but is simple and consistent.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from macro_place.benchmark import Benchmark


SMALL_CANVAS_THRESHOLD_DEFAULT: float = 200.0  # μm


@dataclass
class SpacingConfig:
    """Configuration for spacing metrics, candidate scoring, and winner repair."""

    clearance_um: float = 12.0
    spacing_weight: float = 0.0
    repair: bool = False
    log: bool = False
    log_dir: Optional[str] = None
    small_canvas_threshold: float = SMALL_CANVAS_THRESHOLD_DEFAULT
    repair_max_iters: int = 50

    @classmethod
    def from_env(cls) -> "SpacingConfig":
        """Create config from environment variables."""
        return cls(
            clearance_um=float(os.environ.get("MPC_SPACING_CLEARANCE", "12.0")),
            spacing_weight=float(os.environ.get("MPC_SPACING_WEIGHT", "0.0")),
            repair=os.environ.get("MPC_SPACING_REPAIR", "0").strip().lower()
            in {"1", "true", "yes", "on"},
            log=os.environ.get("MPC_FINAL_LOG", "0").strip().lower()
            in {"1", "true", "yes", "on"},
            log_dir=os.environ.get("MPC_FINAL_LOG_DIR") or None,
            small_canvas_threshold=float(
                os.environ.get(
                    "MPC_SPACING_CANVAS_THRESHOLD",
                    str(SMALL_CANVAS_THRESHOLD_DEFAULT),
                )
            ),
        )

    def is_large_canvas(self, benchmark: Benchmark) -> bool:
        """True when canvas is large enough for ≥12 μm spacing to be meaningful."""
        return (
            max(float(benchmark.canvas_width), float(benchmark.canvas_height))
            >= self.small_canvas_threshold
        )


def _compute_gaps(pos: np.ndarray, sizes: np.ndarray, n: int) -> np.ndarray:
    """
    Vectorised edge-to-edge clearance for all N*(N-1)/2 pairs of hard macros.

    Returns clearance array shape (n_pairs,) per the L∞ approximation in this
    module's docstring.  Pairs where both rectangles overlap have clearance 0.
    """
    if n <= 1:
        return np.empty(0, dtype=np.float64)

    hw = sizes[:n, 0] / 2.0
    hh = sizes[:n, 1] / 2.0

    ii, jj = np.triu_indices(n, k=1)
    dx = np.abs(pos[ii, 0] - pos[jj, 0])
    dy = np.abs(pos[ii, 1] - pos[jj, 1])

    gap_x = dx - (hw[ii] + hw[jj])
    gap_y = dy - (hh[ii] + hh[jj])

    clearance = np.where(
        gap_x <= 0,
        np.where(gap_y <= 0, 0.0, gap_y),
        np.where(gap_y <= 0, gap_x, np.minimum(gap_x, gap_y)),
    )
    return clearance


def hard_macro_spacing_violation_count(
    placement: torch.Tensor,
    benchmark: Benchmark,
    clearance_um: float = 12.0,
) -> int:
    """Number of hard-macro pairs with edge-to-edge clearance < clearance_um."""
    n = benchmark.num_hard_macros
    if n <= 1:
        return 0
    pos = placement[:n].detach().cpu().numpy().astype(np.float64)
    sizes = benchmark.macro_sizes[:n].detach().cpu().numpy().astype(np.float64)
    clr = _compute_gaps(pos, sizes, n)
    return int(np.sum(clr < clearance_um))


def min_hard_macro_clearance(placement: torch.Tensor, benchmark: Benchmark) -> float:
    """Minimum edge-to-edge clearance across all hard-macro pairs (μm)."""
    n = benchmark.num_hard_macros
    if n <= 1:
        return float("inf")
    pos = placement[:n].detach().cpu().numpy().astype(np.float64)
    sizes = benchmark.macro_sizes[:n].detach().cpu().numpy().astype(np.float64)
    clr = _compute_gaps(pos, sizes, n)
    return float(np.min(clr)) if len(clr) > 0 else float("inf")


def spacing_penalty(
    placement: torch.Tensor,
    benchmark: Benchmark,
    clearance_um: float = 12.0,
) -> float:
    """
    Normalised spacing penalty in [0, ∞).

    Each pair contributes (shortfall / clearance_um)² where
    shortfall = max(0, clearance_um − actual_clearance).
    Result is divided by the number of pairs so benchmarks with different
    macro counts are comparable.
    """
    n = benchmark.num_hard_macros
    if n <= 1:
        return 0.0
    pos = placement[:n].detach().cpu().numpy().astype(np.float64)
    sizes = benchmark.macro_sizes[:n].detach().cpu().numpy().astype(np.float64)
    clr = _compute_gaps(pos, sizes, n)
    if len(clr) == 0:
        return 0.0
    shortfall = np.maximum(0.0, clearance_um - clr)
    return float(np.sum((shortfall / clearance_um) ** 2) / len(clr))


def repair_spacing(
    placement: torch.Tensor,
    benchmark: Benchmark,
    config: SpacingConfig,
) -> Tuple[torch.Tensor, Dict]:
    """
    Conservative spacing repair for hard macros.

    Each hard macro is treated as inflated by clearance_um/2 on every side.
    Violating pairs (inflated-box overlaps) are pushed apart along the axis
    with smaller overlap, minimising total displacement.  Fixed macros are
    never moved.  All macros are clamped to canvas after each push.

    Returns ``(repaired_placement, metrics_dict)``.

    Because inflated-box separation implies original-box separation, this
    repair cannot introduce hard-macro overlaps.

    If the canvas is small (< config.small_canvas_threshold), repair is skipped
    to avoid displacing macros by large fractions of the canvas.
    """
    n = benchmark.num_hard_macros
    clearance = config.clearance_um
    pre_viol = hard_macro_spacing_violation_count(placement, benchmark, clearance)
    pre_min = min_hard_macro_clearance(placement, benchmark)

    repaired = placement.clone()

    if not config.is_large_canvas(benchmark):
        return repaired, {
            "repair_ran": False,
            "skipped_reason": "small_canvas",
            "pre_violations": pre_viol,
            "post_violations": pre_viol,
            "pre_min_clearance_um": None if pre_min == float("inf") else pre_min,
            "post_min_clearance_um": None if pre_min == float("inf") else pre_min,
            "max_displacement_um": 0.0,
            "total_displacement_um": 0.0,
        }

    movable = benchmark.get_movable_mask()[:n].numpy()
    pos = repaired[:n].detach().cpu().numpy().copy().astype(np.float64)
    sizes = benchmark.macro_sizes[:n].detach().cpu().numpy().astype(np.float64)
    half_w = sizes[:, 0] / 2.0
    half_h = sizes[:, 1] / 2.0
    original_pos = pos.copy()

    inflated_hw = half_w + clearance / 2.0
    inflated_hh = half_h + clearance / 2.0

    canvas_w = float(benchmark.canvas_width)
    canvas_h = float(benchmark.canvas_height)

    # Precompute pair indices (fixed for the life of the repair)
    ii, jj = np.triu_indices(n, k=1)
    any_movable = movable[ii] | movable[jj]

    for _ in range(config.repair_max_iters):
        dx_all = pos[jj, 0] - pos[ii, 0]
        dy_all = pos[jj, 1] - pos[ii, 1]
        ox_all = inflated_hw[ii] + inflated_hw[jj] - np.abs(dx_all)
        oy_all = inflated_hh[ii] + inflated_hh[jj] - np.abs(dy_all)

        viol = (ox_all > 0) & (oy_all > 0) & any_movable
        if not np.any(viol):
            break

        # Process worst violations first (deterministic by inflated-overlap area)
        scores = np.where(viol, ox_all * oy_all, -1.0)
        order = np.argsort(-scores)

        changed = False
        for k in order:
            if not viol[k]:
                continue
            i_idx, j_idx = int(ii[k]), int(jj[k])
            mi, mj = bool(movable[i_idx]), bool(movable[j_idx])
            if not mi and not mj:
                continue

            # Recompute with current positions (may have changed within this iter)
            dx = pos[j_idx, 0] - pos[i_idx, 0]
            dy = pos[j_idx, 1] - pos[i_idx, 1]
            ox = inflated_hw[i_idx] + inflated_hw[j_idx] - abs(dx)
            oy = inflated_hh[i_idx] + inflated_hh[j_idx] - abs(dy)
            if ox <= 0 or oy <= 0:
                continue

            # Push along the axis with smaller overlap (less displacement)
            split = 2.0 if (mi and mj) else 1.0
            if ox <= oy:
                push = ox / split
                sx = 1.0 if dx >= 0 else -1.0
                if mi:
                    pos[i_idx, 0] -= push * sx
                if mj:
                    pos[j_idx, 0] += push * sx
            else:
                push = oy / split
                sy = 1.0 if dy >= 0 else -1.0
                if mi:
                    pos[i_idx, 1] -= push * sy
                if mj:
                    pos[j_idx, 1] += push * sy

            # Clamp both to canvas
            for m in (i_idx, j_idx):
                pos[m, 0] = float(np.clip(pos[m, 0], half_w[m], canvas_w - half_w[m]))
                pos[m, 1] = float(np.clip(pos[m, 1], half_h[m], canvas_h - half_h[m]))

            changed = True

        if not changed:
            break

    repaired[:n] = torch.tensor(pos, dtype=repaired.dtype)

    displacement = np.linalg.norm(pos - original_pos, axis=1)
    post_viol = hard_macro_spacing_violation_count(repaired, benchmark, clearance)
    post_min = min_hard_macro_clearance(repaired, benchmark)

    return repaired, {
        "repair_ran": True,
        "pre_violations": pre_viol,
        "post_violations": post_viol,
        "pre_min_clearance_um": None if pre_min == float("inf") else float(pre_min),
        "post_min_clearance_um": None if post_min == float("inf") else float(post_min),
        "max_displacement_um": float(np.max(displacement)) if len(displacement) > 0 else 0.0,
        "total_displacement_um": float(np.sum(displacement)),
    }
