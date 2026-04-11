"""Unified evaluation API built on top of the existing macro_place utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from macro_place.benchmark import Benchmark
from macro_place.objective import compute_proxy_cost
from macro_place.utils import validate_placement, visualize_placement

from submissions.core.types import PlacementLike, as_placement_tensor


def evaluate(
    placement: PlacementLike,
    benchmark: Benchmark,
    plc: Any,
    *,
    weights: Optional[Dict[str, float]] = None,
    check_overlaps: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate a placement with validation and proxy-cost reporting.

    Returns a merged dictionary containing validation status and cost metrics.
    """
    placement_tensor = as_placement_tensor(placement, benchmark)
    valid, violations = validate_placement(
        placement_tensor,
        benchmark,
        check_overlaps=check_overlaps,
    )
    costs = compute_proxy_cost(placement_tensor, benchmark, plc, weights=weights)
    return {
        "placement": placement_tensor,
        "valid": valid,
        "violations": violations,
        **costs,
    }


def validate(
    placement: PlacementLike,
    benchmark: Benchmark,
    *,
    check_overlaps: bool = True,
):
    """Validate a placement against benchmark legality rules."""
    placement_tensor = as_placement_tensor(placement, benchmark)
    return validate_placement(
        placement_tensor,
        benchmark,
        check_overlaps=check_overlaps,
    )


def visualize(
    placement: PlacementLike,
    benchmark: Benchmark,
    *,
    save_path: Optional[Path] = None,
    plc: Any = None,
) -> Optional[Path]:
    """Render a placement figure, optionally saving it to disk."""
    placement_tensor = as_placement_tensor(placement, benchmark)
    resolved_path = str(save_path) if save_path is not None else None
    visualize_placement(placement_tensor, benchmark, save_path=resolved_path, plc=plc)
    return save_path
