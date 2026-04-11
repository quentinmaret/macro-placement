"""Shared types and small helpers for Phase 0 infrastructure."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, Union
import random

import numpy as np
import torch

from macro_place.benchmark import Benchmark


PlacementLike = Union[torch.Tensor, np.ndarray, Sequence[Sequence[float]]]


class Placer(Protocol):
    """Protocol for placement algorithms used by the runner."""

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        """Return a placement tensor with shape [num_macros, 2]."""


@dataclass(frozen=True)
class BenchmarkSpec:
    """Description of a benchmark that can be loaded by the runner."""

    name: str
    benchmark_dir: Path


@dataclass
class RunConfig:
    """Runner configuration with reproducibility and artifact controls."""

    benchmark_root: Path = Path("external/MacroPlacement/Testcases/ICCAD04")
    benchmark_names: Optional[List[str]] = None
    output_dir: Path = Path("submissions/core/outputs")
    seed: int = 42
    check_overlaps: bool = True
    save_visualizations: bool = True
    save_placements: bool = True
    log_filename: str = "results.json"
    config_filename: str = "config.json"


@dataclass
class EvaluationArtifacts:
    """Paths to artifacts emitted for one evaluated benchmark."""

    placement_path: Optional[Path] = None
    visualization_path: Optional[Path] = None


@dataclass
class EvaluationResult:
    """Normalized result record produced by the Phase 0 runner."""

    benchmark_name: str
    proxy_cost: float
    wirelength_cost: float
    density_cost: float
    congestion_cost: float
    overlap_count: int
    total_overlap_area: float
    max_overlap_area: float
    num_macros_with_overlaps: int
    overlap_ratio: float
    valid: bool
    violations: List[str]
    runtime_seconds: float
    seed: int
    artifacts: EvaluationArtifacts = field(default_factory=EvaluationArtifacts)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize result data into JSON-safe primitives."""
        return {
            "benchmark_name": self.benchmark_name,
            "proxy_cost": self.proxy_cost,
            "wirelength_cost": self.wirelength_cost,
            "density_cost": self.density_cost,
            "congestion_cost": self.congestion_cost,
            "overlap_count": self.overlap_count,
            "total_overlap_area": self.total_overlap_area,
            "max_overlap_area": self.max_overlap_area,
            "num_macros_with_overlaps": self.num_macros_with_overlaps,
            "overlap_ratio": self.overlap_ratio,
            "valid": self.valid,
            "violations": list(self.violations),
            "runtime_seconds": self.runtime_seconds,
            "seed": self.seed,
            "artifacts": {
                "placement_path": str(self.artifacts.placement_path)
                if self.artifacts.placement_path
                else None,
                "visualization_path": str(self.artifacts.visualization_path)
                if self.artifacts.visualization_path
                else None,
            },
        }


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def clone_initial_placement(benchmark: Benchmark) -> torch.Tensor:
    """Return a writable copy of the benchmark's current placement."""
    return benchmark.macro_positions.clone()


def as_placement_tensor(
    placement: PlacementLike,
    benchmark: Benchmark,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Convert placement-like data into the standard tensor format."""
    tensor = torch.as_tensor(placement, dtype=dtype)
    expected_shape = (benchmark.num_macros, 2)
    if tensor.shape != expected_shape:
        raise ValueError(
            f"Placement must have shape {expected_shape}, got {tuple(tensor.shape)}"
        )
    return tensor
