"""
Copyable baseline model for the macro placement challenge.

This file is designed to do two jobs at once:

1. It is a valid submission file that can be run directly with the repo's
   evaluation command.
2. It is a simple template that teammates can copy into a new model file and
   modify while keeping the same overall workflow.

Current behavior:
- start from the benchmark's initial placement
- legalize hard macros with small, local moves
- keep soft macros at their original locations
- return a legal, easy-to-debug placement

This is intentionally conservative. It does not try to win the benchmark yet.
Its purpose is to make the competition interface easy to understand and to
provide a clean starting point for stronger methods such as local search,
graph-guided refinement, or RL-based policies.
"""

from __future__ import annotations

import random
from typing import Iterable, List, Optional

import numpy as np
import torch

from macro_place.benchmark import Benchmark


class CorePlacer:
    """
    Minimal-displacement legalization baseline.

    The class is deliberately organized into small helper methods so teammates
    can copy this file and replace only the pieces they want to experiment with.
    The intended flow is:

    1. Seed randomness for reproducibility.
    2. Clone the initial placement from the benchmark.
    3. Legalize hard macros with small moves while preserving as much of the
       original layout as possible.
    4. Run an optional refinement hook for future experiments.
    5. Restore fixed macros and return the final placement tensor.
    """

    def __init__(
        self,
        seed: int = 42,
        search_radii: int = 150,
        step_scale: float = 0.25,
        safety_gap: float = 0.05,
    ) -> None:
        """
        Configure the baseline legalization strategy.

        Args:
            seed: Random seed used by helper methods and future refinements.
            search_radii: Number of expanding search rings used when finding a
                nearby legal location for one hard macro.
            step_scale: Distance between search points, expressed as a fraction
                of the macro's larger dimension.
            safety_gap: Extra spacing between hard macros to avoid tiny
                float-precision overlaps.
        """
        self.seed = seed
        self.search_radii = search_radii
        self.step_scale = step_scale
        self.safety_gap = safety_gap

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        """
        Produce a placement tensor in the competition's expected format.

        This is the only method the evaluator needs. Everything else in the
        class exists to make the workflow understandable and easy to modify.
        """
        self.set_seed()

        # Step 1: start from the given benchmark placement so we preserve the
        # useful structure already present in the data.
        placement = self.clone_initial_placement(benchmark)

        # Step 2: legalize the hard macros with local moves only.
        placement = self.legalize_initial_placement(placement, benchmark)

        # Step 3: future models can replace this with local search, SA, RL,
        # graph-based moves, or any other improvement stage.
        placement = self.refine_placement(placement, benchmark)

        # Step 4: fixed macros should always remain exactly where the benchmark
        # placed them, even if future experiments make larger changes.
        placement = self.restore_fixed_macros(placement, benchmark)

        return placement

    def set_seed(self) -> None:
        """
        Seed Python, NumPy, and PyTorch for reproducible experiments.

        Keeping this in a dedicated method makes it easy for copied models to
        stay deterministic while teammates change the actual algorithm.
        """
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    def clone_initial_placement(self, benchmark: Benchmark) -> torch.Tensor:
        """
        Return a writable copy of the benchmark's initial placement.

        The benchmark already contains positions for both hard and soft macros.
        Starting from this layout is the safest way to understand the system
        before trying more aggressive optimization ideas.
        """
        return benchmark.macro_positions.clone()

    def legalize_initial_placement(
        self, placement: torch.Tensor, benchmark: Benchmark
    ) -> torch.Tensor:
        """
        Legalize hard macros while staying close to the initial layout.

        The current strategy is simple:
        - process larger hard macros first
        - keep any macro that is already legal
        - otherwise search nearby locations in expanding square rings
        - pick the closest legal position that fits in the canvas

        Soft macros are left untouched for now. That keeps the baseline easy to
        reason about while still matching the real benchmark tensor format.
        """
        legalized = placement.clone()
        hard_indices = self.get_hard_macro_indices(benchmark)
        movable_hard_indices = self.get_movable_hard_macro_indices(benchmark)

        if not movable_hard_indices:
            return legalized

        # Larger macros are harder to fit, so we place them first.
        order = sorted(
            hard_indices,
            key=lambda idx: -self.macro_area(benchmark, idx),
        )

        placed_hard: List[int] = []
        for idx in order:
            if idx not in movable_hard_indices:
                placed_hard.append(idx)
                continue

            original_center = legalized[idx].clone()
            legalized[idx] = self.clamp_macro_to_canvas(original_center, benchmark, idx)

            if self.is_hard_macro_legal(legalized, benchmark, idx, placed_hard):
                placed_hard.append(idx)
                continue

            new_center = self.find_nearby_legal_position(
                placement=legalized,
                benchmark=benchmark,
                macro_index=idx,
                reference_position=original_center,
                placed_hard_indices=placed_hard,
            )
            legalized[idx] = new_center
            placed_hard.append(idx)

        return legalized

    def refine_placement(
        self, placement: torch.Tensor, benchmark: Benchmark
    ) -> torch.Tensor:
        """
        Hook for future optimization stages.

        This baseline does nothing on purpose. In copied models, this is the
        cleanest place to add:
        - simulated annealing
        - graph-guided swaps or shifts
        - analytical smoothing
        - RL action loops
        - soft-macro movement policies
        """
        return placement

    def restore_fixed_macros(
        self, placement: torch.Tensor, benchmark: Benchmark
    ) -> torch.Tensor:
        """
        Put fixed macros back at their benchmark positions.

        Even though the current baseline never moves fixed macros on purpose,
        this final step makes the contract explicit and keeps future copied
        models safer.
        """
        fixed_mask = benchmark.macro_fixed
        placement[fixed_mask] = benchmark.macro_positions[fixed_mask]
        return placement

    def get_hard_macro_indices(self, benchmark: Benchmark) -> List[int]:
        """
        Return the benchmark indices of all hard macros.

        Hard macros always occupy the first `num_hard_macros` entries in the
        competition tensor format.
        """
        return list(range(benchmark.num_hard_macros))

    def get_soft_macro_indices(self, benchmark: Benchmark) -> List[int]:
        """
        Return the benchmark indices of all soft macros.

        Keeping this helper in the template makes it easier to add soft-macro
        logic later without having to rediscover the benchmark layout.
        """
        return list(range(benchmark.num_hard_macros, benchmark.num_macros))

    def get_movable_hard_macro_indices(self, benchmark: Benchmark) -> List[int]:
        """
        Return the hard macro indices that are allowed to move.

        Most IBM cases have movable hard macros only, but this helper keeps the
        model honest if future benchmarks include fixed hard macros.
        """
        movable_mask = benchmark.get_movable_mask() & benchmark.get_hard_macro_mask()
        return torch.where(movable_mask)[0].tolist()

    def macro_area(self, benchmark: Benchmark, macro_index: int) -> float:
        """
        Compute the area of one macro.

        Area is used only to decide legalization order. Placing large macros
        first usually makes conservative legalization easier.
        """
        width, height = benchmark.macro_sizes[macro_index].tolist()
        return float(width * height)

    def clamp_macro_to_canvas(
        self,
        center: torch.Tensor,
        benchmark: Benchmark,
        macro_index: int,
    ) -> torch.Tensor:
        """
        Clamp one macro center so the macro stays fully inside the canvas.

        The evaluator expects center coordinates, not lower-left corners, so the
        valid range depends on the macro's width and height.
        """
        width = benchmark.macro_sizes[macro_index, 0].item()
        height = benchmark.macro_sizes[macro_index, 1].item()

        min_x = width / 2.0
        max_x = benchmark.canvas_width - width / 2.0
        min_y = height / 2.0
        max_y = benchmark.canvas_height - height / 2.0

        clamped = center.clone()
        clamped[0] = torch.clamp(clamped[0], min=min_x, max=max_x)
        clamped[1] = torch.clamp(clamped[1], min=min_y, max=max_y)
        return clamped

    def is_hard_macro_legal(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        macro_index: int,
        other_hard_indices: Iterable[int],
    ) -> bool:
        """
        Check whether one hard macro is legal against already placed hard macros.

        This helper enforces the two constraints we care about in the baseline:
        - the macro must stay within the canvas
        - it must not overlap any other relevant hard macro
        """
        center = placement[macro_index]
        clamped = self.clamp_macro_to_canvas(center, benchmark, macro_index)
        if not torch.allclose(center, clamped, atol=1e-6):
            return False

        return self.find_first_hard_overlap(
            placement=placement,
            benchmark=benchmark,
            macro_index=macro_index,
            other_hard_indices=other_hard_indices,
        ) is None

    def find_first_hard_overlap(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        macro_index: int,
        other_hard_indices: Iterable[int],
    ) -> Optional[int]:
        """
        Return the first hard macro index that overlaps `macro_index`.

        Returning the first conflict keeps the function useful for both simple
        legality checks and future local repair policies.
        """
        x_i, y_i = placement[macro_index].tolist()
        w_i, h_i = benchmark.macro_sizes[macro_index].tolist()

        for other_idx in other_hard_indices:
            if other_idx == macro_index:
                continue

            x_j, y_j = placement[other_idx].tolist()
            w_j, h_j = benchmark.macro_sizes[other_idx].tolist()

            min_sep_x = (w_i + w_j) / 2.0 + self.safety_gap
            min_sep_y = (h_i + h_j) / 2.0 + self.safety_gap

            if abs(x_i - x_j) < min_sep_x and abs(y_i - y_j) < min_sep_y:
                return other_idx

        return None

    def find_nearby_legal_position(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        macro_index: int,
        reference_position: torch.Tensor,
        placed_hard_indices: List[int],
    ) -> torch.Tensor:
        """
        Search for the closest nearby legal location for one hard macro.

        The search expands in square rings around the original position. This
        makes the code easy to understand and keeps movement small, which is a
        good baseline behavior before we start optimizing.
        """
        width, height = benchmark.macro_sizes[macro_index].tolist()
        step = max(width, height) * self.step_scale
        best_position = self.clamp_macro_to_canvas(reference_position, benchmark, macro_index)
        best_distance = float("inf")

        for radius in range(1, self.search_radii + 1):
            found_better = False
            for dx_step in range(-radius, radius + 1):
                for dy_step in range(-radius, radius + 1):
                    # Only visit the current ring, not the entire square.
                    if abs(dx_step) != radius and abs(dy_step) != radius:
                        continue

                    candidate = reference_position.clone()
                    candidate[0] = candidate[0] + dx_step * step
                    candidate[1] = candidate[1] + dy_step * step
                    candidate = self.clamp_macro_to_canvas(candidate, benchmark, macro_index)

                    placement[macro_index] = candidate
                    if not self.is_hard_macro_legal(
                        placement,
                        benchmark,
                        macro_index,
                        placed_hard_indices,
                    ):
                        continue

                    distance = torch.sum((candidate - reference_position) ** 2).item()
                    if distance < best_distance:
                        best_distance = distance
                        best_position = candidate.clone()
                        found_better = True

            if found_better:
                break

        placement[macro_index] = best_position
        return best_position


if __name__ == "__main__":
    print(
        "Run this file through the repo evaluator, for example:\n"
        "  uv run evaluate submissions/models/core.py -b ibm01"
    )
