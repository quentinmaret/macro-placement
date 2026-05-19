"""
Ensemble initializer template for the macro placement challenge.

This file is designed to be the "initial placement ideas" counterpart to the
other model templates in `submissions/models`:

1. It is a valid submission file that the evaluator can run directly.
2. It is heavily commented so teammates can understand the flow quickly.
3. It is organized as an initializer ensemble, which makes it easy to add
   more seeding methods later without rewriting the outer workflow.

Current behavior:
- start from the benchmark placement as a safe anchor
- generate a small ensemble of initialization candidates
- include a graph clustering / hierarchy initializer as the main new method
- legalize each candidate with the shared `CorePlacer` logic
- score the candidates with a simple connectivity-aware surrogate
- return the best candidate

Why organize it this way?

Macro placement often benefits from strong starting points. Instead of baking
one hard-coded initialization into the entire placer, this template separates:

- "how do we generate candidate seeds?"
- "how do we legalize them?"
- "how do we choose among them?"

That separation should make it straightforward to add future ensemble members
such as spectral initialization, row packing, force-based spreading, or
learned initializers.
"""

from __future__ import annotations

import importlib.util
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/macro-placement-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/macro-placement-cache")

import numpy as np
import torch

from macro_place.benchmark import Benchmark


def _load_core_placer_class():
    """
    Load `CorePlacer` from the sibling `core.py` file.

    Submission files are imported directly by path during evaluation, so we
    cannot rely on normal package imports here. Loading the sibling file
    explicitly keeps this template copyable and self-contained.
    """
    core_path = Path(__file__).with_name("core.py")
    spec = importlib.util.spec_from_file_location("models_core", str(core_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load CorePlacer from {core_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CorePlacer


CorePlacer = _load_core_placer_class()


class InitializerOperator:
    """
    Small wrapper for seed and placement-transform initializer operators.

    The runner callable receives the shared `EnsembleInitializerPlacer` helper
    as its first argument so operators can reuse the existing graph, geometry,
    and legalization methods without adding another abstraction layer.
    """

    def __init__(
        self,
        name: str,
        kind: str,
        runner: Callable[..., torch.Tensor],
        aliases: Optional[Sequence[str]] = None,
    ) -> None:
        self.name = name
        self.kind = kind
        self.runner = runner
        self.aliases = list(aliases or [])

    def run(
        self,
        problem: Benchmark,
        placement: Optional[torch.Tensor] = None,
        rng: Optional[torch.Generator] = None,
        config: Optional[Dict[str, Any]] = None,
        context: Optional["EnsembleInitializerPlacer"] = None,
    ) -> torch.Tensor:
        """Apply this operator and return a full placement tensor."""
        helper = context or EnsembleInitializerPlacer()
        return self.runner(
            helper,
            problem,
            placement,
            rng or torch.Generator().manual_seed(helper.seed),
            config or {},
        )


INITIALIZER_REGISTRY: Dict[str, InitializerOperator] = {}


def register_initializer_operator(operator: InitializerOperator) -> InitializerOperator:
    """Register one initializer/operator under its name and aliases."""
    names = [operator.name] + operator.aliases
    for name in names:
        existing = INITIALIZER_REGISTRY.get(name)
        if existing is not None and existing is not operator:
            raise ValueError(f"Initializer/operator name already registered: {name}")
        INITIALIZER_REGISTRY[name] = operator
    return operator


def get_initializer_operator(name: str) -> InitializerOperator:
    """Return a registered initializer/operator or fail with a helpful error."""
    try:
        return INITIALIZER_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(INITIALIZER_REGISTRY))
        raise ValueError(f"Unknown initializer/operator `{name}`. Available: {available}") from exc


def list_initializer_operators() -> List[str]:
    """Return registered initializer/operator names."""
    return sorted(INITIALIZER_REGISTRY)


class HierarchyNode:
    """
    One node in the graph-clustering hierarchy tree.

    The node stores:
    - which hard macros belong to this cluster
    - how much total hard-macro area that cluster owns
    - optional child clusters after a recursive split

    A plain class is used instead of `@dataclass` to keep dynamic loading
    simple and consistent with the other submission templates.
    """

    def __init__(
        self,
        macro_indices: Sequence[int],
        total_area: float,
        depth: int,
        children: Optional[List["HierarchyNode"]] = None,
    ) -> None:
        self.macro_indices = list(macro_indices)
        self.total_area = total_area
        self.depth = depth
        self.children = children or []

    @property
    def is_leaf(self) -> bool:
        """Return True when this cluster has no child clusters."""
        return len(self.children) == 0


class CandidatePlacement:
    """
    One candidate produced by the initializer ensemble.

    Keeping candidate metadata explicit makes the ensemble easier to extend.
    When more initializer methods are added later, we can compare them using
    the same scoring and debugging path.
    """

    def __init__(
        self,
        name: str,
        placement: torch.Tensor,
        notes: str,
    ) -> None:
        self.name = name
        self.placement = placement
        self.notes = notes


class EnsembleInitializerPlacer(CorePlacer):
    """
    Ensemble-based initialization template.

    The high-level flow is:

    1. Build one or more candidate initial placements.
    2. Legalize each candidate with the conservative `CorePlacer` logic.
    3. Score the legal candidates with a simple graph-aware surrogate.
    4. Return the best one.

    Right now the ensemble contains:
    - the raw benchmark initialization as a safety baseline
    - a graph clustering / hierarchy initialization

    Future methods can be added by:
    - appending a new method name to `initializer_sequence`
    - implementing a matching `run_<name>_initializer(...)` helper
    """

    def __init__(
        self,
        seed: int = 42,
        search_radii: int = 150,
        step_scale: float = 0.25,
        safety_gap: float = 0.05,
        max_cluster_size: int = 4,
        min_cluster_size_to_split: int = 6,
        wirelength_weight: float = 1.0,
        anchor_weight: float = 0.15,
        spread_weight: float = 0.05,
    ) -> None:
        """
        Configure the ensemble initializer.

        Args:
            seed: Random seed inherited from the shared scaffold.
            search_radii: Number of search rings used by legalization.
            step_scale: Local legalization step size.
            safety_gap: Extra hard-macro spacing margin.
            max_cluster_size: Desired leaf cluster size in the hierarchy tree.
            min_cluster_size_to_split: Small clusters are kept intact because
                splitting them further usually adds complexity without helping.
            wirelength_weight: Weight on the connectivity surrogate used when
                selecting the best candidate from the ensemble.
            anchor_weight: Weight on staying somewhat close to the benchmark's
                original placement.
            spread_weight: Small penalty on overly spread-out layouts.
        """
        super().__init__(
            seed=seed,
            search_radii=search_radii,
            step_scale=step_scale,
            safety_gap=safety_gap,
        )
        self.max_cluster_size = max_cluster_size
        self.min_cluster_size_to_split = min_cluster_size_to_split
        self.wirelength_weight = wirelength_weight
        self.anchor_weight = anchor_weight
        self.spread_weight = spread_weight

        # This ordered list is the core "ensemble" hook.
        #
        # To add another initializer later, add its short name here and create
        # a matching `run_<name>_initializer` method. The outer workflow does
        # not need to change.
        self.initializer_sequence = [
            "benchmark_anchor",
            "graph_hierarchy",
        ]

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        """
        Produce a placement in the evaluator's expected tensor format.

        The initializer ensemble is intentionally kept as the main idea here.
        Unlike the RL template, this file is all about generating a strong
        starting placement rather than performing many local decision steps.
        """
        self.set_seed()

        candidates = self.generate_initializer_candidates(benchmark)
        best = self.select_best_candidate(candidates, benchmark)

        # Keep the same extension hook pattern used by the other model files.
        refined = self.refine_placement(best.placement, benchmark)
        refined = self.restore_fixed_macros(refined, benchmark)
        return refined

    def generate_initializer_candidates(
        self,
        benchmark: Benchmark,
    ) -> List[CandidatePlacement]:
        """
        Build, legalize, and collect all initializer candidates.

        Each initializer method is responsible only for proposing a placement.
        We run the shared legalization pass afterwards so new methods can stay
        simple and focus on structure rather than exact collision handling.
        """
        candidates: List[CandidatePlacement] = []

        for initializer_name in self.initializer_sequence:
            generator = getattr(self, f"run_{initializer_name}_initializer")
            raw_candidate = generator(benchmark)
            legalized_candidate = self.legalize_candidate(raw_candidate, benchmark)
            candidates.append(
                CandidatePlacement(
                    name=initializer_name,
                    placement=legalized_candidate,
                    notes=f"Candidate generated by `{initializer_name}` initializer.",
                )
            )

        return candidates

    def legalize_candidate(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
    ) -> torch.Tensor:
        """
        Apply the shared legality pipeline to one candidate placement.

        Reusing the same legalization logic across all initializer methods is
        important for two reasons:

        1. It keeps new methods easy to implement.
        2. It makes candidate comparisons fairer because every candidate goes
           through the same legality repair stage.
        """
        legalized = self.legalize_initial_placement(placement.clone(), benchmark)
        legalized = self.restore_fixed_macros(legalized, benchmark)
        return legalized

    def select_best_candidate(
        self,
        candidates: Sequence[CandidatePlacement],
        benchmark: Benchmark,
    ) -> CandidatePlacement:
        """
        Choose the best candidate from the current initializer ensemble.

        For now we use a lightweight, easy-to-read surrogate:
        - connected hard macros should be near each other
        - candidates should not drift too far from the benchmark anchor
        - candidates should not be unnecessarily spread out

        Lower score is better.
        """
        if len(candidates) == 1:
            return candidates[0]

        hard_graph = self.build_hard_macro_graph(benchmark)
        best_candidate = candidates[0]
        best_score = self.score_candidate(candidates[0].placement, benchmark, hard_graph)

        for candidate in candidates[1:]:
            score = self.score_candidate(candidate.placement, benchmark, hard_graph)
            if score < best_score:
                best_candidate = candidate
                best_score = score

        return best_candidate

    def score_candidate(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        hard_graph: Dict[int, Dict[int, float]],
    ) -> float:
        """
        Compute a small surrogate score for ensemble candidate selection.

        This is deliberately simpler than the full benchmark proxy objective.
        The placer only receives a `Benchmark`, not a full PlacementCost
        object, so we use a graph-and-geometry score that is available here.
        """
        wirelength_term = self.compute_graph_wirelength_surrogate(
            placement=placement,
            benchmark=benchmark,
            hard_graph=hard_graph,
        )
        anchor_term = self.compute_anchor_distance_penalty(placement, benchmark)
        spread_term = self.compute_layout_spread_penalty(placement, benchmark)

        return (
            self.wirelength_weight * wirelength_term
            + self.anchor_weight * anchor_term
            + self.spread_weight * spread_term
        )

    def run_benchmark_anchor_initializer(self, benchmark: Benchmark) -> torch.Tensor:
        """
        Return the raw benchmark placement as the baseline ensemble member.

        Keeping this candidate in the ensemble is useful for two reasons:
        - it gives us a safe fallback
        - it makes it easy to tell whether a new initializer is helping
        """
        return self.clone_initial_placement(benchmark)

    def run_graph_hierarchy_initializer(self, benchmark: Benchmark) -> torch.Tensor:
        """
        Build a placement using graph clustering plus recursive hierarchy cuts.

        The main intuition is:

        1. Build a hard-macro connectivity graph from the benchmark netlist.
        2. Recursively split the movable hard macros into smaller clusters.
        3. Recursively assign canvas regions to those clusters.
        4. Place leaf-cluster macros inside their assigned regions.
        5. Leave legalization to the shared conservative repair pass.

        This gives us a structured seed that tries to keep strongly connected
        macros near each other before detailed legalization/refinement begins.
        """
        placement = self.clone_initial_placement(benchmark)
        movable_indices = self.get_movable_hard_macro_indices(benchmark)
        if not movable_indices:
            return placement

        hard_graph = self.build_hard_macro_graph(benchmark)
        hierarchy_root = self.build_graph_hierarchy(
            benchmark=benchmark,
            hard_graph=hard_graph,
            macro_indices=movable_indices,
            depth=0,
        )

        canvas_region = (
            0.0,
            0.0,
            float(benchmark.canvas_width),
            float(benchmark.canvas_height),
        )
        region_by_macro: Dict[int, Tuple[float, float, float, float]] = {}

        self.assign_regions_to_hierarchy(
            node=hierarchy_root,
            region=canvas_region,
            benchmark=benchmark,
            region_by_macro=region_by_macro,
        )
        self.place_leaf_clusters_from_regions(
            placement=placement,
            benchmark=benchmark,
            hard_graph=hard_graph,
            region_by_macro=region_by_macro,
        )

        return placement

    def build_hard_macro_graph(self, benchmark: Benchmark) -> Dict[int, Dict[int, float]]:
        """
        Build a simple weighted hard-macro connectivity graph.

        Each hard macro becomes one node.
        Two hard macros receive an undirected weighted edge when they appear
        together in a net. Larger multi-pin nets are down-weighted so they do
        not dominate the graph as strongly as small, focused nets.
        """
        graph: Dict[int, Dict[int, float]] = {
            idx: {} for idx in self.get_hard_macro_indices(benchmark)
        }

        for net_nodes in benchmark.net_nodes:
            hard_nodes = [
                int(node)
                for node in net_nodes.tolist()
                if int(node) < benchmark.num_hard_macros
            ]
            unique_nodes = sorted(set(hard_nodes))

            if len(unique_nodes) < 2:
                continue

            weight = 1.0 / float(len(unique_nodes) - 1)
            for left_pos, left_idx in enumerate(unique_nodes):
                for right_idx in unique_nodes[left_pos + 1 :]:
                    graph[left_idx][right_idx] = graph[left_idx].get(right_idx, 0.0) + weight
                    graph[right_idx][left_idx] = graph[right_idx].get(left_idx, 0.0) + weight

        return graph

    def build_graph_hierarchy(
        self,
        benchmark: Benchmark,
        hard_graph: Dict[int, Dict[int, float]],
        macro_indices: Sequence[int],
        depth: int,
    ) -> HierarchyNode:
        """
        Recursively split a macro set into a small hierarchy tree.

        The splitting rule is intentionally simple and readable:
        - if the cluster is already small, stop
        - otherwise pick two representative seed macros
        - assign each remaining macro to the seed it is more strongly connected to
        - recurse on the two child groups

        This is not meant to be the final word on clustering quality. It is a
        first understandable hierarchy initializer that teammates can improve.
        """
        total_area = sum(self.macro_area(benchmark, idx) for idx in macro_indices)
        node = HierarchyNode(
            macro_indices=macro_indices,
            total_area=total_area,
            depth=depth,
        )

        if len(macro_indices) <= self.max_cluster_size:
            return node
        if len(macro_indices) < self.min_cluster_size_to_split:
            return node

        left_group, right_group = self.partition_cluster(
            benchmark=benchmark,
            hard_graph=hard_graph,
            macro_indices=macro_indices,
        )

        # If partitioning collapses, we stop instead of forcing a bad split.
        if len(left_group) == 0 or len(right_group) == 0:
            return node

        node.children = [
            self.build_graph_hierarchy(
                benchmark=benchmark,
                hard_graph=hard_graph,
                macro_indices=left_group,
                depth=depth + 1,
            ),
            self.build_graph_hierarchy(
                benchmark=benchmark,
                hard_graph=hard_graph,
                macro_indices=right_group,
                depth=depth + 1,
            ),
        ]
        return node

    def partition_cluster(
        self,
        benchmark: Benchmark,
        hard_graph: Dict[int, Dict[int, float]],
        macro_indices: Sequence[int],
    ) -> Tuple[List[int], List[int]]:
        """
        Split one cluster into two child clusters.

        Seed choice:
        - first seed: highest weighted-degree macro in the cluster
        - second seed: macro least strongly connected to the first seed

        Assignment rule:
        - each remaining macro joins the seed it is more strongly connected to
        - if the graph gives no clear preference, we balance by current area

        This makes the partition rule easy to explain and easy to replace later.
        """
        if len(macro_indices) <= 1:
            return list(macro_indices), []

        cluster_list = list(macro_indices)
        first_seed = max(
            cluster_list,
            key=lambda idx: self.cluster_weighted_degree(hard_graph, idx, cluster_list),
        )

        second_seed = min(
            [idx for idx in cluster_list if idx != first_seed],
            key=lambda idx: (
                self.pair_weight(hard_graph, first_seed, idx),
                -self.cluster_weighted_degree(hard_graph, idx, cluster_list),
            ),
        )

        left_group = [first_seed]
        right_group = [second_seed]
        left_area = self.macro_area(benchmark, first_seed)
        right_area = self.macro_area(benchmark, second_seed)

        remaining = [
            idx for idx in cluster_list if idx not in (first_seed, second_seed)
        ]
        remaining.sort(
            key=lambda idx: -self.cluster_weighted_degree(hard_graph, idx, cluster_list)
        )

        for idx in remaining:
            left_score = self.group_affinity(hard_graph, idx, left_group)
            right_score = self.group_affinity(hard_graph, idx, right_group)

            if left_score > right_score:
                left_group.append(idx)
                left_area += self.macro_area(benchmark, idx)
                continue
            if right_score > left_score:
                right_group.append(idx)
                right_area += self.macro_area(benchmark, idx)
                continue

            # Ties are broken by current area to keep the split reasonably
            # balanced, which helps later region assignment.
            if left_area <= right_area:
                left_group.append(idx)
                left_area += self.macro_area(benchmark, idx)
            else:
                right_group.append(idx)
                right_area += self.macro_area(benchmark, idx)

        return left_group, right_group

    def cluster_weighted_degree(
        self,
        hard_graph: Dict[int, Dict[int, float]],
        macro_index: int,
        cluster_indices: Sequence[int],
    ) -> float:
        """Return the total edge weight from one macro to others in the cluster."""
        cluster_set = set(cluster_indices)
        return sum(
            weight
            for neighbor, weight in hard_graph[macro_index].items()
            if neighbor in cluster_set
        )

    def pair_weight(
        self,
        hard_graph: Dict[int, Dict[int, float]],
        left_idx: int,
        right_idx: int,
    ) -> float:
        """Return the direct graph edge weight between two hard macros."""
        return hard_graph[left_idx].get(right_idx, 0.0)

    def group_affinity(
        self,
        hard_graph: Dict[int, Dict[int, float]],
        macro_index: int,
        group_indices: Sequence[int],
    ) -> float:
        """Return how strongly one macro connects to a candidate child group."""
        return sum(self.pair_weight(hard_graph, macro_index, other) for other in group_indices)

    def assign_regions_to_hierarchy(
        self,
        node: HierarchyNode,
        region: Tuple[float, float, float, float],
        benchmark: Benchmark,
        region_by_macro: Dict[int, Tuple[float, float, float, float]],
    ) -> None:
        """
        Recursively assign canvas rectangles to the hierarchy tree.

        Region format is `(x_min, y_min, x_max, y_max)`.

        Splitting strategy:
        - even depth: vertical split
        - odd depth: horizontal split
        - split ratio follows child cluster area

        Alternating split direction is a simple way to create a hierarchy of
        placement regions without introducing another large optimization step.
        """
        if node.is_leaf:
            for macro_index in node.macro_indices:
                region_by_macro[macro_index] = region
            return

        if len(node.children) != 2:
            for macro_index in node.macro_indices:
                region_by_macro[macro_index] = region
            return

        left_child, right_child = node.children
        x_min, y_min, x_max, y_max = region
        total_area = max(left_child.total_area + right_child.total_area, 1e-6)
        left_ratio = left_child.total_area / total_area

        if node.depth % 2 == 0:
            split_x = x_min + (x_max - x_min) * left_ratio
            left_region = (x_min, y_min, split_x, y_max)
            right_region = (split_x, y_min, x_max, y_max)
        else:
            split_y = y_min + (y_max - y_min) * left_ratio
            left_region = (x_min, y_min, x_max, split_y)
            right_region = (x_min, split_y, x_max, y_max)

        self.assign_regions_to_hierarchy(
            node=left_child,
            region=left_region,
            benchmark=benchmark,
            region_by_macro=region_by_macro,
        )
        self.assign_regions_to_hierarchy(
            node=right_child,
            region=right_region,
            benchmark=benchmark,
            region_by_macro=region_by_macro,
        )

    def place_leaf_clusters_from_regions(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        hard_graph: Dict[int, Dict[int, float]],
        region_by_macro: Dict[int, Tuple[float, float, float, float]],
    ) -> None:
        """
        Turn assigned leaf regions into concrete hard-macro positions.

        We do not try to solve a perfect packing problem here. Instead we use a
        readable area-aware grid placement inside each region:
        - group macros that share the same leaf region
        - order macros by connectivity and size
        - place them on a small row/column grid inside that region

        The later legalization pass cleans up any remaining conflicts.
        """
        grouped_regions: Dict[Tuple[float, float, float, float], List[int]] = {}
        for macro_index, region in region_by_macro.items():
            grouped_regions.setdefault(region, []).append(macro_index)

        for region, macro_indices in grouped_regions.items():
            self.place_macros_in_region(
                placement=placement,
                benchmark=benchmark,
                hard_graph=hard_graph,
                macro_indices=macro_indices,
                region=region,
            )

    def place_macros_in_region(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        hard_graph: Dict[int, Dict[int, float]],
        macro_indices: Sequence[int],
        region: Tuple[float, float, float, float],
    ) -> None:
        """
        Place a small cluster of macros inside one rectangle.

        This helper intentionally favors clarity over sophistication.
        The layout rule is:
        - sort the macros so important / large ones claim good slots first
        - create a small square-ish slot grid
        - assign each macro to one slot
        - clamp to the canvas and region centerline as needed

        Because this is only an initializer, "reasonable and structured" is the
        goal. The legalization stage handles exact non-overlap repair.
        """
        ordered_indices = sorted(
            macro_indices,
            key=lambda idx: (
                -sum(hard_graph[idx].values()),
                -self.macro_area(benchmark, idx),
            ),
        )

        x_min, y_min, x_max, y_max = region
        region_width = max(x_max - x_min, 1e-6)
        region_height = max(y_max - y_min, 1e-6)

        # Build a compact grid that is roughly square. This tends to produce
        # readable cluster shapes without much code.
        num_macros = len(ordered_indices)
        num_cols = max(1, int(torch.ceil(torch.sqrt(torch.tensor(float(num_macros)))).item()))
        num_rows = max(1, (num_macros + num_cols - 1) // num_cols)

        slot_width = region_width / num_cols
        slot_height = region_height / num_rows

        for position_in_cluster, macro_index in enumerate(ordered_indices):
            row = position_in_cluster // num_cols
            col = position_in_cluster % num_cols

            slot_center_x = x_min + (col + 0.5) * slot_width
            slot_center_y = y_min + (row + 0.5) * slot_height

            candidate = torch.tensor(
                [slot_center_x, slot_center_y],
                dtype=placement.dtype,
                device=placement.device,
            )

            # Move the macro toward the region center if the slot is too close
            # to the region boundary for that macro's actual size.
            candidate = self.adjust_candidate_to_region(
                candidate=candidate,
                benchmark=benchmark,
                macro_index=macro_index,
                region=region,
            )
            candidate = self.clamp_macro_to_canvas(candidate, benchmark, macro_index)
            placement[macro_index] = candidate

    def adjust_candidate_to_region(
        self,
        candidate: torch.Tensor,
        benchmark: Benchmark,
        macro_index: int,
        region: Tuple[float, float, float, float],
    ) -> torch.Tensor:
        """
        Keep a macro center reasonably inside its assigned region.

        This is only a soft regional clamp, not a legality guarantee. It helps
        the hierarchy initializer respect cluster rectangles before the later
        global legalization pass.
        """
        x_min, y_min, x_max, y_max = region
        width = benchmark.macro_sizes[macro_index, 0].item()
        height = benchmark.macro_sizes[macro_index, 1].item()

        region_center_x = 0.5 * (x_min + x_max)
        region_center_y = 0.5 * (y_min + y_max)

        min_x = min(max(x_min + width / 2.0, x_min), x_max)
        max_x = max(min(x_max - width / 2.0, x_max), x_min)
        min_y = min(max(y_min + height / 2.0, y_min), y_max)
        max_y = max(min(y_max - height / 2.0, y_max), y_min)

        adjusted = candidate.clone()

        # If a region is smaller than the macro itself, the min/max bounds may
        # collapse. In that case we simply target the region center and let the
        # later canvas clamp and legalization stages finish the repair.
        if min_x <= max_x:
            adjusted[0] = torch.clamp(adjusted[0], min=min_x, max=max_x)
        else:
            adjusted[0] = float(region_center_x)

        if min_y <= max_y:
            adjusted[1] = torch.clamp(adjusted[1], min=min_y, max=max_y)
        else:
            adjusted[1] = float(region_center_y)

        return adjusted

    def compute_graph_wirelength_surrogate(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        hard_graph: Dict[int, Dict[int, float]],
    ) -> float:
        """
        Compute a simple graph-based wirelength surrogate.

        We sum weighted Manhattan distances over the hard-macro graph.
        Each undirected edge is counted once.
        """
        total = 0.0
        for left_idx in self.get_hard_macro_indices(benchmark):
            for right_idx, weight in hard_graph[left_idx].items():
                if right_idx <= left_idx:
                    continue

                left_pos = placement[left_idx]
                right_pos = placement[right_idx]
                distance = abs(float(left_pos[0] - right_pos[0])) + abs(
                    float(left_pos[1] - right_pos[1])
                )
                total += weight * distance

        canvas_scale = max(float(benchmark.canvas_width + benchmark.canvas_height), 1.0)
        return total / canvas_scale

    def compute_anchor_distance_penalty(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
    ) -> float:
        """
        Measure how far movable hard macros drift from the benchmark anchor.

        This term is intentionally weak. It is only there to keep the ensemble
        selector from preferring a wildly different seed unless the graph score
        is clearly better.
        """
        movable_indices = self.get_movable_hard_macro_indices(benchmark)
        if not movable_indices:
            return 0.0

        total = 0.0
        canvas_scale = max(float(benchmark.canvas_width), float(benchmark.canvas_height), 1.0)

        for macro_index in movable_indices:
            delta = placement[macro_index] - benchmark.macro_positions[macro_index]
            total += float(torch.sqrt(torch.sum(delta * delta)).item()) / canvas_scale

        return total / len(movable_indices)

    def compute_layout_spread_penalty(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
    ) -> float:
        """
        Penalize layouts whose movable hard macros are overly spread out.

        This is a tiny helper term that gently prefers compact seeds.
        """
        movable_indices = self.get_movable_hard_macro_indices(benchmark)
        if len(movable_indices) <= 1:
            return 0.0

        positions = placement[movable_indices]
        center = torch.mean(positions, dim=0)
        distances = torch.sqrt(torch.sum((positions - center) ** 2, dim=1))

        canvas_scale = max(float(benchmark.canvas_width), float(benchmark.canvas_height), 1.0)
        return float(torch.mean(distances).item()) / canvas_scale

    def run_hierarchical_seed(
        self,
        benchmark: Benchmark,
        rng: Optional[torch.Generator] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Registered wrapper around the existing graph hierarchy initializer.

        The existing implementation stays in `run_graph_hierarchy_initializer`;
        this method only gives it the shorter public operator name.
        """
        return self.run_graph_hierarchy_initializer(benchmark)

    def run_random_spread_seed(
        self,
        benchmark: Benchmark,
        rng: Optional[torch.Generator] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Place movable hard macros on shuffled coarse-grid slots with jitter.

        This gives a deterministic random baseline when `rng` is seeded. It is
        intentionally legal-ish rather than fully legal; the `legalize`
        operator should be chained afterwards when zero overlaps are required.
        """
        cfg = config or {}
        placement = self.clone_initial_placement(benchmark)
        movable_indices = self.get_movable_hard_macro_indices(benchmark)
        if not movable_indices:
            return placement

        rng = rng or torch.Generator().manual_seed(self.seed)
        ordered_indices = list(movable_indices)
        permutation = torch.randperm(len(ordered_indices), generator=rng).tolist()
        ordered_indices = [ordered_indices[pos] for pos in permutation]

        self.place_indices_on_grid(
            placement=placement,
            benchmark=benchmark,
            macro_indices=ordered_indices,
            region=(0.0, 0.0, float(benchmark.canvas_width), float(benchmark.canvas_height)),
            jitter_fraction=float(cfg.get("jitter_fraction", 0.35)),
            rng=rng,
        )
        return self.restore_fixed_macros(placement, benchmark)

    def run_grid_seed(
        self,
        benchmark: Benchmark,
        rng: Optional[torch.Generator] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Deterministic coarse-grid spread, with larger macros assigned first.
        """
        placement = self.clone_initial_placement(benchmark)
        movable_indices = self.get_movable_hard_macro_indices(benchmark)
        if not movable_indices:
            return placement

        ordered_indices = sorted(
            movable_indices,
            key=lambda idx: -self.macro_area(benchmark, idx),
        )
        self.place_indices_on_grid(
            placement=placement,
            benchmark=benchmark,
            macro_indices=ordered_indices,
            region=(0.0, 0.0, float(benchmark.canvas_width), float(benchmark.canvas_height)),
            jitter_fraction=0.0,
            rng=rng,
        )
        return self.restore_fixed_macros(placement, benchmark)

    def run_pin_aware_seed(
        self,
        benchmark: Benchmark,
        rng: Optional[torch.Generator] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Seed with grid placement, then nudge macros toward connected IO sides.
        """
        cfg = config or {}
        placement = self.run_grid_seed(benchmark, rng=rng, config=cfg)
        return self.apply_pin_aware_shift(
            placement=placement,
            benchmark=benchmark,
            strength=float(cfg.get("strength", 0.55)),
        )

    def apply_pin_aware_shift(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        strength: float = 0.35,
    ) -> torch.Tensor:
        """
        Move macros toward connected ports or, if ports are absent, neighbors.
        """
        shifted = placement.clone()
        targets, weights = self.compute_pin_attraction_targets(benchmark)
        if not targets:
            return self.restore_fixed_macros(shifted, benchmark)

        for macro_index in self.get_movable_hard_macro_indices(benchmark):
            target = targets.get(macro_index)
            if target is None:
                continue

            weight = min(1.0, max(0.0, weights.get(macro_index, 0.0)))
            local_strength = strength * (0.35 + 0.65 * weight)
            candidate = shifted[macro_index] * (1.0 - local_strength) + target * local_strength
            shifted[macro_index] = self.clamp_macro_to_canvas(candidate, benchmark, macro_index)

        return self.restore_fixed_macros(shifted, benchmark)

    def compute_port_affinity_weights(self, benchmark: Benchmark) -> Dict[int, float]:
        """Return normalized hard-macro affinity to explicit boundary ports."""
        port_count = int(benchmark.port_positions.shape[0])
        if port_count <= 0:
            return {}

        raw_weights: Dict[int, float] = {}
        for net_pos, net_nodes in enumerate(benchmark.net_nodes):
            nodes = [int(node) for node in net_nodes.tolist()]
            hard_nodes = sorted(
                {node for node in nodes if 0 <= node < benchmark.num_hard_macros}
            )
            if not hard_nodes:
                continue

            port_offsets = [
                node - benchmark.num_macros
                for node in nodes
                if benchmark.num_macros <= node < benchmark.num_macros + port_count
            ]
            if not port_offsets:
                continue

            net_weight = (
                float(benchmark.net_weights[net_pos])
                if net_pos < int(benchmark.net_weights.shape[0])
                else 1.0
            )
            contribution = net_weight * float(len(port_offsets)) / max(float(len(hard_nodes)), 1.0)
            for macro_index in hard_nodes:
                raw_weights[macro_index] = raw_weights.get(macro_index, 0.0) + contribution

        max_weight = max(raw_weights.values()) if raw_weights else 0.0
        if max_weight <= 1e-12:
            return {}
        return {idx: value / max_weight for idx, value in raw_weights.items()}

    def compute_periphery_scores(
        self,
        benchmark: Benchmark,
        config: Optional[Dict[str, Any]] = None,
        placement: Optional[torch.Tensor] = None,
        hard_graph: Optional[Dict[int, Dict[int, float]]] = None,
        targets: Optional[Dict[int, torch.Tensor]] = None,
        pin_weights: Optional[Dict[int, float]] = None,
    ) -> Dict[int, float]:
        """
        Score movable hard macros for soft periphery bias.

        The score favors large macros with genuine boundary/IO affinity and
        penalizes internally connected macros whose pin target is central. This
        keeps high graph degree from automatically meaning "move to the edge."
        """
        cfg = config or {}
        movable_indices = self.get_movable_hard_macro_indices(benchmark)
        if not movable_indices:
            return {}

        hard_graph = hard_graph or self.build_hard_macro_graph(benchmark)
        if targets is None or pin_weights is None:
            targets, pin_weights = self.compute_pin_attraction_targets(benchmark)
        placement = placement if placement is not None else benchmark.macro_positions
        port_weights = self.compute_port_affinity_weights(benchmark)

        area_weight = float(cfg.get("area_weight", 0.30))
        io_weight = float(cfg.get("io_weight", 0.34))
        degree_weight = float(cfg.get("degree_weight", 0.18))
        boundary_target_weight = float(cfg.get("boundary_target_weight", 0.24))
        size_weight = float(cfg.get("size_weight", 0.10))
        centrality_penalty = float(cfg.get("centrality_penalty", 0.38))
        center_penalty = float(cfg.get("center_penalty", 0.16))

        max_area = max(self.macro_area(benchmark, idx) for idx in movable_indices)
        degrees = {idx: sum(hard_graph.get(idx, {}).values()) for idx in movable_indices}
        max_degree = max(max(degrees.values()), 1.0)

        canvas_width = float(benchmark.canvas_width)
        canvas_height = float(benchmark.canvas_height)
        edge_span = max(0.5 * min(canvas_width, canvas_height), 1e-6)
        center = torch.tensor(
            [0.5 * canvas_width, 0.5 * canvas_height],
            dtype=placement.dtype,
            device=placement.device,
        )
        max_center_distance = max(
            math.sqrt((0.5 * canvas_width) ** 2 + (0.5 * canvas_height) ** 2),
            1e-6,
        )

        raw_scores: Dict[int, float] = {}
        for idx in movable_indices:
            area_norm = self.macro_area(benchmark, idx) / max(max_area, 1e-6)
            degree_norm = degrees.get(idx, 0.0) / max_degree
            port_norm = port_weights.get(idx, 0.0)
            fallback_pin_norm = pin_weights.get(idx, 0.0) if pin_weights else 0.0
            io_norm = max(port_norm, 0.35 * fallback_pin_norm)

            reference = targets.get(idx) if targets else None
            if reference is None:
                reference = placement[idx]
            reference = reference.to(dtype=placement.dtype, device=placement.device)
            x_pos = min(max(float(reference[0]), 0.0), canvas_width)
            y_pos = min(max(float(reference[1]), 0.0), canvas_height)

            edge_distance = min(x_pos, canvas_width - x_pos, y_pos, canvas_height - y_pos)
            target_edge_affinity = max(0.0, min(1.0, 1.0 - edge_distance / edge_span))
            center_distance = float(torch.norm(reference - center).item())
            center_affinity = max(0.0, min(1.0, 1.0 - center_distance / max_center_distance))

            width = float(benchmark.macro_sizes[idx, 0])
            height = float(benchmark.macro_sizes[idx, 1])
            side_pressure = max(width / max(canvas_width, 1e-6), height / max(canvas_height, 1e-6))
            size_pressure = 0.65 * area_norm + 0.35 * min(side_pressure, 1.0)

            boundary_degree = degree_norm * max(io_norm, 0.35 * target_edge_affinity)
            internal_centrality = degree_norm * (1.0 - io_norm) * (0.45 + 0.55 * center_affinity)
            raw_scores[idx] = (
                area_weight * area_norm
                + io_weight * io_norm
                + boundary_target_weight * target_edge_affinity
                + degree_weight * boundary_degree
                + size_weight * size_pressure
                - centrality_penalty * internal_centrality
                - center_penalty * center_affinity * (1.0 - max(io_norm, target_edge_affinity))
            )

        positives = {idx: max(0.0, score) for idx, score in raw_scores.items()}
        max_positive = max(positives.values()) if positives else 0.0
        if max_positive <= 1e-12:
            return {idx: 0.0 for idx in movable_indices}
        return {idx: min(1.0, positives[idx] / max_positive) for idx in movable_indices}

    def boundary_side_preferences(
        self,
        benchmark: Benchmark,
        macro_indices: Sequence[int],
        targets: Dict[int, torch.Tensor],
        placement: Optional[torch.Tensor] = None,
    ) -> Dict[int, List[str]]:
        """Rank boundary sides for each macro by its pin target or current position."""
        sides = ["left", "right", "bottom", "top"]
        placement = placement if placement is not None else benchmark.macro_positions
        preferences: Dict[int, List[str]] = {}
        for macro_index in macro_indices:
            reference = targets.get(macro_index)
            if reference is None:
                reference = placement[macro_index]
            x_pos = float(reference[0])
            y_pos = float(reference[1])
            distances = {
                "left": x_pos,
                "right": float(benchmark.canvas_width) - x_pos,
                "bottom": y_pos,
                "top": float(benchmark.canvas_height) - y_pos,
            }
            preferred = self.choose_boundary_side(benchmark, macro_index, targets)
            ordered = sorted(sides, key=lambda side: (distances[side], sides.index(side)))
            if preferred in ordered:
                ordered.remove(preferred)
                ordered.insert(0, preferred)
            preferences[macro_index] = ordered
        return preferences

    def estimate_side_capacities(
        self,
        benchmark: Benchmark,
        macro_indices: Sequence[int],
        min_side_gap_fraction: float = 0.035,
    ) -> Dict[str, int]:
        """Estimate rough side quotas so peripheral seeds do not overload one edge."""
        sides = ["left", "right", "bottom", "top"]
        count = len(macro_indices)
        if count <= 0:
            return {side: 0 for side in sides}

        canvas_width = float(benchmark.canvas_width)
        canvas_height = float(benchmark.canvas_height)
        gap = max(0.0, min_side_gap_fraction) * min(canvas_width, canvas_height)
        avg_width = sum(float(benchmark.macro_sizes[idx, 0]) for idx in macro_indices) / count
        avg_height = sum(float(benchmark.macro_sizes[idx, 1]) for idx in macro_indices) / count
        physical = {
            "left": max(1, int(canvas_height / max(avg_height + gap, 1e-6))),
            "right": max(1, int(canvas_height / max(avg_height + gap, 1e-6))),
            "bottom": max(1, int(canvas_width / max(avg_width + gap, 1e-6))),
            "top": max(1, int(canvas_width / max(avg_width + gap, 1e-6))),
        }
        side_lengths = {
            "left": canvas_height,
            "right": canvas_height,
            "bottom": canvas_width,
            "top": canvas_width,
        }
        total_length = max(sum(side_lengths.values()), 1e-6)
        capacities = {
            side: min(
                physical[side],
                max(1, int(round(count * side_lengths[side] / total_length))),
            )
            for side in sides
        }

        while sum(capacities.values()) < count:
            expandable = [
                side for side in sides if capacities[side] < max(physical[side], capacities[side])
            ]
            if not expandable:
                expandable = sides
            side = max(
                expandable,
                key=lambda item: (
                    physical[item] - capacities[item],
                    side_lengths[item],
                    -sides.index(item),
                ),
            )
            capacities[side] += 1

        return capacities

    def choose_capacity_aware_sides(
        self,
        benchmark: Benchmark,
        macro_indices: Sequence[int],
        preferences: Dict[int, List[str]],
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[int, str]:
        """Assign macros to preferred sides while respecting rough side quotas."""
        cfg = config or {}
        sides = ["left", "right", "bottom", "top"]
        capacities = self.estimate_side_capacities(
            benchmark,
            macro_indices,
            min_side_gap_fraction=float(cfg.get("min_side_gap_fraction", 0.035)),
        )
        usage = {side: 0 for side in sides}
        assignments: Dict[int, str] = {}
        for macro_index in macro_indices:
            assigned_side: Optional[str] = None
            for side in preferences.get(macro_index, sides):
                if usage[side] < capacities[side]:
                    assigned_side = side
                    break
            if assigned_side is None:
                assigned_side = min(
                    sides,
                    key=lambda side: (
                        usage[side] / max(float(capacities[side]), 1.0),
                        usage[side],
                        sides.index(side),
                    ),
                )
            assignments[macro_index] = assigned_side
            usage[assigned_side] += 1
        return assignments

    def boundary_target_for_side(
        self,
        benchmark: Benchmark,
        macro_index: int,
        side: str,
        placement: torch.Tensor,
        targets: Dict[int, torch.Tensor],
        config: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Build a clamped boundary target that preserves along-side affinity."""
        cfg = config or {}
        canvas_width = float(benchmark.canvas_width)
        canvas_height = float(benchmark.canvas_height)
        gap = float(cfg.get("min_side_gap_fraction", 0.035)) * min(canvas_width, canvas_height)
        pullback = float(cfg.get("boundary_pullback_fraction", 0.0)) * min(
            canvas_width,
            canvas_height,
        )
        width = float(benchmark.macro_sizes[macro_index, 0])
        height = float(benchmark.macro_sizes[macro_index, 1])
        reference = targets.get(macro_index)
        if reference is None:
            reference = placement[macro_index]
        reference = reference.to(dtype=placement.dtype, device=placement.device)

        def clamp_along(value: float, low: float, high: float) -> float:
            if low > high:
                return 0.5 * (low + high)
            return min(max(value, low), high)

        if side == "left":
            center = torch.tensor(
                [
                    width / 2.0 + pullback,
                    clamp_along(
                        float(reference[1]),
                        height / 2.0 + gap,
                        canvas_height - height / 2.0 - gap,
                    ),
                ],
                dtype=placement.dtype,
                device=placement.device,
            )
        elif side == "right":
            center = torch.tensor(
                [
                    canvas_width - width / 2.0 - pullback,
                    clamp_along(
                        float(reference[1]),
                        height / 2.0 + gap,
                        canvas_height - height / 2.0 - gap,
                    ),
                ],
                dtype=placement.dtype,
                device=placement.device,
            )
        elif side == "bottom":
            center = torch.tensor(
                [
                    clamp_along(
                        float(reference[0]),
                        width / 2.0 + gap,
                        canvas_width - width / 2.0 - gap,
                    ),
                    height / 2.0 + pullback,
                ],
                dtype=placement.dtype,
                device=placement.device,
            )
        else:
            center = torch.tensor(
                [
                    clamp_along(
                        float(reference[0]),
                        width / 2.0 + gap,
                        canvas_width - width / 2.0 - gap,
                    ),
                    canvas_height - height / 2.0 - pullback,
                ],
                dtype=placement.dtype,
                device=placement.device,
            )
        return self.clamp_macro_to_canvas(center, benchmark, macro_index)

    def run_peripheral_seed(
        self,
        benchmark: Benchmark,
        rng: Optional[torch.Generator] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Put important or IO-heavy macros near the boundary, then grid-fill.
        """
        cfg = config or {}
        placement = self.run_grid_seed(benchmark, rng=rng, config=cfg)
        movable_indices = self.get_movable_hard_macro_indices(benchmark)
        if not movable_indices:
            return placement

        hard_graph = self.build_hard_macro_graph(benchmark)
        targets, pin_weights = self.compute_pin_attraction_targets(benchmark)
        periphery_scores = self.compute_periphery_scores(
            benchmark=benchmark,
            config=cfg,
            placement=placement,
            hard_graph=hard_graph,
            targets=targets,
            pin_weights=pin_weights,
        )

        boundary_fraction = float(cfg.get("boundary_fraction", 0.35))
        requested_count = int(math.ceil(len(movable_indices) * boundary_fraction))
        boundary_count = int(cfg.get("boundary_count", requested_count))
        boundary_count = max(0, min(len(movable_indices), boundary_count))

        scored = sorted(
            movable_indices,
            key=lambda idx: (
                -periphery_scores.get(idx, 0.0),
                -self.macro_area(benchmark, idx),
                idx,
            ),
        )
        boundary_indices = scored[:boundary_count]
        interior_indices = scored[boundary_count:]

        side_groups: Dict[str, List[int]] = {"left": [], "right": [], "bottom": [], "top": []}
        preferences = self.boundary_side_preferences(
            benchmark,
            boundary_indices,
            targets,
            placement=placement,
        )
        side_assignments = self.choose_capacity_aware_sides(
            benchmark,
            boundary_indices,
            preferences,
            config=cfg,
        )
        for idx in boundary_indices:
            side_groups[side_assignments[idx]].append(idx)

        for side, indices in side_groups.items():
            indices.sort(key=lambda idx: self.side_sort_key(benchmark, idx, side, targets))
            self.place_indices_on_side(
                placement,
                benchmark,
                indices,
                side,
                min_side_gap_fraction=float(cfg.get("min_side_gap_fraction", 0.035)),
            )

        if interior_indices:
            margin = float(cfg.get("interior_margin", self.estimate_peripheral_margin(benchmark)))
            region = (
                margin,
                margin,
                max(margin, float(benchmark.canvas_width) - margin),
                max(margin, float(benchmark.canvas_height) - margin),
            )
            self.place_indices_on_grid(
                placement=placement,
                benchmark=benchmark,
                macro_indices=interior_indices,
                region=region,
                jitter_fraction=0.0,
                rng=rng,
            )

        return self.restore_fixed_macros(placement, benchmark)

    def run_periphery_bias(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        rng: Optional[torch.Generator] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Softly move high periphery-score macros toward capacity-aware sides.

        This is a transform, not a seed: it preserves the input topology and
        leaves hard legality to later repair stages.
        """
        cfg = config or {}
        biased = placement.clone()
        movable_indices = self.get_movable_hard_macro_indices(benchmark)
        if not movable_indices:
            return self.restore_fixed_macros(biased, benchmark)

        hard_graph = self.build_hard_macro_graph(benchmark)
        targets, pin_weights = self.compute_pin_attraction_targets(benchmark)
        periphery_scores = self.compute_periphery_scores(
            benchmark=benchmark,
            config=cfg,
            placement=biased,
            hard_graph=hard_graph,
            targets=targets,
            pin_weights=pin_weights,
        )

        boundary_fraction = float(cfg.get("boundary_fraction", 0.12))
        requested_count = int(math.ceil(len(movable_indices) * boundary_fraction))
        boundary_count = int(cfg.get("boundary_count", requested_count))
        boundary_count = max(0, min(len(movable_indices), boundary_count))
        if boundary_count <= 0:
            return self.restore_fixed_macros(biased, benchmark)

        min_score = float(cfg.get("min_score", 0.20))
        selected = [
            idx
            for idx in sorted(
                movable_indices,
                key=lambda item: (
                    -periphery_scores.get(item, 0.0),
                    -self.macro_area(benchmark, item),
                    item,
                ),
            )
            if periphery_scores.get(idx, 0.0) >= min_score
        ][:boundary_count]
        if not selected:
            return self.restore_fixed_macros(biased, benchmark)

        preferences = self.boundary_side_preferences(
            benchmark,
            selected,
            targets,
            placement=biased,
        )
        side_assignments = self.choose_capacity_aware_sides(
            benchmark,
            selected,
            preferences,
            config=cfg,
        )

        canvas_scale = max(float(benchmark.canvas_width), float(benchmark.canvas_height), 1.0)
        max_move = float(
            cfg.get("max_move", float(cfg.get("max_move_fraction", 0.030)) * canvas_scale)
        )
        strength = max(0.0, min(1.0, float(cfg.get("strength", 0.18))))

        for macro_index in selected:
            side = side_assignments[macro_index]
            target = self.boundary_target_for_side(
                benchmark,
                macro_index,
                side,
                biased,
                targets,
                config=cfg,
            )
            score = periphery_scores.get(macro_index, 0.0)
            local_strength = strength * (0.25 + 0.75 * score)
            move = (target - biased[macro_index]) * local_strength
            norm = torch.norm(move).item()
            if norm > max_move:
                move = move * (max_move / max(norm, 1e-9))
            biased[macro_index] = self.clamp_macro_to_canvas(
                biased[macro_index] + move,
                benchmark,
                macro_index,
            )

        return self.restore_fixed_macros(biased, benchmark)

    def run_spectral_order(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        rng: Optional[torch.Generator] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Reassign current slots according to a 1D Laplacian/Fiedler ordering.

        This deliberately uses spectral ordering as a cheap local ordering
        heuristic. It keeps the existing slot geometry and only changes which
        macro occupies which slot.
        """
        cfg = config or {}
        reordered = placement.clone()
        max_macros = int(cfg.get("max_macros", 320))
        if benchmark.num_hard_macros > max_macros:
            if bool(cfg.get("raise_on_skip", False)):
                raise RuntimeError(
                    f"spectral_order skipped: {benchmark.num_hard_macros} hard macros "
                    f"exceeds max_macros={max_macros}"
                )
            return self.restore_fixed_macros(reordered, benchmark)
        movable_indices = self.get_movable_hard_macro_indices(benchmark)
        if len(movable_indices) <= 2:
            return self.restore_fixed_macros(reordered, benchmark)

        hard_graph = self.build_hard_macro_graph(benchmark)
        spectral_values = self.compute_spectral_values(hard_graph, movable_indices)
        if spectral_values is None:
            raise RuntimeError("spectral_order could not build a connected macro graph")

        positions = reordered[movable_indices].clone()
        spread = torch.max(positions, dim=0).values - torch.min(positions, dim=0).values
        primary_axis = 0 if float(spread[0]) >= float(spread[1]) else 1
        secondary_axis = 1 - primary_axis

        macro_order = [
            idx
            for _, idx in sorted(
                zip(spectral_values.tolist(), movable_indices),
                key=lambda item: (item[0], item[1]),
            )
        ]
        slot_order = sorted(
            range(len(movable_indices)),
            key=lambda pos_idx: (
                float(positions[pos_idx, primary_axis]),
                float(positions[pos_idx, secondary_axis]),
            ),
        )

        for macro_index, slot_pos in zip(macro_order, slot_order):
            reordered[macro_index] = positions[slot_pos]

        return self.restore_fixed_macros(reordered, benchmark)

    def run_force_smooth(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        rng: Optional[torch.Generator] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Lightweight connectivity attraction plus overlap/boundary repulsion.
        """
        cfg = config or {}
        smoothed = placement.clone()
        iterations = int(cfg.get("iterations", 20))
        if iterations <= 0:
            return self.restore_fixed_macros(smoothed, benchmark)
        attraction = float(cfg.get("attraction", 0.10))
        repulsion = float(cfg.get("repulsion", 0.45))
        canvas_scale = max(float(benchmark.canvas_width), float(benchmark.canvas_height), 1.0)
        max_move = float(cfg.get("max_move", 0.020 * canvas_scale))

        hard_graph = self.build_hard_macro_graph(benchmark)
        hard_indices = self.get_hard_macro_indices(benchmark)
        movable_set = set(self.get_movable_hard_macro_indices(benchmark))
        edges = self.graph_edges(hard_graph)

        deadline = cfg.get("deadline_sec")
        for _ in range(iterations):
            if deadline is not None and time.perf_counter() >= float(deadline):
                break
            forces = torch.zeros_like(smoothed)

            for left_idx, right_idx, weight in edges:
                delta = smoothed[right_idx] - smoothed[left_idx]
                force = attraction * weight * delta / max(canvas_scale, 1e-6)
                if left_idx in movable_set:
                    forces[left_idx] = forces[left_idx] + force
                if right_idx in movable_set:
                    forces[right_idx] = forces[right_idx] - force

            self.add_overlap_repulsion(
                placement=smoothed,
                benchmark=benchmark,
                hard_indices=hard_indices,
                movable_set=movable_set,
                forces=forces,
                strength=repulsion,
            )

            self.apply_forces(smoothed, benchmark, hard_indices, movable_set, forces, max_move)

        return self.restore_fixed_macros(smoothed, benchmark)

    def run_analytical_stage1(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        rng: Optional[torch.Generator] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Continuous physical Stage-1 relaxation before hard legalization.

        The loop combines graph attraction, overlap/crowding repulsion, density
        overflow pressure, boundary cleanup, and optional periphery attraction.
        It intentionally does not legalize; later repair stages commit the soft
        placement to a zero-overlap result.
        """
        cfg = config or {}
        relaxed = placement.clone()
        iterations = int(cfg.get("iterations", 24))
        if iterations <= 0:
            return self.restore_fixed_macros(relaxed, benchmark)

        hard_indices = self.get_hard_macro_indices(benchmark)
        movable_set = set(self.get_movable_hard_macro_indices(benchmark))
        if not movable_set:
            return self.restore_fixed_macros(relaxed, benchmark)

        hard_graph = self.build_hard_macro_graph(benchmark)
        edges = self.graph_edges(hard_graph)
        max_edge_weight = max((weight for _left, _right, weight in edges), default=1.0)
        max_edge_weight = max(max_edge_weight, 1.0)

        canvas_scale = max(float(benchmark.canvas_width), float(benchmark.canvas_height), 1.0)
        max_move = float(cfg.get("max_move", 0.020 * canvas_scale))
        attraction = float(cfg.get("attraction", 0.035))
        overlap_repulsion = float(cfg.get("overlap_repulsion", 0.95))
        density_repulsion = float(cfg.get("density_repulsion", 0.060))
        boundary_repulsion = float(cfg.get("boundary_repulsion", 0.10))
        periphery_attraction = float(cfg.get("periphery_attraction", 0.0))
        spread_repulsion = float(cfg.get("spread_repulsion", 0.12))
        bin_count = int(cfg.get("bin_count", cfg.get("density_grid_size", 7)))
        target_density = float(cfg.get("target_density", 0.72))
        max_pairwise_macros = int(cfg.get("max_pairwise_macros", 360))

        periphery_scores: Dict[int, float] = {}
        side_assignments: Dict[int, str] = {}
        targets: Dict[int, torch.Tensor] = {}
        if periphery_attraction > 0.0:
            targets, pin_weights = self.compute_pin_attraction_targets(benchmark)
            periphery_scores = self.compute_periphery_scores(
                benchmark=benchmark,
                config=cfg,
                placement=relaxed,
                hard_graph=hard_graph,
                targets=targets,
                pin_weights=pin_weights,
            )
            boundary_fraction = float(cfg.get("boundary_fraction", 0.20))
            periphery_count = int(
                cfg.get("boundary_count", math.ceil(len(movable_set) * boundary_fraction))
            )
            periphery_count = max(0, min(len(movable_set), periphery_count))
            periphery_indices = sorted(
                movable_set,
                key=lambda item: (
                    -periphery_scores.get(item, 0.0),
                    -self.macro_area(benchmark, item),
                    item,
                ),
            )[:periphery_count]
            preferences = self.boundary_side_preferences(
                benchmark,
                periphery_indices,
                targets,
                placement=relaxed,
            )
            side_assignments = self.choose_capacity_aware_sides(
                benchmark,
                periphery_indices,
                preferences,
                config=cfg,
            )

        deadline = cfg.get("deadline_sec")
        for iteration in range(iterations):
            if deadline is not None and time.perf_counter() >= float(deadline):
                break
            forces = torch.zeros_like(relaxed)
            anneal = 0.55 + 0.45 * (1.0 - iteration / max(float(iterations), 1.0))

            for left_idx, right_idx, weight in edges:
                normalized_weight = min(float(weight) / max_edge_weight, 4.0)
                delta = relaxed[right_idx] - relaxed[left_idx]
                force = attraction * normalized_weight * delta / max(canvas_scale, 1e-6)
                if left_idx in movable_set:
                    forces[left_idx] = forces[left_idx] + force
                if right_idx in movable_set:
                    forces[right_idx] = forces[right_idx] - force

            if overlap_repulsion > 0.0 and len(hard_indices) <= max_pairwise_macros:
                self.add_overlap_repulsion(
                    placement=relaxed,
                    benchmark=benchmark,
                    hard_indices=hard_indices,
                    movable_set=movable_set,
                    forces=forces,
                    strength=overlap_repulsion,
                )

            if density_repulsion > 0.0:
                self.add_density_repulsion(
                    placement=relaxed,
                    benchmark=benchmark,
                    hard_indices=hard_indices,
                    movable_set=movable_set,
                    forces=forces,
                    strength=density_repulsion,
                    bin_count=bin_count,
                    target_density=target_density,
                )

            if boundary_repulsion > 0.0:
                self.add_boundary_repulsion(
                    placement=relaxed,
                    benchmark=benchmark,
                    hard_indices=hard_indices,
                    movable_set=movable_set,
                    forces=forces,
                    strength=boundary_repulsion,
                    margin_fraction=float(cfg.get("boundary_margin_fraction", 0.014)),
                )

            if periphery_attraction > 0.0 and side_assignments:
                for macro_index, side in side_assignments.items():
                    if macro_index not in movable_set:
                        continue
                    score = periphery_scores.get(macro_index, 0.0)
                    if score <= float(cfg.get("periphery_min_score", 0.08)):
                        continue
                    target = self.boundary_target_for_side(
                        benchmark,
                        macro_index,
                        side,
                        relaxed,
                        targets,
                        config=cfg,
                    )
                    forces[macro_index] = forces[macro_index] + (
                        periphery_attraction * score * (target - relaxed[macro_index])
                    )

            if spread_repulsion > 0.0 and len(hard_indices) <= max_pairwise_macros:
                self.add_spread_repulsion(
                    placement=relaxed,
                    benchmark=benchmark,
                    hard_indices=hard_indices,
                    movable_set=movable_set,
                    forces=forces,
                    strength=spread_repulsion,
                    distance_factor=float(cfg.get("spread_distance_factor", 0.55)),
                )

            self.apply_forces(
                relaxed,
                benchmark,
                hard_indices,
                movable_set,
                forces,
                max_move=max_move * anneal,
            )
            relaxed = self.restore_fixed_macros(relaxed, benchmark)

        return self.restore_fixed_macros(relaxed, benchmark)

    def add_density_repulsion(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        hard_indices: Sequence[int],
        movable_set: Iterable[int],
        forces: torch.Tensor,
        strength: float,
        bin_count: int,
        target_density: float,
    ) -> None:
        """Push macros out of overfull coarse density bins."""
        bin_count = max(1, min(int(bin_count), 32))
        movable_lookup = set(movable_set)
        canvas_width = float(benchmark.canvas_width)
        canvas_height = float(benchmark.canvas_height)
        bin_width = canvas_width / bin_count
        bin_height = canvas_height / bin_count
        bin_area = max(bin_width * bin_height, 1e-6)

        utilizations = np.zeros((bin_count, bin_count), dtype=np.float64)
        members: Dict[Tuple[int, int], List[int]] = {}
        for macro_index in hard_indices:
            x_pos = min(max(float(placement[macro_index, 0]), 0.0), canvas_width - 1e-9)
            y_pos = min(max(float(placement[macro_index, 1]), 0.0), canvas_height - 1e-9)
            col = min(bin_count - 1, max(0, int(x_pos / max(bin_width, 1e-6))))
            row = min(bin_count - 1, max(0, int(y_pos / max(bin_height, 1e-6))))
            utilizations[row, col] += self.macro_area(benchmark, macro_index) / bin_area
            members.setdefault((row, col), []).append(macro_index)

        for (row, col), indices in members.items():
            overflow = float(utilizations[row, col]) - target_density
            if overflow <= 0.0:
                continue
            bin_center = torch.tensor(
                [(col + 0.5) * bin_width, (row + 0.5) * bin_height],
                dtype=placement.dtype,
                device=placement.device,
            )
            for macro_index in indices:
                if macro_index not in movable_lookup:
                    continue
                away = placement[macro_index] - bin_center
                norm = torch.norm(away).item()
                if norm <= 1e-9:
                    angle = 2.399963229728653 * float(macro_index + row * bin_count + col)
                    away = torch.tensor(
                        [math.cos(angle), math.sin(angle)],
                        dtype=placement.dtype,
                        device=placement.device,
                    )
                    norm = 1.0
                macro_scale = max(float(torch.max(benchmark.macro_sizes[macro_index]).item()), 1.0)
                magnitude = strength * min(overflow, 3.0) * macro_scale
                forces[macro_index] = forces[macro_index] + away * (magnitude / norm)

    def add_boundary_repulsion(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        hard_indices: Sequence[int],
        movable_set: Iterable[int],
        forces: torch.Tensor,
        strength: float,
        margin_fraction: float,
    ) -> None:
        """Accumulate soft inward forces for macros near or outside the canvas."""
        movable_lookup = set(movable_set)
        canvas_width = float(benchmark.canvas_width)
        canvas_height = float(benchmark.canvas_height)
        margin = max(0.0, margin_fraction) * min(canvas_width, canvas_height)
        for macro_index in hard_indices:
            if macro_index not in movable_lookup:
                continue
            width = float(benchmark.macro_sizes[macro_index, 0])
            height = float(benchmark.macro_sizes[macro_index, 1])
            x_pos = float(placement[macro_index, 0])
            y_pos = float(placement[macro_index, 1])
            left_gap = x_pos - width / 2.0
            right_gap = canvas_width - (x_pos + width / 2.0)
            bottom_gap = y_pos - height / 2.0
            top_gap = canvas_height - (y_pos + height / 2.0)
            if left_gap < margin:
                forces[macro_index, 0] += strength * (margin - left_gap)
            if right_gap < margin:
                forces[macro_index, 0] -= strength * (margin - right_gap)
            if bottom_gap < margin:
                forces[macro_index, 1] += strength * (margin - bottom_gap)
            if top_gap < margin:
                forces[macro_index, 1] -= strength * (margin - top_gap)

    def add_spread_repulsion(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        hard_indices: Sequence[int],
        movable_set: Iterable[int],
        forces: torch.Tensor,
        strength: float,
        distance_factor: float,
    ) -> None:
        """Apply weak anti-collapse forces before boxes actually overlap."""
        movable_lookup = set(movable_set)
        for left_pos, left_idx in enumerate(hard_indices):
            left_center = placement[left_idx]
            left_w, left_h = benchmark.macro_sizes[left_idx].tolist()
            for right_idx in hard_indices[left_pos + 1 :]:
                if left_idx not in movable_lookup and right_idx not in movable_lookup:
                    continue
                right_center = placement[right_idx]
                right_w, right_h = benchmark.macro_sizes[right_idx].tolist()
                delta = right_center - left_center
                distance = torch.norm(delta).item()
                radius = distance_factor * max(left_w + right_w, left_h + right_h)
                if distance >= radius:
                    continue
                if distance <= 1e-9:
                    angle = 2.399963229728653 * float(left_idx + right_idx)
                    direction = torch.tensor(
                        [math.cos(angle), math.sin(angle)],
                        dtype=placement.dtype,
                        device=placement.device,
                    )
                    distance = 1.0
                else:
                    direction = delta / distance
                magnitude = strength * (radius - distance)
                force = direction * magnitude
                if left_idx in movable_lookup:
                    forces[left_idx] = forces[left_idx] - force
                if right_idx in movable_lookup:
                    forces[right_idx] = forces[right_idx] + force

    def run_macro_spread(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        rng: Optional[torch.Generator] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Push overlapping or densely clustered hard macros apart.
        """
        cfg = config or {}
        spread = placement.clone()
        iterations = int(cfg.get("iterations", 30))
        if iterations <= 0:
            return self.restore_fixed_macros(spread, benchmark)
        strength = float(cfg.get("strength", 0.90))
        canvas_scale = max(float(benchmark.canvas_width), float(benchmark.canvas_height), 1.0)
        max_move = float(cfg.get("max_move", 0.030 * canvas_scale))

        hard_indices = self.get_hard_macro_indices(benchmark)
        movable_set = set(self.get_movable_hard_macro_indices(benchmark))
        deadline = cfg.get("deadline_sec")
        for _ in range(iterations):
            if deadline is not None and time.perf_counter() >= float(deadline):
                break
            forces = torch.zeros_like(spread)
            self.add_overlap_repulsion(
                placement=spread,
                benchmark=benchmark,
                hard_indices=hard_indices,
                movable_set=movable_set,
                forces=forces,
                strength=strength,
            )

            positions = spread[list(movable_set)] if movable_set else torch.zeros(0, 2)
            if positions.numel() > 0:
                center = torch.mean(positions, dim=0)
                for idx in movable_set:
                    away = spread[idx] - center
                    norm = torch.norm(away).item()
                    if norm > 1e-6:
                        forces[idx] = forces[idx] + 0.03 * away / norm

            self.apply_forces(spread, benchmark, hard_indices, movable_set, forces, max_move)

        return self.restore_fixed_macros(spread, benchmark)

    def run_legalize_operator(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        rng: Optional[torch.Generator] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Repair bounds and hard-macro overlaps with the existing CorePlacer pass.
        """
        legalized = self.legalize_initial_placement(placement.clone(), benchmark)
        return self.restore_fixed_macros(legalized, benchmark)

    def run_local_swap(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        rng: Optional[torch.Generator] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Small-budget greedy pair swap refinement using the graph surrogate.
        """
        cfg = config or {}
        rng = rng or torch.Generator().manual_seed(self.seed)
        refined = placement.clone()
        iterations = int(cfg.get("iterations", 80))
        if iterations <= 0:
            return self.restore_fixed_macros(refined, benchmark)
        movable_indices = self.get_movable_hard_macro_indices(benchmark)
        if len(movable_indices) < 2:
            return self.restore_fixed_macros(refined, benchmark)

        hard_graph = self.build_hard_macro_graph(benchmark)
        default_cap = max(1, len(movable_indices) * int(cfg.get("iteration_cap_multiplier", 2)))
        iterations = min(iterations, int(cfg.get("max_iterations", default_cap)))
        require_legal = bool(cfg.get("require_legal", True))
        best_score = self.operator_score(refined, benchmark, hard_graph, cfg)

        deadline = cfg.get("deadline_sec")
        for _ in range(iterations):
            if deadline is not None and time.perf_counter() >= float(deadline):
                break
            left_pos = int(torch.randint(len(movable_indices), (1,), generator=rng).item())
            right_pos = int(torch.randint(len(movable_indices), (1,), generator=rng).item())
            if left_pos == right_pos:
                continue

            left_idx = movable_indices[left_pos]
            right_idx = movable_indices[right_pos]
            old_left = refined[left_idx].clone()
            old_right = refined[right_idx].clone()
            refined[left_idx] = self.clamp_macro_to_canvas(old_right, benchmark, left_idx)
            refined[right_idx] = self.clamp_macro_to_canvas(old_left, benchmark, right_idx)

            if require_legal and not self.changed_macros_are_legal(
                refined,
                benchmark,
                [left_idx, right_idx],
            ):
                refined[left_idx] = old_left
                refined[right_idx] = old_right
                continue

            score = self.operator_score(refined, benchmark, hard_graph, cfg)
            if score < best_score:
                best_score = score
            else:
                refined[left_idx] = old_left
                refined[right_idx] = old_right

        return self.restore_fixed_macros(refined, benchmark)

    def run_local_shift(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        rng: Optional[torch.Generator] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Greedy small local moves around the current placement.
        """
        cfg = config or {}
        refined = placement.clone()
        iterations = int(cfg.get("iterations", 2))
        if iterations <= 0:
            return self.restore_fixed_macros(refined, benchmark)
        movable_indices = self.get_movable_hard_macro_indices(benchmark)
        if not movable_indices:
            return self.restore_fixed_macros(refined, benchmark)

        hard_graph = self.build_hard_macro_graph(benchmark)
        shift_fraction = float(cfg.get("shift_fraction", 0.50))
        require_legal = bool(cfg.get("require_legal", True))
        best_score = self.operator_score(refined, benchmark, hard_graph, cfg)

        ordered_indices = sorted(
            movable_indices,
            key=lambda idx: -sum(hard_graph[idx].values()),
        )
        deadline = cfg.get("deadline_sec")
        for _ in range(iterations):
            if deadline is not None and time.perf_counter() >= float(deadline):
                break
            for idx in ordered_indices:
                if deadline is not None and time.perf_counter() >= float(deadline):
                    break
                width, height = benchmark.macro_sizes[idx].tolist()
                step = max(width, height) * shift_fraction
                directions = [
                    torch.tensor([step, 0.0], dtype=refined.dtype, device=refined.device),
                    torch.tensor([-step, 0.0], dtype=refined.dtype, device=refined.device),
                    torch.tensor([0.0, step], dtype=refined.dtype, device=refined.device),
                    torch.tensor([0.0, -step], dtype=refined.dtype, device=refined.device),
                ]

                for delta in directions:
                    candidate = refined.clone()
                    candidate[idx] = self.clamp_macro_to_canvas(
                        candidate[idx] + delta,
                        benchmark,
                        idx,
                    )

                    if require_legal and not self.changed_macros_are_legal(
                        candidate,
                        benchmark,
                        [idx],
                    ):
                        continue

                    score = self.operator_score(candidate, benchmark, hard_graph, cfg)
                    if score < best_score:
                        refined = candidate
                        best_score = score

        return self.restore_fixed_macros(refined, benchmark)

    def place_indices_on_grid(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        macro_indices: Sequence[int],
        region: Tuple[float, float, float, float],
        jitter_fraction: float = 0.0,
        rng: Optional[torch.Generator] = None,
    ) -> None:
        """Assign macros to a regular slot grid inside `region`."""
        if not macro_indices:
            return

        x_min, y_min, x_max, y_max = region
        region_width = max(x_max - x_min, 1e-6)
        region_height = max(y_max - y_min, 1e-6)
        aspect = max(region_width / max(region_height, 1e-6), 1e-6)
        num_macros = len(macro_indices)
        num_cols = max(1, int(math.ceil(math.sqrt(num_macros * aspect))))
        num_rows = max(1, int(math.ceil(num_macros / num_cols)))
        slot_width = region_width / num_cols
        slot_height = region_height / num_rows

        for position_in_grid, macro_index in enumerate(macro_indices):
            row = position_in_grid // num_cols
            col = position_in_grid % num_cols
            center_x = x_min + (col + 0.5) * slot_width
            center_y = y_min + (row + 0.5) * slot_height

            if jitter_fraction > 0.0 and rng is not None:
                jitter = torch.rand(2, generator=rng, dtype=placement.dtype)
                jitter = (jitter - 0.5) * torch.tensor(
                    [slot_width, slot_height],
                    dtype=placement.dtype,
                )
                center_x += float(jitter[0]) * jitter_fraction
                center_y += float(jitter[1]) * jitter_fraction

            candidate = torch.tensor(
                [center_x, center_y],
                dtype=placement.dtype,
                device=placement.device,
            )
            candidate = self.adjust_candidate_to_region(
                candidate=candidate,
                benchmark=benchmark,
                macro_index=macro_index,
                region=region,
            )
            placement[macro_index] = self.clamp_macro_to_canvas(
                candidate,
                benchmark,
                macro_index,
            )

    def compute_pin_attraction_targets(
        self,
        benchmark: Benchmark,
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, float]]:
        """
        Compute per-macro attraction targets from connected IO ports.

        If explicit port locations are unavailable for a macro, connected macro
        and soft-cluster positions provide a weaker fallback target.
        """
        target_sums: Dict[int, torch.Tensor] = {}
        weights: Dict[int, float] = {}
        fallback_sums: Dict[int, torch.Tensor] = {}
        fallback_weights: Dict[int, float] = {}
        port_count = int(benchmark.port_positions.shape[0])

        for net_pos, net_nodes in enumerate(benchmark.net_nodes):
            nodes = [int(node) for node in net_nodes.tolist()]
            hard_nodes = [node for node in nodes if 0 <= node < benchmark.num_hard_macros]
            if not hard_nodes:
                continue

            net_weight = (
                float(benchmark.net_weights[net_pos])
                if net_pos < int(benchmark.net_weights.shape[0])
                else 1.0
            )
            port_offsets = [
                node - benchmark.num_macros
                for node in nodes
                if benchmark.num_macros <= node < benchmark.num_macros + port_count
            ]
            if port_offsets:
                port_positions = benchmark.port_positions[port_offsets]
                target = torch.mean(port_positions.to(benchmark.macro_positions.device), dim=0)
                weight = net_weight * float(len(port_offsets)) / max(float(len(hard_nodes)), 1.0)
                for macro_index in hard_nodes:
                    target_sums[macro_index] = target_sums.get(
                        macro_index,
                        torch.zeros(2, dtype=benchmark.macro_positions.dtype),
                    ) + target.cpu() * weight
                    weights[macro_index] = weights.get(macro_index, 0.0) + weight
                continue

            other_macros = [node for node in nodes if 0 <= node < benchmark.num_macros]
            if len(other_macros) <= 1:
                continue

            for macro_index in hard_nodes:
                neighbors = [node for node in other_macros if node != macro_index]
                if not neighbors:
                    continue
                target = torch.mean(benchmark.macro_positions[neighbors], dim=0)
                weight = 0.25 * net_weight / max(float(len(neighbors)), 1.0)
                fallback_sums[macro_index] = fallback_sums.get(
                    macro_index,
                    torch.zeros(2, dtype=benchmark.macro_positions.dtype),
                ) + target.cpu() * weight
                fallback_weights[macro_index] = fallback_weights.get(macro_index, 0.0) + weight

        targets: Dict[int, torch.Tensor] = {}
        normalized_weights: Dict[int, float] = {}
        max_weight = max(weights.values()) if weights else 0.0
        max_fallback_weight = max(fallback_weights.values()) if fallback_weights else 0.0

        for macro_index, total_weight in weights.items():
            targets[macro_index] = target_sums[macro_index] / max(total_weight, 1e-6)
            normalized_weights[macro_index] = total_weight / max(max_weight, 1e-6)

        for macro_index, total_weight in fallback_weights.items():
            if macro_index in targets:
                continue
            targets[macro_index] = fallback_sums[macro_index] / max(total_weight, 1e-6)
            normalized_weights[macro_index] = 0.5 * total_weight / max(
                max_fallback_weight,
                1e-6,
            )

        return targets, normalized_weights

    def choose_boundary_side(
        self,
        benchmark: Benchmark,
        macro_index: int,
        targets: Dict[int, torch.Tensor],
    ) -> str:
        """Pick a boundary side from a macro's pin-attraction target."""
        target = targets.get(macro_index)
        if target is None:
            return ["left", "right", "bottom", "top"][macro_index % 4]

        x_pos = float(target[0])
        y_pos = float(target[1])
        distances = {
            "left": x_pos,
            "right": float(benchmark.canvas_width) - x_pos,
            "bottom": y_pos,
            "top": float(benchmark.canvas_height) - y_pos,
        }
        return min(distances, key=distances.get)

    def side_sort_key(
        self,
        benchmark: Benchmark,
        macro_index: int,
        side: str,
        targets: Dict[int, torch.Tensor],
    ) -> float:
        """Sort macros along one side by their attraction target."""
        target = targets.get(macro_index)
        if target is None:
            return float(macro_index)
        if side in ("left", "right"):
            return float(target[1])
        return float(target[0])

    def place_indices_on_side(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        macro_indices: Sequence[int],
        side: str,
        min_side_gap_fraction: float = 0.0,
    ) -> None:
        """Place macros evenly along one canvas side."""
        if not macro_indices:
            return

        count = len(macro_indices)
        gap = max(0.0, min_side_gap_fraction) * min(
            float(benchmark.canvas_width),
            float(benchmark.canvas_height),
        )

        def spaced_coordinate(pos: int, low: float, high: float) -> float:
            if low > high:
                return 0.5 * (low + high)
            fraction = (pos + 1.0) / (count + 1.0)
            return low + fraction * (high - low)

        for pos, macro_index in enumerate(macro_indices):
            width, height = benchmark.macro_sizes[macro_index].tolist()
            if side == "left":
                center = torch.tensor(
                    [
                        width / 2.0,
                        spaced_coordinate(
                            pos,
                            height / 2.0 + gap,
                            float(benchmark.canvas_height) - height / 2.0 - gap,
                        ),
                    ],
                    dtype=placement.dtype,
                    device=placement.device,
                )
            elif side == "right":
                center = torch.tensor(
                    [
                        float(benchmark.canvas_width) - width / 2.0,
                        spaced_coordinate(
                            pos,
                            height / 2.0 + gap,
                            float(benchmark.canvas_height) - height / 2.0 - gap,
                        ),
                    ],
                    dtype=placement.dtype,
                    device=placement.device,
                )
            elif side == "bottom":
                center = torch.tensor(
                    [
                        spaced_coordinate(
                            pos,
                            width / 2.0 + gap,
                            float(benchmark.canvas_width) - width / 2.0 - gap,
                        ),
                        height / 2.0,
                    ],
                    dtype=placement.dtype,
                    device=placement.device,
                )
            else:
                center = torch.tensor(
                    [
                        spaced_coordinate(
                            pos,
                            width / 2.0 + gap,
                            float(benchmark.canvas_width) - width / 2.0 - gap,
                        ),
                        float(benchmark.canvas_height) - height / 2.0,
                    ],
                    dtype=placement.dtype,
                    device=placement.device,
                )

            placement[macro_index] = self.clamp_macro_to_canvas(center, benchmark, macro_index)

    def estimate_peripheral_margin(self, benchmark: Benchmark) -> float:
        """Estimate a conservative interior margin for peripheral placement."""
        hard_indices = self.get_hard_macro_indices(benchmark)
        if not hard_indices:
            return 0.0
        max_width = max(float(benchmark.macro_sizes[idx, 0]) for idx in hard_indices)
        max_height = max(float(benchmark.macro_sizes[idx, 1]) for idx in hard_indices)
        return 0.60 * max(max_width, max_height)

    def compute_spectral_values(
        self,
        hard_graph: Dict[int, Dict[int, float]],
        macro_indices: Sequence[int],
    ) -> Optional[np.ndarray]:
        """Return the Fiedler vector for the induced hard-macro graph."""
        if len(macro_indices) <= 1:
            return None

        index_by_macro = {macro_index: pos for pos, macro_index in enumerate(macro_indices)}
        matrix = np.zeros((len(macro_indices), len(macro_indices)), dtype=np.float64)
        for left_idx in macro_indices:
            left_pos = index_by_macro[left_idx]
            for right_idx, weight in hard_graph[left_idx].items():
                right_pos = index_by_macro.get(right_idx)
                if right_pos is None or right_pos == left_pos:
                    continue
                matrix[left_pos, right_pos] += float(weight)

        if float(np.sum(matrix)) <= 1e-12:
            return None

        matrix = 0.5 * (matrix + matrix.T)
        degrees = np.sum(matrix, axis=1)
        laplacian = np.diag(degrees) - matrix
        values, vectors = np.linalg.eigh(laplacian)
        order = np.argsort(values)
        if len(order) < 2:
            return None
        return vectors[:, order[1]]

    def graph_edges(
        self,
        hard_graph: Dict[int, Dict[int, float]],
    ) -> List[Tuple[int, int, float]]:
        """Return unique undirected graph edges."""
        edges: List[Tuple[int, int, float]] = []
        for left_idx, neighbors in hard_graph.items():
            for right_idx, weight in neighbors.items():
                if right_idx <= left_idx:
                    continue
                edges.append((left_idx, right_idx, float(weight)))
        return edges

    def add_overlap_repulsion(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        hard_indices: Sequence[int],
        movable_set: Iterable[int],
        forces: torch.Tensor,
        strength: float,
    ) -> None:
        """Accumulate pairwise hard-macro overlap repulsion forces."""
        movable_lookup = set(movable_set)
        for left_pos, left_idx in enumerate(hard_indices):
            left_center = placement[left_idx]
            left_w, left_h = benchmark.macro_sizes[left_idx].tolist()
            for right_idx in hard_indices[left_pos + 1 :]:
                right_center = placement[right_idx]
                right_w, right_h = benchmark.macro_sizes[right_idx].tolist()
                delta = right_center - left_center
                min_sep_x = (left_w + right_w) / 2.0 + self.safety_gap
                min_sep_y = (left_h + right_h) / 2.0 + self.safety_gap
                overlap_x = min_sep_x - abs(float(delta[0]))
                overlap_y = min_sep_y - abs(float(delta[1]))
                if overlap_x <= 0.0 or overlap_y <= 0.0:
                    continue

                if overlap_x < overlap_y:
                    direction = torch.tensor(
                        [1.0 if float(delta[0]) >= 0.0 else -1.0, 0.0],
                        dtype=placement.dtype,
                        device=placement.device,
                    )
                    magnitude = overlap_x
                else:
                    direction = torch.tensor(
                        [0.0, 1.0 if float(delta[1]) >= 0.0 else -1.0],
                        dtype=placement.dtype,
                        device=placement.device,
                    )
                    magnitude = overlap_y

                force = direction * (strength * magnitude)
                if left_idx in movable_lookup:
                    forces[left_idx] = forces[left_idx] - force
                if right_idx in movable_lookup:
                    forces[right_idx] = forces[right_idx] + force

    def apply_forces(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        hard_indices: Sequence[int],
        movable_set: Iterable[int],
        forces: torch.Tensor,
        max_move: float,
    ) -> None:
        """Apply capped forces and clamp moved macros to the canvas."""
        movable_lookup = set(movable_set)
        for idx in hard_indices:
            if idx not in movable_lookup:
                continue
            move = forces[idx]
            norm = torch.norm(move).item()
            if norm <= 1e-9:
                continue
            if norm > max_move:
                move = move * (max_move / norm)
            placement[idx] = self.clamp_macro_to_canvas(placement[idx] + move, benchmark, idx)

    def changed_macros_are_legal(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        changed_indices: Sequence[int],
    ) -> bool:
        """Check legality of a small set of changed hard macros."""
        hard_indices = self.get_hard_macro_indices(benchmark)
        for idx in changed_indices:
            if not self.is_hard_macro_legal(placement, benchmark, idx, hard_indices):
                return False
        return True

    def operator_score(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        hard_graph: Dict[int, Dict[int, float]],
        config: Dict[str, Any],
    ) -> float:
        """Score local refinements with proxy cost when explicitly provided."""
        plc = config.get("plc")
        if plc is not None and bool(config.get("use_proxy", False)):
            from macro_place.objective import compute_proxy_cost

            return float(compute_proxy_cost(placement, benchmark, plc)["proxy_cost"])
        return self.score_candidate(placement, benchmark, hard_graph)

    def refine_placement(
        self, placement: torch.Tensor, benchmark: Benchmark
    ) -> torch.Tensor:
        """
        Hook for future post-initialization refinement stages.

        Good future extensions here include:
        - running a local policy after initialization
        - doing light simulated annealing
        - adjusting soft macros after hard-macro initialization
        - chaining multiple initializers and then refining the winner
        """
        return placement


def parse_initializer_chain(chain: Union[Sequence[str], str]) -> List[str]:
    """Parse comma-separated or list-style chain input."""
    if isinstance(chain, str):
        names: List[str] = []
        for chunk in chain.split(","):
            name = chunk.strip()
            if name:
                names.append(name)
        return names
    return [str(name).strip() for name in chain if str(name).strip()]


def run_initializer_chain(
    problem: Benchmark,
    chain: Union[Sequence[str], str],
    seed: int = 0,
    config: Optional[Dict[str, Any]] = None,
    collect_metrics: bool = True,
    plc: Optional[Any] = None,
    placement: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Run a manually specified initializer/operator chain.

    Args:
        problem: Benchmark to place.
        chain: Comma-separated string or list of operator names.
        seed: Random seed controlling all stochastic operators.
        config: Optional nested dictionary keyed by operator name.
        collect_metrics: If True, collect per-stage metrics.
        plc: Optional PlacementCost object for proxy-cost metrics.
        placement: Optional existing placement. If absent, the first operator
            must be a seed operator.

    Returns:
        `(placement, metadata)` where metadata contains chain, seed, runtime,
        and per-stage metrics.
    """
    cfg = config or {}
    if plc is None:
        plc = cfg.get("plc")
    if placement is None:
        placement = cfg.get("placement")

    names = parse_initializer_chain(chain)
    if not names:
        raise ValueError("Initializer chain is empty")

    operators = [get_initializer_operator(name) for name in names]
    if placement is None and operators[0].kind != "seed":
        raise ValueError(
            f"First operator `{names[0]}` is kind `{operators[0].kind}`. "
            "Start with a seed operator or pass an existing placement."
        )

    helper_config = dict(cfg.get("placer", {}))
    helper = EnsembleInitializerPlacer(seed=seed, **helper_config)
    helper.set_seed()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed)

    current = placement.clone() if placement is not None else None
    metadata: Dict[str, Any] = {
        "chain": names,
        "seed": seed,
        "benchmark": problem.name,
        "num_hard_macros": problem.num_hard_macros,
        "runtime_sec": None,
        "stage_metrics": [],
        "budget_stopped": False,
        "stopped_after_operator": None,
        "final_operator": None,
    }

    start_time = time.perf_counter()
    previous_metrics: Optional[Dict[str, Any]] = None
    chain_budget_sec = cfg.get("chain_budget_sec")
    stage_budget_sec = cfg.get("stage_budget_sec")
    chain_deadline = (
        start_time + float(chain_budget_sec) if chain_budget_sec is not None else None
    )

    for requested_name, operator in zip(names, operators):
        if chain_deadline is not None and time.perf_counter() >= chain_deadline:
            if current is None:
                raise RuntimeError("Initializer chain budget expired before any placement was produced")
            metadata["budget_stopped"] = True
            metadata["stopped_after_operator"] = metadata.get("final_operator")
            break

        operator_config: Dict[str, Any] = {}
        operator_config.update(cfg.get("*", {}))
        operator_config.update(cfg.get(operator.name, {}))
        operator_config.update(cfg.get(requested_name, {}))
        if plc is not None:
            operator_config.setdefault("plc", plc)
        if stage_budget_sec is not None:
            operator_config.setdefault("stage_budget_sec", float(stage_budget_sec))

        stage_start = time.perf_counter()
        if stage_budget_sec is not None or chain_deadline is not None:
            deadlines = []
            if stage_budget_sec is not None:
                deadlines.append(stage_start + float(stage_budget_sec))
            if chain_deadline is not None:
                deadlines.append(chain_deadline)
            operator_config["deadline_sec"] = min(deadlines)

        current = operator.run(
            problem=problem,
            placement=current,
            rng=rng,
            config=operator_config,
            context=helper,
        )
        current = helper.restore_fixed_macros(current, problem)
        stage_runtime = time.perf_counter() - stage_start
        metadata["final_operator"] = requested_name

        if collect_metrics:
            try:
                metrics = collect_initializer_metrics(current, problem, plc=plc)
            except Exception as exc:
                metrics = {
                    "metrics_error": f"metric collection failed: {exc}",
                    "valid": None,
                    "legal": None,
                    "overlap_count": None,
                    "boundary_violations": None,
                }
            metrics.update(
                {
                    "operator": requested_name,
                    "canonical_operator": operator.name,
                    "kind": operator.kind,
                    "runtime_sec": stage_runtime,
                    "macro_count": problem.num_hard_macros,
                    "stage_budget_sec": float(stage_budget_sec)
                    if stage_budget_sec is not None
                    else None,
                    "budget_exceeded": bool(
                        stage_budget_sec is not None and stage_runtime > float(stage_budget_sec)
                    ),
                }
            )
            if (
                operator.name == "legalize"
                and previous_metrics is not None
                and previous_metrics.get("proxy_cost") is not None
                and metrics.get("proxy_cost") is not None
            ):
                metrics["legalization_damage"] = (
                    float(metrics["proxy_cost"]) - float(previous_metrics["proxy_cost"])
                )
            else:
                metrics["legalization_damage"] = None

            metadata["stage_metrics"].append(metrics)
            previous_metrics = metrics

        elapsed = time.perf_counter() - start_time
        if (
            (chain_budget_sec is not None and elapsed > float(chain_budget_sec))
            or (stage_budget_sec is not None and stage_runtime > float(stage_budget_sec))
        ):
            if current is None:
                raise RuntimeError("Initializer chain budget expired before any placement was produced")
            metadata["budget_stopped"] = True
            metadata["stopped_after_operator"] = requested_name
            break

    metadata["runtime_sec"] = time.perf_counter() - start_time
    if current is None:
        raise RuntimeError("Initializer chain produced no placement")
    return current, metadata


def collect_initializer_metrics(
    placement: torch.Tensor,
    benchmark: Benchmark,
    plc: Optional[Any] = None,
) -> Dict[str, Any]:
    """Collect available comparable metrics without making metrics mandatory."""
    metrics: Dict[str, Any] = {
        "macro_count": benchmark.num_hard_macros,
        "proxy_cost": None,
        "hpwl": None,
        "wirelength_cost": None,
        "density_cost": None,
        "congestion_cost": None,
        "overlap": None,
        "overlap_count": None,
        "total_overlap_area": None,
        "boundary_violations": None,
        "max_bin_density": None,
        "max_macro_bin_utilization": None,
        "density_overflow_energy": None,
        "approximate_narrow_channel_count": None,
        "movable_macro_spread": None,
        "macro_crowding_energy": None,
        "valid": None,
        "legal": None,
        "metrics_error": None,
    }

    fixed_ok = True
    if benchmark.macro_fixed.any():
        fixed_ok = bool(
            torch.allclose(
                placement[benchmark.macro_fixed],
                benchmark.macro_positions[benchmark.macro_fixed],
                atol=1e-3,
            )
    )
    shape_ok = placement.shape == (benchmark.num_macros, 2)
    finite_ok = not bool(torch.isnan(placement).any() or torch.isinf(placement).any())

    try:
        metrics["boundary_violations"] = count_boundary_violations(placement, benchmark)
    except Exception as exc:
        metrics["metrics_error"] = f"boundary metrics failed: {exc}"

    try:
        from macro_place.objective import compute_overlap_metrics

        overlap_metrics = compute_overlap_metrics(placement, benchmark)
        metrics.update(overlap_metrics)
        metrics["overlap"] = overlap_metrics.get("total_overlap_area")
    except Exception as exc:  # metrics should not make experiments unusable
        metrics["metrics_error"] = f"overlap metrics failed: {exc}"

    try:
        metrics.update(compute_stage1_readiness_metrics(placement, benchmark))
    except Exception as exc:
        existing_error = metrics.get("metrics_error")
        readiness_error = f"stage1 readiness metrics failed: {exc}"
        metrics["metrics_error"] = (
            f"{existing_error}; {readiness_error}" if existing_error else readiness_error
        )

    if plc is not None:
        try:
            from macro_place.objective import compute_proxy_cost

            cost_metrics = compute_proxy_cost(placement, benchmark, plc)
            metrics.update(cost_metrics)
            metrics["hpwl"] = cost_metrics.get("wirelength_cost")
            metrics["overlap"] = cost_metrics.get("total_overlap_area")
        except Exception as exc:
            existing_error = metrics.get("metrics_error")
            proxy_error = f"proxy metrics failed: {exc}"
            metrics["metrics_error"] = (
                f"{existing_error}; {proxy_error}" if existing_error else proxy_error
            )

    overlap_count = metrics.get("overlap_count")
    no_overlaps = overlap_count == 0 if overlap_count is not None else None
    bounds_ok = metrics["boundary_violations"] == 0
    metrics["legal"] = (
        bool(shape_ok and finite_ok and bounds_ok and fixed_ok and no_overlaps)
        if no_overlaps is not None
        else None
    )
    metrics["valid"] = metrics["legal"]
    return normalize_metric_values(metrics)


def compute_stage1_readiness_metrics(
    placement: torch.Tensor,
    benchmark: Benchmark,
    bin_count: int = 8,
    target_density: float = 0.72,
    max_pairwise_macros: int = 520,
) -> Dict[str, Any]:
    """Compute lightweight physical-readiness metrics without requiring a PLC."""
    hard_count = int(benchmark.num_hard_macros)
    if hard_count <= 0:
        return {
            "max_bin_density": 0.0,
            "max_macro_bin_utilization": 0.0,
            "density_overflow_energy": 0.0,
            "approximate_narrow_channel_count": 0,
            "movable_macro_spread": 0.0,
            "macro_crowding_energy": 0.0,
        }

    bin_count = max(1, min(int(bin_count), 32))
    canvas_width = float(benchmark.canvas_width)
    canvas_height = float(benchmark.canvas_height)
    bin_width = canvas_width / bin_count
    bin_height = canvas_height / bin_count
    bin_area = max(bin_width * bin_height, 1e-6)
    utilizations = np.zeros((bin_count, bin_count), dtype=np.float64)

    hard_positions = placement[:hard_count].detach().cpu()
    hard_sizes = benchmark.macro_sizes[:hard_count].detach().cpu()
    for idx in range(hard_count):
        x_pos = min(max(float(hard_positions[idx, 0]), 0.0), canvas_width - 1e-9)
        y_pos = min(max(float(hard_positions[idx, 1]), 0.0), canvas_height - 1e-9)
        col = min(bin_count - 1, max(0, int(x_pos / max(bin_width, 1e-6))))
        row = min(bin_count - 1, max(0, int(y_pos / max(bin_height, 1e-6))))
        area = float(hard_sizes[idx, 0] * hard_sizes[idx, 1])
        utilizations[row, col] += area / bin_area

    overflow = np.maximum(utilizations - target_density, 0.0)
    movable_mask = (benchmark.get_movable_mask() & benchmark.get_hard_macro_mask())[:hard_count]
    movable_positions = hard_positions[movable_mask.detach().cpu()]
    canvas_scale = max(canvas_width, canvas_height, 1.0)
    if int(movable_positions.shape[0]) > 1:
        centroid = torch.mean(movable_positions, dim=0)
        spread = torch.mean(torch.norm(movable_positions - centroid, dim=1)).item() / canvas_scale
    else:
        spread = 0.0

    narrow_channel_count: Optional[int] = None
    crowding_energy: Optional[float] = None
    if hard_count <= max_pairwise_macros:
        narrow_channel_count = 0
        crowding_energy = 0.0
        threshold = 0.030 * min(canvas_width, canvas_height)
        for left_idx in range(hard_count):
            left_x = float(hard_positions[left_idx, 0])
            left_y = float(hard_positions[left_idx, 1])
            left_w = float(hard_sizes[left_idx, 0])
            left_h = float(hard_sizes[left_idx, 1])
            for right_idx in range(left_idx + 1, hard_count):
                right_x = float(hard_positions[right_idx, 0])
                right_y = float(hard_positions[right_idx, 1])
                right_w = float(hard_sizes[right_idx, 0])
                right_h = float(hard_sizes[right_idx, 1])
                gap_x = abs(right_x - left_x) - (left_w + right_w) / 2.0
                gap_y = abs(right_y - left_y) - (left_h + right_h) / 2.0
                if 0.0 <= gap_x < threshold and gap_y < 0.0:
                    narrow_channel_count += 1
                if 0.0 <= gap_y < threshold and gap_x < 0.0:
                    narrow_channel_count += 1
                positive_gap = max(0.0, min(max(gap_x, 0.0), max(gap_y, 0.0)))
                if positive_gap < threshold:
                    crowding_energy += ((threshold - positive_gap) / max(threshold, 1e-6)) ** 2

    max_density = float(np.max(utilizations)) if utilizations.size else 0.0
    return {
        "max_bin_density": max_density,
        "max_macro_bin_utilization": max_density,
        "density_overflow_energy": float(np.sum(overflow * overflow)),
        "approximate_narrow_channel_count": narrow_channel_count,
        "movable_macro_spread": float(spread),
        "macro_crowding_energy": crowding_energy,
    }


def normalize_metric_values(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Convert tensor/NumPy scalar metric values into plain Python values."""
    normalized: Dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            normalized[key] = value.item() if value.numel() == 1 else value.detach().cpu().tolist()
        elif isinstance(value, np.generic):
            normalized[key] = value.item()
        else:
            normalized[key] = value
    return normalized


def count_boundary_violations(placement: torch.Tensor, benchmark: Benchmark) -> int:
    """Count macros whose boxes extend outside the canvas."""
    widths = benchmark.macro_sizes[:, 0]
    heights = benchmark.macro_sizes[:, 1]
    x_min = placement[:, 0] - widths / 2.0
    x_max = placement[:, 0] + widths / 2.0
    y_min = placement[:, 1] - heights / 2.0
    y_max = placement[:, 1] + heights / 2.0
    violations = (x_min < 0) | (x_max > benchmark.canvas_width) | (y_min < 0) | (
        y_max > benchmark.canvas_height
    )
    return int(torch.sum(violations).item())


def _seed_hierarchical(
    helper: EnsembleInitializerPlacer,
    benchmark: Benchmark,
    placement: Optional[torch.Tensor],
    rng: torch.Generator,
    config: Dict[str, Any],
) -> torch.Tensor:
    return helper.run_hierarchical_seed(benchmark, rng=rng, config=config)


def _seed_anchor(
    helper: EnsembleInitializerPlacer,
    benchmark: Benchmark,
    placement: Optional[torch.Tensor],
    rng: torch.Generator,
    config: Dict[str, Any],
) -> torch.Tensor:
    return helper.run_benchmark_anchor_initializer(benchmark)


def _seed_random_spread(
    helper: EnsembleInitializerPlacer,
    benchmark: Benchmark,
    placement: Optional[torch.Tensor],
    rng: torch.Generator,
    config: Dict[str, Any],
) -> torch.Tensor:
    return helper.run_random_spread_seed(benchmark, rng=rng, config=config)


def _seed_grid(
    helper: EnsembleInitializerPlacer,
    benchmark: Benchmark,
    placement: Optional[torch.Tensor],
    rng: torch.Generator,
    config: Dict[str, Any],
) -> torch.Tensor:
    return helper.run_grid_seed(benchmark, rng=rng, config=config)


def _seed_pin_aware(
    helper: EnsembleInitializerPlacer,
    benchmark: Benchmark,
    placement: Optional[torch.Tensor],
    rng: torch.Generator,
    config: Dict[str, Any],
) -> torch.Tensor:
    return helper.run_pin_aware_seed(benchmark, rng=rng, config=config)


def _seed_peripheral(
    helper: EnsembleInitializerPlacer,
    benchmark: Benchmark,
    placement: Optional[torch.Tensor],
    rng: torch.Generator,
    config: Dict[str, Any],
) -> torch.Tensor:
    return helper.run_peripheral_seed(benchmark, rng=rng, config=config)


def _transform_pin_aware_shift(
    helper: EnsembleInitializerPlacer,
    benchmark: Benchmark,
    placement: Optional[torch.Tensor],
    rng: torch.Generator,
    config: Dict[str, Any],
) -> torch.Tensor:
    if placement is None:
        raise ValueError("pin_aware_shift requires an existing placement")
    return helper.apply_pin_aware_shift(
        placement=placement,
        benchmark=benchmark,
        strength=float(config.get("strength", 0.35)),
    )


def _transform_spectral_order(
    helper: EnsembleInitializerPlacer,
    benchmark: Benchmark,
    placement: Optional[torch.Tensor],
    rng: torch.Generator,
    config: Dict[str, Any],
) -> torch.Tensor:
    if placement is None:
        raise ValueError("spectral_order requires an existing placement")
    return helper.run_spectral_order(placement, benchmark, rng=rng, config=config)


def _transform_periphery_bias(
    helper: EnsembleInitializerPlacer,
    benchmark: Benchmark,
    placement: Optional[torch.Tensor],
    rng: torch.Generator,
    config: Dict[str, Any],
) -> torch.Tensor:
    if placement is None:
        raise ValueError("periphery_bias requires an existing placement")
    return helper.run_periphery_bias(placement, benchmark, rng=rng, config=config)


def _transform_force_smooth(
    helper: EnsembleInitializerPlacer,
    benchmark: Benchmark,
    placement: Optional[torch.Tensor],
    rng: torch.Generator,
    config: Dict[str, Any],
) -> torch.Tensor:
    if placement is None:
        raise ValueError("force_smooth requires an existing placement")
    return helper.run_force_smooth(placement, benchmark, rng=rng, config=config)


def _transform_analytical_stage1(
    helper: EnsembleInitializerPlacer,
    benchmark: Benchmark,
    placement: Optional[torch.Tensor],
    rng: torch.Generator,
    config: Dict[str, Any],
) -> torch.Tensor:
    if placement is None:
        raise ValueError("analytical_stage1 requires an existing placement")
    return helper.run_analytical_stage1(placement, benchmark, rng=rng, config=config)


def _transform_macro_spread(
    helper: EnsembleInitializerPlacer,
    benchmark: Benchmark,
    placement: Optional[torch.Tensor],
    rng: torch.Generator,
    config: Dict[str, Any],
) -> torch.Tensor:
    if placement is None:
        raise ValueError("macro_spread requires an existing placement")
    return helper.run_macro_spread(placement, benchmark, rng=rng, config=config)


def _repair_legalize(
    helper: EnsembleInitializerPlacer,
    benchmark: Benchmark,
    placement: Optional[torch.Tensor],
    rng: torch.Generator,
    config: Dict[str, Any],
) -> torch.Tensor:
    if placement is None:
        raise ValueError("legalize requires an existing placement")
    return helper.run_legalize_operator(placement, benchmark, rng=rng, config=config)


def _refine_local_swap(
    helper: EnsembleInitializerPlacer,
    benchmark: Benchmark,
    placement: Optional[torch.Tensor],
    rng: torch.Generator,
    config: Dict[str, Any],
) -> torch.Tensor:
    if placement is None:
        raise ValueError("local_swap requires an existing placement")
    return helper.run_local_swap(placement, benchmark, rng=rng, config=config)


def _refine_local_shift(
    helper: EnsembleInitializerPlacer,
    benchmark: Benchmark,
    placement: Optional[torch.Tensor],
    rng: torch.Generator,
    config: Dict[str, Any],
) -> torch.Tensor:
    if placement is None:
        raise ValueError("local_shift requires an existing placement")
    return helper.run_local_shift(placement, benchmark, rng=rng, config=config)


register_initializer_operator(
    InitializerOperator(
        name="hierarchical",
        kind="seed",
        runner=_seed_hierarchical,
        aliases=["hierarchical_partition", "graph_hierarchy"],
    )
)
register_initializer_operator(
    InitializerOperator(name="anchor", kind="seed", runner=_seed_anchor, aliases=["benchmark_anchor"])
)
register_initializer_operator(
    InitializerOperator(name="random_spread", kind="seed", runner=_seed_random_spread)
)
register_initializer_operator(InitializerOperator(name="grid", kind="seed", runner=_seed_grid))
register_initializer_operator(
    InitializerOperator(name="pin_aware", kind="seed", runner=_seed_pin_aware)
)
register_initializer_operator(
    InitializerOperator(name="peripheral", kind="seed", runner=_seed_peripheral)
)
register_initializer_operator(
    InitializerOperator(
        name="pin_aware_shift",
        kind="transform",
        runner=_transform_pin_aware_shift,
    )
)
register_initializer_operator(
    InitializerOperator(
        name="spectral_order",
        kind="transform",
        runner=_transform_spectral_order,
    )
)
register_initializer_operator(
    InitializerOperator(
        name="periphery_bias",
        kind="transform",
        runner=_transform_periphery_bias,
    )
)
register_initializer_operator(
    InitializerOperator(
        name="force_smooth",
        kind="transform",
        runner=_transform_force_smooth,
        aliases=["analytical_smooth"],
    )
)
register_initializer_operator(
    InitializerOperator(
        name="analytical_stage1",
        kind="transform",
        runner=_transform_analytical_stage1,
        aliases=["analytical_physical", "physical_smooth"],
    )
)
register_initializer_operator(
    InitializerOperator(name="macro_spread", kind="transform", runner=_transform_macro_spread)
)
register_initializer_operator(
    InitializerOperator(name="legalize", kind="repair", runner=_repair_legalize)
)
register_initializer_operator(
    InitializerOperator(name="local_swap", kind="refine", runner=_refine_local_swap)
)
register_initializer_operator(
    InitializerOperator(name="local_shift", kind="refine", runner=_refine_local_shift)
)


if __name__ == "__main__":
    print(
        "Run this file through the repo evaluator, for example:\n"
        "  uv run evaluate submissions/models/initializer.py -b ibm01"
    )
