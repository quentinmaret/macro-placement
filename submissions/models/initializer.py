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
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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


if __name__ == "__main__":
    print(
        "Run this file through the repo evaluator, for example:\n"
        "  uv run evaluate submissions/models/initializer.py -b ibm01"
    )
