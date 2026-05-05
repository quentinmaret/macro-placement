"""
RL-style macro placement template built on top of the core scaffold.

This file is meant to answer a practical question for the team:
"What could a reinforcement-learning-based placer look like in this repo,
without pulling in a full training stack yet?"

The design is intentionally simple and educational:

1. Start from the benchmark's initial placement.
2. Run the same conservative legalization step used by `core.py`.
3. Visit movable hard macros one by one, which mirrors the sequential action
   structure used in the RL placement literature.
4. For each macro, build a small local action set of nearby legal moves.
5. Score those candidate actions with a lightweight policy over hand-crafted
   state/action features.
6. Choose one action and update the placement.

Why this shape?

- The Nature 2021 paper and the open-source AlphaChip code frame macro
  placement as a sequential decision problem with a policy over macro actions.
- Later assessments and meta-analyses emphasize the importance of strong
  baselines, careful evaluation, and avoiding brittle design choices.
- This template therefore keeps the trustworthy parts of our scaffold
  (initial-placement anchoring and legality-first behavior) while exposing the
  core RL ideas: state, action candidates, policy scores, rewards, and
  trajectories.

This is not a trained PPO agent. It is a clean bridge between `core.py` and a
future learned method.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch

from macro_place.benchmark import Benchmark


def _load_core_placer_class():
    """
    Load `CorePlacer` from the sibling `core.py` file.

    The competition evaluator imports submission files directly by path instead
    of as part of an installed Python package. Because of that, regular imports
    like `from submissions.models.core import CorePlacer` fail unless the
    `submissions` directory is packaged.

    Loading the sibling file explicitly keeps the "copy and run this file"
    workflow simple for the team.
    """
    core_path = Path(__file__).with_name("core.py")
    spec = importlib.util.spec_from_file_location("models_core", str(core_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load CorePlacer from {core_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CorePlacer


CorePlacer = _load_core_placer_class()


class ActionRecord:
    """
    One decision made by the RL-style local policy.

    Keeping decisions in an explicit record makes it easy to evolve this file
    into a true learning setup later, because a real RL trainer would also keep
    track of states, actions, rewards, and action probabilities.
    """

    def __init__(
        self,
        macro_index: int,
        chosen_action: int,
        action_probabilities: torch.Tensor,
        feature_matrix: torch.Tensor,
        reward: float,
    ) -> None:
        """
        Store one macro-level decision from the policy episode.

        A plain Python class is used here instead of `@dataclass` because the
        repo's evaluator loads submission files with a minimal import path that
        can confuse `dataclass` module lookups during dynamic execution.
        """
        self.macro_index = macro_index
        self.chosen_action = chosen_action
        self.action_probabilities = action_probabilities
        self.feature_matrix = feature_matrix
        self.reward = reward


class RLLocalPolicyPlacer(CorePlacer):
    """
    RL-inspired local policy placer.

    The overall strategy is:

    1. Use `core.py`'s legalization pass so we begin from a legal baseline.
    2. Build a simple hard-macro connectivity graph from the benchmark netlist.
    3. Place macros in an order that prioritizes highly connected hard macros.
    4. For each macro, evaluate a discrete set of local legal moves.
    5. Score each move with a lightweight policy over interpretable features.

    This mirrors the "sequential policy over macro actions" idea from the RL
    literature, but keeps the implementation small enough for teammates to read
    and modify quickly.
    """

    def __init__(
        self,
        seed: int = 42,
        search_radii: int = 150,
        step_scale: float = 0.25,
        safety_gap: float = 0.05,
        policy_temperature: float = 0.20,
        local_step_scale: float = 0.75,
        max_candidate_rings: int = 2,
        sample_actions: bool = False,
    ) -> None:
        """
        Configure the RL-style local policy.

        Args:
            seed: Random seed inherited from the core scaffold.
            search_radii: Passed through to the legalization stage.
            step_scale: Passed through to the legalization stage.
            safety_gap: Extra hard-macro spacing margin.
            policy_temperature: Softmax temperature used when turning policy
                scores into action probabilities.
            local_step_scale: Distance used for local candidate moves, measured
                as a fraction of the macro's larger dimension.
            max_candidate_rings: Number of rings of local actions generated
                around the current macro position.
            sample_actions: If True, sample from the policy distribution. If
                False, take the highest-probability action for deterministic
                debugging and evaluation.
        """
        super().__init__(
            seed=seed,
            search_radii=search_radii,
            step_scale=step_scale,
            safety_gap=safety_gap,
        )
        self.policy_temperature = policy_temperature
        self.local_step_scale = local_step_scale
        self.max_candidate_rings = max_candidate_rings
        self.sample_actions = sample_actions

        # These weights define a tiny hand-built policy. In a future learned
        # agent, these values would come from a neural network instead.
        self.policy_weights = torch.tensor(
            [
                0.00,   # bias
                3.50,   # improve connectivity to neighboring hard macros
                -1.25,  # avoid large displacement from the initial placement
                0.80,   # prefer actions with more spacing to nearby macros
                0.30,   # modest preference for staying away from boundaries
            ],
            dtype=torch.float32,
        )

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        """
        Produce a placement using an RL-style sequential local policy.

        The method reuses the stable parts of the core scaffold and swaps in a
        policy-driven refinement stage. This makes it easier to understand how
        an RL method fits the competition interface without rewriting the full
        pipeline from scratch.
        """
        self.set_seed()

        placement = self.clone_initial_placement(benchmark)
        placement = self.legalize_initial_placement(placement, benchmark)
        placement, _trajectory = self.run_policy_episode(placement, benchmark)
        placement = self.restore_fixed_macros(placement, benchmark)
        return placement

    def run_policy_episode(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
    ) -> Tuple[torch.Tensor, List[ActionRecord]]:
        """
        Run one sequential decision-making episode over the current placement.

        In RL terms:
        - state: current placement plus macro connectivity context
        - action: one legal local move for one macro
        - reward: local surrogate improvement after taking that action

        We keep the current implementation to a single episode because the repo
        evaluator expects only a final placement, not a long-running training
        job. The code structure still leaves room for future online fine-tuning
        or offline imitation/RL training.
        """
        updated = placement.clone()
        hard_graph = self.build_hard_macro_graph(benchmark)
        action_order = self.compute_macro_action_order(benchmark, hard_graph)

        trajectory: List[ActionRecord] = []
        hard_indices = self.get_hard_macro_indices(benchmark)

        for macro_index in action_order:
            candidates = self.generate_local_action_candidates(
                placement=updated,
                benchmark=benchmark,
                macro_index=macro_index,
                hard_indices=hard_indices,
            )
            if len(candidates) == 1:
                continue

            features = self.build_action_feature_matrix(
                placement=updated,
                benchmark=benchmark,
                macro_index=macro_index,
                candidates=candidates,
                hard_graph=hard_graph,
            )
            probabilities = self.compute_action_probabilities(features)
            chosen_action = self.select_action(probabilities)

            before_step = updated.clone()
            previous_position = updated[macro_index].clone()
            updated[macro_index] = candidates[chosen_action]
            reward = self.compute_local_reward(
                placement_before=before_step,
                placement_after=updated,
                benchmark=benchmark,
                macro_index=macro_index,
                previous_position=previous_position,
                hard_graph=hard_graph,
            )

            trajectory.append(
                ActionRecord(
                    macro_index=macro_index,
                    chosen_action=chosen_action,
                    action_probabilities=probabilities,
                    feature_matrix=features,
                    reward=reward,
                )
            )

        return updated, trajectory

    def build_hard_macro_graph(self, benchmark: Benchmark) -> Dict[int, Dict[int, float]]:
        """
        Build a simple hard-macro connectivity graph from benchmark nets.

        The cited RL work uses graph structure as a core part of the state.
        This helper keeps the same spirit in a lightweight form by aggregating
        hard-to-hard relationships from the public benchmark netlist.
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
            for left_index, left_node in enumerate(unique_nodes):
                for right_node in unique_nodes[left_index + 1 :]:
                    graph[left_node][right_node] = graph[left_node].get(right_node, 0.0) + weight
                    graph[right_node][left_node] = graph[right_node].get(left_node, 0.0) + weight

        return graph

    def compute_macro_action_order(
        self,
        benchmark: Benchmark,
        hard_graph: Dict[int, Dict[int, float]],
    ) -> List[int]:
        """
        Decide which movable hard macros the policy should visit first.

        A real RL policy may learn the placement order jointly with other
        decisions. For a simple readable template, we use a deterministic order:
        highly connected hard macros go first, with larger macros breaking ties.
        """
        movable = self.get_movable_hard_macro_indices(benchmark)
        return sorted(
            movable,
            key=lambda idx: (
                -sum(hard_graph[idx].values()),
                -self.macro_area(benchmark, idx),
            ),
        )

    def generate_local_action_candidates(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        macro_index: int,
        hard_indices: Sequence[int],
    ) -> List[torch.Tensor]:
        """
        Generate a discrete action set of nearby legal moves for one macro.

        The main lesson here is practical: even if the eventual model is RL,
        it is much easier to reason about and debug if actions are local and
        legality-filtered instead of unconstrained moves across the full canvas.
        """
        current = placement[macro_index].clone()
        width, height = benchmark.macro_sizes[macro_index].tolist()
        step = max(width, height) * self.local_step_scale

        candidates: List[torch.Tensor] = [current.clone()]
        seen = {
            (round(float(current[0].item()), 6), round(float(current[1].item()), 6))
        }

        for ring in range(1, self.max_candidate_rings + 1):
            for dx_step, dy_step in self.iter_ring_offsets(ring):
                candidate = current.clone()
                candidate[0] = candidate[0] + dx_step * step
                candidate[1] = candidate[1] + dy_step * step
                candidate = self.clamp_macro_to_canvas(candidate, benchmark, macro_index)

                key = (
                    round(float(candidate[0].item()), 6),
                    round(float(candidate[1].item()), 6),
                )
                if key in seen:
                    continue
                seen.add(key)

                placement[macro_index] = candidate
                if self.is_hard_macro_legal(
                    placement=placement,
                    benchmark=benchmark,
                    macro_index=macro_index,
                    other_hard_indices=[idx for idx in hard_indices if idx != macro_index],
                ):
                    candidates.append(candidate.clone())

        placement[macro_index] = current
        return candidates

    def iter_ring_offsets(self, ring: int) -> Iterable[Tuple[int, int]]:
        """
        Yield integer offsets on the perimeter of one square action ring.

        This keeps the candidate set discrete, structured, and easy to visualize.
        A future learned method can replace this action generator with something
        richer without touching the rest of the file.
        """
        for dx_step in range(-ring, ring + 1):
            for dy_step in range(-ring, ring + 1):
                if abs(dx_step) != ring and abs(dy_step) != ring:
                    continue
                yield dx_step, dy_step

    def build_action_feature_matrix(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        macro_index: int,
        candidates: Sequence[torch.Tensor],
        hard_graph: Dict[int, Dict[int, float]],
    ) -> torch.Tensor:
        """
        Turn candidate moves into policy features.

        The features are intentionally small and interpretable:
        - bias term
        - connectivity improvement to neighboring hard macros
        - displacement from the initial benchmark position
        - local spacing to nearby hard macros
        - canvas-margin comfort

        This is the simplest place to later swap in a neural encoder.
        """
        rows: List[torch.Tensor] = []
        current_position = placement[macro_index].clone()
        initial_position = benchmark.macro_positions[macro_index].clone()

        for candidate in candidates:
            connectivity_improvement = self.compute_connectivity_improvement(
                benchmark=benchmark,
                placement=placement,
                macro_index=macro_index,
                candidate=candidate,
                hard_graph=hard_graph,
            )
            anchor_penalty = self.compute_anchor_penalty(
                benchmark=benchmark,
                macro_index=macro_index,
                candidate=candidate,
                initial_position=initial_position,
            )
            spacing_score = self.compute_spacing_score(
                benchmark=benchmark,
                placement=placement,
                macro_index=macro_index,
                candidate=candidate,
            )
            canvas_margin_score = self.compute_canvas_margin_score(
                benchmark=benchmark,
                macro_index=macro_index,
                candidate=candidate,
            )

            rows.append(
                torch.tensor(
                    [
                        1.0,
                        connectivity_improvement,
                        anchor_penalty,
                        spacing_score,
                        canvas_margin_score,
                    ],
                    dtype=torch.float32,
                )
            )

        placement[macro_index] = current_position
        return torch.stack(rows, dim=0)

    def compute_connectivity_improvement(
        self,
        benchmark: Benchmark,
        placement: torch.Tensor,
        macro_index: int,
        candidate: torch.Tensor,
        hard_graph: Dict[int, Dict[int, float]],
    ) -> float:
        """
        Measure how much a candidate helps the macro's local connectivity.

        We use a simple weighted Manhattan-distance surrogate over connected hard
        macros. Positive values mean the candidate pulls the macro closer to its
        current neighbors, which is often directionally aligned with reducing
        wirelength.
        """
        current_position = placement[macro_index].clone()
        baseline_cost = self.connected_distance_cost(
            placement=placement,
            macro_index=macro_index,
            position=current_position,
            hard_graph=hard_graph,
        )
        candidate_cost = self.connected_distance_cost(
            placement=placement,
            macro_index=macro_index,
            position=candidate,
            hard_graph=hard_graph,
        )

        canvas_scale = max(float(benchmark.canvas_width), float(benchmark.canvas_height), 1.0)
        return float((baseline_cost - candidate_cost) / canvas_scale)

    def connected_distance_cost(
        self,
        placement: torch.Tensor,
        macro_index: int,
        position: torch.Tensor,
        hard_graph: Dict[int, Dict[int, float]],
    ) -> float:
        """
        Compute a local hard-to-hard distance surrogate for one macro.

        This is not the full proxy objective. It is only a lightweight local
        score that gives the RL-style policy a directional signal.
        """
        total = 0.0
        for neighbor_index, edge_weight in hard_graph[macro_index].items():
            neighbor_position = placement[neighbor_index]
            distance = abs(float(position[0] - neighbor_position[0])) + abs(
                float(position[1] - neighbor_position[1])
            )
            total += edge_weight * distance
        return total

    def compute_anchor_penalty(
        self,
        benchmark: Benchmark,
        macro_index: int,
        candidate: torch.Tensor,
        initial_position: torch.Tensor,
    ) -> float:
        """
        Penalize large moves away from the benchmark's initial placement.

        The later critique papers argue for careful baselines and fair
        comparisons. Staying anchored early on is a strong default because it
        preserves useful structure while we learn how the system behaves.
        """
        delta = candidate - initial_position
        distance = torch.sqrt(torch.sum(delta * delta)).item()
        canvas_scale = max(float(benchmark.canvas_width), float(benchmark.canvas_height), 1.0)
        return float(distance / canvas_scale)

    def compute_spacing_score(
        self,
        benchmark: Benchmark,
        placement: torch.Tensor,
        macro_index: int,
        candidate: torch.Tensor,
    ) -> float:
        """
        Reward candidates that keep healthy spacing from other hard macros.

        Since legality is already enforced, this feature is a soft preference
        for moves that do not crowd neighbors too tightly.
        """
        best_gap = float("inf")
        for other_index in self.get_hard_macro_indices(benchmark):
            if other_index == macro_index:
                continue

            other_position = placement[other_index]
            width_i, height_i = benchmark.macro_sizes[macro_index].tolist()
            width_j, height_j = benchmark.macro_sizes[other_index].tolist()

            gap_x = abs(float(candidate[0] - other_position[0])) - (width_i + width_j) / 2.0
            gap_y = abs(float(candidate[1] - other_position[1])) - (height_i + height_j) / 2.0
            best_gap = min(best_gap, max(min(gap_x, gap_y), 0.0))

        if best_gap == float("inf"):
            return 0.0

        canvas_scale = max(float(benchmark.canvas_width), float(benchmark.canvas_height), 1.0)
        return float(best_gap / canvas_scale)

    def compute_canvas_margin_score(
        self,
        benchmark: Benchmark,
        macro_index: int,
        candidate: torch.Tensor,
    ) -> float:
        """
        Measure how comfortably a candidate sits inside the canvas.

        This weak feature is useful because placements pressed hard against
        boundaries can reduce downstream flexibility.
        """
        width = benchmark.macro_sizes[macro_index, 0].item()
        height = benchmark.macro_sizes[macro_index, 1].item()

        left_margin = float(candidate[0] - width / 2.0)
        right_margin = float(benchmark.canvas_width - (candidate[0] + width / 2.0))
        bottom_margin = float(candidate[1] - height / 2.0)
        top_margin = float(benchmark.canvas_height - (candidate[1] + height / 2.0))

        min_margin = min(left_margin, right_margin, bottom_margin, top_margin)
        canvas_scale = max(float(benchmark.canvas_width), float(benchmark.canvas_height), 1.0)
        return float(max(min_margin, 0.0) / canvas_scale)

    def compute_action_probabilities(self, feature_matrix: torch.Tensor) -> torch.Tensor:
        """
        Convert action features into a policy distribution.

        A learned RL system would usually replace this with a neural policy
        network. Here we keep it linear on purpose so the action logic remains
        easy to inspect.
        """
        logits = feature_matrix @ self.policy_weights
        scaled_logits = logits / max(self.policy_temperature, 1.0e-6)
        return torch.softmax(scaled_logits, dim=0)

    def select_action(self, probabilities: torch.Tensor) -> int:
        """
        Choose one action from the policy distribution.

        Deterministic argmax is better for debugging and regression checks.
        Sampling is still supported because stochastic exploration is a core RL
        idea and can be useful once the team starts experimenting more deeply.
        """
        if self.sample_actions:
            return int(torch.multinomial(probabilities, num_samples=1).item())
        return int(torch.argmax(probabilities).item())

    def compute_local_reward(
        self,
        placement_before: torch.Tensor,
        placement_after: torch.Tensor,
        benchmark: Benchmark,
        macro_index: int,
        previous_position: torch.Tensor,
        hard_graph: Dict[int, Dict[int, float]],
    ) -> float:
        """
        Compute a simple immediate reward for the chosen local action.

        The reward favors two things:
        - better connectivity to neighboring hard macros
        - staying relatively close to the previous and initial positions

        A stronger future version could replace this with proxy-cost deltas or a
        learned value model.
        """
        new_position = placement_after[macro_index]
        initial_position = benchmark.macro_positions[macro_index]

        connectivity_gain = self.connected_distance_cost(
            placement_before,
            macro_index,
            previous_position,
            hard_graph,
        ) - self.connected_distance_cost(
            placement_after,
            macro_index,
            new_position,
            hard_graph,
        )

        move_distance = torch.sqrt(torch.sum((new_position - previous_position) ** 2)).item()
        anchor_distance = torch.sqrt(torch.sum((new_position - initial_position) ** 2)).item()

        canvas_scale = max(float(benchmark.canvas_width), float(benchmark.canvas_height), 1.0)
        reward = (
            connectivity_gain / canvas_scale
            - 0.25 * move_distance / canvas_scale
            - 0.50 * anchor_distance / canvas_scale
        )
        return float(reward)

    def explain_future_training_recipe(self) -> str:
        """
        Describe how this file could evolve into a true RL training setup.

        This helper is not used by the evaluator. It exists purely as in-code
        guidance for teammates who want to build the next iteration.
        """
        return (
            "Future RL upgrades could keep the same action and feature helpers, "
            "replace `policy_weights` with a neural network, collect trajectories "
            "across many benchmarks, and optimize the policy with PPO or "
            "REINFORCE-style updates against a stronger reward signal."
        )


if __name__ == "__main__":
    print(
        "Run this RL-style template through the repo evaluator, for example:\n"
        "  uv run evaluate submissions/models/rl_local_policy.py -b ibm01"
    )
