"""
Final deterministic tournament placer.

The challenge evaluator gives a submission only a Benchmark, so this placer
loads the matching PlacementCost object itself when available. It then builds a
small portfolio of legal candidates from the strongest checked-in baseline and
the initializer ensemble, scores each candidate with the real proxy objective,
and returns the best zero-overlap placement.
"""

from __future__ import annotations

import importlib.util
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/macro-placement-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/macro-placement-cache")

import numpy as np
import torch

from macro_place.benchmark import Benchmark
from macro_place.objective import compute_proxy_cost


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_symbol(module_path: Path, module_name: str, symbol: str):
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {symbol} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, symbol)


WillSeedPlacer = _load_symbol(
    _repo_root() / "submissions" / "will_seed" / "placer.py",
    "final_will_seed",
    "WillSeedPlacer",
)
EnsembleInitializerPlacer = _load_symbol(
    _repo_root() / "submissions" / "models" / "initializer.py",
    "final_initializer",
    "EnsembleInitializerPlacer",
)
run_initializer_chain = _load_symbol(
    _repo_root() / "submissions" / "models" / "initializer.py",
    "final_initializer_chain",
    "run_initializer_chain",
)


def _load_plc(name: str):
    from macro_place.loader import load_benchmark, load_benchmark_from_dir

    root = _repo_root() / "external" / "MacroPlacement" / "Testcases" / "ICCAD04" / name
    if root.exists():
        _, plc = load_benchmark_from_dir(str(root))
        return plc

    ng45 = {
        "ariane133_ng45": "ariane133",
        "ariane136_ng45": "ariane136",
        "nvdla_ng45": "nvdla",
        "mempool_tile_ng45": "mempool_tile",
        "ariane133": "ariane133",
        "ariane136": "ariane136",
        "nvdla": "nvdla",
        "mempool_tile": "mempool_tile",
    }
    design = ng45.get(name)
    if design:
        base = (
            _repo_root()
            / "external"
            / "MacroPlacement"
            / "Flows"
            / "NanGate45"
            / design
            / "netlist"
            / "output_CT_Grouping"
        )
        netlist = base / "netlist.pb.txt"
        plc_file = base / "initial.plc"
        if netlist.exists() and plc_file.exists():
            _, plc = load_benchmark(str(netlist), str(plc_file), name=name)
            return plc
    return None


class FinalMacroPlacer:
    def __init__(self) -> None:
        self.seed = 314159
        self.will_seeds = (42, 7, 13, 23, 101, 271)
        self.will_refine_iters = 3600
        self.safety_gap = 0.0

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        self._set_seed(self.seed)
        if benchmark.num_hard_macros > 320:
            return WillSeedPlacer(seed=42, refine_iters=3000).place(benchmark)

        plc = _load_plc(benchmark.name)
        candidates: List[Tuple[str, torch.Tensor]] = []

        candidates.append(("will_seed_baseline", WillSeedPlacer(seed=42, refine_iters=3000).place(benchmark)))

        for seed in self.will_seeds:
            placer = WillSeedPlacer(seed=seed, refine_iters=self.will_refine_iters)
            candidates.append((f"will_seed_{seed}", placer.place(benchmark)))

        if benchmark.num_hard_macros <= 320:
            try:
                initializer = EnsembleInitializerPlacer(seed=self.seed)
                candidates.append(("initializer_ensemble", initializer.place(benchmark)))
            except Exception:
                pass

        for chain in self._chain_portfolio(benchmark):
            try:
                placement, _ = run_initializer_chain(
                    benchmark,
                    chain,
                    seed=self.seed,
                    config=self._chain_config(),
                    collect_metrics=False,
                    plc=None,
                )
                candidates.append((f"chain:{chain}", placement))
            except Exception:
                continue

        best_score = float("inf")
        best_placement: Optional[torch.Tensor] = None
        for name, placement in candidates:
            candidate = self._repair_candidate(placement, benchmark)
            if not self._is_valid(candidate, benchmark):
                continue
            score = self._score(candidate, benchmark, plc)
            if score < best_score:
                best_score = score
                best_placement = candidate

        if best_placement is None:
            fallback = WillSeedPlacer(seed=42, refine_iters=self.will_refine_iters).place(benchmark)
            best_placement = self._repair_candidate(fallback, benchmark)

        return best_placement

    def _chain_portfolio(self, benchmark: Benchmark) -> Sequence[str]:
        if benchmark.num_hard_macros > 320:
            return ()
        if benchmark.num_hard_macros <= 260:
            return (
                "hierarchical,macro_spread,legalize",
                "hierarchical,legalize,local_swap",
            )
        return (
            "hierarchical,macro_spread,legalize",
            "hierarchical,force_smooth,legalize",
        )

    def _chain_config(self) -> Dict[str, Any]:
        return {
            "macro_spread": {"iterations": 18, "strength": 0.9},
            "force_smooth": {"iterations": 18, "attraction": 0.10},
            "local_swap": {"iterations": 80, "require_legal": True},
            "placer": {"search_radii": 150, "step_scale": 0.25, "safety_gap": 0.03},
        }

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _repair_candidate(self, placement: torch.Tensor, benchmark: Benchmark) -> torch.Tensor:
        repaired = placement.clone()
        repaired[benchmark.macro_fixed] = benchmark.macro_positions[benchmark.macro_fixed]
        repaired = self._clamp_all_hard(repaired, benchmark)
        if self._hard_overlap_count(repaired, benchmark) == 0:
            return repaired
        legalizer = WillSeedPlacer(seed=42, refine_iters=0)
        n = benchmark.num_hard_macros
        sizes = benchmark.macro_sizes[:n].numpy().astype(np.float64)
        pos = repaired[:n].numpy().copy().astype(np.float64)
        movable = benchmark.get_movable_mask()[:n].numpy()
        half_w = sizes[:, 0] / 2.0
        half_h = sizes[:, 1] / 2.0
        legal = legalizer._legalize(
            pos,
            movable,
            sizes,
            half_w,
            half_h,
            float(benchmark.canvas_width),
            float(benchmark.canvas_height),
            n,
        )
        repaired[:n] = torch.tensor(legal, dtype=repaired.dtype)
        repaired[benchmark.macro_fixed] = benchmark.macro_positions[benchmark.macro_fixed]
        return self._clamp_all_hard(repaired, benchmark)

    def _clamp_all_hard(self, placement: torch.Tensor, benchmark: Benchmark) -> torch.Tensor:
        clamped = placement.clone()
        n = benchmark.num_hard_macros
        sizes = benchmark.macro_sizes[:n]
        clamped[:n, 0] = torch.maximum(clamped[:n, 0], sizes[:, 0] / 2.0)
        clamped[:n, 0] = torch.minimum(
            clamped[:n, 0], torch.tensor(float(benchmark.canvas_width)) - sizes[:, 0] / 2.0
        )
        clamped[:n, 1] = torch.maximum(clamped[:n, 1], sizes[:, 1] / 2.0)
        clamped[:n, 1] = torch.minimum(
            clamped[:n, 1], torch.tensor(float(benchmark.canvas_height)) - sizes[:, 1] / 2.0
        )
        return clamped

    def _is_valid(self, placement: torch.Tensor, benchmark: Benchmark) -> bool:
        if placement.shape != (benchmark.num_macros, 2):
            return False
        if torch.isnan(placement).any() or torch.isinf(placement).any():
            return False
        if self._hard_overlap_count(placement, benchmark) != 0:
            return False
        return True

    def _score(self, placement: torch.Tensor, benchmark: Benchmark, plc: Optional[Any]) -> float:
        if plc is not None:
            costs = compute_proxy_cost(placement, benchmark, plc)
            if costs["overlap_count"] == 0:
                return float(costs["proxy_cost"])
        return self._surrogate_score(placement, benchmark)

    def _surrogate_score(self, placement: torch.Tensor, benchmark: Benchmark) -> float:
        hpwl = 0.0
        for nodes in benchmark.net_nodes:
            macro_nodes = [int(idx) for idx in nodes.tolist() if int(idx) < benchmark.num_hard_macros]
            if len(macro_nodes) < 2:
                continue
            pts = placement[macro_nodes]
            hpwl += float(torch.max(pts[:, 0]) - torch.min(pts[:, 0]))
            hpwl += float(torch.max(pts[:, 1]) - torch.min(pts[:, 1]))
        movement = torch.mean(torch.abs(placement[: benchmark.num_hard_macros] - benchmark.macro_positions[: benchmark.num_hard_macros]))
        spread = self._spread_penalty(placement, benchmark)
        return hpwl / max(float(benchmark.num_nets), 1.0) + 0.01 * float(movement) + spread

    def _spread_penalty(self, placement: torch.Tensor, benchmark: Benchmark) -> float:
        n = benchmark.num_hard_macros
        if n == 0:
            return 0.0
        positions = placement[:n]
        cx = float(benchmark.canvas_width) / 2.0
        cy = float(benchmark.canvas_height) / 2.0
        dist = torch.sqrt((positions[:, 0] - cx) ** 2 + (positions[:, 1] - cy) ** 2)
        return 0.001 * float(torch.mean(dist))

    def _hard_overlap_count(self, placement: torch.Tensor, benchmark: Benchmark) -> int:
        n = benchmark.num_hard_macros
        if n <= 1:
            return 0
        pos = placement[:n].detach().cpu().numpy()
        sizes = benchmark.macro_sizes[:n].detach().cpu().numpy()
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                min_sep_x = (sizes[i, 0] + sizes[j, 0]) / 2.0 + self.safety_gap
                min_sep_y = (sizes[i, 1] + sizes[j, 1]) / 2.0 + self.safety_gap
                if abs(pos[i, 0] - pos[j, 0]) < min_sep_x and abs(pos[i, 1] - pos[j, 1]) < min_sep_y:
                    count += 1
        return count
