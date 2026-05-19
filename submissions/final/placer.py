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
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/macro-placement-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/macro-placement-cache")

import numpy as np
import torch

from macro_place.benchmark import Benchmark
from macro_place.objective import compute_proxy_cost


CandidateFactory = Callable[[], torch.Tensor]
CandidateSpec = Tuple[str, str, CandidateFactory]


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
    CHAIN_PORTFOLIO_BUCKETS: Sequence[Tuple[int, Sequence[str]]] = (
        (
            220,
            (
                "hierarchical,macro_spread,legalize",
                "hierarchical,legalize,local_swap",
                "anchor,analytical_stage1,legalize",
                "anchor,periphery_bias,analytical_stage1,macro_spread,legalize",
            ),
        ),
        (
            280,
            (
                "anchor,analytical_stage1,legalize",
                "anchor,periphery_bias,analytical_stage1,macro_spread,legalize",
                "hierarchical,macro_spread,legalize",
                "hierarchical,legalize",
            ),
        ),
        (
            320,
            ("anchor,analytical_stage1,legalize",),
        ),
    )

    def __init__(self, debug: Optional[bool] = None, log_path: Optional[str] = None) -> None:
        self.seed = 314159
        self.will_seeds = (42, 7, 13, 23, 101, 271)
        self.will_refine_iters = 3600
        self.safety_gap = 0.0
        self.debug = (
            self._env_bool("FINAL_PLACER_DEBUG", default=self._env_bool("MACRO_PLACER_DEBUG"))
            if debug is None
            else debug
        )
        self.log_path = (
            log_path
            or os.environ.get("FINAL_PLACER_LOG_PATH")
            or os.environ.get("MACRO_PLACER_LOG_PATH")
        )
        self._candidate_records: List[Dict[str, Any]] = []
        self._candidate_metadata_by_name: Dict[str, Any] = {}

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        self._set_seed(self.seed)
        plc = _load_plc(benchmark.name)
        self._candidate_records = []
        self._candidate_metadata_by_name = {}
        best_score = float("inf")
        best_placement: Optional[torch.Tensor] = None
        best_candidate_name: Optional[str] = None

        for name, candidate_type, factory in self._generate_candidates(benchmark):
            placement: Optional[torch.Tensor] = None
            generation_runtime = 0.0
            candidate_start = time.perf_counter()
            try:
                generation_start = time.perf_counter()
                placement = factory()
                generation_runtime = time.perf_counter() - generation_start
            except Exception as exc:
                candidate_metadata = self._candidate_metadata_by_name.pop(name, None)
                self._record_candidate(
                    benchmark=benchmark,
                    candidate=name,
                    candidate_type=candidate_type,
                    generation_runtime_sec=time.perf_counter() - candidate_start,
                    repair_runtime_sec=0.0,
                    score_runtime_sec=0.0,
                    total_runtime_sec=time.perf_counter() - candidate_start,
                    valid=False,
                    overlap_count=None,
                    score=None,
                    became_best=False,
                    error=str(exc),
                    candidate_metadata=candidate_metadata,
                )
                continue

            candidate, repair_runtime = self._timed_repair_candidate(placement, benchmark)
            valid = self._is_valid(candidate, benchmark)
            overlap_count = self._hard_overlap_count(candidate, benchmark)
            score: Optional[float] = None
            score_runtime = 0.0
            error: Optional[str] = None

            if valid:
                try:
                    score_start = time.perf_counter()
                    score, score_metrics = self._score_with_metrics(candidate, benchmark, plc)
                    score_runtime = time.perf_counter() - score_start
                    overlap_count = int(score_metrics.get("overlap_count", overlap_count))
                except Exception as exc:
                    error = str(exc)

            became_best = bool(valid and score is not None and score < best_score)
            if score is not None and score < best_score:
                best_score = score
                best_placement = candidate
                best_candidate_name = name

            candidate_metadata = self._candidate_metadata_by_name.pop(name, None)
            self._record_candidate(
                benchmark=benchmark,
                candidate=name,
                candidate_type=candidate_type,
                generation_runtime_sec=generation_runtime,
                repair_runtime_sec=repair_runtime,
                score_runtime_sec=score_runtime,
                total_runtime_sec=time.perf_counter() - candidate_start,
                valid=valid,
                overlap_count=overlap_count,
                score=score,
                became_best=became_best,
                error=error,
                candidate_metadata=candidate_metadata,
            )

        if best_placement is None:
            fallback_start = time.perf_counter()
            fallback = WillSeedPlacer(seed=42, refine_iters=self.will_refine_iters).place(benchmark)
            best_placement, repair_runtime = self._timed_repair_candidate(fallback, benchmark)
            self._record_candidate(
                benchmark=benchmark,
                candidate="fallback_will_seed",
                candidate_type="fallback",
                generation_runtime_sec=time.perf_counter() - fallback_start - repair_runtime,
                repair_runtime_sec=repair_runtime,
                score_runtime_sec=0.0,
                total_runtime_sec=time.perf_counter() - fallback_start,
                valid=self._is_valid(best_placement, benchmark),
                overlap_count=self._hard_overlap_count(best_placement, benchmark),
                score=None,
                became_best=True,
            )
            best_candidate_name = "fallback_will_seed"

        self._mark_winner(best_score, best_candidate_name)
        self._flush_candidate_log()
        if self.debug:
            self._print_summary(benchmark)
        return best_placement

    def _env_bool(self, name: str, default: bool = False) -> bool:
        value = os.environ.get(name)
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "on"}

    def _generate_candidates(self, benchmark: Benchmark) -> Sequence[CandidateSpec]:
        specs: List[CandidateSpec] = [
            (
                "will_seed_baseline",
                "will_seed",
                lambda: WillSeedPlacer(seed=42, refine_iters=3000).place(benchmark),
            )
        ]

        if benchmark.num_hard_macros > 320:
            return specs

        for seed in self.will_seeds:
            specs.append(
                (
                    f"will_seed_{seed}",
                    "will_seed",
                    lambda seed=seed: WillSeedPlacer(
                        seed=seed,
                        refine_iters=self.will_refine_iters,
                    ).place(benchmark),
                )
            )

        specs.append(
            (
                "initializer_ensemble",
                "initializer_ensemble",
                lambda: EnsembleInitializerPlacer(seed=self.seed).place(benchmark),
            )
        )

        for chain in self._chain_portfolio(benchmark):
            specs.append(
                (
                    f"chain:{chain}",
                    "chain",
                    lambda chain=chain: self._run_chain_candidate(benchmark, chain),
                )
            )
        return specs

    def _chain_portfolio(self, benchmark: Benchmark) -> Sequence[str]:
        n = benchmark.num_hard_macros
        for max_macros, chains in self.CHAIN_PORTFOLIO_BUCKETS:
            if n <= max_macros:
                return chains
        return ()

    def _chain_config(self, benchmark: Benchmark) -> Dict[str, Any]:
        n = benchmark.num_hard_macros
        if n <= 220:
            macro_spread_iterations = 18
            force_smooth_iterations = 18
            local_swap_iterations = 80
            search_radii = 150
            chain_budget_sec = 20.0
            stage_budget_sec = 8.0
        elif n <= 280:
            macro_spread_iterations = 12
            force_smooth_iterations = 8
            local_swap_iterations = 30
            search_radii = 100
            chain_budget_sec = 20.0
            stage_budget_sec = 8.0
        else:
            macro_spread_iterations = 4
            force_smooth_iterations = 4
            analytical_iterations = 4
            local_swap_iterations = 0
            search_radii = 60
            chain_budget_sec = 14.0
            stage_budget_sec = 6.0
            periphery_strength = 0.10
            periphery_fraction = 0.08
        if n <= 220:
            analytical_iterations = 16
            periphery_strength = 0.10
            periphery_fraction = 0.08
        elif n <= 280:
            analytical_iterations = 16
            periphery_strength = 0.10
            periphery_fraction = 0.08

        canvas_scale = max(float(benchmark.canvas_width), float(benchmark.canvas_height), 1.0)
        return {
            "chain_budget_sec": chain_budget_sec,
            "stage_budget_sec": stage_budget_sec,
            "macro_spread": {"iterations": macro_spread_iterations, "strength": 0.9},
            "force_smooth": {"iterations": force_smooth_iterations, "attraction": 0.10},
            "periphery_bias": {
                "strength": periphery_strength,
                "boundary_fraction": periphery_fraction,
                "area_weight": 0.24,
                "io_weight": 0.44,
                "degree_weight": 0.10,
                "centrality_penalty": 0.50,
                "min_side_gap_fraction": 0.035,
                "max_move_fraction": 0.018,
                "min_score": 0.25,
            },
            "analytical_stage1": {
                "iterations": analytical_iterations,
                "attraction": 0.025,
                "overlap_repulsion": 0.80,
                "density_repulsion": 0.05,
                "boundary_repulsion": 0.08,
                "periphery_attraction": 0.0,
                "spread_repulsion": 0.10,
                "max_move": (0.010 if n <= 280 else 0.008) * canvas_scale,
                "bin_count": 7 if n <= 280 else 6,
                "boundary_fraction": periphery_fraction,
                "target_density": 0.74,
                "max_pairwise_macros": 340,
                "min_side_gap_fraction": 0.035,
            },
            "local_swap": {"iterations": local_swap_iterations, "require_legal": True},
            "local_shift": {"iterations": 0},
            "spectral_order": {"max_macros": 320},
            "placer": {"search_radii": search_radii, "step_scale": 0.25, "safety_gap": 0.03},
        }

    def _run_chain_candidate(self, benchmark: Benchmark, chain: str) -> torch.Tensor:
        placement, metadata = run_initializer_chain(
            benchmark,
            chain,
            seed=self.seed,
            config=self._chain_config(benchmark),
            collect_metrics=True,
            plc=None,
        )
        self._candidate_metadata_by_name[f"chain:{chain}"] = metadata
        return placement

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

    def _timed_repair_candidate(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
    ) -> Tuple[torch.Tensor, float]:
        start = time.perf_counter()
        repaired = self._repair_candidate(placement, benchmark)
        return repaired, time.perf_counter() - start

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
        score, _ = self._score_with_metrics(placement, benchmark, plc)
        return score

    def _score_with_metrics(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        plc: Optional[Any],
    ) -> Tuple[float, Dict[str, Any]]:
        if plc is not None:
            costs = compute_proxy_cost(placement, benchmark, plc)
            if costs["overlap_count"] == 0:
                return float(costs["proxy_cost"]), costs
            return self._surrogate_score(placement, benchmark), costs
        overlap_count = self._hard_overlap_count(placement, benchmark)
        return self._surrogate_score(placement, benchmark), {"overlap_count": overlap_count}

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

    def _record_candidate(
        self,
        benchmark: Benchmark,
        candidate: str,
        candidate_type: str,
        generation_runtime_sec: float,
        repair_runtime_sec: float,
        score_runtime_sec: float,
        total_runtime_sec: float,
        valid: bool,
        overlap_count: Optional[int],
        score: Optional[float],
        became_best: bool,
        winner: bool = False,
        error: Optional[str] = None,
        candidate_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        record: Dict[str, Any] = {
            "benchmark": benchmark.name,
            "num_hard_macros": benchmark.num_hard_macros,
            "candidate": candidate,
            "candidate_type": candidate_type,
            "generation_runtime_sec": float(generation_runtime_sec),
            "repair_runtime_sec": float(repair_runtime_sec),
            "score_runtime_sec": float(score_runtime_sec),
            "total_runtime_sec": float(total_runtime_sec),
            "valid": bool(valid),
            "overlap_count": overlap_count,
            "score": float(score) if score is not None else None,
            "became_best": bool(became_best),
            "winner": bool(winner),
        }
        if error:
            record["error"] = error
        if candidate_metadata is not None:
            record["chain_runtime_sec"] = candidate_metadata.get("runtime_sec")
            record["chain_budget_stopped"] = candidate_metadata.get("budget_stopped")
            record["chain_final_operator"] = candidate_metadata.get("final_operator")
            stage_metrics = candidate_metadata.get("stage_metrics", [])
            if stage_metrics:
                record["stage_metrics"] = stage_metrics
                final_stage = stage_metrics[-1]
                record["chain_final_overlap_count"] = final_stage.get("overlap_count")
                record["chain_final_total_overlap_area"] = final_stage.get("total_overlap_area")
                record["chain_final_max_bin_density"] = final_stage.get("max_bin_density")
                record["chain_final_density_overflow_energy"] = final_stage.get(
                    "density_overflow_energy"
                )
        self._candidate_records.append(record)

    def _mark_winner(self, best_score: float, best_candidate_name: Optional[str]) -> None:
        if best_candidate_name is not None:
            for record in self._candidate_records:
                if record.get("candidate") == best_candidate_name:
                    record["winner"] = True
                    return
        best_index: Optional[int] = None
        for index, record in enumerate(self._candidate_records):
            if record.get("score") is None:
                continue
            if abs(float(record["score"]) - best_score) <= 1e-12 and record.get("valid"):
                best_index = index
        if best_index is not None:
            self._candidate_records[best_index]["winner"] = True
            return
        for index in range(len(self._candidate_records) - 1, -1, -1):
            if self._candidate_records[index].get("became_best"):
                self._candidate_records[index]["winner"] = True
                return

    def _flush_candidate_log(self) -> None:
        if not self.log_path:
            return
        path = Path(self.log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            for record in self._candidate_records:
                f.write(json.dumps(record, sort_keys=True) + "\n")

    def _print_summary(self, benchmark: Benchmark) -> None:
        print(f"Benchmark {benchmark.name}, hard_macros={benchmark.num_hard_macros}")
        print("Candidate summary:")
        for record in self._candidate_records:
            score = record.get("score")
            score_text = f"{score:.4f}" if isinstance(score, (float, int)) else "-"
            runtime = record.get("total_runtime_sec")
            runtime_text = f"{float(runtime):.2f}s" if runtime is not None else "-"
            suffix = f" error={record['error']}" if record.get("error") else ""
            print(
                f"  {record['candidate']:<44} "
                f"score={score_text:<8} valid={record['valid']!s:<5} "
                f"runtime={runtime_text}{suffix}"
            )
        winner = next((record for record in self._candidate_records if record.get("winner")), None)
        if winner is not None:
            score = winner.get("score")
            score_text = f"{score:.4f}" if isinstance(score, (float, int)) else "-"
            print(f"Winner: {winner['candidate']} score={score_text}")
