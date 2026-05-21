"""
Adaptive branch-family parameter tuner for macro placement.

This model keeps the useful lesson from the IBM01 experiments without turning
the implementation into an IBM01-only script:

1. Probe a few known-good branch families.
2. Pick the best basin for the current benchmark.
3. Spend the remaining budget on a small directed sweep around that basin.

It uses the existing initializer/operator framework as the execution primitive.
There is no offline training, broad chain search, or operator-internal
parallelism here.
"""

from __future__ import annotations

import copy
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


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


MODE = os.environ.get("ML_PARAM_MODE", "adaptive").strip().lower()
FAST_MODE = _env_bool("ML_PARAM_FAST_MODE", False)
TOTAL_BUDGET_SEC = _env_float("ML_PARAM_TOTAL_BUDGET_SEC", 8.0 * 60.0)
BRANCH_PROBE_BUDGET_SEC = _env_float("ML_PARAM_BRANCH_PROBE_BUDGET_SEC", 60.0)
SWEEP_BUDGET_SEC = _env_float("ML_PARAM_SWEEP_BUDGET_SEC", 180.0)
MAX_BRANCH_FAMILIES = _env_int("ML_PARAM_MAX_BRANCH_FAMILIES", 1)
MAX_VARIANTS = _env_int("ML_PARAM_MAX_VARIANTS", 10)
ENABLE_BROAD_FAMILIES = _env_bool(
    "ML_PARAM_ENABLE_BROAD_FAMILIES",
    _env_bool("ML_PARAM_ENABLE_EXTRA_BRANCHES", False),
)
ENABLE_LOCAL_REFINEMENT = _env_bool(
    "ML_PARAM_ENABLE_LOCAL_REFINEMENT",
    False,
)
USE_PREFIX_REUSE = _env_bool("ML_PARAM_USE_PREFIX_REUSE", False)
RANDOM_SEED = _env_int("ML_PARAM_RANDOM_SEED", 314159)
LOCAL_REFINEMENT_USE_PROXY = _env_bool("ML_PARAM_LOCAL_USE_PROXY", False)

SAFE_BASELINE_CHAIN = "anchor,legalize"
STAGE1_SPREAD_CHAIN = "anchor,periphery_bias,analytical_stage1,macro_spread,legalize"
STAGE1_CHAIN = "anchor,analytical_stage1,legalize"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_initializer_module = _load_module(
    _repo_root() / "submissions" / "models" / "initializer.py",
    "ml_param_initializer_runtime",
)
run_initializer_chain = _initializer_module.run_initializer_chain
collect_initializer_metrics = _initializer_module.collect_initializer_metrics
EnsembleInitializerPlacer = _initializer_module.EnsembleInitializerPlacer
_WillSeedPlacer = None


def _will_seed_placer_class():
    global _WillSeedPlacer
    if _WillSeedPlacer is None:
        module = _load_module(
            _repo_root() / "submissions" / "will_seed" / "placer.py",
            "ml_param_will_seed_runtime",
        )
        _WillSeedPlacer = module.WillSeedPlacer
    return _WillSeedPlacer


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
    if not design:
        return None
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


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.item() if value.numel() == 1 else value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _last_stage(metadata: Dict[str, Any]) -> Dict[str, Any]:
    stages = metadata.get("stage_metrics") or []
    return stages[-1] if stages else {}


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class ParamPreset:
    def __init__(
        self,
        name: str,
        config: Dict[str, Any],
        variant_family: str = "probe",
        notes: str = "",
    ) -> None:
        self.name = name
        self.config = copy.deepcopy(config)
        self.variant_family = variant_family
        self.notes = notes


class CandidateResult:
    def __init__(
        self,
        benchmark: str,
        mode: str,
        stage: str,
        branch_family: str,
        candidate_name: str,
        chain: Optional[str],
        config: Dict[str, Any],
        seed: int,
        label: str,
        variant_family: str = "probe",
        parent_candidate: Optional[str] = None,
        prefix_reuse_requested: bool = False,
        prefix_reuse_used: bool = False,
    ) -> None:
        self.benchmark = benchmark
        self.mode = mode
        self.stage = stage
        self.branch_family = branch_family
        self.candidate_name = candidate_name
        self.chain = chain
        self.config = copy.deepcopy(config)
        self.seed = seed
        self.label = label
        self.variant_family = variant_family
        self.parent_candidate = parent_candidate
        self.placement: Optional[torch.Tensor] = None
        self.metadata: Dict[str, Any] = {}
        self.runtime_sec: Optional[float] = None
        self.proxy_cost: Optional[float] = None
        self.overlap_count: Optional[int] = None
        self.boundary_violations: Optional[int] = None
        self.legal = False
        self.fallback_score = float("inf")
        self.score_key: Tuple[Any, ...] = (9, float("inf"))
        self.error: Optional[str] = None
        self.selected = False
        self.accepted = False
        self.rejected_reason: Optional[str] = None
        self.delta_vs_parent: Optional[float] = None
        self.prefix_reuse_requested = prefix_reuse_requested
        self.prefix_reuse_used = prefix_reuse_used


VariantBuilder = Callable[[Benchmark, CandidateResult, int], Sequence[ParamPreset]]
ConfigBuilder = Callable[[Benchmark], Dict[str, Any]]
Suitability = Callable[[Benchmark], bool]


class BranchFamily:
    def __init__(
        self,
        name: str,
        kind: str,
        probe_presets: Sequence[ParamPreset],
        chain: Optional[str] = None,
        base_config_builder: Optional[ConfigBuilder] = None,
        variant_builder: Optional[VariantBuilder] = None,
        enabled_by_default: bool = True,
        explore_only: bool = False,
        suitability: Optional[Suitability] = None,
    ) -> None:
        self.name = name
        self.kind = kind
        self.probe_presets = tuple(probe_presets)
        self.chain = chain
        self.base_config_builder = base_config_builder
        self.variant_builder = variant_builder
        self.enabled_by_default = enabled_by_default
        self.explore_only = explore_only
        self.suitability = suitability

    def suitable(self, benchmark: Benchmark) -> bool:
        return True if self.suitability is None else bool(self.suitability(benchmark))


def final_like_config(benchmark: Benchmark) -> Dict[str, Any]:
    """Config shape shared with the current final tournament chain candidates."""
    n = int(benchmark.num_hard_macros)
    if n <= 220:
        macro_spread_iterations = 18
        force_smooth_iterations = 18
        local_swap_iterations = 80
        search_radii = 150
        chain_budget_sec = 20.0
        stage_budget_sec = 8.0
        analytical_iterations = 16
        periphery_strength = 0.10
        periphery_fraction = 0.08
    elif n <= 280:
        macro_spread_iterations = 12
        force_smooth_iterations = 8
        local_swap_iterations = 30
        search_radii = 100
        chain_budget_sec = 20.0
        stage_budget_sec = 8.0
        analytical_iterations = 16
        periphery_strength = 0.10
        periphery_fraction = 0.08
    else:
        macro_spread_iterations = 4
        force_smooth_iterations = 4
        local_swap_iterations = 0
        search_radii = 60
        chain_budget_sec = 14.0
        stage_budget_sec = 6.0
        analytical_iterations = 4
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


def safe_anchor_config(_benchmark: Benchmark) -> Dict[str, Any]:
    return {
        "chain_budget_sec": min(BRANCH_PROBE_BUDGET_SEC, 90.0),
        "stage_budget_sec": min(BRANCH_PROBE_BUDGET_SEC, 90.0),
        "placer": {"search_radii": 220, "step_scale": 0.25, "safety_gap": 0.03},
    }


def low_gap_wide_legalizer_config() -> Dict[str, Any]:
    """IBM01-discovered basin, expressed as branch parameters, not a benchmark gate."""
    return {
        "macro_spread": {"iterations": 12, "strength": 0.88},
        "periphery_bias": {
            "strength": 0.0803758500091099,
            "max_move_fraction": 0.020869102336449195,
        },
        "analytical_stage1": {
            "attraction": 0.02366152508632894,
            "overlap_repulsion": 0.6082540067636194,
        },
        "placer": {"search_radii": 180, "step_scale": 0.22, "safety_gap": 0.006},
    }


def legalizer_wide_config() -> Dict[str, Any]:
    return {
        "macro_spread": {"iterations": 12, "strength": 0.88},
        "placer": {"search_radii": 180, "step_scale": 0.22, "safety_gap": 0.02},
    }


def _variant(name: str, override: Dict[str, Any], family: str, notes: str = "") -> ParamPreset:
    return ParamPreset(name=name, config=override, variant_family=family, notes=notes)


def build_stage1_spread_variants(
    _benchmark: Benchmark,
    parent: CandidateResult,
    limit: int,
) -> Sequence[ParamPreset]:
    base = copy.deepcopy(parent.config)
    variants = [
        _variant("LEGALIZER_GAP_ZERO", {"placer": {"safety_gap": 0.0}}, "legalizer"),
        _variant("LEGALIZER_GAP_012", {"placer": {"safety_gap": 0.012}}, "legalizer"),
        _variant(
            "LEGALIZER_RADIUS_220",
            {"placer": {"search_radii": 220, "step_scale": 0.20}},
            "legalizer",
        ),
        _variant(
            "LEGALIZER_RADIUS_160",
            {"placer": {"search_radii": 160, "step_scale": 0.22}},
            "legalizer",
        ),
        _variant(
            "STAGE1_OVERLAP_LOW",
            {"analytical_stage1": {"overlap_repulsion": 0.48}},
            "stage1",
        ),
        _variant(
            "STAGE1_OVERLAP_MED",
            {"analytical_stage1": {"overlap_repulsion": 0.70}},
            "stage1",
        ),
        _variant(
            "MACRO_SPREAD_SOFT",
            {"macro_spread": {"iterations": 12, "strength": 0.78}},
            "macro_spread",
        ),
        _variant(
            "MACRO_SPREAD_MORE",
            {"macro_spread": {"iterations": 16, "strength": 0.92}},
            "macro_spread",
        ),
        _variant(
            "PERIPHERY_SOFT",
            {"periphery_bias": {"strength": 0.070, "max_move_fraction": 0.016}},
            "periphery",
        ),
        _variant(
            "PERIPHERY_WIDER",
            {"periphery_bias": {"boundary_fraction": 0.10, "max_move_fraction": 0.024}},
            "periphery",
        ),
        _variant(
            "DENSITY_RELAXED",
            {"analytical_stage1": {"density_repulsion": 0.040, "target_density": 0.80}},
            "stage1",
        ),
        _variant(
            "STAGE1_ATTRACTION_LOW",
            {"analytical_stage1": {"attraction": 0.020}},
            "stage1",
        ),
    ]
    jobs: List[ParamPreset] = []
    seen = {json.dumps(_jsonable(base), sort_keys=True, separators=(",", ":"))}
    for item in variants:
        config = _deep_merge(base, item.config)
        signature = json.dumps(_jsonable(config), sort_keys=True, separators=(",", ":"))
        if signature in seen:
            continue
        seen.add(signature)
        jobs.append(
            ParamPreset(
                name=f"{parent.candidate_name}_{item.name}",
                config=config,
                variant_family=item.variant_family,
                notes=f"Directed {item.variant_family} sweep around {parent.label}",
            )
        )
        if len(jobs) >= limit:
            break
    return jobs


def build_stage1_variants(
    _benchmark: Benchmark,
    parent: CandidateResult,
    limit: int,
) -> Sequence[ParamPreset]:
    base = copy.deepcopy(parent.config)
    variants = [
        _variant("LEGALIZER_WIDE", {"placer": {"search_radii": 180, "step_scale": 0.22}}, "legalizer"),
        _variant("LEGALIZER_GAP_LOW", {"placer": {"safety_gap": 0.006}}, "legalizer"),
        _variant("STAGE1_OVERLAP_LOW", {"analytical_stage1": {"overlap_repulsion": 0.60}}, "stage1"),
        _variant("STAGE1_DENSITY_RELAXED", {"analytical_stage1": {"density_repulsion": 0.040}}, "stage1"),
        _variant("STAGE1_MORE_ITERS", {"analytical_stage1": {"iterations": 22}}, "stage1"),
    ]
    jobs: List[ParamPreset] = []
    for item in variants[:limit]:
        jobs.append(
            ParamPreset(
                name=f"{parent.candidate_name}_{item.name}",
                config=_deep_merge(base, item.config),
                variant_family=item.variant_family,
                notes=f"Directed {item.variant_family} sweep around {parent.label}",
            )
        )
    return jobs


def build_branch_families() -> List[BranchFamily]:
    small_chain = lambda benchmark: int(benchmark.num_hard_macros) <= 320
    large_macro = lambda benchmark: int(benchmark.num_hard_macros) > 320
    return [
        BranchFamily(
            name="anchor_stage1_spread",
            kind="chain",
            chain=STAGE1_SPREAD_CHAIN,
            base_config_builder=final_like_config,
            probe_presets=(
                ParamPreset(
                    "LOW_GAP_WIDE_LEGALIZER",
                    low_gap_wide_legalizer_config(),
                    notes="Best discovered Stage-1/spread basin from IBM01 logs.",
                ),
                ParamPreset("FINAL_REPLICA", {}, notes="Current final-tournament chain config."),
                ParamPreset("LEGALIZER_WIDE", legalizer_wide_config()),
            ),
            variant_builder=build_stage1_spread_variants,
            suitability=small_chain,
        ),
        BranchFamily(
            name="anchor_stage1",
            kind="chain",
            chain=STAGE1_CHAIN,
            base_config_builder=final_like_config,
            probe_presets=(
                ParamPreset("FINAL_RUNNER_UP", {}, notes="Current final-tournament runner-up chain."),
            ),
            variant_builder=build_stage1_variants,
            suitability=small_chain,
        ),
        BranchFamily(
            name="will_seed_large",
            kind="will_seed",
            probe_presets=(
                ParamPreset("SEED_42_3600", {"seed": 42, "refine_iters": 3600}),
            ),
            enabled_by_default=True,
            suitability=large_macro,
        ),
        BranchFamily(
            name="will_seed",
            kind="will_seed",
            probe_presets=(
                ParamPreset("SEED_101_3600", {"seed": 101, "refine_iters": 3600}),
                ParamPreset("SEED_42_3600", {"seed": 42, "refine_iters": 3600}),
                ParamPreset("SEED_13_3600", {"seed": 13, "refine_iters": 3600}),
            ),
            enabled_by_default=False,
            explore_only=True,
        ),
        BranchFamily(
            name="initializer_ensemble",
            kind="initializer_ensemble",
            probe_presets=(
                ParamPreset("FINAL_ENSEMBLE", {"seed": RANDOM_SEED}),
                ParamPreset("WIDE_LEGALIZER", {"seed": RANDOM_SEED, "search_radii": 220, "safety_gap": 0.02}),
            ),
            enabled_by_default=False,
            explore_only=True,
        ),
    ]


class MLParamTournamentPlacer:
    """Compact adaptive tuner over known-good branch families."""

    def __init__(
        self,
        seed: int = RANDOM_SEED,
        log_path: Optional[str] = None,
        debug: Optional[bool] = None,
    ) -> None:
        self.seed = seed
        self.mode = "fast" if FAST_MODE else (MODE if MODE in {"adaptive", "fast", "explore"} else "adaptive")
        self.total_budget_sec = TOTAL_BUDGET_SEC
        self.probe_budget_sec = BRANCH_PROBE_BUDGET_SEC
        self.sweep_budget_sec = SWEEP_BUDGET_SEC
        self.max_branch_families = max(1, MAX_BRANCH_FAMILIES)
        self.max_variants = max(0, MAX_VARIANTS)
        self.enable_broad_families = ENABLE_BROAD_FAMILIES
        self.enable_local_refinement = ENABLE_LOCAL_REFINEMENT
        self.use_prefix_reuse = USE_PREFIX_REUSE
        self.log_path = (
            log_path
            or os.environ.get("ML_PARAM_TOURNAMENT_LOG_PATH")
            or os.environ.get("MACRO_PLACER_LOG_PATH")
        )
        self.debug = _env_bool("ML_PARAM_TOURNAMENT_DEBUG", False) if debug is None else debug
        self._results: List[CandidateResult] = []
        self._start_time = 0.0
        self._current_benchmark: Optional[Benchmark] = None

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        self._set_seed(self.seed)
        self._results = []
        self._start_time = time.perf_counter()
        self._current_benchmark = benchmark
        plc = _load_plc(benchmark.name)

        baseline_family = BranchFamily(
            name="safe_anchor_legalize",
            kind="chain",
            chain=SAFE_BASELINE_CHAIN,
            base_config_builder=safe_anchor_config,
            probe_presets=(ParamPreset("SAFE_ANCHOR", {}),),
        )
        baseline = self.evaluate_family(
            benchmark=benchmark,
            family=baseline_family,
            preset=baseline_family.probe_presets[0],
            plc=plc,
            stage="baseline",
        )
        best = baseline
        if baseline.placement is None:
            return benchmark.macro_positions.clone()

        probe_results: List[CandidateResult] = []
        for family in self.enabled_families(benchmark):
            for preset in self.probe_presets_for(family):
                if self.budget_exhausted():
                    break
                result = self.evaluate_family(benchmark, family, preset, plc, stage="branch_probe")
                probe_results.append(result)
                best = self.select_best_result(best, result, baseline)

        selected_families = self.select_sweep_parents(probe_results)
        variant_limit = 3 if self.mode == "fast" else self.max_variants
        for parent in selected_families:
            if self.budget_exhausted() or variant_limit <= 0:
                break
            family = self.family_by_name(parent.branch_family)
            if family is None or family.variant_builder is None:
                continue
            for preset in family.variant_builder(benchmark, parent, variant_limit):
                if self.budget_exhausted():
                    break
                result = self.evaluate_family(
                    benchmark,
                    family,
                    preset,
                    plc,
                    stage="directed_sweep",
                    parent_candidate=parent.label,
                )
                if result.proxy_cost is not None and parent.proxy_cost is not None:
                    result.delta_vs_parent = float(result.proxy_cost) - float(parent.proxy_cost)
                best = self.select_best_result(best, result, baseline)

        if self.enable_local_refinement and not self.budget_exhausted():
            for parent in self.rank_results(self._results)[:2]:
                if self.budget_exhausted() or not parent.legal:
                    continue
                best = self.run_local_refinement(benchmark, plc, parent, best, baseline)

        selected = self.final_selection(best, baseline)
        self.mark_selected(selected)
        self.flush_log()
        if self.debug:
            self.print_summary(benchmark, selected)
        placement = selected.placement if selected.placement is not None else baseline.placement
        if placement is None:
            placement = benchmark.macro_positions.clone()
        return self.restore_fixed_macros(placement.clone(), benchmark)

    def enabled_families(self, benchmark: Benchmark) -> List[BranchFamily]:
        families = []
        for family in build_branch_families():
            if not family.suitable(benchmark):
                continue
            if self.mode == "explore" or self.enable_broad_families:
                if family.enabled_by_default or family.explore_only:
                    families.append(family)
            elif family.enabled_by_default and not family.explore_only:
                families.append(family)
        return families

    def probe_presets_for(self, family: BranchFamily) -> Sequence[ParamPreset]:
        if self.mode == "fast":
            return family.probe_presets[:1]
        if self.mode == "explore" or self.enable_broad_families:
            return family.probe_presets
        return family.probe_presets[: min(3, len(family.probe_presets))]

    def select_sweep_parents(self, probe_results: Sequence[CandidateResult]) -> List[CandidateResult]:
        best_by_family: Dict[str, CandidateResult] = {}
        for result in self.rank_results(probe_results):
            if not result.legal:
                continue
            if result.branch_family not in best_by_family:
                best_by_family[result.branch_family] = result
        limit = 1 if self.mode == "fast" else self.max_branch_families
        return self.rank_results(list(best_by_family.values()))[:limit]

    def family_by_name(self, name: str) -> Optional[BranchFamily]:
        for family in build_branch_families():
            if family.name == name:
                return family
        return None

    def evaluate_family(
        self,
        benchmark: Benchmark,
        family: BranchFamily,
        preset: ParamPreset,
        plc: Optional[Any],
        stage: str,
        parent_candidate: Optional[str] = None,
        placement: Optional[torch.Tensor] = None,
        local_operator: Optional[str] = None,
    ) -> CandidateResult:
        effective_config = self.effective_config(benchmark, family, preset, stage)
        label = f"{stage}:{family.name}:{preset.name}"
        result = CandidateResult(
            benchmark=benchmark.name,
            mode=self.mode,
            stage=stage,
            branch_family=family.name,
            candidate_name=preset.name,
            chain=family.chain,
            config=effective_config,
            seed=self.seed,
            label=label,
            variant_family=preset.variant_family,
            parent_candidate=parent_candidate,
            prefix_reuse_requested=self.use_prefix_reuse,
            prefix_reuse_used=False,
        )
        start = time.perf_counter()
        try:
            if family.kind == "chain":
                out, metadata = run_initializer_chain(
                    benchmark,
                    family.chain,
                    seed=self.seed,
                    config=effective_config,
                    collect_metrics=True,
                    plc=plc,
                    placement=placement,
                )
                result.metadata = metadata
            elif family.kind == "will_seed":
                placer_cls = _will_seed_placer_class()
                out = placer_cls(
                    seed=int(effective_config.get("seed", 42)),
                    refine_iters=int(effective_config.get("refine_iters", 3600)),
                ).place(benchmark)
            elif family.kind == "initializer_ensemble":
                out = EnsembleInitializerPlacer(**effective_config).place(benchmark)
            else:
                raise ValueError(f"Unknown branch family kind: {family.kind}")

            result.runtime_sec = time.perf_counter() - start
            result.placement = self.restore_fixed_macros(out, benchmark)
            self.score_result(result, benchmark, plc)
        except Exception as exc:
            result.runtime_sec = time.perf_counter() - start
            result.error = str(exc)
            result.rejected_reason = "error"
        if local_operator is not None:
            result.variant_family = local_operator
        self._results.append(result)
        return result

    def effective_config(
        self,
        benchmark: Benchmark,
        family: BranchFamily,
        preset: ParamPreset,
        stage: str,
    ) -> Dict[str, Any]:
        base = family.base_config_builder(benchmark) if family.base_config_builder else {}
        config = _deep_merge(base, preset.config)
        if family.kind == "chain":
            budget = self.probe_budget_sec if stage in {"baseline", "branch_probe"} else self.sweep_budget_sec
            config.setdefault("chain_budget_sec", budget)
            config.setdefault("stage_budget_sec", min(float(config["chain_budget_sec"]), budget))
        return config

    def score_result(
        self,
        result: CandidateResult,
        benchmark: Benchmark,
        plc: Optional[Any],
    ) -> None:
        placement = result.placement
        metadata = result.metadata or {}
        final = _last_stage(metadata)
        proxy_cost = final.get("proxy_cost")
        overlap_count = final.get("overlap_count")
        boundary_violations = final.get("boundary_violations")
        legal = final.get("legal")

        if placement is not None:
            try:
                metrics = collect_initializer_metrics(placement, benchmark, plc=plc)
                result.metadata.setdefault("final_metrics", metrics)
                proxy_cost = proxy_cost if proxy_cost is not None else metrics.get("proxy_cost")
                overlap_count = overlap_count if overlap_count is not None else metrics.get("overlap_count")
                boundary_violations = (
                    boundary_violations
                    if boundary_violations is not None
                    else metrics.get("boundary_violations")
                )
                legal = legal if legal is not None else metrics.get("legal")
            except Exception as exc:
                result.metadata["metrics_error"] = str(exc)

        if placement is not None and plc is not None and proxy_cost is None:
            try:
                costs = compute_proxy_cost(placement, benchmark, plc)
                result.metadata["proxy_rescore_metrics"] = costs
                proxy_cost = costs.get("proxy_cost")
                overlap_count = costs.get("overlap_count", overlap_count)
            except Exception as exc:
                result.metadata["proxy_error"] = str(exc)

        overlap_int = None if overlap_count is None else int(overlap_count)
        boundary_int = None if boundary_violations is None else int(boundary_violations)
        legal_bool = bool(
            legal is True
            and overlap_int == 0
            and (boundary_int is None or boundary_int == 0)
        )
        proxy_float = _as_float(proxy_cost)
        fallback = self.fallback_score(result)

        if legal_bool and proxy_float is not None:
            score_key = (0, proxy_float, float(result.runtime_sec or 0.0))
        elif legal_bool:
            score_key = (1, fallback, float(result.runtime_sec or 0.0))
        else:
            score_key = (
                2,
                float(overlap_int if overlap_int is not None else 1e9),
                float(boundary_int if boundary_int is not None else 1e9),
                fallback,
            )

        result.legal = legal_bool
        result.proxy_cost = proxy_float
        result.overlap_count = overlap_int
        result.boundary_violations = boundary_int
        result.fallback_score = fallback
        result.score_key = score_key
        if not legal_bool:
            result.rejected_reason = "illegal_or_unverified"
        elif proxy_float is None:
            result.rejected_reason = "no_proxy_available"
        else:
            result.rejected_reason = "not_best_proxy"

    def fallback_score(self, result: CandidateResult) -> float:
        if result.proxy_cost is not None:
            return float(result.proxy_cost)
        metadata = result.metadata or {}
        metrics = metadata.get("final_metrics") or _last_stage(metadata)
        total_overlap = float(metrics.get("total_overlap_area") or 0.0)
        overlap_count = float(metrics.get("overlap_count") or result.overlap_count or 0.0)
        boundary = float(metrics.get("boundary_violations") or result.boundary_violations or 0.0)
        density = float(metrics.get("density_overflow_energy") or metrics.get("density_cost") or 0.0)
        hpwl = float(metrics.get("wirelength_cost") or metrics.get("hpwl") or 0.0)
        return 1e6 * boundary + 1e5 * overlap_count + 1e3 * total_overlap + 10.0 * density + hpwl

    def rank_results(self, results: Sequence[CandidateResult]) -> List[CandidateResult]:
        usable = [result for result in results if result.placement is not None]
        return sorted(usable, key=lambda result: result.score_key)

    def select_best_result(
        self,
        current_best: CandidateResult,
        candidate: CandidateResult,
        baseline: CandidateResult,
    ) -> CandidateResult:
        if candidate.placement is None or not candidate.legal:
            return current_best
        if current_best.placement is None or not current_best.legal:
            candidate.accepted = True
            candidate.rejected_reason = None
            return candidate
        if baseline.legal and not current_best.legal:
            current_best = baseline
        if self.is_better(candidate, current_best):
            candidate.accepted = True
            candidate.rejected_reason = None
            return candidate
        return current_best

    def is_better(self, candidate: CandidateResult, current: CandidateResult) -> bool:
        if candidate.proxy_cost is not None and current.proxy_cost is not None:
            return float(candidate.proxy_cost) < float(current.proxy_cost) - 1e-12
        if candidate.proxy_cost is not None and current.proxy_cost is None:
            return True
        if candidate.proxy_cost is None and current.proxy_cost is not None:
            return False
        return candidate.score_key < current.score_key

    def final_selection(self, best: CandidateResult, baseline: CandidateResult) -> CandidateResult:
        if baseline.legal:
            if not best.legal:
                return baseline
            if baseline.proxy_cost is not None and best.proxy_cost is not None:
                return best if float(best.proxy_cost) < float(baseline.proxy_cost) - 1e-12 else baseline
            return best if self.is_better(best, baseline) else baseline
        return best if best.placement is not None else baseline

    def run_local_refinement(
        self,
        benchmark: Benchmark,
        plc: Optional[Any],
        parent: CandidateResult,
        best: CandidateResult,
        baseline: CandidateResult,
    ) -> CandidateResult:
        source = parent
        for operator, config in (
            (
                "local_shift",
                {
                    "chain_budget_sec": self.sweep_budget_sec,
                    "stage_budget_sec": min(self.sweep_budget_sec, 90.0),
                    "local_shift": {
                        "iterations": 2,
                        "shift_fraction": 0.30,
                        "require_legal": True,
                        "use_proxy": LOCAL_REFINEMENT_USE_PROXY,
                    },
                },
            ),
            (
                "local_swap",
                {
                    "chain_budget_sec": self.sweep_budget_sec,
                    "stage_budget_sec": min(self.sweep_budget_sec, 90.0),
                    "local_swap": {
                        "iterations": 80,
                        "require_legal": True,
                        "use_proxy": LOCAL_REFINEMENT_USE_PROXY,
                    },
                },
            ),
        ):
            if source.placement is None or self.budget_exhausted():
                break
            family = BranchFamily(
                name=f"{operator}_refine",
                kind="chain",
                chain=operator,
                base_config_builder=lambda _benchmark, cfg=config: copy.deepcopy(cfg),
                probe_presets=(ParamPreset(operator.upper(), {}, variant_family=operator),),
            )
            result = self.evaluate_family(
                benchmark,
                family,
                family.probe_presets[0],
                plc,
                stage="final_polish",
                parent_candidate=source.label,
                placement=source.placement,
                local_operator=operator,
            )
            if result.proxy_cost is not None and source.proxy_cost is not None:
                result.delta_vs_parent = float(result.proxy_cost) - float(source.proxy_cost)
            best = self.select_best_result(best, result, baseline)
            if result.legal:
                source = result
        return best

    def restore_fixed_macros(self, placement: torch.Tensor, benchmark: Benchmark) -> torch.Tensor:
        restored = placement.clone()
        if benchmark.macro_fixed.any():
            restored[benchmark.macro_fixed] = benchmark.macro_positions[benchmark.macro_fixed]
        return restored

    def budget_exhausted(self) -> bool:
        return (time.perf_counter() - self._start_time) >= self.total_budget_sec

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def mark_selected(self, selected: CandidateResult) -> None:
        for result in self._results:
            result.selected = bool(result.label == selected.label)
            if result.selected:
                result.rejected_reason = None

    def log_record(self, result: CandidateResult) -> Dict[str, Any]:
        metadata = result.metadata or {}
        final = _last_stage(metadata)
        return {
            "benchmark": result.benchmark,
            "mode": result.mode,
            "stage": result.stage,
            "branch_family": result.branch_family,
            "candidate_name": result.candidate_name,
            "chain": result.chain,
            "config": _jsonable(result.config),
            "runtime_sec": result.runtime_sec,
            "proxy_cost": result.proxy_cost,
            "legal": result.legal,
            "overlap_count": result.overlap_count,
            "boundary_violations": result.boundary_violations,
            "selected": result.selected,
            "accepted": result.accepted,
            "parent_candidate": result.parent_candidate,
            "variant_family": result.variant_family,
            "delta_vs_parent": result.delta_vs_parent,
            "error": result.error,
            "rejected_reason": result.rejected_reason,
            "seed": result.seed,
            "label": result.label,
            "stage_metrics": _jsonable(metadata.get("stage_metrics") or []),
            "final_stage_proxy_cost": final.get("proxy_cost"),
            "final_stage_legal": final.get("legal"),
            "final_stage_overlap_count": final.get("overlap_count"),
            "chain_budget_stopped": metadata.get("budget_stopped"),
            "final_operator": metadata.get("final_operator"),
            "prefix_reuse_requested": result.prefix_reuse_requested,
            "prefix_reuse_used": result.prefix_reuse_used,
        }

    def flush_log(self) -> None:
        path_text = self.log_path
        if not path_text:
            path_text = str(
                _repo_root()
                / "runs"
                / "ml_param_tournament"
                / f"{int(time.time())}_{os.getpid()}.jsonl"
            )
        try:
            path = Path(path_text)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as f:
                for result in self._results:
                    f.write(json.dumps(_jsonable(self.log_record(result)), sort_keys=True) + "\n")
        except Exception:
            return

    def print_summary(self, benchmark: Benchmark, selected: CandidateResult) -> None:
        print(f"Benchmark {benchmark.name}, mode={self.mode}, hard_macros={benchmark.num_hard_macros}")
        for result in self.rank_results(self._results):
            proxy = result.proxy_cost
            proxy_text = f"{proxy:.4f}" if proxy is not None else "-"
            runtime = result.runtime_sec
            runtime_text = f"{runtime:.2f}s" if runtime is not None else "-"
            marker = "*" if result.label == selected.label else " "
            print(
                f"{marker} {result.stage:<15} {result.branch_family:<22} "
                f"{result.candidate_name:<38.38} proxy={proxy_text:<8} "
                f"legal={str(result.legal):<5} time={runtime_text}"
            )


if __name__ == "__main__":
    print(
        "Run this model through the repo evaluator, for example:\n"
        "  uv run evaluate submissions/models/ml_param_tournament.py -b ibm01"
    )
