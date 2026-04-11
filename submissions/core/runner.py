"""Experiment runner for benchmark sweeps, logging, and artifact output."""

from __future__ import annotations

from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import time
from typing import List, Optional

import torch

from macro_place.loader import load_benchmark_from_dir

from submissions.core.eval import evaluate, visualize
from submissions.core.types import (
    BenchmarkSpec,
    EvaluationArtifacts,
    EvaluationResult,
    Placer,
    RunConfig,
    set_seed,
)


def load_placer_from_file(path: str | Path) -> Placer:
    """Load the first class in a Python file that exposes a `place` method."""
    placer_path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(placer_path.stem, str(placer_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load placer from {placer_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for attr in vars(module).values():
        if (
            isinstance(attr, type)
            and attr.__module__ == placer_path.stem
            and callable(getattr(attr, "place", None))
        ):
            return attr()

    raise RuntimeError(
        f"No placer class found in {placer_path}. "
        "Expected a class with a place(self, benchmark) method."
    )


class ExperimentRunner:
    """Run a placer across benchmarks and save results in one place."""

    def __init__(self, config: Optional[RunConfig] = None):
        self.config = config or RunConfig()

    def discover_benchmarks(self) -> List[BenchmarkSpec]:
        """Resolve benchmark names into on-disk benchmark directories."""
        if not self.config.benchmark_root.exists():
            raise FileNotFoundError(
                f"Benchmark root not found: {self.config.benchmark_root}"
            )

        if self.config.benchmark_names is None:
            benchmark_dirs = sorted(
                path for path in self.config.benchmark_root.iterdir() if path.is_dir()
            )
        else:
            benchmark_dirs = [self.config.benchmark_root / name for name in self.config.benchmark_names]

        specs = []
        for benchmark_dir in benchmark_dirs:
            if not benchmark_dir.exists():
                raise FileNotFoundError(f"Benchmark directory not found: {benchmark_dir}")
            specs.append(BenchmarkSpec(name=benchmark_dir.name, benchmark_dir=benchmark_dir))
        return specs

    def run(self, placer: Placer) -> List[EvaluationResult]:
        """Run a placer across all configured benchmarks."""
        set_seed(self.config.seed)
        output_dir = self._prepare_output_dir()
        self._write_config(output_dir)

        results = []
        for spec in self.discover_benchmarks():
            results.append(self._run_single(placer, spec, output_dir))

        self._write_results(output_dir, results)
        return results

    def _run_single(
        self,
        placer: Placer,
        spec: BenchmarkSpec,
        output_dir: Path,
    ) -> EvaluationResult:
        benchmark, plc = load_benchmark_from_dir(str(spec.benchmark_dir))

        start = time.perf_counter()
        placement = placer.place(benchmark)
        runtime_seconds = time.perf_counter() - start

        summary = evaluate(
            placement,
            benchmark,
            plc,
            check_overlaps=self.config.check_overlaps,
        )

        artifacts = self._save_artifacts(
            output_dir=output_dir,
            benchmark_name=spec.name,
            placement=summary["placement"],
            benchmark=benchmark,
            plc=plc,
        )

        return EvaluationResult(
            benchmark_name=spec.name,
            proxy_cost=float(summary["proxy_cost"]),
            wirelength_cost=float(summary["wirelength_cost"]),
            density_cost=float(summary["density_cost"]),
            congestion_cost=float(summary["congestion_cost"]),
            overlap_count=int(summary["overlap_count"]),
            total_overlap_area=float(summary["total_overlap_area"]),
            max_overlap_area=float(summary["max_overlap_area"]),
            num_macros_with_overlaps=int(summary["num_macros_with_overlaps"]),
            overlap_ratio=float(summary["overlap_ratio"]),
            valid=bool(summary["valid"]),
            violations=list(summary["violations"]),
            runtime_seconds=runtime_seconds,
            seed=self.config.seed,
            artifacts=artifacts,
        )

    def _prepare_output_dir(self) -> Path:
        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "placements").mkdir(exist_ok=True)
        (output_dir / "visualizations").mkdir(exist_ok=True)
        return output_dir

    def _save_artifacts(
        self,
        *,
        output_dir: Path,
        benchmark_name: str,
        placement: torch.Tensor,
        benchmark,
        plc,
    ) -> EvaluationArtifacts:
        artifacts = EvaluationArtifacts()

        if self.config.save_placements:
            placement_path = output_dir / "placements" / f"{benchmark_name}.pt"
            torch.save(placement.cpu(), placement_path)
            artifacts.placement_path = placement_path

        if self.config.save_visualizations:
            visualization_path = output_dir / "visualizations" / f"{benchmark_name}.png"
            visualize(
                placement,
                benchmark,
                save_path=visualization_path,
                plc=plc,
            )
            artifacts.visualization_path = visualization_path

        return artifacts

    def _write_config(self, output_dir: Path) -> None:
        config_path = output_dir / self.config.config_filename
        payload = asdict(self.config)
        payload["benchmark_root"] = str(self.config.benchmark_root)
        payload["output_dir"] = str(self.config.output_dir)
        with config_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

    def _write_results(self, output_dir: Path, results: List[EvaluationResult]) -> None:
        results_path = output_dir / self.config.log_filename
        payload = {
            "summary": self._summarize(results),
            "results": [result.to_dict() for result in results],
        }
        with results_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

    @staticmethod
    def _summarize(results: List[EvaluationResult]) -> dict:
        if not results:
            return {
                "num_benchmarks": 0,
                "avg_proxy_cost": None,
                "avg_runtime_seconds": None,
                "num_valid": 0,
                "total_overlaps": 0,
            }

        return {
            "num_benchmarks": len(results),
            "avg_proxy_cost": sum(result.proxy_cost for result in results) / len(results),
            "avg_runtime_seconds": (
                sum(result.runtime_seconds for result in results) / len(results)
            ),
            "num_valid": sum(1 for result in results if result.valid),
            "total_overlaps": sum(result.overlap_count for result in results),
        }


def run_placer(placer: Placer, config: Optional[RunConfig] = None) -> List[EvaluationResult]:
    """Convenience entry point for running a placer across benchmarks."""
    return ExperimentRunner(config=config).run(placer)
