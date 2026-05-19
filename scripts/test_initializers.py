"""
Run small initializer-chain experiments on one benchmark.

Examples:
    uv run python scripts/test_initializers.py --benchmark ibm01
    uv run python scripts/test_initializers.py --benchmark ibm01 --chains "hierarchical,legalize"
    uv run python scripts/test_initializers.py --benchmark ibm01 --default-suite
    uv run python scripts/test_initializers.py --benchmark ibm01 --output results/initializer_ibm01.jsonl
    uv run python scripts/test_initializers.py --benchmark ibm01 --debug
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/macro-placement-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/macro-placement-cache")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from macro_place.benchmark import Benchmark
from macro_place.loader import load_benchmark, load_benchmark_from_dir


DEFAULT_SUITE = [
    "hierarchical,legalize",
    "hierarchical,macro_spread,legalize",
    "grid,legalize",
    "peripheral,legalize",
]

NG45_BENCHMARKS = {
    "ariane133": "external/MacroPlacement/Flows/NanGate45/ariane133/netlist/output_CT_Grouping",
    "ariane133_ng45": "external/MacroPlacement/Flows/NanGate45/ariane133/netlist/output_CT_Grouping",
    "ariane136": "external/MacroPlacement/Flows/NanGate45/ariane136/netlist/output_CT_Grouping",
    "ariane136_ng45": "external/MacroPlacement/Flows/NanGate45/ariane136/netlist/output_CT_Grouping",
    "mempool_tile": "external/MacroPlacement/Flows/NanGate45/mempool_tile/netlist/output_CT_Grouping",
    "mempool_tile_ng45": "external/MacroPlacement/Flows/NanGate45/mempool_tile/netlist/output_CT_Grouping",
    "nvdla": "external/MacroPlacement/Flows/NanGate45/nvdla/netlist/output_CT_Grouping",
    "nvdla_ng45": "external/MacroPlacement/Flows/NanGate45/nvdla/netlist/output_CT_Grouping",
}


def load_initializer_module():
    path = REPO_ROOT / "submissions" / "models" / "initializer.py"
    spec = importlib.util.spec_from_file_location("initializer_model", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load initializer module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize_benchmark_name(name: str) -> str:
    lowered = name.lower()
    if lowered.startswith("ibm") and lowered[3:].isdigit():
        return f"ibm{int(lowered[3:]):02d}"
    return name


def load_requested_benchmark(name_or_path: str) -> Tuple[Benchmark, Optional[Any]]:
    path = Path(name_or_path)
    if path.exists() and path.is_dir():
        return load_benchmark_from_dir(str(path))
    if path.exists() and path.suffix == ".pt":
        return Benchmark.load(str(path)), None

    name = normalize_benchmark_name(name_or_path)
    ibm_dir = REPO_ROOT / "external" / "MacroPlacement" / "Testcases" / "ICCAD04" / name
    if ibm_dir.exists():
        return load_benchmark_from_dir(str(ibm_dir))

    ng45_dir = NG45_BENCHMARKS.get(name)
    if ng45_dir and (REPO_ROOT / ng45_dir).exists():
        base = REPO_ROOT / ng45_dir
        return load_benchmark(str(base / "netlist.pb.txt"), str(base / "initial.plc"), name=name)

    processed_pt = REPO_ROOT / "benchmarks" / "processed" / "public" / f"{name}.pt"
    if processed_pt.exists():
        return Benchmark.load(str(processed_pt)), None

    raise FileNotFoundError(f"Could not find benchmark `{name_or_path}`")


def default_suite_for_macro_count(num_hard_macros: int, force_full_suite: bool) -> List[str]:
    if force_full_suite or num_hard_macros <= 280:
        return list(DEFAULT_SUITE)
    if num_hard_macros <= 320:
        return ["hierarchical,legalize", "grid,legalize"]
    return ["hierarchical,legalize"]


def final_stage(metadata: Dict[str, Any]) -> Dict[str, Any]:
    stages = metadata.get("stage_metrics", [])
    return stages[-1] if stages else {}


def result_row(result: Dict[str, Any]) -> Dict[str, Any]:
    metadata = result.get("metadata", {})
    final = final_stage(metadata)
    return {
        "chain": result["chain"],
        "legal": final.get("legal"),
        "proxy_cost": final.get("proxy_cost"),
        "overlap_count": final.get("overlap_count"),
        "runtime_sec": metadata.get("runtime_sec"),
        "final_operator": metadata.get("final_operator"),
        "budget_stopped": metadata.get("budget_stopped"),
        "error": result.get("error") or metadata.get("error"),
    }


def format_value(value: Any, width: int, precision: int = 4) -> str:
    if value is None:
        return f"{'-':>{width}}"
    if isinstance(value, bool):
        return f"{str(value):>{width}}"
    if isinstance(value, (float, int)):
        return f"{float(value):>{width}.{precision}f}"
    return f"{str(value):>{width}}"


def print_table(results: Sequence[Dict[str, Any]]) -> None:
    print()
    print(f"{'chain':<42} {'legal':>7} {'proxy':>10} {'overlap':>8} {'time':>8} {'final':>14} {'stop':>6}")
    print("-" * 102)
    for result in results:
        row = result_row(result)
        print(
            f"{row['chain']:<42.42} "
            f"{format_value(row['legal'], 7)} "
            f"{format_value(row['proxy_cost'], 10)} "
            f"{format_value(row['overlap_count'], 8, 0)} "
            f"{format_value(row['runtime_sec'], 8, 2)} "
            f"{format_value(row['final_operator'], 14)} "
            f"{format_value(row['budget_stopped'], 6)}"
        )
        if row["error"]:
            print(f"  error: {row['error']}")
    print()


def write_jsonl(path: Path, results: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for result in results:
            f.write(json.dumps(result, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run initializer chains on one benchmark.")
    parser.add_argument("--benchmark", "-b", default="ibm01", help="Benchmark name, path, or .pt file")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic seed")
    parser.add_argument("--chains", nargs="*", default=None, help="Explicit chains to run")
    parser.add_argument("--default-suite", action="store_true", help="Run the small default suite")
    parser.add_argument("--force-full-suite", action="store_true", help="Do not shrink defaults for large benchmarks")
    parser.add_argument("--output", "-o", default=None, help="Optional JSONL output path")
    parser.add_argument("--debug", action="store_true", help="Print per-stage details")
    args = parser.parse_args()

    initializer_module = load_initializer_module()
    benchmark, plc = load_requested_benchmark(args.benchmark)
    chains = args.chains or default_suite_for_macro_count(
        benchmark.num_hard_macros,
        force_full_suite=args.force_full_suite,
    )

    results: List[Dict[str, Any]] = []
    for chain in chains:
        if args.debug:
            print(f"running {benchmark.name}: {chain}", flush=True)
        try:
            _placement, metadata = initializer_module.run_initializer_chain(
                benchmark,
                chain,
                seed=args.seed,
                plc=plc,
                collect_metrics=True,
            )
            result: Dict[str, Any] = {
                "benchmark": benchmark.name,
                "num_hard_macros": benchmark.num_hard_macros,
                "chain": chain,
                "seed": args.seed,
                "metadata": metadata,
            }
        except Exception as exc:
            result = {
                "benchmark": benchmark.name,
                "num_hard_macros": benchmark.num_hard_macros,
                "chain": chain,
                "seed": args.seed,
                "metadata": {"stage_metrics": [], "runtime_sec": None, "final_operator": None},
                "error": str(exc),
            }
        if args.debug:
            for stage in result.get("metadata", {}).get("stage_metrics", []):
                print(
                    f"  {stage.get('operator'):<18} "
                    f"time={stage.get('runtime_sec', 0.0):.3f}s "
                    f"legal={stage.get('legal')} overlap={stage.get('overlap_count')}"
                )
        results.append(result)

    print(f"Benchmark {benchmark.name}, hard_macros={benchmark.num_hard_macros}")
    print_table(results)
    if args.output:
        output_path = Path(args.output)
        write_jsonl(output_path, results)
        print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    torch.set_num_threads(1)
    raise SystemExit(main())
