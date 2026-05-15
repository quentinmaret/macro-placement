"""
Run initializer/operator chains on one benchmark and record comparable metrics.

Examples:
    uv run python test_initializers.py --benchmark ibm01
    uv run python test_initializers.py --benchmark ibm01 --seed 0
    uv run python test_initializers.py --benchmark ibm01 --initializers hierarchical,grid
    uv run python test_initializers.py --benchmark ibm01 --chains hierarchical,legalize
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/macro-placement-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/macro-placement-cache")

import torch

from macro_place.benchmark import Benchmark
from macro_place.loader import load_benchmark, load_benchmark_from_dir


DEFAULT_CHAINS = [
    "hierarchical",
    "grid,legalize",
    "random_spread,legalize",
    "peripheral,legalize",
    "hierarchical,pin_aware_shift,legalize",
    "hierarchical,spectral_order,legalize",
    "hierarchical,force_smooth,legalize",
    "hierarchical,macro_spread,legalize",
    "hierarchical,legalize,local_swap",
]

NG45_BENCHMARKS = {
    "ariane133": "external/MacroPlacement/Flows/NanGate45/ariane133/netlist/output_CT_Grouping",
    "ariane136": "external/MacroPlacement/Flows/NanGate45/ariane136/netlist/output_CT_Grouping",
    "mempool_tile": "external/MacroPlacement/Flows/NanGate45/mempool_tile/netlist/output_CT_Grouping",
    "nvdla": "external/MacroPlacement/Flows/NanGate45/nvdla/netlist/output_CT_Grouping",
}


def load_initializer_module():
    """Import submissions/models/initializer.py by path."""
    path = Path("submissions/models/initializer.py")
    spec = importlib.util.spec_from_file_location("initializer_model", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load initializer module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize_benchmark_name(name: str) -> str:
    """Allow ibm1 as a shorthand for ibm01."""
    lowered = name.lower()
    if lowered.startswith("ibm") and lowered[3:].isdigit():
        number = int(lowered[3:])
        return f"ibm{number:02d}"
    return name


def load_requested_benchmark(name_or_path: str) -> Tuple[Benchmark, Optional[Any]]:
    """Load a benchmark from an IBM/NG45 name, directory, or .pt file."""
    path = Path(name_or_path)
    if path.exists() and path.is_dir():
        return load_benchmark_from_dir(str(path))
    if path.exists() and path.suffix == ".pt":
        return Benchmark.load(str(path)), None

    name = normalize_benchmark_name(name_or_path)
    ibm_dir = Path("external/MacroPlacement/Testcases/ICCAD04") / name
    if ibm_dir.exists():
        return load_benchmark_from_dir(str(ibm_dir))

    ng45_dir = NG45_BENCHMARKS.get(name)
    if ng45_dir and Path(ng45_dir).exists():
        return load_benchmark(
            str(Path(ng45_dir) / "netlist.pb.txt"),
            str(Path(ng45_dir) / "initial.plc"),
            name=name,
        )

    processed_pt = Path("benchmarks/processed/public") / f"{name}.pt"
    if processed_pt.exists():
        return Benchmark.load(str(processed_pt)), None

    raise FileNotFoundError(
        f"Could not find benchmark `{name_or_path}` as a directory, IBM/NG45 name, or .pt file"
    )


def split_initializers(value: Optional[str]) -> List[str]:
    """Parse --initializers comma list into one-operator chains."""
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def summarize_final_metrics(result: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten the final stage metrics for table/CSV output."""
    stage_metrics = result["metadata"].get("stage_metrics", [])
    final_metrics = stage_metrics[-1] if stage_metrics else {}
    summary = {
        "benchmark": result["benchmark"],
        "chain": result["chain"],
        "seed": result["seed"],
        "runtime_sec": result["metadata"].get("runtime_sec"),
        "proxy_cost": final_metrics.get("proxy_cost"),
        "hpwl": final_metrics.get("hpwl"),
        "wirelength_cost": final_metrics.get("wirelength_cost"),
        "density_cost": final_metrics.get("density_cost"),
        "congestion_cost": final_metrics.get("congestion_cost"),
        "overlap_count": final_metrics.get("overlap_count"),
        "total_overlap_area": final_metrics.get("total_overlap_area"),
        "boundary_violations": final_metrics.get("boundary_violations"),
        "legal": final_metrics.get("legal"),
        "metrics_error": final_metrics.get("metrics_error"),
    }
    return summary


def format_metric(value: Any, width: int = 9, precision: int = 4) -> str:
    """Format nullable metric values for a compact table."""
    if value is None:
        return f"{'-':>{width}}"
    if isinstance(value, bool):
        return f"{str(value):>{width}}"
    if isinstance(value, (float, int)):
        return f"{float(value):>{width}.{precision}f}"
    return f"{str(value):>{width}}"


def print_summary_table(results: Sequence[Dict[str, Any]]) -> None:
    """Print one readable line per initializer/chain."""
    print()
    print("-" * 118)
    print(
        f"{'Chain':<44} {'Proxy':>9} {'WL':>9} {'Dens':>9} {'Cong':>9} "
        f"{'OvPairs':>8} {'Bound':>6} {'Legal':>7} {'Time':>8}"
    )
    print("-" * 118)
    for result in results:
        summary = summarize_final_metrics(result)
        print(
            f"{summary['chain']:<44.44} "
            f"{format_metric(summary['proxy_cost'])} "
            f"{format_metric(summary['wirelength_cost'])} "
            f"{format_metric(summary['density_cost'])} "
            f"{format_metric(summary['congestion_cost'])} "
            f"{format_metric(summary['overlap_count'], width=8, precision=0)} "
            f"{format_metric(summary['boundary_violations'], width=6, precision=0)} "
            f"{format_metric(summary['legal'], width=7)} "
            f"{format_metric(summary['runtime_sec'], width=8, precision=2)}"
        )
    print("-" * 118)
    print()


def write_results(path: Path, results: Sequence[Dict[str, Any]]) -> None:
    """Write JSONL by default, or CSV when the suffix is .csv."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        rows = []
        for result in results:
            row = summarize_final_metrics(result)
            row["stage_metrics_json"] = json.dumps(result["metadata"].get("stage_metrics", []))
            rows.append(row)
        fieldnames = list(rows[0].keys()) if rows else []
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return

    with path.open("w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")


def build_chains(args: argparse.Namespace) -> List[str]:
    """Build the requested chain list from --initializers/--chains/defaults."""
    chains: List[str] = []
    chains.extend(split_initializers(args.initializers))
    if args.chains:
        chains.extend(args.chains)
    if not chains:
        chains = list(DEFAULT_CHAINS)
    return chains


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run macro-placement initializer chains on one benchmark.",
    )
    parser.add_argument("--benchmark", "-b", default="ibm01", help="Benchmark name/path")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--initializers",
        default=None,
        help="Comma-separated standalone seed initializers to run",
    )
    parser.add_argument(
        "--chains",
        nargs="*",
        default=None,
        help="Manually specified initializer chains, e.g. hierarchical,legalize",
    )
    parser.add_argument("--output", "-o", default=None, help="Optional JSONL or CSV output path")
    args = parser.parse_args()

    initializer_module = load_initializer_module()
    benchmark, plc = load_requested_benchmark(args.benchmark)
    chains = build_chains(args)

    results: List[Dict[str, Any]] = []
    for chain in chains:
        print(f"running {benchmark.name}: {chain}", flush=True)
        try:
            _placement, metadata = initializer_module.run_initializer_chain(
                benchmark,
                chain,
                seed=args.seed,
                plc=plc,
            )
        except Exception as exc:
            print(f"  failed: {exc}", file=sys.stderr)
            metadata = {
                "chain": chain,
                "seed": args.seed,
                "runtime_sec": None,
                "stage_metrics": [],
                "error": str(exc),
            }
        results.append(
            {
                "benchmark": benchmark.name,
                "chain": chain,
                "seed": args.seed,
                "metadata": metadata,
            }
        )

    print_summary_table(results)
    if args.output:
        output_path = Path(args.output)
        write_results(output_path, results)
        print(f"wrote {output_path}")


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
