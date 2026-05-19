"""
Spacing ablation: compare baseline vs spacing-aware placement modes.

Modes
-----
baseline  – proxy-only scoring, no repair (MPC_SPACING_WEIGHT=0)
score     – spacing-aware candidate scoring (MPC_SPACING_WEIGHT=<w>)
repair    – spacing-aware scoring + final spacing repair

Examples
--------
# Quick smoke test on first available benchmark (no args)
python scripts/run_spacing_ablation.py

# Single benchmark, all three modes
python scripts/run_spacing_ablation.py --benchmark ibm01

# All IBM benchmarks (ICCAD04 submodule required)
python scripts/run_spacing_ablation.py --all

# Custom spacing weight and clearance
python scripts/run_spacing_ablation.py --spacing-weight 0.05 --clearance 12.0

# Choose which modes to run
python scripts/run_spacing_ablation.py --modes baseline,repair

# Save results
python scripts/run_spacing_ablation.py --output-csv /tmp/ablation.csv --output-jsonl /tmp/ablation.jsonl
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Load spacing helpers (no import-chain dependency on placer.py).
# Register in sys.modules before exec so @dataclass works on Python ≥ 3.12.
# ---------------------------------------------------------------------------
_sp_spec = importlib.util.spec_from_file_location(
    "_ablation_spacing",
    str(_REPO_ROOT / "submissions" / "final" / "spacing.py"),
)
_sp_mod = importlib.util.module_from_spec(_sp_spec)
sys.modules.setdefault("_ablation_spacing", _sp_mod)
_sp_spec.loader.exec_module(_sp_mod)
_spacing_violation_count = _sp_mod.hard_macro_spacing_violation_count
_spacing_min_clearance = _sp_mod.min_hard_macro_clearance

# ---------------------------------------------------------------------------
# Load FinalMacroPlacer (expensive – triggers WillSeed / Initializer imports)
# Load once so module-level symbols are shared across all mode runs.
# ---------------------------------------------------------------------------
_placer_spec = importlib.util.spec_from_file_location(
    "_ablation_placer",
    str(_REPO_ROOT / "submissions" / "final" / "placer.py"),
)
_placer_mod = importlib.util.module_from_spec(_placer_spec)
_placer_spec.loader.exec_module(_placer_mod)
FinalMacroPlacer = _placer_mod.FinalMacroPlacer


# ---------------------------------------------------------------------------
# Benchmark discovery helpers
# ---------------------------------------------------------------------------

def _ibm_benchmark_dirs() -> List[Path]:
    root = _REPO_ROOT / "external" / "MacroPlacement" / "Testcases" / "ICCAD04"
    if not root.exists():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir())


def _pt_benchmark_files() -> List[Path]:
    root = _REPO_ROOT / "benchmarks" / "processed" / "public"
    if not root.exists():
        return []
    # Prefer IBM .pt files; exclude NG45 unless no IBM files exist
    ibm = sorted(p for p in root.glob("ibm*.pt"))
    return ibm if ibm else sorted(root.glob("*.pt"))


def _load_from_dir(benchmark_dir: Path):
    from macro_place.loader import load_benchmark_from_dir
    return load_benchmark_from_dir(str(benchmark_dir))


def _load_from_pt(pt_path: Path):
    from macro_place.benchmark import Benchmark
    b = Benchmark.load(str(pt_path))
    return b, None  # no plc available from .pt files


# ---------------------------------------------------------------------------
# Per-mode runner
# ---------------------------------------------------------------------------

def _run_mode(
    benchmark,
    plc,
    mode: str,
    spacing_weight: float,
    clearance_um: float,
) -> Dict:
    """
    Run FinalMacroPlacer in the given mode and return a result dict.

    SpacingConfig is read from env in __init__, so env vars must be set
    before constructing the placer instance.
    """
    env_overrides = {
        "MPC_SPACING_CLEARANCE": str(clearance_um),
        "MPC_SPACING_WEIGHT": "0.0" if mode == "baseline" else str(spacing_weight),
        "MPC_SPACING_REPAIR": "1" if mode == "repair" else "0",
        "MPC_FINAL_LOG": "0",  # suppress file logging during ablation
    }
    saved = {k: os.environ.get(k) for k in env_overrides}
    os.environ.update(env_overrides)
    try:
        placer = FinalMacroPlacer()
        t0 = time.perf_counter()
        placement = placer.place(benchmark)
        runtime = time.perf_counter() - t0
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    viol = _spacing_violation_count(placement, benchmark, clearance_um)
    min_clr = _spacing_min_clearance(placement, benchmark)

    proxy: Optional[float] = None
    overlap = 0
    if plc is not None:
        from macro_place.objective import compute_proxy_cost
        costs = compute_proxy_cost(placement, benchmark, plc)
        proxy = float(costs["proxy_cost"])
        overlap = int(costs["overlap_count"])

    winner = next(
        (r for r in placer._candidate_records if r.get("winner")), None
    )
    selected = winner.get("candidate") if winner else None
    repair_info = winner.get("spacing_repair") if winner else None

    return {
        "design": benchmark.name,
        "mode": mode,
        "proxy_score": proxy,
        "overlap_count": overlap,
        "spacing_violation_count": int(viol),
        "min_clearance_um": None if min_clr == float("inf") else round(float(min_clr), 4),
        "runtime_sec": round(runtime, 2),
        "selected_candidate": selected,
        "repair_info": repair_info,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare baseline vs spacing-aware placement modes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--benchmark", "-b",
        metavar="NAME",
        help="Single IBM benchmark name, e.g. ibm01 (requires ICCAD04 submodule)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all available benchmarks (slow)",
    )
    parser.add_argument(
        "--modes", default="baseline,score,repair",
        help="Comma-separated modes to run (default: baseline,score,repair)",
    )
    parser.add_argument(
        "--spacing-weight", type=float, default=0.1,
        help="lambda_spacing for score/repair modes (default: 0.1)",
    )
    parser.add_argument(
        "--clearance", type=float, default=12.0,
        help="Clearance target in μm (default: 12.0)",
    )
    parser.add_argument(
        "--output-csv", metavar="PATH",
        help="Write CSV summary to this path (default: print only)",
    )
    parser.add_argument(
        "--output-jsonl", metavar="PATH",
        help="Write JSONL summary to this path",
    )
    args = parser.parse_args()

    modes = [m.strip() for m in args.modes.split(",")]
    unknown = set(modes) - {"baseline", "score", "repair"}
    if unknown:
        parser.error(f"Unknown modes: {unknown}. Valid: baseline, score, repair")

    # ------------------------------------------------------------------
    # Discover benchmarks
    # ------------------------------------------------------------------
    benchmark_specs: List[Dict] = []  # each: {source, path_or_name, label}

    if args.benchmark:
        # Explicit single benchmark – require ICCAD04
        bd = _REPO_ROOT / "external" / "MacroPlacement" / "Testcases" / "ICCAD04" / args.benchmark
        if not bd.exists():
            print(
                f"ERROR: benchmark directory not found: {bd}\n"
                "Ensure the ICCAD04 submodule is initialised:\n"
                "  git submodule update --init",
                file=sys.stderr,
            )
            sys.exit(1)
        benchmark_specs = [{"source": "dir", "path": bd}]

    else:
        ibm_dirs = _ibm_benchmark_dirs()
        if ibm_dirs:
            if args.all:
                benchmark_specs = [{"source": "dir", "path": p} for p in ibm_dirs]
            else:
                benchmark_specs = [{"source": "dir", "path": ibm_dirs[0]}]
                print(
                    f"Using first IBM benchmark ({ibm_dirs[0].name})."
                    " Pass --all to run all, or --benchmark <name> to pick one.",
                    file=sys.stderr,
                )
        else:
            # Fall back to .pt files (no proxy score available)
            pt_files = _pt_benchmark_files()
            if not pt_files:
                print(
                    "ERROR: no benchmarks found.\n"
                    "Initialise the ICCAD04 submodule:\n"
                    "  git submodule update --init\n"
                    "or place .pt files in benchmarks/processed/public/.",
                    file=sys.stderr,
                )
                sys.exit(1)
            target = pt_files if args.all else [pt_files[0]]
            benchmark_specs = [{"source": "pt", "path": p} for p in target]
            print(
                f"ICCAD04 submodule not initialised – loading from .pt files.\n"
                "Proxy scores will not be available (requires full netlist).\n"
                "To enable proxy scores, run: git submodule update --init",
                file=sys.stderr,
            )

    # ------------------------------------------------------------------
    # Run ablation
    # ------------------------------------------------------------------
    all_results: List[Dict] = []

    for spec in benchmark_specs:
        path = spec["path"]
        label = path.name if spec["source"] == "dir" else path.stem
        print(f"\n=== {label} ===")
        try:
            if spec["source"] == "dir":
                benchmark, plc = _load_from_dir(path)
            else:
                benchmark, plc = _load_from_pt(path)
        except Exception as exc:
            print(f"  load FAILED: {exc}", file=sys.stderr)
            continue

        canvas_max = max(float(benchmark.canvas_width), float(benchmark.canvas_height))
        # SpacingConfig default threshold is 200 μm; repair/scoring active only above it
        _sp_thresh = _sp_mod.SMALL_CANVAS_THRESHOLD_DEFAULT
        print(
            f"  macros={benchmark.num_hard_macros}  "
            f"canvas={benchmark.canvas_width:.0f}×{benchmark.canvas_height:.0f} μm  "
            f"large_canvas(>={_sp_thresh:.0f}μm)={canvas_max >= _sp_thresh}"
        )

        for mode in modes:
            print(f"  [{mode}] ... ", end="", flush=True)
            try:
                result = _run_mode(
                    benchmark, plc, mode, args.spacing_weight, args.clearance
                )
                all_results.append(result)
                proxy = result["proxy_score"]
                viol = result["spacing_violation_count"]
                min_clr = result["min_clearance_um"]
                rt = result["runtime_sec"]
                proxy_str = f"{proxy:.4f}" if proxy is not None else "N/A"
                min_clr_str = f"{min_clr:.2f}" if min_clr is not None else "inf"
                print(
                    f"proxy={proxy_str}  violations={viol}"
                    f"  min_clearance={min_clr_str} μm"
                    f"  runtime={rt:.1f}s"
                    f"  winner={result['selected_candidate']}"
                )
            except Exception as exc:
                import traceback
                print(f"FAILED: {exc}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                all_results.append({"design": label, "mode": mode, "error": str(exc)})

    if not all_results:
        print("No results collected.", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Output summary table
    # ------------------------------------------------------------------
    _print_table(all_results)

    # ------------------------------------------------------------------
    # Write CSV
    # ------------------------------------------------------------------
    csv_fields = [
        "design", "mode", "proxy_score", "overlap_count",
        "spacing_violation_count", "min_clearance_um",
        "runtime_sec", "selected_candidate",
    ]
    csv_path = args.output_csv
    if csv_path:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(all_results)
        print(f"\nCSV → {csv_path}")

    if args.output_jsonl:
        with open(args.output_jsonl, "w") as f:
            for r in all_results:
                # Exclude repair_info blob from JSONL for readability
                row = {k: v for k, v in r.items() if k != "repair_info"}
                f.write(json.dumps(row) + "\n")
        print(f"JSONL → {args.output_jsonl}")


def _print_table(results: List[Dict]) -> None:
    header = (
        f"{'design':<15} {'mode':<10} {'proxy':>9} {'overlaps':>8}"
        f" {'violations':>10} {'min_clr':>9} {'time':>6}"
    )
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for r in results:
        if "error" in r:
            print(f"{r.get('design','?'):<15} {r.get('mode','?'):<10}  ERROR: {r['error']}")
            continue
        proxy = r.get("proxy_score")
        proxy_s = f"{proxy:.4f}" if proxy is not None else "    N/A"
        min_clr = r.get("min_clearance_um")
        min_clr_s = f"{min_clr:.2f}" if min_clr is not None else "    inf"
        print(
            f"{r.get('design',''):<15} {r.get('mode',''):<10}"
            f" {proxy_s:>9} {r.get('overlap_count', 0):>8}"
            f" {r.get('spacing_violation_count', 0):>10} {min_clr_s:>9}"
            f" {r.get('runtime_sec', 0):>5.1f}s"
        )
    print(sep)


if __name__ == "__main__":
    main()
