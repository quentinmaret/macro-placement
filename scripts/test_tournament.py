"""
Run the final tournament placer with candidate-level diagnostics enabled.

Example:
    uv run python scripts/test_tournament.py --benchmark ibm01 --output results/tournament_ibm01.jsonl --debug
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Optional

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/macro-placement-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/macro-placement-cache")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from scripts.test_initializers import load_requested_benchmark


def load_final_module() -> Any:
    path = REPO_ROOT / "submissions" / "final" / "placer.py"
    spec = importlib.util.spec_from_file_location("final_placer", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load final placer module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def winner_record(placer: Any) -> Optional[dict]:
    return next((record for record in placer._candidate_records if record.get("winner")), None)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run final placer tournament diagnostics.")
    parser.add_argument("--benchmark", "-b", default="ibm01", help="Benchmark name, path, or .pt file")
    parser.add_argument("--output", "-o", default=None, help="JSONL candidate diagnostics path")
    parser.add_argument("--debug", action="store_true", help="Print compact candidate summary")
    parser.add_argument("--append", action="store_true", help="Append to output instead of replacing it")
    args = parser.parse_args()

    benchmark, _plc = load_requested_benchmark(args.benchmark)
    output_path = Path(args.output) if args.output else None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists() and not args.append:
            output_path.unlink()

    final_module = load_final_module()
    placer = final_module.FinalMacroPlacer(debug=args.debug, log_path=str(output_path) if output_path else None)
    placement = placer.place(benchmark)
    if placement.shape != (benchmark.num_macros, 2):
        raise RuntimeError(f"Final placement has unexpected shape {tuple(placement.shape)}")

    winner = winner_record(placer)
    if winner is None:
        print("winner: unavailable")
    else:
        score = winner.get("score")
        score_text = f"{score:.4f}" if isinstance(score, (float, int)) else "-"
        print(f"winner: {winner['candidate']} score={score_text}")
    if output_path is not None:
        print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    torch.set_num_threads(1)
    raise SystemExit(main())
