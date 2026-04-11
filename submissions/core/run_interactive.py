"""Interactive runner for evaluating a placer with the Phase 0 API."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from submissions.core.runner import load_placer_from_file, run_placer
from submissions.core.types import RunConfig


def _prompt_text(prompt: str, default: Optional[str] = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    value = input(f"{prompt}{suffix}: ").strip()
    if value:
        return value
    if default is not None:
        return default
    raise ValueError(f"{prompt} is required")


def _prompt_bool(prompt: str, default: bool) -> bool:
    default_text = "y" if default else "n"
    while True:
        value = input(f"{prompt} [y/n, default={default_text}]: ").strip().lower()
        if not value:
            return default
        if value in {"y", "yes"}:
            return True
        if value in {"n", "no"}:
            return False
        print("Please enter y or n.")


def _prompt_int(prompt: str, default: int) -> int:
    while True:
        value = input(f"{prompt} [{default}]: ").strip()
        if not value:
            return default
        try:
            return int(value)
        except ValueError:
            print("Please enter an integer.")


def _prompt_benchmark_names() -> Optional[List[str]]:
    print(
        "Benchmark names: enter a comma-separated list like "
        "`ibm01,ibm02`, or press Enter to run every benchmark folder."
    )
    value = input("benchmark_names: ").strip()
    if not value:
        return None
    names = [name.strip() for name in value.split(",") if name.strip()]
    return names or None


def _print_summary(results) -> None:
    print()
    print("Run complete.")
    print(f"Benchmarks run: {len(results)}")
    if not results:
        return

    avg_proxy = sum(result.proxy_cost for result in results) / len(results)
    avg_runtime = sum(result.runtime_seconds for result in results) / len(results)
    total_overlaps = sum(result.overlap_count for result in results)
    num_valid = sum(1 for result in results if result.valid)

    print(f"Average proxy cost: {avg_proxy:.6f}")
    print(f"Average runtime: {avg_runtime:.4f}s")
    print(f"Valid placements: {num_valid}/{len(results)}")
    print(f"Total overlaps: {total_overlaps}")
    print()

    for result in results:
        print(
            f"{result.benchmark_name}: "
            f"proxy={result.proxy_cost:.6f}, "
            f"valid={result.valid}, "
            f"overlaps={result.overlap_count}, "
            f"runtime={result.runtime_seconds:.4f}s"
        )


def main() -> None:
    print("Phase 0 interactive runner")
    print("This will prompt for the placer file and all RunConfig options.")
    print()

    placer_path = Path(_prompt_text("Path to placer Python file"))
    benchmark_root = Path(
        _prompt_text(
            "Benchmark root",
            "external/MacroPlacement/Testcases/ICCAD04",
        )
    )
    benchmark_names = _prompt_benchmark_names()
    output_dir = Path(
        _prompt_text(
            "Output directory",
            "submissions/core/outputs/interactive_run",
        )
    )
    seed = _prompt_int("Seed", 42)
    check_overlaps = _prompt_bool("Check overlaps during validation", True)
    save_visualizations = _prompt_bool("Save visualization PNGs", True)
    save_placements = _prompt_bool("Save placement tensors", True)
    log_filename = _prompt_text("Results log filename", "results.json")
    config_filename = _prompt_text("Config filename", "config.json")

    config = RunConfig(
        benchmark_root=benchmark_root,
        benchmark_names=benchmark_names,
        output_dir=output_dir,
        seed=seed,
        check_overlaps=check_overlaps,
        save_visualizations=save_visualizations,
        save_placements=save_placements,
        log_filename=log_filename,
        config_filename=config_filename,
    )

    placer = load_placer_from_file(placer_path)
    results = run_placer(placer, config=config)
    _print_summary(results)

    print()
    print(f"Artifacts written to: {output_dir.resolve()}")
    print(f"Results log: {(output_dir / log_filename).resolve()}")
    print(f"Saved config: {(output_dir / config_filename).resolve()}")


if __name__ == "__main__":
    main()
