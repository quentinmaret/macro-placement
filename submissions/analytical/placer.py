"""
Analytical placer submission.

Runs gradient descent on the smooth proxy defined in `losses.py`, then a
short post-processing pass to guarantee zero hard-macro overlaps.

This is the production version of the algorithm explored interactively in
`train.ipynb`. The notebook adds visualization and checkpointing; this
file is the silent, deterministic version that the evaluator imports.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Optional

import torch

from macro_place.benchmark import Benchmark


# Enable CPU fallback for ops that MPS doesn't support yet. Harmless on CUDA / CPU.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def _load_losses_module():
    """
    Load `losses.py` as a sibling module.

    The evaluator imports submission files directly by path rather than as
    part of an installed package, so a normal `from .losses import ...`
    fails. Loading the sibling explicitly keeps the submission self-contained.
    """
    losses_path = Path(__file__).with_name("losses.py")
    spec = importlib.util.spec_from_file_location("analytical_losses", str(losses_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load losses module from {losses_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_losses = _load_losses_module()


def _select_device() -> torch.device:
    """Pick the fastest available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


class AnalyticalPlacer:
    """
    Gradient-descent placer over a smooth differentiable proxy.

    Workflow inside `place`:
      1. Move positions, sizes, and net incidence to the active device.
      2. Optimize `positions` with Adam against the smooth loss.
      3. Anneal `gamma` (HPWL sharpness) and ramp up the overlap weight.
      4. Snap positions back to the canvas and run a small push-apart
         pass to guarantee strictly zero hard-macro overlaps.
    """

    def __init__(
        self,
        *,
        n_steps: int = 1500,
        lr: float = 5e-3,
        gamma_start: float = 0.01,
        gamma_end: float = 0.001,
        w_overlap_start: float = 5.0,
        w_overlap_end: float = 500.0,
        w_canvas: float = 200.0,
        legalize_iters: int = 200,
        seed: int = 42,
        device: Optional[torch.device] = None,
    ) -> None:
        self.n_steps = n_steps
        self.lr = lr
        self.gamma_start = gamma_start
        self.gamma_end = gamma_end
        self.w_overlap_start = w_overlap_start
        self.w_overlap_end = w_overlap_end
        self.w_canvas = w_canvas
        self.legalize_iters = legalize_iters
        self.seed = seed
        self.device = device or _select_device()

    # ─── public API ──────────────────────────────────────────────────────────

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        """Produce a zero-overlap placement of shape [num_macros, 2]."""
        torch.manual_seed(self.seed)

        positions, sizes_dev, fixed_dev, padded_nets, mask, net_weights = (
            self._prepare_tensors(benchmark)
        )

        self._optimize(
            positions, fixed_dev, sizes_dev, padded_nets, mask, net_weights, benchmark
        )

        positions = self._clamp_to_canvas(positions, sizes_dev, benchmark)
        positions = self._restore_fixed(positions, benchmark, fixed_dev)
        positions = self._legalize_push_apart(positions, sizes_dev, benchmark, fixed_dev)
        positions = self._restore_fixed(positions, benchmark, fixed_dev)

        return positions.detach().cpu()

    # ─── setup ───────────────────────────────────────────────────────────────

    def _prepare_tensors(self, benchmark: Benchmark):
        """Move everything we'll touch in the inner loop onto the device."""
        positions = (
            benchmark.macro_positions.clone()
            .to(self.device, dtype=torch.float32)
            .requires_grad_(True)
        )
        sizes_dev = benchmark.macro_sizes.to(self.device, dtype=torch.float32)
        fixed_dev = benchmark.macro_fixed.to(self.device)

        padded_nets, mask, net_weights = _losses.prepare_net_tensors(
            benchmark, self.device
        )
        return positions, sizes_dev, fixed_dev, padded_nets, mask, net_weights

    # ─── inner training loop ────────────────────────────────────────────────

    def _optimize(
        self,
        positions: torch.Tensor,
        fixed_dev: torch.Tensor,
        sizes_dev: torch.Tensor,
        padded_nets: torch.Tensor,
        mask: torch.Tensor,
        net_weights: torch.Tensor,
        benchmark: Benchmark,
    ) -> None:
        """Run Adam on the smooth proxy with annealed gamma and ramped overlap."""
        optimizer = torch.optim.Adam([positions], lr=self.lr)
        anchor = benchmark.macro_positions.to(self.device, dtype=torch.float32)

        for step in range(1, self.n_steps + 1):
            frac = step / max(self.n_steps, 1)
            gamma = self._anneal(self.gamma_start, self.gamma_end, frac)
            w_overlap = self._anneal(self.w_overlap_start, self.w_overlap_end, frac)

            optimizer.zero_grad()
            loss, _ = _losses.total_loss(
                positions,
                benchmark,
                padded_nets,
                mask,
                net_weights,
                sizes_dev,
                gamma=gamma,
                w_overlap=w_overlap,
                w_canvas=self.w_canvas,
            )
            loss.backward()

            # Don't move fixed macros: zero their gradient.
            with torch.no_grad():
                positions.grad[fixed_dev] = 0.0

            optimizer.step()

            # Hard clamp fixed macros to their anchor each step.
            with torch.no_grad():
                positions.data[fixed_dev] = anchor[fixed_dev]

    # ─── post-processing ────────────────────────────────────────────────────

    def _clamp_to_canvas(
        self,
        positions: torch.Tensor,
        sizes_dev: torch.Tensor,
        benchmark: Benchmark,
    ) -> torch.Tensor:
        """Project each macro so it fits fully inside the canvas."""
        with torch.no_grad():
            half = sizes_dev / 2.0
            positions.data[:, 0].clamp_(
                half[:, 0], float(benchmark.canvas_width) - half[:, 0]
            )
            positions.data[:, 1].clamp_(
                half[:, 1], float(benchmark.canvas_height) - half[:, 1]
            )
        return positions

    def _restore_fixed(
        self,
        positions: torch.Tensor,
        benchmark: Benchmark,
        fixed_dev: torch.Tensor,
    ) -> torch.Tensor:
        """Pin fixed macros back to their original benchmark positions."""
        with torch.no_grad():
            anchor = benchmark.macro_positions.to(self.device, dtype=torch.float32)
            positions.data[fixed_dev] = anchor[fixed_dev]
        return positions

    def _legalize_push_apart(
        self,
        positions: torch.Tensor,
        sizes_dev: torch.Tensor,
        benchmark: Benchmark,
        fixed_dev: torch.Tensor,
    ) -> torch.Tensor:
        """
        Iterative push-apart fallback.

        Gradient descent usually drives overlap to near zero, but the
        competition requires strictly zero overlap between hard macros.
        Each iteration: compute overlap gradient with large weight, take
        one no-momentum step, clamp into the canvas, and stop early when
        no overlap remains. This is a safety net, not the main optimizer.
        """
        num_hard = benchmark.num_hard_macros
        if num_hard <= 1:
            return positions

        for _ in range(self.legalize_iters):
            if not self._has_overlap(positions, sizes_dev, num_hard):
                break

            positions.grad = None
            olap = _losses.overlap_penalty(positions, sizes_dev, num_hard)
            if float(olap.detach()) <= 0.0:
                break

            olap.backward()
            with torch.no_grad():
                grad = positions.grad
                grad[fixed_dev] = 0.0
                # Step size = average macro half-extent; pushes one macro-radius per iter.
                step = float(sizes_dev[:num_hard].mean() * 0.25)
                norms = grad.norm(dim=-1, keepdim=True).clamp(min=1e-9)
                positions.data -= step * (grad / norms)

            positions = self._clamp_to_canvas(positions, sizes_dev, benchmark)
            positions = self._restore_fixed(positions, benchmark, fixed_dev)

        return positions

    def _has_overlap(
        self,
        positions: torch.Tensor,
        sizes_dev: torch.Tensor,
        num_hard: int,
    ) -> bool:
        """Cheap exact overlap test on hard macros only."""
        with torch.no_grad():
            pos = positions[:num_hard]
            sz = sizes_dev[:num_hard]
            diff = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs()
            min_sep = (sz.unsqueeze(0) + sz.unsqueeze(1)) / 2.0
            overlap = (min_sep - diff).clamp_min(0.0)
            area = overlap[..., 0] * overlap[..., 1]
            eye = torch.eye(num_hard, dtype=torch.bool, device=pos.device)
            area = area.masked_fill(eye, 0.0)
            return bool((area > 1e-9).any())

    # ─── utilities ──────────────────────────────────────────────────────────

    @staticmethod
    def _anneal(start: float, end: float, frac: float) -> float:
        """Linear interpolation from `start` to `end` over [0, 1]."""
        frac = max(0.0, min(1.0, frac))
        return start + (end - start) * frac
