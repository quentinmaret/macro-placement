"""
Smooth differentiable approximations of the proxy cost.

The TILOS proxy used for evaluation is non-differentiable, so we cannot
backprop through it. Instead we minimize smooth surrogates that are well
correlated with each proxy term:

- Wirelength  -> log-sum-exp smooth HPWL
- Overlap     -> differentiable ReLU pairwise overlap area
- Boundaries  -> hinge penalty for going outside the canvas

These functions are shared between `placer.py` (production submission) and
`train.ipynb` (interactive training notebook).
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

from macro_place.benchmark import Benchmark


def prepare_net_tensors(
    benchmark: Benchmark,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert the benchmark's variable-length nets into padded tensors.

    Returns:
        padded_nets: [num_nets, max_net_size] long tensor of node indices,
            zero-padded for nets shorter than the maximum.
        mask: [num_nets, max_net_size] bool tensor, True for real entries.
        net_weights: [num_nets] float tensor (moved to device).

    The padding lets us compute smooth HPWL for all nets in a single
    batched logsumexp call instead of a Python loop.
    """
    net_nodes = benchmark.net_nodes
    num_nets = len(net_nodes)
    max_size = max((len(n) for n in net_nodes), default=1)

    padded = torch.zeros(num_nets, max_size, dtype=torch.long)
    mask = torch.zeros(num_nets, max_size, dtype=torch.bool)

    for i, nodes in enumerate(net_nodes):
        k = len(nodes)
        if k > 0:
            padded[i, :k] = nodes
            mask[i, :k] = True

    return (
        padded.to(device),
        mask.to(device),
        benchmark.net_weights.to(device),
    )


def smooth_hpwl(
    positions: torch.Tensor,
    padded_nets: torch.Tensor,
    mask: torch.Tensor,
    net_weights: torch.Tensor,
    gamma: float = 0.005,
) -> torch.Tensor:
    """
    Batched log-sum-exp smooth half-perimeter wirelength.

    For each net, the exact HPWL is `(max_x - min_x) + (max_y - min_y)`.
    We approximate max and min with the smooth operators:

        smooth_max(x) =  gamma * logsumexp( x / gamma)
        smooth_min(x) = -gamma * logsumexp(-x / gamma)

    Smaller gamma is a sharper approximation but has noisier gradients.
    Typical schedule: anneal gamma from ~0.01 down to ~0.001 during training.

    Padded (masked-out) entries are set to -inf / +inf so they contribute
    zero weight inside the logsumexp.
    """
    # pts: [num_nets, max_size, 2]
    pts = positions[padded_nets]

    # Mask has shape [num_nets, max_size]; expand to broadcast over (x, y).
    mask_xy = mask.unsqueeze(-1)

    pts_for_max = pts.masked_fill(~mask_xy, float("-inf"))
    pts_for_min = pts.masked_fill(~mask_xy, float("inf"))

    smooth_max = gamma * torch.logsumexp(pts_for_max / gamma, dim=1)        # [N, 2]
    smooth_min = -gamma * torch.logsumexp(-pts_for_min / gamma, dim=1)      # [N, 2]

    span_xy = smooth_max - smooth_min                                       # [N, 2]
    span = span_xy.sum(dim=-1)                                              # [N]

    return (net_weights * span).sum()


def overlap_penalty(
    positions: torch.Tensor,
    sizes: torch.Tensor,
    num_hard: int,
) -> torch.Tensor:
    """
    Differentiable pairwise overlap area for hard macros only.

    For every pair (i, j) we compute the axis-aligned overlap rectangle and
    sum its area. ReLU makes the function piecewise differentiable, and the
    gradient pushes overlapping macros apart along whichever axis has
    smaller separation.

    Soft macros (indices >= num_hard) are excluded because they may
    physically overlap by design.
    """
    pos = positions[:num_hard]
    sz = sizes[:num_hard]

    diff = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs()       # [H, H, 2]
    min_sep = (sz.unsqueeze(0) + sz.unsqueeze(1)) / 2.0      # [H, H, 2]
    overlap = F.relu(min_sep - diff)                         # [H, H, 2]
    area = overlap[..., 0] * overlap[..., 1]                 # [H, H]

    # Zero out self-pairs and double-count.
    eye = torch.eye(num_hard, dtype=torch.bool, device=pos.device)
    area = area.masked_fill(eye, 0.0)
    return area.sum() / 2.0


def canvas_penalty(
    positions: torch.Tensor,
    sizes: torch.Tensor,
    canvas_width: float,
    canvas_height: float,
) -> torch.Tensor:
    """
    Hinge penalty for any part of a macro that lies outside the canvas.

    Equivalent to summing the area that pokes out on each side. Zero when
    every macro fits inside the canvas with its half-extent.
    """
    half = sizes / 2.0
    lo = positions - half
    hi = positions + half

    left   = F.relu(-lo[:, 0]).sum()
    bottom = F.relu(-lo[:, 1]).sum()
    right  = F.relu(hi[:, 0] - canvas_width).sum()
    top    = F.relu(hi[:, 1] - canvas_height).sum()

    return left + right + bottom + top


def total_loss(
    positions: torch.Tensor,
    benchmark: Benchmark,
    padded_nets: torch.Tensor,
    mask: torch.Tensor,
    net_weights: torch.Tensor,
    sizes_dev: torch.Tensor,
    *,
    gamma: float = 0.005,
    w_overlap: float = 50.0,
    w_canvas: float = 100.0,
) -> Tuple[torch.Tensor, dict]:
    """
    Compose the three smooth terms into a single scalar loss.

    Returns the loss and a dict of detached components for logging. The
    dict keys match the field names used in the training notebook so they
    can be plotted directly.
    """
    wl = smooth_hpwl(positions, padded_nets, mask, net_weights, gamma=gamma)
    olap = overlap_penalty(positions, sizes_dev, benchmark.num_hard_macros)
    cnvs = canvas_penalty(
        positions, sizes_dev, benchmark.canvas_width, benchmark.canvas_height
    )

    loss = wl + w_overlap * olap + w_canvas * cnvs

    components = {
        "wl": float(wl.detach()),
        "overlap": float(olap.detach()),
        "canvas": float(cnvs.detach()),
        "total": float(loss.detach()),
    }
    return loss, components
