from __future__ import annotations

import torch
import torch.nn.functional as F


CONTACT_NAMES = ("L_heel", "L_toe", "R_heel", "R_toe")
NUM_CONTACTS = 4


def temporal_opening(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Remove short positive runs from a boolean sequence tensor."""
    values = mask.float().permute(0, 2, 1)
    pad = kernel_size // 2
    eroded = -F.max_pool1d(
        F.pad(-values, (pad, pad), mode="replicate"),
        kernel_size,
        stride=1,
    )
    opened = F.max_pool1d(
        F.pad(eroded, (pad, pad), mode="replicate"),
        kernel_size,
        stride=1,
    )
    return opened.permute(0, 2, 1) > 0.5


def support_aware_contact_labels(
    foot_speed: torch.Tensor,
    contact_height: torch.Tensor,
    root_height: torch.Tensor,
    speed_threshold_mps: float,
    height_threshold_m: float,
    min_stationary_run: int,
    leg_extension_threshold_m: float,
) -> torch.Tensor:
    """Label stationary support including contact on raised surfaces."""
    still = temporal_opening(
        (foot_speed < speed_threshold_mps)[None],
        min_stationary_run,
    )[0]
    raised = contact_height > 2 * height_threshold_m
    leg_extended = (
        root_height[:, None] - contact_height
    ) > leg_extension_threshold_m
    return (still & (~raised | leg_extended)).float()


def flat_floor_contact_labels(
    foot_speed: torch.Tensor,
    contact_height: torch.Tensor,
    speed_threshold_mps: float,
    height_threshold_m: float,
) -> torch.Tensor:
    return (
        (contact_height < height_threshold_m)
        & (foot_speed < speed_threshold_mps)
    ).float()
