from __future__ import annotations

import torch


JOINT_SETS = {
    "L_foot": [7, 10],
    "R_foot": [8, 11],
    "L_lower_leg": [4, 7, 10],
    "R_lower_leg": [5, 8, 11],
    "both_feet": [7, 8, 10, 11],
}


def hold_frames(values: torch.Tensor, length: int) -> torch.Tensor:
    batch, frames = values.shape[:2]
    timeline = torch.arange(frames, device=values.device)[None]
    start = torch.randint(1, frames - length + 1, (batch, 1), device=values.device)
    gather_index = torch.where(
        (timeline >= start) & (timeline < start + length),
        start - 1,
        timeline,
    )
    return values.gather(
        1,
        gather_index[:, :, None, None].expand(-1, -1, 22, 3),
    )


def corrupt(
    relative_joints: torch.Tensor,
    mask_probability: float = 0.0,
    noise_sigma_m: float = 0.0,
    hold_length: int = 0,
    structured_set: str | None = None,
    block_length: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, frames = relative_joints.shape[:2]
    values = relative_joints

    if noise_sigma_m > 0:
        values = values + torch.randn_like(values) * noise_sigma_m
    if hold_length > 0:
        values = hold_frames(values, hold_length)

    visibility = (
        torch.rand(batch, frames, 22, device=values.device) >= mask_probability
    ).float()

    if structured_set is not None:
        timeline = torch.arange(frames, device=values.device)[None]
        length = block_length or frames
        start = torch.randint(0, frames - length + 1, (batch, 1), device=values.device)
        hidden = ((timeline >= start) & (timeline < start + length)).float()
        visibility[:, :, JOINT_SETS[structured_set]] *= 1 - hidden[..., None]

    return values, visibility


def training_corruption(relative_joints: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    batch, frames = relative_joints.shape[:2]
    device = relative_joints.device
    timeline = torch.arange(frames, device=device)[None]

    noise_sigma = torch.rand(batch, 1, 1, 1, device=device) * 0.05
    use_noise = torch.rand(batch, 1, 1, 1, device=device) < 0.7
    values = relative_joints + torch.randn_like(relative_joints) * noise_sigma * use_noise

    hold_length = int(torch.randint(0, 11, (1,), device=device).item())
    if hold_length > 0:
        values = hold_frames(values, hold_length)

    mask_probability = torch.rand(batch, 1, 1, device=device) * 0.5
    use_mask = torch.rand(batch, 1, 1, device=device) < 0.7
    visibility = (
        torch.rand(batch, frames, 22, device=device)
        >= mask_probability * use_mask
    ).float()

    set_matrix = torch.zeros(len(JOINT_SETS) + 1, 22, device=device)
    for row, joints in enumerate(JOINT_SETS.values(), start=1):
        set_matrix[row, joints] = 1
    selected = set_matrix[
        torch.randint(0, len(set_matrix), (batch,), device=device)
    ]
    start = torch.randint(0, frames, (batch, 1), device=device)
    length = torch.randint(10, frames + 1, (batch, 1), device=device)
    hidden = ((timeline >= start) & (timeline < start + length)).float()
    visibility = visibility * (1 - hidden[..., None] * selected[:, None])
    return values, visibility
