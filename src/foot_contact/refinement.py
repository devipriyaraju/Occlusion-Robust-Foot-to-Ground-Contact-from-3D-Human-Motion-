from __future__ import annotations

import torch
import torch.nn.functional as F


def make_root_drift(root: torch.Tensor, level_m: float, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    batch, frames = root.shape[:2]
    drift = torch.cumsum(
        torch.randn(batch, 2, frames, device=root.device),
        dim=-1,
    )
    drift = F.avg_pool1d(
        F.pad(drift, (7, 7), mode="replicate"),
        15,
        stride=1,
    )
    drift = drift - drift[..., :1]
    scale = drift.norm(dim=1).amax(dim=-1)[:, None, None].clamp_min(1e-8)
    drift = drift / scale * level_m
    xyz = torch.cat(
        [drift.transpose(1, 2), torch.zeros(batch, frames, 1, device=root.device)],
        dim=-1,
    )
    return root + xyz


def foot_speed(
    root: torch.Tensor,
    foot_relative: torch.Tensor,
    fps: int,
) -> torch.Tensor:
    world = root[:, :, None] + foot_relative
    return (world[:, 1:] - world[:, :-1]).norm(dim=-1) * fps


def root_error_cm(root: torch.Tensor, reference_root: torch.Tensor) -> float:
    return (root - reference_root).norm(dim=-1).mean().item() * 100


def foot_slide_cm_per_s(
    root: torch.Tensor,
    foot_relative: torch.Tensor,
    contact_labels: torch.Tensor,
    fps: int,
) -> float:
    speed = foot_speed(root, foot_relative, fps)
    weighted = speed * contact_labels[:, 1:]
    return (weighted.sum() / contact_labels[:, 1:].sum().clamp_min(1)).item() * 100


def wham_style_velocity_correction(
    corrupted_root: torch.Tensor,
    foot_relative: torch.Tensor,
    contact_confidence: torch.Tensor,
    fps: int,
) -> torch.Tensor:
    root_velocity = (corrupted_root[:, 1:] - corrupted_root[:, :-1]) * fps
    foot_relative_velocity = (foot_relative[:, 1:] - foot_relative[:, :-1]) * fps
    weights = contact_confidence[:, 1:]
    contact_velocity = -(
        weights[..., None] * foot_relative_velocity
    ).sum(dim=2) / (weights.sum(dim=2, keepdim=True) + 1e-6)
    confidence = weights.max(dim=2, keepdim=True).values
    corrected_velocity = confidence * contact_velocity + (1 - confidence) * root_velocity
    corrected_velocity[..., 2] = root_velocity[..., 2]
    return torch.cat(
        [
            corrupted_root[:, :1],
            corrupted_root[:, :1] + torch.cumsum(corrected_velocity / fps, dim=1),
        ],
        dim=1,
    )


def optimize_root_velocity(
    corrupted_root: torch.Tensor,
    foot_relative: torch.Tensor,
    contact_confidence: torch.Tensor,
    fps: int,
    slide_weight: float = 1.0,
    velocity_weight: float = 0.01,
    acceleration_weight: float = 1e-4,
    steps: int = 300,
    learning_rate: float = 0.02,
) -> torch.Tensor:
    batch = len(corrupted_root)
    initial_velocity = torch.cat(
        [
            torch.zeros(batch, 1, 3, device=corrupted_root.device),
            (corrupted_root[:, 1:] - corrupted_root[:, :-1]) * fps,
        ],
        dim=1,
    )
    velocity = initial_velocity.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([velocity], lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: 1 - step / steps,
    )

    def integrate(values: torch.Tensor) -> torch.Tensor:
        return corrupted_root[:, :1] + torch.cumsum(values / fps, dim=1)

    for _ in range(steps):
        world_feet = integrate(velocity)[:, :, None] + foot_relative
        foot_velocity = (world_feet[:, 1:] - world_feet[:, :-1]) * fps
        acceleration = (velocity[:, 2:] - velocity[:, 1:-1]) * fps

        slide_loss = (
            contact_confidence[:, 1:, :, None] * foot_velocity.pow(2)
        ).sum(dim=-1).mean()
        velocity_loss = (velocity - initial_velocity).pow(2).sum(dim=-1).mean()
        acceleration_loss = acceleration.pow(2).sum(dim=-1).mean()
        loss = (
            slide_weight * slide_loss
            + velocity_weight * velocity_loss
            + acceleration_weight * acceleration_loss
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    return integrate(velocity).detach()
