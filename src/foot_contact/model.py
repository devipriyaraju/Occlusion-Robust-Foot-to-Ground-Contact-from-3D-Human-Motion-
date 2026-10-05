from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


FULL_BODY_JOINTS = list(range(22))
FOOT_JOINTS = [7, 8, 10, 11]


def motion_features(
    joints: torch.Tensor,
    visibility: torch.Tensor,
    joint_indices: list[int],
    use_visibility: bool,
    fps: int,
) -> torch.Tensor:
    velocity = torch.zeros_like(joints)
    velocity[:, 1:] = (joints[:, 1:] - joints[:, :-1]) * fps
    velocity[:, 0] = velocity[:, 1]

    velocity_visibility = visibility.clone()
    velocity_visibility[:, 1:] = visibility[:, 1:] * visibility[:, :-1]

    parts = [
        joints * visibility[..., None],
        velocity * velocity_visibility[..., None],
    ]
    if use_visibility:
        parts.append(visibility[..., None])
    return torch.cat(parts, dim=-1)[:, :, joint_indices].flatten(2)


class ResidualTemporalBlock(nn.Module):
    def __init__(self, width: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            width,
            width,
            kernel_size,
            padding=dilation * (kernel_size // 2),
            dilation=dilation,
        )
        self.norm = nn.GroupNorm(8, width)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + F.gelu(self.norm(self.conv(values)))


class ContactNet(nn.Module):
    def __init__(
        self,
        joint_indices: list[int] | None = None,
        temporal: bool = True,
        use_visibility: bool = True,
        width: int = 256,
        fps: int = 30,
    ) -> None:
        super().__init__()
        self.joint_indices = joint_indices or FULL_BODY_JOINTS
        self.use_visibility = use_visibility
        self.fps = fps
        feature_width = 7 if use_visibility else 6
        self.input_projection = nn.Conv1d(
            len(self.joint_indices) * feature_width,
            width,
            kernel_size=1,
        )
        kernel_size = 3 if temporal else 1
        dilations = (1, 2, 4, 8) if temporal else (1, 1, 1, 1)
        self.body = nn.Sequential(
            *[
                ResidualTemporalBlock(width, kernel_size, dilation)
                for dilation in dilations
            ]
        )
        self.head = nn.Conv1d(width, 4, kernel_size=1)

    def forward(
        self,
        joints: torch.Tensor,
        visibility: torch.Tensor,
    ) -> torch.Tensor:
        features = motion_features(
            joints,
            visibility,
            self.joint_indices,
            self.use_visibility,
            self.fps,
        ).transpose(1, 2)
        logits = self.head(self.body(self.input_projection(features)))
        return logits.transpose(1, 2)
