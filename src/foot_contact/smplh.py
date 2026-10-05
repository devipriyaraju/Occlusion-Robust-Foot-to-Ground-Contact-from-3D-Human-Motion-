from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """Convert axis angle rotations with shape (..., 3) to rotation matrices."""
    angle = axis_angle.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    x, y, z = (axis_angle / angle).unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack(
        [zero, -z, y, z, zero, -x, -y, x, zero],
        dim=-1,
    ).view(*axis_angle.shape[:-1], 3, 3)
    sin = angle.sin()[..., None]
    cos = angle.cos()[..., None]
    eye = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)
    return eye + sin * skew + (1 - cos) * (skew @ skew)


def _dense(array: np.ndarray) -> np.ndarray:
    if array.dtype == object:
        return np.asarray(array.item().todense())
    return np.asarray(array)


class SMPLH:
    """Minimal SMPL-H forward model built from the released model tensors.

    This implementation applies shape blend shapes, joint regression,
    axis angle forward kinematics, pose blend shapes, and linear blend skinning.
    """

    def __init__(
        self,
        model_path: str | Path,
        device: torch.device | str,
        n_betas: int = 16,
    ) -> None:
        model = np.load(model_path, allow_pickle=True)
        self.device = torch.device(device)

        def tensor(name: str) -> torch.Tensor:
            return torch.tensor(
                _dense(model[name]),
                dtype=torch.float32,
                device=self.device,
            )

        self.v_template = tensor("v_template")
        self.shapedirs = tensor("shapedirs")[..., :n_betas]
        self.posedirs = tensor("posedirs")
        self.joint_regressor = tensor("J_regressor")
        self.weights = tensor("weights")

        parents = model["kintree_table"][0].astype(np.int64)
        parents[0] = -1
        self.parents = parents.tolist()
        self.num_joints = len(self.parents)

        expected_pose_basis = (self.num_joints - 1) * 9
        if self.posedirs.shape[-1] != expected_pose_basis:
            raise ValueError("Unexpected SMPL-H pose blend shape dimension")
        if self.joint_regressor.shape != (self.num_joints, len(self.v_template)):
            raise ValueError("Unexpected SMPL-H joint regressor shape")
        if self.weights.shape != (len(self.v_template), self.num_joints):
            raise ValueError("Unexpected SMPL-H skinning weight shape")
        if not all(parent < idx for idx, parent in enumerate(self.parents)):
            raise ValueError("SMPL-H parents must precede children")

    def forward(
        self,
        poses: torch.Tensor,
        betas: torch.Tensor,
        translation: torch.Tensor,
        vertex_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = len(poses)
        beta_count = min(len(betas), self.shapedirs.shape[-1])

        shaped = self.v_template + self.shapedirs[..., :beta_count] @ betas[:beta_count]
        rest_joints = self.joint_regressor @ shaped
        rotations = axis_angle_to_matrix(poses.view(batch, self.num_joints, 3))

        global_rotations = [rotations[:, 0]]
        global_positions = [rest_joints[0].expand(batch, 3)]
        for joint_idx in range(1, self.num_joints):
            parent = self.parents[joint_idx]
            global_rotations.append(global_rotations[parent] @ rotations[:, joint_idx])
            offset = rest_joints[joint_idx] - rest_joints[parent]
            rotated_offset = (
                global_rotations[parent] @ offset[:, None]
            ).squeeze(-1)
            global_positions.append(global_positions[parent] + rotated_offset)

        global_rotations_tensor = torch.stack(global_rotations, dim=1)
        global_positions_tensor = torch.stack(global_positions, dim=1)

        if vertex_indices is None:
            vertex_indices = torch.arange(len(self.v_template), device=self.device)

        identity = torch.eye(3, device=self.device, dtype=poses.dtype)
        pose_features = (rotations[:, 1:] - identity).reshape(batch, -1)
        posed = shaped[vertex_indices] + (
            self.posedirs[vertex_indices] @ pose_features.T
        ).permute(2, 0, 1)

        joint_offsets = global_positions_tensor - (
            global_rotations_tensor @ rest_joints[None, :, :, None]
        ).squeeze(-1)
        selected_weights = self.weights[vertex_indices]
        blended_rotation = torch.einsum(
            "vj,bjmn->bvmn",
            selected_weights,
            global_rotations_tensor,
        )
        blended_translation = torch.einsum(
            "vj,bjm->bvm",
            selected_weights,
            joint_offsets,
        )
        vertices = (
            blended_rotation @ posed[..., None]
        ).squeeze(-1) + blended_translation

        joints_world = global_positions_tensor + translation[:, None]
        vertices_world = vertices + translation[:, None]
        return joints_world, vertices_world

    __call__ = forward


def discover_smplh_models(root: str | Path, device: str | torch.device) -> dict[str, SMPLH]:
    root = Path(root)
    paths = {path.parent.name: path for path in root.glob("**/model.npz")}
    if not paths:
        raise FileNotFoundError(f"No SMPL-H model.npz files found below {root}")
    return {gender: SMPLH(path, device=device) for gender, path in paths.items()}
