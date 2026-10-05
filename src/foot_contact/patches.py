from __future__ import annotations

import torch

from .smplh import SMPLH


def sole_patch_indices(model: SMPLH, vertices_per_patch: int = 20) -> torch.Tensor:
    """Select left and right heel and toe sole vertex patches in the rest pose."""
    rest_joints = model.joint_regressor @ model.v_template
    dominant_joint = model.weights.argmax(dim=1)
    vertical = model.v_template[:, 1]
    forward = model.v_template[:, 2]

    patches: list[torch.Tensor] = []
    for ankle_joint, toe_joint in ((7, 10), (8, 11)):
        heel = torch.where(
            (dominant_joint == ankle_joint)
            & (forward < rest_joints[ankle_joint, 2])
        )[0]
        toe = torch.where(dominant_joint == toe_joint)[0]
        for candidate in (heel, toe):
            if len(candidate) < vertices_per_patch:
                raise ValueError("Too few candidate vertices for a sole patch")
            patches.append(
                candidate[vertical[candidate].argsort()[:vertices_per_patch]]
            )
    return torch.cat(patches)
