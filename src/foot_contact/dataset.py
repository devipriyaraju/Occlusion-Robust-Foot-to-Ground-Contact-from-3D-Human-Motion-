from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .config import MotionConfig
from .labels import NUM_CONTACTS, flat_floor_contact_labels, support_aware_contact_labels
from .smplh import SMPLH


def parse_gender(data: np.lib.npyio.NpzFile) -> str:
    gender = data["gender"].item() if data["gender"].shape == () else data["gender"]
    if isinstance(gender, bytes):
        return gender.decode()
    return str(gender)


def _linear_resample(
    values: torch.Tensor,
    frame_count: int,
    source_fps: float,
    target_fps: int,
) -> torch.Tensor:
    timestamps = torch.arange(
        0,
        (frame_count - 1) / source_fps,
        1 / target_fps,
        device=values.device,
    ) * source_fps
    low = timestamps.floor().long().clamp(max=frame_count - 2)
    weight_shape = [len(timestamps)] + [1] * (values.ndim - 1)
    weight = (timestamps - low).view(*weight_shape)
    return (1 - weight) * values[low] + weight * values[low + 1]


def process_sequence(
    path: str | Path,
    models: dict[str, SMPLH],
    reference_model: SMPLH,
    patch_indices: torch.Tensor,
    motion: MotionConfig,
) -> dict[str, torch.Tensor] | None:
    data = np.load(path, allow_pickle=True)
    if "poses" not in data.files or "trans" not in data.files:
        return None

    fps_key = "mocap_framerate" if "mocap_framerate" in data.files else "mocap_frame_rate"
    source_fps = float(data[fps_key])
    frame_count = len(data["poses"])
    if (frame_count - 1) / source_fps * motion.fps < motion.clip_length:
        return None

    model = models.get(parse_gender(data), reference_model)
    device = model.device
    poses = torch.tensor(
        data["poses"][:, : model.num_joints * 3],
        dtype=torch.float32,
        device=device,
    )
    translation = torch.tensor(data["trans"], dtype=torch.float32, device=device)
    betas = torch.tensor(data["betas"], dtype=torch.float32, device=device)

    joint_chunks = []
    contact_chunks = []
    for start in range(0, frame_count, 4096):
        joints, vertices = model(
            poses[start : start + 4096],
            betas,
            translation[start : start + 4096],
            patch_indices,
        )
        joint_chunks.append(joints[:, :22])
        contact_chunks.append(
            vertices.view(-1, NUM_CONTACTS, motion.sole_patch_vertices, 3).mean(dim=2)
        )

    joints = _linear_resample(
        torch.cat(joint_chunks),
        frame_count,
        source_fps,
        motion.fps,
    )
    contacts = _linear_resample(
        torch.cat(contact_chunks),
        frame_count,
        source_fps,
        motion.fps,
    )

    speed = torch.zeros_like(contacts[..., 0])
    speed[1:] = (contacts[1:] - contacts[:-1]).norm(dim=-1) * motion.fps
    speed[0] = speed[1]

    vertical = contacts[..., 2]
    stationary_candidates = vertical[speed < motion.speed_threshold_mps]
    if stationary_candidates.numel() > 30:
        floor = torch.quantile(stationary_candidates, 0.05)
    else:
        floor = torch.quantile(vertical.flatten(), 0.01)

    contact_height = vertical - floor
    root = joints[:, 0]
    hard = support_aware_contact_labels(
        foot_speed=speed,
        contact_height=contact_height,
        root_height=root[:, 2],
        speed_threshold_mps=motion.speed_threshold_mps,
        height_threshold_m=motion.height_threshold_m,
        min_stationary_run=motion.min_stationary_run,
        leg_extension_threshold_m=motion.leg_extension_threshold_m,
    )
    hard_flat = flat_floor_contact_labels(
        foot_speed=speed,
        contact_height=contact_height,
        speed_threshold_mps=motion.speed_threshold_mps,
        height_threshold_m=motion.height_threshold_m,
    )

    sequence = {
        "rel": joints - root[:, None],
        "cp_rel": contacts - root[:, None],
        "root": root,
        "h": contact_height,
        "hard": hard,
        "hard_flat": hard_flat,
    }

    starts = range(
        0,
        len(joints) - motion.clip_length + 1,
        motion.stride,
    )
    return {
        key: torch.stack(
            [values[start : start + motion.clip_length] for start in starts]
        ).cpu()
        for key, values in sequence.items()
    }


def subject_id(dataset_name: str, sequence_path: str | Path) -> str:
    return f"{dataset_name}_{Path(sequence_path).parent.name}"


def subject_split(
    subjects_per_clip: list[str],
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique_subjects = sorted(set(subjects_per_clip))
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_subjects)
    test_count = max(1, len(unique_subjects) // 10)
    test_subjects = set(unique_subjects[:test_count])
    val_subjects = set(unique_subjects[test_count : 2 * test_count])
    subject_array = np.asarray(subjects_per_clip)

    test = np.where(np.isin(subject_array, list(test_subjects)))[0]
    val = np.where(np.isin(subject_array, list(val_subjects)))[0]
    train = np.where(~np.isin(subject_array, list(test_subjects | val_subjects)))[0]
    return train, val, test
