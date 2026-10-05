from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class DataConfig:
    amass_root: Path
    smplh_root: Path
    cache_path: Path
    datasets: tuple[str, ...] = ("CMU", "KIT")
    max_sequences: int | None = None


@dataclass(frozen=True)
class MotionConfig:
    fps: int = 30
    clip_length: int = 64
    stride: int = 32
    height_threshold_m: float = 0.05
    speed_threshold_mps: float = 0.20
    min_stationary_run: int = 5
    leg_extension_threshold_m: float = 0.60
    sole_patch_vertices: int = 20


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 15
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 0
    width: int = 256


@dataclass(frozen=True)
class RefinementConfig:
    slide_weight: float = 1.0
    velocity_weight: float = 0.01
    acceleration_weight: float = 1e-4
    contact_threshold: float = 0.8
    steps: int = 300
    learning_rate: float = 0.02


@dataclass(frozen=True)
class ProjectConfig:
    data: DataConfig
    motion: MotionConfig
    train: TrainConfig
    refinement: RefinementConfig
    results_dir: Path


def load_config(path: str | Path) -> ProjectConfig:
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    data = raw["data"]
    motion = raw.get("motion", {})
    train = raw.get("train", {})
    refinement = raw.get("refinement", {})

    return ProjectConfig(
        data=DataConfig(
            amass_root=Path(data["amass_root"]).expanduser(),
            smplh_root=Path(data["smplh_root"]).expanduser(),
            cache_path=Path(data["cache_path"]).expanduser(),
            datasets=tuple(data.get("datasets", ["CMU", "KIT"])),
            max_sequences=data.get("max_sequences"),
        ),
        motion=MotionConfig(**motion),
        train=TrainConfig(**train),
        refinement=RefinementConfig(**refinement),
        results_dir=Path(raw["results_dir"]).expanduser(),
    )
