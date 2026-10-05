from __future__ import annotations

from pathlib import Path

import torch


def save_cache(path: str | Path, tensors: dict[str, torch.Tensor], subjects: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(tensors)
    payload["subj"] = subjects
    torch.save(payload, path)


def load_cache(path: str | Path) -> dict:
    return torch.load(Path(path), map_location="cpu", weights_only=False)
