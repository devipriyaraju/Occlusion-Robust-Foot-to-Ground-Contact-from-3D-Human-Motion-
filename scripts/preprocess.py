from __future__ import annotations

import argparse
from pathlib import Path

import torch

from foot_contact.config import load_config
from foot_contact.dataset import process_sequence, subject_id
from foot_contact.io import save_cache
from foot_contact.patches import sole_patch_indices
from foot_contact.smplh import discover_smplh_models


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = discover_smplh_models(cfg.data.smplh_root, device)
    reference = models.get("neutral", next(iter(models.values())))
    patches = sole_patch_indices(reference, cfg.motion.sole_patch_vertices)

    chunks = []
    subjects = []
    with torch.no_grad():
        for dataset_name in cfg.data.datasets:
            files = sorted((cfg.data.amass_root / dataset_name).glob("**/*_poses.npz"))
            if cfg.data.max_sequences:
                step = max(1, len(files) // cfg.data.max_sequences)
                files = files[::step][: cfg.data.max_sequences]

            for index, path in enumerate(files):
                try:
                    clip_dict = process_sequence(
                        path,
                        models,
                        reference,
                        patches,
                        cfg.motion,
                    )
                except Exception as exc:
                    print(f"skip {path}: {exc}")
                    continue
                if clip_dict is None:
                    continue
                chunks.append(clip_dict)
                subjects.extend(
                    [subject_id(dataset_name, path)] * len(clip_dict["rel"])
                )
                if index % 500 == 0:
                    print(f"{dataset_name} {index}/{len(files)}")

    if not chunks:
        raise RuntimeError("No AMASS clips were produced")

    merged = {
        key: torch.cat([chunk[key] for chunk in chunks])
        for key in chunks[0]
    }
    save_cache(cfg.data.cache_path, merged, subjects)
    print(
        f"saved {len(merged['rel'])} clips from {len(set(subjects))} subjects "
        f"to {cfg.data.cache_path}"
    )


if __name__ == "__main__":
    main()
