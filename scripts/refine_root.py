from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from foot_contact.config import load_config
from foot_contact.dataset import subject_split
from foot_contact.io import load_cache
from foot_contact.model import ContactNet
from foot_contact.refinement import (
    foot_slide_cm_per_s,
    make_root_drift,
    optimize_root_velocity,
    root_error_cm,
    wham_style_velocity_correction,
)
from foot_contact.training import predict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data = load_cache(cfg.data.cache_path)
    _, _, test_idx = subject_split(data["subj"], cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    relative = data["rel"][test_idx].to(device)
    root = data["root"][test_idx].to(device)
    foot_relative = data["cp_rel"][test_idx].to(device)
    labels = data["hard"][test_idx].to(device)

    model = ContactNet(width=cfg.train.width, fps=cfg.motion.fps).to(device)
    checkpoint = Path(cfg.results_dir) / "checkpoints" / "tcn_full.pt"
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    probability = predict(model, relative)
    confidence = probability * (probability > cfg.refinement.contact_threshold)

    rows = {}
    for label, level in {"mild 5cm": 0.05, "medium 10cm": 0.10, "hard 25cm": 0.25}.items():
        drifted = make_root_drift(root, level, seed=cfg.train.seed)
        wham = wham_style_velocity_correction(
            drifted,
            foot_relative,
            confidence,
            cfg.motion.fps,
        )
        refined = optimize_root_velocity(
            drifted,
            foot_relative,
            confidence,
            cfg.motion.fps,
            slide_weight=cfg.refinement.slide_weight,
            velocity_weight=cfg.refinement.velocity_weight,
            acceleration_weight=cfg.refinement.acceleration_weight,
            steps=cfg.refinement.steps,
            learning_rate=cfg.refinement.learning_rate,
        )
        for method, estimate in {
            "drifted": drifted,
            "WHAM-style": wham,
            "optimised": refined,
        }.items():
            rows[(label, method)] = {
                "root error (cm)": root_error_cm(estimate, root),
                "foot slide (cm/s)": foot_slide_cm_per_s(
                    estimate,
                    foot_relative,
                    labels,
                    cfg.motion.fps,
                ),
            }

    table = pd.DataFrame(rows).T
    output = Path(cfg.results_dir) / "tables" / "root_refinement.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output)
    print(table.round(2))


if __name__ == "__main__":
    main()
