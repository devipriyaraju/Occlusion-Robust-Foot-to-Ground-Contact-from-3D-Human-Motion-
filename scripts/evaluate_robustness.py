from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from foot_contact.config import load_config
from foot_contact.dataset import subject_split
from foot_contact.io import load_cache
from foot_contact.model import ContactNet, FOOT_JOINTS
from foot_contact.training import evaluate


CONDITIONS = {
    "clean": {},
    "mask 10%": {"mask_probability": 0.1},
    "mask 30%": {"mask_probability": 0.3},
    "mask 50%": {"mask_probability": 0.5},
    "noise 10mm": {"noise_sigma_m": 0.010},
    "noise 25mm": {"noise_sigma_m": 0.025},
    "noise 50mm": {"noise_sigma_m": 0.050},
    "hold 3f": {"hold_length": 3},
    "hold 5f": {"hold_length": 5},
    "hold 10f": {"hold_length": 10},
    "L foot hidden": {"structured_set": "L_foot"},
    "R lower leg hidden": {"structured_set": "R_lower_leg"},
    "both feet 30f": {"structured_set": "both_feet", "block_length": 30},
    "both feet hidden": {"structured_set": "both_feet"},
}


MODEL_SPECS = {
    "mlp_full": {"temporal": False},
    "tcn_foot": {"joint_indices": FOOT_JOINTS},
    "tcn_full_novis": {"use_visibility": False},
    "tcn_full": {},
    "tcn_full_clean": {},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data = load_cache(cfg.data.cache_path)
    _, _, test_idx = subject_split(data["subj"], cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    relative = data["rel"][test_idx].to(device)
    labels = data["hard"][test_idx].to(device)

    rows = {}
    for name, kwargs in MODEL_SPECS.items():
        model = ContactNet(width=cfg.train.width, fps=cfg.motion.fps, **kwargs).to(device)
        checkpoint = Path(cfg.results_dir) / "checkpoints" / f"{name}.pt"
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        rows[name] = {
            condition: evaluate(model, relative, labels, corruption=corruption)["f1"]
            for condition, corruption in CONDITIONS.items()
        }

    table = pd.DataFrame(rows)
    output = Path(cfg.results_dir) / "tables" / "contact_robustness.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output)
    print(table.round(4))


if __name__ == "__main__":
    main()
