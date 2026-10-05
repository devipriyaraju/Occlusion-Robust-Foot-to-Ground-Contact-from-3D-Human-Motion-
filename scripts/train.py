from __future__ import annotations

import argparse
from pathlib import Path

import torch

from foot_contact.config import load_config
from foot_contact.dataset import subject_split
from foot_contact.io import load_cache
from foot_contact.model import ContactNet, FOOT_JOINTS
from foot_contact.training import train_contact_model


VARIANTS = {
    "mlp_full": {"temporal": False},
    "tcn_foot": {"joint_indices": FOOT_JOINTS},
    "tcn_full_novis": {"use_visibility": False},
    "tcn_full": {},
    "tcn_full_clean": {"corrupt_train": False},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--variant", choices=VARIANTS, default="tcn_full")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data = load_cache(cfg.data.cache_path)
    train_idx, val_idx, _ = subject_split(data["subj"], cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    relative = data["rel"].to(device)
    labels = data["hard"].to(device)
    variant = dict(VARIANTS[args.variant])
    corrupt_train = variant.pop("corrupt_train", True)
    seed = cfg.train.seed if args.seed is None else args.seed

    torch.manual_seed(seed)
    model = ContactNet(
        width=cfg.train.width,
        fps=cfg.motion.fps,
        **variant,
    ).to(device)
    train_contact_model(
        name=args.variant if args.seed is None else f"{args.variant}_seed{seed}",
        model=model,
        train_joints=relative[train_idx],
        train_labels=labels[train_idx],
        val_joints=relative[val_idx],
        val_labels=labels[val_idx],
        checkpoint_dir=Path(cfg.results_dir) / "checkpoints",
        epochs=cfg.train.epochs,
        batch_size=cfg.train.batch_size,
        learning_rate=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
        seed=seed,
        corrupt_train=corrupt_train,
    )


if __name__ == "__main__":
    main()
