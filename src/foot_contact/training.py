from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .corruptions import corrupt, training_corruption
from .metrics import contact_metrics
from .model import ContactNet


@torch.no_grad()
def predict(
    model: ContactNet,
    relative_joints: torch.Tensor,
    batch_size: int = 2048,
    corruption: dict | None = None,
) -> torch.Tensor:
    model.eval()
    torch.manual_seed(123)
    corruption = corruption or {}
    outputs = []
    for start in range(0, len(relative_joints), batch_size):
        joints, visibility = corrupt(
            relative_joints[start : start + batch_size],
            **corruption,
        )
        outputs.append(torch.sigmoid(model(joints, visibility)))
    return torch.cat(outputs)


@torch.no_grad()
def evaluate(
    model: ContactNet,
    relative_joints: torch.Tensor,
    labels: torch.Tensor,
    corruption: dict | None = None,
) -> dict[str, float]:
    probabilities = predict(model, relative_joints, corruption=corruption)
    return contact_metrics(probabilities, labels)


def train_contact_model(
    name: str,
    model: ContactNet,
    train_joints: torch.Tensor,
    train_labels: torch.Tensor,
    val_joints: torch.Tensor,
    val_labels: torch.Tensor,
    checkpoint_dir: str | Path,
    epochs: int = 15,
    batch_size: int = 512,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 0,
    corrupt_train: bool = True,
) -> ContactNet:
    torch.manual_seed(seed)
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    steps_per_epoch = (len(train_joints) + batch_size - 1) // batch_size
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        learning_rate,
        total_steps=epochs * steps_per_epoch,
    )

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(len(train_joints), device=train_joints.device)
        total_loss = 0.0
        for start in range(0, len(permutation), batch_size):
            batch = permutation[start : start + batch_size]
            if corrupt_train:
                joints, visibility = training_corruption(train_joints[batch])
            else:
                joints, visibility = corrupt(train_joints[batch])

            logits = model(joints, visibility)
            loss = F.binary_cross_entropy_with_logits(logits, train_labels[batch])
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item() * len(batch)

        if epoch % 5 == 4 or epoch == epochs - 1:
            clean = evaluate(model, val_joints, val_labels)
            masked = evaluate(
                model,
                val_joints,
                val_labels,
                corruption={"mask_probability": 0.3},
            )
            print(
                f"{name:20s} epoch {epoch + 1:3d} "
                f"loss {total_loss / len(train_joints):.4f} "
                f"val F1 clean {clean['f1']:.4f} "
                f"val F1 mask30 {masked['f1']:.4f}"
            )

    torch.save(model.state_dict(), checkpoint_dir / f"{name}.pt")
    return model
