import torch

from foot_contact.model import ContactNet, FOOT_JOINTS


def test_full_body_model_shape() -> None:
    model = ContactNet(width=32)
    joints = torch.randn(2, 64, 22, 3)
    visibility = torch.ones(2, 64, 22)
    output = model(joints, visibility)
    assert output.shape == (2, 64, 4)


def test_feet_only_model_shape() -> None:
    model = ContactNet(joint_indices=FOOT_JOINTS, width=32)
    joints = torch.randn(2, 64, 22, 3)
    visibility = torch.ones(2, 64, 22)
    output = model(joints, visibility)
    assert output.shape == (2, 64, 4)
