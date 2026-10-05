import torch

from foot_contact.smplh import axis_angle_to_matrix


def test_zero_axis_angle_is_identity() -> None:
    axis_angle = torch.zeros(4, 3)
    rotation = axis_angle_to_matrix(axis_angle)
    expected = torch.eye(3).expand(4, 3, 3)
    assert torch.allclose(rotation, expected, atol=1e-6)


def test_rotation_matrices_are_orthonormal() -> None:
    axis_angle = torch.randn(8, 3) * 0.2
    rotation = axis_angle_to_matrix(axis_angle)
    product = rotation.transpose(-1, -2) @ rotation
    expected = torch.eye(3).expand_as(product)
    assert torch.allclose(product, expected, atol=1e-5)
