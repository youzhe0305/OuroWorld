import math

import numpy as np
import pytest
import torch

from ouroworld.geometry.camera import Camera
from ouroworld.geometry.se3 import normalize_quaternion, quaternion_multiply, se3_exp_apply
from ouroworld.geometry.trajectories import (
    horizontal_orbit,
    project_world_point,
    uniform_horizontal_angles,
    unproject_z_depth,
)
from tests.helpers import look_at_camera


def _hat(vector: torch.Tensor) -> torch.Tensor:
    x, y, z = vector.tolist()
    return torch.tensor([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=torch.float64)


def _quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = normalize_quaternion(q).tolist()
    return torch.tensor(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=torch.float64,
    )


@pytest.mark.parametrize("scale", [1e-6, 0.3, 2.0])
def test_se3_matches_matrix_exponential(scale: float) -> None:
    generator = torch.Generator().manual_seed(0)
    twist = scale * torch.randn(6, generator=generator, dtype=torch.float64)
    point = torch.randn(3, generator=generator, dtype=torch.float64)
    quaternion = torch.randn(4, generator=generator, dtype=torch.float64)

    algebra = torch.zeros(4, 4, dtype=torch.float64)
    algebra[:3, :3] = _hat(twist[3:])
    algebra[:3, 3] = twist[:3]
    transform = torch.linalg.matrix_exp(algebra)

    moved, turned = se3_exp_apply(point[None], quaternion[None], twist[None])
    assert torch.allclose(moved[0], transform[:3, :3] @ point + transform[:3, 3], atol=1e-10)
    expected = transform[:3, :3] @ _quaternion_to_matrix(quaternion)
    assert torch.allclose(_quaternion_to_matrix(turned[0]), expected, atol=1e-10)


def test_zero_twist_is_identity_up_to_normalisation() -> None:
    points = torch.randn(5, 3)
    quaternions = torch.randn(5, 4)
    moved, turned = se3_exp_apply(points, quaternions, torch.zeros(5, 6))
    assert torch.equal(moved, points)
    assert torch.allclose(turned, normalize_quaternion(quaternions))


def test_quaternion_product_is_associative() -> None:
    a, b, c = torch.randn(3, 4, dtype=torch.float64)
    assert torch.allclose(
        quaternion_multiply(quaternion_multiply(a, b), c),
        quaternion_multiply(a, quaternion_multiply(b, c)),
    )


def test_camera_rejects_offset_principal_point() -> None:
    K = np.array([[50.0, 0, 40], [0, 50, 18], [0, 0, 1]])
    with pytest.raises(ValueError, match="principal point"):
        Camera(64, 36, K, np.eye(4))


def test_raster_matrices_project_like_pinhole() -> None:
    camera = look_at_camera()
    matrices = camera.raster_matrices()
    point = torch.tensor([0.3, -0.2, 0.5, 1.0])
    clip = point @ matrices.full_projection
    ndc = clip[:2] / clip[3]
    pixel = ((ndc + 1) * torch.tensor([camera.width, camera.height]) - 1) / 2
    expected, _ = project_world_point(camera.K, camera.c2w, point[:3].numpy())
    # The rasterizer's pixel centres sit at integer coordinates, offset by half a pixel.
    assert np.allclose(pixel.numpy(), expected - 0.5, atol=1e-4)
    assert torch.allclose(matrices.camera_center, torch.tensor(camera.center, dtype=torch.float32))


def test_fov_round_trip() -> None:
    camera = look_at_camera()
    assert math.isclose(camera.width / (2 * math.tan(camera.fov_x / 2)), camera.K[0, 0])


def test_unproject_then_project() -> None:
    camera = look_at_camera()
    point = unproject_z_depth(camera.K, camera.c2w, np.array([10.0, 20.0]), 3.0)
    pixel, depth = project_world_point(camera.K, camera.c2w, point)
    assert np.allclose(pixel, [10.0, 20.0]) and math.isclose(depth, 3.0)


def test_orbit_angles_form_a_closed_uniform_path() -> None:
    angles = uniform_horizontal_angles(81, -20.0, 20.0)
    assert angles[0] == 0.0 and angles[-1] == pytest.approx(0.0)
    assert angles.max() == pytest.approx(20.0) and angles.min() == pytest.approx(-20.0)
    assert np.allclose(np.abs(np.diff(angles)), 1.0)


def test_orbit_keeps_pivot_at_target_pixel_and_radius() -> None:
    camera = look_at_camera()
    pivot = np.array([0.2, 0.1, 0.5])
    pixel, _ = project_world_point(camera.K, camera.c2w, pivot)
    poses = horizontal_orbit(camera.c2w, camera.K, pivot, pixel, np.array([0.0, 15.0, -20.0]), 0.95)
    reference_radius = np.linalg.norm(camera.c2w[:3, 3] - pivot)
    for pose in poses:
        seen, _ = project_world_point(camera.K, pose, pivot)
        assert np.allclose(seen, pixel, atol=1e-6)
        assert np.linalg.norm(pose[:3, 3] - pivot) == pytest.approx(0.95 * reference_radius)
