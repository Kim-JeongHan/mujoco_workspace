import subprocess
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from mujoco_lab.utils import Transform

RZ_90 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
RX_90 = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])


def test_identity_and_matrix_round_trip():
    np.testing.assert_array_equal(Transform.identity().as_matrix(), np.eye(4))
    matrix = np.array([[0, -1, 0, 1], [1, 0, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]])
    transform = Transform.from_matrix(matrix)
    np.testing.assert_allclose(transform.as_matrix(), matrix, atol=1e-14)
    np.testing.assert_allclose((Transform() @ transform).as_matrix(), matrix, atol=1e-14)
    np.testing.assert_allclose((transform @ Transform()).as_matrix(), matrix, atol=1e-14)


@pytest.mark.parametrize("shape", [(3,), (5, 3), (2, 4, 3), (0, 3)])
def test_points_and_free_vectors_have_distinct_translation_semantics(shape):
    transform = Transform(rotation=RZ_90, translation=[1, 2, 3])
    points = np.arange(np.prod(shape)).reshape(shape)
    rotated = np.stack([-points[..., 1], points[..., 0], points[..., 2]], axis=-1)
    np.testing.assert_allclose(transform.apply(points), rotated + [1, 2, 3], atol=1e-14)
    np.testing.assert_allclose(transform.apply_vectors(points), rotated, atol=1e-14)


def test_composition_order_matches_homogeneous_matrix_product():
    t_ab = Transform(rotation=RZ_90, translation=[1, 2, 3])
    t_bc = Transform(rotation=RX_90, translation=[-2, 4, 1])
    points = [[1, 2, 3], [-1, 0, 4]]
    t_ac = t_ab @ t_bc
    np.testing.assert_allclose(t_ac.as_matrix(), t_ab.as_matrix() @ t_bc.as_matrix(), atol=1e-14)
    np.testing.assert_allclose(t_ac.apply(points), t_ab.apply(t_bc.apply(points)), atol=1e-14)
    assert not np.array_equal(t_ac.as_matrix(), (t_bc @ t_ab).as_matrix())


def test_inverse_matches_matrix_inverse_and_recovers_points():
    angle = 0.7
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    )
    transform = Transform(rotation=rotation, translation=[0.2, -4.3, 1.8])
    points = np.random.default_rng(3).normal(size=(10, 3))
    np.testing.assert_allclose(
        transform.inverse().as_matrix(), np.linalg.inv(transform.as_matrix()), atol=1e-14
    )
    np.testing.assert_allclose(
        transform.inverse().apply(transform.apply(points)), points, atol=1e-14
    )
    np.testing.assert_allclose((transform.inverse() @ transform).as_matrix(), np.eye(4), atol=1e-14)


def test_transform_is_a_snapshot_of_inputs_and_conversion_results():
    rotation = RZ_90.astype(float)
    translation = np.array([1.0, 2.0, 3.0])
    transform = Transform(rotation, translation)
    expected = transform.as_matrix()
    rotation[:] = 0
    translation[:] = 0
    transform.as_rotation().as_matrix()[:] = 0
    transform.as_translation()[:] = 0
    transform.as_pose_mrad()[:] = 0
    transform.as_pose_mmrad()[:] = 0
    transform.as_xyqquat()[:] = 0
    transform.as_mmdeg()[:] = 0
    transform.as_matrix()[:] = 0
    np.testing.assert_array_equal(transform.as_matrix(), expected)
    restored = Transform.from_matrix(expected)
    expected[:] = 0
    np.testing.assert_allclose(restored.as_matrix(), transform.as_matrix(), atol=1e-14)


def test_float32_rotation_is_normalized_by_scipy():
    angle = np.float32(0.7)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]],
        dtype=np.float32,
    )
    transform = Transform(rotation=rotation)
    actual = transform.as_rotation().as_matrix()
    np.testing.assert_allclose(actual, rotation, atol=1e-7)
    np.testing.assert_allclose(actual.T @ actual, np.eye(3), atol=1e-14)
    assert actual.dtype == np.float64


def test_accepts_scipy_rotation_and_exposes_rotation_conversions():
    rotation = Rotation.from_euler("z", 90, degrees=True)
    transform = Transform(rotation=rotation, translation=[1, 2, 3])
    assert isinstance(transform.as_rotation(), Rotation)
    np.testing.assert_allclose(transform.as_rotation().as_rotvec(), [0, 0, np.pi / 2], atol=1e-14)
    np.testing.assert_allclose(transform.apply([1, 0, 0]), [1, 3, 3], atol=1e-14)


def test_rejects_a_stack_of_scipy_rotations():
    with pytest.raises(ValueError, match="single rotation"):
        Transform(rotation=Rotation.identity(2))


def test_pose_exports_use_fixed_axis_rpy_and_requested_units():
    roll, pitch, yaw = np.deg2rad([20, -30, 40])
    rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    ry = np.array(
        [[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]]
    )
    rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    transform = Transform(rotation=rz @ ry @ rx, translation=[0.125, -0.25, 0.8])
    np.testing.assert_allclose(
        transform.as_pose_mrad(), [0.125, -0.25, 0.8, roll, pitch, yaw], atol=1e-14
    )
    np.testing.assert_allclose(
        transform.as_pose_mmrad(), [125, -250, 800, roll, pitch, yaw], atol=1e-14
    )
    np.testing.assert_allclose(transform.as_mmdeg(), [125, -250, 800, 20, -30, 40], atol=1e-12)
    np.testing.assert_array_equal(transform.as_translation(), [0.125, -0.25, 0.8])


def test_quaternion_pose_uses_meters_and_mujoco_scalar_first_order():
    quat = np.array([1, 2, 3, 4]) / np.sqrt(30)
    transform = Transform(
        rotation=Rotation.from_quat(quat, scalar_first=True), translation=[0.125, -0.25, 0.8]
    )
    result = transform.as_xyqquat()
    assert result.shape == (7,)
    np.testing.assert_allclose(result, [0.125, -0.25, 0.8, *quat], atol=1e-14)


def test_pose_factories_convert_units_and_round_trip_rpy():
    pose_mmdeg = np.array([125, -250, 800, 20, -30, 40], dtype=float)
    pose_mrad = np.array([0.125, -0.25, 0.8, *np.deg2rad([20, -30, 40])])
    from_mmdeg = Transform.from_pose_mmdeg(pose_mmdeg)
    from_mrad = Transform.from_pose_mrad(pose_mrad)
    np.testing.assert_allclose(from_mmdeg.as_matrix(), from_mrad.as_matrix(), atol=1e-14)
    for transform in (from_mmdeg, from_mrad):
        np.testing.assert_allclose(transform.as_mmdeg(), pose_mmdeg, atol=1e-12)
        np.testing.assert_allclose(transform.as_pose_mrad(), pose_mrad, atol=1e-14)
    pose_mmdeg[:] = 0
    pose_mrad[:] = 0
    np.testing.assert_allclose(from_mmdeg.as_translation(), [0.125, -0.25, 0.8], atol=1e-14)
    np.testing.assert_allclose(from_mrad.as_translation(), [0.125, -0.25, 0.8], atol=1e-14)


def test_transform_utils_import_does_not_load_a_simulator(tmp_path):
    code = """
import sys
from mujoco_lab.utils import Transform
assert Transform().apply([1, 2, 3]).tolist() == [1, 2, 3]
assert not {'mujoco', 'mujoco_warp', 'isaaclab', 'isaacsim', 'matplotlib'} & sys.modules.keys()
"""
    subprocess.run([sys.executable, "-c", code], cwd=tmp_path, check=True)
