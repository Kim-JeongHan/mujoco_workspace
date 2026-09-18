import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from mujoco_lab.utils import Transform


@pytest.fixture
def pyplot():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    yield plt
    plt.close("all")


def test_frame_plot_and_quaternion_export_match_mujoco_body(pyplot, monkeypatch, tmp_path):
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body name="frame" pos="0.2 -0.3 0.4" quat="2 3 5 7">'
        '<geom type="sphere" size="0.01"/></body></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    body = data.body("frame")
    rotation = body.xmat.reshape(3, 3)
    transform = Transform(rotation=rotation, translation=body.xpos)
    pose = transform.as_xyqquat()
    np.testing.assert_allclose(pose[:3], body.xpos, atol=1e-14)
    np.testing.assert_allclose(pose[3:], body.xquat, atol=1e-14)
    np.testing.assert_allclose(
        Rotation.from_quat(pose[3:], scalar_first=True).as_matrix(), rotation, atol=1e-14
    )
    calls = []
    monkeypatch.setattr(pyplot, "show", lambda: calls.append(True))
    ax = transform.plot(label="frame", length=0.4, show=False)
    assert not calls
    for index, artist in enumerate(ax.collections):
        tip, origin = artist._segments3d[0]
        np.testing.assert_allclose(origin, body.xpos, atol=1e-14)
        np.testing.assert_allclose(tip, body.xpos + rotation[:, index] * 0.4, atol=1e-14)
    assert ax.get_legend_handles_labels()[1] == ["frame_x", "frame_y", "frame_z"]
    assert [text.get_text() for text in ax.texts] == ["frame"]
    image = tmp_path / "frame.png"
    ax.figure.savefig(image)
    assert image.stat().st_size > 1000


def test_plot_keeps_both_frames_visible_with_equal_scale(pyplot, monkeypatch):
    calls = []
    monkeypatch.setattr(pyplot, "show", lambda: calls.append(True))
    first = Transform.identity()
    second = Transform(
        rotation=Rotation.from_euler("xyz", [20, -30, 40], degrees=True),
        translation=[5, -3, 2],
    )
    ax = first.plot(label="world", length=0.3, show=False)
    ax.view_init(elev=35, azim=20)
    ax.set_autoscale_on(False)
    assert second.plot(ax=ax, label="tool", length=0.8) is ax
    ax.figure.canvas.draw()
    assert not calls
    assert len(ax.collections) == 6
    assert ax.elev == 35 and ax.azim == 20
    bounds = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    for transform, length in [(first, 0.3), (second, 0.8)]:
        points = transform.apply(np.vstack((np.zeros(3), np.eye(3) * length)))
        assert np.all(points >= bounds[:, 0])
        assert np.all(points <= bounds[:, 1])
    np.testing.assert_allclose(np.ptp(bounds, axis=1), np.ptp(bounds[0]))
    np.testing.assert_allclose(ax.get_box_aspect(), np.full(3, ax.get_box_aspect()[0]))


def test_plot_show_defaults_and_override(pyplot, monkeypatch):
    calls = []
    monkeypatch.setattr(pyplot, "show", lambda: calls.append(True))
    ax = Transform.identity().plot()
    assert calls == [True]
    Transform.identity().plot(ax=ax)
    assert calls == [True]
    Transform.identity().plot(ax=ax, show=True)
    assert calls == [True, True]


def test_rpy_export_matches_mjcf_fixed_axis_xyz():
    transform = Transform(
        rotation=Rotation.from_euler("xyz", [0.3, -0.4, 0.5]), translation=[0.2, -0.3, 0.4]
    )
    pose = transform.as_pose_mrad()
    position = " ".join(map(str, pose[:3]))
    angles = " ".join(map(str, pose[3:]))
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><compiler angle="radian" eulerseq="XYZ"/><worldbody>'
        f'<body name="frame" pos="{position}" euler="{angles}">'
        '<geom type="sphere" size="0.01"/></body></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.body("frame").xpos, transform.as_translation(), atol=1e-14)
    np.testing.assert_allclose(
        data.body("frame").xmat.reshape(3, 3), transform.as_rotation().as_matrix(), atol=1e-14
    )
