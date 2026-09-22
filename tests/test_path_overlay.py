import mujoco
import numpy as np

from mujoco_lab import RobotSpec, Simulator, create_environment
from mujoco_lab.rendering.path_overlay import PLANNED_RGBA, TRAIL_RGBA, PathOverlay
from mujoco_lab.utils import Transform


def test_planned_fk_uses_scoped_joints_and_preserves_live_data():
    pose = Transform.from_pose_mmdeg([230, -110, 340, 0, 0, 35])
    sim = Simulator(create_environment("empty"), robots=[RobotSpec("arm", "panda", pose)])
    robot = sim.robots["arm"]
    names = robot.state.joint_names[:7]
    indices = [sim.model.joint(robot.prefix + name).qposadr[0] for name in names]
    start = sim.data.qpos[indices].copy()
    goal = start.copy()
    goal[0] += 0.25
    initial_qpos = sim.data.qpos.copy()
    initial_qvel = sim.data.qvel.copy()
    initial_ctrl = sim.data.ctrl.copy()
    overlay = PathOverlay(robot, names, [start, goal], frame="grasp")

    scratch = mujoco.MjData(sim.model)
    mujoco.mj_copyData(scratch, sim.model, sim.data)
    scratch.qpos[indices] = goal
    mujoco.mj_kinematics(sim.model, scratch)
    site = sim.model.site("arm/grasp").id
    np.testing.assert_allclose(overlay.planned_xyz[0], sim.data.site_xpos[site])
    np.testing.assert_allclose(overlay.planned_xyz[-1], scratch.site_xpos[site])
    np.testing.assert_array_equal(sim.data.qpos, initial_qpos)
    np.testing.assert_array_equal(sim.data.qvel, initial_qvel)
    np.testing.assert_array_equal(sim.data.ctrl, initial_ctrl)

    scene = mujoco.MjvScene(sim.model, 5)
    overlay(scene, sim.data)
    assert scene.ngeom == scene.maxgeom
    assert overlay.trail_xyz.shape == (1, 3)
    np.testing.assert_allclose(overlay.current_xyz, sim.data.site_xpos[site])
    np.testing.assert_array_equal(sim.data.qpos, initial_qpos)

    overlay.record(scratch)
    complete_scene = mujoco.MjvScene(sim.model, 100)
    overlay(complete_scene, scratch)
    capsules = [
        geom
        for geom in complete_scene.geoms[: complete_scene.ngeom]
        if geom.type == mujoco.mjtGeom.mjGEOM_CAPSULE
    ]
    assert any(np.allclose(geom.rgba, PLANNED_RGBA) for geom in capsules)
    assert any(np.allclose(geom.rgba, TRAIL_RGBA) for geom in capsules)
    assert overlay.floor_z == -0.005
    assert any(np.isclose(geom.pos[2], overlay.floor_z + 0.016) for geom in capsules)
    assert any(np.isclose(geom.pos[2], overlay.floor_z + 0.040) for geom in capsules)
    assert "XY floor paths" in [geom.label for geom in complete_scene.geoms[: complete_scene.ngeom]]
