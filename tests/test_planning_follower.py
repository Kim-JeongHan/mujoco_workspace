"""Joint waypoint execution through native robot actuators."""

import numpy as np

from mujoco_lab import RobotSpec, Simulator, create_environment
from mujoco_lab.control import create_controller
from mujoco_lab.planning.follower import WaypointFollower


def test_panda_follows_waypoint_and_preserves_unplanned_finger_target():
    sim = Simulator(create_environment("empty"), robots=[RobotSpec("arm", "panda")])
    robot = sim.robots["arm"]
    robot.change_controller(create_controller("position", robot))
    names = robot.state.joint_names[:7]
    indices = [sim.model.joint(robot.prefix + name).qposadr[0] for name in names]
    start = sim.data.qpos[indices].copy()
    goal = start.copy()
    goal[1] += 0.08
    finger_target = robot.target.position[-1]
    follower = WaypointFollower(robot, names, np.stack([start, goal]), stop_on_finish=True)
    sim.target_updater = follower

    sim.run_steps(1800)

    assert follower.complete and follower.reached_waypoints == 2
    assert follower.steps < 1800
    np.testing.assert_allclose(sim.data.qpos[indices], goal, atol=0.025)
    assert robot.target.position[-1] == finger_target
    assert np.isfinite(sim.data.qpos).all()
    assert np.isfinite(sim.data.qvel).all()
    assert abs(sim.data.qpos[indices[1]] - start[1]) > 0.05


def test_follower_times_out_without_claiming_success():
    sim = Simulator(create_environment("empty"), robots=[RobotSpec("arm", "panda")])
    robot = sim.robots["arm"]
    robot.change_controller(create_controller("position", robot))
    names = robot.state.joint_names[:7]
    indices = [sim.model.joint(robot.prefix + name).qposadr[0] for name in names]
    start = sim.data.qpos[indices].copy()
    goal = start.copy()
    goal[1] += 0.15
    follower = WaypointFollower(robot, names, np.stack([start, goal]), max_steps=1)
    sim.target_updater = follower

    sim.run_steps(2)

    assert follower.failed and not follower.complete
    assert follower.reached_waypoints == 1
    assert "Execution timeout" in follower.result()
