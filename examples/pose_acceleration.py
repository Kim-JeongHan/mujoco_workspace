"""Estimate stationary arm acceleration limits at named, custom, and sampled poses.

Run: uv run python examples/pose_acceleration.py --samples 100
The model's arm torque limits are used unless --torque-limits supplies joint-side
limits in N.m. These instantaneous estimates exclude payload, contact reactions,
position-limit reactions, and hardware transmission coupling omitted by the MJCF.
Finger accelerations are held at zero. No robot settings are written.
"""

import csv
import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import mujoco
import numpy as np
import tyro

from mujoco_lab import RobotSpec, Simulator, create_environment
from mujoco_lab.assets.loader import load_robot_config
from mujoco_lab.utils import Transform


@dataclass
class Config:
    poses: tuple[str, ...] = ("zero", "default")
    pose_deg: tuple[float, ...] | None = None
    samples: int = 0
    seed: int = 0
    gripper_width: float = 0.0  # Total opening width in meters.
    torque_limits: tuple[float, ...] | None = None  # Symmetric joint-side limits in N.m.
    output: Path = Path("log/diagnostics/pose_acceleration")


@dataclass
class AccelerationLimits:
    """Symmetric acceleration bounds in rad/s^2, including static holding feasibility."""

    can_hold: bool
    isolated: np.ndarray  # Other arm and finger accelerations are zero.
    simultaneous: np.ndarray  # All combinations of arm acceleration signs are feasible.


def solve_acceleration(mass: np.ndarray, bias: np.ndarray, torque: np.ndarray) -> np.ndarray:
    """Solve M @ acceleration = torque - bias without forming a matrix inverse."""
    return np.linalg.solve(mass, torque - bias)


def acceleration_limits(
    mass: np.ndarray, bias: np.ndarray, torque_limits: np.ndarray
) -> AccelerationLimits:
    """Find isolated-axis limits, then shrink them into one simultaneous feasible box.

    For either sign of an isolated acceleration a_i, every torque row must satisfy
    abs(bias_j) + abs(M_ji) * a_i <= torque_limit_j. For simultaneous acceleration,
    abs(bias) + abs(M) @ bounds <= torque_limits covers every sign combination.
    The resulting box is one conservative choice, not a unique maximum box.
    """
    margin = torque_limits - np.abs(bias)
    if np.any(margin < 0):
        return AccelerationLimits(False, np.zeros_like(bias), np.zeros_like(bias))
    isolated = np.divide(
        margin[:, None],
        np.abs(mass),
        out=np.full_like(mass, np.inf),
        where=mass != 0,
    ).min(axis=0)
    demand = np.abs(mass) @ isolated
    active = demand > 0
    scale = min(1.0, float(np.min(margin[active] / demand[active]))) if np.any(active) else 1.0
    return AccelerationLimits(True, isolated, isolated * scale)


def main(config: Config) -> None:
    if config.samples < 0:
        raise ValueError("samples must be nonnegative")
    robot_config = load_robot_config("forte")
    constraints = robot_config.constraints
    if constraints is None:
        raise ValueError("The example requires named arm constraints")
    names = constraints.joint_names
    count = len(names)
    bounds = np.asarray(constraints.position_limit)
    poses = {
        name: np.asarray(values[:count]) for name, values in robot_config.pose.named_poses().items()
    }
    selected: list[tuple[str, np.ndarray]] = []
    for name in config.poses:
        if name not in poses:
            raise ValueError(f"Unknown pose {name!r}; choose from {tuple(poses)}")
        selected.append((name, poses[name]))
    if config.pose_deg is not None:
        angles = np.asarray(config.pose_deg)
        if angles.shape != (count,):
            raise ValueError(f"pose-deg must contain {count} arm angles")
        selected.append(("custom", np.deg2rad(angles)))
    rng = np.random.default_rng(config.seed)
    selected.extend(
        (f"sample_{index:04d}", rng.uniform(bounds[:, 0], bounds[:, 1]))
        for index in range(config.samples)
    )
    if not selected:
        raise ValueError("Select at least one named, custom, or sampled pose")
    for name, angles in selected:
        if not np.all(np.isfinite(angles)):
            raise ValueError(f"Pose {name!r} must contain finite arm angles")
        if np.any(angles < bounds[:, 0]) or np.any(angles > bounds[:, 1]):
            raise ValueError(f"Pose {name!r} exceeds configured joint bounds")

    simulator = Simulator(
        create_environment("empty"),
        robots=[RobotSpec("arm", "forte", Transform(translation=[0, 0, 0.8]), config=robot_config)],
    )
    robot = simulator.robots["arm"]
    if robot.gripper is not None:
        robot.gripper.apply_initial_width(config.gripper_width)
    slots = [robot.state.joint_names.index(name) for name in names]
    qpos = [robot.state.qpos_indices[slot] for slot in slots]
    dofs = [robot.state.dof_indices[slot] for slot in slots]
    actuators = [simulator.model.actuator(f"arm/{name}_motor").id for name in names]
    if config.torque_limits is None:
        limits = simulator.model.actuator_ctrlrange[actuators]
        gains = simulator.model.actuator_gainprm[actuators, 0]
        gears = simulator.model.actuator_gear[actuators, 0]
        if not np.all(simulator.model.actuator_ctrllimited[actuators]):
            raise ValueError("Supply --torque-limits when an arm motor has no control limit")
        torque_limits = np.min(np.abs(limits * (gains * gears)[:, None]), axis=1)
    else:
        torque_limits = np.asarray(config.torque_limits)
    if (
        torque_limits.shape != (count,)
        or not np.all(np.isfinite(torque_limits))
        or np.any(torque_limits <= 0)
    ):
        raise ValueError(f"torque-limits must contain {count} positive joint-side torque values")

    print(f"Robot: forte; joint order: {', '.join(names)}")
    print(f"Joint-side torque limits (N.m): {torque_limits.tolist()}")
    print("Stationary, empty-hand estimates; finger accelerations fixed at zero.")
    print("Sampled poses are not filtered for collisions.")
    signs = np.array(list(product((-1.0, 1.0), repeat=count)))
    results = []
    rows = []
    for name, angles in selected:
        simulator.data.qpos[qpos] = angles
        simulator.data.qvel[:] = 0
        mujoco.mj_forward(simulator.model, simulator.data)
        mass = robot.state.get_mass_matrix(slots)
        bias = simulator.data.qfrc_bias[dofs] - simulator.data.qfrc_passive[dofs]
        limits = acceleration_limits(mass, bias, torque_limits)
        # Recover every feasible corner through the linear dynamics equation.
        corners = signs * limits.simultaneous
        torques = corners @ mass.T + bias
        recovered = solve_acceleration(mass, bias[:, None], torques.T).T
        np.testing.assert_allclose(recovered, corners, atol=1e-9)
        if limits.can_hold:
            np.testing.assert_array_less(np.max(np.abs(torques), axis=0), torque_limits + 1e-9)
        if not name.startswith("sample_"):
            print(f"\nPose: {name}; arm degrees: {np.rad2deg(angles).round(2).tolist()}")
            print("Joint                    Bias(N.m)  Isolated(rad/s^2)  Simultaneous(rad/s^2)")
            for index, joint in enumerate(names):
                print(
                    f"{joint:24} {bias[index]:9.3f} {limits.isolated[index]:18.3f} "
                    f"{limits.simultaneous[index]:22.3f}"
                )
            if not limits.can_hold:
                print("Static holding is infeasible; no symmetric acceleration box exists.")
        results.append(
            {
                "pose": name,
                "arm_degrees": np.rad2deg(angles).tolist(),
                "can_hold": limits.can_hold,
                "mass_matrix_kg_m2": mass.tolist(),
                "bias_torque_Nm": bias.tolist(),
                "isolated_acceleration_rad_s2": limits.isolated.tolist(),
                "simultaneous_acceleration_rad_s2": limits.simultaneous.tolist(),
            }
        )
        rows.extend(
            (
                name,
                joint,
                float(np.rad2deg(angles[index])),
                bool(limits.can_hold),
                float(bias[index]),
                float(torque_limits[index]),
                float(limits.isolated[index]),
                float(limits.simultaneous[index]),
            )
            for index, joint in enumerate(names)
        )
    minimum = np.min([result["simultaneous_acceleration_rad_s2"] for result in results], axis=0)
    print(f"\nMinimum simultaneous limits over {len(results)} poses (rad/s^2):")
    print(dict(zip(names, minimum.tolist(), strict=True)))
    config.output.mkdir(parents=True, exist_ok=True)
    with (config.output / "poses.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "pose",
                "joint",
                "position_deg",
                "can_hold",
                "bias_Nm",
                "torque_limit_Nm",
                "isolated_acceleration_rad_s2",
                "simultaneous_acceleration_rad_s2",
            ]
        )
        writer.writerows(rows)
    report = {
        "robot": "forte",
        "joint_names": names,
        "torque_limits_Nm": torque_limits.tolist(),
        "torque_source": "model" if config.torque_limits is None else "explicit_joint_limits",
        "gripper_width_m": config.gripper_width,
        "seed": config.seed,
        "assumptions": [
            "Zero velocity, no payload; finger accelerations held at zero.",
            "Contact and position-limit reactions are excluded; poses are not collision-filtered.",
            "The current model's inertia and direct joint transmissions are used.",
            "The simultaneous limits form one feasible box at each sampled state.",
        ],
        "minimum_simultaneous_acceleration_rad_s2": minimum.tolist(),
        "poses": results,
    }
    (config.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Results saved to {config.output}")


if __name__ == "__main__":
    main(tyro.cli(Config))
