"""Run a manipulator on CUDA: uv run python examples/warp_manipulator.py --robot panda."""

import argparse

import mujoco_warp as mjw
import numpy as np
import warp as wp

from mujoco_lab import ENVIRONMENT_NAMES, ROBOT_NAMES, create_robot


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate manipulator worlds with MuJoCo Warp")
    parser.add_argument("--robot", choices=ROBOT_NAMES, required=True)
    parser.add_argument("--environment", choices=ENVIRONMENT_NAMES, default="empty")
    parser.add_argument("--worlds", type=int, default=1)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    if args.worlds < 1 or args.steps < 0:
        parser.error("--worlds must be positive and --steps must be zero or greater")

    wp.init()
    with wp.ScopedDevice("cuda:0"):
        model, data = create_robot(args.robot, environment=args.environment)
        warp_model = mjw.put_model(model)
        warp_data = mjw.put_data(model, data, nworld=args.worlds, nconmax=32, njmax=128)
        with wp.ScopedCapture() as capture:
            mjw.step(warp_model, warp_data)
        for _ in range(args.steps):
            wp.capture_launch(capture.graph)
        wp.synchronize()
        positions = warp_data.qpos.numpy()
        times = warp_data.time.numpy()
    assert np.isfinite(positions).all()
    assert np.allclose(times, args.steps * model.opt.timestep)
    print(f"Robot: {args.robot}; environment: {args.environment}; device: {warp_data.qpos.device}")
    print(f"Simulated seconds per world: {times}")
    print(f"Joint positions per world:\n{positions}")


if __name__ == "__main__":
    main()
