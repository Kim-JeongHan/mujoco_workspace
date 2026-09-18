"""Select a manipulator: uv run python examples/manipulator.py --robot panda."""

import argparse

from mujoco_lab import ENVIRONMENT_NAMES, ROBOT_NAMES, create_robot
from mujoco_lab.control import CONTROLLER_NAMES, create_controller, run_steps
from mujoco_lab.viewer import show


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a manipulator created by the robot factory")
    parser.add_argument("--robot", choices=ROBOT_NAMES, required=True)
    parser.add_argument("--environment", choices=ENVIRONMENT_NAMES, default="empty")
    parser.add_argument("--controller", choices=CONTROLLER_NAMES, default="none")
    parser.add_argument(
        "--headless", action="store_true", help="Run physics without opening a window"
    )
    parser.add_argument("--steps", type=int, default=1000, help="Physics steps in headless mode")
    args = parser.parse_args()
    if args.steps < 0:
        parser.error("--steps must be zero or greater")

    model, data = create_robot(args.robot, environment=args.environment)
    try:
        controller = create_controller(args.controller, model, data)
    except ValueError as error:
        parser.error(str(error))
    if args.headless:
        stats = run_steps(model, data, args.steps, controller)
        print(
            f"{args.robot}: simulated {data.time:.3f} seconds; "
            f"environment = {args.environment}; qpos = {data.qpos}"
        )
        if controller is not None:
            print(stats.describe("tracking error", "rad" if args.controller == "pd" else "m"))
        return

    show(model, data, controller)


if __name__ == "__main__":
    main()
