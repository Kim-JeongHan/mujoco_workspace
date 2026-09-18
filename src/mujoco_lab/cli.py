"""Command-line entry points for simulation, viewing, and rendering."""

import argparse
import json
from pathlib import Path

import mujoco

from mujoco_lab.control import CONTROLLER_NAMES, create_controller, run_steps
from mujoco_lab.environment import ENVIRONMENT_NAMES, create_environment
from mujoco_lab.robot import ROBOT_NAMES, create_robot
from mujoco_lab.simulation import initialize_data, load_simulation


def nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description="MuJoCo workspace")
    commands = parser.add_subparsers(dest="command", required=True)
    simulate = commands.add_parser("simulate", help="Run physics without rendering")
    simulate.add_argument("--steps", type=nonnegative_int, default=1000)
    view = commands.add_parser("view", help="Open an interactive simulation window")
    render = commands.add_parser("render", help="Save an offscreen 640 x 480 PNG")
    render.add_argument("--steps", type=nonnegative_int, default=0)
    render.add_argument("--output", type=Path, default=Path("outputs/scene.png"))
    for command in (simulate, view, render):
        command.add_argument(
            "--controller",
            choices=CONTROLLER_NAMES,
            default="none",
            help="Optional Forte torque controller",
        )
        command.add_argument(
            "--environment",
            choices=ENVIRONMENT_NAMES,
            help="Select an environment, optionally combined with --robot",
        )
        selection = command.add_mutually_exclusive_group()
        selection.add_argument("--robot", choices=ROBOT_NAMES, help="Select a bundled manipulator")
        selection.add_argument("--model", type=Path, help="Load an explicit MJCF/URDF file")
    args = parser.parse_args()
    if not (args.robot or args.model or args.environment):
        parser.error("select --robot, --model, or --environment")
    if args.environment and args.model:
        parser.error("--environment cannot be combined with --model; use --robot for composition")
    if args.robot:
        model, data = create_robot(args.robot, environment=args.environment or "empty")
    elif args.environment:
        model = create_environment(args.environment).compile()
        data = initialize_data(model)
    else:
        model, data = load_simulation(args.model)
    try:
        controller = create_controller(args.controller, model, data)
    except ValueError as error:
        parser.error(str(error))

    if args.command == "view":
        from mujoco_lab.viewer import show

        try:
            show(model, data, controller)
        except KeyboardInterrupt:
            pass
        return

    stats = run_steps(model, data, args.steps, controller)
    if args.command == "render":
        from mujoco_lab.viewer import save_frame

        print(save_frame(model, data, args.output, controller))
        return

    print(
        json.dumps(
            {
                "mujoco_version": mujoco.__version__,
                "steps": args.steps,
                "simulated_seconds": data.time,
                "qpos": data.qpos.tolist(),
                "warnings": int(data.warning.number.sum()),
                "controller": args.controller,
                "saturated_steps": stats.saturated_steps,
                "tracking_error_mean": sum(stats.errors) / len(stats.errors)
                if stats.errors
                else None,
                "tracking_error_max": max(stats.errors, default=None),
                "tracking_error_unit": "rad"
                if args.controller == "pd"
                else ("m" if args.controller == "osc" else None),
            },
            indent=2,
        )
    )
