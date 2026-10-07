# TL;DR

Simulate Panda and Forte manipulators, run cube stacking or book insertion, collect demonstrations, and train behavior-cloning policies.

## Install

Use Python 3.12 or newer and [uv](https://docs.astral.sh/uv/).

Choose one installation command. The learning installation includes the base
simulation dependencies and adds support for collection, replay, training, and
evaluation.

```bash
# Simulation only
uv sync --locked
# Simulation and learning
uv sync --locked --extra learning
```

## Run a simulation

```bash
uv run mujoco-lab
uv run python examples/simulation.py
```

The CLI supports Panda and Forte, cube-stack scenes, and heuristic or sampling experts.

`examples/simulation.py` accepts a registered environment name or an MJCF/URDF
file path. The robot is optional; selecting one attaches its YAML-configured
controller and holds its initial default pose.

Every robot configuration must define `pose.default` in `robot.yaml`. Simulator uses this pose for initialization, and `reset()` restores it. Arm angles in YAML use degrees and the gripper opening width uses meters (zero closed); dependent finger joints follow their equality constraints. Robot initialization does not read an XML `home` pose.

`pose.zero` is optional and exposes a second pose in the collision viewer. These
are the two supported pose names. `PoseConfig.from_yaml()` converts YAML arm
angles to radians; constructing `PoseConfig` directly uses radians.

The task examples use the same planner configuration as the CLI:

```bash
uv run python examples/cube_task.py
uv run python examples/book_task.py
```

Both task examples support `heuristic` and `sampling` methods. Cube and book sampling accept
RRT Connect, RRT, PRM, RRT Star, PRM Star, and RRG planner configurations.

Book heuristic execution follows the book recipe with Cartesian IK paths and
collision checks.

Cube and book recipes specify `arm` and `gripper` as
`[velocity_ratio, acceleration_ratio]` pairs. Each fraction is in `(0, 1]`
and scales the corresponding joint limits in `robot.yaml`; `[0.9, 1.0]`
uses 90% of the configured velocity limit and 100% of the acceleration limit.
Top-level pairs default to `[1.0, 1.0]`. A stage inherits an omitted pair
and replaces the whole pair when it supplies one:

```yaml
arm: [0.9, 1.0]
gripper: [1.0, 1.0]
stages:
  - name: place
    offset_xyz_m: [0.0, 0.0, 0.035]
    gripper_mode: 1
    arm: [0.3, 0.2]
```

`JointTrajectory` applies the ratios when calculating timing for both continuous
and stop-at-waypoint interpolation. Cube motions that merge recipe stages use
the lowest velocity and acceleration ratios of their members. Bundled Panda
ratios conservatively retain the previous absolute task caps; because Panda's
joint limits differ, a uniform ratio can slow some joints further.

Both cube and book recipes define `gripper_mode` for every stage: `1` closes
to the robot's lower opening-width limit, and `0` fully opens. Targets sent to
controllers and stored in datasets remain opening widths in meters. Closed-mode
gripper-only stages wait for physical finger contact before completing.
Closed-mode arm motion requires contact before starting and monitors grasp loss
during execution. Contact loss lasting longer than `lost_grasp_grace_s` fails
the stage, and closed-mode stages cannot finish while contact is missing.

## Collect demonstrations

```bash
uv run python -m mujoco_lab.learning.collect_cube
uv run python -m mujoco_lab.learning.collect_book
uv run python -m mujoco_lab.learning.replay --path <PATH>
```

[`collect_cube.py`](src/mujoco_lab/learning/collect_cube.py) and [`collect_book.py`](src/mujoco_lab/learning/collect_book.py) save demonstrations in the same compressed NPZ format.
Episodes include the recorded scene and physics state for playback with [`replay.py`](src/mujoco_lab/learning/replay.py).
Collection supports resuming and records both successful and failed attempts; training loads successful episodes.
Resuming requires matching scene, action timing, randomization settings, and model hash.
Auxiliary metadata such as `visual_sha256` and the MuJoCo version may differ.

## Download a dataset

Install the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli)
and set `HUGGINGFACE_API_KEY` in the project-root `.env` file using a
shell-compatible assignment:

```dotenv
HUGGINGFACE_API_KEY=hf_your_token
```

Both dataset scripts load this token and pass it through `HF_TOKEN`, so
`hf auth login` is unnecessary. Downloads require read access; uploads require
write access to the dataset repository.

```bash
./scripts/dataset_download.sh YOUR_HF_USERNAME/mujoco-lab-datasets
```

This downloads the repository under `data/`; see [the script](scripts/dataset_download.sh) for options.

## Train and evaluate

```bash
uv run python -m mujoco_lab.learning.train
uv run python -m mujoco_lab.learning.evaluate --checkpoint <PATH>
uv run python -m mujoco_lab.learning.evaluate_cube --checkpoint <PATH>
uv run python -m mujoco_lab.learning.evaluate_book --checkpoint <PATH>
```

[`TrainConfig`, `EvalConfig`, and their shared `RolloutConfig`](src/mujoco_lab/learning/config/config.py) define training and evaluation settings.

Run training and evaluation from the project root. Their logger automatically
loads `.env` from the working directory, preserving existing environment
variables. On each server, put your W&B API key in this file:

```dotenv
WANDB_API_KEY=your_api_key
```

The commands above use the key without a separate `source .env` or `wandb login`.
The `.env` file is ignored by Git; copy or create it separately on the server.
