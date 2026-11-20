# TL;DR

Simulate Panda and Forte robots, collect manipulation demonstrations, and train policies with behavior cloning, DAPG, or DDPG with HER.

## Install

Use Python 3.12 or newer and [uv](https://docs.astral.sh/uv/).

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

[`examples/simulation.py`](examples/simulation.py) accepts an environment name or MJCF/URDF path, with an optional robot.

Robot YAML settings select `position`, `pd`, or `computed_torque` control and an
`jerk_lqr` or `min_jerk` reference. `Robot.set_goal()` accepts arm joint positions
followed by optional gripper opening width in meters. `Simulator.step()` advances
the reference, controller, and physics together.

Task examples support `heuristic` and `sampling` methods:

```bash
uv run python examples/cube_task.py
uv run python examples/book_task.py
```

## Collect demonstrations

```bash
uv run python -m mujoco_lab.learning.collect_cube
uv run python -m mujoco_lab.learning.collect_book
uv run python -m mujoco_lab.learning.replay --path <PATH>
```

The [cube](src/mujoco_lab/learning/collect_cube.py) and [book](src/mujoco_lab/learning/collect_book.py) collectors save NPZ episodes and support resuming. Training uses successful episodes.

Cube sampling collection supports the shared planner configurations, including RRT*:

```bash
uv run python -m mujoco_lab.learning.collect_cube --method sampling --cubes 4 \
  --count 1000 --seed 30000 --output-dir data/forte_cube4_rrt_star_1000 \
  planning:rrt-star-config --planning.max-iterations 1000 \
  --planning.step-size 0.2 --planning.goal-tolerance 0.04 --planning.seed 7 \
  --planning.no-return-first-solution
```

`--count` includes failed attempts. Disabling `return_first_solution` lets RRT*
continue optimizing within its iteration budget after finding a route.

## Download a dataset

Install the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) and set your token in the project-root `.env` file:

```dotenv
HUGGINGFACE_API_KEY=hf_your_token
```

The dataset scripts load this token automatically.

```bash
./scripts/dataset_download.sh YOUR_HF_USERNAME/mujoco-lab-datasets
```

Downloads are saved under `data/`. See [the script](scripts/dataset_download.sh) for options.

## Train and evaluate

```bash
uv run python -m mujoco_lab.learning.train
uv run python -m mujoco_lab.learning.evaluate --checkpoint <PATH>
uv run python -m mujoco_lab.learning.evaluate_cube --checkpoint <PATH>
uv run python -m mujoco_lab.learning.evaluate_book --checkpoint <PATH>
```

Add `--evaluation.viewer` to watch live policy evaluation. The viewer runs at
real-time speed and closes after the requested episodes; closing its window ends
evaluation early. Use `--evaluation.video-episodes 0` to watch without saving MP4s,
or a positive count to record episodes while viewing.

Cube placement succeeds when all cubes simultaneously have an XY goal error
below 3 cm and a height error below 1.2 cm, touch their designated support, and
have no robot contact. Placement is accepted immediately without a velocity
threshold or stabilization delay.

BC supports `--policy-type mse`, `flow`, and `diffusion`. Diffusion uses an
observation-conditioned temporal U-Net, epsilon-prediction training, and DDIM
sampling. It shares the dataset, normalization, EMA, and action-chunk execution
pipeline with the other BC policies.

Train the hover-5mm PD MSE baselines on Colab with W&B enabled:

```bash
# Export WANDB_API_KEY before launching; optional WANDB_ENTITY and WANDB_PROJECT select the workspace.
bash scripts/colab_mse_train.sh --gpu T4
# Minimum-jerk interpolation with zero desired PD velocity.
bash scripts/colab_mse_minjerk_train.sh --gpu T4
# Constrained quintic interpolation with reference-velocity PD tracking.
bash scripts/colab_mse_quintic_train.sh --gpu L4
```

This uploads the selected dataset and current source, preserves the collection's
ordered seed splits, and applies the recorded PD gains, reference settings, and
desired velocity inside the Colab runtime. Position hold and unconstrained
minimum jerk use zero desired PD velocity. Constrained quintic uses the reference
velocity and enforces motion limits. Each experiment trains for 400 epochs with
batch size 128, observation history 2, and masked action chunks of length 16.
Every 30,000 optimizer steps and at the final step, it evaluates seeds 5000–5002
for up to 15 simulated seconds each and logs three videos to W&B. Results are
downloaded to `log/colab/` before the owned session is stopped. The launcher
must remain running to retrieve results and stop the session automatically.
Use `--session NAME` to reuse an existing session with the same cleanup behavior.

Colab runtimes can terminate before training finishes. Every periodic inference
checkpoint is uploaded to W&B as a model artifact, independently of the final
result download. These checkpoints contain policy weights and normalization;
they do not contain AdamW state or support an exact BC training resume.

For local BC training, `--split-file <DATA_DIR>/splits.json` reuses recorded
train/validation/test seed assignments, omitting unsuccessful attempts without
replacing them or changing the remaining order.

```bash
uv run --extra learning python -m mujoco_lab.learning.train \
  --policy-type diffusion --scheduler.num-train-timesteps 40 --sampling-num-steps 7
```

## DAPG fine-tuning

Fine-tune an MSE BC checkpoint with DAPG.

```bash
uv run python -m mujoco_lab.learning.trainers.train_dapg --init-from <BC_PATH> --data-dir <DATA_DIR>

uv run python -m mujoco_lab.learning.trainers.train_dapg --resume <DAPG_PATH> --total-steps 200000

uv run python -m mujoco_lab.learning.evaluate_dapg --checkpoint <DAPG_PATH>
```

See [`DAPGConfig`](src/mujoco_lab/learning/config/dapg.py) for settings.

## DDPG with hindsight experience replay

Train a bounded, single-action MLP from scratch using the existing robot joint
and gripper targets. HER relabels 80% of sampled transitions with goals achieved
later in the same episode. Set `--her.ratio 0` for the DDPG-only baseline.

```bash
uv run --extra learning python -m mujoco_lab.learning.trainers.train_ddpg --robot forte --cubes 1

uv run --extra learning python -m mujoco_lab.learning.trainers.train_ddpg --robot forte --cubes 2 --her.ratio 0

uv run --extra learning python -m mujoco_lab.learning.trainers.train_ddpg --scene book_insertion --book small --max-seconds 180 --evaluation.max-seconds 180

uv run --extra learning python -m mujoco_lab.learning.trainers.train_ddpg --resume <DDPG_PATH> --total-steps 200000

uv run --extra learning python -m mujoco_lab.learning.evaluate_ddpg --checkpoint <DDPG_PATH>
```
