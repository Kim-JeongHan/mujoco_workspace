# TL;DR

Simulate Panda and Forte robots, collect manipulation demonstrations, and train policies with behavior cloning or DAPG.

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
`analytic_jerk` or `min_jerk` reference. `Robot.set_goal()` accepts arm joint positions
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

BC supports `--policy-type mse`, `flow`, and `diffusion`. Diffusion uses an
observation-conditioned temporal U-Net, epsilon-prediction training, and DDIM
sampling. It shares the dataset, normalization, EMA, and action-chunk execution
pipeline with the other BC policies.

```bash
uv run python -m mujoco_lab.learning.train --policy-type diffusion \
  --data-dir data/forte_mixed_1 --run-name forte-diffusion-cube1 \
  --diffusion-num-train-timesteps 100 --sampling-num-steps 20
```

`sampling_num_steps` controls Flow Euler steps or Diffusion DDIM steps. Diffusion
defaults to a cosine noise schedule, U-Net widths `(128, 256, 512)`, observation
history 2, action chunk 16, and execution horizon 4. DDIM uses fresh initial
Gaussian noise and `eta=0`; standardized actions are not clipped to `[-1, 1]`.
Checkpoints preserve the U-Net architecture and complete scheduler settings;
evaluation uses the saved sampling step count. Diffusion validation loss measures
noise prediction, while rollout success measures physical task performance.

See [`BCTrainConfig` and `BCEvalConfig`](src/mujoco_lab/learning/config/bc.py) and
[`EvaluationConfig`](src/mujoco_lab/learning/config/evaluation.py) for settings.

Run from the project root. Training and evaluation load W&B credentials from `.env`:

```dotenv
WANDB_API_KEY=your_api_key
```

The `.env` file is ignored by Git.

## DAPG fine-tuning

Fine-tune an MSE BC checkpoint with DAPG.

```bash
uv run python -m mujoco_lab.learning.trainers.train_dapg --init-from <BC_PATH> --data-dir <DATA_DIR>

uv run python -m mujoco_lab.learning.trainers.train_dapg --resume <DAPG_PATH> --total-steps 200000

uv run python -m mujoco_lab.learning.evaluate_dapg --checkpoint <DAPG_PATH>
```

See [`DAPGConfig`](src/mujoco_lab/learning/config/dapg.py) for settings.
