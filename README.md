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

See [`TrainConfig`, `EvalConfig`, and `RolloutConfig`](src/mujoco_lab/learning/config/config.py) for settings.

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
