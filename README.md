# TL;DR

Simulate Panda and Forte manipulators, run cube stacking, collect demonstrations, and train behavior-cloning policies.

## Install

Use Python 3.12 or newer and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --locked
uv sync --locked --extra learning
```

## Run a simulation

```bash
uv run mujoco-lab --robot forte --cubes 2
uv run mujoco-lab --robot forte --cubes 2 --headless --steps 30000
uv run python examples/manipulator.py --robot panda --environment warehouse
```

The CLI supports Panda and Forte, cube-stack scenes, and heuristic or sampling experts. See [Forte asset notes](src/mujoco_lab/assets/robot/forte/README.md) for details about its model and simulated zero pose.

## Collect demonstrations

```bash
uv run python -m mujoco_lab.learning.collect --count 20 --seed 42 --output-dir data/demos/pilot
uv run python -m mujoco_lab.learning.replay --path data/demos/pilot/episode_000000.npz
```

Collection and replay settings are in [`Config`](src/mujoco_lab/learning/collect.py) and [`Config`](src/mujoco_lab/learning/replay.py).

## Download a dataset

Install the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) and run `hf auth login` if the repository is private.

```bash
./scripts/dataset_download.sh YOUR_HF_USERNAME/mujoco-lab-datasets
```

This downloads the repository under `data/`; see [the script](scripts/dataset_download.sh) for options.

## Train and evaluate

```bash
uv run --locked --extra learning python -m mujoco_lab.learning.train
uv run --locked --extra learning python -m mujoco_lab.learning.evaluate --checkpoint log/bc/mse/RUN/checkpoint.pt
```

Replace `RUN` with the training run directory. Training and evaluation settings are in [`TrainConfig` and `EvalConfig`](src/mujoco_lab/learning/config/config.py).

Cube models and stacking layouts derive from OGBench task 5; upstream provenance and licenses are recorded in the [OGBench source manifest](third_party/ogbench.SOURCE.json). Planning code includes upstream components documented in the [planning guide](third_party/planning/README.md) and [planning source manifest](third_party/planning.SOURCE.json).
