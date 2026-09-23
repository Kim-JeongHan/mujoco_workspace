# MuJoCo Lab Demonstration Episodes

This repository contains simulated robot manipulation episodes collected with
the MuJoCo Lab workspace. The available directories contain demonstrations for
cube stacking with a Forte robot; see each directory's episode metadata for its
collection settings and recorded outcomes.

## Format

Each `*.npz` file stores one episode. The required arrays are:

- `states`: observations with shape `(T + 1, state_dim)`.
- `actions`: physical action targets with shape `(T, action_dim)`.
- `metadata`: a scalar JSON string with episode details such as seed, success,
  scene, and simulation settings.

Optional arrays include per-action `rewards`, `terminated`, and `truncated`
values. Episodes with replay data may also contain `qpos`, `frame_times`,
`mocap_pos`, and `mocap_quat`, each recorded at `T + 1` frames. Exact feature
and action dimensions can vary by task; metadata and the workspace's collection
code describe the corresponding setup.

Load episodes with the workspace's `mujoco_lab.learning.datasets.load_episodes`
API. It keeps successful episodes by default; set `success_only=False` to
include failed and unlabeled attempts.
