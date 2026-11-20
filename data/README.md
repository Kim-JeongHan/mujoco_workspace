# MuJoCo Lab Demonstration Episodes

This repository contains simulated robot manipulation episodes collected with
the MuJoCo Lab workspace. The available directories contain demonstrations for
cube stacking with a Forte robot; see each directory's episode metadata for its
collection settings and recorded outcomes.

## Joint coordinate convention

These datasets use the Forte coordinate convention adopted on 2026-10-16.
In the extended zero pose, positive shoulder yaw follows the base +Z axis and
positive shoulder, elbow, and wrist pitch follow the +Y direction, using the
right-hand rule. Body-frame origins and orientations match the updated model.

The arm joint order is shoulder yaw, shoulder pitch, shoulder roll, elbow pitch,
lower-arm roll, wrist pitch, and wrist roll. Relative to the previous convention,
shoulder yaw and elbow pitch angles, velocities, and position targets change
sign. The migration updates:

- `states` columns 0 and 3 (angles), and 7 and 10 (velocities).
- `actions` columns 0 and 3 (joint position targets).
- The matching named joints in full-scene replay `qpos`.
- Replay model signatures and dataset file-hash manifests.

World-space cube, goal, and grasp poses, gripper values, rewards, timing, and
recorded outcomes are preserved. Body-frame changes alone do not change the
recorded joint angles. Replay `qpos` indices depend on the scene and must not be
confused with observation column indices.

This was a one-time migration from Hugging Face revision
[`1b50e252fa21ca0c901a5d4a1aa765402b1c1ae7`](https://huggingface.co/datasets/jeonghan/mujoco-lab-datasets/tree/1b50e252fa21ca0c901a5d4a1aa765402b1c1ae7).
Episodes migrated from that revision record their source revision and original
model signature under `metadata.coordinate_migration`. See [coordinate_migration.json](coordinate_migration.json)
for model fingerprints and verification results, and each dataset's
`file_hashes.json` for the migrated episode SHA-256 digests.

Pre-migration checkpoints and normalization statistics are unsupported;
retrain policies with the migrated data. Loading, training, and replay do not
perform automatic conversion of old joint conventions.

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
