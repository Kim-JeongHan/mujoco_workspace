# MuJoCo Lab Demonstration Episodes

This repository contains simulated robot manipulation episodes collected with
the MuJoCo Lab workspace. The available directories contain demonstrations for
cube stacking with a Forte robot; see each directory's episode metadata for its
collection settings and recorded outcomes.

## Available datasets

| Directory | Cubes | Recorded attempts | Successful episodes |
| --- | ---: | ---: | ---: |
| `forte_mixed_1` | 1 | 2000 | 1999 |
| `forte_cube2_mix_2000` | 2 | 2000 | 1989 |
| `forte_cube4_mix_2000` | 4 | 2000 | 1618 |
| `forte_ctc_hover5mm_mixed_1` | 1 | 2000 | 2000 |
| `forte_pd_hold_hover5mm_mixed_1` | 1 | 2000 | 1995 |
| `forte_pd_minjerk_hover5mm_mixed_1` | 1 | 2000 | 1997 |
| `forte_pd_quintic_hover5mm_mixed_1` | 1 | 2000 | 1948 |

The two- and four-cube mixtures each contain 1000 heuristic and 1000 sampling
attempts. Failed attempts are retained. These datasets use 500 Hz physics and
100 Hz action commands. Observation dimensions are 39, 54, and 84 for one, two,
and four cubes; each action has seven arm targets and one gripper opening width.

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

The pre-hover PD datasets `forte_pd_mixed_1`, `forte_pd_minjerk_mixed_1`, and
`forte_pd_quintic_mixed_1` are archived at
[Hugging Face revision `529dd198b3c1`](https://huggingface.co/datasets/jeonghan/mujoco-lab-datasets/tree/529dd198b3c148cc2b8c845bc1c91485e68462a3)
and were removed locally after file-hash and replacement verification.
Each PD hover variant reuses its own 1,000 original heuristic attempts and
recollects 1,000 sampling attempts with the 5 mm placement hover.
The hold, fixed-duration minimum-jerk, and constrained quintic variants
retain their original PD gains and reference behavior. See
[pd_hover5mm_migration.json](pd_hover5mm_migration.json) and
[reference_ablation_hover5mm_1cube.json](reference_ablation_hover5mm_1cube.json).

The obsolete
`forte_cube4_mix_rrt_star_snapshot_20261012-195554` dataset was removed.
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

Load episodes with the workspace's `mujoco_lab.learning.datasets.Episode.load_all`
API. It keeps successful episodes by default; set `success_only=False` to
include failed and unlabeled attempts.
