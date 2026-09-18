# MuJoCo workspace

A standalone workspace for robot simulation, feedback control, reusable furniture,
and composable environments. It includes **MuJoCo 3.13.0** and **MuJoCo Warp 3.13.0**,
with UR20, UR30, Panda, and Forte assets. Development uses Python 3.12 on Ubuntu.

## Installation

```bash
cd ~/workspace/mujoco-lab
uv sync
```

The project uses its own `.venv`. MuJoCo and MuJoCo Warp are pinned in
`pyproject.toml`; `uv.lock` records the complete dependency resolution. The existing
`~/workspace/.venv-d4rl` is a separate environment.

## Select a robot, model, or environment

Every CLI command requires an explicit `--robot`, `--model`, or `--environment`.
A robot without an environment uses `empty`. `--model` loads a standalone MJCF or
URDF file and cannot be combined with `--robot` or `--environment`.

```bash
uv run mujoco-lab view --robot panda
uv run mujoco-lab simulate --robot ur20 --steps 1000
uv run mujoco-lab render --robot ur30 --output outputs/ur30.png
uv run mujoco-lab view --environment warehouse
uv run mujoco-lab view --model /path/to/scene.xml
```

`--steps` counts physics steps. Rendering defaults to zero steps and saves the
current scene as a 640 x 480 PNG; the default output path is `outputs/scene.png`.
The GUI requires a desktop display. Close its window to exit.
On Linux without a display, set `MUJOCO_GL=egl` when rendering.

## Forte control integration

The `control/` package contains the integrated Forte PD and OSC algorithms,
controller creation, and controlled stepping. The `state/` package provides joint
state and MuJoCo dynamics access. Model loading and viewing live in `simulation.py`
and `viewer.py`. Select a controller through the CLI or manipulator example:

```bash
# Original joint-space PD routine in the workspace environment
uv run python examples/manipulator.py --robot forte --environment empty --controller pd

# OSC follows a circle expressed in the robot base frame
uv run python examples/manipulator.py --robot forte --environment warehouse --controller osc

# Twelve simulated seconds, with tracking statistics
uv run mujoco-lab simulate --robot forte --environment warehouse --controller osc --steps 6000

# PNG with original torque arrows and controller target annotations
MUJOCO_GL=egl uv run mujoco-lab render --robot forte --controller pd --steps 1000 --output outputs/forte_pd.png
```

Controller choices are `none` (default), `pd`, and `osc`. UR20, UR30, and Panda use
their asset-defined position servos with `none`. Forte uses torque motors, so
`none` leaves zero torque and allows the arm to fall under gravity. `pd` and `osc`
require the Forte torque-motor layout and are rejected for the other robots.

The source equations, gains, PD waypoints, clipping, and tracking statistics are
reused. The OSC path is transformed to the actual robot base pose, including the
0.8 m mount height in furnished environments. These reference trajectories do not
perform obstacle-avoiding planning; furniture collisions remain enabled.

The GUI runs feedback through a scoped MuJoCo control callback and restores any
previous callback on exit. It uses the native blocking viewer, without the upstream
process-exit workaround. PNG images include upstream controller annotations. See
the [project layout](#project-layout) for module locations.

## Manipulator example and robot factory

```bash
uv run python examples/manipulator.py --robot ur20
uv run python examples/manipulator.py --robot ur30 --environment table_shelf
uv run python examples/manipulator.py --robot panda --environment warehouse
uv run python examples/manipulator.py --robot forte --controller pd --headless --steps 6000
```

`robot.py` defines `create_robot(name, environment="empty")`. Its `ROBOT_SCENES`
registry points to robot-only MJCF files. The factory attaches the selected robot
to the environment, then initializes native MuJoCo model/data objects from the
`home` keyframe. Add a robot asset and one registry entry to extend the selectors.

## Python API

```python
import mujoco
from mujoco_lab import create_robot, load_simulation
from mujoco_lab.control import create_controller, run_steps

model, data = create_robot("forte", environment="warehouse")
controller = create_controller("osc", model, data)
stats = run_steps(model, data, 6000, controller)
print(stats.describe("tracking error", "m"))

# Explicit files remain supported
model, data = load_simulation("src/mujoco_lab/assets/panda/scene.xml")
mujoco.mj_step(model, data, nstep=1000)
```

`model` and `data` are native MuJoCo objects. `load_simulation()` requires a model
path. Models with a named `home` keyframe start with that state and control input.
Control units are asset-specific: the UR/Panda servos use position targets;
Forte's direct motors use torques in Nm.

## Objects and environments

Object assets define geometry, dimensions, material, and collisions. Environments
provide placements, a floor, lights, and a `robot_mount` site.

- `empty`: floor only, robot mounted at z=0.
- `table_shelf`: table and large shelf, robot mounted at z=0.8 m.
- `warehouse`: table, large shelf, and small shelf, robot mounted at z=0.8 m.

```bash
uv run mujoco-lab view --robot ur20 --environment warehouse
uv run mujoco-lab simulate --robot ur30 --environment warehouse --steps 1000
uv run mujoco-lab render --robot panda --environment table_shelf --output outputs/panda_table_shelf.png
```

`create_environment(name)` returns an editable `mujoco.MjSpec`. Register new scenes
in `ENVIRONMENT_SCENES` in `environment.py`. The furnished layouts preserve MPD's
robot-relative geometry. The table and small shelf are solid-box approximations;
the large shelf contains ten panels with open compartments. See the
[environment guide](src/mujoco_lab/assets/environments/README.md) and
[object guide](src/mujoco_lab/assets/objects/README.md).

## MuJoCo Warp

MuJoCo Warp is installed by `uv sync`. Import it as `mujoco_warp` and the underlying
NVIDIA runtime as `warp`. The regular CLI and Forte feedback controllers use native
MuJoCo physics. A separate CUDA example runs asset-defined controls in parallel:

```bash
uv run python examples/warp_manipulator.py --robot panda --worlds 1 --steps 100
```

The example also accepts `--environment`. It requires a CUDA-capable NVIDIA GPU.
It captures a simulation step in a CUDA graph and replays it for each step.
The first run compiles kernels and can take longer. Start with a small world count
on this laptop's 4 GiB GPU. The Python PD/OSC controllers are not GPU kernels;
the Warp example uses the asset's home control values.

## Transform utilities

`Transform` is a simulator-independent interface to SciPy's
[`RigidTransform`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.RigidTransform.html).
`T_ab` maps points from frame b into frame a: `p_a = R_ab @ p_b + t_ab`. Compose
transforms with `T_ab @ T_bc`; the right-hand transform is applied first.

```python
from mujoco_lab.utils import Transform
from scipy.spatial.transform import Rotation

T_world_robot = Transform(
    rotation=Rotation.from_euler("z", 90, degrees=True),
    translation=[0.0, 0.0, 0.8],
)
point_world = T_world_robot.apply([0.5, 0.0, 0.2])
point_robot = T_world_robot.inverse().apply(point_world)
matrix = T_world_robot.as_matrix()  # 4x4 homogeneous matrix
restored = Transform.from_matrix(matrix)

# Equivalent poses using fixed-axis RPY and different units
from_mmdeg = Transform.from_pose_mmdeg([500, 200, 400, 20, -15, 35])
from_mrad = Transform.from_pose_mrad(from_mmdeg.as_pose_mrad())
```

The constructor accepts a single SciPy `Rotation` or a proper 3x3 rotation matrix,
with translation in meters. Conversion methods replace the rotation/translation
properties:

| Method | Output | Units / order |
|---|---|---|
| `as_rotation()` | SciPy `Rotation` | Use `.as_matrix()` or `.as_quat()` as needed |
| `as_translation()` | `[x, y, z]` | m |
| `as_pose_mrad()` | `[x, y, z, roll, pitch, yaw]` | m, rad |
| `as_pose_mmrad()` | `[x, y, z, roll, pitch, yaw]` | mm, rad |
| `as_xyqquat()` | `[x, y, z, qw, qx, qy, qz]` | m, MuJoCo scalar-first quaternion |
| `as_mmdeg()` | `[x, y, z, roll, pitch, yaw]` | mm, deg |

RPY uses fixed-axis (extrinsic) `xyz` Euler angles:
`R = Rz(yaw) @ Ry(pitch) @ Rx(roll)`. SciPy's Euler angle ranges and gimbal-lock
behavior apply. MuJoCo quaternion exports use `wxyz`. When calling SciPy directly,
use `Rotation.from_quat(quat, scalar_first=True)` and
`rotation.as_quat(scalar_first=True)` to keep that order. Quaternion signs may
differ while representing the same rotation.

Frames are right-handed. Rotation matrices map local vectors into parent
coordinates; `data.xpos` and `data.xmat` describe body frames in world coordinates.
The retained RPY convention corresponds to MJCF `eulerseq="XYZ"`, with
`angle="radian"` for radian exports; it differs from the default MJCF intrinsic
`eulerseq="xyz"`. Prefer `quat` for MuJoCo orientation interchange. See the
[MuJoCo frame conventions](https://mujoco.readthedocs.io/en/stable/modeling.html#frame-orientations)
and [Euler sequence reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html#compiler-eulerseq).

Inputs and exported values are independent copies.
`apply()` accepts a point `(3,)` or points `(..., 3)`;
`apply_vectors()` rotates free 3D vectors without translation. SciPy handles
composition, inversion, application, matrix/coordinate validation, and rotation
normalization. SciPy is installed by `uv sync`.

To capture a MuJoCo body pose after forward kinematics:

```python
body = data.body("base_link")
T_world_body = Transform(rotation=body.xmat.reshape(3, 3), translation=body.xpos)
```

The transform is a snapshot and remains unchanged when simulation data advances.

### Plot coordinate frames

`plot()` draws local x/y/z axes in red/green/blue, using meters and a Z-up view.
It returns a Matplotlib 3D Axes for overlaying frames or saving an image. The view
includes arrow endpoints and existing data with equal spatial scales. Matplotlib
and its PyQt6 GUI backend are installed by `uv sync`; plotting imports are lazy.

```python
import matplotlib.pyplot as plt

ax = Transform.identity().plot(label="world", length=0.3, show=False)
T_world_robot.plot(ax=ax, label="robot", length=0.3)
ax.figure.savefig("frames.png", dpi=150)
plt.show()
```

By default, `plot()` shows a window only when it creates a new figure. Pass
`show=False` to suppress it or `show=True` to show a supplied Axes. All overlaid
transforms must be expressed relative to the same parent frame. For headless
rendering, set `MPLBACKEND=Agg` and use `show=False`.

The `if __name__ == "__main__":` block in `transform_utils.py` creates a world frame
with `from_pose_mrad()` and a tool frame with `from_pose_mmdeg()`, prints their
pose representations, and plots both in one window. Run it directly:

```bash
uv run python src/mujoco_lab/utils/transform_utils.py
```

## Project layout

```text
src/mujoco_lab/
  robot.py                # Robot factory
  environment.py          # Environment factory
  simulation.py           # MuJoCo model loading and state initialization
  viewer.py               # Native GUI, callbacks, and offscreen rendering
  cli.py                  # Model/environment/controller selection
  utils/
    transform_utils.py    # Rigid transforms, composition, inverse, point mapping
  state/
    joint_state.py        # JointState data and read_state()
    dynamics.py           # Frame positions, Jacobian, and mass matrix
  control/
    pd.py                 # Original PD algorithm and gains
    osc.py                # Original OSC algorithm and frame adaptation
    trajectory.py         # Minimum-jerk interpolation
    base.py               # Controller base class
    stats.py              # Tracking and saturation statistics
    factory.py            # Actuator validation and controller creation
    runner.py             # Torque application and controlled stepping
    visualization.py      # Controller markers and torque arrows
  assets/
    ur20/, ur30/, panda/, forte/
    objects/
    environments/
examples/
  manipulator.py          # --robot, --environment, --controller
  warp_manipulator.py     # CUDA example with --robot and --environment
scripts/                  # Asset import tools
third_party/              # Original Forte sources with recorded removals
tests/                   # Workspace validation
```

Robot assets include converted XML and their mesh/texture files, so no ROS or
Isaac Sim installation is required for normal workspace execution. See the
[asset guide](src/mujoco_lab/assets/README.md) for sources and licenses.

## MuJoCo state and dynamics

The workspace uses MuJoCo directly. Robot and environment factories compose MJCF
assets and return native MuJoCo objects. The concrete
[`Dynamics` class](src/mujoco_lab/state/dynamics.py) owns its reusable NumPy
buffers and exposes them through getters. `JointState` and `read_state()` live in
[`state/joint_state.py`](src/mujoco_lab/state/joint_state.py):

```python
from mujoco_lab import create_robot
from mujoco_lab.state import read_state
from mujoco_lab.state.dynamics import Dynamics

model, data = create_robot("forte")
joint_state = read_state(data)
dynamics = Dynamics(model, data)
position = dynamics.get_frame_position("ee_site")  # (3,), world coordinates, m
jacobian = dynamics.get_jacobian("ee_site")  # (6, nv), linear then angular
mass = dynamics.get_mass_matrix()  # (nv, nv), SI units
```

Frame names resolve to MJCF sites. The geometric Jacobian
maps generalized velocity to the frame origin's linear velocity followed by
angular velocity, both in world coordinates. Matrix columns and mass-matrix axes
follow the same generalized velocity order. Position-only OSC uses `jacobian[:3]`;
its equations, gains, reference trajectories, and update order are preserved.
Its frame selector is now `frame="ee_site"` instead of `site="ee_site"`.

Treat getter results as read-only borrowed arrays. A later call of the same getter
may overwrite its buffer, including a Jacobian query for a different frame.
Simulation updates can also change live views. Use `.copy()` for a persistent
snapshot. Getters neither advance physics nor implicitly call `mj_forward`.

The controllers operate on one robot with NumPy arrays. `state/` owns state data
and read access; it does not depend on `control/`. Controllers consume `JointState`
and the dynamics getters. `control/factory.py` creates and validates controllers,
while `control/runner.py` applies torque limits and advances physics.
`control/base.py` holds the controller base class. Controller equations remain
separate from rendering and file loading.

## Original Forte source

The upstream snapshot is retained under
`third_party/forte-arm-isaac-mujoco-demos/` at commit
`d0846d26a940f630fbb6bd54aaa2de821aee8f14`.

```bash
uv run python third_party/forte-arm-isaac-mujoco-demos/Forte_mujoco/scripts/pd_control.py --headless --duration 12
uv run python third_party/forte-arm-isaac-mujoco-demos/Forte_mujoco/scripts/osc_control.py --headless --duration 12
```

The workspace integrates its algorithms and control helpers into `control/`,
state and dynamics access into `state/`, model loading into `simulation.py`, and
rendering into `viewer.py`, preserving the equations and numeric settings.
The complete `Forte_mujoco` subtree is unchanged. Sixteen unrelated Isaac template
files were removed as requested and recorded in the source manifest. The actual
Forte reaching task and training tools remain available. See
[upstream source notes](third_party/README.md).

## Validation and packaging

```bash
uv sync --locked
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv build
```

The wheel includes the workspace, packaged controller code, assets, and provenance.
The source distribution additionally includes the retained third-party snapshot
and its documented removal list.
Use the commands above to check GUI control and rendering, and the Warp example
for a CUDA smoke test. All workspace documentation is maintained in English.

To use the package from another uv project:

```bash
uv add --editable /home/jeonghan/workspace/mujoco-lab
```

## References

- [MuJoCo Python API](https://mujoco.readthedocs.io/en/stable/python.html)
- [MuJoCo Warp](https://mujoco.readthedocs.io/en/latest/mjwarp/index.html)
- [Original Forte repository](https://github.com/jahirsadik/forte-arm-isaac-mujoco-demos)
- [uv project management](https://docs.astral.sh/uv/guides/projects/)
