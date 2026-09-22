# MuJoCo workspace

A Python workspace for MuJoCo robot simulation with Panda and Forte assets.

## Install

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --locked
```

## Run

Choose a robot with `--robot` or a scene with `--environment`:

```bash
uv run mujoco-lab --command view --robot panda
uv run mujoco-lab --command view --environment warehouse
uv run mujoco-lab --command simulate --robot panda --controller position --steps 1000
uv run mujoco-lab --command simulate --robot forte --controller pd --steps 6000
uv run mujoco-lab --command render --robot panda --output outputs/panda.png
```

Robots are `panda` and `forte`. Environments are `empty`
(the default for a robot), `table_shelf`, `warehouse`, and `cube_stack_2`,
`cube_stack_3`, or `cube_stack_4` (each also has a `_warehouse` variant). `forte` is the
ForteV1_RobStride CAD arm with seven controlled arm axes and a parallel gripper.
It also supports `--controller osc`; its arm motors have zero torque without a
controller.
The `position` controller uses each robot's existing position servos, including
the Panda's coupled finger actuator.
`pd` and `osc` require torque or force actuators; they do not change an asset's
actuator type. A custom torque-controlled robot can use `pd` by supplying
per-joint `kp` and `kd` to `create_controller`, in actuator order.
`--steps` counts physics steps. Control runs once per step. The default step
period is 0.002 seconds; use `--dt` to change it. The viewer needs a desktop display;
for headless PNG rendering on Linux, set `MUJOCO_GL=egl`.

To open an external MJCF or URDF file, or run the other examples:

```bash
uv run python examples/model.py --model /path/to/scene.xml
uv run python examples/multi_robot.py --layout forte_panda --headless --steps 1000
uv run python examples/forte_gripper.py
uv run python examples/warp_manipulator.py --robot panda --worlds 1 --steps 100
```

The Warp example requires a CUDA-capable NVIDIA GPU.

## Panda and Forte cube stacking

Panda and Forte can physically pick and stack two, three, or four 4 cm cubes on the
`table_shelf` or `warehouse` tabletop. The cube model, colors, and stacking
start/goal layouts come from OGBench task 5. This is a robot demonstration,
not an OGBench robot or benchmark score implementation. The cubes have free
joints and move through gripper and support contacts.

```bash
uv run python examples/cube_stack.py --cubes 2 --headless
uv run python examples/cube_stack.py --cubes 4 --headless --output outputs/four_cubes.png
uv run python examples/cube_stack.py --cubes 3
uv run python examples/cube_stack.py --robot forte --cubes 4 --headless
```

The example stops after the fingers release and the aligned stack remains
supported by table/cube contacts for 0.5 seconds. It reports both the OGBench
goal-distance check (each center within 4 cm) and this stricter physical-stack
check. The default limit is 22,000 physics steps for Panda and 30,000 for
Forte, at 0.002 seconds per step. Forte uses a top-down grasp site and its
native coupled gripper; the shipped robot mount and OGBench cube layouts are
unchanged.
Use `MUJOCO_GL=egl` for headless PNG rendering on Linux.

For Python use, import `create_cube_stack` from `mujoco_lab`, call
`create_cube_stack(cubes, environment="table_shelf", robot="panda")` (or
`robot="forte"`), and attach `CubeStackTask(sim, cubes)`
before `sim.run_steps(22000)` (30,000 for Forte). Call `task.reset()` to restore the initial cube,
goal, robot, and task state together. The exact upstream cube XML and MIT
license are recorded in `third_party/ogbench.SOURCE.json`.

Choose `CubeStackTask(sim, cubes, method="sampling", planner="rrt_connect")`
or `--method sampling --planner rrt_connect` in the example to search each
pick, lift, place, and retract motion from the measured arm pose. The stage
targets and physical grasp/release checks are the same as for the default
`heuristic` method. `rrt` and `prm` are also available, with `seed` and
`planning_budget` controls. Sampling checks edges at discrete points against
a scene snapshot taken on stage entry. Each carrying stage requires observed
contact from both fingers. A grasped cube follows its measured grasp pose
only in the checker's private scratch state; the real cube remains
under physics control. These checks do not guarantee continuous or dynamic
collision avoidance. Planning and execution failures are reported without
switching methods.

## Planning

`mujoco_lab.planning` provides AStar graph search, RRT and RRT* variants,
RRG, and PRM and PRM* planners. The imported `AStar` uses edge costs only
(uniform-cost search). Planners accept a `CollisionChecker`; the
`MuJoCoCollisionChecker` checks robot joint states against a snapshot of the
MuJoCo scene; call `refresh()` after scene changes. See the
[planning guide](src/mujoco_lab/planning/README.md) for a Python
example and the [source manifest](third_party/planning.SOURCE.json) for
upstream provenance. To plan and drive the robot through a joint-space path in
the viewer:

```bash
uv run python examples/planning.py --robot panda --planner rrt
```

The viewer shows the planned end-effector path and actual trail, plus labeled
XY floor projections for paths hidden behind the arm. Add
`--headless` for a bounded physics run, `--output path.npz` to save the planned
joint waypoints, or `--image path.png` for a headless path image. The example
also supports Forte and exits with an error if the measured robot does not
reach the goal within `--steps` physics steps.

## Python API

```python
from mujoco_lab import RobotSpec, Simulator, create_environment
from mujoco_lab.control import create_controller, demo_target_updater

sim = Simulator(create_environment("empty"), robots=[RobotSpec("arm", "forte")])
arm = sim.robots["arm"]
arm.change_controller(create_controller("pd", arm))
arm.controller.set_gripper_target(-0.02)  # meters per finger; 0 opens
sim.target_updater = demo_target_updater(sim, {"arm": "pd"})
stats = sim.run_steps(1000)
print(stats["arm"].describe("tracking error", "rad"))
```

For a position-controlled arm, use `create_controller("position", arm)` and
assign `arm.target = ControlTarget(angles)` before stepping. The target follows
actuator order and includes the Panda's driven finger joint. Import
`ControlTarget` from `mujoco_lab.control`.

Forte `RobotState` includes both finger joints (`nq = nv = 9`). The PD target
contains seven arm angles; the gripper uses a separate position target from
`-0.02` m (closed) to `0` m (open). One native actuator drives the coupled
finger pair. `sim.reset()` reopens it. The CAD archive did not include motor
ratings or collision shapes; the bundled torque ranges and contact proxies are
simulation choices documented with the [asset](src/mujoco_lab/assets/robot/forte/README.md).

For a world-space `Transform` named `target_pose`, solve arm joint positions with
`arm.state.solve_ik(target_pose, frame="ee_site", seed=previous_q)`. Use `"grasp"`
for Panda's grasp frame. The seed is optional and defaults to the current
positions. Only robot-owned hinge/slide joints on the selected site's ancestor
chain are optimized; returned positions follow their `joint_names` order.
Descendant finger joints remain fixed, and continuous joints have no artificial
position bounds. Each query uses a private copy of the current scene data and
does not change live state or advance physics.

IK keeps the existing planner's least-squares tolerances: 8 mm position error
and 1/3 rad orientation error, with orientation residual weight 0.18. It raises
`ValueError` for an unreachable pose and does not check collision-free motion.
The Panda cube task now calls this shared method while retaining its existing
downward grasp targets and motion sequence.
