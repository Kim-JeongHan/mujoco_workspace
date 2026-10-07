# Environments

Environments place reusable objects and define the floor, lighting, camera defaults, and a `robot_mount` site. They contain no robot geometry.

| Environment | Objects | Robot mount (m) |
|---|---|---|
| `empty` | Floor only | (0, 0, 0) |
| `table_shelf` | Table and large shelf | (0, 0, 0.8) |
| `warehouse` | Table, large shelf, and small shelf | (0, 0, 0.8) |
| `book_shelf` | Existing table and side shelf with book-specific board heights (no book until added by the factory) | (0, 0, 0.8) |
| `cube_stack_1` through `cube_stack_4` | Table, large shelf, and 1–4 cubes with goal markers | (0, 0, 0.8) |
| `cube_stack_1_warehouse` through `cube_stack_4_warehouse` | Cube layouts with the warehouse small shelf | (0, 0, 0.8) |

## Usage

```bash
# Inspect furniture without a robot
uv run python examples/simulation.py --environment warehouse

# Inspect a robot in the default empty environment
uv run python examples/simulation.py --robot panda

# Compose a robot with an environment
uv run python examples/simulation.py --robot panda --environment warehouse
uv run python examples/simulation.py --robot panda --environment cube_stack_3

# Load an external MJCF or URDF file without a robot
uv run python examples/simulation.py --environment path.xml

# Run the cube stacking task
uv run mujoco-lab --cubes 3 --environment warehouse --headless

# Run the same cube task through its example entry point
uv run python examples/cube_task.py --cubes 3 --environment warehouse --headless
uv run python examples/cube_task.py --method sampling --headless planning:rrg-config --planning.seed 7

# Preview book physics or sample an insertion motion
uv run python examples/book_task.py --book medium --method preview
uv run python examples/book_task.py --headless planning:rrt-star-config --planning.seed 7
```

`create_environment(name)` in `assets/loader.py` returns a fresh, editable MuJoCo `MjSpec`. `Simulator(create_environment(...), robots=[RobotSpec(...)])` attaches robot-only `robot.xml` assets and combines their home states with the environment home. Each robot's entities receive an instance-name prefix. A single robot with no explicit pose uses `robot_mount`; explicit poses are world placements, and multiple robots require them. The environment supplies the only floor in the composed scene.

Every `RobotSpec` requires an explicit `config=` argument. Use
`load_robot_config(robot_type)` to load bundled settings, or supply a
`RobotConfig` for a custom asset. Simulator copies each config before filling
its model metadata, so reusing a config preserves the caller's metadata and
keeps each robot's asset names independent.

Add an environment directory and register its scene in `ENVIRONMENT_SCENES` in `assets/__init__.py`. Every registered environment must define a site named `robot_mount`.

For rigid book experiments, `mujoco_lab.create_book_insertion("medium")` loads a
complete XML scene with a centered free book and an insertion target in
`book_shelf`. Four fixed book types share box collision geometry and common
friction; their explicit inertias follow the uniform-box formula. Book models
are defined in `objects/book/`, and the separate shelf model in
`objects/book_shelf/` defines book-specific board heights. The complete scenes
in `book_shelf/{small,medium,thick,large}.xml` also load directly in MuJoCo.
See the [book workspace notes](book_shelf/README.md)
for frames, specifications, preview commands, and the staged experiment plan.

The `cube_stack_1` scene contains one cube and a translucent goal. The `cube_stack_2`, `cube_stack_3`, and `cube_stack_4` scenes contain the OGBench task-5 starting positions and translucent goals. Their warehouse variants add the existing small shelf. All eight scenes can be loaded directly with `create_environment`; `mujoco_lab.create_cube_stack(cubes, environment=...)` returns the corresponding editable `MjSpec` for use with `Simulator(scene, robots=[RobotSpec(...)])`. Cube geometry, contact settings, and marker geometry match the OGBench source retained at [`third_party/ogbench/cube.xml`](../../../../third_party/ogbench/cube.xml), with colors and positions specified in each count's `layout.xml`. The source is MIT licensed; see `objects/ogbench_cube/LICENSE.txt`.

## Frames and source layout

The furnished layouts are derived from MPD's `EnvTableShelf` and `EnvWarehouse` at zero environment rotation, without their optional extra obstacles. MPD places the robot base at z=0 and the bottoms of the furniture at z=-0.8. Here the scene is shifted upward by 0.8 m: furniture bottoms are at nominal ground level z=0, the tabletop is at z=0.8, and the robot mount is at z=0.8. The table has been enlarged and moved toward the mount so the bundled robot bases fit on its top; the shelf placements retain their source layout. The floor is at z=-0.005 for a small clearance.

Object origins are bottom centers. Default placements are:

| Object | Position (m) | Yaw |
|---|---|---|
| Table | (0.35, 0, 0) | 90° |
| Large shelf | (0.4, 0.58, 0) | 0° |
| Small shelf | (0.4, -0.6, 0) | 0° |

The table is 1.2 m long, 0.8 m wide, and 0.8 m high. Its top spans x=-0.25 to 0.95 m and y=-0.4 to 0.4 m, with 0.03 m clearance to the large shelf and 0.06 m to the small shelf. The robot base remains at MPD's original mount location; no separate mounting fixture is modeled. Colors, lighting, and the floor are local visualization choices. This adapts MPD's geometry and layout, without porting its planners, SDF implementation, or simulator settings. The table geometry is defined in [its object model](objects/table/model.xml).
