# Environments

Environments place reusable objects and define the floor, lighting, camera defaults, and a `robot_mount` site. They contain no robot geometry.

| Environment | Objects | Robot mount (m) |
|---|---|---|
| `empty` | Floor only | (0, 0, 0) |
| `table_shelf` | Table and large shelf | (0, 0, 0.8) |
| `warehouse` | Table, large shelf, and small shelf | (0, 0, 0.8) |
| `cube_stack_2`, `cube_stack_3`, `cube_stack_4` | Table, large shelf, and 2–4 cubes with goal markers | (0, 0, 0.8) |
| `cube_stack_2_warehouse`, `cube_stack_3_warehouse`, `cube_stack_4_warehouse` | Cube layouts with the warehouse small shelf | (0, 0, 0.8) |

## Usage

```bash
# Inspect furniture without a robot
uv run python examples/model.py --model src/mujoco_lab/assets/environment/warehouse/scene.xml

# Compose a robot with an environment
uv run python examples/manipulator.py --robot panda --environment warehouse
uv run python examples/manipulator.py --robot panda --environment cube_stack_3

# Run the cube stacking task
uv run mujoco-lab --cubes 3 --environment warehouse --headless
```

`create_environment(name)` in `environment.py` returns a fresh, editable MuJoCo `MjSpec`. `Simulator(create_environment(...), robots=[RobotSpec(...)])` attaches robot-only `robot.xml` assets and combines their home states with the environment home. Each robot's entities receive an instance-name prefix. A single robot with no explicit pose uses `robot_mount`; explicit poses are world placements, and multiple robots require them. The environment supplies the only floor in the composed scene.

Add an environment directory and register its scene in `ENVIRONMENT_SCENES` in `assets/__init__.py`. Every registered environment must define a site named `robot_mount`.

The `cube_stack_2`, `cube_stack_3`, and `cube_stack_4` scenes contain the OGBench task-5 starting positions and translucent goals. Their warehouse variants add the existing small shelf. All six scenes can be loaded directly with `create_environment`; `mujoco_lab.create_cube_stack(cubes, environment=...)` returns the corresponding editable `MjSpec` for use with `Simulator(scene, robots=[RobotSpec(...)])`. Cube geometry, contact settings, and marker geometry match the OGBench source retained at [`third_party/ogbench/cube.xml`](../../../../third_party/ogbench/cube.xml), with colors and positions specified in each count's `layout.xml`. The source is MIT licensed; see `objects/ogbench_cube/LICENSE.txt`.

## Frames and source layout

The furnished layouts are derived from MPD's `EnvTableShelf` and `EnvWarehouse` at zero environment rotation, without their optional extra obstacles. MPD places the robot base at z=0 and the bottoms of the furniture at z=-0.8. Here the scene is shifted upward by 0.8 m: furniture bottoms are at nominal ground level z=0, the tabletop is at z=0.8, and the robot mount is at z=0.8. The table has been enlarged and moved toward the mount so the bundled robot bases fit on its top; the shelf placements retain their source layout. The floor is at z=-0.005 for a small clearance.

Object origins are bottom centers. Default placements are:

| Object | Position (m) | Yaw |
|---|---|---|
| Table | (0.35, 0, 0) | 90° |
| Large shelf | (0.4, 0.58, 0) | 0° |
| Small shelf | (0.4, -0.6, 0) | 0° |

The table is 1.2 m long, 0.8 m wide, and 0.8 m high. Its top spans x=-0.25 to 0.95 m and y=-0.4 to 0.4 m, with 0.03 m clearance to the large shelf and 0.06 m to the small shelf. The robot base remains at MPD's original mount location; no separate mounting fixture is modeled. Colors, lighting, and the floor are local visualization choices. This adapts MPD's geometry and layout, without porting its planners, SDF implementation, or simulator settings. The table geometry is defined in [its object model](objects/table/model.xml).
