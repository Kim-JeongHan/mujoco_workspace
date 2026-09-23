# ForteV1 RobStride

This fixed-base MuJoCo model comes from the supplied `ForteV1_RobStride.zip` CAD URDF. The complete archive contents are retained byte for byte under `third_party/fortev1-robstride/`; the adjacent `fortev1-robstride.SOURCE.json` records every file hash and the archive hash. The archive contains no explicit license declaration. The earlier Forte model remains available separately in `third_party/forte-arm-isaac-mujoco-demos/`.

The runtime model keeps all 830 source CAD visual instances and copies their referenced STL files without changing their bytes. Its nine coordinates are the seven selected arm joints and two parallel gripper sliders. All other CAD mates are fixed. Rigidly connected source inertias are combined, retaining the source total mass of 7.497537857 kg. The fixed root is named `base_link`. Its local height of 0.058 m seats the bottom of the CAD base on the standard 0.8 m table mount. `ee_site` is a simulation reference point placed on the source link `part_8_2`; the CAD archive does not define or calibrate a tool frame.

The runtime `home` keyframe defines a simulated zero pose with the upper arm upright, the long forearm horizontal, the purple `Part_62` wrist link vertical, and the `part_8_2` to `ee_site` tool link pointing down. The pad-defined `grasp` frame is a separate reference and remains tilted at home. The generator changes the wrist mount orientation without moving its pivot and applies a 7.44° fixed gripper mount correction about the wrist-roll pivot. It re-expresses the wrist-roll joint axis in the corrected body frame to preserve the rotation axis. These runtime mount orientations differ from the supplied CAD assembly; they are not a calibration of physical encoder zeros, which have not been measured.

The `grasp` site on `part_8_2` is centered between the open gripper contact pads and shifted 10 mm toward the fingertips. Its +X axis crosses the jaw gap, its +Z axis points toward the fingertips, and +Y completes the right-handed frame. For a top-down cube grasp, orient +Z toward the table. With the standard table mount and a 40 mm cube, placing this site 10 mm above the cube center leaves about 15 mm between the cube and the closest palm collision hull. The unshifted pad midpoint can overlap that hull. The site is a simulation grasp reference, not a measured hardware tool calibration. `ee_site` remains a separate link reference.

| MuJoCo joint | Source URDF joint | Control |
| --- | --- | --- |
| `shoulder_yaw` | `shoulderyaw` | Torque motor |
| `shoulder_pitch` | `shoulderpich` | Torque motor |
| `shoulder_roll` | `shoulderroll` | Torque motor |
| `elbow_pitch` | `revolute_1` | Torque motor |
| `lower_arm_roll` | `revolute_16` | Torque motor |
| `wrist_pitch` | `revolute_17` | Torque motor |
| `wrist_roll` | `revolute_9` | Torque motor |
| `gripper_left_joint` | `slider_1_1` | Coupled position servo |
| `gripper_right_joint` | `slider_2_1` | Coupled to left slider |

Arm actuator names append `_motor` to the joint names, followed by `gripper_motor`. The seven arm controls are torques in Nm, with simulation limits of ±87 Nm for the first five and ±12 Nm for the last two. These inherited Forte limits are for controller compatibility; the archive does not establish RobStride motor ratings. The gripper control is a slider position in meters: `0` is open and `-0.02` is closed. Its position servo uses `kp=2000 N/m`, `kv=5 N·s/m`, and a ±20 N force limit. An equality constraint keeps both sliders parallel. These gripper settings are simulation choices, not measured motor ratings. The first three arm joints retain their source limits; the remaining four source joints are continuous. Each arm joint has a simulation armature of 0.01 kg·m² and damping of 0.05 N·m·s/rad to represent omitted motor rotor inertia and stabilize control at a 2 ms step. These values are modeling assumptions, not CAD-derived or measured motor properties; the CAD body inertias remain unchanged.

The source URDF has no collision geometry. The runtime model uses 48 convex collision geoms from selected structural CAD meshes, one forearm capsule, and two flat contact pads. The component hulls cover the base, shoulder supports, upper arm, elbow, wrist, palm, and finger exteriors. They reuse the unchanged visual mesh files and transforms; small fasteners and internal electronics remain visual-only. The 285.5 mm forearm tube uses a 19 mm radius capsule. Collision geoms are in group 3, and visual geoms are in group 1.

The base collision hulls have 0.1 mm mounting clearance to avoid roundoff contacts at home. Finger outer hulls are offset outward by 0.5 mm so the pads make contact first. The pads keep sliding friction 1.0 and use `condim=4`, `solref="0.006 1"`, and `solimp="0.995 0.999 0.001"`; their higher contact priority keeps these settings when an object uses default contact parameters. No global solver options or other robot assets are changed by this tuning.

Native-physics tests start with a free 40 mm, 79.36 g cube already between the fingers. With Earth gravity and the normal PD arm controller, the gripper holds it for 10 seconds with less than 1 mm of relative slip, lifts it by more than 40 mm, then opens to release it onto the floor. The object is never welded to the gripper. This verifies retention, lifting, and release from an initial grasp, not a pickup planner or a calibrated hardware model. Payload-induced arm deflection is separate from slip relative to the gripper. Convex hulls still approximate concave parts, and the tested poses do not cover the full joint workspace. `scene.xml` previews the robot on a table.

Regenerate the asset from the original archive with:

```bash
uv run python scripts/import_forte_robstride.py
```
