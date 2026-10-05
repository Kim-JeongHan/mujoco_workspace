"""Inspect a named robot's collision model in Viser without an environment.

Run from the workspace root:
    uv run --extra viewer python -m mujoco_lab.assets.view_collision --robot forte2
"""

from __future__ import annotations

import argparse
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import mujoco
import numpy as np
import trimesh
import viser

from mujoco_lab.assets.robot.robot import RobotConfig
from mujoco_lab.utils.transform_utils import Transform

ROBOT_PATH = Path(__file__).parent / "robot"
CAPSULE_COLOR = (255, 0, 0)


def local_name(name):
    """Remove the scene's robot instance prefix."""
    return (name or "").rsplit("/", 1)[-1]


def robot_names():
    """Discover assets by directory convention, including experimental variants."""
    return tuple(sorted(path.parent.name for path in ROBOT_PATH.glob("*/robot.xml")))


def geom_mesh(model, geom_id, *, convex=False):
    """Construct geometry in the compiled geom frame, including convex hulls."""
    kind = int(model.geom_type[geom_id])
    size = model.geom_size[geom_id]
    if kind == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = model.geom_dataid[geom_id]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        face_start, face_count = model.mesh_faceadr[mesh], model.mesh_facenum[mesh]
        result = trimesh.Trimesh(
            vertices=model.mesh_vert[start : start + count],
            faces=model.mesh_face[face_start : face_start + face_count],
            process=False,
        )
        return result.convex_hull if convex else result
    if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        result = trimesh.creation.capsule(height=2 * size[1], radius=size[0], count=[16, 24])
        # Normalize the axial center across trimesh versions.
        result.vertices[:, 2] -= result.bounds[:, 2].mean()
        return result
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        return trimesh.creation.box(extents=2 * size)
    if kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return trimesh.creation.cylinder(radius=size[0], height=2 * size[1], sections=32)
    if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        return trimesh.creation.icosphere(subdivisions=2, radius=size[0])
    if kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        result = trimesh.creation.icosphere(subdivisions=2)
        result.vertices *= size
        return result
    raise ValueError(f"Unsupported geom type {kind} for {model.geom(geom_id).name}")


@dataclass(frozen=True)
class JointControl:
    joint: int
    name: str
    scale: float
    unit: str
    lower: float
    upper: float


class RobotPreview:
    """Load robot.xml and derive independent scalar controls from its joints."""

    def __init__(self, name):
        if name not in robot_names():
            raise ValueError(
                f"Unknown robot {name!r}; available robots: {', '.join(robot_names())}"
            )
        self.name = name
        self.model = mujoco.MjModel.from_xml_path(str(ROBOT_PATH / name / "robot.xml"))
        self.data = mujoco.MjData(self.model)
        model = self.model
        home = next((i for i in range(model.nkey) if local_name(model.key(i).name) == "home"), None)
        if home is not None:
            mujoco.mj_resetDataKeyframe(model, self.data, home)
        self.home = self.data.qpos.copy()
        self.zero = next(
            (i for i in range(model.nkey) if local_name(model.key(i).name) == "zero"), None
        )
        # MuJoCo joint equality: q1-q1_ref = poly(q2-q2_ref). Controls expose
        # drivers only; dependent coordinates follow even in kinematic preview.
        pending = [
            i
            for i in range(model.neq)
            if int(model.eq_type[i]) == mujoco.mjtEq.mjEQ_JOINT and model.eq_active0[i]
        ]
        dependent = {int(model.eq_obj1id[i]) for i in pending}
        resolved = set(range(model.njnt)) - dependent
        self.relations = []
        while pending:
            ready = [i for i in pending if model.eq_obj2id[i] < 0 or model.eq_obj2id[i] in resolved]
            if not ready:
                raise ValueError(f"Robot {name!r} has cyclic joint equality dependencies")
            for equality in ready:
                self.relations.append(equality)
                resolved.add(int(model.eq_obj1id[equality]))
                pending.remove(equality)
        self.controls = []
        for joint in range(model.njnt):
            kind = int(model.jnt_type[joint])
            if joint in dependent or kind not in (
                mujoco.mjtJoint.mjJNT_HINGE,
                mujoco.mjtJoint.mjJNT_SLIDE,
            ):
                continue
            is_slide = kind == mujoco.mjtJoint.mjJNT_SLIDE
            scale, unit = (1000.0, "mm") if is_slide else (180 / np.pi, "deg")
            value = self.home[model.jnt_qposadr[joint]]
            if model.jnt_limited[joint]:
                lower, upper = model.jnt_range[joint]
            elif is_slide:
                lower, upper = value - 0.1, value + 0.1
            else:
                lower, upper = min(-np.pi, value), max(np.pi, value)
            self.controls.append(
                JointControl(joint, model.joint(joint).name, scale, unit, lower, upper)
            )
        self.poses = {
            local_name(model.key(index).name): model.key(index).qpos.copy()
            for index in range(model.nkey)
        }
        config_path = ROBOT_PATH / name / "robot.yaml"
        if config_path.exists():
            config = RobotConfig.load(config_path)
            if config.pose is not None:
                self.poses = {}
                for preset, values in config.pose.named_poses().items():
                    if len(values) != len(self.controls):
                        raise ValueError(
                            f"Robot {name!r} pose {preset!r} must contain "
                            f"{len(self.controls)} independent joint positions"
                        )
                    qpos = self.home.copy()
                    for control, value in zip(self.controls, values, strict=True):
                        qpos[model.jnt_qposadr[control.joint]] = value
                    self.poses[preset] = qpos
        if not self.poses:
            self.poses = {"default": self.home.copy()}
        self.default_pose = (
            "default"
            if "default" in self.poses
            else "home"
            if "home" in self.poses
            else next(iter(self.poses))
        )
        self.selected_pose = self.default_pose
        self.apply_pose(self.default_pose)
        self.home = self.data.qpos.copy()
        # Continuous-joint sliders must also reach every configured preset.
        self.controls = [
            JointControl(
                control.joint,
                control.name,
                control.scale,
                control.unit,
                min(
                    control.lower,
                    *(q[model.jnt_qposadr[control.joint]] for q in self.poses.values()),
                ),
                max(
                    control.upper,
                    *(q[model.jnt_qposadr[control.joint]] for q in self.poses.values()),
                ),
            )
            if not model.jnt_limited[control.joint]
            else control
            for control in self.controls
        ]
        self.set_values({})

    def values(self):
        """Return independent joint coordinates in native radians or meters."""
        return {
            control.name: float(self.data.qpos[self.model.jnt_qposadr[control.joint]])
            for control in self.controls
        }

    def set_values(self, values):
        """Set independent coordinates and evaluate active joint equalities."""
        model = self.model
        for control in self.controls:
            if control.name in values:
                self.data.qpos[model.jnt_qposadr[control.joint]] = values[control.name]
        for equality in self.relations:
            first, second = model.eq_obj1id[equality], model.eq_obj2id[equality]
            first_address = model.jnt_qposadr[first]
            displacement = 0.0
            if second >= 0:
                second_address = model.jnt_qposadr[second]
                displacement = self.data.qpos[second_address] - model.qpos0[second_address]
            self.data.qpos[first_address] = model.qpos0[
                first_address
            ] + np.polynomial.polynomial.polyval(displacement, model.eq_data[equality, :5])
        mujoco.mj_forward(model, self.data)

    def reset_home(self):
        self.apply_pose(self.default_pose)

    def apply_pose(self, name):
        """Apply a named YAML pose or an XML keyframe and update coupled joints."""
        self.data.qpos[:] = self.poses[name]
        self.selected_pose = name
        self.set_values({})

    def reset_zero(self):
        """Restore a named zero preset when the asset defines one."""
        if self.zero is None:
            raise ValueError(f"Robot {self.name!r} has no zero preset")
        if "zero" in self.poses:
            self.apply_pose("zero")
            return
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.zero)
        self.set_values({})


class ModelView:
    """Own one robot's scene handles and lazily loaded CAD details."""

    def __init__(self, preview, server):
        self.preview = preview
        self.server = server
        self.root_path = f"/models/{preview.name}"
        self.root = server.scene.add_frame(self.root_path, show_axes=False)
        self.frames = {
            body: server.scene.add_frame(f"{self.root_path}/body_{body}", show_axes=False)
            for body in range(preview.model.nbody)
        }
        self.coordinate_frames = {}
        self.build_coordinate_frames()
        self.visuals, self.details, self.collisions = [], [], []
        self.detail_built = False
        self.build_visuals(detail=False)
        self.build_collisions()
        self.sliders = {}
        with server.gui.add_folder(f"Pose: {preview.name}", order=0) as folder:
            self.folder = folder
            for control in preview.controls:
                value = preview.values()[control.name]
                self.sliders[control.name] = server.gui.add_slider(
                    f"{control.name} ({control.unit})",
                    float(control.lower * control.scale),
                    float(control.upper * control.scale),
                    0.1,
                    float(value * control.scale),
                    hint="Independent joint coordinate; coupled joints follow automatically.",
                )

    def body_name(self, body):
        return self.preview.model.body(body).name or f"body_{body}"

    def build_coordinate_frames(self):
        """Attach local XYZ axes at joint pivots and end-effector sites."""
        model = self.preview.model
        frames = []
        for joint in range(model.njnt):
            if int(model.jnt_type[joint]) not in (
                mujoco.mjtJoint.mjJNT_HINGE,
                mujoco.mjtJoint.mjJNT_SLIDE,
            ):
                continue
            frames.append(
                (
                    f"joint_{joint}",
                    local_name(model.joint(joint).name),
                    int(model.jnt_bodyid[joint]),
                    model.jnt_pos[joint],
                    (1.0, 0.0, 0.0, 0.0),
                )
            )
        for site in range(model.nsite):
            name = local_name(model.site(site).name)
            if name in ("ee_site", "grasp"):
                frames.append(
                    (
                        f"site_{site}",
                        name,
                        int(model.site_bodyid[site]),
                        model.site_pos[site],
                        model.site_quat[site],
                    )
                )
        for identifier, name, body, position, quaternion in frames:
            path = f"{self.root_path}/body_{body}/{identifier}_coordinates"
            handle = self.server.scene.add_frame(
                path,
                position=position,
                wxyz=quaternion,
                axes_length=0.07,
                axes_radius=0.0015,
                visible=False,
            )
            self.server.scene.add_label(f"{path}/label", name, position=(0.0, 0.0, 0.085))
            self.coordinate_frames[name] = handle

    def build_visuals(self, *, detail):
        """Group visual geoms by body and color; defer components under 40 mm."""
        model = self.preview.model
        groups = {}
        visual = np.flatnonzero(
            (model.geom_group == 1)
            | ((model.geom_group != 3) & (model.geom_type != int(mujoco.mjtGeom.mjGEOM_PLANE)))
        )
        self.has_details = bool(np.any(model.geom_rbound[visual] < 0.02))
        for geom in visual:
            if (model.geom_rbound[geom] < 0.02) != detail:
                continue
            body = int(model.geom_bodyid[geom])
            material = model.geom_matid[geom]
            rgba = model.mat_rgba[material] if material >= 0 else model.geom_rgba[geom]
            color = tuple(int(v) for v in np.clip(rgba[:3] * 255, 0, 255))
            mesh = geom_mesh(model, geom)
            rotation = np.empty(9)
            mujoco.mju_quat2Mat(rotation, model.geom_quat[geom])
            transform = Transform(rotation.reshape(3, 3), model.geom_pos[geom])
            mesh.vertices = transform.apply(mesh.vertices)
            groups.setdefault((body, color), []).append(mesh)
        target = self.details if detail else self.visuals
        for index, ((body, color), meshes) in enumerate(groups.items()):
            mesh = cast(trimesh.Trimesh, trimesh.util.concatenate(meshes))
            target.append(
                self.server.scene.add_mesh_simple(
                    f"{self.root_path}/body_{body}/{'detail' if detail else 'visual'}_{index}",
                    mesh.vertices.astype(np.float32),
                    mesh.faces.astype(np.uint32),
                    color=color,
                )
            )

    def build_collisions(self):
        model = self.preview.model
        collision_ids = np.flatnonzero(
            (model.geom_group == 3) | (model.geom_contype != 0) | (model.geom_conaffinity != 0)
        )
        for geom in collision_ids:
            if int(model.geom_type[geom]) == mujoco.mjtGeom.mjGEOM_PLANE:
                continue
            body = int(model.geom_bodyid[geom])
            mesh = geom_mesh(model, geom, convex=True)
            is_capsule = int(model.geom_type[geom]) == mujoco.mjtGeom.mjGEOM_CAPSULE
            color = CAPSULE_COLOR if is_capsule else (160, 170, 185)
            handle = self.server.scene.add_mesh_simple(
                f"{self.root_path}/body_{body}/collision_{geom}",
                mesh.vertices.astype(np.float32),
                mesh.faces.astype(np.uint32),
                color=color,
                opacity=0.35,
                position=model.geom_pos[geom],
                wxyz=model.geom_quat[geom],
            )
            self.collisions.append((geom, body, color, handle))


class CollisionViewer:
    """Inspect an asset by robot name; discover all other assets for comparison."""

    def __init__(self, robot, host="127.0.0.1", port=8080):
        preview = RobotPreview(robot)
        self.lock = threading.RLock()
        self.switching = False
        self.server = viser.ViserServer(host=host, port=port, label="Robot collision inspector")
        self.server.scene.set_up_direction("+z")
        self.server.scene.add_grid("/grid", width=2, height=2, cell_size=0.1, section_size=0.5)
        self.server.scene.add_frame("/world", axes_length=0.12, axes_radius=0.003)
        self.views = {robot: ModelView(preview, self.server)}
        self.current = self.views[robot]
        with self.server.gui.add_folder("Display", order=-1):
            self.robot = self.server.gui.add_dropdown("Model", robot_names(), initial_value=robot)
            self.pose = self.server.gui.add_dropdown(
                "Pose preset", tuple(preview.poses), initial_value=preview.default_pose
            )
            self.reset = self.server.gui.add_button("Reset pose")
            self.show_visual = self.server.gui.add_checkbox("CAD", True)
            self.full_cad = self.server.gui.add_checkbox(
                "Full CAD details", False, visible=self.current.has_details
            )
            self.show_collision = self.server.gui.add_checkbox("Collision geometry", True)
            self.opacity = self.server.gui.add_slider("Collision opacity", 0.05, 1.0, 0.05, 0.35)
            self.wireframe = self.server.gui.add_checkbox("Collision wireframe", False)
            self.show_coordinates = self.server.gui.add_dropdown(
                "Coordinate frames",
                self.coordinate_options(),
                initial_value="Off",
                hint="Local XYZ axes at joint origins and tool frames: X red, Y green, Z blue.",
            )
            self.focus = self.server.gui.add_dropdown("Collision body", self.body_options())
        for handle in (
            self.show_visual,
            self.full_cad,
            self.show_collision,
            self.opacity,
            self.wireframe,
            self.show_coordinates,
            self.focus,
        ):
            handle.on_update(self.update)
        for slider in self.current.sliders.values():
            slider.on_update(self.update)
        self.robot.on_update(self.switch_model)
        self.pose.on_update(self.select_pose)
        self.reset.on_click(self.select_pose)
        self.update()

    def body_options(self):
        return (
            "All",
            *dict.fromkeys(
                self.current.body_name(body) for _, body, _, _ in self.current.collisions
            ),
        )

    def coordinate_options(self):
        return ("Off", "All", *self.current.coordinate_frames)

    def switch_model(self, _event=None):
        with self.lock:
            name = self.robot.value
            if self.switching or name == self.current.preview.name:
                return
            self.switching = True
            try:
                values = self.current.preview.values()
                if name not in self.views:
                    self.views[name] = ModelView(RobotPreview(name), self.server)
                    for slider in self.views[name].sliders.values():
                        slider.on_update(self.update)
                self.current = self.views[name]
                self.full_cad.visible = self.current.has_details
                for control in self.current.preview.controls:
                    if control.name in values:
                        value = np.clip(values[control.name], control.lower, control.upper)
                        self.current.sliders[control.name].value = float(value * control.scale)
                self.focus.options = self.body_options()
                self.focus.value = "All"
                self.pose.options = tuple(self.current.preview.poses)
                self.pose.value = self.current.preview.selected_pose
                selection = self.show_coordinates.value
                self.show_coordinates.options = self.coordinate_options()
                self.show_coordinates.value = (
                    selection if selection in self.coordinate_options() else "Off"
                )
            finally:
                self.switching = False
            self.update()

    def reset_home(self, _event=None):
        self.reset_pose(self.current.preview.default_pose)

    def reset_zero(self, _event=None):
        self.reset_pose("zero")

    def select_pose(self, _event=None):
        if not self.switching:
            self.reset_pose(self.pose.value)

    def reset_pose(self, preset):
        with self.lock:
            self.switching = True
            try:
                self.current.preview.apply_pose(preset)
                self.pose.value = preset
                values = self.current.preview.values()
                for control in self.current.preview.controls:
                    self.current.sliders[control.name].value = values[control.name] * control.scale
            finally:
                self.switching = False
            self.update()

    def update(self, _event=None):
        with self.lock:
            if self.switching:
                return
            view = self.current
            preview = view.preview
            data = preview.data
            preview.set_values(
                {
                    control.name: view.sliders[control.name].value / control.scale
                    for control in preview.controls
                }
            )
            penetrating = set()
            for contact in data.contact:
                if contact.dist >= -0.0001:
                    continue
                penetrating.update((int(contact.geom1), int(contact.geom2)))
            if self.full_cad.value and view.has_details and not view.detail_built:
                view.build_visuals(detail=True)
                view.detail_built = True
            with self.server.atomic():
                for cached in self.views.values():
                    cached.root.visible = cached is view
                    cached.folder.visible = cached is view
                for body, frame in view.frames.items():
                    pose = Transform(data.xmat[body].reshape(3, 3), data.xpos[body]).as_xyzquat()
                    frame.position, frame.wxyz = pose[:3], pose[3:]
                for name, frame in view.coordinate_frames.items():
                    frame.visible = self.show_coordinates.value in ("All", name)
                for handle in view.visuals:
                    handle.visible = self.show_visual.value
                for handle in view.details:
                    handle.visible = self.show_visual.value and self.full_cad.value
                for geom, body, color, handle in view.collisions:
                    selected = self.focus.value in ("All", view.body_name(body))
                    handle.visible = self.show_collision.value and selected
                    handle.opacity = self.opacity.value
                    handle.wireframe = self.wireframe.value
                    handle.color = CAPSULE_COLOR if geom in penetrating else color


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--robot", choices=robot_names(), default="forte2")
    args = parser.parse_args()
    viewer = CollisionViewer(args.robot, host=args.host, port=args.port)
    print(f"Open http://{args.host}:{viewer.server.get_port()} to inspect the robot.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        viewer.server.stop()


if __name__ == "__main__":
    main()
