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

import mujoco
import numpy as np
import trimesh
import viser

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
        self.data.qpos[:] = self.home
        self.set_values({})

    def reset_zero(self):
        """Restore a named zero preset when the asset defines one."""
        if self.zero is None:
            raise ValueError(f"Robot {self.name!r} has no zero preset")
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
        self.visuals, self.details, self.collisions, self.labels = [], [], [], []
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

    def geom_name(self, geom):
        model = self.preview.model
        return model.geom(geom).name or f"{self.body_name(model.geom_bodyid[geom])}/geom_{geom}"

    def build_visuals(self, *, detail):
        """Group visual geoms by body and color; defer components under 40 mm."""
        model = self.preview.model
        groups = {}
        visual = np.flatnonzero(
            (model.geom_group == 1)
            | ((model.geom_group != 3) & (model.geom_type != int(mujoco.mjtGeom.mjGEOM_PLANE)))
        )
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
            mesh = trimesh.util.concatenate(meshes)
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
            if is_capsule:
                label = self.server.scene.add_label(
                    f"{self.root_path}/body_{body}/label_{geom}",
                    f"{self.geom_name(geom)} | r={model.geom_size[geom, 0] * 1000:.0f} mm",
                    position=model.geom_pos[geom],
                    visible=False,
                )
                self.labels.append((body, label))


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
            self.show_visual = self.server.gui.add_checkbox("CAD exterior", True)
            self.full_cad = self.server.gui.add_checkbox("Full CAD details", False)
            self.show_collision = self.server.gui.add_checkbox("Collision geometry", True)
            self.opacity = self.server.gui.add_slider("Collision opacity", 0.05, 1.0, 0.05, 0.35)
            self.wireframe = self.server.gui.add_checkbox("Collision wireframe", False)
            self.show_labels = self.server.gui.add_checkbox("Capsule labels", False)
            self.focus = self.server.gui.add_dropdown("Collision body", self.body_options())
            self.home = self.server.gui.add_button("Reset home")
            self.zero = self.server.gui.add_button("Zero pose", visible=preview.zero is not None)
        for handle in (
            self.show_visual,
            self.full_cad,
            self.show_collision,
            self.opacity,
            self.wireframe,
            self.show_labels,
            self.focus,
        ):
            handle.on_update(self.update)
        for slider in self.current.sliders.values():
            slider.on_update(self.update)
        self.robot.on_update(self.switch_model)
        self.home.on_click(self.reset_home)
        self.zero.on_click(self.reset_zero)
        self.update()

    def body_options(self):
        return (
            "All",
            *dict.fromkeys(
                self.current.body_name(body) for _, body, _, _ in self.current.collisions
            ),
        )

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
                for control in self.current.preview.controls:
                    if control.name in values:
                        value = np.clip(values[control.name], control.lower, control.upper)
                        self.current.sliders[control.name].value = float(value * control.scale)
                self.focus.options = self.body_options()
                self.focus.value = "All"
                self.zero.visible = self.current.preview.zero is not None
            finally:
                self.switching = False
            self.update()

    def reset_home(self, _event=None):
        self.reset_pose("home")

    def reset_zero(self, _event=None):
        self.reset_pose("zero")

    def reset_pose(self, preset):
        with self.lock:
            self.switching = True
            try:
                if preset == "zero":
                    self.current.preview.reset_zero()
                else:
                    self.current.preview.reset_home()
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
            if self.full_cad.value and not view.detail_built:
                view.build_visuals(detail=True)
                view.detail_built = True
            with self.server.atomic():
                for cached in self.views.values():
                    cached.root.visible = cached is view
                    cached.folder.visible = cached is view
                for body, frame in view.frames.items():
                    pose = Transform(data.xmat[body].reshape(3, 3), data.xpos[body]).as_xyzquat()
                    frame.position, frame.wxyz = pose[:3], pose[3:]
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
                for body, label in view.labels:
                    label.visible = (
                        self.show_collision.value
                        and self.show_labels.value
                        and self.focus.value in ("All", view.body_name(body))
                    )


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
