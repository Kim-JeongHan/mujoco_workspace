"""Build a Forte variant with six arm capsules and two proximal cylinders."""

from __future__ import annotations

import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/mujoco_lab/assets/robot/forte"
TARGET = SOURCE.with_name("forte2")

# Body, geom name, mesh selection (None means all collision meshes), axis,
# nominal radius. All coordinates and radii are in the body's local frame.
FITS = (
    ("upperarmright", "upper_arm_capsule", None, 0, 0.075),
    ("elbowlink", "elbow_capsule", None, 0, 0.075),
    ("spur_gear__40_teeth_", "forearm_end_capsule", {"Part_58"}, 2, 0.040),
    (
        "spiral_gear_2",
        "wrist_link_capsule",
        {"Part_61", "Part_62", "Spiral_Gear_2"},
        2,
        0.055,
    ),
    ("spiral_gear_2", "wrist_motor_capsule", {"EL05_merged"}, 2, 0.045),
)
CYLINDERS = (
    ("main_drum", "base_link", "base_drum_cylinder", 2),
    ("lefthinge", "upperarmright", "shoulder_cylinder", 0),
)
# Named exterior parts supplement the structural collision meshes. Electronics,
# screws, washers, and unnamed internal detail are deliberately not retained.
VISUAL_EXTRAS = {
    "base_link": {"A1_adapter"},
    "main_drum": {
        "turntable_cap",
        "bottomcable_cover",
        "Cover_3DPrinting",
        "ShoulderCable",
        "ShoulderCable1",
        "ShoulderCable3",
        "ShoulderCable4",
    },
    "lefthinge": {"Belt1"},
    "elbowlink": {"Spur_gear__20_teeth_"},
    "spur_gear__40_teeth_": {"Part_57", "Part_59", "grommet"},
    "part_8_2": {"Part_2_4", "Part_3_2", "Part_6_2", "Part_7_3", "Part_8_3"},
}


def collision_vertices(model, data, body_name, meshes=None, reference_body=None):
    """Return selected collision vertices in the requested body-local frame."""
    body = model.body(body_name).id
    reference = model.body(reference_body).id if reference_body else body
    vertices, names = [], []
    for geom in np.flatnonzero((model.geom_bodyid == body) & (model.geom_group == 3)):
        if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mesh = model.geom_dataid[geom]
        name = model.mesh(mesh).name
        if meshes is not None and name not in meshes:
            continue
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        points = model.mesh_vert[start : start + count]
        points = points @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
        vertices.append((points - data.xpos[reference]) @ data.xmat[reference].reshape(3, 3))
        names.append(name)
    if not vertices:
        raise ValueError(f"No collision meshes found for {body_name}: {meshes}")
    return np.concatenate(vertices), names


def fit_capsule(points, axis, nominal_radius, margin=0.003):
    """Enclose vertices plus a radial margin using an axis-aligned segment."""
    center = (points.min(axis=0) + points.max(axis=0)) / 2
    delta = points - center
    radial = delta.copy()
    radial[:, axis] = 0
    distances = np.linalg.norm(radial, axis=1)
    required = float(distances.max()) + margin
    radius = max(nominal_radius, np.ceil(required / 0.005) * 0.005)
    # Shorten the cylindrical segment while keeping vertices inside the caps.
    reach = np.sqrt((radius - margin) ** 2 - distances**2)
    low = float(np.min(delta[:, axis] + reach))
    high = float(np.max(delta[:, axis] - reach))
    if low >= high:
        low, high = (low + high) / 2 - 0.0005, (low + high) / 2 + 0.0005
    start, end = center.copy(), center.copy()
    start[axis] += low
    end[axis] += high
    return start, end, float(radius)


def add_capsule(body, name, start, end, radius):
    """Append a collision-only geom without changing source body inertials."""
    ET.SubElement(
        body,
        "geom",
        name=name,
        type="capsule",
        fromto=" ".join(f"{v:.12g}" for v in np.r_[start, end]),
        size=f"{radius:.12g}",
        contype="1",
        conaffinity="1",
        density="0",
        group="3",
        rgba="1 0 0 0.3",
    )


def fit_cylinder(points, axis, margin=0.003):
    """Enclose vertices around an axis through the receiving body's origin.

    Keeping the radial center on the joint axis makes the proxy invariant
    under the joint's rotation, including when its source is a parent body.
    """
    radial = points.copy()
    radial[:, axis] = 0
    radius = float(np.ceil((np.linalg.norm(radial, axis=1).max() + margin) / 0.005) * 0.005)
    start, end = np.zeros(3), np.zeros(3)
    start[axis], end[axis] = points[:, axis].min() - margin, points[:, axis].max() + margin
    return start, end, radius


def add_cylinder(body, name, start, end, radius):
    """Append a flat-ended collision cylinder with explicit body inertials."""
    ET.SubElement(
        body,
        "geom",
        name=name,
        type="cylinder",
        fromto=" ".join(f"{v:.12g}" for v in np.r_[start, end]),
        size=f"{radius:.12g}",
        contype="1",
        conaffinity="1",
        density="0",
        group="3",
        rgba="1 0 0 0.3",
    )


def build():
    """Keep exterior visuals, dynamics, base hulls, and gripper collisions."""
    tree = ET.parse(SOURCE / "robot.xml")
    xml = tree.getroot()
    xml.set("model", "forte2_capsules")
    for mesh in xml.findall("asset/mesh"):
        mesh.set("file", "../forte/" + mesh.attrib["file"])
    bodies = {body.get("name"): body for body in xml.iter("body")}
    visual_parts = {
        name: {
            geom.get("mesh")
            for geom in body.findall("geom")
            if geom.get("group") == "3" and geom.get("mesh")
        }
        | VISUAL_EXTRAS.get(name, set())
        for name, body in bodies.items()
    }
    original_visual_count = sum(geom.get("group") == "1" for geom in xml.iter("geom"))
    for name, body in bodies.items():
        for geom in list(body.findall("geom")):
            if geom.get("group") == "1" and geom.get("mesh") not in visual_parts[name]:
                body.remove(geom)
    replaced = {fit[0] for fit in FITS} | {fit[0] for fit in CYLINDERS}
    for name in replaced:
        for geom in list(bodies[name].findall("geom")):
            if geom.get("group") == "3":
                bodies[name].remove(geom)

    model = mujoco.MjModel.from_xml_path(str(SOURCE / "robot.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    records = []
    for body, name, meshes, axis, nominal in FITS:
        points, source_meshes = collision_vertices(model, data, body, meshes)
        start, end, radius = fit_capsule(points, axis, nominal)
        add_capsule(bodies[body], name, start, end, radius)
        records.append(
            dict(
                body=body,
                name=name,
                fromto=np.r_[start, end].tolist(),
                radius=radius,
                source_meshes=source_meshes,
                radial_margin=0.003,
            )
        )

    # The tube capsule also encloses both proximal 21 mm radius spur gears.
    # The wider distal Part_58 connector has its own capsule above.
    start, end, radius = np.array([0, 0, 0.030]), np.array([0, 0, -0.2665]), 0.025
    body = "spur_gear__40_teeth_"
    name = "forearm_tube_capsule"
    add_capsule(bodies[body], name, start, end, radius)
    records.append(
        dict(
            body=body,
            name=name,
            fromto=np.r_[start, end].tolist(),
            radius=radius,
            source_meshes=["Spur_gear__40_teeth_", "Spur_gear__40_teeth__2"],
            source_geom="forearm_collision",
        )
    )
    cylinders = []
    for source_body, target_body, name, axis in CYLINDERS:
        points, source_meshes = collision_vertices(
            model, data, source_body, reference_body=target_body
        )
        start, end, radius = fit_cylinder(points, axis)
        add_cylinder(bodies[target_body], name, start, end, radius)
        cylinders.append(
            dict(
                body=target_body,
                source_body=source_body,
                name=name,
                axis=axis,
                fromto=np.r_[start, end].tolist(),
                radius=radius,
                source_meshes=source_meshes,
                margin=0.003,
            )
        )
    TARGET.mkdir(parents=True, exist_ok=True)
    used_meshes = {geom.get("mesh") for geom in xml.iter("geom") if geom.get("mesh")}
    assets = xml.find("asset")
    original_mesh_count = len(assets.findall("mesh"))
    for mesh in list(assets.findall("mesh")):
        if mesh.get("name") not in used_meshes:
            assets.remove(mesh)
    ET.indent(tree, space="  ")
    tree.write(TARGET / "robot.xml", encoding="utf-8", xml_declaration=True)
    shutil.copyfile(SOURCE / "robot.yaml", TARGET / "robot.yaml")
    (TARGET / "capsules.json").write_text(json.dumps(records, indent=2) + "\n")
    (TARGET / "cylinders.json").write_text(json.dumps(cylinders, indent=2) + "\n")
    retained_visual_count = sum(geom.get("group") == "1" for geom in xml.iter("geom"))
    (TARGET / "visuals.json").write_text(
        json.dumps(
            {
                "policy": "Retain structural collision mesh visuals and named exterior parts only.",
                "source_visual_count": original_visual_count,
                "retained_visual_count": retained_visual_count,
                "source_mesh_count": original_mesh_count,
                "retained_mesh_count": len(assets.findall("mesh")),
                "visual_meshes_by_body": {
                    name: sorted(
                        {
                            geom.get("mesh")
                            for geom in body.findall("geom")
                            if geom.get("group") == "1"
                        }
                    )
                    for name, body in bodies.items()
                },
            },
            indent=2,
        )
        + "\n"
    )
    for record in [*records, *cylinders]:
        print(f"{record['name']}: radius={record['radius'] * 1000:.0f} mm")
    print(f"Visual geoms: {original_visual_count} -> {retained_visual_count}")
    print(f"Mesh assets: {original_mesh_count} -> {len(assets.findall('mesh'))}")


if __name__ == "__main__":
    build()
