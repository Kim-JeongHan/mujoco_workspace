"""Regenerate the ForteV1 RobStride MuJoCo asset from its supplied CAD archive."""

import json
import shutil
import tempfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from mujoco_lab.utils.logger import Logger

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = Path.home() / "Downloads/ForteV1_RobStride.zip"
SOURCE = ROOT / "third_party/fortev1-robstride"
MANIFEST = ROOT / "third_party/fortev1-robstride.SOURCE.json"
ASSET = ROOT / "src/mujoco_lab/assets/robot/forte"
SOURCE_URDF = SOURCE / "fortev1_robstride/urdf/fortev1_robstride.urdf"
NAMES = {
    "shoulderyaw": "shoulder_yaw",
    "shoulderpich": "shoulder_pitch",
    "shoulderroll": "shoulder_roll",
    "revolute_1": "elbow_pitch",
    "revolute_16": "lower_arm_roll",
    "revolute_17": "wrist_pitch",
    "revolute_9": "wrist_roll",
    "slider_1_1": "gripper_left_joint",
    "slider_2_1": "gripper_right_joint",
}
ARM = list(NAMES.values())[:7]
# Define the simulated zero pose; physical encoder zeros have not been measured.
MODEL_ZERO_POSE = {
    "shoulder_yaw": 0.031713138037,
    "shoulder_pitch": -0.392582419038,
    "shoulder_roll": -0.35409561867,
    "elbow_pitch": -0.005483411844,
    "wrist_pitch": 2.53044826746345,
    "wrist_roll": 0.0548281178675077,
}
FOREARM_VISUAL_ROLL = 0.9461201993
# Reorient the CAD wrist assembly around the forearm without moving its pivot.
# This makes the long axis of the purple Part_62 wrist link vertical at zero.
WRIST_MOUNT_ROLL = -0.4689923000897805
# Reorient the gripper about the wrist-roll pivot so its tool link points down.
# The wrist-roll axis is re-expressed in the new body frame below.
GRIPPER_MOUNT_QUAT = (
    0.9978939966412655,
    -0.06219329765287789,
    0.018427098689429634,
    -0.00008501924807374775,
)

# Separate component hulls preserve gaps between assemblies. Tiny fasteners and
# internal electronics stay visual-only; the straight forearm uses a capsule.
COLLISION_PARTS = {
    "base_link": {"bottom_base", "EL05_merged", "act_drum"},
    "main_drum": {
        "main_drum",
        "LeftColumn",
        "RightColumn",
        "RS60_merged",
        "Part_41",
        "Part_42",
    },
    "lefthinge": {
        "LeftHinge",
        "ShoulderRoll",
        "RS60_merged",
        "HingeCover",
        "ShoulderRollActAdapter",
        "ShoulderPitchActDrum",
        "Part_30_1",
        "Part_46",
    },
    "upperarmright": {
        "UpperArmRight",
        "UpperArmLeft",
        "RS60_merged",
        "ElbowPCB_cover",
        "Part_64",
        "Part_65",
        "Part_63",
        "Part_66",
        "Part_67",
    },
    "elbowlink": {"ElbowLink", "EL05_merged", "Part_56"},
    "spur_gear__40_teeth_": {"Part_58", "Spur_gear__40_teeth_", "Spur_gear__40_teeth__2"},
    "spiral_gear_2": {"Part_61", "Part_62", "EL05_merged", "Spiral_Gear_2"},
    "part_8_2": {"Part_8_2", "Part_2_5", "EL05_merged_1", "Part_4_2", "Part_5_2", "Part_6_3"},
    "part_1_35": {"Part_7_2", "Part_1_9"},
    "part_1_40": {"Part_7_2", "Part_1_9"},
}


def digest(data: bytes) -> str:
    return sha256(data).hexdigest()


def unpack_source() -> None:
    """Retain each archive member byte for byte and record its SHA-256."""
    files = {}
    with ZipFile(ARCHIVE) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            relative = Path(member.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe archive member: {member.filename}")
            data = archive.read(member)
            target = SOURCE / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            files[str(relative)] = digest(data)
    manifest = {
        "source_archive": ARCHIVE.name,
        "source_archive_sha256": digest(ARCHIVE.read_bytes()),
        "snapshot": "Complete, byte-identical ForteV1_RobStride.zip contents.",
        "license": "No license file or declaration found in the supplied archive.",
        "removed_files": [],
        "files_sha256": files,
        "runtime_asset": "src/mujoco_lab/assets/robot/forte/robot.xml",
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")


def origin(element):
    if element is None:
        return np.eye(3), np.zeros(3)
    xyz = np.fromstring(element.get("xyz", "0 0 0"), sep=" ")
    rpy = np.fromstring(element.get("rpy", "0 0 0"), sep=" ")
    return Rotation.from_euler("xyz", rpy).as_matrix(), xyz


def number(values):
    return " ".join(f"{value:.12g}" for value in values)


def prepare_urdf() -> tuple[ET.Element, dict]:
    """Freeze unselected mates and sum each rigid group's CAD inertials."""
    root = ET.parse(SOURCE_URDF).getroot()
    links = {link.get("name"): link for link in root.findall("link")}
    joints = root.findall("joint")
    by_child = {joint.find("child").get("link"): joint for joint in joints}
    poses = {"bottom_base": (np.eye(3), np.zeros(3))}

    def pose(link):
        if link not in poses:
            joint = by_child[link]
            parent_r, parent_p = pose(joint.find("parent").get("link"))
            joint_r, joint_p = origin(joint.find("origin"))
            poses[link] = parent_r @ joint_r, parent_r @ joint_p + parent_p
        return poses[link]

    def anchor(link):
        while link in by_child:
            joint = by_child[link]
            if joint.get("name") in NAMES:
                return link
            link = joint.find("parent").get("link")
        return "bottom_base"

    groups = defaultdict(list)
    for name, link in links.items():
        inertial = link.find("inertial")
        if inertial is None:
            continue
        mass = float(inertial.find("mass").get("value"))
        tensor = inertial.find("inertia")
        matrix = np.array(
            [
                [float(tensor.get("ixx")), float(tensor.get("ixy")), float(tensor.get("ixz"))],
                [float(tensor.get("ixy")), float(tensor.get("iyy")), float(tensor.get("iyz"))],
                [float(tensor.get("ixz")), float(tensor.get("iyz")), float(tensor.get("izz"))],
            ]
        )
        link_r, link_p = pose(name)
        inertia_r, inertia_p = origin(inertial.find("origin"))
        anchor_name = anchor(name)
        anchor_r, anchor_p = pose(anchor_name)
        center = anchor_r.T @ (link_r @ inertia_p + link_p - anchor_p)
        rotation = anchor_r.T @ link_r @ inertia_r
        groups[anchor_name].append((mass, center, rotation @ matrix @ rotation.T))
        link.remove(inertial)

    aggregate = {}
    for name, entries in groups.items():
        mass = sum(entry[0] for entry in entries)
        center = sum(entry[0] * entry[1] for entry in entries) / mass
        matrix = np.zeros((3, 3))
        for part_mass, part_center, part_inertia in entries:
            offset = part_center - center
            matrix += part_inertia + part_mass * (
                np.dot(offset, offset) * np.eye(3) - np.outer(offset, offset)
            )
        if np.linalg.eigvalsh(matrix).min() <= 0:
            raise ValueError(f"Nonpositive aggregate inertia for {name}")
        aggregate[name] = (mass, center, matrix)
        inertial = ET.SubElement(links[name], "inertial")
        ET.SubElement(inertial, "origin", xyz=number(center), rpy="0 0 0")
        ET.SubElement(inertial, "mass", value=f"{mass:.12g}")
        ET.SubElement(
            inertial,
            "inertia",
            **{
                key: f"{value:.12g}"
                for key, value in zip(
                    ("ixx", "ixy", "ixz", "iyy", "iyz", "izz"),
                    (
                        matrix[0, 0],
                        matrix[0, 1],
                        matrix[0, 2],
                        matrix[1, 1],
                        matrix[1, 2],
                        matrix[2, 2],
                    ),
                    strict=True,
                )
            },
        )

    for joint in joints:
        source_name = joint.get("name")
        if source_name not in NAMES:
            joint.set("type", "fixed")
            continue
        joint.set("name", NAMES[source_name])
        if source_name.startswith("slider_"):
            joint.find("limit").set("lower", "-0.02")
            joint.find("limit").set("upper", "0")

    ET.SubElement(ET.SubElement(root, "mujoco"), "compiler", discardvisual="false")
    for mesh in root.iter("mesh"):
        filename = mesh.get("filename")
        if filename and filename.startswith("package://fortev1_robstride/meshes/"):
            mesh.set("filename", str(SOURCE / "fortev1_robstride/meshes" / Path(filename).name))
    return root, aggregate


def add_collisions(bodies):
    """Reuse selected CAD component hulls in their exact visual frames."""
    for name, parts in COLLISION_PARTS.items():
        body = bodies[name]
        visuals = [geom for geom in body.findall("geom") if geom.get("mesh") in parts]
        for index, visual in enumerate(visuals):
            geom = deepcopy(visual)
            geom.attrib.update(
                name=f"{name}_collision_{index}",
                contype="1",
                conaffinity="1",
                group="3",
                rgba="0.4 0.4 0.45 0.3",
            )
            pos = np.fromstring(geom.get("pos", "0 0 0"), sep=" ")
            if name == "base_link":
                # Avoid a roundoff contact with the mounting surface at home.
                pos[2] += 0.0001
            elif name in ("part_1_35", "part_1_40"):
                # The flat contact pads stand 0.5 mm proud of the outer hulls.
                pos[0] += 0.0005 if name == "part_1_35" else -0.0005
            geom.set("pos", number(pos))
            body.append(geom)

    # Part_57 is a 285.5 mm tube with an approximately 18.5 mm outer radius.
    ET.SubElement(
        bodies["spur_gear__40_teeth_"],
        "geom",
        name="forearm_collision",
        type="capsule",
        fromto="0 0 -0.019 0 0 -0.2665",
        size="0.019",
        group="3",
        rgba="0.4 0.4 0.45 0.3",
    )


def generate_runtime(urdf, aggregate):
    with tempfile.TemporaryDirectory(prefix="forte-robstride-build-") as temporary:
        urdf_file = Path(temporary) / "source.urdf"
        mjcf_file = Path(temporary) / "compiled.xml"
        ET.ElementTree(urdf).write(urdf_file, encoding="unicode")
        model = mujoco.MjModel.from_xml_path(str(urdf_file))
        mujoco.mj_saveLastXML(str(mjcf_file), model)
        xml = ET.parse(mjcf_file).getroot()

    xml.set("model", "forte_robstride")
    compiler = xml.find("compiler")
    compiler.set("angle", "radian")
    ET.SubElement(xml, "option", timestep="0.002", integrator="implicitfast")
    meshes = ASSET / "meshes"
    if meshes.exists():
        shutil.rmtree(meshes)
    meshes.mkdir(parents=True)
    for mesh in xml.findall("asset/mesh"):
        source_file = Path(mesh.get("file"))
        shutil.copy2(source_file, meshes / source_file.name)
        mesh.set("file", f"meshes/{source_file.name}")

    world = xml.find("worldbody")
    base = ET.Element("body", name="base_link", pos="0 0 0.058")
    for child in list(world):
        world.remove(child)
        base.append(child)
    world.append(base)
    base_mass, base_center, base_matrix = aggregate["bottom_base"]
    base.insert(
        0,
        ET.Element(
            "inertial",
            mass=f"{base_mass:.12g}",
            pos=number(base_center),
            fullinertia=number(
                (
                    base_matrix[0, 0],
                    base_matrix[1, 1],
                    base_matrix[2, 2],
                    base_matrix[0, 1],
                    base_matrix[0, 2],
                    base_matrix[1, 2],
                )
            ),
        ),
    )
    bodies = {body.get("name"): body for body in xml.iter("body")}
    # mj_saveLastXML rounds imported inertials; restore the full-precision CAD sums.
    for source_name, (mass, center, matrix) in aggregate.items():
        if source_name == "bottom_base":
            continue
        body = bodies[source_name]
        old = body.find("inertial")
        if old is not None:
            body.remove(old)
        body.insert(
            0,
            ET.Element(
                "inertial",
                mass=f"{mass:.12g}",
                pos=number(center),
                fullinertia=number(
                    (
                        matrix[0, 0],
                        matrix[1, 1],
                        matrix[2, 2],
                        matrix[0, 1],
                        matrix[0, 2],
                        matrix[1, 2],
                    )
                ),
            ),
        )
    source_bodies = [
        "main_drum",
        "lefthinge",
        "upperarmright",
        "elbowlink",
        "spur_gear__40_teeth_",
        "spiral_gear_2",
        "part_8_2",
    ]
    contact = ET.SubElement(xml, "contact")
    for parent, child in zip(["base_link", *source_bodies], source_bodies, strict=True):
        ET.SubElement(contact, "exclude", body1=parent, body2=child)
    add_collisions(bodies)

    for body in bodies.values():
        for joint in body.findall("joint"):
            joint.attrib.pop("actuatorfrcrange", None)
            if joint.get("name") in ARM:
                joint.set("armature", "0.01")
                joint.set("damping", "0.05")
            if joint.get("name") in MODEL_ZERO_POSE:
                offset = MODEL_ZERO_POSE[joint.get("name")]
                axis = np.fromstring(joint.get("axis", "0 0 1"), sep=" ")
                quat = np.fromstring(body.get("quat", "1 0 0 0"), sep=" ")
                base_rotation = Rotation.from_quat(np.r_[quat[1:], quat[0]])
                x, y, z, w = (base_rotation * Rotation.from_rotvec(offset * axis)).as_quat()
                body.set("quat", number((w, x, y, z)))
                if joint.get("range") is not None:
                    limits = np.fromstring(joint.get("range"), sep=" ") - offset
                    joint.set("range", number(limits))
    # Keep the forearm at its CAD roll and the wrist pivot on the forearm axis.
    forearm_roll = bodies["spur_gear__40_teeth_"].find("joint")
    roll_axis = np.fromstring(forearm_roll.get("axis", "0 0 1"), sep=" ")
    position_correction = Rotation.from_rotvec(FOREARM_VISUAL_ROLL * roll_axis)
    orientation_correction = Rotation.from_rotvec(
        (FOREARM_VISUAL_ROLL + WRIST_MOUNT_ROLL) * roll_axis
    )
    wrist = bodies["spiral_gear_2"]
    wrist.set("pos", number(position_correction.apply(np.fromstring(wrist.get("pos"), sep=" "))))
    quat = np.fromstring(wrist.get("quat"), sep=" ")
    wrist_rotation = Rotation.from_quat(np.r_[quat[1:], quat[0]])
    x, y, z, w = (orientation_correction * wrist_rotation).as_quat()
    wrist.set("quat", number((w, x, y, z)))
    # The gripper is rigidly mounted to the wrist-roll pivot. Preserve that
    # pivot's world rotation axis when changing the gripper body frame.
    gripper = bodies["part_8_2"]
    correction = Rotation.from_quat(np.r_[GRIPPER_MOUNT_QUAT[1:], GRIPPER_MOUNT_QUAT[0]])
    quat = np.fromstring(gripper.get("quat"), sep=" ")
    gripper_rotation = Rotation.from_quat(np.r_[quat[1:], quat[0]])
    x, y, z, w = (gripper_rotation * correction).as_quat()
    gripper.set("quat", number((w, x, y, z)))
    wrist_roll = gripper.find("joint")
    axis = np.fromstring(wrist_roll.get("axis", "0 0 1"), sep=" ")
    wrist_roll.set("axis", number(correction.inv().apply(axis)))
    # The archive defines no TCP; this is a simulation reference on its tool link.
    ET.SubElement(
        bodies["part_8_2"],
        "site",
        name="ee_site",
        pos="0.0487 0.1643 -0.0146",
        size="0.008",
        rgba="1 0.2 0.1 1",
    )

    # The pads lie within the broad planar faces of the two CAD finger paddles.
    ET.SubElement(
        bodies["part_1_35"],
        "geom",
        name="gripper_left_pad",
        type="box",
        pos="-0.027 -0.044 -0.010106",
        quat="0.305268 0.952267 0 0",
        size="0.002 0.015 0.01",
        rgba="0.12 0.12 0.13 0.85",
        friction="1 0.02 0.001",
        group="3",
        condim="4",
        priority="1",
        solref="0.006 1",
        solimp="0.995 0.999 0.001",
    )
    ET.SubElement(
        bodies["part_1_40"],
        "geom",
        name="gripper_right_pad",
        type="box",
        pos="0.027 -0.044 0.010106",
        quat="0.305268 0.952267 0 0",
        size="0.002 0.015 0.01",
        rgba="0.12 0.12 0.13 0.85",
        friction="1 0.02 0.001",
        group="3",
        condim="4",
        priority="1",
        solref="0.006 1",
        solimp="0.995 0.999 0.001",
    )
    # Open-pad midpoint, 10 mm toward the fingertips to clear the palm hull.
    # +X crosses the jaws; +Z points along the fingers toward their tips.
    ET.SubElement(
        bodies["part_8_2"],
        "site",
        name="grasp",
        pos="0.0398841241051 0.134757709764 -0.0268140270887",
        quat="-0.452770631242 0.880023262695 -0.127496711913 0.0655926905333",
        size="0.004",
        rgba="0.1 0.8 0.2 1",
    )

    equality = ET.SubElement(xml, "equality")
    ET.SubElement(
        equality,
        "joint",
        name="gripper_parallel",
        joint1="gripper_left_joint",
        joint2="gripper_right_joint",
        polycoef="0 1 0 0 0",
        solref="0.004 1",
        solimp="0.99 0.999 0.001",
    )
    actuators = ET.SubElement(xml, "actuator")
    for index, name in enumerate(ARM):
        torque = 87 if index < 5 else 12
        ET.SubElement(
            actuators,
            "motor",
            name=f"{name}_motor",
            joint=name,
            gear="1",
            ctrlrange=f"-{torque} {torque}",
        )
    ET.SubElement(
        actuators,
        "position",
        name="gripper_motor",
        joint="gripper_left_joint",
        kp="2000",
        kv="5",
        ctrlrange="-0.02 0",
        forcerange="-20 20",
    )
    ET.SubElement(ET.SubElement(xml, "keyframe"), "key", name="home", qpos="0 0 0 0 0 0 0 0 0")
    ET.indent(xml, space="  ")
    ET.ElementTree(xml).write(ASSET / "robot.xml", encoding="utf-8", xml_declaration=True)


def main():
    unpack_source()
    urdf, aggregate = prepare_urdf()
    generate_runtime(urdf, aggregate)
    logger = Logger()
    logger.info(f"Source mass: {sum(group[0] for group in aggregate.values()):.9f} kg")
    logger.info(f"Wrote {ASSET / 'robot.xml'}")


if __name__ == "__main__":
    main()
