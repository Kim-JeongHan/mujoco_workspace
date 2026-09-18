"""Import the three requested robot descriptions from pinned upstream checkouts.

Run with ``uv run --group asset-import python scripts/import_robot_assets.py --help``.
The generated assets need neither ROS nor the asset-import dependency group at runtime.
"""

import argparse
import copy
import hashlib
import json
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import trimesh
import xacro

UR_COMMIT = "89bbe795f38a7ab00fb66fe8831dfff79dc99edf"
PANDA_COMMIT = "62335618e7f141755524836c738efe9c01396440"
ASSETS = Path(__file__).resolve().parents[1] / "src/mujoco_lab/assets"


def write_xml(root: ET.Element, path: Path) -> None:
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def numbers(values) -> str:
    return " ".join(f"{float(value):.10g}" for value in values)


def prepare_xacro(source: Path, package: str, destination: Path) -> Path:
    """Resolve the two known ROS package locations without requiring a ROS installation."""
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns(".git"))
    for path in destination.rglob("*.xacro"):
        path.write_text(path.read_text().replace(f"$(find {package})", str(destination)))
    return destination


def convert_visual(source: Path, destination: Path, prefix: str) -> list[dict]:
    """Bake Collada scene transforms and keep each material as a separate visual geom."""
    scene = trimesh.load_scene(source, process=False, ignore_broken=False)
    result = []
    for index, node in enumerate(sorted(scene.graph.nodes_geometry)):
        transform, geometry = scene.graph[node]
        mesh = scene.geometry[geometry].copy()
        mesh.apply_transform(transform)
        name = f"{prefix}_{index}"
        relative = Path("meshes/visual") / f"{name}.obj"
        text = trimesh.exchange.obj.export_obj(
            mesh,
            include_normals=True,
            include_color=False,
            include_texture=True,
            write_texture=False,
        )
        # MuJoCo material/texture bindings are written explicitly in MJCF below.
        text = (
            "\n".join(
                line for line in text.splitlines() if not line.startswith(("mtllib ", "usemtl "))
            )
            + "\n"
        )
        (destination / relative).write_text(text)
        material = mesh.visual.material
        color = getattr(material, "baseColorFactor", None)
        if color is None:
            color = [255, 255, 255, 255]
        item = {"name": name, "file": str(relative), "rgba": numbers(np.array(color) / 255)}
        image = getattr(material, "baseColorTexture", None)
        if image is not None:
            # Hash image content so every link can reuse the original texture, including logos.
            digest = hashlib.sha256(image.tobytes()).hexdigest()[:16]
            texture = Path("textures") / f"{digest}.png"
            if not (destination / texture).exists():
                image.save(destination / texture)
            item["texture"] = str(texture)
        result.append(item)
    return result


def import_robot(name: str, source: Path, prepared: Path, destination: Path) -> None:
    is_ur = name in {"ur20", "ur30"}
    package = "ur_description" if is_ur else "franka_panda_description"
    commit = UR_COMMIT if is_ur else PANDA_COMMIT
    actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True)
    if actual.strip() != commit:
        raise ValueError(f"{package} must be checked out at {commit}")
    for folder in ["meshes/visual", "meshes/collision", "textures", "source"]:
        (destination / folder).mkdir(parents=True, exist_ok=True)
    if is_ur:
        entrypoint = prepared / "urdf/ur.urdf.xacro"
        mappings = {"name": name, "ur_type": name}
    else:
        entrypoint = prepared / "robots/panda_arm_hand.urdf.xacro"
        mappings = {"load_gripper": "true", "load_gazebo": "false"}
    document = xacro.process_file(str(entrypoint), mappings=mappings)
    original = document.toprettyxml(indent="  ")
    (destination / "source/robot.urdf").write_text(original)
    robot = ET.fromstring(original)
    converted = {}
    visuals = {}
    provenance = []
    for link in robot.findall("link"):
        for visual in list(link.findall("visual")):
            mesh_element = visual.find("geometry/mesh")
            if mesh_element is None:
                continue
            relative = mesh_element.attrib["filename"].removeprefix(f"package://{package}/")
            dae = source / relative
            if relative not in converted:
                prefix = "visual_" + "_".join(Path(relative).with_suffix("").parts[1:])
                parts = convert_visual(dae, destination, prefix)
                converted[relative] = parts
                visuals.update((part["name"], part) for part in parts)
                raw = destination / "source" / relative
                raw.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(dae, raw)
                provenance.append(relative)
                # Keep original images beside the original DAE files as well.
                ns = {"c": "http://www.collada.org/2005/11/COLLADASchema"}
                for image in ET.parse(dae).findall(".//c:library_images//c:init_from", ns):
                    texture = dae.parent / image.text
                    shutil.copy2(texture, raw.parent / texture.name)
            link.remove(visual)
            for part in converted[relative]:
                clone = copy.deepcopy(visual)
                clone.set("name", f"{link.attrib['name']}_{part['name']}")
                clone.find("geometry/mesh").set("filename", part["file"])
                old_material = clone.find("material")
                if old_material is not None:
                    clone.remove(old_material)
                material = ET.SubElement(clone, "material", name=part["name"])
                ET.SubElement(material, "color", rgba=part["rgba"])
                link.append(clone)
        for collision in link.findall("collision"):
            mesh_element = collision.find("geometry/mesh")
            if mesh_element is None:
                continue
            relative = mesh_element.attrib["filename"].removeprefix(f"package://{package}/")
            original_mesh = source / relative
            target = Path("meshes/collision") / f"collision_{original_mesh.stem}.stl"
            shutil.copy2(original_mesh, destination / target)
            mesh_element.set("filename", str(target))
            provenance.append(relative)
    extension = ET.SubElement(robot, "mujoco")
    ET.SubElement(
        extension, "compiler", discardvisual="false", strippath="false", fusestatic="false"
    )
    adapted_urdf = destination / "robot.urdf"
    write_xml(robot, adapted_urdf)
    spec = mujoco.MjSpec.from_file(str(adapted_urdf))
    for mesh in spec.meshes:
        if mesh.name in visuals:
            mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
    source_model = spec.compile()
    mjcf = ET.fromstring(spec.to_xml())
    mjcf.set("model", name)
    asset = mjcf.find("asset")
    texture_names = {}
    for part in visuals.values():
        attributes = {"name": part["name"] + "_material", "rgba": part["rgba"]}
        if "texture" in part:
            texture_file = part["texture"]
            if texture_file not in texture_names:
                texture_name = "texture_" + Path(texture_file).stem
                texture_names[texture_file] = texture_name
                ET.SubElement(asset, "texture", name=texture_name, type="2d", file=texture_file)
            attributes["texture"] = texture_names[texture_file]
        ET.SubElement(asset, "material", attributes)
    for geom in mjcf.findall(".//worldbody//geom"):
        mesh_name = geom.get("mesh")
        if mesh_name in visuals:
            geom.set("material", mesh_name + "_material")
            geom.set("group", "1")
            geom.set("contype", "0")
            geom.set("conaffinity", "0")
            geom.attrib.pop("rgba", None)
        else:
            geom.set("group", "3")
    option = mjcf.find("option")
    if option is None:
        option = ET.SubElement(mjcf, "option")
    option.set("integrator", "implicitfast")
    option.set("timestep", "0.002")
    # Connected links share a mechanical joint and should not collide at that joint.
    contact = ET.SubElement(mjcf, "contact")
    body_names = {body.attrib["name"] for body in mjcf.findall(".//worldbody//body")}
    for joint in robot.findall("joint"):
        parent = joint.find("parent").attrib["link"]
        child = joint.find("child").attrib["link"]
        if parent in body_names and child in body_names:
            ET.SubElement(contact, "exclude", body1=parent, body2=child)
    actuator = ET.SubElement(mjcf, "actuator")
    equality = mjcf.find("equality")
    if equality is None:
        equality = ET.SubElement(mjcf, "equality")
    moving_joints = [j for j in robot.findall("joint") if j.attrib["type"] != "fixed"]
    moving_joints.sort(key=lambda joint: source_model.joint(joint.attrib["name"]).id)
    for joint in moving_joints:
        joint_name = joint.attrib["name"]
        mimic = joint.find("mimic")
        if mimic is not None:
            if not any(e.get("joint1") == joint_name for e in equality):
                ET.SubElement(
                    equality,
                    "joint",
                    joint1=joint_name,
                    joint2=mimic.attrib["joint"],
                    polycoef=f"{mimic.get('offset', '0')} {mimic.get('multiplier', '1')} 0 0 0",
                )
            continue
        limit = joint.find("limit")
        lower, upper = limit.get("lower", "-6.283185307"), limit.get("upper", "6.283185307")
        effort = float(limit.attrib["effort"])
        finger = joint.attrib["type"] == "prismatic"
        kp, kv = (400, 20) if finger else ((2000, 200) if is_ur else (300, 30))
        ET.SubElement(
            actuator,
            "position",
            name=joint_name,
            joint=joint_name,
            kp=str(kp),
            kv=str(kv),
            ctrlrange=f"{lower} {upper}",
            forcerange=numbers([-effort, effort]),
        )
    # Generic demonstration poses/controllers; these are not factory control parameters.
    if is_ur:
        home = dict(
            zip(
                [j.attrib["name"] for j in moving_joints],
                [0, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0],
                strict=True,
            )
        )
    else:
        home = {
            f"panda_joint{i + 1}": value
            for i, value in enumerate([0, -np.pi / 4, 0, -3 * np.pi / 4, 0, np.pi / 2, np.pi / 4])
        }
        home.update(panda_finger_joint1=0.04, panda_finger_joint2=0.04)
    robot_xml = destination / "robot.xml"
    write_xml(mjcf, robot_xml)
    compiled = mujoco.MjModel.from_xml_path(str(robot_xml))
    qpos = compiled.qpos0.copy()
    for joint_name, value in home.items():
        joint_id = mujoco.mj_name2id(compiled, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        qpos[compiled.jnt_qposadr[joint_id]] = value
    controls = [home[child.attrib["joint"]] for child in actuator]
    keys = ET.SubElement(mjcf, "keyframe")
    ET.SubElement(keys, "key", name="home", qpos=numbers(qpos), ctrl=numbers(controls))
    write_xml(mjcf, robot_xml)
    scene = ET.Element("mujoco", model=name + " scene")
    ET.SubElement(scene, "include", file="robot.xml")
    ET.SubElement(
        scene,
        "statistic",
        center="0 0 0.7" if is_ur else "0.3 0 0.4",
        extent="2.0" if is_ur else "1.0",
    )
    world = ET.SubElement(scene, "worldbody")
    ET.SubElement(world, "light", pos="1 -2 3", dir="-0.2 0.5 -1", diffuse="0.8 0.8 0.8")
    ET.SubElement(
        world,
        "geom",
        name="floor",
        type="plane",
        pos="0 0 -0.005",
        size="3 3 0.1",
        rgba="0.2 0.23 0.27 1",
    )
    write_xml(scene, destination / "scene.xml")
    shutil.copy2(source / "README.md", destination / "source/README.upstream.md")
    shutil.copy2(source / "package.xml", destination / "source/package.xml")
    if is_ur:
        shutil.copy2(source / "LICENSE", destination / "LICENSE.description.txt")
        shutil.copy2(source / f"meshes/{name}/LICENSE.txt", destination / "LICENSE.meshes.txt")
        shutil.copytree(
            source / f"config/{name}", destination / "source/config", dirs_exist_ok=True
        )
    manifest = {
        "repository": "https://github.com/UniversalRobots/Universal_Robots_ROS2_Description"
        if is_ur
        else "https://github.com/justagist/franka_panda_description",
        "commit": commit,
        "robot": name,
        "source_meshes": sorted(set(provenance)),
        "mesh_license": "Universal Robots Graphical Documentation Terms 1.01"
        if is_ur
        else "Not specified upstream; package.xml contains <license>TODO</license>",
        "conversion": "Expanded upstream Xacro; preserved origins, limits and supplied inertias; "
        "baked DAE node transforms into per-material OBJ meshes; retained colors and textures. "
        "Added position servos, a home keyframe and a floor. Collision STL shapes use MuJoCo's "
        "convex mesh collision representation; collisions between directly connected links "
        "are excluded. Panda finger mimic is a joint equality.",
    }
    (destination / "SOURCE.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Imported {name}: {compiled.nv} velocity DOFs, {compiled.nu} actuators", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ur-description", type=Path, required=True)
    parser.add_argument("--panda-description", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ASSETS)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="mujoco-xacro-") as directory:
        root = Path(directory)
        ur_source = args.ur_description.resolve()
        panda_source = args.panda_description.resolve()
        ur = prepare_xacro(ur_source, "ur_description", root / "ur_description")
        panda = prepare_xacro(panda_source, "franka_panda_description", root / "panda")
        for name in ["ur20", "ur30", "panda"]:
            source, prepared = (panda_source, panda) if name == "panda" else (ur_source, ur)
            import_robot(name, source, prepared, args.output.resolve() / name)


if __name__ == "__main__":
    main()
