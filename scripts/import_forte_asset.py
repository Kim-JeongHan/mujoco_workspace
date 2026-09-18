"""Import the pinned native MuJoCo Forte model without Isaac Sim dependencies."""

import argparse
import hashlib
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

COMMIT = "d0846d26a940f630fbb6bd54aaa2de821aee8f14"
REPOSITORY = "https://github.com/jahirsadik/forte-arm-isaac-mujoco-demos"
OUTPUT = Path(__file__).resolve().parents[1] / "src/mujoco_lab/assets/forte"


def write_xml(root: ET.Element, path: Path) -> None:
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Upstream repository checkout")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True)
    if actual.strip() != COMMIT:
        raise ValueError(f"Check out upstream commit {COMMIT} before importing")

    original = source / "Forte_mujoco/model/forte.xml"
    meshes = source / "Forte_mujoco/assets/meshes"
    (output / "source").mkdir(parents=True, exist_ok=True)
    shutil.copytree(meshes, output / "meshes", dirs_exist_ok=True)
    shutil.copy2(original, output / "source/forte.xml")
    for filename in ["README.md", "HOW_TO_RUN.md"]:
        shutil.copy2(source / "Forte_mujoco" / filename, output / "source" / filename)
    shutil.copy2(source / "README.md", output / "source/README.repository.md")

    robot = ET.parse(original).getroot()
    robot.find("compiler").attrib.pop("meshdir")
    # The environment owns the floor, lights, and render-target dimensions.
    robot.remove(robot.find("visual"))
    world = robot.find("worldbody")
    for element in list(world):
        if element.tag == "light" or element.get("name") == "floor":
            world.remove(element)
    asset = robot.find("asset")
    for element in list(asset):
        if element.get("name") == "groundplane":
            asset.remove(element)
        elif element.tag == "mesh":
            element.set("file", "meshes/" + element.attrib["file"])
    write_xml(robot, output / "robot.xml")

    scene = ET.Element("mujoco", model="forte scene")
    ET.SubElement(scene, "include", file="robot.xml")
    ET.SubElement(scene, "statistic", center="0.4 0 0.4", extent="1.3")
    world = ET.SubElement(scene, "worldbody")
    ET.SubElement(world, "light", pos="1 -2 3", dir="-0.2 0.5 -1")
    ET.SubElement(
        world,
        "geom",
        name="floor",
        type="plane",
        pos="0 0 -0.005",
        size="3 3 0.1",
        rgba="0.2 0.23 0.27 1",
    )
    write_xml(scene, output / "scene.xml")

    source_files = [original, *sorted(meshes.glob("*.obj"))]
    manifest = {
        "repository": REPOSITORY,
        "commit": COMMIT,
        "source_model": "Forte_mujoco/model/forte.xml",
        "source_files": {
            str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_files
        },
        "control_mode": "Seven direct torque motors; home controls are zero.",
        "collision_geometry": "Upstream shoulder-yaw convex mesh and upper-arm, lower-arm, "
        "and wrist capsules. Other links have visual geometry only.",
        "license": "No explicit license file or license declaration was found for the "
        "MuJoCo model and meshes at this commit. No license is assigned locally.",
        "adaptations": [
            "Copied the upstream OBJ meshes without geometry changes.",
            "Rewrote mesh paths to be relative to this asset directory.",
            "Removed the upstream floor, lights, ground materials, and offscreen dimensions "
            "from robot.xml so environments can supply them.",
            "Preserved link transforms, inertias, joint limits, damping, armature, torque "
            "limits, collision exclusions, ee_site, and the home keyframe.",
            "Added a standalone scene.xml preview with a local floor and light.",
        ],
    }
    (output / "SOURCE.json").write_text(json.dumps(manifest, indent=2) + "\n")
    model = mujoco.MjModel.from_xml_path(str(output / "scene.xml"))
    assert (model.nq, model.nv, model.nu) == (7, 7, 7)
    print(f"Imported Forte: {model.nq} joints, {model.nu} torque motors")


if __name__ == "__main__":
    main()
