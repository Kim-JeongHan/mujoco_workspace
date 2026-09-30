"""Offline shell-launcher checks with a fake Colab CLI."""

import hashlib
import io
import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

FAKE_COLAB = """#!/usr/bin/env python3
import io
import json
import os
import shutil
import signal
import stat
import sys
import tarfile
from pathlib import Path

args = sys.argv[1:]
assert args[:2] == ["--auth", "oauth2"]
args = args[2:]
action = args[0]
home = Path(os.environ["FAKE_COLAB_HOME"])
mode = os.environ["FAKE_COLAB_MODE"].removeprefix("online_")
with (home / "calls.jsonl").open("a") as log:
    log.write(json.dumps(args) + "\\n")
if action == "upload":
    if args[4].endswith("-wandb-key"):
        assert stat.S_IMODE(Path(args[3]).stat().st_mode) == 0o600
        assert stat.S_IMODE(Path(args[3]).parent.stat().st_mode) == 0o700
        assert "WANDB_API_KEY" not in os.environ
    if mode == "upload_error":
        sys.exit(1)
    if mode == "key_upload_error" and args[4].endswith("-wandb-key"):
        shutil.copyfile(args[3], home / "uploads" / Path(args[4]).name)
        sys.exit(1)
    if mode == "term_upload":
        os.kill(os.getppid(), signal.SIGTERM)
    shutil.copyfile(args[3], home / "uploads" / Path(args[4]).name)
elif action == "exec" and mode == "term_exec":
    os.kill(os.getppid(), signal.SIGTERM)
elif action == "download":
    if mode == "download_error":
        sys.exit(1)
    run_id = Path(args[3]).name.removesuffix("-result.tar.gz")
    with tarfile.open(args[4], "w:gz") as archive:
        if mode != "missing_status":
            code = 1 if mode == "train_error" else 0
            status = json.dumps({"run_id": run_id, "returncode": code}).encode()
            member = tarfile.TarInfo("result/status.json")
            member.size = len(status)
            archive.addfile(member, io.BytesIO(status))
        if mode in ("success", "stop_error"):
            checkpoint = b"checkpoint"
            member = tarfile.TarInfo("result/log/bc/flow/run/checkpoint.pt")
            member.size = len(checkpoint)
            archive.addfile(member, io.BytesIO(checkpoint))
elif action == "stop" and mode == "stop_error":
    sys.exit(1)
"""


@pytest.mark.parametrize(
    ("mode", "expected_actions", "expected_status"),
    [
        ("success", ["new", "upload", "upload", "upload", "exec", "download", "stop"], 0),
        ("train_error", ["new", "upload", "upload", "exec", "download", "stop"], 1),
        ("missing_status", ["new", "upload", "upload", "exec", "download"], 1),
        ("download_error", ["new", "upload", "upload", "exec", "download"], 1),
        ("upload_error", ["new", "upload", "stop"], 1),
        ("stop_error", ["new", "upload", "upload", "exec", "download", "stop"], 1),
        ("term_upload", ["new", "upload", "stop"], 143),
        ("term_exec", ["new", "upload", "upload", "exec"], 143),
    ],
)
def test_shell_launcher_offline(tmp_path, mode, expected_actions, expected_status):
    source = Path(__file__).resolve().parents[1] / "scripts"
    root = tmp_path / "project with spaces"
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    for name in ("colab_flow_train.sh", "colab_flow_worker.sh", "colab_flow_worker.py"):
        shutil.copyfile(source / name, scripts / name)
    for name in ("pyproject.toml", "uv.lock", "README.md", ".python-version"):
        (root / name).write_text(name)
    (root / "src").mkdir()
    (root / "src" / "current.py").write_text("local edits")
    (root / "src" / "asset.stl").write_bytes(b"asset")
    (root / "src" / ".env").write_text("PRIVATE")
    (root / "src" / "token.json").write_text("PRIVATE")
    (root / "src" / "api_token.json").write_text("PRIVATE")
    dataset = root / "data with spaces"
    dataset.mkdir()
    episode = os.urandom(9 * 1024 * 1024) if mode == "success" else b"episode"
    (dataset / "episode.npz").write_bytes(episode)
    (dataset / "private.txt").write_text("PRIVATE")
    fake_home = tmp_path / "fake-colab"
    (fake_home / "uploads").mkdir(parents=True)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "colab"
    fake.write_text(FAKE_COLAB)
    fake.chmod(0o755)
    env = os.environ | {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "FAKE_COLAB_HOME": str(fake_home),
        "FAKE_COLAB_MODE": mode,
        "WANDB_MODE": "disabled",
    }
    env.pop("WANDB_API_KEY", None)
    process = subprocess.run(
        ["bash", "-x", str(scripts / "colab_flow_train.sh"), "--dataset", str(dataset)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert process.returncode == expected_status, process.stderr
    calls = [json.loads(line) for line in (fake_home / "calls.jsonl").read_text().splitlines()]
    assert [call[0] for call in calls] == expected_actions
    if mode == "success":
        chunks = [
            (fake_home / "uploads" / Path(call[4]).name).read_bytes()
            for call in calls
            if call[0] == "upload"
            and call[4].endswith(".part-0000")
            or call[0] == "upload"
            and call[4].endswith(".part-0001")
        ]
        assert len(chunks) == 2
        archive = tmp_path / "joined.tar.gz"
        archive.write_bytes(b"".join(chunks))
        with tarfile.open(archive, "r:gz") as bundle:
            names = bundle.getnames()
            assert "src/current.py" in names
            assert "src/asset.stl" in names
            assert "src/.env" not in names
            assert "src/token.json" not in names
            assert "src/api_token.json" not in names
            assert "data/forte_heuristic_final_100hz/episode.npz" in names
            assert "data/forte_heuristic_final_100hz/private.txt" not in names
        result_dir = next((root / "log" / "colab").iterdir())
        assert list((result_dir / "result" / "log").rglob("checkpoint.pt"))
    if mode == "stop_error":
        result_dir = next((root / "log" / "colab").iterdir())
        assert list((result_dir / "result" / "log").rglob("checkpoint.pt"))
    if mode == "term_exec":
        assert "remains active" in process.stderr


@pytest.mark.parametrize(
    ("mode", "actions", "code"),
    [
        ("online_success", ["new", "upload", "upload", "upload", "exec", "download", "stop"], 0),
        (
            "online_download_error",
            ["new", "upload", "upload", "upload", "exec", "download", "rm"],
            1,
        ),
        ("online_key_upload_error", ["new", "upload", "upload", "upload", "stop"], 1),
        ("online_term_exec", ["new", "upload", "upload", "upload", "exec", "rm"], 143),
        ("online_missing_key", [], 2),
    ],
)
def test_online_key_transfer_is_separate_and_private(tmp_path, mode, actions, code):
    source = Path(__file__).resolve().parents[1] / "scripts"
    root = tmp_path / "project"
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    for name in ("colab_flow_train.sh", "colab_flow_worker.sh", "colab_flow_worker.py"):
        shutil.copyfile(source / name, scripts / name)
    for name in ("pyproject.toml", "uv.lock", "README.md", ".python-version"):
        (root / name).write_text(name)
    (root / "src").mkdir()
    (root / "src" / "current.py").write_text("current local code")
    dataset = root / "data"
    dataset.mkdir()
    (dataset / "episode.npz").write_bytes(b"episode")
    fake_home = tmp_path / "fake-colab"
    (fake_home / "uploads").mkdir(parents=True)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "colab"
    fake.write_text(FAKE_COLAB)
    fake.chmod(0o755)
    dummy_key = "test-key-do-not-use-123"
    env = os.environ | {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "FAKE_COLAB_HOME": str(fake_home),
        "FAKE_COLAB_MODE": mode,
        "WANDB_PROJECT": "test-project",
        "WANDB_ENTITY": "test-entity",
        "WANDB_RUN_GROUP": "test-group",
    }
    env.pop("WANDB_API_KEY", None)
    env.pop("WANDB_MODE", None)
    if mode != "online_missing_key":
        env["WANDB_API_KEY"] = dummy_key
    process = subprocess.run(
        ["bash", "-x", str(scripts / "colab_flow_train.sh"), "--dataset", str(dataset)],
        env=env,
        text=True,
        capture_output=True,
    )
    assert process.returncode == code, process.stderr
    calls_file = fake_home / "calls.jsonl"
    calls = (
        [json.loads(line) for line in calls_file.read_text().splitlines()]
        if calls_file.exists()
        else []
    )
    assert [call[0] for call in calls] == actions
    assert dummy_key not in process.stdout + process.stderr
    if calls:
        assert dummy_key not in calls_file.read_text()
    if not calls:
        return
    result_dir = next((root / "log" / "colab").iterdir())
    assert dummy_key not in (result_dir / "session.json").read_text()
    key_calls = [call for call in calls if call[0] == "upload" and call[4].endswith("-wandb-key")]
    assert len(key_calls) == 1
    assert not Path(key_calls[0][3]).exists()
    assert (fake_home / "uploads" / Path(key_calls[0][4]).name).read_text() == dummy_key
    parts = sorted(
        (call[4], fake_home / "uploads" / Path(call[4]).name)
        for call in calls
        if call[0] == "upload" and ".part-" in call[4]
    )
    with tarfile.open(
        fileobj=io.BytesIO(b"".join(path.read_bytes() for _, path in parts)), mode="r:gz"
    ) as bundle:
        for member in bundle.getmembers():
            assert dummy_key not in member.name
            if member.isfile():
                assert dummy_key.encode() not in bundle.extractfile(member).read()
    if mode != "online_key_upload_error":
        exec_args = next(call for call in calls if call[0] == "exec")
        assert "MLAB_WANDB_MODE=online" in exec_args
        assert "WANDB_PROJECT=test-project" in exec_args
        assert "WANDB_ENTITY=test-entity" in exec_args
        assert "WANDB_RUN_GROUP=test-group" in exec_args
    if mode == "online_success":
        assert dummy_key.encode() not in (result_dir / "result" / "status.json").read_bytes()


@pytest.mark.parametrize("mode", ["success", "train_error", "unsafe_archive"])
@pytest.mark.parametrize("wandb_mode", ["disabled", "online"])
def test_remote_shell_worker_packages_checked_status(tmp_path, mode, wandb_mode):
    source = Path(__file__).resolve().parents[1] / "scripts/colab_flow_worker.sh"
    run_id = "flow-20260101-000000-1234abcd"
    content = tmp_path / "content"
    content.mkdir()
    worker = tmp_path / "worker.sh"
    worker.write_text(
        source.read_text().replace(
            'base="/content/$MLAB_RUN_ID"', 'base="$MLAB_TEST_CONTENT/$MLAB_RUN_ID"'
        )
    )
    bundle = content / f"{run_id}-input.tar.gz"
    with tarfile.open(bundle, "w:gz") as archive:
        data = b"[project]\nname='example'\n"
        member = tarfile.TarInfo("../escape" if mode == "unsafe_archive" else "pyproject.toml")
        member.size = len(data)
        archive.addfile(member, io.BytesIO(data))
    payload = bundle.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    midpoint = len(payload) // 2
    for index, part in enumerate((payload[:midpoint], payload[midpoint:])):
        (content / f"{run_id}-input.tar.gz.part-{index:04d}").write_bytes(part)
    bundle.unlink()
    dummy_key = "test-key-do-not-use-123"
    key_file = content / f"{run_id}-wandb-key"
    if wandb_mode == "online":
        key_file.write_text(dummy_key)
        key_file.chmod(0o644)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python = bin_dir / "mock-python"
    python.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "$1" == "-m" && "$2" == "pip" ]]; then exit 0; fi\n'
        f'exec {shlex.quote(sys.executable)} "$@"\n'
    )
    python.chmod(0o755)
    uv = bin_dir / "uv"
    uv.write_text(
        "#!" + sys.executable + "\n"
        "import os, pathlib, sys\n"
        "args = sys.argv[1:]\n"
        "if 'mujoco_lab.learning.train' in args:\n"
        "    assert '--policy-type' in args and 'flow' in args\n"
        "    assert os.environ['WANDB_MODE'] == os.environ['MLAB_WANDB_MODE']\n"
        "    keyfile = pathlib.Path(os.environ['MLAB_TEST_CONTENT'])\n"
        "    keyfile /= os.environ['MLAB_RUN_ID'] + '-wandb-key'\n"
        "    assert not keyfile.exists()\n"
        "    assert args[args.index('--eval-interval') + 1] == '0'\n"
        "    if os.environ['WANDB_MODE'] == 'online':\n"
        "        assert os.environ['WANDB_API_KEY'] == 'test-key-do-not-use-123'\n"
        "    else:\n"
        "        assert 'WANDB_API_KEY' not in os.environ\n"
        "    out = pathlib.Path(args[args.index('--output-dir') + 1])\n"
        "    periodic = out / 'bc/flow/run/checkpoint_step_00000030.pt'\n"
        "    periodic.parent.mkdir(parents=True)\n"
        "    periodic.write_bytes(b'periodic')\n"
        "    if os.environ['FAKE_TRAIN_MODE'] == 'train_error': sys.exit(7)\n"
        "    checkpoint = out / 'bc/flow/run/checkpoint.pt'\n"
        "    checkpoint.write_bytes(b'checkpoint')\n"
    )
    uv.chmod(0o755)
    env = os.environ | {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "MLAB_RUN_ID": run_id,
        "MLAB_CHUNKS": "2",
        "MLAB_SHA256": digest,
        "MLAB_PYTHON": str(python),
        "MLAB_TEST_CONTENT": str(content),
        "FAKE_TRAIN_MODE": mode,
        "MLAB_WANDB_MODE": wandb_mode,
    }
    env.pop("WANDB_API_KEY", None)
    process = subprocess.run(["bash", "-x", str(worker)], env=env, capture_output=True, text=True)
    expected = 0 if mode == "success" else 1 if mode == "unsafe_archive" else 7
    assert process.returncode == expected, process.stderr
    assert dummy_key not in process.stdout + process.stderr
    assert not key_file.exists()
    result_tar = content / f"{run_id}-result.tar.gz"
    assert dummy_key.encode() not in result_tar.read_bytes()
    with tarfile.open(result_tar, "r:gz") as archive:
        status = json.load(archive.extractfile("result/status.json"))
        assert status == {"run_id": run_id, "returncode": process.returncode}
        assert "result/console.log" in archive.getnames()
        assert ("result/log/bc/flow/run/checkpoint.pt" in archive.getnames()) == (mode == "success")
        periodic_name = "result/log/bc/flow/run/checkpoint_step_00000030.pt"
        assert (periodic_name in archive.getnames()) == (mode in ("success", "train_error"))
        for member in archive.getmembers():
            if member.isfile():
                assert dummy_key.encode() not in archive.extractfile(member).read()
    assert not (tmp_path / "escape").exists()
    if mode == "success":
        assert (content / run_id / "result/log/bc/flow/run/checkpoint_step_00000030.pt").exists()
