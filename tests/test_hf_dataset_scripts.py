import os
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_script(tmp_path, script_name, *args, with_hf=True, fail_create=False):
    project = tmp_path / "project"
    scripts = project / "scripts"
    scripts.mkdir(parents=True)
    (project / "data/forte_heuristic_final_100hz").mkdir(parents=True)
    (project / "data/other dataset").mkdir(parents=True)
    (project / "data/README.md").write_text("# Dataset card fixture\n")
    source_script = PROJECT_ROOT / "scripts" / script_name
    script = scripts / script_name
    script.write_bytes(source_script.read_bytes())
    script.chmod(0o755)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    capture = tmp_path / "hf-args"
    env = os.environ.copy()
    env["HF_CAPTURE"] = str(capture)
    if fail_create:
        env["HF_FAIL_CREATE"] = "1"
    env["PATH"] = str(bin_dir) + os.pathsep + env.get("PATH", "")
    if with_hf:
        fake_hf = bin_dir / "hf"
        fake_hf.write_text(
            "#!/usr/bin/env bash\n"
            'if [[ "${1:-}" == repos && "${2:-}" == create &&\n'
            '      "${HF_FAIL_CREATE:-}" == 1 ]]; then exit 17; fi\n'
            'printf \'%s\\n\' "$@" >> "$HF_CAPTURE"\n'
            "printf '%s\\n' '###' >> \"$HF_CAPTURE\"\n"
        )
        fake_hf.chmod(0o755)

    result = subprocess.run(
        [str(script), *args],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    calls = []
    if capture.exists():
        calls = [
            call.decode().splitlines() for call in capture.read_bytes().split(b"###\n") if call
        ]
    return result, calls, project


def test_upload_defaults_to_whole_data_tree_and_creates_private_repo(tmp_path):
    result, calls, project = run_script(tmp_path, "dataset_upload.sh", "user/robot-data")

    assert result.returncode == 0, result.stderr
    assert calls == [
        [
            "repos",
            "create",
            "user/robot-data",
            "--repo-type",
            "dataset",
            "--private",
            "--exist-ok",
        ],
        [
            "upload",
            "user/robot-data",
            str(project / "data"),
            ".",
            "--repo-type",
            "dataset",
            "--exclude",
            ".cache/huggingface/**",
        ],
    ]
    assert (project / "data/README.md").read_text() == "# Dataset card fixture\n"


def test_upload_selects_named_dataset_without_flattening(tmp_path):
    result, calls, project = run_script(
        tmp_path, "dataset_upload.sh", "user/robot-data", "other dataset"
    )

    assert result.returncode == 0, result.stderr
    assert calls[1][2:4] == [
        str(project / "data/other dataset"),
        "other dataset",
    ]
    assert calls[1][-2:] == ["--exclude", ".cache/huggingface/**"]
    assert calls[2] == [
        "upload",
        "user/robot-data",
        str(project / "data/README.md"),
        "README.md",
        "--repo-type",
        "dataset",
    ]


def test_upload_stops_if_private_repo_creation_fails(tmp_path):
    result, calls, _ = run_script(
        tmp_path, "dataset_upload.sh", "user/robot-data", fail_create=True
    )

    assert result.returncode == 17
    assert calls == []


def test_download_defaults_to_whole_repo_and_main_revision(tmp_path):
    result, calls, project = run_script(tmp_path, "dataset_download.sh", "user/robot-data")

    assert result.returncode == 0, result.stderr
    assert calls == [
        [
            "download",
            "user/robot-data",
            "--repo-type",
            "dataset",
            "--local-dir",
            str(project / "data"),
            "--revision",
            "main",
        ]
    ]


def test_download_selects_named_dataset_with_revision(tmp_path):
    result, calls, project = run_script(
        tmp_path,
        "dataset_download.sh",
        "user/robot-data",
        "other dataset",
        "v2",
    )

    assert result.returncode == 0, result.stderr
    assert calls == [
        [
            "download",
            "user/robot-data",
            "--repo-type",
            "dataset",
            "--local-dir",
            str(project / "data"),
            "--revision",
            "v2",
            "--include",
            "other dataset/**",
        ]
    ]


def test_dataset_name_cannot_escape_data_directory(tmp_path):
    result, calls, _ = run_script(tmp_path, "dataset_upload.sh", "user/robot-data", "../outside")

    assert result.returncode == 2
    assert "single directory name" in result.stderr
    assert calls == []


def test_upload_requires_repository_argument(tmp_path):
    result, calls, _ = run_script(tmp_path, "dataset_upload.sh")

    assert result.returncode == 2
    assert "Usage:" in result.stderr
    assert calls == []


def test_download_rejects_path_traversal(tmp_path):
    result, calls, _ = run_script(tmp_path, "dataset_download.sh", "user/robot-data", "../outside")

    assert result.returncode == 2
    assert "single directory name" in result.stderr
    assert calls == []


def test_download_empty_selector_allows_revision_without_filter(tmp_path):
    result, calls, project = run_script(
        tmp_path, "dataset_download.sh", "user/robot-data", "", "v2"
    )

    assert result.returncode == 0, result.stderr
    assert calls == [
        [
            "download",
            "user/robot-data",
            "--repo-type",
            "dataset",
            "--local-dir",
            str(project / "data"),
            "--revision",
            "v2",
        ]
    ]


def test_help_does_not_require_hf_cli(tmp_path):
    result, calls, _ = run_script(tmp_path, "dataset_upload.sh", "--help", with_hf=False)

    assert result.returncode == 0
    assert "whole data directory" in result.stdout
    assert calls == []
