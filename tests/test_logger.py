import logging
import re

import pytest

from mujoco_lab.utils import Logger


def test_logger_writes_levels_to_console_and_file(tmp_path, capsys):
    log_file = tmp_path / "simulation.log"
    logger = Logger(log_file)

    logger.debug("controller ready")
    logger.info("simulation started")
    logger.warn("joint limit approaching")
    with pytest.raises(SystemExit) as exit_info:
        logger.error("simulation stopped")

    expected_messages = [
        "[DEBUG] controller ready",
        "[INFO] simulation started",
        "[WARNING] joint limit approaching",
        "[ERROR] simulation stopped",
    ]
    console_lines = capsys.readouterr().err.splitlines()
    file_lines = log_file.read_text(encoding="utf-8").splitlines()

    assert exit_info.value.code == 1
    assert console_lines == file_lines
    assert len(file_lines) == len(expected_messages)
    for line, message in zip(file_lines, expected_messages, strict=True):
        timestamp = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}"
        assert re.fullmatch(rf"{timestamp} {re.escape(message)}", line)


def test_logger_can_disable_console_and_multiple_instances_append_once(tmp_path, capsys):
    log_file = tmp_path / "simulation.log"

    Logger(log_file, console=False).info("first")
    Logger(log_file, console=False).info("second")

    captured = capsys.readouterr()
    assert captured.out == captured.err == ""
    lines = log_file.read_text(encoding="utf-8").splitlines()
    assert [line.rsplit(" ", 1)[-1] for line in lines] == ["first", "second"]


def test_logger_does_not_propagate_to_root_handlers(capsys):
    records = []

    class RecordingHandler(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = RecordingHandler()
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        Logger().debug("isolated")
    finally:
        root.removeHandler(handler)

    assert records == []
    assert capsys.readouterr().err.endswith("[DEBUG] isolated\n")
