"""Small standard-library logger for console and file output."""

import logging
import sys
from pathlib import Path
from typing import NoReturn


class _ConsoleHandler(logging.StreamHandler):
    """Write each record to the current standard error stream."""

    def emit(self, record: logging.LogRecord) -> None:
        self.stream = sys.stderr
        super().emit(record)


class Logger:
    """Write timestamped messages without changing global logging configuration."""

    def __init__(self, log_file: str | Path | None = None, console: bool = True) -> None:
        self._logger = logging.Logger(f"{__name__}.{id(self)}", level=logging.DEBUG)
        self._logger.propagate = False
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        if console:
            console_handler = _ConsoleHandler()
            console_handler.setFormatter(formatter)
            self._logger.addHandler(console_handler)
        if log_file is not None:
            file_handler = logging.FileHandler(Path(log_file), encoding="utf-8")
            file_handler.setFormatter(formatter)
            self._logger.addHandler(file_handler)

    def debug(self, message: str) -> None:
        self._logger.debug(message)

    def info(self, message: str) -> None:
        self._logger.info(message)

    def warn(self, message: str) -> None:
        self._logger.warning(message)

    def error(self, message: str, exit_code: int = 1) -> NoReturn:
        """Log an error and terminate with ``exit_code``."""
        self._logger.error(message)
        raise SystemExit(exit_code)
