"""Small standard-library logger for console and file output."""

import logging
import sys
from pathlib import Path
from typing import NoReturn


class Logger:
    """Write timestamped messages without changing global logging configuration."""

    def __init__(self, log_file: str | Path | None = None) -> None:
        self._logger = logging.Logger(f"{__name__}.{id(self)}", level=logging.DEBUG)
        self._logger.propagate = False
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self._console_handler = logging.StreamHandler()
        self._console_handler.setFormatter(formatter)
        self._logger.addHandler(self._console_handler)
        if log_file is not None:
            file_handler = logging.FileHandler(Path(log_file), encoding="utf-8")
            file_handler.setFormatter(formatter)
            self._logger.addHandler(file_handler)

    def debug(self, message: str) -> None:
        self._log(logging.DEBUG, message)

    def info(self, message: str) -> None:
        self._log(logging.INFO, message)

    def warn(self, message: str) -> None:
        self._log(logging.WARNING, message)

    def error(self, message: str, exit_code: int = 1) -> NoReturn:
        """Log an error and terminate with ``exit_code``."""
        self._log(logging.ERROR, message)
        raise SystemExit(exit_code)

    def _log(self, level: int, message: str) -> None:
        self._console_handler.acquire()
        try:
            self._console_handler.stream = sys.stderr
            self._logger.log(level, message)
        finally:
            self._console_handler.release()
