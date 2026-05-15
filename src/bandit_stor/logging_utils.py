"""Logging setup for Bandit-STOR runs."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path


class UtcFormatter(logging.Formatter):
    """Logging formatter that emits ISO-like UTC timestamps."""

    converter = time.gmtime


def setup_run_logging(run_dir: str | Path, *, level: int = logging.INFO) -> Path:
    """Configure console and file logging for a pipeline run.

    Args:
        run_dir: Output directory for run artifacts.
        level: Root logger level.

    Returns:
        Path to the run log file.
    """
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    log_path = run_path / "pipeline.log"
    fmt = "%(asctime)sZ | %(levelname)s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%dT%H:%M:%S"
    formatter = UtcFormatter(fmt=fmt, datefmt=datefmt)

    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    root.addHandler(console)
    root.addHandler(file_handler)
    logging.captureWarnings(True)
    logging.getLogger(__name__).info("Logging initialized at %s", log_path)
    return log_path
