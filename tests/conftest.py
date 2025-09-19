"""Pytest configuration to isolate Inspect AI logs into a temp directory.

This prevents test runs from writing to ./logs or whatever INSPECT_LOG_DIR
might be in the developer environment. We set INSPECT_LOG_DIR to a temporary
session directory and default the log level to 'warning' to reduce noise.

Refs:
- Inspect AI logs: https://inspect.aisi.org.uk/eval-logs.html
- Options/env vars: https://inspect.aisi.org.uk/options.html
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Generator

import pytest


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def pytest_configure(config: Any) -> None:  # runs before tests are collected
    # Respect an explicit override if the developer set one, otherwise isolate.
    if "INSPECT_LOG_DIR" not in os.environ:
        tmp_dir = Path(tempfile.mkdtemp(prefix="inspect_logs_"))
        _ensure_dir(tmp_dir)
        os.environ["INSPECT_LOG_DIR"] = str(tmp_dir)
    # Reduce log noise during tests unless explicitly overridden
    os.environ.setdefault("INSPECT_LOG_LEVEL", "warning")


@pytest.fixture(scope="session", autouse=True)
def _report_log_dir() -> Generator[None, None, None]:
    # Optional: print the temporary log dir at start for debugging.
    # Pytest may capture this; it's harmless and helps diagnose if needed.
    log_dir = os.environ.get("INSPECT_LOG_DIR")
    if log_dir:
        print(f"[tests] INSPECT_LOG_DIR={log_dir}")
    yield
