"""Structured JSON logging setup for ahf.

Usage:
    from ahf.core.logging import setup_logging, get_logger
    setup_logging(level="INFO")
    log = get_logger(__name__)
    log.info("message", extra={"event": "trade_step", "action": 0.7})
"""
from __future__ import annotations

import json
import logging
import sys
from typing import Any


class JsonFormatter(logging.Formatter):
    """Emit each log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        data: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Merge any structured fields passed via extra={}
        for key, val in record.__dict__.items():
            if key not in logging.LogRecord.__dict__ and not key.startswith("_"):
                data[key] = val
        if record.exc_info:
            data["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(data, default=str)


def setup_logging(level: str = "INFO", *, json_output: bool = True) -> None:
    """Configure the root logger.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR).
        json_output: If True, emit JSON lines. If False, use human-readable format.
    """
    root = logging.getLogger()
    root.setLevel(level.upper())

    if root.handlers:
        root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    if json_output:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-8s %(name)s — %(message)s")
        )
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger.  Call setup_logging() once at startup first."""
    return logging.getLogger(name)
