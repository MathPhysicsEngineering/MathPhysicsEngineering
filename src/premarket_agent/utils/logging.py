"""Logging utilities."""

from __future__ import annotations

import logging
import logging.config
from typing import Any, Dict

from ..config.settings import get_settings


def configure_logging() -> None:
    """Configure application-wide logging."""
    settings = get_settings()
    log_config = _build_logging_config(json_logs=settings.log_json, level=settings.log_level)
    logging.config.dictConfig(log_config)


def _build_logging_config(json_logs: bool, level: str) -> Dict[str, Any]:
    formatter = "json" if json_logs else "default"
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            },
            "json": {
                "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                "format": "%(asctime)s %(levelname)s %(name)s %(message)s",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": level,
                "formatter": formatter,
                "stream": "ext://sys.stdout",
            }
        },
        "root": {
            "handlers": ["console"],
            "level": level,
        },
    }


