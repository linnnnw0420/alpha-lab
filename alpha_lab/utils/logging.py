from __future__ import annotations

import logging
import sys
from typing import Literal

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

_SETUP_DONE = False


def setup_logging(
    log_level: LogLevel = "INFO",
    log_format: str | None = None,
) -> None:
    """
    Configure global logging (call once at app startup).

    Args:
        log_level: minimum severity level (DEBUG/INFO/WARNING/ERROR/CRITICAL)
        log_format: optional custom format string (uses default if None)

    Notes:
        - Only sets up once; subsequent calls are no-op to avoid duplicate handlers.
        - Logs to stdout (not stderr) for better compatibility with notebooks.
    """
    global _SETUP_DONE
    if _SETUP_DONE:
        return
    
    if log_format is None:
        log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True, #override any existing config
    )

    _SETUP_DONE = True

def get_logger(name: str | None = None) -> logging.Logger:
    """
    Get a logger for the given module/name.

    Args:
        name: logger name (typically __name__), or None for root logger

    Returns:
        A Logger instance
    """
    if not _SETUP_DONE:
        setup_logging() # auto-setup with defaults
    
    return logging.getLogger(name)

__all__ = ["setup_logging", "get_logger"]