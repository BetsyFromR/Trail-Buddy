from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs"
DEFAULT_LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
MAX_LOG_BYTES = 5 * 1024 * 1024
BACKUP_COUNT = 3
QUIET_LOGGERS = ("LiteLLM", "litellm", "httpx", "httpcore")


def _path_env(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    if not raw:
        return default
    path = Path(raw).expanduser()
    if not path.is_absolute():
        return PROJECT_ROOT / path
    return path


def _log_level() -> int:
    raw = os.getenv("TRAIL_BUDDY_LOG_LEVEL", DEFAULT_LOG_LEVEL).upper()
    return logging.getLevelNamesMapping().get(raw, logging.INFO)


def _remove_stream_handlers(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        if isinstance(handler, logging.StreamHandler) and not isinstance(
            handler, RotatingFileHandler
        ):
            logger.removeHandler(handler)
            handler.close()


def _quiet_third_party_loggers() -> None:
    for name in QUIET_LOGGERS:
        logger = logging.getLogger(name)
        logger.setLevel(logging.WARNING)
        _remove_stream_handlers(logger)


def configure_logging() -> None:
    """Configure app and error file logs once per process."""
    root = logging.getLogger()
    if getattr(root, "_trail_buddy_logging_configured", False):
        return

    log_dir = _path_env("TRAIL_BUDDY_LOG_DIR", DEFAULT_LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    app_file = RotatingFileHandler(
        log_dir / "trail_buddy.log",
        maxBytes=MAX_LOG_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    app_file.setLevel(_log_level())
    app_file.setFormatter(formatter)

    error_file = RotatingFileHandler(
        log_dir / "trail_buddy.error.log",
        maxBytes=MAX_LOG_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    error_file.setLevel(logging.ERROR)
    error_file.setFormatter(formatter)

    _remove_stream_handlers(root)
    _quiet_third_party_loggers()

    root.setLevel(_log_level())
    root.addHandler(app_file)
    root.addHandler(error_file)
    root._trail_buddy_logging_configured = True
