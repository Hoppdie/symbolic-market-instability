"""Tests for the logging utility (src/utils/logger.py)."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.logger import setup_logger


def test_setup_logger_returns_configured_logger(tmp_path):
    logger = setup_logger(name="test_returns", log_dir=str(tmp_path))
    assert isinstance(logger, logging.Logger)
    assert logger.level == logging.INFO
    assert logger.handlers  # console + file handlers attached


def test_setup_logger_creates_dated_log_file(tmp_path):
    logger = setup_logger(name="test_file", log_dir=str(tmp_path))
    logger.info("hello world")
    log_files = list(tmp_path.glob("system_*.log"))
    assert len(log_files) == 1


def test_setup_logger_does_not_duplicate_handlers(tmp_path):
    first = setup_logger(name="test_dupe", log_dir=str(tmp_path))
    handler_count = len(first.handlers)
    second = setup_logger(name="test_dupe", log_dir=str(tmp_path))
    assert first is second
    assert len(second.handlers) == handler_count  # handlers not re-added


def test_setup_logger_respects_custom_level(tmp_path):
    logger = setup_logger(name="test_level", log_dir=str(tmp_path),
                          level=logging.DEBUG)
    assert logger.level == logging.DEBUG
