"""Dark Pool Detection System — Utility functions."""

import logging
import sys
from pathlib import Path

import yaml


def load_config(config_path: str | None = None) -> dict:
    default_path = Path(__file__).parent.parent / "config" / "config.default.yaml"
    with open(default_path) as f:
        cfg = yaml.safe_load(f)
    if config_path is not None:
        with open(config_path) as f:
            override = yaml.safe_load(f)
        _deep_merge(cfg, override)
    return cfg


def _deep_merge(base: dict, override: dict) -> None:
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def get_logger(name: str = "darkpool", level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
    return logger


def ensure_dirs(config: dict) -> None:
    base = Path(config["system"]["data_dir"])
    out = Path(config["system"]["output_dir"])
    dirs = [
        base / "raw" / "finra",
        base / "raw" / "yfinance",
        base / "parquet",
        base / "simulated",
        out,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
