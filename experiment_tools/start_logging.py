import logging
from logging import getLogger, FileHandler, Formatter


def get_logger(cfg: dict) -> logging.Logger:
    logger = getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = FileHandler(cfg["log"]["log_file"], mode="w")
    formatter = Formatter(cfg["log"]["log_formatter"])
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
