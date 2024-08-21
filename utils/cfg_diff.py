import difflib
import json
import logging


def get_config(default: str, experiment: str) -> tuple[dict, dict, str, str]:
    with open(default) as f:
        default_cfg = json.load(f)
    with open(experiment) as f:
        exp_cfg = json.load(f)

    default_cfg_str = json.dumps(default_cfg, indent=4)
    exp_cfg_str = json.dumps(exp_cfg, indent=4)

    return default_cfg, exp_cfg, default_cfg_str, exp_cfg_str


def get_diff(default: str, exp: str, logger: logging.Logger) -> logging.Logger:
    differ = difflib.ndiff(
        default.splitlines(keepends=True), exp.splitlines(keepends=True)
    )

    diff_parts = "\n"
    for line in differ:
        if line.startswith("+") or line.startswith("-"):
            diff_parts += "\n" + line.strip()
    diff_parts += "\n"

    logger.info(f"diff: {diff_parts}")

    return logger
