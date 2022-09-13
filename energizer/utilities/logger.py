import logging

import colorlog

"""
Define the logger
"""
logger = logging.getLogger("energizer_logger")
formatter = colorlog.ColoredFormatter(
    "%(log_color)s[%(asctime)s] energizer/%(levelname)s%(reset)s ~ %(bold_yellow)s%(module)s:%(lineno)s%(reset)s$ %(message_log_color)s%(message)s",  # noqa: E501
    datefmt="%Y-%m-%d %H:%M:%S",
    # reset=True,
    log_colors={
        "DEBUG": "bold_cyan",
        "INFO": "bold_green",
        "WARNING": "bold_yellow",
        "ERROR": "bold_red",
        "CRITICAL": "bold_red,bg_white",
    },
    secondary_log_colors={
        "message": {
            "DEBUG": "white",
            "INFO": "white",
            "WARNING": "white",
            "ERROR": "white",
            "CRITICAL": "bg_red,white",
        }
    },
    style="%",
)

handler = colorlog.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


"""
Methods to interactively modify the logging level
"""
level_map = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


def set_logging_level(level: str, also_pl: bool = False):
    """Changes the internal logging level."""
    assert level in level_map, ValueError(f"`level` must be one of {list(level_map.keys())}")

    logger = logging.getLogger("energizer_logger")
    logger.setLevel(level_map[level])

    if also_pl:
        logger = logging.getLogger("pytorch_lightning")
        logger.setLevel(level_map[level])
