import logging

import colorlog

logger = logging.getLogger("energizer_logger")

formatter = colorlog.ColoredFormatter(
    "%(log_color)s[%(asctime)s] energizer/%(levelname)s%(reset)s ~ %(bold_yellow)s%(module)s:%(lineno)s%(reset)s$ %(message_log_color)s%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    # reset=True,
    log_colors={
        "DEBUG": "bold_white",
        "DETAIL": "bold_cyan",
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
