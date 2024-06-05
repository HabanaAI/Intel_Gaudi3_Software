import logging
import os
import time

log_levels = {
    "d": logging.DEBUG,
    "i": logging.INFO,
    "w": logging.WARNING,
    "e": logging.ERROR,
    "c": logging.CRITICAL,
}

LOG = logging.getLogger("mpm")


def set_env():
    if os.environ.get("ENABLE_CONSOLE") is None:
        os.environ["ENABLE_CONSOLE"] = "true"
    if os.environ.get("LOG_LEVEL_ALL") is None:
        os.environ["LOG_LEVEL_ALL"] = "4"


def prepare_logger_format():
    # Time printing is added to format according to the value of PRINT_TIME env var.
    # This behaviour is aligned to hl_logger's behaviour.
    print_time = os.environ.get("PRINT_TIME", "True")
    time_format = "" if print_time in ["0", "False", "false"] else "%(asctime)s "
    return f"{time_format}(%(threadName)s) [%(levelname)s]: %(message)s"


def translate_log_level(log_level):
    return log_levels[log_level]


def config_logger(log_level, output_folder="."):
    if LOG.handlers:
        return
    set_env()
    LOG_LEVEL = translate_log_level(log_level)
    foramt = prepare_logger_format()
    os.makedirs(output_folder, exist_ok=True)
    logging.basicConfig(
        filename=output_folder + "/mpm_" + time.strftime("%Y%m%d_%H%M%S") + ".log",
        format=foramt,
        level=LOG_LEVEL,
    )
    LOG.setLevel(LOG_LEVEL)
    ch = logging.StreamHandler()
    formatter = logging.Formatter(foramt)
    ch.setFormatter(formatter)
    LOG.addHandler(ch)
