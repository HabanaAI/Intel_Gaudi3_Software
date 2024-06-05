import logging
import os
import time
import shlex
import subprocess
from typing import Tuple, List


def config_logger(logger, log_level=logging.DEBUG, output_folder="."):
    """test logger setup"""
    foramt = "%(asctime)s (%(threadName)s) [%(levelname)s]: %(message)s"
    os.makedirs(output_folder, exist_ok=True)
    logging.basicConfig(
        filename=output_folder
        + "/synrec_tests"
        + time.strftime("%H%M%S_%d%m%y")
        + ".log",
        format=foramt,
        level=log_level,
    )
    logger.setLevel(log_level)
    ch = logging.StreamHandler()
    formatter = logging.Formatter(foramt)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

def exe(cmd: str, log=None) -> Tuple[int, List[str]]:
    """run subprocess"""
    process = subprocess.Popen(
        args=cmd,
        env=os.environ,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    process_prints = []
    while True:
        line = process.stdout.readline()
        if line:
            process_prints.append(line.decode().replace("\n", ""))
            if log:
                log.debug(line)
        else:
            break
    process.wait()
    return process.returncode, process_prints


def convert_command_line_to_dict(cmd: str):
    """
    Convert str with command line args to dictionary {arg: arg_value}
    for args with no velus, dict value is an empty string
    """
    # test_str = f"--path {RECORDING_OUTPUT} --test -s"
    tokens: list = shlex.split(cmd)
    args: dict = {}
    for i, token in enumerate(tokens):
        key, val = '', ''
        if token.startswith('-'):
            key = token.lstrip('-')
            # Check if the next token is a value or another flag
            if i + 1 < len(tokens) and not tokens[i + 1].startswith('-'):
                val = tokens[i + 1]
                i += 1
            args[key] = val
    return args
