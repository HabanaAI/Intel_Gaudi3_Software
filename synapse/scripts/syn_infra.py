import errno
import glob
import logging
import os
import subprocess
from typing import List
import xml.etree.ElementTree as ET
from xml.dom import minidom
import signal

SOFTWARE_LFS_DATA = os.environ.get("SOFTWARE_LFS_DATA")
if SOFTWARE_LFS_DATA is None:
    raise RuntimeError("SOFTWARE_LFS_DATA EV is not set")
CHIP_TYPES = {"greco": 1, "goya2": 1, "gaudi": 2, "gaudiM": 3, "gaudi2": 4, "gaudi3": 5}


def create_folder(folder):
    try:
        os.makedirs(folder)
    except OSError as exc:
        if not (exc.errno == errno.EEXIST and os.path.isdir(folder)):
            raise RuntimeError("failed to create results folder")


def prepare_logger_format():
    # Time printing is added to format according to the value of PRINT_TIME env var.
    # This behaviour is aligned to hl_logger's behaviour.
    print_time = os.environ.get("PRINT_TIME", "True")
    time_format = "" if print_time in ["0", "False", "false"] else "%(asctime)s "
    return f"{time_format}%(message)s"


def config_logger(name, log_level, file_path=None):
    format = prepare_logger_format()
    logging.basicConfig(
        filename=file_path,
        format=format,
        level=log_level,
    )
    return logging.getLogger(name)


def get_files(folder, postfix, prefix=""):
    if not folder:
        return
    yield from (
        f
        for f in glob.iglob(f"{folder}/**/{prefix}*{postfix}", recursive=True)
        if os.path.isfile(f)
    )


def get_latest(files):
    times = {}
    for f in files:
        times[f] = os.path.getmtime(f)
    return max(times, key=times.get)


def run_external_in_bash(args: List[str], venv=None, log=None, log_cmd=False):
    if venv:
        args = ["source", venv, "&&", "cd", os.getcwd(), "&&"] + args
    args_str = " ".join(args)
    cmd = f'bash -ic "{args_str}"'
    if log is not None:
        if log_cmd:
            log.info(f'run_external: {" ".join(args)}')
    subprocess.run(args=cmd, shell=True, env=os.environ)


def run_external(args: List[str], env_ext=None, log=None, log_cmd=False):
    env = os.environ.copy()
    if env_ext:
        env.update(env_ext)
    p = subprocess.Popen(
        "exec " + " ".join(args),
        env=env,
        shell=True,
        stdout=subprocess.PIPE if log else None,
        stderr=subprocess.STDOUT if log else None,
    )
    try:
        if log is not None:
            if log_cmd:
                log.info(f'run_external: {" ".join(args)}')
            for line in iter(p.stdout):
                try:
                    log.info(line.decode().replace("\n", ""))
                except Exception:
                    log.info(line)
        p.wait()
    except:
        os.kill(p.pid, signal.SIGKILL)
        return 1
    return p.poll()


def git_clone(repo, folder, env=None, log=None):
    run_external(["git", "clone", repo, folder], env, log)


def git_pull(folder, env=None, log=None):
    run_external(["git", "-C", folder, "pull"], env, log)
    run_external(["git", "-C", folder, "lfs", "pull"], env, log)


def git_checkout(folder, commit, env=None, log=None):
    run_external(["git", "-C", folder, "checkout", commit], env, log)


def git_hash(folder) -> str:
    return (
        subprocess.check_output(["git", "-C", folder, "rev-parse", "HEAD"])
        .decode("ascii")
        .strip()
    )


def generate_xml(name):
    ret = ET.Element("testsuites")
    testsuite = ET.SubElement(ret, "testsuite")
    testsuite.set("name", name)

    return ret


def add_testcase_to_xml(testxml, feilds_dict):
    testcase = ET.SubElement(testxml.getchildren()[0], "testcase")
    for k, v in feilds_dict.items():
        if v is not None:
            testcase.set(k, str(v))


def save_xml_file(testxml, file_name):
    with open(file_name, "w") as o_file:
        o_file.write(
            "\n".join(
                line
                for line in minidom.parseString(ET.tostring(testxml))
                .toprettyxml(indent="   ")
                .split("\n")
                if line.strip()
            )
        )
