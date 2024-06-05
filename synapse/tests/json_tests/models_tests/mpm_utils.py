import glob
import json
import math
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

from mpm_logger import LOG
from mpm_model_list import ModelsList
from mpm_types import ERROR_CODES, GlobalConfig, ModelInfo, ModelStats, ProcessPriority

try:
    import psutil
except ModuleNotFoundError:
    LOG.info("installing psutil moduled")
    subprocess.check_call(
        ["python", "-m", "pip", "install", "psutil"], stdout=subprocess.DEVNULL
    )
finally:
    import psutil


def get_tools():
    return {
        "WHICH": "which",
        "GIT": "git",
        "BASH": "bash",
    }


def calculate_change(res, ref):
    return ((ref - res) / ref) * 100.0


class Executer:
    def __init__(
        self,
        tool,
        args,
        fatal=False,
        log=True,
        env_ext=None,
        ld_preload=None,
        process_priority=ProcessPriority.NORMAL,
    ):
        self.max_mem = 0
        self.process_prints = []
        self._begin_time = None
        self._end_time = None
        self.process: subprocess.Popen = None
        self.mem_tracker = ThreadPoolExecutor(1)
        self.mem_tracker.submit(self.check_mem)
        self._start_process(tool, args, env_ext, ld_preload, process_priority)
        self._wait(log)
        self._status_check(tool, args, fatal)

    def get_process_mem(self) -> int:
        rss = 0
        try:
            proc = psutil.Process(self.process.pid)
            subprocesses = list(proc.children(True)) + [proc]

            for subproc in subprocesses:
                rss += subproc.memory_info().rss
        except:
            pass
        return rss

    def check_mem(self):
        while self.process is None or self.process.poll() is None:
            mem = self.get_process_mem()
            if mem > self.max_mem:
                self.max_mem = mem
            time.sleep(0.1)

    def _start_process(self, tool, args, env_ext, ld_preload, process_priority):
        env = os.environ.copy()
        if ld_preload:
            env["LD_PRELOAD"] = ld_preload
        if env_ext:
            LOG.debug(f"external process env: {' '.join(f'{k}={v}' for k, v in env_ext.items())}")
            env.update(env_ext)
        priority = f"nice -n {process_priority}"
        cmd = f"{priority} {tool} {args}"
        LOG.debug(f"run external process: {cmd}")
        self._begin_time = time.time()
        self.process = subprocess.Popen(
            cmd,
            env=env,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

    def _wait(self, log):
        while True:
            line = self.process.stdout.readline()
            if not line:
                break
            line_str = line.decode().replace("\n", "")
            self.process_prints.append(line_str)
            if log:
                LOG.debug(line_str)
        self.process.wait()
        self._end_time = time.time()
        self.mem_tracker.shutdown()

    def _status_check(self, tool, args, fatal):
        if self.process.returncode != 0:
            msg = "failed to run: {} with args: {}".format(tool, args)
            if fatal:
                LOG.error(msg)
                raise RuntimeError(msg)
            else:
                LOG.warn(msg)

    def get_status(self) -> int:
        return self.process.returncode

    def get_prints(self) -> List[str]:
        return self.process_prints

    def get_max_mem(self) -> int:
        return self.max_mem

    def get_execution_time(self) -> Optional[int]:
        if self._end_time and self._begin_time:
            return math.floor((self._end_time - self._begin_time) * 1e3)
        return None


def git_clone(repo, local_path):
    LOG.debug(f"clone : {repo}")
    Executer(
        get_tools().get("GIT"),
        f"clone {repo} {local_path}",
        True,
        False,
    )


def git_pull(repo_path, allow_fail=False):
    LOG.debug(f"pull repo: {repo_path}")
    Executer(
        get_tools().get("GIT"),
        f"-C {repo_path} pull",
        not allow_fail,
        False,
    )
    Executer(
        get_tools().get("GIT"),
        f"-C {repo_path} lfs pull",
        not allow_fail,
        False,
    )


def checkout(commit, repo_path):
    LOG.debug(f"checkout: {commit} from repo: {repo_path}")
    Executer(
        get_tools().get("GIT"),
        f"-C {repo_path} checkout {commit}",
        True,
        False,
    )


def get_git_commit(path, allow_fail, commit="HEAD"):
    LOG.debug("get git commit")
    if not path or not commit:
        return None
    exe = Executer(
        get_tools().get("GIT"),
        f"-C {path} rev-parse {commit}",
        not allow_fail,
        False,
    )
    if exe.get_status() != 0:
        return None
    return exe.get_prints()[0]


def get_git_branch(global_config: GlobalConfig):
    LOG.debug("get git branch")
    ret = Executer(
        get_tools().get("GIT"),
        f"-C {global_config.CONSTS.NPU_STACK_PATH}/{global_config.repo} rev-parse --abbrev-ref HEAD",
        True,
        False,
    ).get_prints()
    return ret[0]


def build(build_tests, global_config: GlobalConfig):
    LOG.info(f"build {global_config.repo}")
    exe_exists = os.path.exists(global_config.models_tests_binary_path)
    skip_tests_flag = " -l" if exe_exists and not build_tests else ""
    env_file = "/tmp/mpm-env"
    env = [f'{k}="{v}"\n' for k, v in os.environ.items()]
    with open(env_file, "w") as f:
        f.writelines(env)
    sts = Executer(
        get_tools().get("BASH"),
        f'-ic "source {env_file} && build_{global_config.repo} -r{skip_tests_flag}"',
    ).get_status()
    os.remove(env_file)
    if sts != 0:
        raise RuntimeError(f"failed to build {global_config.repo}")


def get_files(folder, postfix):
    if not folder:
        return
    yield from (
        f
        for f in glob.iglob(f"{folder}/**/*{postfix}", recursive=True)
        if os.path.isfile(f)
    )


def remove_files_from_directory(directory_path):
    for fileName in os.listdir(directory_path):
        file_full_path = os.path.join(directory_path, fileName)
        if os.path.isfile(file_full_path):
            os.remove(file_full_path)


def get_models_stats(global_config: GlobalConfig) -> Dict[str, ModelStats]:
    ret = {}
    if os.path.exists(global_config.models_file):
        with open(global_config.models_file) as f:
            models= ModelsList.deserialize(json.load(f)).models
            for k, v in models.items():
                device = v.devices.get(global_config.chip_type)
                if device:
                    ret[k] = device.stats
    return ret


def get_models_infos(
    global_config: GlobalConfig, default_model_stats: ModelStats
) -> List[ModelInfo]:
    LOG.debug("get all models")
    ret: List[ModelInfo] = []
    models_data = {}
    models_stats: Dict[str, ModelStats] = get_models_stats(global_config)
    if global_config.accuracy:
        for tf in get_files(global_config.tensors_data_folder, ".db"):
            models_data[os.path.basename(tf).replace(".db", "")] = tf
    for tf in get_files(global_config.models_folder, ".json"):
        name = os.path.basename(tf).replace(".json", "")
        stats: ModelStats = models_stats.get(name, default_model_stats)
        ret.append(
            ModelInfo(
                name,
                tf,
                models_data.get(name, None),
                stats,
            )
        )
    models = global_config.models
    models_filter_file_path = os.path.join(
        global_config.work_folder, global_config.CONSTS.MODELS_FILTER_FILE_NAME
    )
    if os.path.exists(models_filter_file_path):
        with open(models_filter_file_path) as f:
            models = json.load(f)
    missing_json = [
        model_name
        for model_name in models
        if not any(model_name == model_info.name for model_info in ret)
    ]

    if missing_json:
        separator = "'\n'"
        LOG.warning(
            f"Json files for models are missing\n{separator.join(missing_json)}"
        )

    ret = [m for m in ret if m.name in models]
    for m in ret:
        if m.stats.compile is None:
            m.stats.compile = default_model_stats.compile
        if m.stats.run is None:
            m.stats.run = default_model_stats.run
        if "compile" in global_config.actions and (m.stats.compile.duration == 0 or m.stats.compile.max_memory_usage == 0):
            msg = f"missing compile model stats for model: {m.name}, this might impact the execution time of the test"
            if global_config.models_stats_required:
                LOG.error(f"{msg}, models_stats_required flag is enabled, exiting test")
                exit(ERROR_CODES.ERROR)
            else:
                LOG.warning(msg)
        if "run" in global_config.actions and (m.stats.run.duration == 0 or m.stats.run.max_memory_usage == 0):
            msg = f"missing run model stats for model: {m.name}, this might impact the execution time of the test"
            if global_config.models_stats_required:
                LOG.error(f"{msg}, models_stats_required flag is enabled, exiting test")
                exit(ERROR_CODES.ERROR)
            else:
                LOG.warning(msg)
    return ret
