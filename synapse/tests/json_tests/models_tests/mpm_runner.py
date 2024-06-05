

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Set

import mpm_utils
from mpm_logger import LOG
from mpm_types import (ERROR_CODES, ComObj, GlobalConfig, LaunchConfig, MemSts,
                       ProcessPriority, Stats)


def rename_logs(logs_folder, model_name):
    for _, fileName in enumerate(os.listdir(logs_folder)):
        if model_name not in fileName:
            os.rename(f"{logs_folder}/{fileName}", f"{logs_folder}/{model_name}_{fileName}")


def config_profiler(launch_config: LaunchConfig, name: str):
    env = dict()
    model_folder = os.path.join(
        launch_config.test.folder_path, launch_config.model.info.name
    )
    prof_time_measure = launch_config.global_config.time_measurement == "profiler"
    if launch_config.global_config.profiler:
        trace_dir = os.path.join(model_folder, "trace", name)
        os.makedirs(trace_dir, exist_ok=True)
        prof_file = f"{trace_dir}/prof_{name}_conf.json"
        if not os.path.exists(prof_file):
            mpm_utils.Executer(
                "hl-prof-config",
                f"--edit-existing off --config-filename {prof_file} --chip {launch_config.global_config.chip_type} --session-name {launch_config.model.info.name}_{name}_prof --output-dir {trace_dir} --invoc hltv --trace-analyzer off --hw-trace on --buffer-size 80 --merged hltv {'--instrumentation' if prof_time_measure else ''}",
                True,
                True,
            )
    if launch_config.global_config.profiler:
        env["HABANA_PROF_CONFIG"] = prof_file
        env["HABANA_PROFILE"] = str(1)
    elif prof_time_measure:
        env["HABANA_PROFILE"] = "profile_api_light_recipe_busy"
    return env


def compile_model(launch_config: LaunchConfig):
    LOG.info(
        f"compile test: {launch_config.test.name}, model: {launch_config.model.info.name}"
    )
    model_folder = os.path.join(
        launch_config.test.folder_path, launch_config.model.info.name
    )
    os.makedirs(model_folder, exist_ok=True)
    logs_folder = f"{model_folder}/logs-compile"
    if os.path.exists(logs_folder):
        mpm_utils.remove_files_from_directory(logs_folder)
    env = {
        "HABANA_LOGS": logs_folder,
        "FUSER_DEBUG_PATH": logs_folder,
    }
    env.update(config_profiler(launch_config, "host"))
    env.update(launch_config.test.env)
    exe = mpm_utils.Executer(
        launch_config.global_config.models_tests_binary_path,
        f"model --compile --com-file {launch_config.com_file_path}",
        False,
        True,
        env,
        launch_config.test.lib_path,
        ProcessPriority.BELOW_NORMAL,
    )
    with open(launch_config.com_file_path) as f:
        com = ComObj.deserialize(json.load(f))
    serialize_com = False
    debug_marker_file = os.path.join(logs_folder, launch_config.global_config.CONSTS.DEBUG_MARKER_FILE_NAME)
    if launch_config.global_config.skip_models:
        serialize_com = True
        if os.path.exists(debug_marker_file):
            com.config.marked_for_debug = True
        else:
            com.config.marked_for_debug = False
            launch_config.model.skip_run = True
    rename_logs(logs_folder, launch_config.model.info.name)
    if com.compile:
        serialize_com = True
        com.compile.stats = Stats()
        com.compile.stats.duration = exe.get_execution_time()
        com.compile.stats.max_memory_usage = exe.get_max_mem()
        execution_time_limit = launch_config.global_config.execution_time_limit * 1e3 if launch_config.global_config.execution_time_limit else 0
        if (execution_time_limit > 0 and execution_time_limit < com.compile.stats.duration):
            com.compile.error = f"compile execution time exceeds limit, limit {launch_config.global_config.execution_time_limit * 1e3} [ms], duration: {com.compile.stats.duration} [ms]"
    if serialize_com:
        with open(launch_config.com_file_path, "w") as f:
            json.dump(com.serialize(), f)
    return exe.get_status()


def run_model(launch_config: LaunchConfig):
    LOG.info(
        f"run test: {launch_config.test.name}, model: {launch_config.model.info.name}"
    )
    model_folder = os.path.join(
        launch_config.test.folder_path, launch_config.model.info.name
    )
    if not os.path.exists(model_folder):
        os.makedirs(model_folder, exist_ok=True)
    logs_folder = f"{model_folder}/logs-runtime"
    env = {
        "HABANA_LOGS": logs_folder,
    }
    env.update(config_profiler(launch_config, "hw"))
    env.update(launch_config.test.env)
    exe = mpm_utils.Executer(
        launch_config.global_config.models_tests_binary_path,
        f"model --run --com-file {launch_config.com_file_path}",
        False,
        True,
        env,
        launch_config.test.lib_path,
        ProcessPriority.BELOW_NORMAL,
    )
    rename_logs(logs_folder, launch_config.model.info.name)
    with open(launch_config.com_file_path) as f:
        com = ComObj.deserialize(json.load(f))
    if com.run:
        if com.run.error and "failed to capture device run time" in com.run.error: # w/a to time capture failure
            raise RuntimeError(com.run.error)
        com.run.stats = Stats()
        com.run.stats.duration = exe.get_execution_time()
        com.run.stats.max_memory_usage = exe.get_max_mem()
        execution_time_limit = launch_config.global_config.execution_time_limit * 1e3 if launch_config.global_config.execution_time_limit else 0
        if (execution_time_limit > 0 and execution_time_limit < com.run.stats.duration):
            com.run.error = f"run execution time exceeds limit, limit {launch_config.global_config.execution_time_limit * 1e3} [ms], duration: {com.run.stats.duration} [ms]"
        with open(launch_config.com_file_path, "w") as f:
            json.dump(com.serialize(), f)
    return exe.get_status()


def sort_by_compile_duration(config: LaunchConfig):
    return config.model.info.stats.compile.duration


def sort_by_run_duration(config: LaunchConfig):
    return config.model.info.stats.run.duration


class Dispatcher:
    def __init__(self, max_executers_count, global_config: GlobalConfig):
        LOG.debug(f"max number of compilation threads is set to: {max_executers_count}")
        self.OUT_OF_MEMORY_ERROR_CODE = 137

        self.results: List[ComObj] = []
        self.max_mem_limit = (
            global_config.max_mem_limit
            if global_config.max_mem_limit
            else int(MemSts().available * 0.8)
        )
        self.compile_threads = max_executers_count
        self.run_threads = global_config.cards
        self.retry_on_failure = global_config.retry_on_failure
        self.time_measurement = global_config.time_measurement
        self.models_filter_file_path = os.path.join(global_config.work_folder, global_config.CONSTS.MODELS_FILTER_FILE_NAME)
        self.compile_launch_configs: List[LaunchConfig] = []
        self.run_launch_configs: List[LaunchConfig] = []
        self.finished_launch_configs: List[LaunchConfig] = []
        self.submit_tpe = ThreadPoolExecutor(1)
        self.compile_tpe = ThreadPoolExecutor(self.compile_threads)
        self.run_tpe = ThreadPoolExecutor(self.run_threads)
        self.compile_fail_counter: Dict[LaunchConfig, int] = {}
        self.run_fail_counter: Dict[LaunchConfig, int] = {}
        self.active_mem = 0
        self.compile_activated: int = 0
        self.run_activated: int = 0
        self.marked_for_debug: Set[str] = set()

    def add_result(self, launch_config: LaunchConfig):
        with open(launch_config.com_file_path) as f:
            config_data = json.load(f)
        com = ComObj.deserialize(config_data)
        if com.config.marked_for_debug:
            self.marked_for_debug.add(launch_config.model.info.name)
        self.results.append(com)

    def should_submit(self, max_memory_usage) -> bool:
        if self.active_mem + max_memory_usage < self.max_mem_limit:
            return True
        if self.active_mem == 0 and max_memory_usage >= self.max_mem_limit:
            return True
        return False

    def scan_compile(self):
        sorted_compile_configs = sorted(
            self.compile_launch_configs, key=sort_by_compile_duration, reverse=True
        )
        for c in sorted_compile_configs:
            if self.compile_activated >= self.compile_threads:
                break
            if self.should_submit(c.model.info.stats.compile.max_memory_usage):
                self.active_mem += c.model.info.stats.compile.max_memory_usage
                self.compile_tpe.submit(self.try_compile, c)
                self.compile_launch_configs.remove(c)
                self.compile_activated += 1
        LOG.info(
            f"scan compile: launch configs size: {len(self.compile_launch_configs)}, activated: {self.compile_activated}"
        )

    def scan_run(self):
        sorted_run_configs = sorted(
            self.run_launch_configs, key=sort_by_run_duration, reverse=True
        )
        for c in sorted_run_configs:
            if self.run_activated > self.run_threads:
                break
            if self.should_submit(c.model.info.stats.run.max_memory_usage):
                self.active_mem += c.model.info.stats.run.max_memory_usage
                self.run_tpe.submit(self.try_run, c)
                self.run_launch_configs.remove(c)
                self.run_activated += 1
        LOG.info(
            f"scan run: launch configs size: {len(self.run_launch_configs)}, activated: {self.run_activated}"
        )

    def scan(self):
        self.scan_run()
        self.scan_compile()

    def submit_compile(self, config: LaunchConfig):
        self.compile_launch_configs.append(config)
        self.scan()

    def submit_run(self, config: LaunchConfig):
        self.run_launch_configs.append(config)
        self.scan()

    def remove_compile(self, config: LaunchConfig):
        self.active_mem -= config.model.info.stats.compile.max_memory_usage
        self.compile_activated -= 1
        self.scan()

    def remove_run(self, config: LaunchConfig):
        self.active_mem -= config.model.info.stats.run.max_memory_usage
        self.run_activated -= 1
        self.scan()

    def finish(self, config: LaunchConfig):
        self.add_result(config)
        self.finished_launch_configs.append(config)

    def compile(self, launch_config: LaunchConfig):
        if launch_config not in self.compile_fail_counter:
            self.compile_fail_counter[launch_config] = 0
        launch_config.status = compile_model(launch_config)
        if launch_config.status == self.OUT_OF_MEMORY_ERROR_CODE:
            raise RuntimeError("compilation failed with out of memory error")
        else:
            self.submit_tpe.submit(self.submit_run, launch_config)
            self.submit_tpe.submit(self.remove_compile, launch_config)

    def run(self, launch_config: LaunchConfig):
        if launch_config not in self.run_fail_counter:
            self.run_fail_counter[launch_config] = 0
        should_run = (
            not launch_config.model.skip_run
            and launch_config.status == ERROR_CODES.SUCCESS.value
        )
        if should_run:
            launch_config.status = run_model(launch_config)
        if should_run and launch_config.status == self.OUT_OF_MEMORY_ERROR_CODE:
            raise RuntimeError("run failed with out of memory error")
        else:
            self.submit_tpe.submit(self.remove_run, launch_config)
            self.submit_tpe.submit(self.finish, launch_config)

    def try_compile(self, launch_config: LaunchConfig):
        try:
            self.compile(launch_config)
        except Exception as exc:
            LOG.warning(f"compile model: {launch_config.model.info.name} failed with status: {launch_config.status}, fail number: {self.compile_fail_counter.get(launch_config)}, exception: {exc}")
            if self.compile_fail_counter.get(launch_config, self.retry_on_failure) < self.retry_on_failure:
                LOG.info(f" re-submiting model compilation, model: {launch_config.model.info.name}")
                self.compile_fail_counter[launch_config] += 1
                self.compile_tpe.submit(self.try_compile, launch_config)
            else:
                self.submit_tpe.submit(self.remove_compile, launch_config)

    def try_run(self, launch_config: LaunchConfig):
        try:
            self.run(launch_config)
        except Exception as exc:
            LOG.warning(f"run model: {launch_config.model.info.name} failed with status: {launch_config.status}, fail number: {self.run_fail_counter.get(launch_config)}, exception: {exc}")
            if self.run_fail_counter.get(launch_config, self.retry_on_failure) < self.retry_on_failure:
                self.run_fail_counter[launch_config] += 1
                if "failed to capture device run time" in str(exc):
                    LOG.info(f" re-submiting model compilation, model: {launch_config.model.info.name}")
                    self.submit_tpe.submit(self.submit_compile, launch_config)
                    self.submit_tpe.submit(self.remove_run, launch_config)
                else:
                    LOG.info(f" re-submiting model run, model: {launch_config.model.info.name}")
                    self.run_tpe.submit(self.try_run, launch_config)
            else:
                if launch_config.status == 0:
                    launch_config.status = -1
                self.submit_tpe.submit(self.remove_run, launch_config)
                self.submit_tpe.submit(self.finish, launch_config)

    def put(self, launch_config: LaunchConfig):
        if not launch_config.model.skip_compile:
            self.submit_tpe.submit(self.submit_compile, launch_config)
        else:
            if not launch_config.model.skip_run:
                self.submit_tpe.submit(self.submit_run, launch_config)

    def write_models_file(self):
        if self.marked_for_debug:
            with open(self.models_filter_file_path, "w") as f:
                json.dump(list(self.marked_for_debug), f, indent=4, sort_keys=True)

    def dispatch(self):
        f = self.submit_tpe.submit(self.scan)
        f.result()
        while (self.compile_activated + self.run_activated) > 0:
            time.sleep(1)
        self.submit_tpe.shutdown()
        self.compile_tpe.shutdown()
        self.run_tpe.shutdown()
        self.write_models_file()
