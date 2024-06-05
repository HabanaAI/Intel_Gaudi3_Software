#!/usr/bin/env python3

import argparse
import glob
import json
import os
import shutil
import signal
import sys
import time
from typing import Dict, List

import mpm_cl as mpm
import mpm_types as mtypes

class FileLock:
    def __init__(self, lock_file_path: str, timeout: int = 0):
        self.lock_file_path: str = lock_file_path
        self.timeout: int = timeout
        self.file = None

    def __enter__(self):
        start = time.time()
        elapsed = time.time() - start
        while self.timeout == 0 or elapsed <= self.timeout:
            try:
                with open(self.lock_file_path, "xt"):
                    pass
                break
            except OSError:
                time_since_creation = time.time() - os.path.getctime(self.lock_file_path)
                if time_since_creation > self.timeout:
                    print(f"Seems that the lock file failed to be released. The lock file was created {time_since_creation:.2f} seconds ago which exceeded the timeout: {self.timeout}")
                    os.remove(self.lock_file_path)
                time.sleep(0.1)
                elapsed = time.time() - start
        if self.timeout != 0 and elapsed > self.timeout:
            raise RuntimeError(
                f"failed to take file lock, file path: {self.lock_file_path} timeout: {self.timeout}"
            )

    def __exit__(self, exc_type, exc_val, exc_tb):
        if os.path.exists(self.lock_file_path):
            os.remove(self.lock_file_path)


def get_files(root_folder: str, postfix: str, recursive: bool):
    if not root_folder:
        return
    cmd = f"{root_folder}/**/*{postfix}" if recursive else f"{root_folder}/*{postfix}"
    yield from (
        f
        for f in glob.iglob(cmd, recursive=recursive)
        if os.path.isfile(f)
    )


def get_folders(root_folder, folder_name):
    if not root_folder:
        return
    yield from (
        f
        for f in glob.iglob(f"{root_folder}/**/{folder_name}", recursive=True)
        if os.path.isdir(f)
    )


class SignalCatcher:
    def __init__(self, lock_file_path: str):
        self.lock_file_path = lock_file_path
        signal.signal(signal.SIGINT, self.on_kill)
        signal.signal(signal.SIGTERM, self.on_kill)

    def on_kill(self, *args):
        print("received kill signal")
        os.remove(self.lock_file_path)
        exit(-1)


def copy_files(src_folder, dst_folder, file_ext, recursive):
    files = get_files(src_folder, f".{file_ext}", recursive)
    for f in files:
        shutil.copy(f, dst_folder)


class CommonConfig():
    def __init__(self, args):
        self.chip_type = args.chip_type
        self.job = args.job
        self.models = args.models
        self.max_mem_limit = args.max_mem_limit
        self.models_folder = args.models_folder
        self.models_stats_required = args.models_stats_required
        self.execution_time_limit = args.execution_time_limit
        self.unit_tests = args.unit_tests
        self.keep_going = args.keep_going
        self.test_name = args.test_name
        self.work_folder = f"mpm_{args.test_type}_{args.test_name}"
        self.local_results_path = os.path.join(self.work_folder, f"{args.test_type}_results")
        self.logs_folder = os.environ.get("HABANA_LOGS")
        self.results_folder = os.path.join(self.logs_folder, "results")
        self.graphs_stats_folder = os.path.join(self.logs_folder, "graphs_stats")


class AccuracyConfig(CommonConfig):
    def __init__(self, args):
        super().__init__(args)
        self.tensors_data_recording_folder = args.tensors_data_recording_folder


class CompileConfig(CommonConfig):
    def __init__(self, args):
        super().__init__(args)


class CompilationConsistencyConfig(CommonConfig):
    def __init__(self, args):
        super().__init__(args)
        self.compile_consistency_iters = args.compile_consistency_iters


class PerfConfig(CommonConfig):
    def __init__(self, args):
        super().__init__(args)
        self.compilation_threads = args.compilation_threads
        self.eager = False
        self.lock_timeout = args.lock_timeout
        self.publish_results = args.publish_results
        self.regression_threshold = args.regression_threshold
        self.run_iterations = args.run_iterations
        self.local_xml_results_path = os.path.join(self.work_folder, f"{args.test_type}_full_results")
        self.local_single_results_path = "single_results"
        self.local_data_path_curr = os.path.join(self.work_folder, f"{args.test_name}.db.json")
        self.local_input_data_path_status = os.path.join(self.work_folder, "status.input.db.json")
        self.local_output_data_path_status = os.path.join(self.work_folder, "status.output.db.json")
        self.db_full_name = f"{args.database}_{args.branch}_{args.chip_type}"
        self.remote_results_folder = os.path.join(args.public_results_folder, self.db_full_name)
        self.remote_data_path_status = os.path.join(self.remote_results_folder, "status.db.json")
        self.remote_db_folder = os.path.join(self.remote_results_folder, "db")
        self.allowed_regression_file = os.path.join(self.remote_results_folder, "allowed_regression.json")
        self.lock_file = os.path.join(self.remote_results_folder, f".{self.db_full_name}.lock")
        self.regression_threshold_file = os.path.join(args.public_results_folder, f"{args.chip_type}_models_threshold.json")

def accuracy_test(args):
    config = AccuracyConfig(args)
    os.makedirs(config.results_folder, exist_ok=True)
    os.makedirs(config.graphs_stats_folder, exist_ok=True)

    base_cmd=["--accuracy", "--metrics", "accuracy", "--names", config.test_name, "--work_folder", config.work_folder, "--log_level", "d", "--chip_type", config.chip_type, "--tensors_data_folder", config.tensors_data_recording_folder, "--time_measurement", "none"]

    base_cmd += ["--jobs", (config.job if config.job else "accuracy")]

    if config.models_folder:
        base_cmd += ["--models_folder", config.models_folder]

    if config.max_mem_limit:
        base_cmd += ["--max_mem_limit"] + [str(config.max_mem_limit)]

    if config.unit_tests:
        base_cmd += ["--unit_tests"]

    if config.keep_going:
        base_cmd += ["--keep_going"]

    if config.execution_time_limit:
        base_cmd += ["--execution_time_limit", str(config.execution_time_limit)]

    if config.models_stats_required:
        base_cmd += ["--models_stats_required"]

    # for debug
    if config.models:
        base_cmd += ["-m"] + config.models

    compile_cmd=["--actions", "compile"]
    sts = mpm.main(mpm.parse_args(base_cmd + compile_cmd))
    if sts != mtypes.ERROR_CODES.SUCCESS.value:
        print(f"ERROR: failed to compile models, error code: {sts}")

    run_cmd = ["--actions", "run", "report", "--release_device", "--report_path", config.local_results_path, "--report_format", "xml", "csv"]
    sts = mpm.main(mpm.parse_args(base_cmd + run_cmd))
    if sts != mtypes.ERROR_CODES.SUCCESS.value:
        print(f"ERROR: failed to execute models tests, error code: {sts}")

    test_folder = os.path.join(config.work_folder, config.test_name)
    if sts != mtypes.ERROR_CODES.SUCCESS.value:
        folders = get_folders(test_folder, "logs-compile")
        for d in folders:
            shutil.copytree(d, config.logs_folder, dirs_exist_ok=True)
        folders = get_folders(test_folder, "logs-runtime")
        for d in folders:
            shutil.copytree(d, config.logs_folder, dirs_exist_ok=True)

    copy_files(test_folder, config.graphs_stats_folder, "csv", True)

    copy_files(config.work_folder, config.results_folder, "log", False)
    copy_files(config.work_folder, config.results_folder, "csv", False)
    copy_files(config.work_folder, config.results_folder, "xml", False)
    copy_files(config.work_folder, ".", "xml", False)

    if sts != mtypes.ERROR_CODES.SUCCESS.value:
        print(f"ERROR: accuracy test failed, error code: {sts}")
    else:
        print("Model accuracy tests fininshed successfully!")
    return sts


def compilation_consistency_test(args):
    config = CompilationConsistencyConfig(args)
    os.makedirs(config.results_folder, exist_ok=True)
    os.makedirs(config.graphs_stats_folder, exist_ok=True)

    base_cmd=["--actions", "compile", "--compile_consistency_iters", str(config.compile_consistency_iters), "--metrics", "compile_consistency", "--names", config.test_name, "--work_folder", config.work_folder, "--log_level", "d", "--chip_type", config.chip_type, "--time_measurement", "none", "--report_path", config.local_results_path, "--report_format", "xml", "csv"]

    base_cmd += ["--jobs", (config.job if config.job else "compile_consistency")]

    if config.models_folder:
        base_cmd += ["--models_folder", config.models_folder]

    if config.max_mem_limit:
        base_cmd += ["--max_mem_limit"] + [str(config.max_mem_limit)]

    if config.unit_tests:
        base_cmd += ["--unit_tests"]

    if config.keep_going:
        base_cmd += ["--keep_going"]

    if config.execution_time_limit:
        base_cmd += ["--execution_time_limit", str(config.execution_time_limit)]

    if config.models_stats_required:
        base_cmd += ["--models_stats_required"]

    # for debug
    if config.models:
        base_cmd += ["-m"] + config.models

    sts = mpm.main(mpm.parse_args(base_cmd))

    test_folder = os.path.join(config.work_folder, config.test_name)
    if sts != mtypes.ERROR_CODES.SUCCESS.value:
        folders = get_folders(test_folder, "logs-compile")
        for d in folders:
            shutil.copytree(d, config.logs_folder, dirs_exist_ok=True)
        folders = get_folders(test_folder, "logs-runtime")
        for d in folders:
            shutil.copytree(d, config.logs_folder, dirs_exist_ok=True)

    copy_files(test_folder, config.graphs_stats_folder, "csv", True)

    copy_files(config.work_folder, config.results_folder, "log", False)
    copy_files(config.work_folder, config.results_folder, "csv", False)
    copy_files(config.work_folder, config.results_folder, "xml", False)
    copy_files(config.work_folder, ".", "xml", False)

    if sts != mtypes.ERROR_CODES.SUCCESS.value:
        print(f"ERROR: compile consistency test failed, error code: {sts}")
    else:
        print("Model compile consistency tests fininshed successfully!")
    return sts


def compilation_test(args):
    config = CompileConfig(args)
    os.makedirs(config.results_folder, exist_ok=True)
    os.makedirs(config.graphs_stats_folder, exist_ok=True)

    compile_cmd=["--actions", "compile", "--names", config.test_name, "--work_folder", config.work_folder, "--log_level", "d", "--chip_type", config.chip_type, "--report_format", "xml", "csv"]

    compile_cmd += ["--jobs", (config.job if config.job else "compilation")]

    if config.models_folder:
        compile_cmd += ["--models_folder", config.models_folder]

    if config.max_mem_limit:
        compile_cmd += ["--max_mem_limit"] + [str(config.max_mem_limit)]

    if config.unit_tests:
        compile_cmd += ["--unit_tests"]

    if config.models_stats_required:
        compile_cmd += ["--models_stats_required"]

    # for debug
    if config.models:
        compile_cmd += ["-m"] + config.models

    if config.execution_time_limit:
        compile_cmd += ["--execution_time_limit", str(config.execution_time_limit)]

    sts = mpm.main(mpm.parse_args(compile_cmd))
    if sts != mtypes.ERROR_CODES.SUCCESS.value:
        print(f"ERROR: failed to compile models, error code: {sts}")

    test_folder = os.path.join(config.work_folder, config.test_name)
    if sts != mtypes.ERROR_CODES.SUCCESS.value:
        folders = get_folders(test_folder, "logs-compile")
        for d in folders:
            shutil.copytree(d, config.logs_folder, dirs_exist_ok=True)

    copy_files(test_folder, config.graphs_stats_folder, "csv", True)

    copy_files(config.work_folder, config.results_folder, "log", False)
    copy_files(config.work_folder, config.results_folder, "csv", False)
    copy_files(config.work_folder, config.results_folder, "xml", False)
    copy_files(config.work_folder, ".", "xml", False)

    if sts != mtypes.ERROR_CODES.SUCCESS.value:
        print(f"ERROR: compilation test failed, error code: {sts}")
    else:
        print("Model compilation tests fininshed successfully!")
    return sts


def read_json_file(json_file_path) -> dict:
    with open(json_file_path) as jf:
        return json.load(jf)


def perf_eager_reg_test(args):
    config = PerfConfig(args)
    config.eager = True
    return perf_reg_test(config)

def perf_graph_reg_test(args):
    config = PerfConfig(args)
    return perf_reg_test(config)


def get_allowed_regression_models(file_path: str) -> Dict[str,float]:
    if not os.path.exists(file_path):
        return {}
    with open(file_path) as jf:
        return json.load(jf)


def ignore_results(config):
    allowed_regression = get_allowed_regression_models(config.allowed_regression_file)
    if not allowed_regression:
        return False
    for m, r in allowed_regression.items():
        ignore_results_cmd = ["--chip_type", config.chip_type, "--ignore_results", str(r), "--ref_results", config.local_input_data_path_status, "--data_path", config.local_input_data_path_status, "-m", m]
        if config.models_folder:
            ignore_results_cmd += ["--models_folder", config.models_folder]
        mpm.main(mpm.parse_args(ignore_results_cmd))
    return True


def perf_reg_test(config):
    signal_catcher = SignalCatcher(config.lock_file)

    os.makedirs(config.results_folder, exist_ok=True)
    os.makedirs(config.graphs_stats_folder, exist_ok=True)
    os.makedirs(config.remote_db_folder, exist_ok=True)


    if not os.path.exists(config.regression_threshold_file):
        with open(config.regression_threshold_file, "w") as f:
            f.write("{}")

    base_cmd=["--names", config.test_name, "--work_folder", config.work_folder, "--log_level", "d", "--chip_type", config.chip_type, "--threads", str(config.compilation_threads)]

    if config.job:
        base_cmd += ["--jobs", config.job]

    if config.models_folder:
        base_cmd += ["--models_folder", config.models_folder]

    if config.eager:
        base_cmd += ["compilation_mode", "eager", "time_measurement", "profiler"]

    if config.max_mem_limit:
        base_cmd += ["--max_mem_limit"] + [str(config.max_mem_limit)]

    if config.unit_tests:
        base_cmd += ["--unit_tests"]

    if config.models_stats_required:
        base_cmd += ["--models_stats_required"]

    # for debug
    if config.models:
        base_cmd += ["-m"] + config.models

    if config.execution_time_limit:
        base_cmd += ["--execution_time_limit", str(config.execution_time_limit)]

    compile_and_run_cmd = base_cmd + ["--actions", "compile", "run", "--report_path", config.local_single_results_path, "--data_path", config.local_data_path_curr, "--dump_curr_tests", "--metrics", "compile", "run"]
    sts = mpm.main(mpm.parse_args(compile_and_run_cmd))
    if sts != mtypes.ERROR_CODES.SUCCESS.value:
        print(f"ERROR: failed to execute models, error code: {sts}")

    report_cmd = base_cmd + ["--actions", "run", "report", "--report_format", "xml", "csv", "--report_path", config.local_xml_results_path, "--data_path", config.local_output_data_path_status, "--dump_curr_status", "--threshold", str(config.regression_threshold), "--threshold_file", config.regression_threshold_file, "--ref_best"]

    print(f"Try lock file: {config.lock_file}")
    with FileLock(config.lock_file, config.lock_timeout):
        if os.path.exists(config.remote_data_path_status):
            shutil.copy(config.remote_data_path_status, config.local_input_data_path_status)
            report_cmd += ["--ref_results", config.local_input_data_path_status]
        sts = mpm.main(mpm.parse_args(report_cmd))
        absorb_regression = False
        if sts != mtypes.ERROR_CODES.SUCCESS.value and ignore_results(config):
            sts = mpm.main(mpm.parse_args(report_cmd))
            absorb_regression = sts == mtypes.ERROR_CODES.SUCCESS.value
        if config.publish_results:
            if absorb_regression:
                print(f"clear allowed regression list in: {config.allowed_regression_file}")
                with open(config.allowed_regression_file, "w") as f:
                    json.dump({}, f, indent=4, sort_keys=True)
            print("publish results")
            shutil.copy(config.local_data_path_curr, config.remote_db_folder)
            shutil.copy(config.local_output_data_path_status, config.remote_data_path_status)
    print(f"Lock file: {config.lock_file} is released")

    # keep the logs in case of failure
    test_folder = os.path.join(config.work_folder, config.test_name)
    if sts != mtypes.ERROR_CODES.SUCCESS.value:
        folders = get_folders(test_folder, "logs-compile")
        for d in folders:
            shutil.copytree(d, os.path.join(config.logs_folder, "logs-compile"), dirs_exist_ok=True)
        folders = get_folders(test_folder, "logs-runtime")
        for d in folders:
            shutil.copytree(d, os.path.join(config.logs_folder, "logs-runtime"), dirs_exist_ok=True)

    copy_files(test_folder, config.graphs_stats_folder, "csv", True)

    copy_files(config.work_folder, config.results_folder, "log", False)
    copy_files(config.work_folder, config.results_folder, "csv", False)
    copy_files(config.work_folder, config.results_folder, "json", False)
    copy_files(config.work_folder, config.results_folder, "xml", False)
    copy_files(config.work_folder, config.results_folder, "txt", False)
    shutil.copy(f"{config.local_xml_results_path}.xml", ".") # jenkins job searches the results in the current folder

    if sts != mtypes.ERROR_CODES.SUCCESS.value:
        print(f"Model perf tests fininshed with error: {sts}")
    else:
        print("Model perf tests fininshed successfully!")

    return sts


def main(args):
    if args.test_type == "accuracy":
        return accuracy_test(args)
    if args.test_type == "compilation":
        return compilation_test(args)
    if args.test_type == "compilation_consistency":
        return compilation_consistency_test(args)
    if args.test_type == "perf":
        return perf_graph_reg_test(args)
    if args.test_type == "perf_eager":
        return perf_eager_reg_test(args)

    print(f"ERROR: wrong test type: {args.test_type}")
    return -1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-type",
        choices=["accuracy", "compilation", "compilation_consistency", "perf", "perf_eager"],
        required=True,
    )
    parser.add_argument(
        "--test-name",
        default=f"{time.strftime('%Y%m%d-%H%M%S')}",
    )
    parser.add_argument(
        "--models-folder",
        help="Models folder path",
    )
    parser.add_argument(
        "--job",
        help="Models job group",
    )
    parser.add_argument(
        "--publish-results",
        action="store_true",
    )
    parser.add_argument(
        "--database",
        help="Database name, partial identifier of the full database path",
    )
    parser.add_argument(
        "--branch",
        help="Git branch, partial identifier of the full database path",
    )
    parser.add_argument(
        "--chip-type",
        choices=["gaudi", "gaudi2", "gaudi3"],
        help="Select chip type, partial identifier of the full database path",
        required=True,
    )
    parser.add_argument(
        "--public-results-folder",
        default="/software/ci/models_tests",
        help="Path to remote results",
    )
    parser.add_argument(
        "--tensors-data-recording-folder",
        default="/software/ci/models_tests/data_rec",
        help="Path to recorded tensors folder",
    )
    parser.add_argument(
        "--lock-timeout",
        type=int,
        default=600,
        help="File lock timeout",
    )
    parser.add_argument(
        "--regression-threshold",
        type=float,
        default=2,
        help="Set the defaults regression threshold that fails the test",
    )
    parser.add_argument(
        "--run-iterations",
        type=int,
        default=10,
        help="Number of synLaunch for each graph",
    )
    parser.add_argument(
        "--max-mem-limit",
        type=int,
        help="Limit the max memory consumption",
    )
    parser.add_argument(
        "--compilation-threads",
        type=int,
        default=0,
        help="Number of models compilation threads",
    )
    parser.add_argument(
        "-m",
        "--models",
        nargs="+",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--unit-tests",
        action="store_true",
        help="Run recorded unit tests instead of models",
    )
    parser.add_argument(
        "--keep-going", action="store_true", help="Continue on non fatal errors"
    )
    parser.add_argument(
        "--execution-time-limit",
        type=float,
        help="Max allowed execution time in seconds",
    )
    parser.add_argument(
        "--compile-consistency-iters",
        type=int,
        help="Number of compilation consistency iterations"
    )
    parser.add_argument(
        "--models-stats-required",
        action="store_true",
        help="Stop execution on missing models stats (max_memory_usage or duration)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(f"models tests ci args: {' '.join(sys.argv)}")
    args = parse_args()
    exit(main(args))
