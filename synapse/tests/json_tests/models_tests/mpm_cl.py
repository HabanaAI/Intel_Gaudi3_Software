#!/usr/bin/env python3
import argparse
import os
import shutil
import sys
import time

import mpm
import mpm_base_metric
import mpm_logger
import mpm_utils
from mpm_types import ERROR_CODES, Devices, GlobalConfig, GlobalConstants, MetricType


def main(args):
    global_config = GlobalConfig(args)
    if args.ignore_tests:
        mpm.ignore_tests(
            args.ref_results, args.ignore_tests, args.models, args.data_path
        )
        return ERROR_CODES.SUCCESS.value
    elif args.ignore_results:
        mpm.ignore_results(global_config, args.ignore_results)
        return ERROR_CODES.SUCCESS.value

    status = mpm.run(global_config)

    if status is None:
        print("Models test run finished with unknown status")
        return ERROR_CODES.ERROR.value

    if status is ERROR_CODES.SUCCESS.value:
        print("Models test run finished successfully!")
    else:
        print("One or more actions failed, run with '-l d' for more details")
        print(f"Exit with error, code: {status}")

    return status


def parse_args(sys_args):
    global_constants = GlobalConstants()
    dep_parser = argparse.ArgumentParser(add_help=False)
    dep_parser.add_argument(
        "-w",
        "--work_folder",
        help="Work folder path",
        default="mpm-{}".format(time.strftime("%Y%m%d-%H%M%S")),
    )
    models_source = dep_parser.add_mutually_exclusive_group()
    models_source.add_argument(
        "--models_folder",
        help="Full path to models folder",
    )
    models_source.add_argument(
        "--models_revision",
        help="Models git revision",
        default="master_next",
    )
    dep_parser.add_argument(
        "-l",
        "--log_level",
        choices=mpm_logger.log_levels.keys(),
        help="Set mpm log level",
        default="i",
    )
    dep_parser.add_argument(
        "-u",
        "--unit_tests",
        action="store_true",
        help="Run recorded unit tests instead of models",
    )
    dep_args, _ = dep_parser.parse_known_args(sys_args)

    if dep_args.log_level:
        mpm_logger.config_logger(dep_args.log_level, dep_args.work_folder)
        mpm_logger.LOG.info(f"mpm args: {' '.join(sys_args)}")

    if not dep_args.models_folder:
        mpm.checkout_models_revision(dep_args.models_revision, global_constants)
    models_folder = dep_args.models_folder or global_constants.DEFAULT_MODELS_FOLDER

    if not os.path.exists(models_folder):
        dep_parser.error(f"models folder path {models_folder} doesn't exists")

    dep_parser.add_argument(
        "--models_file",
        help="Path to models file",
        default=os.path.join(models_folder, (".default.tests-list.json" if dep_args.unit_tests else ".default.models-list.json")),
    )

    dep_args, _ = dep_parser.parse_known_args(sys_args)

    models_file = dep_args.models_file if os.path.exists(dep_args.models_file) else None

    available_jobs = mpm.get_jobs(models_file)

    dep_parser.add_argument(
        "--jobs",
        nargs="+",
        choices=available_jobs,
        help="Filter models by job types",
    )
    dep_parser.add_argument(
        "-c",
        "--chip_type",
        choices=[d.value for d in Devices],
        help="Select chip type",
    )

    dep_args, _ = dep_parser.parse_known_args(sys_args)

    dep_parser.add_argument(
        "--time_measurement",
        choices=["none", "events", "profiler"],
        help="Set time measurement mechanism",
        default=("events" if dep_args.chip_type == Devices.GAUDI_1.value else "profiler")
    )

    dep_args, _ = dep_parser.parse_known_args(sys_args)

    parser = argparse.ArgumentParser(parents=[dep_parser])

    # avoid running disabled tests
    if not dep_args.jobs and "default" in available_jobs:
        dep_args.jobs = ["default"]

    # if a models file is set, use it with the requested models folder.
    # else if a models folder contains models file, use it.
    # otherwise run all models in the requested models folder.
    filterd_models = mpm.get_supported_models(models_folder, models_file, dep_args.chip_type, dep_args.jobs)
    all_models = mpm.get_supported_models(models_folder, None)

    mpm.log_models_info(models_folder, models_file, filterd_models)

    parser.add_argument(
        "-m",
        "--models",
        nargs="+",
        choices=all_models,
        help=", ".join(all_models),
        metavar="",
        default=filterd_models,
    )
    names_group = parser.add_mutually_exclusive_group()
    names_group.add_argument(
        "-n",
        "--names",
        nargs="+",
        help="List of test names",
        default=[dep_args.chip_type],
    )
    names_group.add_argument(
        "-g",
        "--git_revs",
        nargs="+",
        help="List of git revs (hash/branch/tag)",
    )
    names_group.add_argument(
        "--config_compare_file",
        help=("Allows perf compare between runs of the same model with different configurations. "
            "Requires a json file that contains test runs configurations. The file format is as follows:"
            "list of runs configurations where each run configurations are written as dictionary such that key=<config_name> and value=<config_value>")
    )
    names_group.add_argument(
        "--config_compare_values",
        nargs="+",
        help=("Allows perf compare between runs of the same model with different configurations. "
        "Requires one or more configuration pairs of this type: <config_name, config_value>, each pair is set on its own compilation and run process."
        "In case only one pair is specified then it is compared to the default configurations.")
    )
    parser.add_argument(
        "-i",
        "--iterations",
        type=int,
        help="Number of model run iterations",
        default = 10 if dep_args.time_measurement == "events" else 1,
    )
    parser.add_argument(
        "-a",
        "--actions",
        nargs="+",
        choices=mpm.action_types,
        help="Select which actions to take",
        default=mpm.action_types,
    )
    parser.add_argument(
        "-f",
        "--report_format",
        nargs="+",
        choices=mpm.file_format,
        help="Select the report file format",
        default=[mpm.file_format[0]],
    )
    parser.add_argument(
        "-r",
        "--report_path",
        help="Report file path",
        default="{}/results".format(dep_args.work_folder),
    )
    parser.add_argument(
        "-d",
        "--data_path",
        help="Data file path",
        default="{}/data.json".format(dep_args.work_folder),
    )
    dump_group = parser.add_mutually_exclusive_group()
    dump_group.add_argument(
        "--dump_curr_tests",
        action="store_true",
        help="If --data_path is set, dump only the results from the work folder, ignoring the ref results",
    )
    dump_group.add_argument(
        "--dump_curr_status",
        action="store_true",
        help="If --data_path is set, dump only the best results from the work folder, ignoring the ref results,"
        "in case perf regression was detected dump the regressed result instead the best result",
    )
    release_device_group = parser.add_mutually_exclusive_group()
    release_device_group.add_argument(
        "--acquire_device_once",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    release_device_group.add_argument(
        "--release_device",
        action="store_true",
        help="Release device after each graph run",
    )
    parser.add_argument(
        "--retry_on_failure",
        type=int,
        help="Number of retrys in case of compile/run failures",
        default=5,
    )
    parser.add_argument(
        "--retry_on_perf_reg",
        type=int,
        default=1,
        help="Number of retrys in case of performance regression",
    )
    parser.add_argument(
        "--accuracy",
        action="store_true",
        help="Check data accuracy in case tensors data recording file is available in (--tensors_data_folder)",
    )
    parser.add_argument(
        "--tensors_data_folder",
        help="Full path to models folder",
        default="/software/ci/models_tests/data_rec"
    )
    parser.add_argument(
        "--data_comparator_config",
        help="Data comparator config file path",
        default=os.path.join(os.path.dirname(os.path.realpath(__file__)), "data_comparator_config.json"),
    )
    ref_test_group = parser.add_mutually_exclusive_group()
    ref_test_group.add_argument(
        "--ref_name",
        help="Reference test name for relative performance comparison",
    )
    ref_test_group.add_argument(
        "--ref_best",
        action="store_true",
        help="Calculate performance gain compare to best results",
    )
    ref_test_group.add_argument(
        "--regression_check",
        action="store_true",
        help="Compare performance with the best known run of the selected device",
    )
    ref_test_group.add_argument(
        "--ref_file",
        help="Path to Json file with ref method per model, format: { '<model_name>': '<test_name>' }",
    )
    parser.add_argument(
        "--gen_ref_file",
        help="Generate Json file with ref method per model in the provided path, format: { '<model_name>': '<test_name>' }",
    )
    parser.add_argument(
        "--ref_results",
        nargs="+",
        help="List of json files with previous measurements",
    )
    parser.add_argument(
        "-p",
        "--precompiled_test",
        help="Precompiled test name, run compiled recipes form the test folder.\n"
        "The new results are written to a different folder according to the test name.\n"
        "Compile action is ignored",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        help="Max allowed perf regression percentage",
    )
    parser.add_argument(
        "--threshold_file",
        help="File path to threshold per model, format: { '<model_name>': <threshold> }",
    )
    parser.add_argument(
        "--workspace_size_threshold",
        type=float,
        help="Max allowed workspace size regression percentage",
    )
    parser.add_argument(
        "-b", "--build", action="store_true", help="Build synapse before execution"
    )
    syn_build = os.getenv("SYNAPSE_RELEASE_BUILD")
    parser.add_argument(
        "--models_tests_bin",
        help="Full path to models_tests binary",
        default=os.path.join(syn_build, "bin", "json_tests") if syn_build else None,
    )
    parser.add_argument(
        "--repo",
        choices=mpm.REPOS,
        help="Select the repository to validate",
        default=mpm.REPOS[0],
    ),
    parser.add_argument(
        "--prof",
        action="store_true",
        help=argparse.SUPPRESS,
    ),
    parser.add_argument(
        "--set_const_tensor_max_size",
        type=int,
        help="Set const tensor max size, set 0 to use synapse defaults",
        default=0x1000000,
    ),
    parser.add_argument(
        "--ignore_tests",
        nargs="+",
        help="A list of tests names that are set to be ignored when checking regression.\n"
        "This will create a copy of the results file (--ref_results) at the requsted data path (--data_path) with the requsted test marked as ignored.\n"
        "Only the requested models (--models) are ignored.",
    )
    parser.add_argument(
        "--ignore_results",
        type=float,
        help="Any result for a specified model that is better than the requestd result is set to be ignored when checking regression.\n"
        "This will create a copy of the results file (--ref_results) at the requsted data path (--data_path) with the relevant results mark as ignored.\n"
        "Only the requested model (--models) is ignored, if more than one model is set, the run will fail.",
    )
    parser.add_argument(
        "--show_precompiled", action="store_true", help="Show also precompiled tests results"
    )
    parser.add_argument(
        "--graphs", action="store_true", help="Show model run result per graph"
    )
    parser.add_argument(
        "--keep_going", action="store_true", help="Continue on non fatal errors"
    )
    parser.add_argument(
        "--ignore_errors", action="store_true", help="Continue on env configuration errors"
    )
    threads_group = parser.add_mutually_exclusive_group()
    threads_group.add_argument(
        "--threads",
        type=int,
        help="Limit the number of compilation threads, 0 for no limit",
        default=0,
    )
    parser.add_argument(
        "--max_mem_limit",
        type=int,
        help="Limit the max memory consumption (bytes)",
    )
    parser.add_argument(
        "--cards",
        type=int,
        help="Run on multiple habana cards",
        default=1,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing results",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        help="Select which metrics should be execute to validate the run results",
        choices=[m.name.lower() for m in mpm_base_metric.MetricType],
        default=[m.name.lower() for m in mpm_base_metric.DEFAULT_METRICS],
    )
    parser.add_argument(
        "--skip_models",
        action="store_true",
        help="Generates a file that contains a list of models that were marked by the compiler.\n"
        "A model that wasn't marked by the compiler will not run.",
    )
    parser.add_argument(
        "--compilation_mode",
        type=str,
        choices=['graph', 'eager'],
        default='graph',
        help="Force a compilation mode on all graphs in json",
    )
    parser.add_argument(
        "--execution_time_limit",
        type=float,
        help="Max allowed compile/run time in seconds",
    )
    parser.add_argument(
        "--compile_consistency_iters",
        type=int,
        help="Number of compilation consistency iterations"
    )
    parser.add_argument(
        "--models_stats_required",
        action="store_true",
        help="Stop execution on missing models stats (max_memory_usage or duration).\n"
    )
    args = parser.parse_args(sys_args)
    args.models_folder = models_folder
    if args.accuracy:
        local_tensors_data_folder = global_constants.DEFAULT_DATA_FOLDER
        mpm_logger.LOG.info(f"copy tensors data folder locally, from: {args.tensors_data_folder}, to: {local_tensors_data_folder}")
        shutil.copytree(args.tensors_data_folder, local_tensors_data_folder, dirs_exist_ok=True)
        args.tensors_data_folder = local_tensors_data_folder
    args.models_revision = mpm_utils.get_git_commit(models_folder, True)
    if os.environ.get("HABANA_PROFILE") == str(1):
        args.prof = True
    if args.acquire_device_once:
        parser.error('--acquire_device_once is activated by default, please remove it from the args list')
    if args.time_measurement == "none" and MetricType.RUN_TIME.name.lower() in args.metrics:
        args.metrics.remove(MetricType.RUN_TIME.name.lower())
    if (args.prof and args.time_measurement == "profiler") or os.environ.get("HABANA_PROFILE") is not None:
        parser.error("Argument --time_measurement shouldn't be set to profiler if profiler is enabled, run with --time_measurement none")
    if not mpm.is_report_only(args.actions) and args.chip_type is None:
        parser.error('Argument --chip_type is required for compile/run')
    if args.regression_check:
        if args.ref_results is None:
            args.ref_results = [mpm.get_device_default_ref_results(args.chip_type)]
        if args.threshold is None:
            args.threshold = 2
        args.ref_best = True

    if os.getcwd().startswith(os.path.join(os.environ['HOME'], "qnpu")) and "QNPU_PATH" not in os.environ and args.ignore_errors is False:
        parser.error('Running within qnpu dir without activating it. Use --ignore_errors to skip this check')

    os.environ[
        "LD_LIBRARY_PATH"
    ] = f'{syn_build}/lib:{os.environ.get("LD_LIBRARY_PATH", "")}'

    return args


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    exit(main(args))
