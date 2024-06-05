#!/usr/bin/env python
import argparse
import collections
import copy
import itertools
import logging
import os
import re
import subprocess
import textwrap
import time
import xml.etree.ElementTree as ET
from multiprocessing import Pool
from typing import Dict, List, Tuple

import compilation_time_analyzer
import graph_editor
import parse_callgrind_file
import syn_infra as syn

BUNDLE_REPRODUCTION_MODE_FAILURE="Bundle reproduction mode failed"

MODELS_REPO = "ssh://gerrit:29418/mpm-test-data"
MODELS_REVISION = "master_next"
LOCAL_MODELS_FOLDER = f"/tmp/.mpm"
MODELS_TESTS_PATH = f"{LOCAL_MODELS_FOLDER}/models"

CHIP_TYPE_NAME = "gaudi"
DEFAULT_DATA_COMPARATOR_CONFIG_FILE=os.path.join(os.path.dirname(os.path.abspath(__file__)), "default_data_comparator_config.json")
LOG = syn.config_logger("json_runner", logging.INFO)


try:
    import termcolor
    colored=termcolor.colored
except:
    colored=lambda x, y, z: x


def set_const_tensor_max_size(max_size):
    LOG.info(f"set const tensor max size to: {max_size}")
    if os.environ.get("ENABLE_EXPERIMENTAL_FLAGS") is None:
        os.environ["ENABLE_EXPERIMENTAL_FLAGS"] = "true"
    if os.environ.get("MAX_CONST_TENSOR_SIZE_BYTES") is None:
        os.environ["MAX_CONST_TENSOR_SIZE_BYTES"] = str(max_size)
    if os.environ.get("HBM_GLOBAL_MEM_SIZE_MEGAS") is None:
        os.environ["HBM_GLOBAL_MEM_SIZE_MEGAS"] = "256"


def get_all_models(folder, skip_models_download):
    LOG.debug("get all models")
    if not skip_models_download:
        if os.path.exists(f"{LOCAL_MODELS_FOLDER}/.git"):
            syn.git_pull(LOCAL_MODELS_FOLDER)
        else:
            LOG.info("clone models folder")
            syn.create_folder(LOCAL_MODELS_FOLDER)
            syn.git_clone(MODELS_REPO, LOCAL_MODELS_FOLDER)
        syn.git_checkout(LOCAL_MODELS_FOLDER, MODELS_REVISION)
    models = {}
    for tf in syn.get_files(folder, ".json"):
        models[os.path.basename(tf).replace(".json", "")] = tf
    return models


def get_models_set(models_file):
    MODELS_SET = set()
    try:
        LOG.debug(f"Parsing models filter file: {models_file}")
        with open(models_file) as f:
            models_list = [
                l for l in (line.strip() for line in f) if l and not l.startswith("#")
            ]
        if isinstance(models_list, list):
            MODELS_SET = set(models_list)
            return MODELS_SET
    except:
        raise RuntimeError(
            "Failed to read models_file: {models_file}, the file might be malformed, it should be a single list of strings []"
        )


def get_supported_models(folder, models_file, skip_models_download):
    models = get_all_models(folder, skip_models_download)
    if models_file is None:
        return [m for m in models]
    models_set = get_models_set(models_file)
    return [m for m in models if m in models_set]


def validate_json_path(json_file):
    if not os.path.exists(json_file):
        raise RuntimeError(f'json file {json_file} does not exist')
    LOG.info("executing json file: {}".format(json_file))


def validate_recipe_path(recipe_file):
    if not os.path.exists(recipe_file):
        raise RuntimeError(f'recipe file {recipe_file} does not exist')
    LOG.info("executing recipe file: {}".format(args.recipe))


def get_consistency_cmd_args(args, json_file) -> "List[str]":
    validate_json_path(json_file)
    cmd_args = ["--json-file", json_file]
    if args.graph_indices is not None:
        cmd_args += ["--graphs-indices"] + [str(i) for i in args.graph_indices]
    if args.exclude_graphs:
        cmd_args += ["--exclude-graphs"]
    if args.keep_going:
        cmd_args += ["--keep-going"]
    if args.test_iters:
        cmd_args += ["--test-iter", str(args.test_iters)]
    return cmd_args


def get_playback_cmd_args(args, json_file) -> "List[str]":
    validate_json_path(json_file)
    cmd_args = ["--json-file", json_file]
    if args.stats_json_path is not None:
        cmd_args += ["--stats-file", args.stats_json_path]
    if (args.run or args.recipe) and args.iterations:
        cmd_args += ["--run-iter", str(args.iterations)]
    if args.serialize_recipe:
        syn.create_folder(args.serialize_recipe)
        cmd_args += ["--serialize-recipe", args.serialize_recipe]
    if args.test_iters:
        cmd_args += ["--test-iter", str(args.test_iters)]
    if args.iterations_filter:
        cmd_args += ["--run-iter-filter"] + [str(i) for i in args.iterations_filter]
    if args.run or args.prof or args.synthetic_data:
        cmd_args += ["--run"]
        if not args.run:
            LOG.warning("--run was enabled automatically since --prof is enabled")
    if args.groups is not None:
        cmd_args += ["--groups"] + [str(i) for i in args.groups]
    if args.graph_indices is not None:
        cmd_args += ["--graphs-indices"] + [str(i) for i in args.graph_indices]
    if args.exclude_graphs:
        cmd_args += ["--exclude-graphs"]
    if args.synthetic_data:
        cmd_args += ["--synthetic-data"]
    if args.compilation_mode is not None:
        cmd_args += ["--compilation-mode", args.compilation_mode]
    if args.quiet:
        cmd_args += ["--quiet"]
    if args.const:
        cmd_args += ["--const", args.const]
    if args.time_measurement:
        cmd_args += ["--time-measurement", args.time_measurement]
    if args.comp_config_file:
        cmd_args += ["--comp-config-file", args.comp_config_file]
    if args.reset_device:
        cmd_args += ["--reset-device"]
    if args.keep_going:
        cmd_args += ["--keep-going"]
    return cmd_args


def get_perf_cmd_args(args, json_file) -> "List[str]":
    cmd_args = get_playback_cmd_args(args, json_file)
    return cmd_args


def get_config_compare_cmd_args(args, json_file) -> "List[str]":
    validate_json_path(json_file)
    cmd_args = ["--json-file", json_file]
    if args.stats_json_path is not None:
        cmd_args += ["--stats-file", args.stats_json_path]
    if args.graph_indices is not None:
        cmd_args += ["--graphs-indices"] + [str(i) for i in args.graph_indices]
    if args.exclude_graphs:
        cmd_args += ["--exclude-graphs"]
    if args.compilation_mode is not None:
        cmd_args += ["--compilation-mode", args.compilation_mode]
    if args.data_file:
        cmd_args += ["--data-file", args.data_file]
    if args.comp_config_file:
        cmd_args += ["--comp-config-file", args.comp_config_file]
    if args.config_compare_values:
        cmd_args += ["--config-compare-values"] + args.config_compare_values
    if args.config_compare_file:
        cmd_args += ["--config-compare-file", args.config_compare_file]
    return cmd_args


def get_recipe_cmd_args(args) -> "List[str]":
    validate_recipe_path(args.recipe)
    cmd_args = ["--recipe-file", args.recipe]
    if (args.run or args.recipe) and args.iterations:
        cmd_args += ["--run-iter", str(args.iterations)]
    if args.stats_json_path is not None:
        cmd_args += ["--stats-file", args.stats_json_path]
    if args.time_measurement:
        cmd_args += ["--time-measurement", args.time_measurement]
    return cmd_args


def build_env(args, json_file) -> Tuple[Dict[str, str], List[str]]:
    curr_env: Dict[str, str] = {}

    if args.set_const_tensor_max_size:
        set_const_tensor_max_size(args.set_const_tensor_max_size)

    if os.environ.get("ENABLE_CONSOLE") is None:
        curr_env["ENABLE_CONSOLE"] = "true"

    if os.environ.get("LOG_LEVEL_ALL") is None:
        curr_env["LOG_LEVEL_ALL"] = "4"

    if args.pass_time:
        curr_env["LOG_LEVEL_PASS_MANAGER"] = "0"
        curr_env["ENABLE_CONSOLE"] = "false"

    if args.prof:
        curr_env["HABANA_PROFILE"] = "1"

    if args.time_measurement == "profiler":
        curr_env["HABANA_PROFILE"] = "profile_api_light_recipe_busy"

    test_type = None
    if args.consistency_check:
        test_type = "consistency"
    elif json_file is not None:
        if args.config_compare_values or args.config_compare_file:
            test_type = "config-compare"
        elif args.st_perf:
            test_type = "st_perf"
        elif args.mt_perf:
            test_type = "mt_perf"
        else:
            test_type = "playback"
    else:
        test_type = "recipe"

    cmd_args: List[str] = [
        args.json_tests_bin,
        test_type,
        "--device-type",
        args.chip_type,
    ]

    if test_type == "consistency":
        cmd_args += get_consistency_cmd_args(args, json_file)
    elif test_type == "playback":
        cmd_args += get_playback_cmd_args(args, json_file)
    elif test_type == "st_perf":
        cmd_args += get_perf_cmd_args(args, json_file)
    elif test_type == "mt_perf":
        cmd_args += get_perf_cmd_args(args, json_file)
    elif test_type == "config-compare":
        cmd_args += get_config_compare_cmd_args(args, json_file)
    else:
        cmd_args += get_recipe_cmd_args(args)

    return curr_env, cmd_args


class IrReport():
    def __init__(self, keys = [], results = {}):
        if keys != {} and results != {}:
            self.results = {**dict(zip(["cold_run_" + k for k in keys], map(int, results[0]))),
                            **dict(zip(["warm_run_" + k for k in keys],
                                       [int(double_iter) - int(single_iter)
                                        for single_iter, double_iter in zip(*results[:2])]))}


class TimeReport():
    def __init__(self, data):
        import numpy as np
        from scipy import stats

        self.results = {"min"         : np.min(data) / 1000,
                        "mean"        : int(np.round(np.mean(data))) / 1000,
                        "mean_trim_1" : np.round(stats.trim_mean(data, 0.01)) / 1000,
                        "mean_trim_5" : int(np.round(stats.trim_mean(data, 0.05))) / 1000,
                        "max"         : np.max(data) / 1000}


def run_measure_ir(cmd_args: List[str], curr_env: Dict[str, str], fns_pat: str) -> Dict[str, IrReport]:
    run_iters = [1, 2]

    keys = None
    aggregated_results = collections.defaultdict(list)
    for iter_num, iters in enumerate(run_iters, 1):
        iter_arg = ["--test-iter", str(iters)]
        args = (
            [
                "valgrind",
                "--quiet",
                "--tool=callgrind",
                f"--callgrind-out-file=callgrind.out.iters_{iters}",
                "--",
            ]
            + cmd_args
            + iter_arg
        )

        LOG.info(f"Running command ({iter_num}/{len(run_iters)}):")
        LOG.info(
            " ".join(f"{k}={v}" for k, v in curr_env.items()) + " " + " ".join(args)
        )
        sts = syn.run_external(args, curr_env, LOG)
        if sts:
            ret = IrReport()
            ret.results = {"Instructions_error" : "Could not run valgrind tool"}
            return {'Error': ret}

        new_keys, result = parse_callgrind_file.parse_callgrind_file(f"callgrind.out.iters_{iters}", fns_pat)
        if keys is None:
            keys = new_keys
        else:
            assert keys == new_keys
        for fn, cost in result.items():
            aggregated_results[f"graph_0_{fn}"] += [cost]

    return {k: IrReport(keys, v) for k, v in aggregated_results.items()}


def print_instructions(report : Dict[str, IrReport]):
    # keep includes here to avoid issues with people who do not have them, not needing them by default
    try:
        import tabulate
    except:
        LOG.info("Instruction print NA, please pip install tabulate")
        return 0

    results = copy.deepcopy(report) # avoid adding color esc codes to the original results
    for measure_name, result_ir in results.items():
        for k, v in result_ir.results.items():
            if k.startswith('warm_') and k.endswith('_ir'):
                result_ir.results[k] = colored(v, "white", "on_blue")

    cold_prefix = 'cold_'
    warm_prefix = 'warm_'
    cold_len = len(cold_prefix)
    warm_len = len(warm_prefix)

    headers, tbl = [], []
    for measure_name, result_ir in results.items():
        cold = [[k[cold_len:], v] for k, v in result_ir.results.items() if k.startswith(cold_prefix)]
        warm = [[k[warm_len:], v] for k, v in result_ir.results.items() if k.startswith(warm_prefix)]
        assert len(cold) + len(warm) == len(result_ir.results), 'something other than cold/warm was present'
        if not headers:
            headers = [v[0] for v in cold]
        assert headers == [v[0] for v in cold] == [v[0] for v in warm], \
               f'mismatched cold/warm keys in {measure_name}: [{headers}, {[v[0] for v in cold]}, {[v[0] for v in warm]}]'

        tbl += [[measure_name] + [v[1] for v in cold] + [v[1] for v in warm]]

    headers = ["Measure Name"] + ["Cold " + v for v in headers] + ["Warm " + v for v in headers]

    try:
        import tabulate
        got_tabulate = True
    except:
        got_tabulate = False

    if got_tabulate:
        LOG.info("Final run summary:\n" + tabulate.tabulate(sorted(tbl), headers=headers))
    else:
        LOG.info("Final run summary:\n" + ', '.join(headers) + '\n' + '\n'.join(', '.join(map(str, line)) for line in tbl))

def plot_syntime(plottables: Dict[str, List[int]], category: str, aggregate: List[str]):
    # keep includes here to avoid issues with people who do not have them, not needing them by default
    import matplotlib.pyplot as plt

    data = plottables[category]
    min_val = min(data)
    data_without_high = [v for v in data if v <= 1.5 * min_val]
    indexed_data_without_high = [
        (i, v) for i, v in enumerate(data) if v <= 1.5 * min_val
    ]

    fig, axs = plt.subplots(2, 2, gridspec_kw={"width_ratios": [3, 1]})
    fig.suptitle = category
    plt.figtext(0.5, 0.01, ", ".join(aggregate), ha="center")

    axs[0, 0].set_title("Scatter")
    axs[1, 0].set_title("Scatter - Ignoring Values Above (1.5 * min)")
    axs[0, 1].set_title("Bins")
    axs[1, 1].set_title("Bins - Ignoring Values Above (1.5 * min)")

    runs_per_test = len(data) / len(plottables["iters"])
    if category.endswith("_deviceRuntime") and runs_per_test > 1:
        axs[0, 0].scatter(
            range(len(data)),
            data,
            s=2,
            color=[
                ("red" if i % runs_per_test == 0 else "blue") for i in range(len(data))
            ],
        )
        axs[1, 0].scatter(
            range(len(data_without_high)),
            data_without_high,
            s=2,
            color=[
                ("red" if i % runs_per_test == 0 else "blue")
                for i, v in indexed_data_without_high
            ],
        )
    else:
        axs[0, 0].scatter(range(len(data)), data, s=2)
        axs[1, 0].scatter(range(len(data_without_high)), data_without_high, s=2)

    axs[0, 1].hist(data, max(100, int(len(data) / 100)), orientation="horizontal")
    axs[1, 1].hist(
        data_without_high,
        max(100, int(len(data_without_high) / 100)),
        orientation="horizontal",
    )

    fig.set_size_inches(21, 16, forward=True)
    fig.set_dpi(100)
    png_fn = category + "_plot.png"
    fig.savefig(png_fn)
    plt.close(fig)
    LOG.info(f'Created file "{png_fn}"')


def get_json_data(file_path):
    import json
    with open(file_path) as f:
        return json.load(f)

def run_measure_syntime(cmd_args: List[str], curr_env: Dict[str, str], args) -> dict:
    # keep includes here to avoid issues with people who do not have them, not needing them by default

    log_fn = args.stats_json_path
    if args.stats_json_path is None:
        log_fn = "stats.latest.json"

    cmd_args += [
        "--stats-file",
        log_fn,
        "--synapse-api-funcs " + " ".join(args.measure_syntime) if args.measure_syntime else ""
    ]

    if args.test_iters is None:
        cmd_args += ["--test-iter", "10000"]

    LOG.info("Running command:")
    LOG.info(
        " ".join(f"{k}={v}" for k, v in curr_env.items()) + " " + " ".join(cmd_args)
    )

    sts = syn.run_external(cmd_args, curr_env, LOG)
    if sts:
        return sts

    j = get_json_data(log_fn)

    plottables: Dict[str, List[int]] = {"iters": j["iters"]}
    for g, vals in j["graphs"].items():
        for stat in vals:
            if stat == "workspaceSize":
                continue
            if stat == "deviceRuntime":
                plottables[f"graph_{g}_deviceRuntime"] = list(
                    itertools.chain.from_iterable(vals.get("deviceRuntime", []))
                )
            else:
                plottables[f"graph_{g}_{stat}"] = vals.get(stat, [])
    return plottables


def print_times(plottables):
    import numpy as np
    from scipy import stats

    if os.getenv('EAGER_NOPLOT', '').lower() in ['t', 'true', 'y', 'yes', '1']:
        LOG.info("EAGER_NOPLOT is set, skipping plots")
        do_plot = False
    else:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            do_plot = True
        except:
            LOG.info("pip install matplotlib to enable measure_syntime plot dumps to png")
            do_plot = False

    for category, data in plottables.items():
        if len(data) == 0:
            LOG.info(f'No data provided for "{category}"!')
            continue

        aggregate = [
            f"Min: {np.min(data)}ns",
            f"Mean: {int(np.round(np.mean(data)))}ns",
            f"MeanTrim1%: {int(np.round(stats.trim_mean(data, 0.01)))}ns",
            f"MeanTrim5%: {int(np.round(stats.trim_mean(data, 0.05)))}ns",
            f"Max: {np.max(data)}ns",
        ]

        agg_with_color = aggregate[:]
        agg_with_color[0] = colored(agg_with_color[0], "white", "on_blue")
        LOG.info(f"Summary for \"{category}\": {', '.join(agg_with_color)}")

        if do_plot:
            plot_syntime(plottables, category, aggregate)


def print_measured_data(results):
    for file_name, measures in results.items():
        print(colored("Results for {}".format(file_name), "white", "on_magenta"))
        for key, report in sorted(measures.items()):
            if key == "instructions":
                print_instructions(report)
            else:
                print_times(report)


def get_properties_dict(results):
    return {f'properties.{k}': v for k, v in results.items()}


def generate_eager_xml_report(results, run_times, append=False):
    if append:
        try:
            xmltree = ET.parse(args.generate_eager_xml_report).getroot()
        except:
            xmltree = syn.generate_xml("eager_test_results")
    else:
        xmltree = syn.generate_xml("eager_test_results")
    for file_name, measures in results.items():
        for metricName in measures["instructions"]:
            ir_report = measures["instructions"][metricName]
            if metricName.endswith('_synGraphCompile') and metricName not in measures["time"]:
                metricName = metricName.replace('_synGraphCompile', '_compileTime')
            time_report = TimeReport(measures["time"][metricName]).results if metricName in measures["time"] else {}
            feilds_dict = {"name" : file_name.split('/')[-1].split('.')[0] + "_" + metricName.split('_')[-1]}
            feilds_dict = {
                **feilds_dict,
                **run_times[file_name],
                **get_properties_dict(ir_report.results),
                **get_properties_dict(time_report),
            }
            syn.add_testcase_to_xml(xmltree, feilds_dict)

    syn.save_xml_file(xmltree, args.generate_eager_xml_report)


def multi_cards_run(cmd: Tuple[List[str],Dict[str, str]]) -> int:
    return syn.run_external(cmd[0], cmd[1], LOG)


def run_normal(cmd_args: List[str], curr_env: Dict[str, str], args) -> int:
    sts = 0
    env_str = " ".join(f"{k}={v}" for k, v in curr_env.items())
    if args.cards > 1:
        commands = []
        LOG.info("Running commands:")
        for i in range(args.cards):
            curr_cmd = cmd_args + ["--groups", str(i)]
            env = curr_env.copy()
            env["OMPI_COMM_WORLD_RANK"] = str(i)
            LOG.info(f'{env_str} {" ".join(curr_cmd)}')
            LOG.info('vscode args: "args": ["%s"]', '", "'.join(curr_cmd[1:]))
            commands.append((curr_cmd, env))
        pool = Pool(len(commands))
        for ret in pool.imap(multi_cards_run, commands):
            LOG.info(f"Command finished with status: {ret}")
            sts |= ret
    else:
        LOG.info("Running command:")
        LOG.info(f'{env_str} {" ".join(cmd_args)}')
        LOG.info('vscode args: "args": ["%s"]', '", "'.join(cmd_args[1:]))
        sts = syn.run_external(cmd_args, curr_env, LOG)
    return sts


def get_graph_bundles_json(json_files, args):
    # check that the user's arguments fits to bundle reproduction mode
    if args.graph_indices is None:
        args.graph_indices = [0]
    if len(args.graph_indices) != 1:
        LOG.error(f"Bundle reproduction mode works with one graph only")
        exit(BUNDLE_REPRODUCTION_MODE_FAILURE)
    if (len(json_files) != 1):
        LOG.error(f"In bundle reproduction mode only one json is allowed")
        exit(BUNDLE_REPRODUCTION_MODE_FAILURE)

    # use graph_editor's bundle reproduction tool api
    json_file, validation_status = graph_editor.get_graph_bundles_json(json_files[0], args.bundles, args.graph_indices[0], args.chip_type)

    if json_file is None or validation_status != 0:
        exit(BUNDLE_REPRODUCTION_MODE_FAILURE)
    args.graph_indices = None
    LOG.info(f"Continue run_from_json run with the bundles pre graphs")
    return [json_file]

def get_graph_name_to_graph_index(json_files) -> Dict[str, int]:
    d = {}
    for json_file in json_files:
        json_data = get_json_data(json_file)
        for graph in json_data.get("graphs"):
            if graph.get("nodes") == None or len(graph.get("nodes")) == 0:
                continue
            d[graph.get("name")] = graph.get("nodes")[0].get("graph_index")
    return d


def generate_pass_time_csv_file(graph_name_to_graph_index: Dict[str, int]):
    csv_file_path = compilation_time_analyzer.main(
        compilation_time_analyzer.parse_args([])
    )

    # add graph index column to the csv
    import csv

    with open(csv_file_path, "r") as csvinput:
        reader = csv.reader(csvinput, lineterminator="\n")

        COL_GRAPH_INDEX = "Graph Index"
        edited_rows = []
        row = next(reader)
        row.insert(
            compilation_time_analyzer.PASS_TIME_TABLE_HEADER.index(
                compilation_time_analyzer.COL_RECIPE_NAME
            )
            + 1,
            COL_GRAPH_INDEX,
        )
        edited_rows.append(row)

        col_index_of_graph_index = row.index(COL_GRAPH_INDEX)
        col_index_of_recipe_name = row.index(compilation_time_analyzer.COL_RECIPE_NAME)

        for row in reader:
            row.insert(
                col_index_of_graph_index,
                graph_name_to_graph_index[row[col_index_of_recipe_name]],
            )
            edited_rows.append(row)

    compilation_time_analyzer.save_list_to_csv_file(edited_rows, csv_file_path)

def main(args):
    curr_env: Dict[str, str]
    cmd_args: List[str]
    results = collections.defaultdict(dict)
    if args.json_files:
        json_files = args.json_files
    else:
        models = get_all_models(MODELS_TESTS_PATH, True)
        json_files = models.values() if args.all_models else [models.get(args.model)]
    # bundle reproduction mode:
    if args.bundles:
        json_files = get_graph_bundles_json(json_files, args)
    run_times = {}

    if args.pass_time:
        # Empty the relevant log files
        for log_file in syn.get_files(os.getenv("HABANA_LOGS"), compilation_time_analyzer.LOG_FILES_POSTFIX, compilation_time_analyzer.LOG_FILES_PREFIX):
            os.remove(log_file)

    sts = 0
    for json_file in json_files:
        curr_env, cmd_args = build_env(args, json_file)
        json_dict = results[json_file]
        start = time.time()
        if args.measure_compile_ir or args.generate_eager_xml_report:
            if args.st_perf and args.generate_eager_xml_report:
                synapse_function_names = 'syn[A-Z][a-zA-Z0-9_]*' # get_kibana_eager_api_funcs()
            else:
                synapse_function_names = args.measure_compile_ir or 'synGraphCompile'
            json_dict["instructions"] = run_measure_ir(cmd_args, curr_env, synapse_function_names)

        if args.measure_syntime is not None or args.generate_eager_xml_report:
            json_dict["time"] = run_measure_syntime(cmd_args, curr_env, args)
            if type(json_dict["time"]) == int:
                return json_dict["time"]

        if not json_dict:
            sts |= run_normal(cmd_args, curr_env, args)
            if sts != 0 and not args.keep_going:
                break

        run_times[json_file] = {"time" : time.time() - start}

    if args.generate_eager_xml_report is not None or args.measure_compile_ir is not None or args.measure_syntime is not None:
        print_measured_data(results)

    if args.generate_eager_xml_report:
        generate_eager_xml_report(results, run_times, args.append)

    if args.pass_time:
        # Expecting that there are no graphs with the same names
        generate_pass_time_csv_file(get_graph_name_to_graph_index(json_files))

    return sts


class SynHelpFormatter(argparse.RawDescriptionHelpFormatter):
    def _split_lines(self, text, width):
        if text.startswith("split_lines"):
            return sorted(text.replace("split_lines", "").split(" "))
        return textwrap.wrap(text)


def get_kibana_eager_api_funcs():
    return ["synGraphCreateEager",
            "synNodeCreateWithId",
            "synTensorHandleCreate",
            "synSectionCreate",
            "synGraphCompile",
            "synTensorSetGeometry",
            "synTensorSetDeviceFullLayout",
            "synTensorSetHostPtr",
            "synTensorSetPermutation",
            "synTensorSetExternal",
            "synTensorSetAllowPermutation",
            "synTensorAssignToSection",
            "synSectionSetPersistent",
            "synSectionSetRMW",
            "synNodeSetRoundingMode",
            "synNodeDependencySet",
            "synGraphDestroy",
            "synTensorDestroy",
            "synSectionDestroy",
            "synGraphDuplicate",
            "synGraphInferShapes"]

def get_supported_api_funcs():
    return ["synGraphCreate",
            "synGraphCreateEager",
            "synNodeCreateWithId",
            "synTensorHandleCreate",
            "synSectionCreate",
            "synGraphCompile",
            "synTensorSetGeometry",
            "synTensorSetDeviceFullLayout",
            "synTensorSetHostPtr",
            "synTensorSetPermutation",
            "synTensorSetExternal",
            "synTensorSetAllowPermutation",
            "synTensorAssignToSection",
            "synSectionSetPersistent",
            "synSectionSetRMW",
            "synNodeSetRoundingMode",
            "synNodeDependencySet",
            "synGraphDestroy",
            "synTensorDestroy",
            "synSectionDestroy",
            "synGraphDuplicate",
            "synGraphInferShapes",
            "synWorkspaceGetSize",
            "synTensorRetrieveLaunchInfoByIdExt",
            "synLaunchWithExternalEventsExt"]


def validateArgs(parser, args):
    if args.chip_type is None:
        parser.error('Argument --chip_type is required')
    if args.append and args.generate_eager_xml_report is None:
        parser.error('Argument "--append" used without "--generate_eager_xml_report"')
    if args.eager:
        parser.error('Argument "--eager" is deprecated, please use: "--compilation_mode eager"')
    if os.getcwd().startswith(os.path.join(os.environ['HOME'], "qnpu")) and "QNPU_PATH" not in os.environ and args.ignore_errors is False:
        parser.error('Running within qnpu dir without activating it. Use --ignore_errors to skip this check')
    if args.time_measurement == "profiler" and os.environ.get("HABANA_PROFILE", "0").lower() in ["true", "1"]:
        parser.error("Argument --time_measurement shouldn't be set to profiler if profiler is enabled, run with --time_measurement none")


def get_dependency_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "-r", "--release", action="store_true", help="Run release binary"
    )
    parser.add_argument(
        "--skip_models_download", action="store_true", help="Avoid clone/pull the models repo"
    )
    parser.add_argument(
        "-c",
        "--chip_type",
        choices=list(syn.CHIP_TYPES.keys()),
        help="Select chip type (default: gaudi)",
    )
    parser.add_argument(
        "--sanitizer", action="store_true", help="Run json_tests sanitizer build"
    )
    parser.add_argument(
        "--prof",
        action="store_true",
        help="Capture synapse profiler trace",
    )
    args, _ = parser.parse_known_args()
    return parser, args


def parse_args():
    dep_parser, dep_args = get_dependency_args()
    parser = argparse.ArgumentParser(parents=[dep_parser], formatter_class=SynHelpFormatter)
    models = list(get_supported_models(MODELS_TESTS_PATH, None, dep_args.skip_models_download))
    supported_api_funcs = get_supported_api_funcs()
    executer = parser.add_mutually_exclusive_group()
    time_measurement = ("events" if dep_args.chip_type == "gaudi" else "profiler")
    if os.environ.get("HABANA_PROFILE", "0").lower() in ["true", "1"] or dep_args.prof:
        time_measurement = "none"
    executer.add_argument("-j", "--json_files", help="Json file to run", nargs="+")
    executer.add_argument(
        "-m",
        "--model",
        choices=models,
        help=f'split_lines{", ".join(models)}' if models else "No models found",
        metavar="",
    )
    parser.add_argument(
        "--all_models",
        action="store_true",
        help="Run all availabe models",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run graph after compilation",
    )
    parser.add_argument(
        "-d",
        "--data_file",
        help="Data file for input data and results comparison",
    )
    parser.add_argument(
        "--const", action="store_true", help="Skip persistent tensors, use only const tensors data",
    )
    parser.add_argument(
        "--comp_config_file",
        help="Data comparator config file",
        default=DEFAULT_DATA_COMPARATOR_CONFIG_FILE
    )
    parser.add_argument(
        "--exclude_graphs",
        action="store_true",
        help="-g/--graph_indices should be excluded",
    )
    filter_group = parser.add_mutually_exclusive_group()
    filter_group.add_argument(
        "-g",
        "--graph_indices",
        type=int,
        nargs="+",
        help="Run only the specified graph index",
    )
    filter_group.add_argument(
        "--groups",
        type=int,
        nargs="+",
        help="Use graphs from the specified groups. Default: use all groups",
    )
    parser.add_argument(
        "-i",
        "--iterations",
        type=int,
        help="Number of graph run iterations",
        default=1,
    )
    parser.add_argument(
        "--test_iters",
        help="Number of whole test (including compilation + iteration number of runs) repeats",
        type=int,
    )
    parser.add_argument(
        "--iterations_filter",
        type=int,
        nargs="+",
        help="Run only specific iterations, if not set, run all iterations",
    )
    parser.add_argument(
        "--compilation_mode",
        type=str,
        choices=['graph', 'eager', 'from_recording'],
        default='from_recodring',
        help="Force a compilation mode on all graphs in json",
    )
    parser.add_argument(
        "--eager",
        action="store_true",
        help="Use eager graph mode",
    )
    executer.add_argument(
        "--recipe",
        help="Recipe file to run",
    )
    parser.add_argument(
        "--serialize_recipe",
        help="Folder path for serialized recipes",
    )
    parser.add_argument(
        "--stats_json_path",
        help="Log per-{iter,compilation,run} times to a json file",
    )
    syn_build = (
        os.getenv("SYNAPSE_RELEASE_BUILD")
        if dep_args.release
        else os.getenv("SYNAPSE_DEBUG_SANITIZER_BUILD")
        if dep_args.sanitizer
        else os.getenv("SYNAPSE_DEBUG_BUILD")
    )
    parser.add_argument(
        "--json_tests_bin",
        help="Full path to json_tests binary",
        default=os.path.join(syn_build, "bin", "json_tests") if syn_build else None,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--measure_compile_ir",
        nargs="?",
        const="synGraphCompile",
        # choices=supported_api_funcs,
        metavar="",
        help="Use callgrind to extract synGraphCompile Ir and other stats",
    )
    mode.add_argument(
        "--measure_syntime",
        nargs="*",
        choices=supported_api_funcs,
        metavar="",
        help="collect per-{iter,compilation,run} timmings into timmings.json.latest and summarize them.",
    )
    mode.add_argument(
        "--generate_eager_xml_report",
        nargs='?',
        const="eager_report.xml",
        help="Generate eager XML report for kibana view",
    )
    parser.add_argument(
        "--append",
        help="Update the xml with results; Use for incremental generation of XML for Kibana",
        action="store_true"
    )
    parser.add_argument(
        "--quiet",
        help="Suppress prints and only print progress every approx 0.05 of the iterations. Use together with high --test_iters counts.",
        action="store_true",
    )
    mode.add_argument(
        "--synthetic_data",
        action="store_true",
        help="Run with synthetic tensors data",
    )
    parser.add_argument(
        "--consistency_check",
        action="store_true",
        help="Compile multiple times and check recipe compilation consistency",
    )
    config_compare_modes = parser.add_mutually_exclusive_group()
    config_compare_modes.add_argument(
        "--config_compare_values",
        nargs="+",
        help=("Allows accuracy compare between runs of the same graphs with different configurations. "
        "Requires one or more configuration pairs of this type: <config_name, config_value>, each pair is set on its own compilation and run process. In case only one pair is specified then it is compared to the default configurations.")
    )
    config_compare_modes.add_argument(
        "--config_compare_file",
        help=("Allows accuracy compare between runs of the same graphs with different configurations. "
            "Requires a json file that contains test runs configurations. The file format is as follows: list of runs configurations where each run configurations are written as dictionary such that key=<config_name> and value=<config_value>")
    )
    parser.add_argument(
        "--st_perf",
        action="store_true",
        help='''Works simillarly to playback mode with more accurate time measurments and ability to measure
                each of the Synapse available API calls using both measure_syntime and measure_compile_ir.
                Allowed arguments for measure_syntime are either of the supported API calls or no args to record everything.
                Allowed arguments for measure_compile_ir are either of the supported API calls or no args to profile graph compilation.
                Currently supported API calls: ''' + " ,".join(supported_api_funcs),
    )
    parser.add_argument(
        "--mt_perf",
        action="store_true",
        help='''This mode aims to imitate the Pytorch Eager mode operation, where we utilize a pipeline composed of four steps:
                graph construction, compilation, execution, graph and recipe resource cleanup.
                Where each such phase utilizes dedicated threads from a thread pool (currently a single thread per phase).
                The purpose of this mode is to allow us to find bubbles and concentrate our efforts on the graphs and phases
                that matter most.'''
    )
    parser.add_argument(
        "--reset_device",
        action="store_true",
        help="Release and acquire the device before each graph run",
    )
    parser.add_argument(
        "--keep_going",
        action="store_true",
        help="When running multiple graphs from jsons, continue on compilation error and report failed graphs",
    )
    parser.add_argument(
        "--ignore_errors",
        action="store_true",
        help="Continue on env configuration errors (such as running from qnpu dir w/o activating it)",
    )
    parser.add_argument(
        "--set_const_tensor_max_size",
        type=int,
        help="Set const tensor max size, set 0 to use synapse defaults",
        default=0x1000000,
    )
    parser.add_argument(
        "--time_measurement",
        choices=["none", "events", "profiler"],
        default=time_measurement,
        help="set time measurement mechanism",
    )
    parser.add_argument(
        "--bundles",
        type=int,
        nargs="+",
        help='''Generates pre graphs of selected bundles and operates on them.
                Release binaries are being used in the creation of the bundles pre graphs,
                the later json_runner operations use binaries depending on the addition of -r flag.''',
    )
    parser.add_argument(
        "--cards",
        type=int,
        default=1,
        help='Run on multiple habana cards, the executed json number of groups must be equel to the number of cards',
    )
    parser.add_argument(
        "--pass_time",
        action="store_true",
        help='''Generates csv of passes working times. With this mode HABANA_LOGS directory will be edited since the passes working time being written to the log during compilation'''
    )
    args = parser.parse_args()
    validateArgs(parser, args)

    os.environ[
        "LD_LIBRARY_PATH"
    ] = f'{syn_build}/lib:{os.environ.get("LD_LIBRARY_PATH", "")}'

    return args


if __name__ == "__main__":
    args = parse_args()
    exit(main(args))
