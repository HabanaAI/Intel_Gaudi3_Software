import copy
import errno
import json
import multiprocessing
import os
import os.path
import shutil
import time
from typing import Dict, List

import mpm_report
import mpm_runner
import mpm_utils
from mpm_logger import LOG
from mpm_metrics import MpmMetrics
from mpm_model_list import Filter, ModelMetadata, ModelsDeviceInfo, ModelsList
from mpm_run_time_metric import RunTimeMetric
from mpm_runner import Dispatcher
from mpm_types import (ERROR_CODES, ComConfig, ComObj, GlobalConfig,
                       GlobalConstants, LaunchConfig, MetricType, ModelConfig, ModelInfo,
                       MpmStatusLists, ModelStats, TestConfig)

REPOS = ["synapse", "mme", "tpc_kernels", "tpc_fuser"]

action_types = ["compile", "run", "report"]

file_format = ["csv", "json", "xml"]  # first is default

REPO_TO_LIB_PATH = (
    None
    if os.environ.get("SYNAPSE_RELEASE_BUILD") is None
    or os.environ.get("TPC_KERNELS_RELEASE_BUILD") is None
    or os.environ.get("TPC_FUSER_RELEASE_BUILD") is None
    else {
        "synapse": os.path.join(
            os.environ.get("SYNAPSE_RELEASE_BUILD"), "lib", "libSynapse.so"
        ),
        "mme": os.path.join(
            os.environ.get("SYNAPSE_RELEASE_BUILD"), "lib", "libSynapse.so"
        ),
        "tpc_kernels": os.path.join(
            os.environ.get("TPC_KERNELS_RELEASE_BUILD"), "lib", "libtpc_kernels.so"
        ),
        "tpc_fuser": os.path.join(
            os.environ.get("TPC_FUSER_RELEASE_BUILD"), "lib", "libTPCFuser.so"
        ),
    }
)


def log_models_info(models_folder, models_file, models_list):
    LOG.info(f"models folder: {models_folder}")
    LOG.info(f"models file: {models_file}")
    LOG.info(f"default models list: {models_list}")


def get_timestamp():
    return time.mktime(time.gmtime())


def convert_to_ms(time_units, time):
    if time_units == "ns":
        return time / 1000000.0
    if time_units == "us":
        return time / 1000.0


def set_const_tensor_max_size(max_size):
    LOG.info(f"set const tensor max size to: {max_size}")
    if os.environ.get("ENABLE_EXPERIMENTAL_FLAGS") is None:
        os.environ["ENABLE_EXPERIMENTAL_FLAGS"] = "true"
    if os.environ.get("MAX_CONST_TENSOR_SIZE_BYTES") is None:
        os.environ["MAX_CONST_TENSOR_SIZE_BYTES"] = str(max_size)
    if os.environ.get("HBM_GLOBAL_MEM_SIZE_MEGAS") is None:
        os.environ["HBM_GLOBAL_MEM_SIZE_MEGAS"] = "256"


def set_env():
    if os.environ.get("ENABLE_CONSOLE") is None:
        os.environ["ENABLE_CONSOLE"] = "true"
    if os.environ.get("LOG_LEVEL_ALL") is None:
        os.environ["LOG_LEVEL_ALL"] = "4"


def create_folder(folder):
    try:
        os.makedirs(folder, exist_ok=True)
    except OSError as exc:
        if not (exc.errno == errno.EEXIST and os.path.isdir(folder)):
            raise RuntimeError("failed to create results folder")


def get_device_default_ref_results(device):
    ret = f"/software/ci/models_tests/promotion_master_{device}.json"
    if not os.path.exists(ret):
        raise RuntimeError(f"default_ref_results doesn't exists, path: {ret}")
    return ret


def checkout_models_revision(models_revision: str, global_constants: GlobalConstants):
    LOG.debug(f"set models folder revision to: {models_revision}")
    if os.path.exists(f"{global_constants.LOCAL_MODELS_FOLDER}/.git"):
        mpm_utils.git_pull(global_constants.LOCAL_MODELS_FOLDER, True)
    else:
        LOG.info("clone models folder")
        os.makedirs(global_constants.LOCAL_MODELS_FOLDER, exist_ok=True)
        mpm_utils.git_clone(
            global_constants.MODELS_REPO, global_constants.LOCAL_MODELS_FOLDER
        )
    mpm_utils.checkout(models_revision, global_constants.LOCAL_MODELS_FOLDER)


def get_models_names(folder: str) -> List[str]:
    ret = []
    for tf in mpm_utils.get_files(folder, ".json"):
        ret.append(os.path.basename(tf).replace(".json", ""))
    return ret


def get_jobs(models_file: str):
    if models_file is None:
        return []
    with open(models_file) as f:
        ml = ModelsList.deserialize(json.load(f))
    return list(ml.get_jobs())


def get_supported_models(
    folder: str, models_file: str, device: str = None, jobs: List[str] = None
) -> List[str]:
    models = get_models_names(folder)
    if models_file is None:
        return models
    with open(models_file) as f:
        ml = ModelsList.deserialize(json.load(f))
    all_models = []
    if jobs:
        for j in jobs:
            all_models += ml.get_names(Filter(device, j, True))
        all_models = list(set(all_models))
    else:
        all_models = ml.get_names(Filter(device, None, True))
    return [m for m in models if m in all_models]


def validate_requirements(global_config: GlobalConfig):
    LOG.info("check required tools")
    models_tests_bin = global_config.models_tests_binary_path
    if not models_tests_bin or not os.path.isfile(models_tests_bin):
        raise RuntimeError(
            f"models_tests executable path {'does not exists' if models_tests_bin else 'was not defined'}, path: {models_tests_bin}"
        )
    for tool in mpm_utils.get_tools().values():
        mpm_utils.Executer(mpm_utils.get_tools().get("WHICH"), tool, True)


def is_report_only(actions):
    return actions is None or not ("compile" in actions or "run" in actions)


def compile_request(actions):
    return "compile" in actions


def run_request(actions):
    return "run" in actions


def get_com_files(folder):
    return mpm_utils.get_files(folder, ".com.json")


def split_to_graphs(com_json):
    graphs = []
    com = ComObj.deserialize(com_json)
    comp = com_json.get("compile")
    comp_graphs = comp.get("graphs") if comp else None
    if comp_graphs:
        graph_count = len(comp_graphs)
        for i in range(graph_count):
            g = copy.deepcopy(com)
            g.config.model = (
                f"{g.config.model}-{g.compile.graphs.get(str(i)).name}".replace(
                    "/", "_"
                )
            )
            create_folder(
                os.path.join(g.config.work_folder, g.config.name, g.config.model)
            )
            if g.compile and g.compile.graphs:
                g.compile.graphs = {"0": g.compile.graphs.get(str(i))}
            if g.run and g.run.graphs:
                g.run.graphs = {"0": g.run.graphs.get(str(i))}
            graphs.append(g)
    else:  # in case there are no graphs, use the original com
        graphs.append(com)
    return graphs


def get_precompiled_tests(coms: List[ComObj]) -> set:
    ret = set()
    for com in coms:
        if com and com.config and com.config.recipe_folder:
            ret.add(com.config.recipe_folder)
    return ret


def collect_curr_results(global_config: GlobalConfig) -> List[ComObj]:
    LOG.debug("collect current results")
    all_coms = []
    results = get_com_files(global_config.work_folder)
    status = []
    for res in results:
        with open(res) as i_file:
            result_data = json.load(i_file)
        coms = (
            split_to_graphs(result_data)
            if global_config.graphs
            else [ComObj.deserialize(result_data)]
        )
        for com in coms:
            com.under_test = True
            all_coms.append(com)
            status.append(com)
    pre_compiled_tests = get_precompiled_tests(all_coms)
    ret = []
    for com in all_coms:
        if (
            com.config.name in pre_compiled_tests
            and not global_config.show_precompiled
            and com.config.recipe_folder is None
        ):
            continue
        ret.append(com)
    return ret


def collect_ref_results(global_config: GlobalConfig) -> List[ComObj]:
    LOG.debug("collect ref results")
    ret = []
    if global_config.ref_results:
        for res_file in global_config.ref_results:
            with open(res_file) as i_file:
                result_data = json.load(i_file)
            for res in result_data:
                coms = (
                    split_to_graphs(res)
                    if global_config.graphs
                    else [ComObj.deserialize(res)]
                )
                for com in coms:
                    if com.use_for_ref and com.config is not None:
                        com.under_test = False
                        ret.append(com)
    return ret


def get_best_run_time_results(coms: List[ComObj]) -> Dict[str, ComObj]:
    best_result: Dict[str, ComObj] = {}
    for com in coms:
        if com.report is None or com.report.avg_run_time is None:
            continue
        if com.report.model not in best_result:
            best_result[com.report.model] = com
        min_run_time = best_result.get(com.report.model).report.avg_run_time
        if com.report.avg_run_time is not None and (
            min_run_time is None or com.report.avg_run_time < min_run_time
        ):
            best_result[com.report.model] = com
    return best_result


def update_report_status(results_data: List[ComObj], errors_file: str):
    LOG.debug("update report status")
    for com in results_data:
        lines: List[str] = []
        assert com.report is not None
        line_space = os.linesep + '\t\t'
        errors = [f"\t\t{line_space.join(e.messages)}" for e in com.status.values() if e.status == ERROR_CODES.ERROR]
        warnings = [f"\t\t{line_space.join(w.messages)}" for w in com.status.values() if w.status == ERROR_CODES.WARNING]
        if errors:
            com.report.status = f'"fail, errors file: {errors_file}"'
            lines += [f"\terrors:"] + errors
        if warnings:
            if not errors:
                com.report.status = f'"pass with warnings, errors file: {errors_file}"'
            lines += [f"\twarnings:"] + warnings
        if not lines:
            com.report.status = "pass"
        else:
            with open(errors_file, "a") as f:
                f.write(
                    f"test name: {com.report.name}, model: {com.report.model}{os.linesep}"
                )
                f.write(f"{os.linesep}".join(lines) + os.linesep)
                f.write(os.linesep)


def show_results(global_config: GlobalConfig, status: MpmStatusLists):
    print()
    LOG.info(f"Status:")
    if status.errors:
        LOG.info(f"Failed tests ({len(status.errors)}/{status.count}):")
        for t in status.errors:
            LOG.error(f"error details:{os.linesep}{os.linesep.join(t.messages)}")
            print()
            LOG.error(f"repro instructions:{os.linesep}{os.linesep.join(t.repro)}")
            print()
    if status.warnings:
        LOG.info(f"Passed with warning ({len(status.warnings)}/{status.count}):")
        for t in status.warnings:
            LOG.warning(f"warning details:{os.linesep}{os.linesep.join(t.messages)}")
            print()
    LOG.info(
        f"Run summary: passed: ({len(status.success)}/{status.count}), passed with warning: ({len(status.warnings)}/{status.count}), failed: ({len(status.errors)}/{status.count}), test case count: {len(global_config.names)}, models count: {len(global_config.models)}, metrics count: {len(global_config.metrics.metrics)}"
    )


def get_status(status_lists: MpmStatusLists) -> int:
    if status_lists.errors:
        return ERROR_CODES.ERROR.value
    return ERROR_CODES.SUCCESS.value


def dump_coms_data(coms: List[ComObj], file_path, keep_report=False):
    LOG.debug(f"dump coms to: {os.path.abspath(file_path)}")
    lines = []
    for c in coms:
        if keep_report:
            lines.append(c.serialize())
        else:
            com_copy = ComObj()
            com_copy.config = c.config
            com_copy.compile = c.compile
            com_copy.run = c.run
            com_copy.use_for_ref = c.use_for_ref
            lines.append(com_copy.serialize())
    folder = os.path.dirname(file_path)
    if folder != "" and not os.path.exists(folder):
        os.makedirs(os.path.dirname(file_path))
    with open(file_path, "w") as o_file:
        json.dump(lines, o_file, sort_keys=True)


def update_stats(com: ComObj, model_list: ModelsList):
    model_metadata: ModelMetadata = model_list.models.setdefault(com.config.model, ModelMetadata())
    models_device_info: ModelsDeviceInfo = model_metadata.devices.setdefault(com.config.device, ModelsDeviceInfo())
    if com.compile and com.compile.status == "pass":
        models_device_info.stats.compile = com.compile.stats
    if com.run and com.run.status == "pass":
        models_device_info.stats.run = com.run.stats


def dump_models_stats(coms: List[ComObj], global_config: GlobalConfig):
    org_stats = ModelsList()
    combined_models_list = ModelsList()
    tests: Dict[str, ModelsList] = {}
    if global_config.models_file and os.path.exists(global_config.models_file):
        with open(global_config.models_file) as f:
            org_stats = ModelsList.deserialize(json.load(f))
            combined_models_list = copy.deepcopy(org_stats)
    for com in coms:
        if com is None or com.config is None or not com.under_test:
            continue
        curr_file_path = os.path.join(com.config.work_folder, com.config.name, f"{com.config.name}_models_list.json")
        curr_model_list: ModelsList = tests.setdefault(curr_file_path, copy.deepcopy(org_stats))
        update_stats(com, curr_model_list)
        update_stats(com, combined_models_list)
    for k, v in tests.items():
        with open(k, "w") as f:
            json.dump(v.serialize(), f, sort_keys=True, indent=4)
    combined_file_path = os.path.join(global_config.work_folder, "combined_models_list.json")
    with open(combined_file_path, "w") as f:
        json.dump(combined_models_list.serialize(), f, sort_keys=True, indent=4)


def dump_curr_tests(coms: List[ComObj], file_path, keep_report=False):
    dump_coms_data([c for c in coms if c.under_test], file_path, keep_report)


def dump_curr_status(coms: List[ComObj], file_path, keep_report=False):
    best_results = get_best_run_time_results(coms)
    for com in coms:
        if (
            com.report
            and com.under_test
            and MetricType.RUN_TIME in com.status
            and com.status.get(MetricType.RUN_TIME).status == ERROR_CODES.ERROR
        ):
            best_results[com.report.model] = com
    dump_coms_data(list(best_results.values()), file_path, keep_report)


def update_com(
    com: ComObj,
    run_config: GlobalConfig,
    test_config: TestConfig,
    model_info: ModelInfo,
):
    assert com.config is not None
    assert com.config.model_data_file == model_info.data_file_path
    com.config.actions = list(set(com.config.actions + run_config.actions))
    com.config.work_folder = os.path.abspath(run_config.work_folder)
    com.config.name = test_config.name
    com.config.data_comparator_config_file = run_config.data_comparator_config_file
    com.config.iterations = run_config.iterations
    com.config.release_device = run_config.release_device
    com.config.time_measurement = run_config.time_measurement
    com.config.recipe_folder = (
        os.path.join(run_config.precompiled_test, model_info.name, "recipes")
        if run_config.precompiled_test
        else None
    )
    model_folder = os.path.join(test_config.folder_path, model_info.name)
    os.makedirs(model_folder, exist_ok=True)


def generate_com(
    run_config: GlobalConfig,
    test_config: TestConfig,
    model_info: ModelInfo,
    timestamp,
):
    com = ComObj()
    com.use_for_ref = True
    com.config = ComConfig()
    com.config.work_folder = os.path.abspath(run_config.work_folder)
    com.config.actions = run_config.actions
    com.config.name = test_config.name
    com.config.git_hash = test_config.commit
    com.config.model = model_info.name
    com.config.model_file = model_info.file_path
    com.config.model_data_file = model_info.data_file_path
    com.config.data_comparator_config_file = run_config.data_comparator_config_file
    com.config.iterations = run_config.iterations
    com.config.timestamp = timestamp
    com.config.device = run_config.chip_type
    com.config.repo = run_config.repo
    com.config.release_device = run_config.release_device
    com.config.time_measurement = run_config.time_measurement
    com.config.groups = [0]
    com.config.compilation_mode = run_config.compilation_mode
    com.config.compile_consistency_iters = run_config.compile_consistency_iters
    com.config.models_revision = run_config.models_revision
    recipes_folder = os.path.join(test_config.folder_path, model_info.name, "recipes")
    os.makedirs(recipes_folder, exist_ok=True)
    if com.config.models_revision is None:
        com.config.models_revision = "unknown"
    return com


def get_com(
    global_config: GlobalConfig,
    test_config: TestConfig,
    model_info: ModelInfo,
    timestamp: int,
) -> ComObj:
    test_name = (
        test_config.name
        if global_config.precompiled_test is None
        else global_config.precompiled_test
    )
    com_file_path = os.path.join(
        global_config.work_folder,
        test_name,
        model_info.name,
        f"{model_info.name}.com.json",
    )
    if os.path.exists(com_file_path):
        with open(com_file_path) as f:
            config_data = json.load(f)
        com = ComObj.deserialize(config_data)
        update_com(com, global_config, test_config, model_info)
    else:
        com = generate_com(global_config, test_config, model_info, timestamp)
    com_file_path = os.path.join(
        test_config.folder_path, model_info.name, f"{model_info.name}.com.json"
    )
    with open(com_file_path, "w") as f:
        json.dump(com.serialize(), f, indent=4, sort_keys=True)
    return com


def create_tests_configs_from_config_compare_file(global_config: GlobalConfig, commit: str) -> List[TestConfig]:
    with open(global_config.config_compare_file) as f:
        config_compare_data = json.load(f)
    ret = []
    i = 0
    for conf in config_compare_data:
        str_conf: Dict[str, str] = {}
        name = str()
        delimiter = ""
        for k, v in conf.items():
            name += f'{delimiter}{k}={v}'
            str_conf[k] = str(v)
            delimiter = "_"
        ret.append(TestConfig(name, commit, global_config, False, str_conf))
        i += 1
    if len(ret) == 1:
        ret.insert(0, TestConfig(f"base", commit, global_config, False, {}))
    return ret


def create_tests_configs_from_config_compare_values(global_config: GlobalConfig, commit: str) -> List[TestConfig]:
    ret = [TestConfig(f"base", commit, global_config, False, {})]
    length = int(len(global_config.config_compare_values) / 2)
    for i in range(length):
        env: Dict[str, str] = {}
        index = i * 2
        config_name = global_config.config_compare_values[index]
        config_value = str(global_config.config_compare_values[index + 1])
        env[config_name] = config_value
        ret.append(TestConfig(f"{config_name}={config_value}", commit, global_config, False, env))
    return ret


def create_tests_configs(global_config: GlobalConfig) -> List[TestConfig]:
    if global_config.git_revs:
        ret = []
        for n in global_config.git_revs:
            git_hash = mpm_utils.get_git_commit(
                f"{global_config.CONSTS.NPU_STACK_PATH}/{global_config.repo}", False, n
            )
            ret.append(TestConfig(n, git_hash, global_config, True, {}))
        build_revisions(ret, global_config)
        return ret
    commit = mpm_utils.get_git_commit(
        f"{global_config.CONSTS.NPU_STACK_PATH}/{global_config.repo}", False
    )
    if global_config.config_compare_file:
        return create_tests_configs_from_config_compare_file(global_config, commit)
    if global_config.config_compare_values:
        return create_tests_configs_from_config_compare_values(global_config, commit)
    return [TestConfig(n, commit, global_config, False, {}) for n in global_config.names]


def build_revisions(tests_configs: List[TestConfig], run_config: GlobalConfig):
    LOG.debug(f"build tests revisions: {[t.name for t in tests_configs]}")
    if REPO_TO_LIB_PATH is None:
        raise RuntimeError(
            f"repos environment variables are not set, can't build git revisions: {[t.name for t in tests_configs]}"
        )
    org_rev = mpm_utils.get_git_branch(run_config)
    if org_rev == "HEAD":
        org_rev = mpm_utils.get_git_commit(
            f"{run_config.CONSTS.NPU_STACK_PATH}/{run_config.repo}", False
        )
    for t in tests_configs:
        if os.path.exists(t.lib_path):
            LOG.debug(
                f"test revision: {t.name}, commit: {t.commit} is already built, skipping... "
            )
            continue
        else:
            shutil.rmtree(
                os.path.join(run_config.work_folder, t.name), ignore_errors=True
            )
        mpm_utils.checkout(
            t.name, f"{run_config.CONSTS.NPU_STACK_PATH}/{run_config.repo}"
        )
        mpm_utils.build(False, run_config)
        os.makedirs(os.path.dirname(t.lib_path), exist_ok=True)
        shutil.copy2(REPO_TO_LIB_PATH.get(run_config.repo), t.lib_path)
    mpm_utils.checkout(org_rev, f"{run_config.CONSTS.NPU_STACK_PATH}/{run_config.repo}")


def dispatch_tests(global_config: GlobalConfig):
    LOG.info(f"start performance measurements for models: {global_config.models}")
    tests = create_tests_configs(global_config)
    if len(global_config.models) == 0:
        raise RuntimeError(f"models list is empty, nothing to do")
    timestamp = get_timestamp()
    models = mpm_utils.get_models_infos(global_config, ModelStats())
    launch_configs: List[LaunchConfig] = []
    for mi in models:
        should_skip_run = True
        curr = []
        for t in tests:
            com = get_com(global_config, t, mi, timestamp)
            should_skip_run &= (
                bool(com.run)
                and com.run.status == "pass"
                and not global_config.overwrite
            )
            com_file_path = os.path.join(
                t.folder_path, com.config.model, f"{com.config.model}.com.json"
            )
            skip_compile = not compile_request(global_config.actions) or bool(
                com.compile
                and com.compile.status == "pass"
                and not global_config.overwrite
            )
            skip_run = not run_request(global_config.actions)
            model_config = ModelConfig(mi, skip_compile, skip_run)
            curr.append(LaunchConfig(global_config, t, model_config, com_file_path))
        if should_skip_run:
            LOG.info(f"found previous run results for model: {mi.name}, skipping...")
        else:
            launch_configs += curr
    max_number_of_threads = (
        max(
            min(
                len(launch_configs),
                (multiprocessing.cpu_count() - 2)
                - (global_config.cards if run_request(global_config.actions) else 0),
            ),
            1,
        )
        if global_config.threads == 0
        else global_config.threads
    )
    dispatcher = Dispatcher(max_number_of_threads, global_config)
    for mi in sorted(launch_configs, key=mpm_runner.sort_by_compile_duration, reverse=True):
        dispatcher.put(mi)
    dispatcher.dispatch()


def ignore_tests(ref_results, tests_names, models, data_path):
    LOG.info(
        f"set ignored results for models: {models}, in tests: {tests_names} from data files: {ref_results}"
    )
    data = []
    for res_file in ref_results:
        with open(res_file, "r") as i_file:
            result_data = json.load(i_file)
        for res in result_data:
            com = ComObj.deserialize(res)
            com.under_test = False
            if com.config.model in models and com.config.name in tests_names:
                com.use_for_ref = False
            data.append(com)
    dump_coms_data(data, data_path)
    LOG.info(f"data file with ignored results: {data_path}")


def ignore_results(global_config: GlobalConfig, test_result: float):
    if len(global_config.models) != 1:
        raise RuntimeError("ignore results can update only one model at a time")
    model = global_config.models[0]
    LOG.info(
        f"set ignored results for model: {model}, which are below (better) than: {test_result} from data files: {global_config.ref_results}"
    )
    data = []
    run_time_metric = RunTimeMetric(LOG, global_config)
    for res_file in global_config.ref_results:
        with open(res_file, "r") as i_file:
            result_data = json.load(i_file)
        for res in result_data:
            com = ComObj.deserialize(res)
            com.under_test = False
            if run_time_metric.run_available(com):
                (
                    _,
                    _,
                    avg_run_time,
                    _,
                    _,
                ) = run_time_metric.calculate_total_run_time(com)
                if com.config.model == model:
                    if convert_to_ms(com.run.time_units, avg_run_time) < test_result:
                        com.use_for_ref = False
            data.append(com)
    dump_coms_data(data, global_config.data_path)
    LOG.info(f"data file with ignored results: {global_config.data_path}")


def get_report_file_path(file_path, file_format):
    return (
        file_path
        if file_path.endswith(".{}".format(file_format))
        else "{}.{}".format(file_path, file_format)
    )


def has_regression(coms: List[ComObj]) -> List[str]:
    ret = []
    for com in coms:
        if not com.under_test:
            continue
        reg_error = MetricType.RUN_TIME in com.status and com.status[MetricType.RUN_TIME].status == ERROR_CODES.ERROR
        if com.config and reg_error:
            ret.append(com.config.model)
    return ret


def update_curr_results(
    global_config, curr_results: List[ComObj], ref_results: List[ComObj]
) -> List[ComObj]:
    unified_results = ref_results + curr_results

    global_config.metrics.set_report(unified_results)
    global_config.metrics.set_status(curr_results)
    return unified_results


def run_and_collect_results(global_config) -> List[ComObj]:
    ref_results = collect_ref_results(global_config)
    unified_results: List[ComObj] = []
    curr_results: List[ComObj] = []

    if is_report_only(global_config.actions):
        LOG.info("Generating report for existing measurement")
        curr_results = collect_curr_results(global_config)
        unified_results = update_curr_results(global_config, curr_results, ref_results)
    else:
        validate_requirements(global_config)

        if global_config.set_const_tensor_max_size:
            set_const_tensor_max_size(global_config.set_const_tensor_max_size)

        regressed_models: List[str] = []
        gc_copy = copy.deepcopy(global_config)
        for i in range(global_config.retry_on_perf_reg + 1):
            retry = bool(regressed_models)
            if retry:
                LOG.info(
                    f'regression found, re-run failed models: {" ".join(regressed_models)}'
                )
                failed_coms = [
                    com
                    for com in curr_results
                    if com.config and com.config.model in regressed_models
                ]
                dump_coms_data(
                    failed_coms,
                    global_config.data_path.replace(".json", f".reg-fail-{i-1}.json"),
                    True,
                )
                gc_copy.models = regressed_models
                gc_copy.actions = ["run"]
                gc_copy.overwrite = True
            dispatch_tests(gc_copy)
            curr_results = collect_curr_results(global_config)
            unified_results = update_curr_results(
                global_config, curr_results, ref_results
            )
            regressed_models = has_regression(unified_results)
            if not regressed_models:
                break
    return unified_results


def run(global_config: GlobalConfig) -> int:
    global_config.metrics = MpmMetrics(LOG, global_config)
    unified_results = run_and_collect_results(global_config)
    status_lists = MpmStatusLists(unified_results)

    show_results(global_config, status_lists)

    errors_file = os.path.abspath(
        os.path.join(os.path.dirname(global_config.report_path), "errors.txt")
    )

    update_report_status(unified_results, errors_file)

    if global_config.report_format:
        print()
        LOG.info("Report files:")
    for f in global_config.report_format:
        report_file_path = get_report_file_path(global_config.report_path, f)
        mpm_report.export_report(
            unified_results, f, report_file_path, global_config.graphs
        )

    if global_config.dump_curr_tests:
        dump_curr_tests(unified_results, global_config.data_path, False)
    elif global_config.dump_curr_status:
        dump_curr_status(
            unified_results, global_config.data_path, False
        )
    else:
        dump_coms_data(unified_results, global_config.data_path, False)

    dump_models_stats(unified_results, global_config)

    if os.path.exists(errors_file):
        LOG.info(f"errors report: {errors_file}")

    status = get_status(status_lists)

    if status != ERROR_CODES.SUCCESS.value:
        mpm_report.create_repro_instructions(
            unified_results,
            os.path.join(os.path.dirname(global_config.report_path), "repro.txt"),
        )
    return status
