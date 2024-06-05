import os
from enum import Enum, IntEnum
from math import isnan
from numbers import Number
from typing import Dict, List, Optional, Set, Tuple


class MetricType(IntEnum):
    ACCURACY = 1,
    COMPILE = 2,
    COMPILE_CONSISTENCY = 3,
    COMPILE_MEMORY = 4,
    COMPILE_TIME = 5,
    PERSISTENT_TENSORS = 6,
    RUN = 7,
    RUN_MEMORY = 8,
    RUN_TIME = 9,
    WORKSPACE = 10,


class ERROR_CODES(IntEnum):
    SUCCESS = 0
    WARNING = 1
    ERROR = 2


class Devices(Enum):
    GRECO = "greco"
    GAUDI_M = "gaudiM"
    GAUDI_1 = "gaudi"
    GAUDI_2 = "gaudi2"
    GAUDI_3 = "gaudi3"


class MpmObj:
    def from_dict(self, data):
        if data is not None:
            for k, v in data.items():
                if hasattr(self, k):
                    setattr(self, k, v)

    def serialize_list(self, data):
        ret = list()
        for v in data:
            if v == None:
                continue
            if isinstance(v, MpmObj):
                ret.append(v.serialize())
            elif isinstance(v, list):
                ret.append(self.serialize_list(v))
            elif isinstance(v, dict):
                ret.append(self.serialize_dict(v))
            else:
                ret.append(v)
        return ret

    def serialize_dict(self, data):
        ret = dict()
        for k, v in data.items():
            if v == None:
                continue
            if isinstance(v, MpmObj):
                ret[k] = v.serialize()
            elif isinstance(v, list):
                ret[k] = self.serialize_list(v)
            elif isinstance(v, dict):
                ret[k] = self.serialize_dict(v)
            else:
                ret[k] = v
        return ret

    def serialize(self):
        ret = dict()
        for k, v in self.__dict__.items():
            if v == None:
                continue
            if k.startswith("_") == None:
                continue
            if isinstance(v, MpmObj):
                ret[k] = v.serialize()
            elif isinstance(v, list):
                ret[k] = self.serialize_list(v)
            elif isinstance(v, dict):
                ret[k] = self.serialize_dict(v)
            else:
                ret[k] = v
        return ret

    @classmethod
    def deserialize(cls, data):
        ret = cls()
        ret.from_dict(data)
        return ret


class ComConfig(MpmObj):
    def __init__(self):
        self.actions = None
        self.data_comparator_config_file = None
        self.device = None
        self.git_hash = None
        self.groups = None
        self.iterations = None
        self.model = None
        self.model_data_file = None
        self.model_file = None
        self.models_revision = None
        self.name = None
        self.recipe_folder = None
        self.release_device = None
        self.repo = None
        self.time_measurement = None
        self.timestamp = None
        self.work_folder = None
        self.marked_for_debug: Optional[bool] = None
        self.compilation_mode = None
        self.compile_consistency_iters = None

    def key(self):
        return (self.name, self.git_hash, self.model, self.timestamp)


class Graph(MpmObj):
    def __init__(self):
        self.name = None
        self.recipe_file = None
        self.time = None
        self.times = None
        self.warnings = None
        self.errors = None
        self.workspace_size = None
        self.num_persistent_tensors = None
        self.persistent_tensors_size = None
        self.index_in_file = None


class Stats(MpmObj):
    def __init__(self):
        self.duration = 0
        self.max_memory_usage = 0


class ModelStats(MpmObj):
    def __init__(self):
        self.compile: Stats = Stats()
        self.run: Stats = Stats()

    @classmethod
    def deserialize(cls, data):
        ret = cls()
        ret.from_dict(data)
        if data:
            compile_stats = data.get("compile", None)
            ret.compile = Stats.deserialize(compile_stats) if compile_stats else None
            run_stats = data.get("run", None)
            ret.run = Stats.deserialize(run_stats) if run_stats else None
        return ret


class Result(MpmObj):
    def __init__(self):
        self.status = None
        self.error = None
        self.time_units = None
        self.graphs = None
        self.stats: Optional[Stats] = None

    @classmethod
    def deserialize(cls, data):
        ret = cls()
        ret.from_dict(data)
        if data:
            if "graphs" in data and data["graphs"]:
                for k, v in data["graphs"].items():
                    ret.graphs[k] = Graph.deserialize(v)
            if "stats" in data:
                ret.stats = Stats.deserialize(data["stats"])
        return ret


class MpmStatus(MpmObj):
    def __init__(self, status: ERROR_CODES):
        self.status: ERROR_CODES = status
        self.messages: List[str] = []
        self.repro: List[str] = []

class ComObj(MpmObj):
    def __init__(self):
        self.config: Optional[ComConfig] = None
        self.compile: Optional[Result] = None
        self.run: Optional[Result] = None
        self.report: Optional[Report] = None
        self.under_test: Optional[bool] = None
        self.use_for_ref: Optional[bool] = None
        self.status: Dict[MetricType, MpmStatus] = {}

    @classmethod
    def deserialize(cls, data):
        ret = cls()
        ret.config = ComConfig.deserialize(data["config"]) if "config" in data else None
        ret.compile = Result.deserialize(data["compile"]) if "compile" in data else None
        ret.run = Result.deserialize(data["run"]) if "run" in data else None
        ret.use_for_ref = data["use_for_ref"] if "use_for_ref" in data else True
        return ret


class MpmStatusLists():
    def __init__(self, coms: List[ComObj]):
        status, count = self.parse(coms)
        self.success: List[MpmStatus] = status.get(ERROR_CODES.SUCCESS) or []
        self.warnings: List[MpmStatus] = status.get(ERROR_CODES.WARNING) or []
        self.errors: List[MpmStatus] = status.get(ERROR_CODES.ERROR) or []
        self.count: int = count

    def parse(self, coms: List[ComObj]) -> Tuple[Dict[ERROR_CODES, List[MpmStatus]], int]:
        status: Dict[ERROR_CODES, List[MpmStatus]] = {}
        count = 0
        for e in ERROR_CODES:
            status[e] = []
        for com in coms:
            if not com.under_test:
                continue
            for s in com.status.values():
                if s is None:
                    continue
                count += 1
                status[s.status].append(s)
        return status, count



class Report(MpmObj):
    def __init__(self):
        self.model = None
        self.name = None
        self.device = None
        self.under_test = None
        self.status = None
        self.avg_run_time = None
        self.run_ref_avg_run_time = None
        self.run_gain = None
        self.run_ref_name = None
        self._run_time_units = None
        self._min_run_time = None
        self.run_mem = None
        self.compile_time = None
        self.compile_gain = None
        self.compile_ref_name = None
        self.compile_mem = None
        self.largest_ws_size = None
        self.largest_ws_size_gain = None
        self.largest_ws_size_ref = None
        self.largest_ws_size_ref_name = None
        self.total_ws_size = None
        self._std_dev = None
        self.iterations = None
        self.timestamp = None
        self.repo = None
        self.git_hash = None
        self.model_version = None

    def convert_to_ms(self, time_units):
        if time_units == "ns":
            self._min_run_time /= 1000000.0
            self.avg_run_time /= 1000000.0
            self._std_dev /= 1000000.0
        if time_units == "us":
            self._min_run_time /= 1000.0
            self.avg_run_time /= 1000.0
            self._std_dev /= 1000.0

    def total_time(self):
        time = float(self.compile_time if self.compile_time else 0)
        time += (
            float(self.avg_run_time) * float(self.iterations)
            if self.avg_run_time
            else 0
        )
        return time

    def get_xml_fields(self):
        ret = {}
        properties = [
            "avg_run_time",
            "min_run_time",
            "iterations",
            "compile_time",
            "run_gain",
            "compile_gain",
            "model_version",
            "largest_ws_size",
            "largest_ws_size_gain",
            "total_ws_size",
        ]
        total_time = self.total_time() * 0.001
        if not isnan(total_time):
            ret["time"] = str(total_time)
        for k, v in self.__dict__.items():
            if v is not None and (not isinstance(v, Number) or not isnan(v)):
                key = k if k != "model" else "classname"
                if key in properties:
                    key = "properties.{}".format(key)
                ret[key] = str(v)
        return ret


class MemSts():
    def __init__(self):
        self.total, self.used, _ = map(int, os.popen('free -t -b').readlines()[-1].split()[1:])
        self.available = self.total - self.used

class ModelInfo:
    def __init__(self, name: str = None, file_path: str = None, data_file_path: str = None, stats: ModelStats = None):
        self.name: str = name
        self.file_path: str = file_path
        self.data_file_path: str = data_file_path
        self.stats: ModelStats = stats


class GlobalConstants:
    def __init__(self):
        self.LOCAL_MODELS_FOLDER = f"/tmp/.mpm"
        self.MODELS_REPO = "ssh://gerrit:29418/mpm-test-data"
        self.DEFAULT_MODELS_FOLDER = os.path.join(self.LOCAL_MODELS_FOLDER, "models")
        self.DEFAULT_DATA_FOLDER = os.path.join(self.LOCAL_MODELS_FOLDER, "data")
        self.NPU_STACK_PATH = os.environ.get("HABANA_NPU_STACK_PATH")
        self.PROF_CONFIG_TEMPLATE = os.path.join(os.path.dirname(__file__), "shim_config.json")
        self.MODELS_FILTER_FILE_NAME = "models_filter.json"
        self.DEBUG_MARKER_FILE_NAME = "mpm_debug_marker"


class GlobalConfig:
    def __init__(self, args):
        self.CONSTS = GlobalConstants()
        self.accuracy: bool = args.accuracy
        self.actions: List[str] = args.actions
        self.build: bool = args.build
        self.cards: int = args.cards
        self.chip_type: str = args.chip_type
        self.compilation_mode: str = args.compilation_mode
        self.config_compare_file: str = args.config_compare_file
        self.config_compare_values: List[str] = args.config_compare_values
        self.compile_consistency_iters: int = args.compile_consistency_iters
        self.data_comparator_config_file: str = args.data_comparator_config
        self.data_path: str = args.data_path
        self.dump_curr_status: bool = args.dump_curr_status
        self.dump_curr_tests: bool = args.dump_curr_tests
        self.execution_time_limit: float = args.execution_time_limit
        self.gen_ref_file: str = args.gen_ref_file
        self.git_revs: List[str] = args.git_revs
        self.graphs: bool = args.graphs
        self.iterations: int = args.iterations
        self.keep_going: bool = args.keep_going
        self.max_mem_limit: int = args.max_mem_limit
        self.models_file: str = args.models_file
        self.metrics = None
        self.metrics_names: Set[str] = set(args.metrics)
        self.models: List[str] = args.models
        self.models_folder: str = args.models_folder
        self.models_revision: str = args.models_revision
        self.models_stats_required: bool = args.models_stats_required
        self.models_tests_binary_path: str = args.models_tests_bin
        self.names: List[str] = args.names
        self.overwrite: bool = args.overwrite
        self.precompiled_test:str = args.precompiled_test
        self.profiler: bool = args.prof
        self.ref_best: bool = args.ref_best
        self.ref_file: str = args.ref_file
        self.ref_name: str = args.ref_name
        self.ref_results: List[str] = args.ref_results
        self.release_device: bool = args.release_device
        self.repo: str = args.repo
        self.report_format: str = args.report_format
        self.report_path: str = args.report_path
        self.retry_on_failure: int = args.retry_on_failure
        self.retry_on_perf_reg: int = args.retry_on_perf_reg
        self.set_const_tensor_max_size: int = args.set_const_tensor_max_size
        self.show_precompiled: bool = args.show_precompiled
        self.skip_models: bool = args.skip_models
        self.tensors_data_folder: str = args.tensors_data_folder
        self.threads: int = args.threads
        self.threshold: float = args.threshold
        self.workspace_size_threshold: float = args.workspace_size_threshold
        self.threshold_file: str = args.threshold_file
        self.time_measurement: str = args.time_measurement
        self.work_folder: str = args.work_folder


class TestConfig:
    def __init__(self, name: str, commit: str, global_config: GlobalConfig, create_lib_path: bool, env: Dict[str, str]):
        self.name: str = name
        self.commit: str = commit
        self.folder_path: str = os.path.join(global_config.work_folder, name)
        self.lib_path: str = os.path.join(self.folder_path, f"{global_config.repo}-{commit}.so") if create_lib_path else None
        self.env: Dict[str, str] = env


class ModelConfig:
    def __init__(self, info: ModelInfo = None, skip_compile: bool = False, skip_run: bool = False):
        self.info: ModelInfo = info
        self.skip_compile: bool = skip_compile
        self.skip_run: bool = skip_run


class LaunchConfig:
    def __init__(self, global_config: GlobalConfig = None, test: TestConfig = None, model: ModelConfig = None, com_file_path: str = None, status: int = ERROR_CODES.SUCCESS.value):
        self.global_config: GlobalConfig = global_config
        self.test: TestConfig = test
        self.model: ModelConfig = model
        self.com_file_path: str = com_file_path
        self.status: int = status


class ProcessPriority:
    ABOVE_NORMAL = -10
    NORMAL = 0
    BELOW_NORMAL = 10