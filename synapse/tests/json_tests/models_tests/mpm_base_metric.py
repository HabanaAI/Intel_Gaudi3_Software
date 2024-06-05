from typing import List

from mpm_types import ERROR_CODES, ComObj, GlobalConfig, MetricType, MpmStatus

DEFAULT_METRICS = [
    MetricType.COMPILE,
    MetricType.COMPILE_TIME,
    MetricType.COMPILE_MEMORY,
    MetricType.WORKSPACE,
    MetricType.PERSISTENT_TENSORS,
    MetricType.RUN,
    MetricType.RUN_TIME,
    MetricType.RUN_MEMORY,
]


class MpmBaseMetric:
    def __init__(
        self,
        metric_type: MetricType,
        log,
        global_config: GlobalConfig,
    ):
        self.type: MetricType = metric_type
        self.log = log
        self.global_config = global_config

    def get_name(self):
        return self.type.name.lower()

    def update_com_status(self, com: ComObj, status: ERROR_CODES, messages: List[str] = []):
        name = com.config.name if com.config else "unknown"
        model = com.config.model if com.config else "unknown"
        sts = MpmStatus(status)
        sts.messages.append(f"metric: {self.get_name()}, test: {name}, model: {model}")
        for m in messages:
            sts.messages.append(m)
        com.status[self.type] = sts

    def set_success_status(self, com: ComObj, messages: List[str] = []):
        self.update_com_status(com, ERROR_CODES.SUCCESS, messages)

    def get_graphs_stats(self, com: ComObj):
        raise NotImplementedError()

    def set_report(self, com: ComObj):
        pass

    def set_report_relative_results(self, coms: List[ComObj]):
        pass

    def set_status(self, com: ComObj):
        pass

    def should_compile(self, com):
        return com.under_test and "compile" in com.config.actions and com.config.marked_for_debug != False

    def should_run(self, com):
        return com.under_test and "run" in com.config.actions and com.config.marked_for_debug != False

    def compile_available(self, com):
        return com.compile and com.compile.status == "pass"

    def run_available(self, com):
        return com.run and com.run.status == "pass"

    def get_graph_indexes(self, com: ComObj):
        ret = []
        if not self.compile_available(com):
            return None
        graphs = com.compile.graphs
        if graphs is None:
            return None
        for i in range(len(graphs)):
            v = graphs.get(str(i))
            if v is None or v.index_in_file is None:
                return None
            ret.append(v.index_in_file)
        return ret

    def get_graphs_names(self, com: ComObj):
        ret = []
        if not self.compile_available(com):
            return None
        graphs = com.compile.graphs
        if graphs is None:
            return None
        for i in range(len(graphs)):
            v = graphs.get(str(i))
            if v is None or v.name is None:
                return None
            ret.append(v.name)
        return ret
