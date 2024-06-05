import os
import sys
import mpm_utils

from typing import Dict, List

from mpm_base_metric import MetricType, MpmBaseMetric
from mpm_types import ERROR_CODES, ComObj, GlobalConfig, MpmStatus


class WorkspaceMetric(MpmBaseMetric):
    def __init__(self, log, global_config: GlobalConfig):
        super().__init__(MetricType.WORKSPACE, log, global_config)
        self.ref_coms: Dict[ComObj, ComObj] = {}

    def get_workspace_sizes(self, com: ComObj):
        ret = []
        if not self.compile_available(com):
            return None
        graphs = com.compile.graphs
        if graphs is None:
            return None
        for i in range(len(graphs)):
            v = graphs.get(str(i))
            if v is None or v.workspace_size is None:
                return None
            ret.append(v.workspace_size)
        return ret

    def get_graphs_stats(self, com: ComObj):
        if (
            com is None
            or com.config is None
            or not os.path.exists(
                os.path.join(com.config.work_folder, com.config.name, com.config.model)
            )
        ):
            return "unavailable"
        graphs_names = self.get_graphs_names(com)
        workspace_sizes = self.get_workspace_sizes(com)
        graph_indexes = self.get_graph_indexes(com)
        if not workspace_sizes:
            return None
        file_path = os.path.join(
            com.config.work_folder,
            com.config.name,
            com.config.model,
            f"{com.config.model}_{self.get_name()}_graphs_stats.csv",
        )
        header = "Model,Index,Name,Workspace Size [bytes]"
        lines = [header]
        num_lines = len(workspace_sizes)
        for i in range(num_lines):
            model = com.config.model if com.config is not None else ""
            index = (
                f"{graph_indexes[i]}"
                if graph_indexes is not None and len(graph_indexes) > i
                else "unknown"
            )
            graph_name = (
                f"{graphs_names[i]}"
                if graphs_names is not None and len(graphs_names) > i
                else ""
            )
            workspace_size = (
                f"{workspace_sizes[i]}"
                if workspace_sizes and len(workspace_sizes) > i
                else ""
            )
            lines.append(f"{model},{index},{graph_name},{workspace_size}")
        with open(file_path, "w") as f:
            f.write("\n".join(lines))
        return file_path

    def set_report(self, com: ComObj):
        assert com.report is not None
        ws_sizes = self.get_workspace_sizes(com)
        com.report.total_ws_size = sum(ws_sizes) if ws_sizes is not None else 0
        com.report.largest_ws_size = max(ws_sizes) if ws_sizes is not None else 0

    def get_ref_workspace_size_results(self, coms: List[ComObj]) -> Dict[str, ComObj]:
        ref_results: Dict[str, ComObj] = {}
        if not (self.global_config.ref_best or self.global_config.ref_name):
            return ref_results
        for com in coms:
            if com.report is None:
                continue
            if self.global_config.ref_best:
                if com.report.model not in ref_results:
                    ref_results[com.report.model] = com
                largest_ws_size = ref_results.get(
                    com.report.model
                ).report.largest_ws_size
                if com.report.largest_ws_size is not None and (
                    largest_ws_size is None
                    or com.report.largest_ws_size < largest_ws_size
                ):
                    ref_results[com.report.model] = com
            else:
                if self.global_config.ref_name == com.report.name:
                    ref_results[com.report.model] = com
        return ref_results

    def set_report_relative_results(self, coms: List[ComObj]):
        ref_results = self.get_ref_workspace_size_results(coms)
        if ref_results:
            for res in coms:
                assert res.report is not None
                ref = ref_results.get(res.report.model)
                if ref is None or ref.report is None:
                    self.log.warn(
                        f"in metric: {self.get_name()}, missing model {res.report.model} in reference results"
                    )
                    continue
                if res.report.largest_ws_size and ref.report.largest_ws_size:
                    res.report.largest_ws_size_ref_name = ref.report.name
                    res.report.largest_ws_size_ref = ref.report.largest_ws_size
                    self.ref_coms[res] = ref
                    res.report.largest_ws_size_gain = mpm_utils.calculate_change(
                        res.report.largest_ws_size, ref.report.largest_ws_size
                    )

    def set_status(self, com: ComObj):
        model_relative_threshold = self.global_config.workspace_size_threshold
        if model_relative_threshold is None:
            model_relative_threshold = sys.maxsize
        assert com.report is not None
        report = com.report
        if report.largest_ws_size_gain and report.largest_ws_size_gain < -abs(model_relative_threshold):
            message = f"metric: {self.get_name()}{os.linesep}found workspace size regression in test: {report.name}, model: {report.model}, best test name: {report.largest_ws_size_ref_name}, {report.name} largest workspace size: {report.largest_ws_size}, {report.largest_ws_size_ref_name} largest workspace size: {report.largest_ws_size_ref}, largest workspace size regression: {round(report.largest_ws_size_gain, 2)} [%]"
            assert com.config is not None
            graphs_regression_file = os.path.join(
                com.config.work_folder,
                com.config.name,
                f"{com.config.model}_graphs_regression.json",
            )
            repro = [
                f"manual compare with vs without commit:{os.linesep}\tbuild the relevant repo WITHOUT the change that created this regression and run:{os.linesep}"
                f"\trun_models_tests -l d -a compile -n ref -c {com.config.device} -m {com.config.model} -w /tmp/{com.config.model}-reg-check{os.linesep}"
                f"\tbuild the relevant repo WITH the change that created this regression and run:{os.linesep}"
                f"\trun_models_tests -l d -a compile -n regression -c {com.config.device} -m {com.config.model} -w /tmp/{com.config.model}-reg-check --ref_name ref --threshold {model_relative_threshold}{os.linesep}"
                f"\tsee /tmp/{com.config.model}-reg-check/results.csv for run's details",
            ]
            sts = MpmStatus(ERROR_CODES.ERROR)
            sts.messages.append(
                f"{message}{os.linesep}graphs stats: {self.get_graphs_stats(com)}{os.linesep}graphs regression report: {graphs_regression_file}"
            )
            sts.repro = repro
            com.status[self.type] = sts
        else:
            self.set_success_status(
                com, [f"graphs stats: {self.get_graphs_stats(com)}"]
            )
