import math
import os
from typing import Dict, List

import mpm_utils
from mpm_base_metric import MetricType, MpmBaseMetric
from mpm_types import ComObj, GlobalConfig


class CompileTimeMetric(MpmBaseMetric):
    def __init__(self, log, global_config: GlobalConfig):
        super().__init__(MetricType.COMPILE_TIME, log, global_config)

    def get_compile_times(self, com: ComObj):
        ret = []
        if not self.compile_available(com):
            return None
        graphs = com.compile.graphs
        if graphs is None:
            return None
        for i in range(len(graphs)):
            v = graphs.get(str(i))
            if v is None or v.time is None:
                return None
            ret.append(v.time)
        return ret

    def calculate_total_compile_time(self, com: ComObj):
        compile_times = self.get_compile_times(com)
        if compile_times is None:
            return math.nan
        return sum(compile_times)

    def get_graphs_stats(self, com: ComObj):
        if (
            com is None
            or com.config is None
            or com.compile is None
            or not os.path.exists(
                os.path.join(com.config.work_folder, com.config.name, com.config.model)
            )
        ):
            return "unavailable"
        graphs_names = self.get_graphs_names(com)
        compile_times = self.get_compile_times(com)
        graph_indexes = self.get_graph_indexes(com)
        if not compile_times:
            return None
        total_compile_time = sum(compile_times)
        file_path = os.path.join(
            com.config.work_folder,
            com.config.name,
            com.config.model,
            f"{com.config.model}_{self.get_name()}_graphs_stats.csv",
        )
        header = f"Model,Index,Name,Compile Time [{com.compile.time_units}],Total Compile Time [%]"
        lines = [header]
        num_lines = len(compile_times)
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
            compile_time = f"{compile_times[i]:.3}" if len(compile_times) > i else ""
            compile_percent = (
                f"{100*float(compile_times[i])/float(total_compile_time):.3}"
                if len(compile_times) > i and total_compile_time > 0
                else ""
            )
            lines.append(
                f"{model},{index},{graph_name},{compile_time},{compile_percent}"
            )
        with open(file_path, "w") as f:
            f.write("\n".join(lines))
        return file_path

    def get_ref_compile_time_results(self, coms: List[ComObj]) -> Dict[str, ComObj]:
        ref_results: Dict[str, ComObj] = {}
        if not (self.global_config.ref_best or self.global_config.ref_name):
            return ref_results
        for com in coms:
            if com.report is None:
                continue
            if self.global_config.ref_best:
                if com.report.model not in ref_results:
                    ref_results[com.report.model] = com
                min_compile_time = ref_results.get(com.report.model).report.compile_time
                if com.report.compile_time is not None and (
                    min_compile_time is None
                    or com.report.compile_time < min_compile_time
                ):
                    ref_results[com.report.model] = com
            else:
                if self.global_config.ref_name == com.report.name:
                    ref_results[com.report.model] = com
        return ref_results

    def set_report(self, com: ComObj):
        assert com.config is not None
        assert com.report is not None
        com.report.compile_time = self.calculate_total_compile_time(com)

    def set_report_relative_results(self, coms: List[ComObj]):
        ref_results = self.get_ref_compile_time_results(coms)
        if ref_results:
            for res in coms:
                assert res.report is not None
                ref = ref_results.get(res.report.model)
                if ref is None or ref.report is None:
                    self.log.warn(f"in metric: {self.get_name()}, missing model {res.report.model} in reference results")
                    continue
                if res.report.compile_time and ref.report.compile_time:
                    res.report.compile_ref_name = ref.report.name
                    res.report.compile_gain = mpm_utils.calculate_change(
                        res.report.compile_time, ref.report.compile_time
                    )

    def set_status(self, com: ComObj):
        if self.compile_available(com):
            self.set_success_status(
                com, [f"graphs stats: {self.get_graphs_stats(com)}"]
            )
