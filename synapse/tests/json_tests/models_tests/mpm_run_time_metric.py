import json
import os
import sys
from statistics import mean, median, stdev
from typing import Dict, List

import mpm_utils
from mpm_base_metric import MetricType, MpmBaseMetric
from mpm_logger import LOG
from mpm_types import ERROR_CODES, ComObj, GlobalConfig, MpmStatus


class RunTimeMetric(MpmBaseMetric):
    def __init__(self, log, global_config: GlobalConfig):
        super().__init__(MetricType.RUN_TIME, log, global_config)
        LOG.debug(
            f"{self.get_name()} metric, relative threshold: {self.global_config.threshold}, threshold file: {self.global_config.threshold_file}"
        )
        self.ref_coms: Dict[ComObj, ComObj] = {}

    def remove_graph_outliers(self, graph_times, rng=None):
        times = sorted(graph_times, reverse=True)
        if rng is None:
            return times
        std = stdev(times) if len(times) > 1 else 0
        med = median(times)
        rng *= std
        return [t for t in times if (med - rng < t < med + rng)]

    def remove_graphs_outliers(
        self, graphs, min_iterations, max_search_range
    ):
        graphs_times = []
        min_times = []
        for i in range(len(graphs)):
            v = graphs.get(str(i))
            if v is None or v.times is None:
                return [], []
            min_times.append(min(v.times))
            iter_count = len(v.times)
            for r in range(1, max_search_range):
                times = self.remove_graph_outliers(v.times, r)
                if len(times) > min_iterations * iter_count:
                    break
            if len(times) < min_iterations * iter_count:
                times = self.remove_graph_outliers(v.times)
            graphs_times.append(times)
        return graphs_times, min_times

    def get_run_times(self, com: ComObj):
        if com.run is None or com.run.graphs is None:
            return [], []
        return self.remove_graphs_outliers(com.run.graphs, 0.5, 4)

    def calculate_total_run_time(self, com: ComObj):
        graphs, min_times = self.get_run_times(com)
        min_sum = sum(min_times)
        max_sum = 0
        avg_sum = 0
        std_sum = 0
        calculated_times = 0
        for times in graphs:
            max_sum += max(times)
            avg_sum += sum(times) / float(len(times))
            std_sum += stdev(times) if len(times) > 1 else 0
            calculated_times += len(times)
        return min_sum, max_sum, avg_sum, std_sum, calculated_times / len(graphs)

    def get_graphs_stats(self, com: ComObj):
        if (
            com is None
            or com.config is None
            or com.run is None
            or not os.path.exists(
                os.path.join(com.config.work_folder, com.config.name, com.config.model)
            )
        ):
            return "unavailable"
        graphs_names = self.get_graphs_names(com)
        graph_indexes = self.get_graph_indexes(com)
        all_run_times, _ = self.get_run_times(com)
        run_times = [mean(t) for t in all_run_times]
        total_run_time = sum(run_times) if run_times else 0
        file_path = os.path.join(
            com.config.work_folder,
            com.config.name,
            com.config.model,
            f"{com.config.model}_{self.get_name()}_graphs_stats.csv",
        )
        header = f"Model,Index,Name,Run Time [{com.run.time_units}],Total Run Time [%],Iterations"
        lines = [header]
        num_lines = len(run_times)
        for i in range(num_lines):
            model = com.config.model if com.config is not None else ""
            run_time = f"{run_times[i]}" if len(run_times) > i else ""
            run_percent = (
                f"{100*float(run_times[i])/float(total_run_time):.3}"
                if len(run_times) > i and total_run_time > 0
                else ""
            )
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
            lines.append(
                f"{model},{index},{graph_name},{run_time},{run_percent},{len(all_run_times[i])}"
            )
        with open(file_path, "w") as f:
            f.write("\n".join(lines))
        return file_path

    def set_report(self, com: ComObj):
        if not self.run_available(com):
            return
        assert com.report is not None
        (
            com.report._min_run_time,
            _,
            com.report.avg_run_time,
            com.report._std_dev,
            com.report.iterations,
        ) = self.calculate_total_run_time(com)
        com.report.convert_to_ms(com.run.time_units)
        com.report._run_time_units = "ms"


    def get_ref_run_time_results(self, coms: List[ComObj]) -> Dict[str, ComObj]:
        ref_results: Dict[str, ComObj] = {}
        if not (self.global_config.ref_best or self.global_config.ref_name):
            return ref_results
        for com in coms:
            if com.report is None:
                continue
            if self.global_config.ref_best:
                if com.report.model not in ref_results:
                    ref_results[com.report.model] = com
                min_run_time = ref_results.get(com.report.model).report.avg_run_time
                if com.report.avg_run_time is not None and (
                    min_run_time is None or com.report.avg_run_time < min_run_time
                ):
                    ref_results[com.report.model] = com
            else:
                if self.global_config.ref_name == com.report.name:
                    ref_results[com.report.model] = com
        return ref_results

    def set_report_relative_results(self, coms: List[ComObj]):
        ref_results = self.get_ref_run_time_results(coms)
        if ref_results:
            for res in coms:
                assert res.report is not None
                ref = ref_results.get(res.report.model)
                if ref is None or ref.report is None:
                    self.log.warn(f"in metric: {self.get_name()}, missing model {res.report.model} in reference results")
                    continue
                if res.report.avg_run_time and ref.report.avg_run_time:
                    res.report.run_ref_name = ref.report.name
                    res.report.run_ref_avg_run_time = ref.report.avg_run_time
                    self.ref_coms[res] = ref
                    res.report.run_gain = mpm_utils.calculate_change(
                        res.report.avg_run_time, ref.report.avg_run_time
                    )

    def get_runtimes(self, com):
        graphs = dict()
        for k, v in com.run.graphs.items():
            graphs[k] = sum(v.times) / float(len(v.times)) if len(v.times) > 0 else 0
        return graphs

    def get_runtime_diff(self, ref, curr):
        ref_time = self.get_runtimes(ref)
        curr_time = self.get_runtimes(curr)
        if len(curr_time) != len(ref_time):
            LOG.warn(
                f"can't compare {ref.config.model} model graphs results, the refernce run has "
                f"different graphs count, curr: {len(curr_time) if curr_time else 0}, ref: {len(ref_time) if ref_time else 0}"
            )
            return {}
        diffs = dict()
        for k, v in curr_time.items():
            diffs[k] = v - ref_time.get(k)
        return dict(sorted(diffs.items(), key=lambda item: item[1], reverse=True))

    def set_status(self, com: ComObj):
        threshold_per_model: Dict[str, float] = {}
        if self.global_config.threshold_file:
            with open(self.global_config.threshold_file) as f:
                threshold_per_model = json.load(f)
        relative_threshold = self.global_config.threshold
        if relative_threshold is None:
            relative_threshold = sys.maxsize
        assert com.report is not None
        report = com.report
        model_relative_threshold = float(
            relative_threshold
            if not report.model in threshold_per_model
            else threshold_per_model.get(report.model)
        )
        if report.run_gain and report.run_gain < -abs(model_relative_threshold):
            ref = (
                f' --ref_results {" ".join(self.global_config.ref_results)}'
                if self.global_config.ref_results
                else " --regression_check"
            )
            message = f"metric: {self.get_name()}{os.linesep}found performance regression in test: {report.name}, model: {report.model}, best test name: {report.run_ref_name}, {report.name} run time: {float(report.avg_run_time):.3f} [{report._run_time_units}], {report.run_ref_name} run time: {float(report.run_ref_avg_run_time):.3f} [{report._run_time_units}], performance regression: {round(report.run_gain, 2)} [%]"
            ref_com = self.ref_coms.get(com)
            diffs = self.get_runtime_diff(ref_com, com)
            assert com.config is not None
            graphs_regression_file = os.path.join(
                com.config.work_folder,
                com.config.name,
                f"{com.config.model}_graphs_regression.json",
            )
            with open(graphs_regression_file, "w") as o_file:
                json.dump(diffs, o_file, indent=4)
            repro = [
                f"option 1 - compare to best result:{os.linesep}\trun_models_tests -l d -c {com.config.device} -m {com.config.model} --ref_best --threshold {model_relative_threshold}{ref}",
                f"option 2 - manual compare with vs without commit:{os.linesep}\tbuild the relevant repo WITHOUT the change that created this regression and run:{os.linesep}"
                f"\trun_models_tests -l d -n ref -c {com.config.device} -m {com.config.model} -w /tmp/{com.config.model}-reg-check{os.linesep}"
                f"\tbuild the relevant repo WITH the change that created this regression and run:{os.linesep}"
                f"\trun_models_tests -l d -n regression -c {com.config.device} -m {com.config.model} -w /tmp/{com.config.model}-reg-check --ref_name ref --threshold {model_relative_threshold}{os.linesep}"
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
