from typing import List

from mpm_accuracy_metric import AccuracyMetric
from mpm_base_metric import MetricType, MpmBaseMetric
from mpm_compile_consistency_metric import CompileConsistencyMetric
from mpm_compile_memory_metric import CompileMemoryMetric
from mpm_compile_metric import CompileMetric
from mpm_compile_time_metric import CompileTimeMetric
from mpm_logger import LOG
from mpm_persistent_tensors_metric import PersistentTensorsMetric
from mpm_run_memory_metric import RunMemoryMetric
from mpm_run_metric import RunMetric
from mpm_run_time_metric import RunTimeMetric
from mpm_types import ComObj, GlobalConfig, Report
from mpm_workspcae_metric import WorkspaceMetric


def create_metric(name: str, log, global_config: GlobalConfig) -> MpmBaseMetric:
    metric_type: MetricType = MetricType[name.upper()]
    if metric_type == MetricType.COMPILE:
        return CompileMetric(log, global_config)
    if metric_type == MetricType.COMPILE_CONSISTENCY:
        return CompileConsistencyMetric(log, global_config)
    if metric_type == MetricType.ACCURACY:
        return AccuracyMetric(log, global_config)
    if metric_type == MetricType.COMPILE_TIME:
        return CompileTimeMetric(log, global_config)
    if metric_type == MetricType.COMPILE_MEMORY:
        return CompileMemoryMetric(log, global_config)
    if metric_type == MetricType.PERSISTENT_TENSORS:
        return PersistentTensorsMetric(log, global_config)
    if metric_type == MetricType.RUN:
        return RunMetric(log, global_config)
    if metric_type == MetricType.RUN_TIME:
        return RunTimeMetric(log, global_config)
    if metric_type == MetricType.RUN_MEMORY:
        return RunMemoryMetric(log, global_config)
    if metric_type == MetricType.WORKSPACE:
        return WorkspaceMetric(log, global_config)
    raise RuntimeError(f"unsupported metric name: {name}")


class MpmMetrics:
    def __init__(self, log, global_config: GlobalConfig):
        self.metrics: List[MpmBaseMetric] = []
        if MetricType.COMPILE.name.lower() in global_config.actions:
            global_config.metrics_names.add(MetricType.COMPILE.name.lower())
        if MetricType.RUN.name.lower() in global_config.actions:
            global_config.metrics_names.add(MetricType.RUN.name.lower())
        if global_config.accuracy:
            global_config.metrics_names.add(MetricType.ACCURACY.name.lower())
        if global_config.compile_consistency_iters:
            global_config.metrics_names.add(MetricType.COMPILE_CONSISTENCY.name.lower())
        for m in global_config.metrics_names:
            self.metrics.append(create_metric(m, log, global_config))

    def init_report(self, com: ComObj):
        assert com.config is not None
        if com.report is None:
            com.report = Report.deserialize(com.config.__dict__)
            com.report.model_version = com.config.models_revision if com.config else None

    def get_graphs_stats(self, com: ComObj):
        for m in self.metrics:
            m.get_graphs_stats(com)

    def set_report(self, coms: List[ComObj]):
        for m in self.metrics:
            LOG.debug(f"generate current {m.get_name()} report")
            for com in coms:
                self.init_report(com)
                assert com.report is not None
                m.set_report(com)
            m.set_report_relative_results(coms)

    def set_status(self, coms: List[ComObj]):
        for com in coms:
            for m in self.metrics:
                m.set_status(com)
