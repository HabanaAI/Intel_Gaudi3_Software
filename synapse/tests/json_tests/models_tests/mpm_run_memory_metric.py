from mpm_base_metric import MetricType, MpmBaseMetric
from mpm_types import ERROR_CODES, ComObj, GlobalConfig


class RunMemoryMetric(MpmBaseMetric):
    def __init__(self, log, global_config: GlobalConfig):
        super().__init__(MetricType.RUN_MEMORY, log, global_config)

    def set_report(self, com: ComObj):
        if com.run and com.report and com.run.stats:
            com.report.run_mem = com.run.stats.max_memory_usage

    def set_status(self, com: ComObj):
        if not self.run_available(com):
            return
        if com.report and com.report.run_mem is not None:
            self.set_success_status(com)
        else:
            self.update_com_status(
                com, ERROR_CODES.ERROR, ["failed to capture run memory usage"]
            )
