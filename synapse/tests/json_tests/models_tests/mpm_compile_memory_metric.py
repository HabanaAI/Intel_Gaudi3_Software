from mpm_base_metric import MetricType, MpmBaseMetric
from mpm_types import ERROR_CODES, ComObj, GlobalConfig


class CompileMemoryMetric(MpmBaseMetric):
    def __init__(self, log, global_config: GlobalConfig):
        super().__init__(MetricType.COMPILE_MEMORY, log, global_config)

    def set_report(self, com: ComObj):
        if com.compile and com.report and com.compile.stats:
            com.report.compile_mem = com.compile.stats.max_memory_usage

    def set_status(self, com: ComObj):
        if not self.compile_available(com):
            return
        if com.report and com.report.compile_mem is not None:
            self.set_success_status(com)
        else:
            self.update_com_status(
                com, ERROR_CODES.ERROR, ["failed to capture compile memory usage"]
            )
