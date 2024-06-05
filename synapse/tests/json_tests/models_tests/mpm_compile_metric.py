from mpm_base_metric import MetricType, MpmBaseMetric
from mpm_types import ERROR_CODES, ComObj, GlobalConfig, MpmStatus


class CompileMetric(MpmBaseMetric):
    def __init__(self, log, global_config: GlobalConfig):
        super().__init__(MetricType.COMPILE, log, global_config)

    def set_status(self, com: ComObj):
        assert com.config is not None
        if not self.should_compile(com):
            return
        is_warning = com.compile is not None and com.compile.status == "pass" and com.compile.error is not None
        if not is_warning and self.compile_available(com):
            self.set_success_status(com)
        else:
            err_msg = com.compile.error if com.compile is not None else "unknown"
            last_index = (
                max(sorted(com.compile.graphs.keys()), key=int)
                if com.compile and com.compile.graphs
                else None
            )
            graph_index = (
                None
                if last_index is None
                else com.compile.graphs[str(last_index)].index_in_file
            )
            graph_index_flag = "" if graph_index is None else f"-g {graph_index}"
            sts = MpmStatus(ERROR_CODES.WARNING) if is_warning else MpmStatus(ERROR_CODES.ERROR)
            sts.messages.append(
                f"metric: {self.get_name()}, test: {com.config.name}, model: {com.config.model} compilation failed, error: {err_msg}"
            )
            if not is_warning:
                sts.repro.append(
                    f"repro: run_models_tests -l d -c {com.config.device} -m {com.config.model} -a compile"
                )
                sts.repro.append(
                    f"debug: run_from_json -r -c {com.config.device} -m {com.config.model} {graph_index_flag}"
                )
            com.status[self.type] = sts
