import mpm_utils
from mpm_base_metric import MetricType, MpmBaseMetric
from mpm_types import ERROR_CODES, ComObj, GlobalConfig, MpmStatus


class RunMetric(MpmBaseMetric):
    def __init__(self, log, global_config: GlobalConfig):
        super().__init__(MetricType.RUN, log, global_config)
        self.device_id: str = self.get_device_id()

    def get_device_id(self):
        exe = mpm_utils.Executer("lspci", "-d 1da3: -n")
        if exe.get_status() != 0 or len(exe.get_prints()) == 0:
            return None
        output = exe.get_prints()[0]
        if "1000" in output:
            return "gaudi 1"
        if "1001" in output:
            return "gaudi M"
        if "1020" in output:
            return "gaudi 2"
        if "1060" in output:
            return "gaudi 3"
        if "ff0b" in output:
            return "gaudi 3 simulator"
        return None

    def set_status(self, com: ComObj):
        assert com.config is not None
        if not self.should_run(com):
            return
        assert com.report is not None
        com.report.device = self.device_id
        is_warning = com.run is not None and com.run.status == "pass" and com.run.error is not None
        if not is_warning and self.run_available(com):
            self.set_success_status(com)
        else:
            err_msg = com.run.error if com.run is not None else "unknown"
            last_index = (
                max(sorted(com.run.graphs.keys()), key=int)
                if com.run and com.run.graphs
                else None
            )
            graph_index = (
                None
                if last_index is None
                else com.run.graphs[str(last_index)].index_in_file
            )
            graph_index_flag = "" if graph_index is None else f"-g {graph_index}"
            sts = MpmStatus(ERROR_CODES.WARNING) if is_warning else MpmStatus(ERROR_CODES.ERROR)
            sts.messages.append(
                f"metric: {self.get_name()}, test: {com.config.name}, model: {com.config.model} run failed, error: {err_msg}"
            )
            if not is_warning:
                sts.repro.append(
                    f"repro: run_models_tests -l d -c {com.config.device} -m {com.config.model}"
                )
                sts.repro.append(
                    f"debug: run_from_json -r --run -c {com.config.device} -m {com.config.model} {graph_index_flag}"
                )
            com.status[self.type] = sts
