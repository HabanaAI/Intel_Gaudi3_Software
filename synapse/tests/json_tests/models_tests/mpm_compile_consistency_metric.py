import os
from typing import List

import mpm_utils
from mpm_base_metric import MetricType, MpmBaseMetric
from mpm_logger import LOG
from mpm_types import ERROR_CODES, ComObj, GlobalConfig, ModelStats, MpmStatus


class CompileConsistencyMetric(MpmBaseMetric):
    def __init__(self, log, global_config: GlobalConfig):
        super().__init__(MetricType.COMPILE_CONSISTENCY, log, global_config)

    def set_status(self, com: ComObj):
        sts = MpmStatus(ERROR_CODES.SUCCESS)
        if not com.under_test or com.compile is None or com.compile.graphs is None:
            return
        assert com.report is not None
        assert com.config is not None
        report = com.report
        for v in com.compile.graphs.values():
            if v.errors:
                if not sts.repro:
                    repro = f"repro: run_models_tests -l d -c {com.config.device} -m {com.config.model} --compile_consistency_iters {self.global_config.compile_consistency_iters}"
                    sts.repro.append(repro)
                sts.status = ERROR_CODES.ERROR
                error = f"metric: {self.get_name()}, found compilation consistency failure in test: {report.name}, model: {report.model}, graph: {v.index_in_file} error/s: {os.linesep.join(v.errors)}"
                sts.messages.append(error)
                debug = f"debug: run_from_json -r -c {com.config.device} -m {com.config.model} -g {v.index_in_file} --consistency_check --test_iters {self.global_config.compile_consistency_iters}"
                sts.repro.append(debug)
        if sts.status != ERROR_CODES.SUCCESS:
            com.status[self.type] = sts
        else:
            self.set_success_status(com)
