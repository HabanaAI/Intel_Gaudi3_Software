import os
from typing import List

import mpm_utils
from mpm_base_metric import MetricType, MpmBaseMetric
from mpm_logger import LOG
from mpm_types import ERROR_CODES, ComObj, GlobalConfig, ModelStats, MpmStatus


class AccuracyMetric(MpmBaseMetric):
    def __init__(self, log, global_config: GlobalConfig):
        super().__init__(MetricType.ACCURACY, log, global_config)
        self.missing_data_files: List[str] = self.validate_accuracy_requirments(
            global_config
        )

    def validate_accuracy_requirments(self, global_config: GlobalConfig) -> List[str]:
        try:
            ret: List[str] = []
            if not global_config.accuracy:
                return ret
            if global_config.tensors_data_folder is None or not os.path.exists(
                global_config.tensors_data_folder
            ):
                raise RuntimeError(
                    "--accuracy is set but --tensors_data_folder doesn't exist"
                )
            models_infos = mpm_utils.get_models_infos(global_config, ModelStats())
            for m in models_infos:
                if m.data_file_path is None:
                    msg = f"tensors data file for model: {m} is missing"
                    if global_config.keep_going:
                        LOG.warning(msg)
                        ret.append(m)
                    else:
                        raise RuntimeError(msg)
            return ret
        except Exception as e:
            LOG.error(f"can't run accuracy tests, error: {str(e)}")
            status = ERROR_CODES.FAIL.value
            print(f"Exit with error: {status}")
            exit(status)

    def set_status(self, com: ComObj):
        sts = MpmStatus(ERROR_CODES.SUCCESS)
        if not com.under_test or com.run is None or com.run.graphs is None:
            return
        assert com.report is not None
        assert com.config is not None
        if self.missing_data_files and com.config.model in self.missing_data_files:
            sts.status = ERROR_CODES.WARNING
            sts.messages.append("missing tensors data file")
            com.status[self.type] = sts
        else:
            report = com.report
            for v in com.run.graphs.values():
                if v.errors:
                    if not sts.repro:
                        repro = f"repro: run_models_tests -l d -c {com.config.device} -m {com.config.model} --accuracy"
                        sts.repro.append(repro)
                    sts.status = ERROR_CODES.ERROR
                    error = f"metric: {self.get_name()}, found accuracy failure in test: {report.name}, model: {report.model}, graph: {v.index_in_file} error/s: {os.linesep.join(v.errors)}"
                    sts.messages.append(error)
                    debug = f"debug: run_from_json -r --run -c {com.config.device} -m {com.config.model} -d {com.config.model_data_file} -g {v.index_in_file}"
                    sts.repro.append(debug)
        if sts.status != ERROR_CODES.SUCCESS:
            com.status[self.type] = sts
        else:
            self.set_success_status(com)
