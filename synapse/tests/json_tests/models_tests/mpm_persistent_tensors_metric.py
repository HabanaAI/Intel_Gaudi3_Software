import os

from mpm_base_metric import MetricType, MpmBaseMetric
from mpm_types import ERROR_CODES, ComObj, GlobalConfig


class PersistentTensorsMetric(MpmBaseMetric):
    def __init__(self, log, global_config: GlobalConfig):
        super().__init__(MetricType.PERSISTENT_TENSORS, log, global_config)

    def get_num_persistent_tensors(self, com: ComObj):
        ret = []
        if not self.compile_available(com):
            return None
        graphs = com.compile.graphs
        if graphs is None:
            return None
        for i in range(len(graphs)):
            v = graphs.get(str(i))
            if v is None or v.num_persistent_tensors is None:
                return None
            ret.append(v.num_persistent_tensors)
        return ret

    def get_persistent_tensors_sizes(self, com: ComObj):
        ret = []
        if not self.compile_available(com):
            return None
        graphs = com.compile.graphs
        if graphs is None:
            return None
        for i in range(len(graphs)):
            v = graphs.get(str(i))
            if v is None or v.persistent_tensors_size is None:
                return None
            ret.append(v.persistent_tensors_size)
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
        num_persistent_tensors = self.get_num_persistent_tensors(com)
        persistent_tensors_sizes = self.get_persistent_tensors_sizes(com)
        graph_indexes = self.get_graph_indexes(com)
        file_path = os.path.join(
            com.config.work_folder,
            com.config.name,
            com.config.model,
            f"{com.config.model}_{self.get_name()}_graphs_stats.csv",
        )
        header = (
            "Model,Index,Name,Num Persistent Tensors,Persistent Tensors Size [bytes]"
        )
        lines = [header]
        num_lines = len(persistent_tensors_sizes)
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
            persistent_tensors = (
                f"{num_persistent_tensors[i]}"
                if num_persistent_tensors and len(num_persistent_tensors) > i
                else ""
            )
            persistent_tensors_size = (
                f"{persistent_tensors_sizes[i]}"
                if persistent_tensors_sizes and len(persistent_tensors_sizes) > i
                else ""
            )
            lines.append(
                f"{model},{index},{graph_name},{persistent_tensors},{persistent_tensors_size}"
            )
        with open(file_path, "w") as f:
            f.write("\n".join(lines))
        return file_path

    def set_status(self, com: ComObj):
        if self.compile_available(com):
            self.set_success_status(com, [f"graphs stats: {self.get_graphs_stats(com)}"])
