import json
import logging
import re
import os
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass, field


from record_playback.utils import config_logger, exe


LOG = logging.getLogger("recorder")

# patterns for synrec log
GRAPH_PATH_PATTERN = ".*Record graph path: (.*)$"
TENSOR_PATH_PATTERN = ".*Record tensors path: (.*)$"

config_logger(LOG)


@dataclass
class SynApp:
    """
    Represents APP that can be recorded by synrec
    Parameters:
    exe_file_path - execution file's path, or command line if app is bash
    args - optional, dictionary of args {arg_name (without '--'): arg_value (can be empty string)}
    """
    exe_file_path: str
    args: Optional[dict]
    cmd: str = field(init=False)

    def __post_init__(self):
        arg_list = (
            [f"--{key} {value}" for key, value in self.args.items()]
            if self.args
            else ""
        )
        self.cmd = f"{self.exe_file_path} {' '.join(arg_list)}"

    def run(self):
        """execute app without recording"""
        return exe(self.cmd, LOG)


@dataclass
class SynRecording:
    """
    Represents syn recording
    Parameters:
    exe_file_path - synrec execution file's path
    args - optional, dictionary of args {arg_name (without '--'): arg_value (can be empty string)}
    syn_app - app to record
    """
    name: str
    exe_file_path: str
    args: dict
    syn_app: SynApp
    cmd: str = field(init=False)
    rc: int = field(init=False, default=None)
    prints: List[str] = field(init=False, default=None)
    recorded_graph_path: str = field(init=False, default=None)
    recorded_tensor_path: str = field(init=False, default=None)
    is_output_dir: bool = field(init=False, default=None)

    def __post_init__(self):
        arg_list = [f"--{key} {value}" for key, value in self.args.items()]
        self.cmd = f"{self.exe_file_path} {' '.join(arg_list)} -- {self.syn_app.cmd}"
        self.run()
        self._calc_is_output_dir()


    def _parse_prints(self, prints):
        """
        Parse synrrec execution log
        """
        self.recorded_tensor_path = self.recorded_graph_path = None
        for line in reversed(prints):
            match = re.match(GRAPH_PATH_PATTERN, line)
            if match:
                # path of output graph
                self.recorded_graph_path = match.group(1)
            else:
                match = re.match(TENSOR_PATH_PATTERN, line)
                if match:
                    # path of output tensor
                    self.recorded_tensor_path = match.group(1)
                elif self.recorded_graph_path and self.recorded_tensor_path:
                    break


    def _calc_is_output_dir(self):
        """
        returns True if output path is a directory (i.e. it is a splitted recorfing)
        """
        self.is_output_dir = False
        if self.is_recorded_graph_exists() and os.path.isdir(self.recorded_graph_path):
                self.is_output_dir = True


    def get_graphs_names(self, recorded_graph_path: str = None) -> list:
        """
        Get graphs name. from output of this recording or from output file of other recording
        The function can handle splitted records or regular recordings

        Input params:
        recorded_graph_path - path of file to analyze, if it's None then analyzion is done on self.recorded_graph_path

        Output params:
        list of graphs names
        """
        graph_names = []
        is_exist = os.path.exists(recorded_graph_path) if recorded_graph_path else self.is_recorded_graph_exists()
        is_dir = os.path.isdir(recorded_graph_path) if recorded_graph_path else self.is_output_dir
        print(f"yoav {self.is_output_dir} {is_dir}")
        if is_exist:
            real_graph_path = recorded_graph_path if recorded_graph_path else self.recorded_graph_path
            if is_dir:
                for filename in os.listdir(real_graph_path):
                    if filename.endswith('json'):
                        graph_names.append(filename.rsplit('.'))
            else:
                with open(real_graph_path) as file:
                    json_content = json.load(file)
                    graph_names = [g.get("name") for g in json_content.get("graphs")]
        return graph_names


    def is_recorded_graph_exists(self) -> bool:
        """
        Indicates if synrec execution created output graph
        """
        if self.recorded_graph_path and os.path.exists(self.recorded_graph_path):
            return True
        return False

    def is_recorded_tensor_exists(self) -> bool:
        """
        Indicates if synrec execution created output tensor
        """
        if self.recorded_tensor_path and os.path.exists(self.recorded_tensor_path):
            return True
        return False


    def get_recording_dir_content(self) -> tuple:
        """
        Returns numer of output files of splitted rercording

        Output params:
        graph_count - number of graph json files
        tensor_count - number of tensor db files
        """
        graph_count, tensor_count = None, None
        if self.recorded_graph_path:
            path = Path(self.recorded_graph_path)
            if path.is_dir():
                graph_count = sum(1 for file in path.glob("*.json"))
                tensor_count = sum(1 for file in path.glob("*.db"))
        return graph_count, tensor_count


    def get_recording_dir_graphs_content(self):
        """
        Returns number of graphs output files of splitted rercording.

        Output params:
        pre_graph_count - number of pre graphs files
        post_graph_count - number of post graph files
        """
        pre_graph_count, post_graph_count = None, None
        print(self.recorded_graph_path)
        if self.recorded_graph_path:
            path = Path(self.recorded_graph_path)
            if path.is_dir():
                json_files = list(path.glob('*.json'))
                pre_graph_count = sum(1 for file in json_files if not file.stem.endswith('post') )
                post_graph_count = sum(1 for file in json_files if file.stem.endswith('post')
                                       and file.name.count('.') == 2)
        return pre_graph_count, post_graph_count


    def diff_recorded_graph(self, other: str) -> bool:
        """
        Compare output graph with other graph
        """
        is_equal = False
        my_graph_names = self.get_graphs_names()
        other_graph_names = self.get_graphs_names(other)
        if my_graph_names and other_graph_names and my_graph_names == other_graph_names:
            is_equal = True
        return is_equal

    def run(self):
        """
        Execute synrec

        Returns:
        rc - return code of the execution
        prints - log of the execution
        """
        LOG.info(f"executing {self.cmd}")
        self.rc, self.prints = exe(self.cmd, LOG)
        self._parse_prints(self.prints)
