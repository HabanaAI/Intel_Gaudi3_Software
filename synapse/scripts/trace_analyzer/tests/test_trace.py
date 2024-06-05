import sys
import os
import shutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
from trace_analyzer import Trace

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),'data')
hw_csv_file_name = 'debug_hw_prof_accel0_001.csv'
nodes_csv_name = 'debug_hw_prof_accel0_001_analyzed_nodes.csv'

def prepare_fs(tmp_path):
    shutil.copyfile(os.path.join(data_dir, hw_csv_file_name), os.path.join(tmp_path, hw_csv_file_name))
    shutil.copyfile(os.path.join(data_dir, nodes_csv_name), os.path.join(tmp_path, nodes_csv_name))


def test_main_flow(tmp_path: str):
    prepare_fs(tmp_path)
    expected_generated_file_name = 'trace_analyser_debug_1.csv'
    generated_file_path = os.path.join(tmp_path, expected_generated_file_name)
    hw_csv_path = os.path.join(tmp_path, hw_csv_file_name)
    nodes_csv_path = os.path.join(tmp_path, nodes_csv_name)
    t = Trace(model='debug',
              graph_index=1,
              output_dir=tmp_path,
              log_file_path=None,
              nodes_csv_path=nodes_csv_path,
              hw_events_csv_path=hw_csv_path,
              hltv_path=None,
              overwrite=True)
    assert os.path.exists(generated_file_path)