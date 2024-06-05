import sys
import os
import shutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
from models_tests_analyzer import ModelsTestsAnalyzer
from config import RUN_1_DIR, RUN_2_DIR, TRACE_DIR, OUTPUT_SUFFIX_RUN_1, OUTPUT_SUFFIX_RUN_2

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),'data')
hw_csv_file_name = 'debug_hw_prof_accel0_001.csv'
nodes_csv_name = 'debug_hw_prof_accel0_001_analyzed_nodes.csv'
model = 'debug'

def prepare_fs(tmp_path):
    trace_dir_1 = os.path.join(tmp_path,RUN_1_DIR, model, TRACE_DIR)
    trace_dir_2 = os.path.join(tmp_path,RUN_2_DIR, model, TRACE_DIR)

    os.makedirs(trace_dir_1)
    os.makedirs(trace_dir_2)
    shutil.copyfile(os.path.join(data_dir, hw_csv_file_name), os.path.join(trace_dir_1, hw_csv_file_name))
    shutil.copyfile(os.path.join(data_dir, nodes_csv_name), os.path.join(trace_dir_1, nodes_csv_name))
    shutil.copyfile(os.path.join(data_dir, hw_csv_file_name), os.path.join(trace_dir_2, hw_csv_file_name))
    shutil.copyfile(os.path.join(data_dir, nodes_csv_name), os.path.join(trace_dir_2, nodes_csv_name))
    return trace_dir_1, trace_dir_2

def test_main_flow(tmp_path: str):
    trace_dir_1, trace_dir_2 = prepare_fs(tmp_path)
    expected_generated_file_names = ['trace_analyser_debug_1_f.csv','trace_analyser_debug_1_t.csv']
    generated_file_pathes = [os.path.join(trace_dir_1, expected_generated_file_names[0]),
                             os.path.join(trace_dir_2, expected_generated_file_names[1])]

    ModelsTestsAnalyzer(root_dir=str(tmp_path),
                        overwrite_analyze=True,
                        overwrite_compare=True,
                        model=None,
                        graph_index=None,
                        nodes=None,
                        output_dir=None)
    for p in generated_file_pathes:
        assert  os.path.exists(p)