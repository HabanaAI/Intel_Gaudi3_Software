from os.path import join, abspath
from os import environ

DEVICE = 'gaudi'
SYN_ROOT = environ.get("SYNAPSE_ROOT")
SCRIPTS_ROOT = join(SYN_ROOT,'scripts')
SYNREC_PATH: str = join(SCRIPTS_ROOT,'synrec.py')
JSON_TESTS_PATH: str = join(SCRIPTS_ROOT,'json_runner.py')
MODEL_TEST_FILE_PATH = join(SYN_ROOT,'tests','record_playback_tests','data','test_model.json')
RECORDING_OUTPUT = 'recording_output'
SYNREC_PARAMS = 'synrec_params'
PATH_ARG = 'path'
TEST_ARG = 'test'
OVERWRITE_ARG = 'overwrite'
TENSORS_DATA_ARG = 'tensors-data'
SPLIT_ARG = 'split'
COMPRESSION_ARG = 'compression'
SYN_APP = 'syn_app'
LZ4 = 'lz4'
SYN_RECORDINGS = 'syn_recordings'
SYNREC_NAME = 'synrec_name'
SYNREC_NAME_1 = 'synrec_name_1'
SYNREC_NAME_2 = 'synrec_name_2'
ITERATION_NUM = 5
MIN_ITER_ARG = 'min-iter'
MAX_ITER_ARG = 'max-iter'
METADATA_ARG = 'metadata'
GRAPH_STATES_ARG = 'graph-states'
PRE = 'pre'
POST = 'post'