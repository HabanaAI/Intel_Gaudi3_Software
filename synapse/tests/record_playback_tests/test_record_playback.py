"""Synapse record and playback test suite"""

import os
import logging
from typing import Optional
import pytest

from _pytest.fixtures import FixtureRequest

from record_playback.recorder import SynApp, SynRecording
from record_playback.utils import config_logger, convert_command_line_to_dict
from record_playback.config import (
    DEVICE,
    SYNREC_PATH,
    JSON_TESTS_PATH,
    MODEL_TEST_FILE_PATH,
    SYNREC_PARAMS,
    PATH_ARG,
    TEST_ARG,
    OVERWRITE_ARG,
    TENSORS_DATA_ARG,
    SPLIT_ARG,
    SYN_APP,
    SYN_RECORDINGS,
    SYNREC_NAME,
    SYNREC_NAME_1,
    SYNREC_NAME_2,
    COMPRESSION_ARG,
    LZ4,
    ITERATION_NUM,
    MIN_ITER_ARG,
    MAX_ITER_ARG,
    METADATA_ARG,
    GRAPH_STATES_ARG,
    PRE,
    POST
)

LOG = logging.getLogger("synrec-tests")

config_logger(LOG)


JSON_SYN_APP_BASIC_ARGS = {
        "release": "",
        "run": "",
        "chip_type": DEVICE,
        "json_files": MODEL_TEST_FILE_PATH,
        "skip_models_download": "",
}

JSON_SYN_APP_SYNTHETIC_DATA_ARG = {
    "synthetic_data": ""
}

JSON_SYN_APP_ITERATIONS_ARG = {
    "iterations": ITERATION_NUM
}

JSON_SYN_APP_BASIC = SynApp(
    JSON_TESTS_PATH,
    JSON_SYN_APP_BASIC_ARGS,
)

JSON_SYN_APP_WITH_SYNTHETIC_DATA = SynApp(
    JSON_TESTS_PATH,
    {
        **JSON_SYN_APP_BASIC_ARGS,
        **JSON_SYN_APP_SYNTHETIC_DATA_ARG
    }
)

JSON_SYN_APP_WITH_ITERATION = SynApp(
    JSON_TESTS_PATH,
    {
        **JSON_SYN_APP_BASIC_ARGS,
        **JSON_SYN_APP_SYNTHETIC_DATA_ARG,
        **JSON_SYN_APP_ITERATIONS_ARG
    }
)


def _create_syn_recording(tmp_path: str,
                          recording_cmd: str,
                          syn_app: SynApp,
                          recording_name: Optional[str] = SYNREC_NAME_1) -> SynRecording:
    """
    Create SynRecording object

    Input params:
    tmp_path - pytest temp dir in file system
    recording_cmd - str of the recording atgs
    syn_app - syn app to record
    recording_name - optional recording name. by default its value is SYNREC_NAME_1

    Returns SynRecording object
    """
    recording_params = convert_command_line_to_dict(recording_cmd)
    if PATH_ARG in recording_params:
        recording_params[PATH_ARG] = os.path.join(tmp_path, recording_params[PATH_ARG])
    else:
        recording_params[PATH_ARG] = os.path.join(tmp_path, recording_name)
    recording_params[TEST_ARG] = ''
    return SynRecording(
        recording_name,
        SYNREC_PATH,
        recording_params,
        syn_app,
    )

@pytest.fixture
def one_syn_recording(tmp_path: str, request: FixtureRequest) -> SynRecording:
    """
    Prepare syn recording object with recording of synrec app

    Input params:
    request.param is a dictionary with those keys/values:
    SYNREC_PARAMS -   string of the command for example '--path recording_output --test --overwrite'.
    SYN_APP (OPtional) - synapp to record. if not included JSON_SYN_APP_BASIC is used

    Returns SynRecording object
    """
    recording_cmd = request.param.get(SYNREC_PARAMS)
    syn_app = request.param.get(SYN_APP, JSON_SYN_APP_BASIC)
    return _create_syn_recording(tmp_path, recording_cmd, syn_app)


@pytest.fixture
def syn_recordings(tmp_path: str, request: FixtureRequest) -> dict:
    """
    Prepare multiple syn recordings objects
    Input params:
    request.param is a dictionary with those keys/values:
    SYN_RECORDINGS - list of dictionaries {SYNREC_NAME: <recording name>,
                                           SYNREC_PARAMS: <strings of args>,
                                           SYN_APP: <optional app to record>}
    Returns - dictionary  {recording_name: SynRecording}
    """
    recordings_input: list = request.param.get(SYN_RECORDINGS)
    recordings = {}

    for recording_input in recordings_input:
        recording_name = recording_input[SYNREC_NAME]
        recording_cmd = recording_input[SYNREC_PARAMS]
        syn_app = JSON_SYN_APP_BASIC if SYN_APP not in recording_input else recording_input[SYN_APP]
        recordings[recording_name] = _create_syn_recording(tmp_path,
                                                           recording_cmd=recording_cmd,
                                                           syn_app=syn_app,
                                                           recording_name=recording_name)
    return recordings

@pytest.mark.parametrize(
    "one_syn_recording",
    [
        ({SYNREC_PARAMS: ""}),
        ({SYNREC_PARAMS: f"--{OVERWRITE_ARG}"})
    ],
    indirect=["one_syn_recording"],
)
def test_graphs_recording(one_syn_recording):
    """
    basic record test - graph
    verify the creation and content of output files
    """
    expected_record_path = f"{one_syn_recording.args[PATH_ARG]}.json"

    assert one_syn_recording.rc == 0, one_syn_recording.prints
    assert expected_record_path == one_syn_recording.recorded_graph_path
    assert (
        one_syn_recording.is_recorded_graph_exists()
    ), f"failed to record file: {expected_record_path}"
    assert one_syn_recording.diff_recorded_graph(MODEL_TEST_FILE_PATH)
    assert one_syn_recording.recorded_tensor_path is None


@pytest.mark.parametrize(
    "one_syn_recording",
    [
        ({SYNREC_PARAMS: f"--{TENSORS_DATA_ARG}"}),
        ({SYNREC_PARAMS: f"--{TENSORS_DATA_ARG} --{OVERWRITE_ARG}"})
    ],
    indirect=["one_syn_recording"],
)
def test_tensor_recording(one_syn_recording):
    """
    basic record test - tensor
    verify the creation of output files
    """

    expected_record_path = f"{one_syn_recording.args[PATH_ARG]}.json"
    expected_tensor_path = f"{one_syn_recording.args[PATH_ARG]}.db"


    assert one_syn_recording.rc == 0, one_syn_recording.prints
    assert expected_record_path == one_syn_recording.recorded_graph_path
    assert expected_tensor_path == one_syn_recording.recorded_tensor_path


@pytest.mark.parametrize(
    "one_syn_recording, expected_graph_count, expected_tensor_count",
    [
        ({SYNREC_PARAMS: f"--{TENSORS_DATA_ARG} --{SPLIT_ARG}"},2,2,),
        ({SYNREC_PARAMS: f"--{TENSORS_DATA_ARG} --{SPLIT_ARG} --{OVERWRITE_ARG}"},2,2,)
    ],
    indirect=["one_syn_recording"],
)
def test_split_recording(one_syn_recording, expected_graph_count, expected_tensor_count):
    """
    Test split.
    Verify that the corrent ampount of files are created
    """
    expected_record_path = one_syn_recording.args[PATH_ARG]

    assert one_syn_recording.rc == 0, one_syn_recording.prints
    assert (
        expected_record_path
        == one_syn_recording.recorded_graph_path
        == one_syn_recording.recorded_tensor_path
    ),f"pathes are not the same. {one_syn_recording.recorded_graph_path}, {one_syn_recording.recorded_tensor_path}"
    assert (one_syn_recording.is_output_dir),f"output is not a directory,{one_syn_recording.recorded_graph_path}"
    graph_count, tensor_count = one_syn_recording.get_recording_dir_content()
    assert (graph_count == expected_graph_count),f"wrong graph files number:{graph_count}, expected:{expected_graph_count}"
    assert (tensor_count == expected_tensor_count),f"wrong tensor files number:{tensor_count}, expected:{expected_tensor_count}"

@pytest.mark.parametrize(
    "one_syn_recording",
    [
        ({SYNREC_PARAMS: ""}),
        ({SYNREC_PARAMS: f"--{OVERWRITE_ARG}"})
    ],
    indirect=["one_syn_recording"],
)
def test_overwrite(one_syn_recording):
    """
    Test overwrite.
    record twice and verify that:
    1. second record fails if no overwrite
    2. second record successed if overwrite arg exists
    """
    assert one_syn_recording.rc == 0, one_syn_recording.prints

    one_syn_recording.run()

    if OVERWRITE_ARG not in one_syn_recording.args:
        assert (one_syn_recording.rc != 0),"synrec should fail, directory exists, no overwrite"
    else:
        assert (one_syn_recording.rc == 0),"synrec should not fail, overwrite arg was given"


@pytest.mark.skip(reason="Need to check compression feature")
@pytest.mark.parametrize(
    "syn_recordings",
    [
        (
            {
                SYN_RECORDINGS: [
                    {
                        SYNREC_NAME: SYNREC_NAME_1,
                        SYNREC_PARAMS: f"--{TENSORS_DATA_ARG} --{COMPRESSION_ARG} {LZ4}",
                        SYN_APP: JSON_SYN_APP_WITH_SYNTHETIC_DATA,
                    },
                    {
                        SYNREC_NAME: SYNREC_NAME_2,
                        SYNREC_PARAMS: f"--{TENSORS_DATA_ARG}",
                        SYN_APP: JSON_SYN_APP_WITH_SYNTHETIC_DATA,
                    },
                ]
            }
        ),
    ],
    indirect=["syn_recordings"],
)
def test_compression(syn_recordings):
    """
    Test compression flag - record with and without compression and check the size of the tensors files
    """
    assert syn_recordings[SYNREC_NAME_1].rc == 0, syn_recordings[SYNREC_NAME_1].prints
    assert syn_recordings[SYNREC_NAME_2].rc == 0, syn_recordings[SYNREC_NAME_2].prints

    # TODO check why same size
    tensors_file_size1 = os.path.getsize(syn_recordings[SYNREC_NAME_1].recorded_tensor_path)
    tensors_file_size2 = os.path.getsize(syn_recordings[SYNREC_NAME_2].recorded_tensor_path)


@pytest.mark.skip(reason="TODO check db files content")
@pytest.mark.parametrize(
    "syn_recordings",
    [
        (
            {
                SYN_RECORDINGS: [
                    {
                        SYNREC_NAME: SYNREC_NAME_1,
                        SYNREC_PARAMS: f"--{TENSORS_DATA_ARG}",
                        SYN_APP: JSON_SYN_APP_WITH_ITERATION,
                    },
                    {
                        SYNREC_NAME: SYNREC_NAME_2,
                        SYNREC_PARAMS: f"--{TENSORS_DATA_ARG} --{MIN_ITER_ARG} 2 --{MAX_ITER_ARG} 3",
                        SYN_APP: JSON_SYN_APP_WITH_ITERATION,
                    },
                ]
            }
        ),
    ],
    indirect=["syn_recordings"],
)
def test_iteration(syn_recordings):

    assert syn_recordings[SYNREC_NAME_1].rc == 0, syn_recordings[SYNREC_NAME_1].prints
    assert syn_recordings[SYNREC_NAME_2].rc == 0, syn_recordings[SYNREC_NAME_2].prints

    tensors_file_size1 = os.path.getsize(syn_recordings[SYNREC_NAME_1].recorded_tensor_path)
    tensors_file_size2 = os.path.getsize(syn_recordings[SYNREC_NAME_2].recorded_tensor_path)

    # TODO
    # assert tensors_file_size2 < tensors_file_size1

@pytest.mark.parametrize(
    "syn_recordings",
    [
        (
            {
                SYN_RECORDINGS: [
                    {
                        SYNREC_NAME: SYNREC_NAME_1,
                        SYNREC_PARAMS: f"--{TENSORS_DATA_ARG}",
                        SYN_APP: JSON_SYN_APP_WITH_ITERATION,
                    },
                    {
                        SYNREC_NAME: SYNREC_NAME_2,
                        SYNREC_PARAMS: f"--{TENSORS_DATA_ARG} --{METADATA_ARG}",
                        SYN_APP: JSON_SYN_APP_WITH_ITERATION,
                    },
                ]
            }
        ),
    ],
    indirect=["syn_recordings"],
)
def test_metadata(syn_recordings):
    assert syn_recordings[SYNREC_NAME_1].rc == 0, syn_recordings[SYNREC_NAME_1].prints
    assert syn_recordings[SYNREC_NAME_2].rc == 0, syn_recordings[SYNREC_NAME_2].prints

    tensors_file_size1 = os.path.getsize(syn_recordings[SYNREC_NAME_1].recorded_tensor_path)
    tensors_file_size2 = os.path.getsize(syn_recordings[SYNREC_NAME_2].recorded_tensor_path)

    assert (tensors_file_size2 < tensors_file_size1),"Tensor metadata file size is smalled than regular tensor file"

@pytest.mark.parametrize(
    "syn_recordings",
    [
        (
            {
                SYN_RECORDINGS: [
                    {
                        SYNREC_NAME: SYNREC_NAME_1,
                        SYNREC_PARAMS: f"--{SPLIT_ARG} --{GRAPH_STATES_ARG} {PRE}"
                    },
                    {
                        SYNREC_NAME: SYNREC_NAME_2,
                        SYNREC_PARAMS: f"--{SPLIT_ARG} --{GRAPH_STATES_ARG} {POST}"
                    },
                ]
            }
        ),
    ],
    indirect=["syn_recordings"],
)
def test_graph_states(syn_recordings):

    assert syn_recordings[SYNREC_NAME_1].rc == 0, syn_recordings[SYNREC_NAME_1].prints
    assert syn_recordings[SYNREC_NAME_2].rc == 0, syn_recordings[SYNREC_NAME_2].prints

    pre_graph_count_first, post_graph_count_first = syn_recordings[SYNREC_NAME_1].get_recording_dir_graphs_content()
    pre_graph_count_sec, post_graph_count_sec = syn_recordings[SYNREC_NAME_2].get_recording_dir_graphs_content()
    assert pre_graph_count_first is not None
    assert post_graph_count_first is not None
    assert pre_graph_count_sec is not None
    assert post_graph_count_sec is not None
    assert post_graph_count_first == 0
    assert pre_graph_count_sec == 0
    assert pre_graph_count_first == post_graph_count_sec
