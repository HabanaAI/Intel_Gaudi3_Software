cmake_minimum_required(VERSION 3.5.1)

project(json_tests LANGUAGES CXX)

add_executable(json_tests
    base_test.cpp
    json_tests.cpp
    playback_tests.cpp
    db_parser.cpp
    performance_base_test.cpp
    performance_playback_tests.cpp
    multi_threaded_playback_tests.cpp
    model_tests.cpp
    recipe_runner.cpp
    graph_loader.cpp
    model_loader.cpp
    consistency_tests.cpp
    config_compare_tests.cpp
    file_loader.cpp
    $ENV{SYNAPSE_ROOT}/tests/utils/launcher.cpp
)

target_link_libraries(json_tests PRIVATE Synapse DataSerialize)

target_include_directories(json_tests
    PRIVATE
    $ENV{HABANALABS_ROOT}/include/uapi
    $ENV{HABANA_SOFTWARE_STACK}/3rd-parties
    $ENV{HL_LOGGER_INCLUDE_DIRS}
    $ENV{SCAL_ROOT}/include
    $ENV{SWTOOLS_SDK_ROOT}/hl_gcfg/include
    $ENV{SYNAPSE_ROOT}/include
    $ENV{SYNAPSE_ROOT}/src/graph_compiler/habana_nodes
    $ENV{SYNAPSE_ROOT}/src/include
    $ENV{SYNAPSE_ROOT}/src/infra
    $ENV{SYNAPSE_ROOT}/src/runtime/common
    $ENV{SYNAPSE_ROOT}/tests
)
