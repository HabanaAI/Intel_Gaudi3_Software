cmake_minimum_required(VERSION 3.5.1)

project(gc_platform_tests LANGUAGES CXX)

set(SYN_ROOT $ENV{SYNAPSE_ROOT})

set(INCLUDE_PATH
    ${SYN_ROOT}/include
    ${SYN_ROOT}/tests
    ${SYN_ROOT}/src/infra
    $ENV{HABANA_SOFTWARE_STACK}/3rd-parties
)

file(GLOB SOURCE_PATH
    *.cpp
    **/*.cpp
    eager_tests/**/*.cpp
    gaudi_tests/**/*.cpp
    gaudi_tests/infra/**/*.cpp
    ${SYN_ROOT}/tests/json_tests/graph_loader.cpp
    ${SYN_ROOT}/tests/json_tests/file_loader.cpp
    ${SYN_ROOT}/tests/utils/*.cpp
)

list(REMOVE_ITEM SOURCE_PATH
    ${SYN_ROOT}/tests/utils/synapse_test.cpp
    ${SYN_ROOT}/tests/utils/synapse_test_common.cpp
    ${SYN_ROOT}/tests/utils/test_hello.cpp
)

add_executable(gc_platform_tests
    ${SOURCE_PATH}
)

add_dependencies(gc_platform_tests googletest)

target_link_libraries(gc_platform_tests
PRIVATE
    Synapse
    ${hl_logger}
    ${hl_gcfg}
    DataSerialize # for data provider
    ${GC_TESTS_COMMON_LIBS}
)

target_include_directories(gc_platform_tests
PRIVATE
    ${INCLUDE_PATH}
)

add_custom_target(generate_graphs_header
    COMMENT "Convert Jsons graphs to strings"
    COMMAND ${SYN_ROOT}/scripts/json_graphs_to_headers.py
    DEPENDS ${SYN_ROOT}/tests/gc_tests/platform_tests/graphs/*.json
    BYPRODUCTS ${SYN_ROOT}/synapse/tests/gc_tests/platform_tests/graphs.h
)

add_dependencies(gc_platform_tests generate_graphs_header)