cmake_minimum_required(VERSION 3.5.1)

project(syn_tests LANGUAGES CXX)

set(SYN_ROOT $ENV{SYNAPSE_ROOT})

include(ExternalProject)

ExternalProject_Get_Property(googletest source_dir)
ExternalProject_Get_Property(googletest binary_dir)

set(GTEST_LIBS
        ${binary_dir}/lib/${libPath}${CMAKE_FIND_LIBRARY_PREFIXES}gtest${libSuffix}${Suffix}
        ${binary_dir}/lib/${libPath}${CMAKE_FIND_LIBRARY_PREFIXES}gmock${libSuffix}${Suffix}
        ${binary_dir}/lib/${libPath}${CMAKE_FIND_LIBRARY_PREFIXES}gtest_main${libSuffix}${Suffix}
        ${binary_dir}/lib/${libPath}${CMAKE_FIND_LIBRARY_PREFIXES}gmock_main${libSuffix}${Suffix}
)

set(INCLUDE_PATH
    ${SYN_ROOT}/include
    ${SYN_ROOT}/tests/runtime_tests/infra
    ${SYN_ROOT}/tests/runtime_tests/recipes
    ${SYN_ROOT}/tests/runtime_tests/tests
    ${SYN_ROOT}/tests/runtime_tests/utils
    ${SYN_ROOT}/tests/utils

    $ENV{THIRD_PARTIES_ROOT}
)

file(GLOB SOURCE_PATH
    *.cpp
    */*.c
    */*.cpp
)

list(REMOVE_ITEM SOURCE_PATH
    ${SYN_ROOT}/tests/utils/test_hello.cpp
)

add_executable(syn_tests
    ${SOURCE_PATH}
)

# Link OldLibrary sources directly to NewExecutable
target_sources(syn_tests PRIVATE
               ${SYN_ROOT}/tests/utils/cpu_calculator.cpp
               ${SYN_ROOT}/tests/utils/test_utils.cpp
               ${SYN_ROOT}/tests/utils/test_dummy_recipe.cpp
               ${SYN_ROOT}/tests/utils/global_conf_test_setter.cpp)

add_dependencies(syn_tests googletest)

target_link_libraries(syn_tests
PRIVATE
    Synapse
    mme_reference
    ${hl_logger}
    ${hl_gcfg}
    ${GTEST_LIBS}
)

target_include_directories(syn_tests
PRIVATE
    ${INCLUDE_PATH}
    ${SYN_ROOT}/tests/utils/
)
