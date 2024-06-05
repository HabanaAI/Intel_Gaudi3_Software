include(ExternalProject)


set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(UNIT_TEST_TARGET fuser_qa_tests)

file(GLOB UNIT_TEST_FILES *.cpp
    ../gaudi_tests/gaudi_test_infra.cpp
    ../utils/global_conf_test_setter.cpp
    ../utils/synapse_test.cpp
    ../utils/test_utils.cpp
    ../utils/device_memory_allocator.cpp
    ../utils/test_configuration.cpp
    )


set (UNIT_TEST_ALL_FILES ${UNIT_TEST_FILES} )

file(GLOB_RECURSE TESTS_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)

set_directory_properties(PROPERTIES EP_PREFIX ${CMAKE_BINARY_DIR}/ThirdParty)

ExternalProject_Get_Property(googletest binary_dir)

include_directories(${binary_dir}/googletest/include
                    ${binary_dir}/googlemock/include
                    ../../include
                    ../../src/include
                    ../../src/graph_compiler/
                    ../../src/graph_compiler/habana_nodes/

                    ../../src/runtime/common/
                    ../../src/runtime/common/recipe/
                    ../../src/runtime/common/recipe/patching/
                    ../../src/runtime/common/stream

                    ../../src/runtime/qman/gaudi/
                    ../../src/runtime/qman/gaudi/common/

                    ../../src/runtime/scal/common/
                    ../../src/runtime/scal/common/entities/

                    ../../src/platform/gaudi/graph_compiler/
                    ../../src/platform/gaudi2/graph_compiler/
                    ../../src/platform/gaudi3/graph_compiler/
                    ../

                    $ENV{SCAL_ROOT}/include
                    $ENV{ENGINES_FW_ROOT}/include
                    $ENV{SPECS_ROOT}
                    $ENV{SPECS_EXT_ROOT}
                    $ENV{HABANALABS_ROOT}/include/uapi
                    $ENV{HLTHUNK_ROOT}/include/uapi
                    $ENV{SPDLOG_ROOT}
                    $ENV{TPC_KERNELS_ROOT}/include
                    $ENV{SCAL_ROOT}
                    ../utils)
if (WIN32)
include_directories(../../src/graph_compiler/win32_workaround)
endif()

#set_target_properties(googletest PROPERTIES EXCLUDE_FROM_ALL TRUE)
ExternalProject_Get_Property(googletest binary_dir)
set(Suffix ".a")
set(Pthread "-pthread")
set (libPath "")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(libSuffix "d")
else()
    set(libSuffix "")
endif()

add_executable(${UNIT_TEST_TARGET} ${UNIT_TEST_ALL_FILES})

target_link_libraries(
    ${UNIT_TEST_TARGET}
    ${binary_dir}/lib/${libPath}${CMAKE_FIND_LIBRARY_PREFIXES}gtest${libSuffix}${Suffix}
    ${binary_dir}/lib/${libPath}${CMAKE_FIND_LIBRARY_PREFIXES}gmock${libSuffix}${Suffix}
    ${binary_dir}/lib/${libPath}${CMAKE_FIND_LIBRARY_PREFIXES}gtest_main${libSuffix}${Suffix}
    ${binary_dir}/lib/${libPath}${CMAKE_FIND_LIBRARY_PREFIXES}gmock_main${libSuffix}${Suffix}
    DataSerialize
    Synapse
    ${hl_thunk}
    ${scal}
    mme_reference fma
)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-narrowing -Werror -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pipe")

add_dependencies(${UNIT_TEST_TARGET} googletest)

if(CMAKE_COMPILER_IS_GNUC OR CMAKE_COMPILER_IS_GNUCXX)
    target_link_libraries(${UNIT_TEST_TARGET} pthread)
endif()

add_custom_target(fuser_qa_test COMMAND ${EXECUTABLE_OUTPUT_PATH}/${UNIT_TEST_TARGET} --gtest_output="xml:${CMAKE_BINARY_DIR}/tests.xml" || true)
add_dependencies(fuser_qa_test ${UNIT_TEST_TARGET})

if(COVERAGE_ENABLED)
    find_package(gcovr REQUIRED)
    target_link_libraries(${UNIT_TEST_TARGET} -lgcov)
    add_custom_target(coverage WORKING_DIRECTORY ${CMAKE_BINARY_DIR} COMMAND ${GCOVR_EXECUTABLE} -r ${CMAKE_SOURCE_PATH}/src --xml -o ${CMAKE_BINARY_DIR}/coverage.xml)
    add_dependencies(coverage check)
endif()



