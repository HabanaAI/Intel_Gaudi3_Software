set(INFRA_TEST_TARGET infra_unit_tests)

include(ExternalProject)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

file(GLOB_RECURSE INFRA_TESTS_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)
set(INFRA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../src/infra)
set(INFRA_TESTS_FILES ${INFRA_TESTS_FILES}
                      ${INFRA_DIR}/../graph_compiler/habana_global_conf.cpp)

ExternalProject_Get_Property(googletest source_dir)
include_directories(
    ${source_dir}/googletest/include
    ../utils)

ExternalProject_Get_Property(googletest binary_dir)
set(Suffix ".a")
set(Pthread "-pthread")
set (libPath "")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(libSuffix "d")
else()
    set(libSuffix "")
endif()

add_executable(${INFRA_TEST_TARGET} ${INFRA_TESTS_FILES})

target_link_libraries(${INFRA_TEST_TARGET}
    ${binary_dir}/lib/${libPath}${CMAKE_FIND_LIBRARY_PREFIXES}gtest${libSuffix}${Suffix}
    ${binary_dir}/lib/${libPath}${CMAKE_FIND_LIBRARY_PREFIXES}gtest_main${libSuffix}${Suffix}
    ${hl_logger}
    ${hl_gcfg}
)

if(VTUNE_PROFILER)
    link_directories($ENV{VTUNE_INSTALL_DIR}/sdk/lib64)
    target_link_libraries(${INFRA_TEST_TARGET} libittnotify.a)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-narrowing -Werror -Wall -rdynamic")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pipe")

add_dependencies(${INFRA_TEST_TARGET} googletest)

if(CMAKE_COMPILER_IS_GNUC OR CMAKE_COMPILER_IS_GNUCXX)
    target_link_libraries(${INFRA_TEST_TARGET} pthread)
endif()

#add_custom_target(infra_check COMMAND ${EXECUTABLE_OUTPUT_PATH}/${INFRA_TEST_TARGET} --gtest_output="xml:${CMAKE_BINARY_DIR}/tests.xml" || true)
#add_dependencies(infra_check ${INFRA_TEST_TARGET})

if(COVERAGE_ENABLED)
    find_package(gcovr REQUIRED)
    target_link_libraries(${INFRA_TEST_TARGET} -lgcov)
    add_custom_target(coverage WORKING_DIRECTORY ${CMAKE_BINARY_DIR} COMMAND ${GCOVR_EXECUTABLE} -r ${CMAKE_SOURCE_PATH}/src --xml -o ${CMAKE_BINARY_DIR}/coverage.xml)
    add_dependencies(coverage check)
endif()
