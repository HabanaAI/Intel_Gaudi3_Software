set(RUNTIME_TEST_TARGET runtime_unit_tests)

include(ExternalProject)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

file(GLOB RUNTIME_TEST_FILES
    *.cpp
    ../utils/*.cpp
    infra_tests/containers/*.cpp
    infra_tests/event_dispatcher_tests.cpp)

file(GLOB RUNTIME_PLATFORM_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/common/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/gaudi/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/gaudi2/*.cpp")

set (RUNTIME_ALL_FILES ${RUNTIME_TEST_FILES} ${RUNTIME_PLATFORM_FILES})

file(GLOB_RECURSE TESTS_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)

set_directory_properties(PROPERTIES EP_PREFIX ${CMAKE_BINARY_DIR}/ThirdParty)

ExternalProject_Get_Property(googletest source_dir)
include_directories(
    ${source_dir}/googletest/include
    ../utils
    ../
    ../src/runtime/common/)

ExternalProject_Get_Property(googletest binary_dir)
set(Suffix ".a")
set(Pthread "-pthread")
set (libPath "")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(libSuffix "d")
else()
    set(libSuffix "")
endif()

add_executable(${RUNTIME_TEST_TARGET} ${RUNTIME_ALL_FILES})

target_link_libraries(${RUNTIME_TEST_TARGET}
    ${binary_dir}/lib/${libPath}${CMAKE_FIND_LIBRARY_PREFIXES}gtest${libSuffix}${Suffix}
    ${binary_dir}/lib/${libPath}${CMAKE_FIND_LIBRARY_PREFIXES}gmock${libSuffix}${Suffix}
    ${binary_dir}/lib/${libPath}${CMAKE_FIND_LIBRARY_PREFIXES}gtest_main${libSuffix}${Suffix}
    ${binary_dir}/lib/${libPath}${CMAKE_FIND_LIBRARY_PREFIXES}gmock_main${libSuffix}${Suffix}
    Synapse
    mme_reference fma
)

if(VTUNE_PROFILER)
    link_directories($ENV{VTUNE_INSTALL_DIR}/sdk/lib64)
    target_link_libraries(${RUNTIME_TEST_TARGET} libittnotify.a)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-narrowing -Werror -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pipe")

add_dependencies(${RUNTIME_TEST_TARGET} googletest)

if(CMAKE_COMPILER_IS_GNUC OR CMAKE_COMPILER_IS_GNUCXX)
    target_link_libraries(${RUNTIME_TEST_TARGET} pthread)
endif()

add_custom_target(runtime_unit_check COMMAND ${EXECUTABLE_OUTPUT_PATH}/${RUNTIME_TEST_TARGET} --gtest_output="xml:${CMAKE_BINARY_DIR}/tests.xml" || true)
add_dependencies(runtime_unit_check ${RUNTIME_TEST_TARGET})

if(COVERAGE_ENABLED)
    find_package(gcovr REQUIRED)
    target_link_libraries(${RUNTIME_TEST_TARGET} -lgcov)
    add_custom_target(coverage WORKING_DIRECTORY ${CMAKE_BINARY_DIR} COMMAND ${GCOVR_EXECUTABLE} -r ${CMAKE_SOURCE_PATH}/src --xml -o ${CMAKE_BINARY_DIR}/coverage.xml)
    add_dependencies(coverage check)
endif()
