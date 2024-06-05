include(ExternalProject)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(SCAL_TEST_TARGET scal_tests)
FILE(GLOB TARGET_H "${CMAKE_SOURCE_DIR}/tests/*.h")
FILE(GLOB TARGET_H2 "${CMAKE_SOURCE_DIR}/utils/*.h")
FILE(GLOB TARGET_CPP "${CMAKE_SOURCE_DIR}/tests/*.cpp")
FILE(GLOB TARGET_CPP2 "${CMAKE_SOURCE_DIR}/utils/*.cpp")
FILE(GLOB TARGET_RDMA "${CMAKE_SOURCE_DIR}/tests/rdma_core/*")
SET(TARGET_SRC ${TARGET_CPP} ${TARGET_CPP2} ${TARGET_H} ${TARGET_H2} ${TARGET_RDMA})

ExternalProject_Add(
    googletest
    URL "$ENV{HABANA_SOFTWARE_STACK}/3rd-parties/googletest_1_13/"
    TIMEOUT 10
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
               -Dgtest_force_shared_crt=ON -DCMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}
    INSTALL_COMMAND ""
    UPDATE_COMMAND ""
    LOG_DOWNLOAD ON
    LOG_CONFIGURE ON
    LOG_BUILD ON)

ExternalProject_Get_Property(googletest source_dir)
include_directories(SYSTEM $ENV{SPDLOG_ROOT})
include_directories(${source_dir}/googletest/include
                    ${source_dir}/googlemock/include
                    ../src/
                    ../src/graphs
                    ../src/scal_api
                    ../include/
                    ../utils/
                    $ENV{ENGINES_FW_ROOT}/include
                    $ENV{ENGINES_FW_ROOT}/include/gaudi3
                    $ENV{SPECS_ROOT}
                    $ENV{SPECS_ROOT_H9}
                    $ENV{SPECS_EXT_ROOT}
                    $ENV{HABANALABS_ROOT}/include/uapi
                    $ENV{HLTHUNK_ROOT}/include/uapi
)

set_target_properties(googletest PROPERTIES EXCLUDE_FROM_ALL TRUE)
ExternalProject_Get_Property(googletest binary_dir)
if(MSVC)
  set(Suffix ".lib")
  set (libPath ${CMAKE_BUILD_TYPE}"/")
else()
  set(Suffix ".a")
  set(Pthread "-pthread")
  set (libPath "")
endif()

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(libSuffix "d")
else()
    set(libSuffix "")
endif()

add_executable(${SCAL_TEST_TARGET} ${TARGET_SRC})

target_link_libraries(
    ${SCAL_TEST_TARGET}
    ${binary_dir}/lib/${libPath}${CMAKE_FIND_LIBRARY_PREFIXES}gtest${libSuffix}${Suffix}
    ${binary_dir}/lib/${libPath}${CMAKE_FIND_LIBRARY_PREFIXES}gmock${libSuffix}${Suffix}
    ${binary_dir}/lib/${libPath}${CMAKE_FIND_LIBRARY_PREFIXES}gtest_main${libSuffix}${Suffix}
    ${binary_dir}/lib/${libPath}${CMAKE_FIND_LIBRARY_PREFIXES}gmock_main${libSuffix}${Suffix}
    ${synapse}
    scal
    ${CMAKE_DL_LIBS}
)
add_dependencies(${SCAL_TEST_TARGET} googletest)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-narrowing -Werror -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pipe")

if(CMAKE_COMPILER_IS_GNUC OR CMAKE_COMPILER_IS_GNUCXX)
    target_link_libraries(${SCAL_TEST_TARGET} pthread)
endif()

add_custom_target(unit_check COMMAND ${EXECUTABLE_OUTPUT_PATH}/${SCAL_TEST_TARGET} --gtest_output="xml:${CMAKE_BINARY_DIR}/tests.xml" || true)
add_dependencies(unit_check ${SCAL_TEST_TARGET})

if(COVERAGE_ENABLED)
    find_package(gcovr REQUIRED)
    target_link_libraries(${SCAL_TEST_TARGET} -lgcov)
    add_custom_target(coverage WORKING_DIRECTORY ${CMAKE_BINARY_DIR} COMMAND ${GCOVR_EXECUTABLE} -r ${CMAKE_SOURCE_PATH}/src --xml -o ${CMAKE_BINARY_DIR}/coverage.xml)
    add_dependencies(coverage check)
endif()
