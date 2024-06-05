set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(UNIT_TEST_TARGET GraphCompiler_tests)

file(GLOB UNIT_TEST_FILES ../../utils/test_utils.cpp *.cpp)

file(GLOB UNIT_TEST_PLATFORM_FILES "${CMAKE_CURRENT_SOURCE_DIR}/dali/*.cpp"
                                   "${CMAKE_CURRENT_SOURCE_DIR}/gaudi/*.cpp"
                                   "${CMAKE_CURRENT_SOURCE_DIR}/gaudi2/*.cpp"
                                   "${CMAKE_CURRENT_SOURCE_DIR}/gaudi3/*.cpp"
                                   ../../../src/runtime/common/recipe/recipe_allocator.cpp
                                   ../../../src/runtime/common/section_handle.cpp
                                   ../../../src/runtime/qman/gaudi/master_qmans_definition.cpp)

set (UNIT_TEST_ALL_FILES ${UNIT_TEST_FILES} ${UNIT_TEST_PLATFORM_FILES})

file(GLOB_RECURSE TESTS_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)

set_directory_properties(PROPERTIES EP_PREFIX ${CMAKE_BINARY_DIR}/ThirdParty)

#ExternalProject_Add(
#    googletest
#    URL "$ENV{HABANA_SOFTWARE_STACK}/3rd-parties/googletest_1_10/"
#    TIMEOUT 10
#    CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
#               -Dgtest_force_shared_crt=ON -DCMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}
#    INSTALL_COMMAND ""
#    UPDATE_COMMAND ""
#    LOG_DOWNLOAD ON
#    LOG_CONFIGURE ON
#    LOG_BUILD ON)

include_directories(
          ${source_dir}/googletest/include
          ../../
          ../../utils
          ../../../src/graph_compiler/habana_nodes
          ../../../src/graph_compiler/sync
          ../../../src/graph_compiler/passes/
          ../../../src/graph_compiler/passes/sram_management
          ../../../src/graph_compiler/passes/sram_management/pipeline_management
          ../../../src/graph_compiler/brain
          ../../../src/graph_compiler/brain/memory_management
          ../../../src/graph_compiler/brain/scheduler
          ../../../src/graph_compiler/utils
          ../../../src/graph_compiler/mme
          $ENV{SPECS_ROOT}
          $ENV{SPECS_EXT_ROOT}
          $ENV{SPDLOG_ROOT}
          $ENV{FUNC_SIM_ROOT}
          $ENV{FUNC_SIM_BOOST_BUILD}/boost
          $ENV{TPC_KERNELS_ROOT}/include)

if (WIN32)
  include_directories(../src/graph_compiler/win32_workaround)
endif()

#set_target_properties(googletest PROPERTIES EXCLUDE_FROM_ALL TRUE)
set(Suffix ".a")
set(Pthread "-pthread")
set (libPath "")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(libSuffix "d")
else()
    set(libSuffix "")
endif()

add_executable(${UNIT_TEST_TARGET} ${UNIT_TEST_ALL_FILES})

target_link_libraries( ${UNIT_TEST_TARGET}
    ${GC_TESTS_COMMON_LIBS}
    HalReader
    GraphCompiler
    ${tpc_kernel_api}
    ${hl_logger}
    ${hl_gcfg}
)

if(VTUNE_PROFILER)
    link_directories($ENV{VTUNE_INSTALL_DIR}/sdk/lib64)
    target_link_libraries(${UNIT_TEST_TARGET} libittnotify.a)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-narrowing -Werror -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pipe")

add_dependencies(${UNIT_TEST_TARGET} googletest)

if(CMAKE_COMPILER_IS_GNUC OR CMAKE_COMPILER_IS_GNUCXX)
    target_link_libraries(${UNIT_TEST_TARGET} pthread)
endif()

add_custom_target(unit_check COMMAND ${EXECUTABLE_OUTPUT_PATH}/${UNIT_TEST_TARGET} --gtest_output="xml:${CMAKE_BINARY_DIR}/tests.xml" || true)
add_dependencies(unit_check ${UNIT_TEST_TARGET})

if(COVERAGE_ENABLED)
    find_package(gcovr REQUIRED)
    target_link_libraries(${UNIT_TEST_TARGET} -lgcov)
    add_custom_target(coverage WORKING_DIRECTORY ${CMAKE_BINARY_DIR} COMMAND ${GCOVR_EXECUTABLE} -r ${CMAKE_SOURCE_PATH}/src --xml -o ${CMAKE_BINARY_DIR}/coverage.xml)
    add_dependencies(coverage check)
endif()
