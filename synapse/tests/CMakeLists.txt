include(ExternalProject)
include(../cmake/DisableAVX512.cmake)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(QA_TESTS_ENABLED "Enable automatic tests" OFF)

set(SYSTEM_TEST_TARGET synapse_tests)
file(GLOB SYSTEM_TEST_FILES *.cpp
                            gaudi_tests/*.cpp
                            gaudi_tests/*.c
                            runtime/scal/*.cpp
                            runtime/common/*.cpp
                            runtime/common/*.c
                            runtime/infra/*.cpp
                            scal_tests/*.cpp
                            utils/*.cpp)

set_directory_properties(PROPERTIES EP_PREFIX ${CMAKE_BINARY_DIR}/ThirdParty)

set(ProtobufBuildDir ${CMAKE_BINARY_DIR}/../ThirdParties/protobuf/build)
include_directories(${ProtobufBuildDir}/include ${ProtobufBuildDir}/include/google/protobuf)
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
include_directories(${source_dir}/googletest/include
                    ${source_dir}/googlemock/include
                    gaudi_tests/
                    ../include
                    ../src/include
                    ../src/infra
                    ../src/graph_compiler/
                    ../src/graph_compiler/habana_nodes/
                    ../src/data_serialize/include

                    ../src/runtime/common/
                    ../src/runtime/common/recipe/
                    ../src/runtime/common/recipe/patching/
                    ../src/runtime/common/stream

                    ../src/runtime/qman/gaudi/

                    ../src/runtime/scal/common/
                    ../src/runtime/scal/common/entities/

                    ../src/platform/gaudi/graph_compiler/
                    ../src/platform/gaudi2/graph_compiler/
                    ../src/platform/gaudi3/graph_compiler/

                    $ENV{SCAL_ROOT}/include
                    $ENV{ENGINES_FW_ROOT}/include

                    $ENV{MME_ROOT}/include
                    $ENV{SPECS_ROOT_H9}
                    $ENV{SPECS_ROOT}
                    $ENV{SPECS_EXT_ROOT}
                    $ENV{THIRD_PARTIES_ROOT}
                    $ENV{HABANALABS_ROOT}/include/uapi
                    $ENV{HLTHUNK_ROOT}/include/uapi
                    $ENV{SPDLOG_ROOT}
                    $ENV{HL_LOGGER_INCLUDE_DIRS}
                    $ENV{HL_GCFG_INCLUDE_DIRS}
                    $ENV{TPC_KERNELS_ROOT}/include
                    $ENV{SCAL_ROOT}
                    ./utils)
include_directories(SYSTEM $<TARGET_PROPERTY:fma,INTERFACE_INCLUDE_DIRECTORIES>)
if (WIN32)
include_directories(../src/graph_compiler/win32_workaround)
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

add_executable(${SYSTEM_TEST_TARGET} ${SYSTEM_TEST_FILES})

target_link_libraries(
    ${SYSTEM_TEST_TARGET}
    ${binary_dir}/lib/${libPath}${CMAKE_FIND_LIBRARY_PREFIXES}gtest${libSuffix}${Suffix}
    ${binary_dir}/lib/${libPath}${CMAKE_FIND_LIBRARY_PREFIXES}gmock${libSuffix}${Suffix}
    ${binary_dir}/lib/${libPath}${CMAKE_FIND_LIBRARY_PREFIXES}gtest_main${libSuffix}${Suffix}
    ${binary_dir}/lib/${libPath}${CMAKE_FIND_LIBRARY_PREFIXES}gmock_main${libSuffix}${Suffix}
    DataSerialize
    Synapse
    ${hl_thunk}
    ${hl_logger}
    ${hl_gcfg}
    ${scal}
    ${ProtobufBuildDir}/lib/libprotobuf.a
    mme_reference
    fma
)

set(LINKER_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/../libSynapse.version")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-narrowing -Werror -Wall -Wno-sign-compare -Wl,--enable-new-dtags -Wl,--version-script=${LINKER_SCRIPT} -Wno-strict-aliasing -rdynamic")

if (NOT (CMAKE_SYSTEM_PROCESSOR MATCHES "^powerpc*" OR CMAKE_SYSTEM_PROCESSOR MATCHES "^ppc64*" )) #PPC doesn't support march=native
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_POWER_PC_")
endif()

if (TSANITIZE_ON)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=thread")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pipe")
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    if (NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS 6.0)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated -Wno-misleading-indentation")
    endif()
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-command-line-argument")
endif()

disable_avx512_if_needed()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${NOAVX512}")

add_dependencies(${SYSTEM_TEST_TARGET} googletest)

if(CMAKE_COMPILER_IS_GNUC OR CMAKE_COMPILER_IS_GNUCXX)
    target_link_libraries(${SYSTEM_TEST_TARGET} pthread)
endif()

add_custom_target(system_check COMMAND ${EXECUTABLE_OUTPUT_PATH}/${SYSTEM_TEST_TARGET} --gtest_output="xml:${CMAKE_BINARY_DIR}/tests.xml" || true)
add_dependencies(system_check ${SYSTEM_TEST_TARGET})

if(COVERAGE_ENABLED)
    find_package(gcovr REQUIRED)
    target_link_libraries(${SYSTEM_TEST_TARGET} -lgcov)
    add_custom_target(coverage WORKING_DIRECTORY ${CMAKE_BINARY_DIR} COMMAND ${GCOVR_EXECUTABLE} -r ${CMAKE_SOURCE_PATH}/src --xml -o ${CMAKE_BINARY_DIR}/coverage.xml)
    add_dependencies(coverage check)
endif()


if(VTUNE_PROFILER)
    link_directories($ENV{VTUNE_INSTALL_DIR}/sdk/lib64)
    target_link_libraries(${SYSTEM_TEST_TARGET} libittnotify.a)
endif()


add_subdirectory(gc_tests)
add_subdirectory(runtime_unit_tests)
add_subdirectory(runtime_tests)
add_subdirectory(json_tests)

if(QA_TESTS_ENABLED)
    add_subdirectory(fuser_qa_tests)
endif()
