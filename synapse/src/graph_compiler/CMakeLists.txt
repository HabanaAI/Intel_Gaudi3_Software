include(ExternalProject)

set(TARGET_NAME GraphCompiler)

set_directory_properties(PROPERTIES EP_PREFIX ${CMAKE_BINARY_DIR}/ThirdParty)

include(cmake/lemon.cmake)

set(ProtobufBuildDir ${CMAKE_BINARY_DIR}/../ThirdParties/protobuf/build)

include_directories(../../include/
                    ../../include/internal/
                    ../runtime/
                    ${ProtobufBuildDir}/include
                    ${ProtobufBuildDir}/include/google/protobuf
                    $ENV{SPECS_ROOT}
                    $ENV{SPECS_EXT_ROOT}
                    $ENV{HABANALABS_ROOT}/include/uapi
                    $ENV{HLTHUNK_ROOT}/include/uapi
                    $ENV{SPDLOG_ROOT}
                    $ENV{FUNC_SIM_ROOT}
                    $ENV{HABANA_SOFTWARE_STACK}/3rd-parties
                    habana_nodes
                    brain
                    passes
                    passes/sram_management
                    passes/sram_management/pipeline_management
                    passes/ir_translation
                    perf_checks
                    mme
                    sync
                    sif
                    smf
                    utils)

if (WIN32)
include_directories(win32_workaround)
endif()

file(GLOB SRC *.cpp
              ../platform/gaudi/utils.cpp
              ../platform/gaudi2/utils.cpp
              ../platform/gaudi/runtime/master_qmans_definition.cpp
              ../dlc_impl.cpp
              habana_nodes/*.cpp
              passes/*.cpp
              code_generation/*.cpp
              brain/*.cpp
              brain/*/*.cpp
              passes/sram_management/*.cpp
              passes/sram_management/pipeline_management/*.cpp
              passes/ir_translation/*.cpp
              mme/*.cpp
              runtime/scal/gaudi2/master_qmans_definition.cpp
              sync/*.cpp
              sif/*.cpp
              smf/*.cpp
              graph_serializers/*.cpp
              perf_checks/*.cpp
              memory_management/*.cpp
              $ENV{HABANA_SOFTWARE_STACK}/3rd-parties/lz4/lz4.c)

file(GLOB HDR *.h
              ../platform/gaudi/utils.hpp
              ../runtime/*.h
              ../../include/*.h
              sif/*.h
              smf/*.h
              code_generation/*.h
              memory_management/*.h)

include_directories(../platform/gaudi/graph_compiler)
include_directories(../platform/gaudi2/graph_compiler)
include_directories(../platform/gaudi3/graph_compiler)

file(GLOB_RECURSE HW_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../protobuf/*.cc"
                         "${CMAKE_CURRENT_SOURCE_DIR}/../runtime/common/memory_utils/*.cpp"
                         "${CMAKE_CURRENT_SOURCE_DIR}/../platform/common/*.cpp"
                         "${CMAKE_CURRENT_SOURCE_DIR}/../platform/common/*.cc"
                         "${CMAKE_CURRENT_SOURCE_DIR}/../platform/gaudi/graph_compiler/*.cpp"
                         "${CMAKE_CURRENT_SOURCE_DIR}/../platform/gaudi/graph_compiler/*.cc"
                         "${CMAKE_CURRENT_SOURCE_DIR}/../platform/gaudi2/graph_compiler/*.cpp"
                         "${CMAKE_CURRENT_SOURCE_DIR}/../platform/gaudi2/graph_compiler/*.cc"
                         "${CMAKE_CURRENT_SOURCE_DIR}/../platform/gaudi3/graph_compiler/*.cpp"
                         "${CMAKE_CURRENT_SOURCE_DIR}/../platform/gaudi3/graph_compiler/*.cc")
file(GLOB_RECURSE HW_HDR "${CMAKE_CURRENT_SOURCE_DIR}/../protobuf/*.h"
                         "${CMAKE_CURRENT_SOURCE_DIR}/../platform/common/*.h"
                         "${CMAKE_CURRENT_SOURCE_DIR}/../platform/gaudi/graph_compiler/*.h"
                         "${CMAKE_CURRENT_SOURCE_DIR}/../platform/gaudi2/graph_compiler/*.h"
                         "${CMAKE_CURRENT_SOURCE_DIR}/../platform/gaudi3/graph_compiler/*.h")

source_group("Header files" FILES ${HDR} ${HW_HDR}) #group them nicely

set(LINKER_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/../../libSynapse.version")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -Wl,--version-script=${LINKER_SCRIPT}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wno-narrowing")

if (TSANITIZE_ON)
    # there is an compilation issue with tsan in release
    # WA is to disable checks
   set(SANITIZE "-fsanitize=thread -Wno-stringop-overflow -Wno-array-bounds")
endif()

if (SANITIZE_ON)
    if (TSANITIZE_ON)
        message(SEND_ERROR "both tsan and asan are on. only one of the can work" )
    endif()

    # TODO[SW-169021]: overriding the default to w/a linker size issue
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2")

    set(SANITIZE "-fsanitize=undefined")
    set(SANITIZE "${SANITIZE} -fno-omit-frame-pointer -fsanitize=address")
endif()

# Build a library from all specified source files
add_library(${TARGET_NAME} STATIC ${SRC} ${HW_SRC} ${HDR} ${HW_HDR})

target_include_directories(${TARGET_NAME} SYSTEM PRIVATE $<TARGET_PROPERTY:fma,INTERFACE_INCLUDE_DIRECTORIES>)

target_link_libraries(${TARGET_NAME}
  PRIVATE
    SynapseInfra
    HalReader
    Eager
    ${ProtobufBuildDir}/lib/libprotobuf.a
    lemon
    mme_stack
    rotator_stack
    stdc++fs
    ${CMAKE_DL_LIBS}
)

target_link_libraries(${TARGET_NAME} PRIVATE ${SANITIZE})
