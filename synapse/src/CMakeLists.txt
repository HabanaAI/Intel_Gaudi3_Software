# Specify .cpp files
set(TARGET_NAME Synapse)

# add libarchive for archiving logs
set (LIBARCHIVE_DIR $ENV{THIRD_PARTIES_ROOT}/libarchive)

include_directories( ${LIBARCHIVE_DIR}/libarchive)

# flags for libarchive
set (ENABLE_MBEDTLS OFF CACHE BOOL "Disable use of MBEDTLS")
set (ENABLE_NETTLE OFF CACHE BOOL "Disable use of NETTLE")
set (ENABLE_OPENSSL OFF CACHE BOOL "Disable use of OPENSSL")
set (ENABLE_LIBB2 OFF CACHE BOOL "Disable use of LIBB2")
set (ENABLE_LZ4 OFF CACHE BOOL "Disable use of LZ4")
set (ENABLE_LZO OFF CACHE BOOL "Disable use of LZO")
set (ENABLE_LZMA OFF CACHE BOOL "Disable use of LZMA")
set (ENABLE_ZSTD OFF CACHE BOOL "Disable use of ZSTD")
set (ENABLE_BZip2 OFF CACHE BOOL "Disable use of BZip2")
set (ENABLE_LIBXML2 OFF CACHE BOOL "Disable use of LIBXML2")
set (ENABLE_EXPAT OFF CACHE BOOL "Disable use of EXPAT")
set (ENABLE_PCREPOSIX OFF CACHE BOOL "Disable use of PCREPOSIX")
set (ENABLE_LibGCC OFF CACHE BOOL "Disable use of LibGCC")
set (ENABLE_CNG OFF CACHE BOOL "Disable use of CNG")
set (ENABLE_XATTR OFF CACHE BOOL "Disable use of XATTR")
set (ENABLE_ACL OFF CACHE BOOL "Disable use of ACL")
set (ENABLE_ICONV OFF CACHE BOOL "Disable use of ICONV")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_subdirectory(${LIBARCHIVE_DIR} ${LIBARCHIVE_DIR})

# build protobuf compiler and lib
set(ProtobufBuildDir ${CMAKE_BINARY_DIR}/../ThirdParties/protobuf/build)
set(ProtobufSrcDir   ${CMAKE_BINARY_DIR}/../ThirdParties/protobuf/src)
set(PROTOC ${ProtobufBuildDir}/bin/protoc)
set(LIBPROTOBUF ${ProtobufBuildDir}/lib/libprotobuf.a)

set(SHOULD_REBUILD_PROTOC FALSE)

# verify protoc exists
if (NOT EXISTS ${PROTOC})
    set(SHOULD_REBUILD_PROTOC TRUE)
else()
    # check version
    message( "Checking Protoc version...")
    set(NextCmd ${PROTOC} --version)
    execute_process(COMMAND ${NextCmd}
            WORKING_DIRECTORY ${ProtobufSrcDir}
            RESULT_VARIABLE   CmdRes
            OUTPUT_VARIABLE   CmdOut
            ERROR_VARIABLE    CmdErr)
    string(REGEX MATCH "([0-9]+)\\.([0-9]+)\\.([0-9]+)" VERSION_NUM ${CmdOut})
    message ("version after regex: ${VERSION_NUM}")
    if (NOT VERSION_NUM STREQUAL "3.9.0")
        message( "Version mismatch in protoc (should be 3.9.0) version is: ${VERSION_NUM}. Rebuilding...")
        set(SHOULD_REBUILD_PROTOC TRUE)
    else()
        message( "protoc version OK.")
    endif()
endif()

message( "should rebuild protoc: " ${SHOULD_REBUILD_PROTOC})

if (SHOULD_REBUILD_PROTOC)
    # create local directories
    file(MAKE_DIRECTORY ${ProtobufBuildDir})
    file(MAKE_DIRECTORY ${ProtobufSrcDir})
    execute_process(COMMAND cp -r $ENV{THIRD_PARTIES_ROOT}/protobuf-3.9.x/. ${ProtobufSrcDir}/.)

    # we need to either touch these files, or re-run autotools from scratch (which takes a couple of minutes)
    # if we do neither, then configure will fail on machines with a newer version of autotools (e.g. Ubuntu 20)
    # the touch command should be sufficient because the source is already autotool-ized
    execute_process(COMMAND touch ${ProtobufSrcDir}/aclocal.m4 ${ProtobufSrcDir}/Makefile.am ${ProtobufSrcDir}/Makefile.in)

    # configure
    message("Configuring Protobuf...")
    if (CCACHE_PROGRAM)
        set(ENV{CC} "${CCACHE_PROGRAM} ${CMAKE_C_COMPILER}")
        set(ENV{CXX} "${CCACHE_PROGRAM} ${CMAKE_CXX_COMPILER}")
    endif()
    set(ENV{CXXFLAGS} "-fPIC -fvisibility=hidden -fvisibility-inlines-hidden")
    set(NextCmd ./configure --prefix=${ProtobufBuildDir} --disable-shared)
    execute_process(COMMAND ${NextCmd}
            WORKING_DIRECTORY ${ProtobufSrcDir}
            RESULT_VARIABLE   CmdRes
            OUTPUT_VARIABLE   CmdOut
            ERROR_VARIABLE    CmdErr)
    if(NOT CmdRes EQUAL 0)
        message( FATAL_ERROR "Res: ${CmdRes}, Out: ${CmdOut}, Err: ${CmdErr}")
    endif()

    # make
    message("Making Protobuf...")
    include(ProcessorCount)
    ProcessorCount(NProc)
    set(NextCmd make -j${NProc})
    execute_process(COMMAND ${NextCmd}
            WORKING_DIRECTORY ${ProtobufSrcDir}
            RESULT_VARIABLE   CmdRes
            OUTPUT_VARIABLE   CmdOut
            ERROR_VARIABLE    CmdErr)
    if(NOT CmdRes EQUAL 0)
        message( FATAL_ERROR "Res: ${CmdRes}, Out: ${CmdOut}, Err: ${CmdErr}")
    endif()

    # make install
    message("Installing Protobuf...")
    set(NextCmd make install)
    execute_process(COMMAND ${NextCmd}
            WORKING_DIRECTORY ${ProtobufSrcDir}
            RESULT_VARIABLE   CmdRes
            OUTPUT_VARIABLE   CmdOut
            ERROR_VARIABLE    CmdErr)
    if(NOT CmdRes EQUAL 0)
        message(FATAL_ERROR "Res: ${CmdRes}, Out: ${CmdOut}, Err: ${CmdErr}")
    endif()

    # Cleaning old proto-file's products
    file(GLOB ProtocOutputs "${CMAKE_CURRENT_SOURCE_DIR}/protobuf/*.pb.*")

    foreach(SINGLE_PROTOC_OUT  ${ProtocOutputs})
        message( "Cleaning Protobuf output: " ${SINGLE_PROTOC_OUT})
        execute_process(COMMAND rm ${SINGLE_PROTOC_OUT})
    endforeach()
endif()

include_directories(${ProtobufBuildDir}/include
                    ${ProtobufBuildDir}/include/google/protobuf
                    $ENV{SYNAPSE_ROOT}/include/internal/
                    $ENV{SPECS_ROOT_H9}
                    $ENV{SPECS_ROOT}
                    $ENV{SPECS_EXT_ROOT}
                    $ENV{THIRD_PARTIES_ROOT}
                    $ENV{HABANALABS_ROOT}/include/uapi
                    $ENV{HLTHUNK_ROOT}/include/uapi
                    $ENV{SPDLOG_ROOT}
                    $ENV{HL_LOGGER_INCLUDE_DIRS}
                    $ENV{HL_GCFG_INCLUDE_DIRS}
                    $ENV{SCAL_ROOT}/include
                    $ENV{HCL_ROOT}/include
                    $ENV{ENGINES_FW_ROOT}/include
                    $ENV{ENGINES_FW_ROOT}/include/gaudi3
                    $ENV{HL_GCFG_DIRS}
                    $ENV{ROTATOR_ROOT}/IRTsim9/src
                    ../src/infra
                    runtime
                    runtime/common
                    include
                    ${CMAKE_CURRENT_BINARY_DIR})

#build DFA read registers
add_custom_command(
        OUTPUT
            ${CMAKE_CURRENT_BINARY_DIR}/dfa_read_gaudi_regs.hpp
            ${CMAKE_CURRENT_BINARY_DIR}/dfa_read_gaudi2_regs.hpp
            ${CMAKE_CURRENT_BINARY_DIR}/dfa_read_gaudi3_regs.hpp
        COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/runtime/scripts/dfa_regs_list.py ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS
            $ENV{SPECS_ROOT}/gaudi/all_regs.txt
            $ENV{SPECS_ROOT}/gaudi2/all_regs.txt
            $ENV{SPECS_ROOT_H9}/gaudi3/all_regs.txt
            $ENV{SYNAPSE_ROOT}/src/runtime/scripts/dfa_regs_list.py
        VERBATIM)

# Create the protobuf source files
# Must be before the file group command below
add_subdirectory(protobuf)

exec_program(
    "git"
    ${CMAKE_CURRENT_SOURCE_DIR}
    ARGS "rev-parse --short HEAD"
    OUTPUT_VARIABLE SYNAPSE_SHA1_VERSION
    RESULT_VARIABLE RET)
if (NOT RET EQUAL 0)
    set (SYNAPSE_SHA1_VERSION 0)
    message(WARNING "Git rev-parse failed, set SYNAPSE_SHA1_VERSION to 0")
endif()

exec_program(
    "git"
    ${CMAKE_CURRENT_SOURCE_DIR}/../../hcl
    ARGS "rev-parse --short HEAD"
    OUTPUT_VARIABLE HCL_SHA1_VERSION
    RESULT_VARIABLE RET)
if (NOT RET EQUAL 0)
    set (HCL_SHA1_VERSION 0)
    message(WARNING "Git rev-parse failed, set HCL_SHA1_VERSION to 0")
endif()

exec_program(
    "git"
    ${CMAKE_CURRENT_SOURCE_DIR}/../../mme
    ARGS "rev-parse --short HEAD"
    OUTPUT_VARIABLE MME_SHA1_VERSION
    RESULT_VARIABLE RET)
if (NOT RET EQUAL 0)
    set (MME_SHA1_VERSION 0)
    message(WARNING "Git rev-parse failed, set MME_SHA1_VERSION to 0")
endif()

exec_program(
    "git"
     ${CMAKE_CURRENT_SOURCE_DIR}/../../scal
     ARGS "rev-parse --short HEAD"
     OUTPUT_VARIABLE SCAL_SHA1_VERSION
     RESULT_VARIABLE RET)
if (NOT RET EQUAL 0)
    set (SCAL_SHA1_VERSION 0)
    message(WARNING "Git rev-parse failed, set SCAL_SHA1_VERSION to 0")
endif()

# generate version.cpp
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/version.cpp.in" "${CMAKE_CURRENT_BINARY_DIR}/version.cpp" @ONLY)

file(GLOB SRC
            common/recipe/*.cpp
            platform/common/*.cpp
            platform/common/graph_compiler/*.cpp
            platform/gaudi/runtime/*.cpp
            protobuf/*.pb.cc
            # graph compiler shared layer agent
            shared_layer/impl/*.cpp
            # runtime common
            runtime/common/*.cpp
            runtime/common/device/*.cpp
            runtime/common/hcl/*.cpp
            runtime/common/osal/*.cpp
            runtime/common/recipe/*.cpp
            runtime/common/recipe/patching/*.cpp
            runtime/common/queues/*.cpp
            runtime/common/streams/*.cpp
            # runtime QMAN common
            runtime/qman/common/*.cpp
            runtime/qman/common/data_chunk/*.cpp
            runtime/qman/common/parser/*.cpp
            runtime/qman/common/syn_enqueue_adapter/*.cpp
            runtime/qman/common/wcm/*.cpp
            # runtime QMAN Gaudi
            runtime/qman/gaudi/*.cpp
            runtime/qman/gaudi/parser/*.cpp
            # runtime SCAL common
            runtime/scal/common/*.cpp
            runtime/scal/common/entities/*.cpp
            runtime/scal/common/infra/*.cpp
            runtime/scal/common/packets/*.cpp
            runtime/scal/common/patching/*.cpp
            runtime/scal/common/recipe_launcher/*.cpp
            # runtime SCAL Gaudi2
            runtime/scal/gaudi2/*.cpp
            runtime/scal/gaudi2/infra/*.cpp
            runtime/scal/gaudi2/packets/*.cpp
            runtime/scal/gaudi2/entities/*.cpp
            # runtime SCAL Gaudi3
            runtime/scal/gaudi3/*.cpp
            runtime/scal/gaudi3/entities/*.cpp
            runtime/scal/gaudi3/packets/*.cpp
            runtime/scal/gaudi3/direct_mode_packets/*.cpp
            ${CMAKE_CURRENT_BINARY_DIR}/version.cpp) # compile all files with *.cpp suffix

file(GLOB HDR ../include/*.h
            common/recipe/*.h
            common/recipe/*.hpp
            platform/common/*.h
            platform/common/graph_compiler/*.h
            platform/gaudi/runtime/*.hpp
            platform/gaudi/protobuf/*.pb.h
            protobuf/*.pb.h.
            # runtime common
            runtime/common/*.h
            runtime/common/*.hpp
            runtime/common/recipe/*.hpp
            runtime/common/recipe/patching/*.hpp
            runtime/deserialize/*.hpp
            # runtime QMAN common
            runtime/qman/common/*.hpp
            # runtime Gaudi
            runtime/qman/gaudi/*.hpp
            # runtime Gaudi2 (QMAN)
            runtime/qman/gaudi2/*.hpp
            # runtime SCAL common
            runtime/scal/common/*.hpp
            runtime/scal/common/entities/*.hpp
            runtime/scal/common/infra/*.hpp
            runtime/scal/common/packets/*.hpp
            runtime/scal/common/patching/*.hpp
            runtime/scal/common/recipe_launcher/*.hpp
            # runtime Gaudi2 (SCAL)
            runtime/scal/gaudi2/*.hpp
            runtime/scal/gaudi2/infra/*.hpp
            runtime/scal/gaudi2/packets/*.hpp
            runtime/scal/gaudi2/entities/*.hpp
            # runtime Gaudi3
            runtime/scal/gaudi3/*.hpp
            runtime/scal/gaudi3/entities/*.hpp
            runtime/scal/gaudi3/packets/*.hpp
            runtime/scal/gaudi3/direct_mode_packets/*.hpp
            ${CMAKE_CURRENT_BINARY_DIR}/dfa_read_gaudi_regs.hpp
            ${CMAKE_CURRENT_BINARY_DIR}/dfa_read_gaudi2_regs.hpp
            ${CMAKE_CURRENT_BINARY_DIR}/dfa_read_gaudi3_regs.hpp)

set (HDR ${HDR}
          ${CMAKE_CURRENT_BINARY_DIR}/dfa_read_gaudi_regs.hpp
          ${CMAKE_CURRENT_BINARY_DIR}/dfa_read_gaudi2_regs.hpp
          ${CMAKE_CURRENT_BINARY_DIR}/dfa_read_gaudi3_regs.hpp)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall -Werror -Wno-sign-compare -Wno-unused-variable")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pipe")
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    if (NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS 6.0)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-misleading-indentation")
    endif()
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-command-line-argument -Wno-missing-braces")
endif()

if (TSANITIZE_ON)
    set(SANITIZE "-Wno-array-bounds -Wno-stringop-overflow -fsanitize=thread")
endif()

if (SANITIZE_ON)
    if (TSANITIZE_ON)
        message(SEND_ERROR "both tsan and asan are on. only one of the can work" )
    endif()
    set(PROTOBUF_SANITIZE_FLAGS "-fno-sanitize=vptr")

    # TODO[SW-169021]: overriding the default to w/a linker size issue
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2")

    set(SANITIZE "-fsanitize=undefined ${PROTOBUF_SANITIZE_FLAGS}")
    set(SANITIZE "${SANITIZE} -fno-omit-frame-pointer -fsanitize=address")
endif()



if (VALGRIND_ON)
    include(CheckIncludeFile)

    CHECK_INCLUDE_FILE("valgrind/memcheck.h" HAVE_VALGRIND_MEMCHECK)
    if (NOT HAVE_VALGRIND_MEMCHECK)
        message(FATAL_ERROR "valgrind/memcheck.h wasn't found")
    endif()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_VALGRIND=1")
    message(STATUS "Enabling Valgrind")
endif()

# CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_OPT_ARCH_NATIVE_SUPPORTED)
set(LINKER_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/../libSynapse.version")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,--version-script=${LINKER_SCRIPT} -rdynamic")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SANITIZE} ${NOAVX512}")

if (NOT (CMAKE_SYSTEM_PROCESSOR MATCHES "^powerpc*" OR CMAKE_SYSTEM_PROCESSOR MATCHES "^ppc64*" )) #PPC doesn't support march=native
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_POWER_PC_ -mcpu=powerpc64le -mpowerpc64")
endif()

source_group("Header files" FILES ${HDR}) #group them nicely

# Build a library from all specified source files
add_library(${TARGET_NAME} SHARED ${SRC} ${HDR})

separate_debug_symbols(${TARGET_NAME})

target_link_libraries(${TARGET_NAME}
                      ${hcl_api}
                      SynapseInfra
                      HalReader
                      GraphCompiler
                      ${tpc_kernel_api}
                      ${hl_thunk}
                      ${hl_logger}
                      ${hl_gcfg}
                      ${scal}
                      ${LIBPROTOBUF}
                      Eager
                      dl
                      fma
                      archive_static)

if(VTUNE_PROFILER)
    link_directories($ENV{VTUNE_INSTALL_DIR}/sdk/lib64)
    target_link_libraries(${TARGET_NAME} libittnotify.a)
endif()

add_subdirectory(infra)
add_subdirectory(hal_reader)
add_subdirectory(graph_compiler)
add_subdirectory(eager)
add_subdirectory(graph_serialize)
add_subdirectory(data_serialize)
add_subdirectory(gc_shim)
