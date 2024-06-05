# COPYRIGHT (c) 2020 Habanalabs Ltd.

cmake_minimum_required(VERSION 3.5.1)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
project(scal C CXX)
set(PROJECT_VERSION 0.1)
set(PROJECT_BRIEF "\"Habana Labs SCAL Library\"")
set(CMAKE_CXX_STANDARD 17)


set(SCAL "scal")
set(SCAL_PACKAGE "scal")
set(SCAL_COMPONENT "lib${SCAL}")
set(SCAL_TARGET "${SCAL}")

# Reduce "warm" compilation time if ccache is available
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    # Support Unix Makefiles and Ninja
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK "${CCACHE_PROGRAM}")
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
set(LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}/lib")
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}/bin")

if(COVERAGE_ENABLED)
    set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -fprofile-arcs -ftest-coverage")
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -fno-omit-frame-pointer -g1")

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS_EQUAL 7.5)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-variable")
    endif()
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected, defaulting to Debug")
    set(CMAKE_BUILD_TYPE "Debug")
endif()

#include_directories(${CMAKE_SOURCE_DIR}/fs_src)

include_directories(${CMAKE_SOURCE_DIR}/src/logger
                    ${CMAKE_SOURCE_DIR}/src
                    $ENV{HL_LOGGER_INCLUDE_DIRS}
                    $ENV{RDMA_CORE_ROOT}/build/include)

# SPECS required for shim_typedefs
include_directories($ENV{SPECS_ROOT})

set(hlthunk $ENV{BUILD_ROOT_LATEST}/libhl-thunk.so)
set(hl_logger $ENV{BUILD_ROOT_LATEST}/libhl_logger.so)

if(TESTS_ENABLED)
    add_subdirectory(tests)
endif()

#include the subdirectory containing our libs
include_directories(include)
add_subdirectory(src)

include(CPack)
