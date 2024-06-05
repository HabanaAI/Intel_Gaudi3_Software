cmake_minimum_required(VERSION 3.5.1)
project(mme LANGUAGES CXX)



set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(PROJECT_VERSION 0.1)
set(PROJECT_BRIEF "\"Habana Labs MME Optimizer\"")
set(CMAKE_CXX_STANDARD 17)


# Enable/disable options
option(TESTS_ENABLED "Enable automatic tests" ON)
option(MME_VER "Compile mme verification libraries" ON)
option(SYN_DEPEND "Compile while depending on synapse sources" ON)
option(SIM_DEPEND "Compile while depending on hl thunk and tpc sim" ON)
option(SWTOOLS_DEP "Compiler with depending on swtools_sdk" ON)

# chip specific compilation - default - all
option(GAUDI_EN "Enable compilation for chip-type - Gaudi" ON)
option(GAUDI2_EN "Enable compilation for chip-type - Gaudi2" ON)
option(GAUDI3_EN "Enable compilation for chip-type - Gaudi3" ON)

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

include(DisableAVX512)
include(SeparateDebugSymbols)


set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected, defaulting to Debug")
    set(CMAKE_BUILD_TYPE "Debug")
endif()

disable_avx512_if_needed()

include_directories(include
                    mme_reference
                    src/mme_common
                    src/sync)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories($ENV{SPECS_ROOT}
                    $ENV{SPECS_EXT_ROOT}
                    $ENV{HABANA_SOFTWARE_STACK}/3rd-parties
                    $ENV{HABANALABS_ROOT}/include/uapi
                    $ENV{SPDLOG_ROOT})

if (SWTOOLS_DEP)
    set(hl_logger $ENV{BUILD_ROOT_LATEST}/libhl_logger.so)
    set(hl_gcfg $ENV{BUILD_ROOT_LATEST}/libhl_gcfg.so)
    include_directories($ENV{HL_LOGGER_INCLUDE_DIRS}
                        $ENV{HL_GCFG_INCLUDE_DIRS}
                       )
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSWTOOLS_DEP=1")
endif()

add_subdirectory(src)
add_subdirectory(mme_reference)

if(MME_VER)
    add_subdirectory(mme_verification)
    if(TESTS_ENABLED)
        add_subdirectory(tests)
    endif()
endif()
