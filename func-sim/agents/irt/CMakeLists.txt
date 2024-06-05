cmake_minimum_required(VERSION 3.5.1)
project(Rotator LANGUAGES CXX)
set(PROJECT_BRIEF "\"Habana Labs Rotator\"")

set(CMAKE_VERBOSE_MAKEFILE OFF)

set(TARGET_ROTATOR_LIB rotator_lib)
set(TARGET_ROTATOR_GAUDI2_SIM rotator_gaudi2_sim)
set(TARGET_ROTATOR_GAUDI3_SIM rotator_gaudi3_sim)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 17)

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    # Support Unix Makefiles and Ninja
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK "${CCACHE_PROGRAM}")
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
set(LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}/lib")
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected, defaulting to Debug")
    set(CMAKE_BUILD_TYPE "Debug")
endif()

option(RUN_WITH_SYNAPSE "use rotator in synapse" ON)
option(RUN_WITH_CORAL "use rotator in Coral-SIM" OFF)
option(RUN_WITH_SV "use rotator in SV environment" OFF)
option(RUN_STANDALONE "compile standalone environment" OFF)

option(TSANITIZE_ON "compile with TSAN" OFF)
option(SANITIZE_ON "compile with both ASAN and UBSAN" OFF)

set(SRC_LIB $ENV{ROTATOR_ROOT}/IRTsim6/src/IRTutils.cpp $ENV{ROTATOR_ROOT}/IRTsim6/src/IRT.cpp)
set(HEADER_LIB $ENV{ROTATOR_ROOT}/IRTsim6/src/IRTutils.h $ENV{ROTATOR_ROOT}/IRTsim6/src/IRT.h)
set(SRC_SIM $ENV{ROTATOR_ROOT}/IRTsim6/src/IRT.cpp $ENV{ROTATOR_ROOT}/IRTsim6/src/IRThl.cpp $ENV{ROTATOR_ROOT}/IRTsim6/src/IRTsim.cpp)
set(HEADER_SIM $ENV{ROTATOR_ROOT}/IRTsim6/src/IRT.h $ENV{ROTATOR_ROOT}/IRTsim6/src/IRTsim.h)

set(SRC_LIB_H9 $ENV{ROTATOR_ROOT}/IRTsim9/src/IRTutils.cpp $ENV{ROTATOR_ROOT}/IRTsim9/src/IRT.cpp)
set(HEADER_LIB_H9 $ENV{ROTATOR_ROOT}/IRTsim9/src/IRTutils.h $ENV{ROTATOR_ROOT}/IRTsim9/src/IRT.h $ENV{CORAL_SIM_ROOT}/header/gaudi3/fs_core/fs_fma_gaudi3.h)
set(SRC_SIM_H9 $ENV{ROTATOR_ROOT}/IRTsim9/src/IRTsim.cpp $ENV{ROTATOR_ROOT}/IRTsim9/src/IRT.cpp $ENV{ROTATOR_ROOT}/IRTsim9/src/IRThl.cpp $ENV{CORAL_SIM_ROOT}/src/gaudi3/fs_core/fs_fma.cpp)
set(HEADER_SIM_H9 $ENV{ROTATOR_ROOT}/IRTsim9/src/IRT.h $ENV{ROTATOR_ROOT}/IRTsim9/src/IRTsim.h $ENV{CORAL_SIM_ROOT}/header/gaudi3/fs_core/fs_fma_gaudi3.h)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}
                    ${CMAKE_CURRENT_SOURCE_DIR}/IRTsim6/src
                    ${CMAKE_CURRENT_SOURCE_DIR}/IRTsim9/src
                    $ENV{CORAL_SIM_ROOT}/header/gaudi3/fs_core/)
#compilation flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -DDISABLE_IRT_TRACE_UTILS_PRINT")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall ") #TODO: add -Werror
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pipe")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-overflow -Wno-sign-compare -Wno-array-bounds -Wno-unused-variable -Wno-reorder -Wno-strict-aliasing")
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-but-set-variable -Wno-invalid-offsetof")
else()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-missing-braces -Wno-overloaded-virtual")
endif()
# Debug\Release flags
set(CMAKE_CXX_FLAGS_DEBUG "-ggdb -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-ggdb -O3 -Wno-unused-result -D_FORTIFY_SOURCE=2 -DNDEBUG")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")

if (TSANITIZE_ON)
    set(SANITIZE "-fsanitize=thread")
endif()
if (SANITIZE_ON)
    if (TSANITIZE_ON)
        message(SEND_ERROR "both tsan and asan are on. only one of the can work" )
    endif()

    set(SANITIZE "-fsanitize=undefined")
    set(SANITIZE "${SANITIZE} -fno-omit-frame-pointer -fsanitize=address")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SANITIZE}")

if(RUN_WITH_SYNAPSE)
    set(TARGET_DEFS "-DGC_BUILD")
endif()
if(RUN_WITH_SV)
    set(TARGET_DEFS "-DRUN_WITH_SV")
endif()
if(RUN_WITH_CORAL)
    set(TARGET_DEFS "-DHABANA_SIMULATION")
endif()
if(RUN_STANDALONE)
    set(TARGET_DEFS "-DSTANDALONE_ROTATOR")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-format -Wno-parentheses -Wno-switch")
endif()

# build descriptor generator library for gaudi2 and gaudi3
add_library(${TARGET_ROTATOR_LIB} STATIC ${SRC_LIB} ${HEADER_LIB} ${SRC_LIB_H9} ${HEADER_LIB_H9})
target_compile_definitions(${TARGET_ROTATOR_LIB} PUBLIC ${TARGET_DEFS})

# build standalone gadui2 simulator
if (RUN_STANDALONE)
    add_executable(${TARGET_ROTATOR_GAUDI2_SIM} ${SRC_SIM} ${HEADER_SIM})
    target_compile_definitions(${TARGET_ROTATOR_GAUDI2_SIM} PUBLIC ${TARGET_DEFS})
    target_link_libraries(${TARGET_ROTATOR_GAUDI2_SIM} PRIVATE ${TARGET_ROTATOR_LIB})
endif()

# build standalone gadui3 simulator
if (RUN_STANDALONE)
    add_executable(${TARGET_ROTATOR_GAUDI3_SIM} ${SRC_SIM_H9} ${HEADER_SIM_H9})
    target_compile_definitions(${TARGET_ROTATOR_GAUDI3_SIM} PUBLIC ${TARGET_DEFS})
    target_link_libraries(${TARGET_ROTATOR_GAUDI3_SIM} PRIVATE ${TARGET_ROTATOR_LIB})
endif()
