set(CMAKE_VERBOSE_MAKEFILE OFF)
cmake_minimum_required(VERSION 3.5.1)
project(Rotator LANGUAGES C CXX)
set(PROJECT_BRIEF "\"Habana Labs Rotator\"")

set(TARGET_ROTATOR_LIB rotator_lib)
set(TARGET_ROTATOR_SIM rotator_sim)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 11)

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

set(SRC_LIB src/IRTutils.cpp src/IRT.cpp)
set(HEADER_LIB src/IRTutils.h
	        src/IRT.h)
set(SRC_SIM src/IRT.cpp src/IRThl.cpp src/IRTsim.cpp)
set(HEADER_SIM src/IRT.h src/IRTsim.h)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}
                    ${CMAKE_CURRENT_SOURCE_DIR}/src)
#compilation flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
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

# build descriptor generator library
add_library(${TARGET_ROTATOR_LIB} STATIC ${SRC_LIB} ${HEADER_LIB})
target_compile_definitions(${TARGET_ROTATOR_LIB} PUBLIC ${TARGET_DEFS})

# build standalone simulator
if (RUN_STANDALONE)
    add_executable(${TARGET_ROTATOR_SIM} ${SRC_SIM} ${HEADER_SIM})
    target_compile_definitions(${TARGET_ROTATOR_SIM} PUBLIC ${TARGET_DEFS})
    target_link_libraries(${TARGET_ROTATOR_SIM} PRIVATE ${TARGET_ROTATOR_LIB})
endif()

