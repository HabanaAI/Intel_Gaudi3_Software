#set(CMAKE_VERBOSE_MAKEFILE ON)
cmake_minimum_required(VERSION 3.5.1)
project(mme_test_gaudi3 LANGUAGES CXX)
set(TARGET_LIB mme_test_gaudi3_lib)
set(TARGET_EXE mme_test_gaudi3)

include(ExternalProject)

# compile all files with *.cpp suffix
file(GLOB_RECURSE SRC *.cpp)
list(REMOVE_ITEM SRC main.cpp)
#add headers
file(GLOB_RECURSE HDR *.h)

#group them nicely
source_group("Header files" FILES ${HDR})

# library search paths
include(findCoralSim)

# Build a library from all specified source files
add_library(${TARGET_LIB} SHARED ${SRC} ${HDR})
target_link_libraries(${TARGET_LIB}
    PUBLIC
        mme_common
        mme_user_gaudi3
        SynapseMmeReference)

if (NOT MSVC)
  target_link_libraries(${TARGET_LIB} PRIVATE pthread)
endif()

#external APIs
target_link_libraries(${TARGET_LIB} PRIVATE coral_gaudi3
                                            ${tpc_api}
                                            ${tpc_kernel_api}
                                            ${hl_thunk})

target_link_libraries(${TARGET_LIB} PRIVATE stdc++fs)

add_executable(${TARGET_EXE} ./main.cpp)
target_link_libraries(${TARGET_EXE} ${TARGET_LIB})

