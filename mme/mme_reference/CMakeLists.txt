cmake_minimum_required(VERSION 3.5.1)
project(mme_reference LANGUAGES CXX)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(compilation_flags)


# compile all files with *.cpp suffix
file(GLOB SRC *.cpp)
file(GLOB CHIP_FMA chip_fma/*.cpp)

#add headers
file(GLOB HDR *.h)

#group them nicely
source_group("Header files" FILES ${HDR})


add_library(fma STATIC
    $ENV{CORAL_SIM_ROOT}/src/gaudi2/fs_core/fs_fma.cpp
    $ENV{CORAL_SIM_ROOT}/src/gaudi3/fs_core/fs_fma.cpp
    ${CHIP_FMA})
target_include_directories(fma
    PUBLIC
        $ENV{CORAL_SIM_ROOT}/header/gaudi2/fs_core
        $ENV{CORAL_SIM_ROOT}/header/gaudi3/fs_core)


add_library(mme_reference_obj OBJECT ${SRC} ${HDR})
target_include_directories(mme_reference_obj
    PUBLIC
        $ENV{CORAL_SIM_ROOT}/header/gaudi2/fs_core
        $ENV{CORAL_SIM_ROOT}/header/gaudi3/fs_core)


add_library(mme_reference STATIC $<TARGET_OBJECTS:mme_reference_obj>)
target_link_libraries(mme_reference PUBLIC fma)


add_library(SynapseMmeReference SHARED $<TARGET_OBJECTS:mme_reference_obj>)
target_link_libraries(SynapseMmeReference PUBLIC SynapseMme)
