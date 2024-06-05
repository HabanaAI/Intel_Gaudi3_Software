cmake_minimum_required(VERSION 3.5)

project(GraphCompilerPlugin CXX)

set(CMAKE_CXX_STANDARD 17)

add_library(GraphCompilerPlugin SHARED
    gc_singleton.h
    gc_singleton.cpp
)

target_include_directories(GraphCompilerPlugin
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src/infra
        ${CMAKE_SOURCE_DIR}/src/runtime
        $ENV{SWTOOLS_SDK_ROOT}/include/shim_sdk
        $ENV{THIRD_PARTIES_ROOT}/json/single_include
)

target_link_libraries(GraphCompilerPlugin
    PRIVATE
        Synapse
        GraphSerialize
        DataSerialize
        stdc++fs
)
