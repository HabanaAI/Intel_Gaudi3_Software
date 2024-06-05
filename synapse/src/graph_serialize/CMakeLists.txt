cmake_minimum_required(VERSION 3.5)

project(GraphSerialize CXX)

set(CMAKE_CXX_STANDARD 17)

add_library(GraphSerialize STATIC
    include/graph_serializer/graph_serializer.h
    graph_serializer.cpp
    include/graph_serializer/strategy_serializer.h
    $ENV{HABANA_SOFTWARE_STACK}/3rd-parties/lz4/lz4.h
    $ENV{HABANA_SOFTWARE_STACK}/3rd-parties/lz4/lz4.c
)

target_include_directories(GraphSerialize
    PUBLIC
        include
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src/infra
        ${CMAKE_SOURCE_DIR}/src/graph_compiler
        ${CMAKE_SOURCE_DIR}/src/data_serialize
        $ENV{HABANA_SOFTWARE_STACK}/3rd-parties
)

target_include_directories(GraphSerialize SYSTEM PRIVATE $<TARGET_PROPERTY:fma,INTERFACE_INCLUDE_DIRECTORIES>)


target_link_libraries(GraphSerialize
    PRIVATE
        HalReader
        stdc++fs
        DataSerialize
)