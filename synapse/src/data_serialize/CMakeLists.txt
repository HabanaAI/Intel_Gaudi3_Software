cmake_minimum_required(VERSION 3.5)

project(DataSerialize CXX)

set(CMAKE_CXX_STANDARD 17)

add_library(DataSerialize STATIC
    include/data_serializer/data_serializer.h
    include/data_serializer/ds_types.h
    data_serializer.cpp
    sql/sql_db.h
    sql/sql_db.cpp
    sql/sql_db_deserializer.h
    sql/sql_db_deserializer.cpp
    sql/sql_db_serializer.h
    sql/sql_db_serializer.cpp
    sql/sql_data_serializer.cpp
    sql/sql_data_deserializer.cpp
    $ENV{HABANA_SOFTWARE_STACK}/3rd-parties/sqlite/sqlite3.h
    $ENV{HABANA_SOFTWARE_STACK}/3rd-parties/sqlite/sqlite3.c
    $ENV{HABANA_SOFTWARE_STACK}/3rd-parties/lz4/lz4.h
    $ENV{HABANA_SOFTWARE_STACK}/3rd-parties/lz4/lz4.c
    $ENV{HABANA_SOFTWARE_STACK}/3rd-parties/lz4/lz4hc.h
    $ENV{HABANA_SOFTWARE_STACK}/3rd-parties/lz4/lz4hc.c
)

# increase the max sql blob size to max int value
target_compile_definitions(DataSerialize PRIVATE SQLITE_MAX_LENGTH=0x7FFFFFFF)

target_include_directories(DataSerialize
    PUBLIC
        include
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src/infra
        ${CMAKE_SOURCE_DIR}/src/runtime
        ${CMAKE_SOURCE_DIR}/src/graph_compiler
        $ENV{HABANA_SOFTWARE_STACK}/3rd-parties
        $ENV{HABANA_SOFTWARE_STACK}/3rd-parties/spdlog/include
)

find_package(Threads REQUIRED)
target_link_libraries(DataSerialize PRIVATE Threads::Threads)
