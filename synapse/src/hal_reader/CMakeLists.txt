cmake_minimum_required(VERSION 3.5)

add_library(HalReader STATIC
    hal_reader.cpp
    gaudi1_hal_reader.cpp
    gaudi2_hal_reader.cpp
    gaudi3_hal_reader.cpp
)

target_include_directories(HalReader
    PUBLIC
        include
    PRIVATE
        include/hal_reader
        ${CMAKE_SOURCE_DIR}/include
        $ENV{MME_ROOT}
)

target_link_libraries(HalReader PRIVATE fma)
