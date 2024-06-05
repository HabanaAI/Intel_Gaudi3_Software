cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(hl_gcfg)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_program(CCACHE_PROGRAM ccache)
if (CCACHE_PROGRAM)
    set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
endif()
set(CMAKE_CXX_FLAGS_DEBUG "-ggdb -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

if (SANITIZE_ON)
    set(SANITIZE "-fsanitize=undefined")
    set(SANITIZE "${SANITIZE} -fno-omit-frame-pointer -fsanitize=address")
endif()

# Enable/disable testing
option(TESTS_ENABLED "Enable automatic tests" ON)

set(LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}/lib")
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}/bin")

add_subdirectory(src)

if(TESTS_ENABLED)
    add_subdirectory(tests)
endif()
