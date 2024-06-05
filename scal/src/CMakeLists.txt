    # Specify .cpp files
set(TARGET_NAME SCAL)

exec_program(
    "git"
    ${CMAKE_CURRENT_SOURCE_DIR}
    ARGS "rev-parse --short HEAD"
    OUTPUT_VARIABLE SCAL_SHA1_VERSION
    RESULT_VARIABLE RET)

if (NOT RET EQUAL 0)
    set (SCAL_SHA1_VERSION 0)
    message(WARNING "Git rev-parse failed, set SCAL_SHA1_VERSION to 0")
endif()

exec_program(
    "date"
    ${CMAKE_CURRENT_SOURCE_DIR}
    ARGS ""
    OUTPUT_VARIABLE SCAL_BUILD_TIME )

# generate version.cpp
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/version.cpp.in" "${CMAKE_CURRENT_BINARY_DIR}/version.cpp" @ONLY)

# 1.build json2cpp
add_custom_command(
    OUTPUT json2cpp.bin
    COMMAND ${CMAKE_CXX_COMPILER} ARGS -O2 -std=c++17 ${CMAKE_CURRENT_SOURCE_DIR}/../tools/json2cpp/json2cpp.cpp -o ${CMAKE_CURRENT_BINARY_DIR}/json2cpp.bin
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../tools/json2cpp/json2cpp.cpp
    VERBATIM)
# 2. collect all the config files to inject into binary
file(GLOB CONFIGS "${CMAKE_CURRENT_SOURCE_DIR}/../configs/**/*.json")
#call json2cpp at build time (if json was changed - it should be rebuilt into cpp)
add_custom_command(
    OUTPUT internal_jsons.cpp internal_jsons.h
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/json2cpp.bin ARGS ${CMAKE_CURRENT_BINARY_DIR}/internal_jsons.cpp  ${CMAKE_CURRENT_SOURCE_DIR}/../configs/ ${CONFIGS}
    DEPENDS json2cpp.bin ${CONFIGS}
    VERBATIM)

#source files - compile all files with *.cpp suffix
file(GLOB SRC
            logger/*.cpp
            common/*.cpp
            platform/gaudi2/*.cpp
            platform/gaudi3/*.cpp
            infra/packets/common/*.cpp
            infra/*.cpp
            ${CMAKE_CURRENT_BINARY_DIR}/version.cpp)
set(SRC ${SRC} ${CMAKE_CURRENT_BINARY_DIR}/internal_jsons.cpp)

file(GLOB PUBLIC_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/../include/*.h")
file(GLOB PRIVATE_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/*.h" "${CMAKE_CURRENT_SOURCE_DIR}/common/*.h" "${CMAKE_CURRENT_SOURCE_DIR}/logger/*.h")
set(HEADERS "${PUBLIC_HEADERS}" "${PRIVATE_HEADERS}" ${CMAKE_CURRENT_BINARY_DIR}/internal_jsons.h common/scal_data.h)

include_directories(SYSTEM $ENV{SPDLOG_ROOT})
include_directories(
        $ENV{SPECS_ROOT_H9}
        $ENV{SPECS_ROOT}
        $ENV{HABANALABS_ROOT}/include/uapi
        $ENV{HLTHUNK_ROOT}/include/uapi
        $ENV{ENGINES_FW_ROOT}/include
        $ENV{ENGINES_FW_ROOT}/include/gaudi3
        ${CMAKE_SOURCE_DIR}/src/klib
        ${CMAKE_CURRENT_SOURCE_DIR}/common
        ${CMAKE_CURRENT_SOURCE_DIR}/platform
        ${CMAKE_CURRENT_BINARY_DIR}
        )

source_group("Header files" FILES "${HEADERS}")

if (NOT MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall -Werror -Wno-sign-compare")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pipe -g")
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        if (NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS 6.0)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-misleading-indentation")
        endif()
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-command-line-argument -Wno-missing-braces")
    endif()
endif()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -g")

if (TSANITIZE_ON)
    set(SANITIZE "-fsanitize=thread")
endif()

if (SANITIZE_ON)
    if (TSANITIZE_ON)
        message(SEND_ERROR "both tsan and asan are on. only one of the can work" )
    endif()
    set(SANITIZE "${SANITIZE} -fno-omit-frame-pointer -fsanitize=address")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SANITIZE}")

if (VALGRIND_ON)
    include(CheckIncludeFile)

    CHECK_INCLUDE_FILE("valgrind/memcheck.h" HAVE_VALGRIND_MEMCHECK)
    if (NOT HAVE_VALGRIND_MEMCHECK)
        message(FATAL_ERROR "valgrind/memcheck.h wasn't found")
    endif()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_VALGRIND=1")
    message(STATUS "Enabling Valgrind")
endif()

if (SANITIZE_ON)
    message(STATUS "Enabling Sanitizer")
    set(PROTOBUF_SANITIZE_FLAGS "-fno-sanitize=vptr")
    set(SANITIZE "-fsanitize=undefined ${PROTOBUF_SANITIZE_FLAGS}")
    set(SANITIZE "${SANITIZE} -fno-omit-frame-pointer -fsanitize=address")

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SANITIZE}")

endif()

# Build a library from all specified source files
add_library(${SCAL_TARGET} SHARED ${SRC} "${HEADERS}")

include(cmake/SeparateDebugSymbols.cmake)
separate_debug_symbols(${SCAL_TARGET})

target_compile_definitions(${SCAL_TARGET} PUBLIC JSON_DIAGNOSTICS=1 JSON_USE_IMPLICIT_CONVERSIONS=0)
target_link_libraries(${SCAL_TARGET} ${hlthunk} ${hl_logger})
set_target_properties(${SCAL_TARGET} PROPERTIES PUBLIC_HEADER "${PUBLIC_HEADERS}")

install(
    TARGETS ${SCAL_TARGET}
    EXPORT ${SCAL_TARGET}Targets
    PUBLIC_HEADER DESTINATION include
    RUNTIME DESTINATION bin
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    INCLUDES DESTINATION include
)
install(
    EXPORT ${SCAL_TARGET}Targets
    FILE ${SCAL_TARGET}Targets.cmake
    DESTINATION "lib/cmake/${SCAL_TARGET}"
)
include(CMakePackageConfigHelpers)
configure_package_config_file(
    cmake/${SCAL_TARGET}Config.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/${SCAL_TARGET}Config.cmake"
    INSTALL_DESTINATION "lib/cmake/${SCAL_TARGET}"
)
install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/${SCAL_TARGET}Config.cmake"
    DESTINATION "lib/cmake/${SCAL_TARGET}"
)
