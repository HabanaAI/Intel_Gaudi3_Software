message("Building logger")

exec_program(
        "git"
        ${CMAKE_CURRENT_SOURCE_DIR}
        ARGS "rev-parse --short HEAD"
        OUTPUT_VARIABLE SWTOOLS_SDK_SHA1_VERSION
        RETURN_VALUE RET)

if (NOT ${RET} EQUAL 0)
    set (SWTOOLS_SDK_SHA1_VERSION 0)
    message(WARNING "Git rev-parse failed, set SCAL_SHA1_VERSION to 0")
endif()

# generate version.cpp
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/version.cpp.in" "${CMAKE_CURRENT_BINARY_DIR}/version.cpp" @ONLY)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SANITIZE}")

add_library(hl_logger SHARED hllog_core.cpp hllog_se.cpp hllog_core_compatibility.cpp ${CMAKE_CURRENT_BINARY_DIR}/version.cpp)

set(LINKER_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/libhl_logger.version")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-narrowing -Werror -Wall -Wno-sign-compare -Wl,--version-script=${LINKER_SCRIPT} -fvisibility=hidden -rdynamic")

target_link_libraries(hl_logger PUBLIC pthread) # dl backtrace) for boost stacktrace
target_include_directories(hl_logger PRIVATE $ENV{SPDLOG_1_10_0_ROOT}
                                     PUBLIC   ../include $ENV{THIRD_PARTIES_ROOT})
