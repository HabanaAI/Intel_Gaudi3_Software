message("Building hl_gcfg")

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

add_library(hl_gcfg SHARED
        hlgcfg.cpp
        logger.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/version.cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-narrowing -Werror -Wall -Wno-sign-compare -fvisibility=hidden -rdynamic")

set(hl_logger $ENV{BUILD_ROOT_LATEST}/libhl_logger.so)

target_link_libraries(hl_gcfg PUBLIC pthread ${hl_logger}) # dl backtrace) for boost stacktrace
target_include_directories(hl_gcfg PRIVATE $ENV{HL_LOGGER_INCLUDE_DIRS}
                                   PUBLIC  $ENV{SWTOOLS_SDK_ROOT}/hl_gcfg/include)
