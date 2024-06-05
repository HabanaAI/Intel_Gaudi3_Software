cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(hl_gcfg_tests)

set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}/tests")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(gtest_lib_suffix "d")
else()
    set(gtest_lib_suffix "")
endif()

include(ExternalProject)
ExternalProject_Add(
        googletest
        URL "$ENV{HABANA_SOFTWARE_STACK}/3rd-parties/googletest_1_13/"
        TIMEOUT 10
        CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -Dgtest_force_shared_crt=ON -DCMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}
        INSTALL_COMMAND ""
        UPDATE_COMMAND ""
        LOG_DOWNLOAD ON
        LOG_CONFIGURE ON
        LOG_BUILD ON)

ExternalProject_Get_Property(googletest source_dir)
set_target_properties(googletest PROPERTIES EXCLUDE_FROM_ALL TRUE)
ExternalProject_Get_Property(googletest binary_dir)
set(Suffix ".a")
set(Pthread "-pthread")
set (libPath "")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(libSuffix "d")
else()
    set(libSuffix "")
endif()


set(TARGET hl_gcfg_tests)

file(GLOB_RECURSE SRC *.cpp *.c )
file(GLOB HDR *.h *.hpp) #add headers
source_group("Header files" FILES ${HDR}) #group them nicely

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=default -rdynamic")

add_executable(${TARGET} ${SRC})
add_dependencies(${TARGET} googletest)

target_include_directories(${TARGET}  PRIVATE $ENV{HL_LOGGER_INCLUDE_DIRS} ${source_dir}/googletest/include $ENV{SWTOOLS_SDK_ROOT}/hl_gcfg/include)

target_link_libraries(
        ${TARGET}
        ${binary_dir}/lib/${libPath}${CMAKE_FIND_LIBRARY_PREFIXES}gtest${libSuffix}${Suffix}
        hl_gcfg
)

