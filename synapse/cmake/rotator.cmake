#
# build and export rotator target
#

if (NOT TARGET RotatorStack)
    include(ExternalProject)

    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(rotator_build_dir $ENV{ROTATOR_DEBUG_BUILD})
    else()
        set(rotator_build_dir $ENV{ROTATOR_RELEASE_BUILD})
    endif()

    ExternalProject_Add(RotatorStack
        TIMEOUT 10
        CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} "-DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}" -DCMAKE_COLOR_MAKEFILE=${CMAKE_COLOR_MAKEFILE} -DSANITIZE_ON=${SANITIZE_ON} -DVALGRIND_ON=${VALGRIND_ON} -DRUN_WITH_SYNAPSE=1
        SOURCE_DIR "$ENV{ROTATOR_ROOT}"
        BINARY_DIR "${rotator_build_dir}"
        INSTALL_COMMAND ""
        UPDATE_COMMAND ""
        LOG_DOWNLOAD OFF
        LOG_CONFIGURE ON
        LOG_BUILD ON
    )
endif()

add_library(rotator_stack STATIC IMPORTED)
set_property(TARGET rotator_stack PROPERTY IMPORTED_LOCATION ${rotator_build_dir}/lib/librotator_lib.a)
SET_PROPERTY(TARGET rotator_stack
             APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES $ENV{ROTATOR_ROOT})
add_dependencies(rotator_stack RotatorStack)

