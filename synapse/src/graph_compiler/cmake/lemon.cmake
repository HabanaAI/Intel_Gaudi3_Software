#
# build and export lemon target
#

if (NOT TARGET lemon_project)
    include(ExternalProject)

    ExternalProject_Add(lemon_project
        URL "$ENV{HABANA_SOFTWARE_STACK}/3rd-parties/lemon/"
        TIMEOUT 10
        CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                -Dgtest_force_shared_crt=ON
        INSTALL_COMMAND ""
        UPDATE_COMMAND ""
        LOG_DOWNLOAD ON
        LOG_CONFIGURE ON
        LOG_BUILD ON
        BUILD_BYPRODUCTS "${binary_dir}/lemon/libemon.a"
    )
endif()

function(add_lemon_target)
    add_library(lemon STATIC IMPORTED)

    # Add include path to lemon headers. We are adding the binary directory (in addition to the obvious
    # source directory) since the lemon's config.h file is a build-artifact that is generated in the
    # binary directory and instead of copying it back to the source, we just include the binary dir.
    ExternalProject_Get_Property(lemon_project source_dir)
    ExternalProject_Get_Property(lemon_project binary_dir)
    list(APPEND lemon_exported_dirs ${source_dir})
    list(APPEND lemon_exported_dirs ${binary_dir})
    set_target_properties(lemon PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${lemon_exported_dirs}"
    )

    set_property(TARGET lemon PROPERTY IMPORTED_LOCATION ${binary_dir}/lemon/libemon.a)
    add_dependencies(lemon lemon_project)
endfunction()

add_lemon_target()