#
# build and export mme_stack, mme_reference and fma targets
#
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    if (SANITIZE_ON)
        set(mme_build_dir $ENV{MME_DEBUG_SANITIZER_BUILD})
    else()
        set(mme_build_dir $ENV{MME_DEBUG_BUILD})
    endif()
else()
    set(mme_build_dir $ENV{MME_RELEASE_BUILD})
endif()

if (EXISTS ${mme_build_dir}/lib/libmme_stack.so)
    add_library(MmeStack SHARED IMPORTED)
    add_library(mme_stack SHARED IMPORTED)
    set_property(TARGET mme_stack PROPERTY IMPORTED_LOCATION ${mme_build_dir}/lib/libmme_stack.so)
    add_dependencies(mme_stack MmeStack)

    add_library(mme_reference SHARED IMPORTED)
    set_property(TARGET mme_reference PROPERTY IMPORTED_LOCATION ${mme_build_dir}/lib/libmme_reference.so)
    add_dependencies(mme_reference MmeStack)
    add_library(fma SHARED IMPORTED)
    set_property(TARGET fma PROPERTY IMPORTED_LOCATION ${mme_build_dir}/lib/libfma.so)

    SET_PROPERTY(TARGET mme_reference 
                 APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES $ENV{MME_ROOT} 
                                                               $ENV{MME_ROOT}/mme_reference)
    SET_PROPERTY(TARGET fma
                 APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES $ENV{MME_ROOT} 
                                                               $ENV{CORAL_SIM_ROOT}/header/gaudi2/fs_core/
                                                               $ENV{CORAL_SIM_ROOT}/header/gaudi3/fs_core/)    
else()

    if (NOT TARGET MmeStack)
        include(ExternalProject)


        ExternalProject_Add(MmeStack
            TIMEOUT 10
            CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} "-DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}" -DCMAKE_COLOR_MAKEFILE=${CMAKE_COLOR_MAKEFILE} -DSANITIZE_ON=${SANITIZE_ON} -DVALGRIND_ON=${VALGRIND_ON} -DMME_VER=OFF -DSYN_DEPEND=ON -DTESTS_ENABLED=OFF
            SOURCE_DIR "$ENV{MME_ROOT}"
            BINARY_DIR "${mme_build_dir}"
            INSTALL_COMMAND ""
            UPDATE_COMMAND ""
            LOG_DOWNLOAD ON
            LOG_CONFIGURE ON
            LOG_BUILD ON
            BUILD_BYPRODUCTS "${mme_build_dir}/lib/libmme_stack.a ${mme_build_dir}/lib/libmme_reference.a ${mme_build_dir}/lib/libfma.a"
        )
    endif()

    add_library(mme_stack STATIC IMPORTED)
    set_property(TARGET mme_stack PROPERTY IMPORTED_LOCATION ${mme_build_dir}/lib/libmme_stack.a)
    add_dependencies(mme_stack MmeStack)

    add_library(mme_reference STATIC IMPORTED)
    set_property(TARGET mme_reference PROPERTY IMPORTED_LOCATION ${mme_build_dir}/lib/libmme_reference.a)
    add_dependencies(mme_reference MmeStack)
    add_library(fma STATIC IMPORTED)
    set_property(TARGET fma PROPERTY IMPORTED_LOCATION ${mme_build_dir}/lib/libfma.a)
    add_dependencies(fma MmeStack)

    SET_PROPERTY(TARGET mme_reference 
                 APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES $ENV{MME_ROOT} 
                                                               $ENV{MME_ROOT}/mme_reference)
    SET_PROPERTY(TARGET fma
                 APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES $ENV{MME_ROOT} 
                                                               $ENV{CORAL_SIM_ROOT}/header/gaudi2/fs_core/
                                                               $ENV{CORAL_SIM_ROOT}/header/gaudi3/fs_core/)
endif()
