if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  if (EXISTS $ENV{CORAL_SIM_DEBUG_BUILD})
    set(CORAL_SIM_BUILD_ROOT $ENV{CORAL_SIM_DEBUG_BUILD})
  else()
    set(CORAL_SIM_BUILD_ROOT $ENV{BUILD_ROOT_LATEST})
  endif()
  if (EXISTS $ENV{FUNC_SIM5_DEBUG_BUILD})
    set(FUNC_SIM5_BUILD_ROOT $ENV{FUNC_SIM5_DEBUG_BUILD})
  else()
    set(FUNC_SIM5_BUILD_ROOT $ENV{BUILD_ROOT_LATEST})
  endif()
else()
    set(CORAL_SIM_BUILD_ROOT $ENV{CORAL_SIM_RELEASE_BUILD})
    set(FUNC_SIM5_BUILD_ROOT $ENV{FUNC_SIM5_RELEASE_BUILD})
endif()

set(CORAL_INCLUDES  $ENV{CORAL_SIM_ROOT}/coral_user/common/header
                    $ENV{CORAL_SIM_ROOT}/header/common/fs_core
                    $ENV{CORAL_SIM_ROOT}/coral_infra
                    $ENV{CORAL_SIM_ROOT}/header/common
                    $ENV{CORAL_SIM_ROOT}/header/
                    $ENV{SPECS_ROOT}/common
                    )
set(FUNC_SIM5_INCLUDES  $ENV{FUNC_SIM5_ROOT}
                        $ENV{FUNC_SIM5_ROOT}/fs_core/header
                        $ENV{FUNC_SIM5_ROOT}/fs_core/header/specs_ext
                        $ENV{FUNC_SIM5_ROOT}/fs_core/header/mme
                        $ENV{FUNC_SIM5_ROOT}/fs_user/header
                        $ENV{FUNC_SIM5_ROOT}/fs_log/header
                        $ENV{SPECS_ROOT}/common
                        )

set(CORAL_GAUDI_INCLUDES    $ENV{CORAL_SIM_ROOT}/coral_user/gaudi/header
                            $ENV{CORAL_SIM_ROOT}/header/gaudi
                            $ENV{CORAL_SIM_ROOT}/header/gaudi/fs_core
                            $ENV{SPECS_ROOT}/gaudi)
set(CORAL_GAUDI2_INCLUDES   $ENV{CORAL_SIM_ROOT}/coral_user/gaudi2/header
                            $ENV{CORAL_SIM_ROOT}/header/gaudi2
                            $ENV{CORAL_SIM_ROOT}/header/gaudi2/fs_core)
set(CORAL_GAUDI3_INCLUDES   $ENV{CORAL_SIM_ROOT}/coral_user/gaudi3/header
                            $ENV{CORAL_SIM_ROOT}/header/gaudi3
                            $ENV{CORAL_SIM_ROOT}/header/gaudi3/fs_core)

if (NOT TARGET coral_gaudi2)
    add_library(coral_user_gaudi2 SHARED IMPORTED)
    set_property(TARGET coral_user_gaudi2 PROPERTY IMPORTED_LOCATION ${CORAL_SIM_BUILD_ROOT}/libcoral_user_gaudi2.so)
    add_library(coral_core_gaudi2 SHARED IMPORTED)
    set_property(TARGET coral_core_gaudi2 PROPERTY IMPORTED_LOCATION ${CORAL_SIM_BUILD_ROOT}/libcoral_core_gaudi2.so)
    add_library(coral_gaudi2 INTERFACE)
  if(EXISTS "${CORAL_SIM_BUILD_ROOT}/libarc_core_g2.so")
    add_library(coral_arc_core_gaudi2 SHARED IMPORTED)
    set_property(TARGET coral_arc_core_gaudi2 PROPERTY IMPORTED_LOCATION ${CORAL_SIM_BUILD_ROOT}/libarc_core_g2.so)
    target_link_libraries(coral_gaudi2 INTERFACE coral_user_gaudi2
                                                 coral_core_gaudi2
                                                 coral_arc_core_gaudi2)
  else()
    target_link_libraries(coral_gaudi2 INTERFACE coral_user_gaudi2
                                                 coral_core_gaudi2)
  endif()

endif()
if (NOT TARGET coral_gaudi3)
    add_library(coral_user_gaudi3 SHARED IMPORTED)
    set_property(TARGET coral_user_gaudi3 PROPERTY IMPORTED_LOCATION ${CORAL_SIM_BUILD_ROOT}/libcoral_user_gaudi3.so)
    add_library(coral_core_gaudi3 SHARED IMPORTED)
    set_property(TARGET coral_core_gaudi3 PROPERTY IMPORTED_LOCATION ${CORAL_SIM_BUILD_ROOT}/libcoral_core_gaudi3.so)
    add_library(coral_gaudi3 INTERFACE)
  if(EXISTS "${CORAL_SIM_BUILD_ROOT}/libarc_core_g3.so")
    add_library(coral_arc_core_gaudi3 SHARED IMPORTED)
    set_property(TARGET coral_arc_core_gaudi3 PROPERTY IMPORTED_LOCATION ${CORAL_SIM_BUILD_ROOT}/libarc_core_g3.so)
    add_library(coral_core_psoc_gaudi3 SHARED IMPORTED)
    set_property(TARGET coral_core_psoc_gaudi3 PROPERTY IMPORTED_LOCATION ${CORAL_SIM_BUILD_ROOT}/libpsoc_g3.so)
    target_link_libraries(coral_gaudi3 INTERFACE coral_user_gaudi3
                                                 coral_core_gaudi3
                                                 coral_arc_core_gaudi3
                                                 coral_core_psoc_gaudi3)
  else()
    target_link_libraries(coral_gaudi3 INTERFACE coral_user_gaudi3
                                                 coral_core_gaudi3)
  endif()
endif()

if (NOT TARGET coral_gaudi)
    add_library(coral_user_gaudi SHARED IMPORTED)
    set_property(TARGET coral_user_gaudi PROPERTY IMPORTED_LOCATION ${CORAL_SIM_BUILD_ROOT}/libcoral_user_gaudi.so)
    add_library(coral_core_gaudi SHARED IMPORTED)
    set_property(TARGET coral_core_gaudi PROPERTY IMPORTED_LOCATION ${CORAL_SIM_BUILD_ROOT}/libcoral_core_gaudi.so)
    add_library(coral_gaudi INTERFACE)
    target_link_libraries(coral_gaudi INTERFACE
                                                coral_user_gaudi
                                                coral_core_gaudi)
endif()