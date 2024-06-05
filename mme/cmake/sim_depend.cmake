if (SIM_DEPEND)

    set(hl_thunk $ENV{BUILD_ROOT_LATEST}/libhl-thunk.so)
    set(tpc_api $ENV{BUILD_ROOT_LATEST}/libtpcsim_shared.so)
    set(tpc_kernel_api $ENV{BUILD_ROOT_LATEST}/libTpcElfReader.so)
    # codec libraries
    set(CODEC_BUILD_ROOT "CODEC")
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(CODEC_BUILD_ROOT "${CODEC_BUILD_ROOT}_DEBUG_BUILD")
    else()
        set(CODEC_BUILD_ROOT "${CODEC_BUILD_ROOT}_RELEASE_BUILD")
    endif()
    link_directories($ENV{${CODEC_BUILD_ROOT}}/lib)
    set(codec_model $ENV{${CODEC_BUILD_ROOT}}/lib/libdwl_cmodel.a $ENV{${CODEC_BUILD_ROOT}}/lib/libcommon_cmodel.a $ENV{${CODEC_BUILD_ROOT}}/lib/libvc8kd.a)

endif()