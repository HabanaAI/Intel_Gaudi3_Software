function hl_logger_build_help()
{
    echo -e "build_hl_logger              -    Build hl_logger binaries and tests"
}


function hl_logger_build_usage()
{
    if [ $1 == "build_hl_logger" ]; then
        echo -e "\nusage: $1 [options]\n"

        echo -e "options:\n"
        echo -e "  -a,  --build-all            Build both debug and release build"
        echo -e "       --no-color             Disable colors in output"
        echo -e "  -c,  --configure            Configure before build"
        echo -e "  -r,  --release              Build only release build"
        echo -e "  -s,  --sanitize             Build with sanitize flags on"
        echo -e "  -v,  --verbose              Build with verbose"
        echo -e "  -j,  --jobs <val>           Overwrite number of jobs"
        echo -e "  -l                          Build hl_logger only (without tests)"
        echo -e "  -h,  --help                 Prints this help"
    fi
}


build_hl_logger ()
{
    _verify_exists_dir "$SWTOOLS_SDK_ROOT" $SWTOOLS_SDK_ROOT
    : "${SWTOOLS_SDK_DEBUG_BUILD:?Need to set SWTOOLS_SDK_DEBUG_BUILD to the build folder}"
    : "${SWTOOLS_SDK_RELEASE_BUILD:?Need to set SWTOOLS_SDK_RELEASE_BUILD to the build folder}"
    local HL_LOGGER_DEBUG_BUILD=${SWTOOLS_SDK_DEBUG_BUILD}/hl_logger
    local HL_LOGGER_RELEASE_BUILD=${SWTOOLS_SDK_RELEASE_BUILD}/hl_logger
    local __scriptname=$(__get_func_name)

    local __jobs=$NUMBER_OF_JOBS
    local __color="ON"
    local __debug="yes"
    local __release=""
    local __verbose=""
    local __all=""
    local __configure=""
    local __org_configure=""
    local __sanitize="NO"
    local __build_res=""
    local __hl_logger_tests="ON"

    # parameter while-loop
    while [ -n "$1" ];
    do
        case $1 in
        -a  | --build-all )
            __all="yes"
            ;;
        --no-color )
            __color="NO"
            ;;
        -c  | --configure )
            __org_configure="yes"
            __configure="yes"
            ;;
        -r  | --release )
            __debug=""
            __release="yes"
            ;;
        -v  | --verbose )
            __verbose="VERBOSE=1"
            ;;
        -j  | --jobs )
            shift
            __jobs=$1
            ;;
        -l  | --hl_logger_only  )
            __hl_logger_tests="OFF"
            ;;
        -s  | --sanitize )
            __sanitize="ON"
            ;;
        -p  | --production )
            ;; # use on demand
        -h  | --help )
            hl_logger_build_usage $__scriptname
            return 0
            ;;
        *)
            echo "The parameter $1 is not allowed"
            hl_logger_build_usage $__scriptname
            return 1 # error
            ;;
        esac
        shift
    done

    if [ -n "$__configure" ]; then
        __check_mandatory_pkgs
        if [ $? -ne 0 ]; then
            return 1
        fi
    fi

    if [ -n "$__all" ]; then
        __debug="yes"
        __release="yes"
    fi

    if [ -n "$__debug" ]; then
        echo -e "Building in debug mode"
        if [ ! -d $HL_LOGGER_DEBUG_BUILD ]; then
            __configure="yes"
        fi

        if [ -n "$__configure" ]; then
            if [ -d $HL_LOGGER_DEBUG_BUILD ]; then
                rm -rf $HL_LOGGER_DEBUG_BUILD
            fi

            mkdir -p $HL_LOGGER_DEBUG_BUILD
        fi

        _verify_exists_dir "$HL_LOGGER_DEBUG_BUILD" $HL_LOGGER_DEBUG_BUILD
        pushd $HL_LOGGER_DEBUG_BUILD
            (set -x; cmake \
            -DCMAKE_BUILD_TYPE="Debug" \
            -DCMAKE_INSTALL_PREFIX=$HOME/.local \
            -DCMAKE_COLOR_MAKEFILE=$__color \
            -DTESTS_ENABLED=$__hl_logger_tests \
            -DSANITIZE_ON=$__sanitize \
            ${SWTOOLS_SDK_ROOT}/hl_logger/)
        make $__verbose -j$__jobs
        __build_res=$?
        popd
        if [ $__build_res -ne 0 ]; then
            return $__build_res
        fi
        mkdir -p $SWTOOLS_SDK_DEBUG_BUILD/lib/
        cp -fs $HL_LOGGER_DEBUG_BUILD/lib/* $SWTOOLS_SDK_DEBUG_BUILD/lib/
        if [ "$__hl_logger_tests" == "ON" ]; then
            mkdir -p $SWTOOLS_SDK_DEBUG_BUILD/tests/
            cp -fs $HL_LOGGER_DEBUG_BUILD/tests/hl_logger_tests $SWTOOLS_SDK_DEBUG_BUILD/tests/
        fi
        _copy_build_products $HL_LOGGER_DEBUG_BUILD
    fi

    __configure=$__org_configure

    if [ -n "$__release" ]; then
        echo "Building in release mode"
        if [ ! -d $HL_LOGGER_RELEASE_BUILD ]; then
            __configure="yes"
        fi

        if [ -n "$__configure" ]; then
            if [ -d $HL_LOGGER_RELEASE_BUILD ]; then
                rm -rf $HL_LOGGER_RELEASE_BUILD
            fi
            mkdir -p $HL_LOGGER_RELEASE_BUILD
        fi

        _verify_exists_dir "$HL_LOGGER_RELEASE_BUILD" $HL_LOGGER_RELEASE_BUILD
        pushd $HL_LOGGER_RELEASE_BUILD
        (set -x; cmake \
            -DCMAKE_BUILD_TYPE="Release" \
            -DCMAKE_INSTALL_PREFIX=$HOME/.local \
            -DCMAKE_COLOR_MAKEFILE=$__color \
            -DTESTS_ENABLED=$__hl_logger_tests \
            -DSANITIZE_ON=$__sanitize \
            ${SWTOOLS_SDK_ROOT}/hl_logger)
        make $__verbose -j$__jobs
        __build_res=$?
        popd
        if [ $__build_res -ne 0 ]; then
            return $__build_res
        fi
        mkdir -p $SWTOOLS_SDK_RELEASE_BUILD/lib/
        cp -fs $HL_LOGGER_RELEASE_BUILD/lib/* $SWTOOLS_SDK_RELEASE_BUILD/lib/
        if [ "$__hl_logger_tests" == "ON" ]; then
            mkdir -p $SWTOOLS_SDK_RELEASE_BUILD/tests/
            cp -fs $HL_LOGGER_RELEASE_BUILD/tests/hl_logger_tests $SWTOOLS_SDK_RELEASE_BUILD/tests/
        fi
        _copy_build_products $HL_LOGGER_RELEASE_BUILD -r
    fi

    return 0
}

function hl_gcfg_build_help()
{
    echo -e "build_hl_gcfg              -    Build hl_gcfg binaries and tests"
}


function hl_gcfg_build_usage()
{
    if [ $1 == "build_hl_gcfg" ]; then
        echo -e "\nusage: $1 [options]\n"

        echo -e "options:\n"
        echo -e "  -a,  --build-all            Build both debug and release build"
        echo -e "       --no-color             Disable colors in output"
        echo -e "  -c,  --configure            Configure before build"
        echo -e "  -r,  --release              Build only release build"
        echo -e "  -s,  --sanitize             Build with sanitize flags on"
        echo -e "  -v,  --verbose              Build with verbose"
        echo -e "  -j,  --jobs <val>           Overwrite number of jobs"
        echo -e "  -l                          Build hl_gcfg only (without tests)"
        echo -e "  -h,  --help                 Prints this help"
    fi
}


build_hl_gcfg ()
{
    _verify_exists_dir "$SWTOOLS_SDK_ROOT" $SWTOOLS_SDK_ROOT
    : "${SWTOOLS_SDK_DEBUG_BUILD:?Need to set SWTOOLS_SDK_DEBUG_BUILD to the build folder}"
    : "${SWTOOLS_SDK_RELEASE_BUILD:?Need to set SWTOOLS_SDK_RELEASE_BUILD to the build folder}"
    local HL_GCFG_DEBUG_BUILD=${SWTOOLS_SDK_DEBUG_BUILD}/hl_gcfg
    local HL_GCFG_RELEASE_BUILD=${SWTOOLS_SDK_RELEASE_BUILD}/hl_gcfg
    local __scriptname=$(__get_func_name)

    local __jobs=$NUMBER_OF_JOBS
    local __color="ON"
    local __debug="yes"
    local __release=""
    local __verbose=""
    local __all=""
    local __configure=""
    local __org_configure=""
    local __sanitize="NO"
    local __build_res=""
    local __hl_gcfg_tests="ON"

    # parameter while-loop
    while [ -n "$1" ];
    do
        case $1 in
        -a  | --build-all )
            __all="yes"
            ;;
        --no-color )
            __color="NO"
            ;;
        -c  | --configure )
            __org_configure="yes"
            __configure="yes"
            ;;
        -r  | --release )
            __debug=""
            __release="yes"
            ;;
        -v  | --verbose )
            __verbose="VERBOSE=1"
            ;;
        -j  | --jobs )
            shift
            __jobs=$1
            ;;
        -l  | --hl_gcfg_only  )
            __hl_gcfg_tests="OFF"
            ;;
        -s  | --sanitize )
            __sanitize="ON"
            ;;
        -p  | --production )
            ;; # use on demand
        -h  | --help )
            hl_gcfg_build_usage $__scriptname
            return 0
            ;;
        *)
            echo "The parameter $1 is not allowed"
            hl_gcfg_build_usage $__scriptname
            return 1 # error
            ;;
        esac
        shift
    done

    if [ -n "$__configure" ]; then
        __check_mandatory_pkgs
        if [ $? -ne 0 ]; then
            return 1
        fi
    fi

    if [ -n "$__all" ]; then
        __debug="yes"
        __release="yes"
    fi

    if [ -n "$__debug" ]; then
        echo -e "Building in debug mode"
        if [ ! -d $HL_GCFG_DEBUG_BUILD ]; then
            __configure="yes"
        fi

        if [ -n "$__configure" ]; then
            if [ -d $HL_GCFG_DEBUG_BUILD ]; then
                rm -rf $HL_GCFG_DEBUG_BUILD
            fi

            mkdir -p $HL_GCFG_DEBUG_BUILD
        fi

        _verify_exists_dir "$HL_GCFG_DEBUG_BUILD" $HL_GCFG_DEBUG_BUILD
        pushd $HL_GCFG_DEBUG_BUILD
            (set -x; cmake \
            -DCMAKE_BUILD_TYPE="Debug" \
            -DCMAKE_INSTALL_PREFIX=$HOME/.local \
            -DCMAKE_COLOR_MAKEFILE=$__color \
            -DTESTS_ENABLED=$__hl_gcfg_tests \
            -DSANITIZE_ON=$__sanitize \
            ${SWTOOLS_SDK_ROOT}/hl_gcfg/)
        make $__verbose -j$__jobs
        __build_res=$?
        popd
        if [ $__build_res -ne 0 ]; then
            return $__build_res
        fi
        mkdir -p $SWTOOLS_SDK_DEBUG_BUILD/lib/
        cp -fs $HL_GCFG_DEBUG_BUILD/lib/* $SWTOOLS_SDK_DEBUG_BUILD/lib/
        if [ "$__hl_gcfg_tests" == "ON" ]; then
            mkdir -p $SWTOOLS_SDK_DEBUG_BUILD/tests/
            cp -fs $HL_GCFG_DEBUG_BUILD/tests/hl_gcfg_tests $SWTOOLS_SDK_DEBUG_BUILD/tests/
        fi
        _copy_build_products $HL_GCFG_DEBUG_BUILD
    fi

    __configure=$__org_configure

    if [ -n "$__release" ]; then
        echo "Building in release mode"
        if [ ! -d $HL_GCFG_RELEASE_BUILD ]; then
            __configure="yes"
        fi

        if [ -n "$__configure" ]; then
            if [ -d $HL_GCFG_RELEASE_BUILD ]; then
                rm -rf $HL_GCFG_RELEASE_BUILD
            fi
            mkdir -p $HL_GCFG_RELEASE_BUILD
        fi

        _verify_exists_dir "$HL_GCFG_RELEASE_BUILD" $HL_GCFG_RELEASE_BUILD
        pushd $HL_GCFG_RELEASE_BUILD
        (set -x; cmake \
            -DCMAKE_BUILD_TYPE="Release" \
            -DCMAKE_INSTALL_PREFIX=$HOME/.local \
            -DCMAKE_COLOR_MAKEFILE=$__color \
            -DTESTS_ENABLED=$__hl_gcfg_tests \
            -DSANITIZE_ON=$__sanitize \
            ${SWTOOLS_SDK_ROOT}/hl_gcfg)
        make $__verbose -j$__jobs
        __build_res=$?
        popd
        if [ $__build_res -ne 0 ]; then
            return $__build_res
        fi
        mkdir -p $SWTOOLS_SDK_RELEASE_BUILD/lib/
        cp -fs $HL_GCFG_RELEASE_BUILD/lib/* $SWTOOLS_SDK_RELEASE_BUILD/lib/
        if [ "$__hl_gcfg_tests" == "ON" ]; then
            mkdir -p $SWTOOLS_SDK_RELEASE_BUILD/tests/
            cp -fs $HL_GCFG_RELEASE_BUILD/tests/hl_gcfg_tests $SWTOOLS_SDK_RELEASE_BUILD/tests/
        fi
        _copy_build_products $HL_GCFG_RELEASE_BUILD -r
    fi

    return 0
}

function swtools_sdk_build_help()
{
    echo -e "build_swtools_sdk              -    Build SW Tools SDK libraries, binaries and tests"
}


function swtools_sdk_build_usage()
{
    if [ $1 == "build_swtools_sdk" ]; then
        echo -e "\nusage: $1 [options]\n"

        echo -e "options:\n"
        echo -e "  -a,  --build-all            Build both debug and release build"
        echo -e "       --no-color             Disable colors in output"
        echo -e "  -c,  --configure            Configure before build"
        echo -e "  -r,  --release              Build only release build"
        echo -e "  -s,  --sanitize             Build with sanitize flags on"
        echo -e "  -v,  --verbose              Build with verbose"
        echo -e "  -j,  --jobs <val>           Overwrite number of jobs"
        echo -e "  -p,  --production           build for production"
        echo -e "  -l                          Build swtools_sdk only (without tests)"
        echo -e "  -h,  --help                 Prints this help"
    fi
}

build_swtools_sdk ()
{
    SECONDS=0

    _verify_exists_dir "$SWTOOLS_SDK_ROOT" $SWTOOLS_SDK_ROOT
    : "${SWTOOLS_SDK_DEBUG_BUILD:?Need to set SWTOOLS_SDK_DEBUG_BUILD to the build folder}"
    : "${SWTOOLS_SDK_RELEASE_BUILD:?Need to set SWTOOLS_SDK_RELEASE_BUILD to the build folder}"

    local __scriptname=$(__get_func_name)

    local __jobs=$NUMBER_OF_JOBS
    local __color="ON"
    local __debug="yes"
    local __release=""
    local __verbose=""
    local __all=""
    local __configure=""
    local __org_configure=""
    local __sanitize="NO"
    local __build_res=""
    local __swtools_sdk_tests="ON"
    local __saved_params="$@"
    local __production="NO"

    # parameter while-loop
    while [ -n "$1" ];
    do
        case $1 in
        -a  | --build-all )
            __all="yes"
            ;;
        --no-color )
            __color="NO"
            ;;
        -c  | --configure )
            __org_configure="yes"
            __configure="yes"
            ;;
        -r  | --release )
            __debug=""
            __release="yes"
            ;;
        -v  | --verbose )
            __verbose="VERBOSE=1"
            ;;
        -j  | --jobs )
            shift
            __jobs=$1
            ;;
        -l  | --swtools_sdk_only  )
            __swtools_sdk_tests="OFF"
            ;;
        -s  | --sanitize )
            __sanitize="ON"
            ;;
        -p  | --production )
            __production="ON"
            ;;
        -h  | --help )
            swtools_sdk_build_usage $__scriptname
            return 0
            ;;
        *)
            echo "The parameter $1 is not allowed"
            swtools_sdk_build_usage $__scriptname
            return 1 # error
            ;;
        esac
        shift
    done

    if [ -n "$__configure" ]; then
        __check_mandatory_pkgs
        if [ $? -ne 0 ]; then
            return 1
        fi
    fi

    if [ -n "$__all" ]; then
        __debug="yes"
        __release="yes"
    fi

    mkdir -p $HABANA_PLUGINS_LIB_PATH


    if [ -n "$__debug" ]; then
        echo -e "Building in debug mode"
        if [ ! -d $SWTOOLS_SDK_DEBUG_BUILD ]; then
            __configure="yes"
        fi

        if [ -n "$__configure" ]; then
            if [ -d $SWTOOLS_SDK_DEBUG_BUILD ]; then
                rm -rf $SWTOOLS_SDK_DEBUG_BUILD
            fi

            mkdir -p $SWTOOLS_SDK_DEBUG_BUILD
        fi

        build_hl_logger $__saved_params
        build_hl_gcfg $__saved_params

        _verify_exists_dir "$SWTOOLS_SDK_DEBUG_BUILD" $SWTOOLS_SDK_DEBUG_BUILD
        pushd $SWTOOLS_SDK_DEBUG_BUILD
            (set -x; cmake \
            -DCMAKE_BUILD_TYPE="Debug" \
            -DCMAKE_INSTALL_PREFIX=$HOME/.local \
            -DCMAKE_COLOR_MAKEFILE=$__color \
            -DTESTS_ENABLED=$__swtools_sdk_tests \
            -DSANITIZE_ON=$__sanitize \
            -DPRODUCTION_ON=$__production \
            $SWTOOLS_SDK_ROOT)
        make $__verbose -j$__jobs
        __build_res=$?
        popd
        if [ $__build_res -ne 0 ]; then
            return $__build_res
        fi
        _copy_build_products $SWTOOLS_SDK_DEBUG_BUILD
        cp -fs $SWTOOLS_SDK_DEBUG_BUILD/bin/* $BUILD_ROOT_DEBUG
        cp -fs $SWTOOLS_SDK_DEBUG_BUILD/bin/* $BUILD_ROOT_LATEST
        if [ "$__swtools_sdk_tests" == "ON" ]; then
            # dummy plugins for tests
            cp -fs $SWTOOLS_SDK_DEBUG_BUILD/plugins/* $HABANA_PLUGINS_LIB_PATH
            cp -fs $SWTOOLS_SDK_DEBUG_BUILD/plugins/* $BUILD_ROOT_LATEST
        fi
    fi

    __configure=$__org_configure

    if [ -n "$__release" ]; then
        echo "Building in release mode"
        if [ ! -d $SWTOOLS_SDK_RELEASE_BUILD ]; then
            __configure="yes"
        fi

        if [ -n "$__configure" ]; then
            if [ -d $SWTOOLS_SDK_RELEASE_BUILD ]; then
                rm -rf $SWTOOLS_SDK_RELEASE_BUILD
            fi
            mkdir -p $SWTOOLS_SDK_RELEASE_BUILD
        fi

        build_hl_logger $__saved_params
        build_hl_gcfg $__saved_params

        _verify_exists_dir "$SWTOOLS_SDK_RELEASE_BUILD" $SWTOOLS_SDK_RELEASE_BUILD
        pushd $SWTOOLS_SDK_RELEASE_BUILD
        (set -x; cmake \
            -DCMAKE_BUILD_TYPE="Release" \
            -DCMAKE_INSTALL_PREFIX=$HOME/.local \
            -DCMAKE_COLOR_MAKEFILE=$__color \
            -DTESTS_ENABLED=$__swtools_sdk_tests \
            -DSANITIZE_ON=$__sanitize \
            -DPRODUCTION_ON=$__production \
            $SWTOOLS_SDK_ROOT)
        make $__verbose -j$__jobs
        __build_res=$?
        popd
        if [ $__build_res -ne 0 ]; then
            return $__build_res
        fi
        _copy_build_products $SWTOOLS_SDK_RELEASE_BUILD -r
        cp -fs $SWTOOLS_SDK_RELEASE_BUILD/bin/* $BUILD_ROOT_RELEASE
        cp -fs $SWTOOLS_SDK_RELEASE_BUILD/bin/* $BUILD_ROOT_LATEST
        if [ "$__swtools_sdk_tests" == "ON" ]; then
            # dummy plugins for tests
            cp -fs $SWTOOLS_SDK_RELEASE_BUILD/plugins/* $HABANA_PLUGINS_LIB_PATH
            cp -fs $SWTOOLS_SDK_RELEASE_BUILD/plugins/* $BUILD_ROOT_LATEST
        fi
    fi


    printf "\nElapsed time: %02u:%02u:%02u \n\n" $(($SECONDS / 3600)) $((($SECONDS / 60) % 60)) $(($SECONDS % 60))
    return 0
}
