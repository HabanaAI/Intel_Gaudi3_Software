#!/bin/bash
#
# Copyright (C) 2021 HabanaLabs, Ltd.
# All Rights Reserved.
#
# Unauthorized copying of this file, via any medium is strictly prohibited.
# Proprietary and confidential.
# Author: Dotan Barak <dbarak@habana.ai>
#

_verify_exists_dir()
{
    # note: if #1 is empty, this still works unless the message string is a dir(unlikely)
    local _msg=$1
    local _dirname=$2
    if [ -z $_dirname ];then
        _fatal_msg "dir does not exist(null dir name): $_msg:$2"
        return 1
    fi
    if [ ! -d $_dirname ];then
        _fatal_msg "directory does not exist: $_msg"
        return 1
    fi
    return 0
}

_verify_exists_file()
{
    local _msg=$1
    local _fname=$2

    #echo "@@_verify_exists_file: $#"

    if [ -z "$_fname" ];then
        _fatal_msg "file does not exist(null file name): $_msg"
    fi

    if [ ! -f $_fname ];then
        _fatal_msg "file does not exist: $_msg"
    fi
}

__get_func_name()
{
    if [ -n "$ZSH_NAME" ]; then
        echo ${funcstack[2]}
    else
        echo ${FUNCNAME[1]}
    fi
}

# --- helper functions ---
function synapse_functions_help()
{
    echo -e "\n- The following is a list of available functions in synapse.sh"
    echo -e "build_engines_fw              - Build ARCs FW for Gaudi2 (default = debug)"
    echo -e "build_gaudi_demo              - Build Gaudi resnet demo"
    echo -e "build_habana_tfrt             - Build the habana tensorflow run-time for Goya"
    echo -e "build_mme                     - Build mme stack library and testing environment (default = debug)"
    echo -e "build_scal                    - Build scal  (default = debug)"
    echo -e "build_synapse                 - Build synapse (default = debug)"
    echo -e "build_weight_lib              - Build the weight compression/decompression library (default = debug)"
    echo -e "build_rotator                 - Build the rotator engine library and standalone simulator (default = debug)"
    echo -e "run_demos_tests               - Run demos module tests"
    echo -e "run_from_json                 - Run model from json file"
    echo -e "run_gemm_benchmarks_test      - Run gemm benchmarks tests"
    echo -e "run_graph_optimizer_test      - Run graph_optimizer test (default = debug)"
    echo -e "run_habana_py_test            - Run habana_py test (default = debug)"
    echo -e "run_mme_test                  - Run mme test (default = debug)"
    echo -e "run_models_tests              - Run models tests"
    echo -e "run_runtime_unit_test         - Run runtime test (default = debug)"
    echo -e "run_synapse_test              - Run synapse test (default = debug)"
    echo -e "run_death_test                - Run synapse test script that crashes the device"
    echo -e "run_synapse_mlir_graph_test   - Run synapse mlir test (default = release)"
    echo -e "check_syn_singleton_interface - Run sanity test on synapse interface version"
    echo
}

function synapse_usage()
{
    if [ $1 == "build_synapse" ]; then
        echo -e "\nusage: $1 [options]\n"

        echo -e "options:\n"
        echo -e "  -a,  --build-all            Build both debug and release build"
        echo -e "       --no-color             Disable colors in output"
        echo -e "  -c,  --configure            Configure before build"
        echo -e "  -n,  --cuda                 Enable CUDA module"
        echo -e "       --no-hcl               *** Deprecated ***"
        echo -e "  -r,  --release              Build only release build"
        echo -e "       --recursive            Build all the pre-requisite modules in a recursive way"
        echo -e "       --recursive_st         Build all the pre-requisite modules (Skip Tests where possible) in a recursive way"
        echo -e "  -s,  --sanitize             Build with address sanitize flags on"
        echo -e "  -ts, --tsanitize            Build with thread sanitizer flags on"
        echo -e "  -v,  --verbose              Build with verbose"
        echo -e "  -V,  --valgrind             Build with valgrind flag on"
        echo -e "  -j,  --jobs <val>           Overwrite number of jobs"
        echo -e "       --doc <pdf>            Generates Synapse documentation (skip synapse build)"
        echo -e "  -l                          Build only synapse (without tests)"
        echo -e "  -qa                         Build synapse and all tests(including qa tests)"
        echo -e "  -o,  --run_from_json        Build synapse and only run_from_json tests"
        echo -e "  -y,  --tidy                 Running clang-tidy during build"
        echo -e "  -h,  --help                 Prints this help"
    fi

    if [ $1 == "build_synapse_mlir" ]; then
        echo -e "\nusage: $1 [options]\n"
        echo -e "\nCurrently only release build is supported\n"
        echo -e "options:\n"
        echo -e "  -c,  --configure            Configure before build"
        echo -e "  -r,  --release              Build only release build"
        echo -e "  -l                          Build only synapse_mlir (without tests)"
        echo -e "  -h,  --help                 Prints this help"
    fi

    if [ $1 == "build_mme" ]; then
        echo -e "\nusage: $1 [options]\n"

        echo -e "options:\n"
        echo -e "  -a,  --build-all            Build both debug and release build"
        echo -e "       --no-color             Disable colors in output"
        echo -e "  -c,  --configure            Configure before build"
        echo -e "  -m,  --mme_verification     Enable building mme descriptor generator library tests"
        echo -e "  -d,  --no_depend            Build mme without dependencies to other repos"
        echo -e "  -k,  --no_synapse_depend    Build mme without dependency on synapse"
        echo -e "  -C,  --chip_type CHIP_TYPE  Build only for specific chip type (H3, H6 or gaudi, gaudi2)"
        echo -e "  --compression_stack STACK   link specific compression stack (local, synapse)"
        echo -e "  -r,  --release              Build only release build"
        echo -e "  -s,  --sanitize             Build with address sanitizer flags on"
        echo -e "  -ts, --tsanitize            Build with thread sanitizer flags on"
        echo -e "  -v,  --verbose              Build with verbose"
        echo -e "  -V   --valgrind             Build with valgrind flag on"
        echo -e "  -j,  --jobs <val>           Overwrite number of jobs"
        echo -e "  -l                          Don't build mme unit tests"
        echo -e "  -h,  --help                 Prints this help"
    fi

    if [ $1 == "build_weight_lib" ]; then
        echo -e "\nusage: $1 [options]\n"

        echo -e "options:\n"
        echo -e "  -a,  --build-all            Build both debug and release build"
        echo -e "       --no-color             Disable colors in output"
        echo -e "  -c,  --configure            Configure before build"
        echo -e "  -r,  --release              Build only release build"
        echo -e "  -s,  --sanitize             Build with sanitize flags on"
        echo -e "  -v,  --verbose              Build with verbose"
        echo -e "  -j,  --jobs <val>           Overwrite number of jobs"
        echo -e "  -h,  --help                 Prints this help"
    fi

    if [ $1 == "build_rotator" ]; then
        echo -e "\nusage: $1 [options]\n"

        echo -e "options:\n"
        echo -e "  -a,  --build-all            Build both debug and release build"
        echo -e "       --no-color             Disable colors in output"
        echo -e "  -t,  --target               build a specific target: synapse, sv, coral, standalone (default: synapse)"
        echo -e "  -c,  --configure            Configure before build"
        echo -e "  -r,  --release              Build only release build"
        echo -e "  -s,  --sanitize             Build with sanitize flags on"
        echo -e "  -v,  --verbose              Build with verbose"
        echo -e "  -j,  --jobs <val>           Overwrite number of jobs"
        echo -e "  -h,  --help                 Prints this help"
    fi

    if [ $1 == "run_synapse_test" ]; then
        echo -e "\nusage: $1 [options]\n"
        echo -e "options:\n"
        echo -e "  -g,  --gdbserver                             Run the app with gdbserver"
        echo -e "  -gdb                                         Run the app under gdb"
        echo -e "  -v   --valgrind                              Run the app under valgrind"
        echo -e "  -spdlog LOG_LEVEL                            0 - TRACE, 1 - DEBUG, 2 - INFO, 3 - WARNING, 4 - ERROR, 5 - CRITICAL, 6 - OFF"
        echo -e "  -notpc                                       Don't run tests which require TPC kernels"
        echo -e "  -sudo                                        Run as sudo"
        echo -e "  -shuffle                                     Randomize tests' orders on every iteration"
        echo -e "  -seed                                        Uses reproduce an order-related tests"
        echo -e "  -prof                                        Enable synapse profiler"
        echo -e "  --sanitizer                                  Run synapse sanitizer build tests"
        echo -e "  -mr   --media-recipes                        Dumps recipes for media in local directory"
        echo -e "  -i,  --iterations NUM                        Run NUM iterations"
        echo -e "  -l,  --list-tests                            List the available tests"
        echo -e "  -r,  --release                               Run the tests with the release build"
        echo -e "  -f,  --run-until-failure                     Run test until first failure"
        echo -e "  -s,  --specific-test TEST                    Run TEST"
        echo -e "  -a,  --disable-additional-tests TEST         Run all tests without TEST and any default skipped tests"
        echo -e "  -c,  --chip-type CHIP_TYPE                   Run only tests related to CHIP_TYPE"
        echo -e '  -m,  --mode all|asic|sim|daily|postCommit    Run only tests marked for asic\\simulator\\daily or all tests (default: all)'
        echo -e "  -n   --num_of_devices                        Run only tests that support the number of devices (default: 1)"
        echo -e "  -d   --disabled-test                         Run tests that are disabled"
        echo -e "  -x,  --xml PATH                              Output XML file PATH"
        echo -e "       --no-color                              Disable colors in output"
        echo -e "       --reset-terminal                        Reset the terminal to clear gtest colors on completion"
        echo -e "       --eager                                 Run only eager mode tests"
        echo -e "       --test-group-id                         Run group of test-packages"
        echo -e "       --list-classes                          Prints the available tests-classes"
        echo -e "  -h,  --help                                  Prints this help"
    fi

    if [ $1 == "run_graph_optimizer_test" ]; then
        echo -e "\nusage: $1 [options]\n"
        echo -e "options:\n"
        echo -e "  -g,  --gdbserver            Run the app with gdbserver"
        echo -e "  -i,  --iterations NUM       Run NUM iterations"
        echo -e "  -spdlog LOG_LEVEL           0 - TRACE, 1 - DEBUG, 2 - INFO, 3 - WARNING, 4 - ERROR, 5 - CRITICAL, 6 - OFF"
        echo -e "  -l,  --list-tests           List the available tests"
        echo -e "  -r,  --release              Run the tests with the release build"
        echo -e "  -s,  --specific-test TEST   Run TEST"
        echo -e "  -x,  --xml                  Output XML file"
        echo -e "  -h,  --help                 Prints this help"
        echo -e "  -shuffle,                   run the tests in a random order"
        echo -e "  -seed,                      Uses reproduce an order-related tests"
        echo -e "  -st,                        Run the tests with single thread"
        echo -e "  --sanitizer                 Run graph optimizer sanitizer build tests"
    fi

    if [ $1 == "run_runtime_unit_test" ]; then
        echo -e "\nusage: $1 [options]\n"
        echo -e "options:\n"
        echo -e "  -g,  --gdbserver            Run the app with gdbserver"
        echo -e "  -i,  --iterations NUM       Run NUM iterations"
        echo -e "  -spdlog LOG_LEVEL           0 - TRACE, 1 - DEBUG, 2 - INFO, 3 - WARNING, 4 - ERROR, 5 - CRITICAL, 6 - OFF"
        echo -e "  -l,  --list-tests           List the available tests"
        echo -e "  -r,  --release              Run the tests with the release build"
        echo -e "  -s,  --specific-test TEST   Run TEST"
        echo -e "  -x,  --xml                  Output XML file"
        echo -e "  -h,  --help                 Prints this help"
        echo -e "  -shuffle,                   run the tests in a random order"
        echo -e "  -seed,                      Uses reproduce an order-related tests"
        echo -e "  -st,                        Run the tests with single thread"
        echo -e "  --sanitizer                 Run tests with sanitizer build"
        echo -e "  -d, --disabled-test         Run tests that are disabled"

    fi

    if [ $1 == "run_mme_test" ]; then
        echo -e "\nusage: $1 [options]\n"
        echo -e "options:\n"
        echo -e "  -g,  --gdbserver                       Run the app with gdbserver"
        echo -e "  -gdb                                   Run the app under gdb"
        echo -e "  -v   --valgrind                        Run the app under valgrind"
        echo -e "       --shuffle                         Randomize tests' orders on every iteration"
        echo -e "       --seed                            Uses reproduce an order-related tests"
        echo -e "  -i,  --iterations NUM                  Run NUM iterations"
        echo -e "  -l,  --list-tests                      List the available tests"
        echo -e "  -r,  --release                         Run the tests with the release build"
        echo -e "  -f,  --run-until-failure               Run test until first failure"
        echo -e "  -s,  --specific-test TEST              Run TEST"
        echo -e "  -a,  --disable-additional-tests TEST   Run all tests without TEST and any default skipped tests"
        echo -e "  -c,  --chip-type CHIP_TYPE             Run only tests related to CHIP_TYPE"
        echo -e "  -o,  --check-roi                       Run roi checker on each test"
        echo -e "  -d   --disabled-test                   Run tests that are disabled"
        echo -e "  -x,  --xml PATH                        Output XML file PATH"
        echo -e "       --no-color                        Disable colors in output"
        echo -e "       --test_on_chip                    Run mme_test to compare FuncSim to chip\\PLDM"
        echo -e "       --skip_sim_test                   When running test_on_chip - skip the run and compare on a simulator"
        echo -e "       --mme_limit NUM                   Limit number of mme cores to NUM"
        echo -e "       --sanity                          Run only sanity tests"
        echo -e "       --regression                      Run only regression tests"
        echo -e "       --networks                        Run tests extracted from networks(gaudi only)"
        echo -e "       --sanitizer                       Run tests with sanitizer build"
        echo -e "  -h,  --help                            Prints this help"
    fi

    if [ $1 == "build_engines_fw" ]; then
        echo -e "\nusage: $1 [options]\n"

        echo -e "options:\n"
        echo -e "  -a,                  --build-all            Build both debug and release build"
        echo -e "  -p,                  --build-coral          Build with coral cyclic build support"
        echo -e "  -noasic,             --build-noasic-fw      Dont build ARC FW binaries for Asic (Synopsys compiler)"
        echo -e "  -nobfm,              --build-nobfm-so       Dont build ARC FW SO libs for Coral"
        echo -e "  -notests,            --build-no-tests       Dont build ARC FW Tests"
        echo -e "  -t,                  --build-fw-tests       Build FW tests (default)"
        echo -e "  -stm TRACE_LEVEL,    --stm-trace            Enable STM tracing 0=disabled, 1=ERROR, 2=WARN, 3=INFO, 4=TRACE, 5=ALL"
        echo -e "  -c,                  --configure            Configure before build"
        echo -e "  -C,                  --chip-type            Chip type: (gaudi2, gaudi3, all) : default = gaudi2"
        echo -e "  -r,                  --release              Build only release build"
        echo -e "                       --recursive            Build all the pre-requisite modules in a recursive way"
        echo -e "  -v,                  --verbose              Build with verbose"
        echo -e "  -s,                  --sanitize             Build with sanitize flags on"
        echo -e "  -h,                  --help                 Prints this help"
    fi

    if [ $1 == "build_habana_tfrt" ]; then
        echo -e "\nusage: $1 [options]\n"

        echo -e "options:\n"
        echo -e "  -j,  --jobs <val>           Overwrite number of jobs"
        echo -e "  -c,  --configure            Configure before build"
        echo -e "  -a,  --build-all            Build both debug and release build"
        echo -e "  -r,  --release              Build only release build"
        echo -e "  -h,  --help                 Prints this help"
    fi

    if [ $1 == "build_scal" ]; then
        echo -e "\nusage: $1 [options]\n"

        echo -e "options:\n"
        echo -e "  -a,  --build-all            Build both debug and release build"
        echo -e "       --no-color             Disable colors in output"
        echo -e "  -c,  --configure            Configure before build"
        echo -e "  -r,  --release              Build only release build"
        echo -e "       --recursive            Build all the pre-requisite modules in a recursive way"
        echo -e "  -s,  --sanitize             Build with address sanitize flags on"
        echo -e "  -ts, --tsanitize            Build with thread sanitize flags on"
        echo -e "  -v,  --verbose              Build with verbose"
        echo -e "  -V   --valgrind             Build with valgrind flag on"
        echo -e "  -j,  --jobs <val>           Overwrite number of jobs"
        echo -e "  -l                          Build only scal (without tests)"
        echo -e "  -h,  --help                 Prints this help"
    fi

    if [ $1 == "build_gaudi_demo" ]; then
        echo -e "\nusage: $1 [options]\n"

        echo -e "options:\n"
        echo -e "  -a,  --build-all            Build both debug and release build"
        echo -e "       --no-color             Disable colors in output"
        echo -e "  -c,  --configure            Configure before build"
        echo -e "  -r,  --release              Build only release build"
        echo -e "  -j,  --jobs <val>           Overwrite number of jobs"
        echo -e "  -h,  --help                 Prints this help"
    fi

    if [ $1 == "run_demos_tests" ]; then
        echo -e "\nusage: $1 [options]\n"
        echo -e "options:\n"
        echo -e "  -l,  --list-tests                   List the available tests"
        echo -e "  -s,  --specific-test TEST           Run TEST or specify filter supported by pytest '-k' parameter"
        echo -e "  -m,  --maxfail NUM                  Stop after NUM failures"
        echo -e "  -p,  --pdb                          Run the app under pdb (python GDB)"
        echo -e "  -seed                               Uses reproduce an order-related tests"
        echo -e "  -prof                               Enable synapse profiler"
        echo -e "  -x,  --xml PATH                     Output XML file to PATH - available in ST mode only"
        echo -e "  -nr                                 Disable random order tests"
        echo -e "       --no-color                     Disable colors in output"
        echo -e "       --no-capture                   Disable all capturing of stdout/stderr output"
        echo -e "  -spdlog LOG_LEVEL                   0 - TRACE, 1 - DEBUG, 2 - INFO, 3 - WARNING, 4 - ERROR, 5 - CRITICAL, 6 - OFF"
        echo -e "  -j                                  Overwrite number of jobs"
        echo -e "  -a,  --marker                       only run tests matching given mark expression. Example: -a 'mark1 and not mark2'"
        echo -e "  -h,  --help                         Prints this help"
    fi

    if [ $1 == "run_gemm_benchmarks_test" ]; then
        if command -v python3.6 >/dev/null 2>&1; then
        __python_cmd="python3.6"
        elif command -v python3 >/dev/null 2>&1; then
            __python_cmd="python3"
        else
            echo "No python3 installation found. Please install and run script again."
            return 1
        fi

        local __gemm_benchmark_root="$SYNAPSE_ROOT/tests/benchmarks_tests/gaudi"
        $__python_cmd $__gemm_benchmark_root/gemm_benchmarks_test.py -h
    fi

    if [ $1 == "run_scal_test" ]; then
        echo -e "\nusage: $1 [options]\n"
        echo -e "options:\n"
        echo -e "  -g,  --gdbserver                       Run the app with gdbserver"
        echo -e "  -gdb                                   Run the app under gdb"
        echo -e "  -v   --valgrind                        Run the app under valgrind"
        echo -e "  -spdlog LOG_LEVEL                      0 - TRACE, 1 - DEBUG, 2 - INFO, 3 - WARNING, 4 - ERROR, 5 - CRITICAL, 6 - OFF"
        echo -e "  -sudo                                  Run as sudo"
        echo -e "  -shuffle                               Randomize tests' orders on every iteration"
        echo -e "  -seed                                  Uses reproduce an order-related tests"
        echo -e "  -prof                                  Enable synapse profiler"
        echo -e "  -i,  --iterations NUM                  Run NUM iterations"
        echo -e "  -l,  --list-tests                      List the available tests"
        echo -e "  -r,  --release                         Run the tests with the release build"
        echo -e "  -f,  --run-until-failure               Run test until first failure"
        echo -e "  -s,  --specific-test TEST              Run TEST"
        echo -e "  -a,  --disable-additional-tests TEST   Run all tests without TEST and any default skipped tests"
        echo -e "  -d   --disabled-test                   Run tests that are disabled"
        echo -e "  -c   --chip-type                       Select chip specific tests <gaudi2/gaudi3> (default: gaudi2)"
        echo -e "  -x,  --xml PATH                        Output XML file PATH"
        echo -e "       --no-color                        Disable colors in output"
        echo -e "       --reset-terminal                  Reset the terminal to clear gtest colors on completion"
        echo -e "  --sanitizer                            Run tests with sanitizer build"
        echo -e "  -h,  --help                            Prints this help"
    fi

    if [ $1 == "run_engines_fw_test" ]; then
        echo -e "\nusage: $1 [options]\n"
        echo -e "options:\n"
        echo -e "  -g,  --gdbserver                       Run the app with gdbserver"
        echo -e "  -C,  --chip-type                       Select chip specific tests <gaudi2/gaudi3> (default: gaudi2)"
        echo -e "  -gdb                                   Run the app under gdb"
        echo -e "  -v   --valgrind                        Run the app under valgrind"
        echo -e "  -spdlog LOG_LEVEL                      0 - TRACE, 1 - DEBUG, 2 - INFO, 3 - WARNING, 4 - ERROR, 5 - CRITICAL, 6 - OFF"
        echo -e "  -sudo                                  Run as sudo"
        echo -e "  -shuffle                               Randomize tests' orders on every iteration"
        echo -e "  -seed                                  Uses reproduce an order-related tests"
        echo -e "  -prof                                  Enable synapse profiler"
        echo -e "  -i,  --iterations NUM                  Run NUM iterations"
        echo -e "  -l,  --list-tests                      List the available tests"
        echo -e "  -r,  --release                         Run the tests with the release build"
        echo -e "  -f,  --run-until-failure               Run test until first failure"
        echo -e "  -b,  --break-on-first-failure          Break on first failure"
        echo -e "  -s,  --specific-test TEST              Run TEST"
        echo -e "  -a,  --disable-additional-tests TEST   Run all tests without TEST and any default skipped tests"
        echo -e "  -d   --disabled-test                   Run tests that are disabled"
        echo -e "  -x,  --xml PATH                        Output XML file PATH"
        echo -e "  -n,  --nic-loopback                    Set nic loopback mask"
        echo -e "       --no-color                        Disable colors in output"
        echo -e "       --reset-terminal                  Reset the terminal to clear gtest colors on completion"
        echo -e "       --shard-index                     Set the GTEST_SHARD_INDEX environment variable to the index of the shard"
        echo -e "       --total-shards                    Set the GTEST_TOTAL_SHARDS environment variable to the total number of shards"
        echo -e "  -h,  --help                            Prints this help"
    fi
    if [ $1 == "run_habana_py_test" ]; then
        echo -e "\nusage: $1 [options]\n"
        echo -e "options:\n"
        echo -e "  -c   --chip                         Select chip specific tests <goya/goya2> (default: goya)"
        echo -e "  -l,  --list-tests                   List the available tests"
        echo -e "  -s,  --specific-test TEST           Run TEST or specify filter supported by pytest '-k' parameter"
        echo -e "  -m,  --maxfail NUM                  Stop after NUM failures"
        echo -e "  -p,  --pdb                          Run the app under pdb (python GDB)"
        echo -e "       --no-color                     Disable colors in output"
        echo -e "  -h,  --help                         Prints this help"
        echo -e "  -x,  --xml PATH                     Output XML file to PATH - available in ST mode only"
        echo -e "  -seed                               Uses reproduce an order-related tests"
        echo -e "  -prof                               Enable synapse profiler"
        echo -e "  -nr                                 Disable random order tests"
        echo -e "  -a,  --marker                       only run tests matching given mark expression. Example: -a 'mark1 and not mark2' default 'not daily and not asic_only'"
        echo -e "  -sm, --skip_measurements            Load dynamic range from a file, in tests marked as @long_measurements"
        echo -e "  -em, --export_measurements          Save measured dynamic range to file (overwrite if exist), in tests marked as @long_measurements"
        echo -e "  -rm, --renew_measurements           Delete all existing dynamic range files and save new measured dynamic range to file, in tests marked as @long_measurements"
        echo -e "  -sx, --skip_execution               Don't execute inference, in tests marked as @long_execution"
        echo -e "  -spdlog LOG_LEVEL                   0 - TRACE, 1 - DEBUG, 2 - INFO, 3 - WARNING, 4 - ERROR, 5 - CRITICAL, 6 - OFF"
    fi
    if [ $1 == "run_synapse_mlir_graph_test" ]; then
        echo -e "\nusage: $1 [options]\n"
        echo -e "options:\n"
        echo -e "  -l,  --list-tests               List the available tests"
        echo -e "  -r,                             Currently has no effect, used for CI needs"
        echo -e "  -s,  --specific-test TEST       Run TEST"
        echo -e "  -a   --specific-test-out TEST   Don't run TEST"
        echo -e "  -h,  --help                     Prints this help"
        echo -e "  -shuffle,                       run the tests in a random order"
        echo -e "  -x,  --xml                      Output XML file"
    fi
    if [ $1 == "run_death_test" ]; then
        echo -e "\nusage: $1\n"
    fi
}

build_synapse ()
{
    SECONDS=0

    _verify_exists_dir "$SYNAPSE_ROOT" $SYNAPSE_ROOT
    : "${SYNAPSE_DEBUG_BUILD:?Need to set SYNAPSE_DEBUG_BUILD to the build folder}"
    : "${SYNAPSE_RELEASE_BUILD:?Need to set SYNAPSE_RELEASE_BUILD to the build folder}"

    local __scriptname=$(__get_func_name)

    local __jobs=$NUMBER_OF_JOBS
    local __color="ON"
    local __debug="yes"
    local __release=""
    local __all=""
    local __configure=""
    local __org_configure=""
    local __other_options=""
    local __build_res=""
    local __sanitize="NO"
    local __tidy=""
    local __tsanitize="NO"
    local __valgrind="NO"
    local __verbose=""
    local __mme_tests="NO"
    local __gen_docs="OFF"
    local __doc_mode="pdf"
    local __synapse_tests="ON"
    local __qa_tests="OFF"
    local __recursive=""
    local __recursive_skip_tests=""

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
        --no-hcl )
            echo -e "WARNING: Option '--no-hcl' is deprecated! Please build hcl lib separately"
            ;;
        -n  | --cuda )
            __other_options="$__other_options -DCUDA_ENABLED=ON"
            ;;
        -o  | --run_from_json )
            __other_options="$__other_options -DTESTS_FROM_JSON_ONLY=ON"
            ;;
        -r  | --release )
            __debug=""
            __release="yes"
            ;;
        --recursive )
            __recursive="yes"
            ;;
        --recursive_st )
            __recursive="yes"
            __recursive_skip_tests="yes"
            ;;
        -j  | --jobs )
            shift
            __jobs=$1
            ;;
        -s  | --sanitize )
            __sanitize="ON"
            ;;
        -ts | --tsanitize )
            __tsanitize="ON"
            ;;

        -v  | --verbose )
            __verbose="VERBOSE=1"
            ;;
        -V  | --valgrind  )
            __valgrind="ON"
            ;;
        --doc )
            # build docs only on Ubunutu
            if  [[ "$OS" =~ "ubuntu" ]] ; then
                __gen_docs="ON"
            fi
            shift
            __doc_mode=$1
            if [ "$__doc_mode" != "pdf" ] ; then
                echo "The option $1 is not allowed with --doc"
                usage $__scriptname
                return 1 # error
            fi
            ;;
        -l  | --synapse_only  )
            __recursive_skip_tests="yes"
            __synapse_tests="OFF"
            ;;
        -qa  | --qa_tests  )
            __qa_tests="ON"
            ;;
        -y  | --tidy )
            __tidy="yes"
            ;;
        -h  | --help )
            usage $__scriptname
            return 0
            ;;
        *)
            echo "The parameter $1 is not allowed"
            usage $__scriptname
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

    CLANG_TIDY_DEFINE="-DCLANG_TIDY=OFF"
    if [ ! -z "$__tidy" ]; then
        CLANG_TIDY_DEFINE="-DCLANG_TIDY=ON"
    fi

    if [ "$__gen_docs" = "ON" ]; then
        if ( ! test -f $SYNAPSE_ROOT/doc/.doc_venv/bin/activate ); then
            $__python_cmd -m venv $SYNAPSE_ROOT/doc/.doc_venv
            conf_file="$SYNAPSE_ROOT/doc/.doc_venv/pip.conf"
            echo "[global]" > ${conf_file}
            echo "index =  https://artifactory-kfs.habana-labs.com/artifactory/api/pypi/pypi-virtual" >> ${conf_file}
            echo "index-url = https://artifactory-kfs.habana-labs.com/artifactory/api/pypi/pypi-virtual/simple" >> ${conf_file}
            echo "trusted-host = artifactory-kfs.habana-labs.com" >> ${conf_file}
        fi
        $SYNAPSE_ROOT/doc/.doc_venv/bin/$__pip_cmd install -r $SYNAPSE_ROOT/doc/common/requirements.txt
    fi

    if [ -n "$__recursive" ]; then
        local __release_par=""
        local __configure_par=""
        local __skip_test_par=""
        local __jobs_par=""

        if [ -n "$__configure" ]; then
            __configure_par="-c"
        fi

        if [ -n "$__release" ]; then
            __release_par="-r"
        fi

        if [ -n "$__all" ]; then
            __release_par="-a"
        fi

        if [ -n "$__recursive_skip_tests" ]; then
            __skip_test_par="-l"
        fi

        __jobs_par="-j $__jobs"

        echo "Building pre-requisite packages for Synapse"

        __common_build_dependency -m "synapse" $__skip_test_par $__configure_par $__release_par $__jobs_par
        if [ $? -ne 0 ]; then
            echo "Failed to build dependency packages"
            return 1
        fi
    fi

    if [ -n "$__debug" ]; then
        echo -e "Building in debug mode"
        local __folder=$SYNAPSE_DEBUG_BUILD
        local __mme_folder=$MME_DEBUG_BUILD
        if [[ $__sanitize = "ON" || $__tsanitize = "ON" ]]; then
            __folder=$SYNAPSE_DEBUG_SANITIZER_BUILD
            __mme_folder=$MME_DEBUG_SANITIZER_BUILD

            if [ $__tsanitize = "ON" ]; then
                if [ -f $__folder/.asan_tag ]; then
                    __configure="yes"
                fi
            else
                if [ -f $__folder/.tsan_tag ]; then
                    __configure="yes"
                fi
            fi
        fi

        if [ ! -d $__folder ]; then
            __configure="yes"
        fi

        if [ -n "$__configure" ]; then
            if [ -d $__folder ]; then
                rm -rf $__folder
                rm -rf $__mme_folder
            fi
            mkdir -p $__folder
        fi

        if [ $__sanitize = "ON" ]; then
            touch $__folder/.asan_tag $__mme_folder/.asan_tag
        fi
        if [ $__tsanitize = "ON" ]; then
            touch $__folder/.tsan_tag $__mme_folder/.tsan_tag
        fi

        _verify_exists_dir "$__folder" $__folder
        pushd $__folder
            (set -x; cmake \
            -DCMAKE_BUILD_TYPE="Debug" \
            -DCMAKE_INSTALL_PREFIX=$HOME/.local \
            -DCMAKE_COLOR_MAKEFILE=$__color \
            -DGEN_DOCS=$__gen_docs \
            -DDOC_MODE=$__doc_mode \
            -DSANITIZE_ON=$__sanitize \
            -DTSANITIZE_ON=$__tsanitize \
            -DVALGRIND_ON=$__valgrind \
            -DTESTS_ENABLED=$__synapse_tests \
            -DQA_TESTS_ENABLED=$__qa_tests \
            $CLANG_TIDY_DEFINE \
            $__other_options \
            $SYNAPSE_ROOT)
        make $__verbose -j$__jobs
        __build_res=$?
        popd
        if [ $__build_res -ne 0 ]; then
            return $__build_res
        fi
        _copy_build_products $__folder

        # create link of GraphCompilerPlugin in plugins folder
        mkdir -p ${HABANA_PLUGINS_LIB_PATH}
        cp -fs ${__folder}/lib/libGraphCompilerPlugin.so ${HABANA_PLUGINS_LIB_PATH}/

    fi

    __configure=$__org_configure

    if [ -n "$__release" ]; then
        echo "Building in release mode"
        if [ ! -d $SYNAPSE_RELEASE_BUILD ]; then
            __configure="yes"
        fi

        if [[ $__sanitize = "ON" || $__tsanitize = "ON" ]]; then

            if [ $__tsanitize = "ON" ]; then
                if [ -f $SYNAPSE_RELEASE_BUILD/.asan_tag ]; then
                    __configure="yes"
                fi
            else
                if [ -f $SYNAPSE_RELEASE_BUILD/.tsan_tag ]; then
                    __configure="yes"
                fi
            fi
        else
            if [[ -f $SYNAPSE_RELEASE_BUILD/.tsan_tag || -f $SYNAPSE_RELEASE_BUILD/.asan_tag ]]; then
                __configure="yes"
            fi
        fi

        if [ -n "$__configure" ]; then
            if [ -d $SYNAPSE_RELEASE_BUILD ]; then
                rm -rf $SYNAPSE_RELEASE_BUILD
                rm -rf $MME_RELEASE_BUILD
            fi
            mkdir -p $SYNAPSE_RELEASE_BUILD
        fi

        if [ $__sanitize = "ON" ]; then
            touch $SYNAPSE_RELEASE_BUILD/.asan_tag $MME_RELEASE_BUILD/.asan_tag
        fi
        if [ $__tsanitize = "ON" ]; then
            touch $SYNAPSE_RELEASE_BUILD/.tsan_tag $MME_RELEASE_BUILD/.tsan_tag
        fi

        _verify_exists_dir "$SYNAPSE_RELEASE_BUILD" $SYNAPSE_RELEASE_BUILD
        pushd $SYNAPSE_RELEASE_BUILD
        (set -x; cmake \
            -DCMAKE_BUILD_TYPE="Release" \
            -DCMAKE_INSTALL_PREFIX=$HOME/.local \
            -DCMAKE_COLOR_MAKEFILE=$__color \
            -DGEN_DOCS=$__gen_docs \
            -DDOC_MODE=$__doc_mode \
            -DSANITIZE_ON=$__sanitize \
            -DTSANITIZE_ON=$__tsanitize \
            -DVALGRIND_ON=$__valgrind \
            -DTESTS_ENABLED=$__synapse_tests \
            -DQA_TESTS_ENABLED=$__qa_tests \
            $CLANG_TIDY_DEFINE \
            $__other_options \
            $SYNAPSE_ROOT)
        make $__verbose -j$__jobs
        __build_res=$?
        popd
        if [ $__build_res -ne 0 ]; then
            return $__build_res
        fi
        _copy_build_products $SYNAPSE_RELEASE_BUILD -r

        # create link of GraphCompilerPlugin in plugins folder
        mkdir -p ${HABANA_PLUGINS_LIB_PATH}
        cp -fs ${SYNAPSE_RELEASE_BUILD}/lib/libGraphCompilerPlugin.so ${HABANA_PLUGINS_LIB_PATH}/

    fi

    printf "\nElapsed time: %02u:%02u:%02u \n\n" $(($SECONDS / 3600)) $((($SECONDS / 60) % 60)) $(($SECONDS % 60))
    return 0
}

build_mme ()
{
    SECONDS=0

    _verify_exists_dir "$MME_ROOT" $MME_ROOT
    : "${MME_DEBUG_BUILD:?Need to set MME_DEBUG_BUILD to the build folder}"
    : "${MME_RELEASE_BUILD:?Need to set MME_RELEASE_BUILD to the build folder}"

    local __scriptname=$(__get_func_name)

    local __jobs=$NUMBER_OF_JOBS
    local __color="ON"
    local __debug="ON"
    local __release=""
    local __all=""
    local __configure=""
    local __org_configure=""
    local __build_res=""
    local __sanitize="OFF"
    local __tsanitize="OFF"
    local __valgrind="OFF"
    local __verbose=""
    local __sim_dependent="ON"
    local __mme_verification="OFF"
    local __synapse_dependent="ON"
    local __swtools_dependent="ON"
    local __mme_tests="ON"
    local __chip_type="all"
    local __chip_type_flags=""
    local __compression_stack=""
    local __use_local_compression_stack="OFF"

    # parameter while-loop
    while [ -n "$1" ];
    do
        case $1 in
        -a  | --build-all )
            __all="ON"
            ;;
        --no-color )
            __color="OFF"
            ;;
        -c  | --configure )
            __org_configure="ON"
            __configure="ON"
            ;;
        -m  | --mme_verification )
            __mme_verification="ON"
            ;;
        -d  | --no_depend )
            __sim_dependent="OFF"
            __synapse_dependent="OFF"
            __use_local_compression_stack="ON"
            __swtools_dependent="OFF"
            ;;
        -k  | --no_synapse_depend )
            __synapse_dependent="OFF"
            __use_local_compression_stack="ON"
            ;;
        -r  | --release )
            __debug="OFF"
            __release="ON"
            ;;
        -j  | --jobs )
            shift
            __jobs=$1
            ;;
        -s  | --sanitize )
            __sanitize="ON"
            ;;
        -ts | --tsanitize )
            __tsanitize="ON"
            ;;
        -v  | --verbose )
            __verbose="VERBOSE=1"
            ;;
        -V  | --valgrind  )
            __valgrind="ON"
            ;;
        -l  )
            __mme_tests="OFF"
            ;;
        -C | --chip-type )
            shift
            __chip_type=$1
            ;;
        --compression_stack )
            shift
            __compression_stack=$1
            ;;
        -h  | --help )
            usage $__scriptname
            return 0
            ;;
        *)
            echo "The parameter $1 is not allowed"
            usage $__scriptname
            return 1 # error
            ;;
        esac
        shift
    done

    if [ -n "$__configure" ]; then
        __check_mandatory_pkgs
    fi


    if [ -n "$__all" ]; then
        __debug="ON"
        __release="ON"
    fi

    case "`echo "$__chip_type" | tr '[:upper:]' '[:lower:]'`" in
        h3 | gaudi )
            __chip_type_flags="-DGAUDI_EN=ON -DGOYA2_EN=OFF -DGAUDI2_EN=OFF -DGAUDI3_EN=OFF"
            ;;
        h5 | goya2 )
            __chip_type_flags="-DGAUDI_EN=OFF -DGOYA2_EN=ON -DGAUDI2_EN=OFF -DGAUDI3_EN=OFF"
            ;;
        h6 | gaudi2 )
            __chip_type_flags="-DGAUDI_EN=OFF -DGOYA2_EN=OFF -DGAUDI2_EN=ON -DGAUDI3_EN=OFF"
            ;;
        h9 | gaudi3 )
            __chip_type_flags="-DGAUDI_EN=OFF -DGOYA2_EN=OFF -DGAUDI2_EN=OFF -DGAUDI3_EN=ON"
            ;;
        all )
            ;;
        * )
            echo " Cannot determine chip type - '$__chip_type'"
            usage $__scriptname
            return 1 # error
    esac

    if [ -n "$__compression_stack" ]; then
        case "`echo "$__compression_stack" | tr '[:upper:]' '[:lower:]'`" in
            local | mme )
                __use_local_compression_stack="ON"
                ;;
            synapse )
                if [ "$__synapse_dependent" == "OFF" ]; then
                    echo "Cannot use synapse compression stack without synapse dependecy"
                    echo "please remove --no_depend/--no_synapse_depend flags"
                    usage $__scriptname
                    return 1 #error
                else
                    __use_local_compression_stack="OFF"
                fi
                ;;
            * )
                echo " Cannot determine compression stack - `$__compression_stack`"
                usage $__scriptname
                return 1 # error
        esac
    fi

    if [ -n "$__debug" ] &&  [ "$__debug" == "ON" ]; then
        echo -e "Building in debug mode"

        local __folder=$MME_DEBUG_BUILD
        if [ $__sanitize = "ON" ]; then
            __folder=$MME_DEBUG_SANITIZER_BUILD
        fi
        if [ $__tsanitize = "ON" ]; then
            __folder=$MME_DEBUG_SANITIZER_BUILD
        fi
        if [ ! -d $__folder ]; then
            __configure="ON"
        fi

        if [ -n "$__configure" ]; then
            if [ -d $__folder ]; then
                rm -rf $__folder
            fi
            mkdir -p $__folder
        fi

        _verify_exists_dir "$__folder" $__folder
        pushd $__folder
            (set -x; cmake \
            -DCMAKE_BUILD_TYPE="Debug" \
            -DCMAKE_INSTALL_PREFIX=$HOME/.local \
            -DCMAKE_COLOR_MAKEFILE=$__color \
            -DSANITIZE_ON=$__sanitize \
            -DTSANITIZE_ON=$__tsanitize \
            -DVALGRIND_ON=$__valgrind \
            -DMME_VER=$__mme_verification \
            -DSYN_DEPEND=$__synapse_dependent \
            -DSIM_DEPEND=$__sim_dependent \
            -DSWTOOLS_DEP=$__swtools_dependent \
            -DTESTS_ENABLED=$__mme_tests \
            -DUSE_LOCAL_COMP_STACK=$__use_local_compression_stack \
            $__chip_type_flags \
            $MME_ROOT)
        make $__verbose -j$__jobs
        __build_res=$?
        popd
        if [ $__build_res -ne 0 ]; then
            return $__build_res
        fi
        _copy_build_products $__folder
    fi

    __configure=$__org_configure

    if [ -n "$__release" ]; then
        echo "Building in release mode"
        if [ ! -d $MME_RELEASE_BUILD ]; then
            __configure="ON"
        fi

        if [ -n "$__configure" ]; then
            if [ -d $MME_RELEASE_BUILD ]; then
                rm -rf $MME_RELEASE_BUILD
            fi
            mkdir -p $MME_RELEASE_BUILD
        fi

        _verify_exists_dir "$MME_RELEASE_BUILD" $MME_RELEASE_BUILD
        pushd $MME_RELEASE_BUILD
        (set -x; cmake \
            -DCMAKE_BUILD_TYPE="Release" \
            -DCMAKE_INSTALL_PREFIX=$HOME/.local \
            -DCMAKE_COLOR_MAKEFILE=$__color \
            -DSANITIZE_ON=$__sanitize \
            -DTSANITIZE_ON=$__tsanitize \
            -DVALGRIND_ON=$__valgrind \
            -DMME_VER=$__mme_verification \
            -DSIM_DEPEND=$__sim_dependent \
            -DSYN_DEPEND=$__synapse_dependent \
            -DSWTOOLS_DEP=$__swtools_dependent \
            -DTESTS_ENABLED=$__mme_tests \
            -DUSE_LOCAL_COMP_STACK=$__use_local_compression_stack \
            $__chip_type_flags \
            $MME_ROOT)
        make $__verbose -j$__jobs
        __build_res=$?
        popd
        if [ $__build_res -ne 0 ]; then
            return $__build_res
        fi
        _copy_build_products $MME_RELEASE_BUILD -r
    fi

    printf "\nElapsed time: %02u:%02u:%02u \n\n" $(($SECONDS / 3600)) $((($SECONDS / 60) % 60)) $(($SECONDS % 60))
    return 0
}

build_weight_lib ()
{
    SECONDS=0

    _verify_exists_dir "$WEIGHT_LIB_ROOT" $WEIGHT_LIB_ROOT
    : "${WEIGHT_LIB_DEBUG_BUILD:?Need to set WEIGHT_LIB_DEBUG_BUILD to the build folder}"
    : "${WEIGHT_LIB_RELEASE_BUILD:?Need to set WEIGHT_LIB_RELEASE_BUILD to the build folder}"

    local __scriptname=$(__get_func_name)

    local __jobs=$NUMBER_OF_JOBS
    local __color="ON"
    local __debug="yes"
    local __release=""
    local __sanitize="NO"
    local __all=""
    local __configure=""
    local __org_configure=""
    local __build_res=""
    local __verbose=""

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
        -s  | --sanitize )
            __sanitize="ON"
            ;;
        -v  | --verbose )
            __verbose="VERBOSE=1"
            ;;
        -j  | --jobs )
            shift
            __jobs=$1
            ;;
        -h  | --help )
            usage $__scriptname
            return 0
            ;;
        *)
            echo "The parameter $1 is not allowed"
            usage $__scriptname
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
        if [ ! -d $WEIGHT_LIB_DEBUG_BUILD ]; then
            __configure="yes"
        fi

        if [ -n "$__configure" ]; then
            if [ -d $WEIGHT_LIB_DEBUG_BUILD ]; then
                rm -rf $WEIGHT_LIB_DEBUG_BUILD
            fi

            mkdir -p $WEIGHT_LIB_DEBUG_BUILD
            pushd $WEIGHT_LIB_DEBUG_BUILD
            (set -x; cmake \
                -DCMAKE_BUILD_TYPE="Debug" \
                -DCMAKE_INSTALL_PREFIX=$HOME/.local \
                -DCMAKE_COLOR_MAKEFILE=$__color \
                -DSANITIZE_ON=$__sanitize \
                $WEIGHT_LIB_ROOT)
            popd
        fi

        _verify_exists_dir "$WEIGHT_LIB_DEBUG_BUILD" $WEIGHT_LIB_DEBUG_BUILD
        pushd $WEIGHT_LIB_DEBUG_BUILD
        make $__verbose -j$__jobs
        __build_res=$?
        popd
        if [ $__build_res -ne 0 ]; then
            return $__build_res
        fi
        #_copy_build_products $WEIGHT_LIB_DEBUG_BUILD
    fi

    __configure=$__org_configure

    if [ -n "$__release" ]; then
        echo "Building in release mode"
        if [ ! -d $WEIGHT_LIB_RELEASE_BUILD ]; then
            __configure="yes"
        fi

        if [ -n "$__configure" ]; then
            if [ -d $WEIGHT_LIB_RELEASE_BUILD ]; then
                rm -rf $WEIGHT_LIB_RELEASE_BUILD
            fi
            mkdir -p $WEIGHT_LIB_RELEASE_BUILD
            pushd $WEIGHT_LIB_RELEASE_BUILD
            (set -x; cmake \
                -DCMAKE_BUILD_TYPE="Release" \
                -DCMAKE_INSTALL_PREFIX=$HOME/.local \
                -DCMAKE_COLOR_MAKEFILE=$__color \
                -DSANITIZE_ON=$__sanitize \
                $WEIGHT_LIB_ROOT)
            popd
        fi

        _verify_exists_dir "$SYNAPSE_RELEASE_BUILD" $WEIGHT_LIB_RELEASE_BUILD
        pushd $WEIGHT_LIB_RELEASE_BUILD
        make $__verbose -j$__jobs
        __build_res=$?
        popd
        if [ $__build_res -ne 0 ]; then
            return $__build_res
        fi
        #_copy_build_products $WEIGHT_LIB_RELEASE_BUILD -r
    fi

    printf "\nElapsed time: %02u:%02u:%02u \n\n" $(($SECONDS / 3600)) $((($SECONDS / 60) % 60)) $(($SECONDS % 60))
    return 0
}

build_rotator ()
{
    SECONDS=0

    _verify_exists_dir "$ROTATOR_ROOT" $ROTATOR_ROOT
    : "${ROTATOR_DEBUG_BUILD:?Need to set ROTATOR_DEBUG_BUILD to the build folder}"
    : "${ROTATOR_RELEASE_BUILD:?Need to set ROTATOR_RELEASE_BUILD to the build folder}"

    local __scriptname=$(__get_func_name)

    local __jobs=$NUMBER_OF_JOBS
    local __color="ON"
    local __debug="ON"
    local __release=""
    local __all=""
    local __configure=""
    local __org_configure=""
    local __build_res=""
    local __sanitize="OFF"
    local __valgrind="OFF"
    local __verbose=""
    local __target="SYNAPSE"
    local __target_define_cmd=""

    # parameter while-loop
    while [ -n "$1" ];
    do
        case $1 in
        -a  | --build-all )
            __all="ON"
            ;;
        --no-color )
            __color="OFF"
            ;;
        -c  | --configure )
            __org_configure="ON"
            __configure="ON"
            ;;
        -t  | --target )
            shift
            __target="$1"
            ;;
        -r  | --release )
            __debug="OFF"
            __release="ON"
            ;;
        -j  | --jobs )
            shift
            __jobs=$1
            ;;
        -s  | --sanitize )
            __sanitize="ON"
            ;;
        -v  | --verbose )
            __verbose="VERBOSE=1"
            ;;
        -V  | --valgrind  )
            __valgrind="ON"
            ;;
        -h  | --help )
            usage $__scriptname
            return 0
            ;;
        *)
            echo "The parameter $1 is not allowed"
            usage $__scriptname
            return 1 # error
            ;;
        esac
        shift
    done

    if [ -n "$__configure" ]; then
        __check_mandatory_pkgs
    fi

    if [ -n "$__all" ]; then
        __debug="ON"
        __release="ON"
    fi

    case "`echo "$__target" | tr '[:upper:]' '[:lower:]'`" in
        gc | synapse )
        __target_define_cmd="-DRUN_WITH_SYNAPSE=1"
        ;;
        standalone | sim | simulator )
        __target_define_cmd="-DRUN_STANDALONE=1"
        ;;
        sv )
        __target_define_cmd="-DRUN_WITH_SV=1"
        ;;
        func-sim | coral )
        __target_define_cmd="-DRUN_WITH_CORAL=1"
        ;;
        * )
            echo "Unsupported target - '$__target'"
            usage $__scriptname
            return 1
        ;;
    esac

    if [ -n "$__debug" ] &&  [ "$__debug" == "ON" ]; then
        echo -e "Building in debug mode"

        local __folder=$ROTATOR_DEBUG_BUILD
        if [ $__sanitize = "ON" ]; then
            __folder=$ROTATOR_DEBUG_SANITIZER_BUILD
        fi
        if [ ! -d $__folder ]; then
            __configure="ON"
        fi

        if [ -n "$__configure" ]; then
            if [ -d $__folder ]; then
                rm -rf $__folder
            fi
            mkdir -p $__folder
        fi

        _verify_exists_dir "$__folder" $__folder
        pushd $__folder
            (set -x; cmake \
            -DCMAKE_BUILD_TYPE="Debug" \
            -DCMAKE_INSTALL_PREFIX=$HOME/.local \
            -DCMAKE_COLOR_MAKEFILE=$__color \
            -DSANITIZE_ON=$__sanitize \
            -DVALGRIND_ON=$__valgrind \
            $__target_define_cmd \
            $ROTATOR_ROOT)
        make $__verbose -j$__jobs
        __build_res=$?
        popd
        if [ $__build_res -ne 0 ]; then
            return $__build_res
        fi
        _copy_build_products $__folder
    fi

    __configure=$__org_configure

    if [ -n "$__release" ]; then
        echo "Building in release mode"
        if [ ! -d $ROTATOR_RELEASE_BUILD ]; then
            __configure="ON"
        fi

        if [ -n "$__configure" ]; then
            if [ -d $ROTATOR_RELEASE_BUILD ]; then
                rm -rf $ROTATOR_RELEASE_BUILD
            fi
            mkdir -p $ROTATOR_RELEASE_BUILD
        fi

        _verify_exists_dir "$ROTATOR_RELEASE_BUILD" $ROTATOR_RELEASE_BUILD
        pushd $ROTATOR_RELEASE_BUILD
        (set -x; cmake \
            -DCMAKE_BUILD_TYPE="Release" \
            -DCMAKE_INSTALL_PREFIX=$HOME/.local \
            -DCMAKE_COLOR_MAKEFILE=$__color \
            -DSANITIZE_ON=$__sanitize \
            -DVALGRIND_ON=$__valgrind \
            $__target_define_cmd \
            $ROTATOR_ROOT)
        make $__verbose -j$__jobs
        __build_res=$?
        popd
        if [ $__build_res -ne 0 ]; then
            return $__build_res
        fi
        _copy_build_products $ROTATOR_RELEASE_BUILD -r
    fi

    printf "\nElapsed time: %02u:%02u:%02u \n\n" $(($SECONDS / 3600)) $((($SECONDS / 60) % 60)) $(($SECONDS % 60))
    return 0
}

run_synapse_gc_test()
{
    if [ -z "$SYNAPSE_ROOT" ]
    then
        echo "SYNAPSE_ROOT path is not defined"
        return 1
    fi

    local __gc_tests_py_exe="$__python_cmd $SYNAPSE_ROOT/scripts/syn_tests.py"
    local __gc_tests_py_args="$*"
    ${__gc_tests_py_exe} $__gc_tests_py_args

    return $?
}

run_synapse_test()
{
    if [ -z "$SYNAPSE_ROOT" ]
    then
        echo "SYNAPSE_ROOT path is not defined"
        return 1
    fi

    local __runtime_tests_py_exe="$__python_cmd $SYNAPSE_ROOT/scripts/runtime_tests.py"
    local __runtime_tests_py_args="$*"
    ${__runtime_tests_py_exe} $__runtime_tests_py_args

    return $?
}

run_death_test()
{
    local __death_test_exe=$SYNAPSE_ROOT/.ci/scripts/device_failure_tester.py
    python3 $__death_test_exe
    return $?
}

run_graph_optimizer_test()
{
    local __graph_optimizer_test_exe=$SYNAPSE_DEBUG_BUILD/bin/GraphCompiler_tests

    local __scriptname=$(__get_func_name)
    local __print_tests=""
    local __num_iterations=1;
    local __filter="*"        # default negative filter - tests not to run
    local __mlir_filter="*mlir*:*MLIR*:*Mlir*" # filter to run only mlir related tests
    local __run_mlir_tests="0"
    local __gdb=""
    local __gdbserver=""
    local __xml=""
    local __spdlog="4"
    local __shuffle=""
    local __seed=0
    local __single_thread=""
    local __color="yes"

    # parameter while-loop
    while [ -n "$1" ];
    do
        case $1 in
        -g  | --gdbserver )
            __gdbserver="yes"
            ;;
        -r  | --release )
            __graph_optimizer_test_exe=$SYNAPSE_RELEASE_BUILD/bin/GraphCompiler_tests
            ;;
       --sanitizer )
            __graph_optimizer_test_exe=$SYNAPSE_DEBUG_SANITIZER_BUILD/bin/GraphCompiler_tests
            ;;
        -gdb )
            __gdb="yes"
        ;;
        -i  | --iterations )
            shift
            __num_iterations=$1
            ;;
        -spdlog )
            shift
            __spdlog=$1
            ;;
        --mlir ) # run only tests related to mlir
            __filter=$__mlir_filter
            __run_mlir_tests="1"
            ;;
        -l  | --list-tests )
            __print_tests="yes"
            ;;
        -s  | --specific-test )
            shift
            __filter=$1
            ;;
        -x  | --xml )
            __xml="--gtest_output=xml"
            ;;
        -h  | --help )
            usage $__scriptname
            return 0
            ;;
        -shuffle )
            __shuffle="--gtest_shuffle"
            __single_thread="yes"
            ;;
        -seed | --gtest_random_seed )
            shift
            __seed=$1
            __shuffle="--gtest_shuffle"
            __single_thread="yes"
            ;;
        -st  | --single_thread )
            __single_thread="yes"
            ;;
        --no-color )
            __color="no"
            ;;
        *)
            echo "The parameter $1 is not allowed"
            usage $__scriptname
            return 1 # error
            ;;
        esac
        shift
    done

    _verify_exists_file "$__graph_optimizer_test_exe" $__graph_optimizer_test_exe

    if [ -n "$__print_tests" ]; then
        ${__graph_optimizer_test_exe} --gtest_list_tests --gtest_color=$__color --gtest_filter=$__filter
        return $?
    fi

    if [ -n "$__gdbserver" ]; then
        __num_iterations=1
        (set -x; RUN_GC_MLIR_TESTS=${__run_mlir_tests} LOG_LEVEL_ALL=${__spdlog} gdbserver localhost:2345 ${__graph_optimizer_test_exe} --gtest_color=$__color --gtest_repeat=$__num_iterations --gtest_filter=$__filter $__xml ${__shuffle} --gtest_random_seed=$__seed)
    elif [ -n "$__gdb" ]; then
        __num_iterations=1
        (set -x; RUN_GC_MLIR_TESTS=${__run_mlir_tests} LOG_LEVEL_ALL=${__spdlog} gdb --args ${__graph_optimizer_test_exe} --gtest_repeat=$__num_iterations --gtest_color=$__color --gtest_filter=$__filter $__xml ${__shuffle} --gtest_random_seed=$__seed)
    else
        (set -x; RUN_GC_MLIR_TESTS=${__run_mlir_tests} LOG_LEVEL_ALL=${__spdlog} ${__graph_optimizer_test_exe} --gtest_repeat=$__num_iterations --gtest_color=$__color --gtest_filter=$__filter $__xml ${__shuffle} --gtest_random_seed=$__seed)
    fi

   # return error code of the test
    return $?
}

run_runtime_unit_test()
{
    local __runtime_unit_test_exe=$SYNAPSE_DEBUG_BUILD/bin/runtime_unit_tests

    local __scriptname=$(__get_func_name)
    local __print_tests=""
    local __num_iterations=1;
    local __filter="*"
    local __gdb=""
    local __gdbserver=""
    local __xml=""
    local __spdlog="4"
    local __shuffle=""
    local __seed=0
    local __single_thread=""
    local __color="yes"
    local __en_disable=""

    # parameter while-loop
    while [ -n "$1" ];
    do
        case $1 in
        -g  | --gdbserver )
            __gdbserver="yes"
            ;;
        -r  | --release )
            __runtime_unit_test_exe=$SYNAPSE_RELEASE_BUILD/bin/runtime_unit_tests
            ;;
       --sanitizer )
            __runtime_unit_test_exe=$SYNAPSE_DEBUG_SANITIZER_BUILD/bin/runtime_unit_tests
            ;;
        -gdb )
            __gdb="yes"
        ;;
        -i  | --iterations )
            shift
            __num_iterations=$1
            ;;
        -spdlog )
            shift
            __spdlog=$1
            ;;
        -l  | --list-tests )
            __print_tests="yes"
            ;;
        -s  | --specific-test )
            shift
            __filter=$1
            ;;
        -x  | --xml )
            __xml="--gtest_output=xml"
            ;;
        -h  | --help )
            usage $__scriptname
            return 0
            ;;
        -shuffle )
            __shuffle="--gtest_shuffle"
            __single_thread="yes"
            ;;
        -seed | --gtest_random_seed )
            shift
            __seed=$1
            __shuffle="--gtest_shuffle"
            __single_thread="yes"
            ;;
        -st  | --single_thread )
            __single_thread="yes"
            ;;
        --no-color )
            __color="no"
            ;;
        -d  | --disabled-test )
            __en_disable="--gtest_also_run_disabled_tests"
            ;;
        *)
            echo "The parameter $1 is not allowed"
            usage $__scriptname
            return 1 # error
            ;;
        esac
        shift
    done

    _verify_exists_file "$__runtime_unit_test_exe" $__runtime_unit_test_exe

    if [ -n "$__print_tests" ]; then
        ${__runtime_unit_test_exe} --gtest_list_tests --gtest_color=$__color --gtest_filter=$__filter
        return $?
    fi

    if [ -n "$__gdbserver" ]; then
        __num_iterations=1
        (set -x; LOG_LEVEL_ALL=${__spdlog} gdbserver localhost:2345 ${__runtime_unit_test_exe} --gtest_color=$__color --gtest_repeat=$__num_iterations $__en_disable --gtest_filter=$__filter $__xml ${__shuffle} --gtest_random_seed=$__seed)
    elif [ -n "$__gdb" ]; then
        __num_iterations=1
        (set -x; LOG_LEVEL_ALL=${__spdlog} gdb --args ${__runtime_unit_test_exe} --gtest_repeat=$__num_iterations $__en_disable --gtest_color=$__color --gtest_filter=$__filter $__xml ${__shuffle} --gtest_random_seed=$__seed)
    else
        (set -x; LOG_LEVEL_ALL=${__spdlog} ${__runtime_unit_test_exe} --gtest_repeat=$__num_iterations $__en_disable --gtest_color=$__color --gtest_filter=$__filter $__xml ${__shuffle} --gtest_random_seed=$__seed)
    fi

   # return error code of the test
    return $?
}

run_mme_test()
{
    local __mme_test_exe_base=${MME_DEBUG_BUILD}/bin/mme_test_runner
    local __ld_lib=$BUILD_ROOT_DEBUG

    local __scriptname=$(__get_func_name)
    local __print_tests=""
    local __num_iterations=1
    local __gdb=""
    local __valgrind=""
    local __filter="*"
    local __negative_filter=""
    local __gdbserver=""
    local __xml=""
    local __en_disable=""
    local __seed=0
    local __color="yes"
    local __run_until_failure=""
    local __shuffle=""
    local __debug_tool=""
    local __chip_type=""
    local __test_flags=""
    local __test_suit=""
    local __mme_limit=0 #default is no limit (=0)
    # parameter while-loop
    while [ -n "$1" ];
    do
        case $1 in
        -g  | --gdbserver )
            __gdbserver="yes"
            __debug_tool="gdbserver localhost:2345"
            ;;
        -gdb )
            __gdb="yes"
            __debug_tool="gdb --args"
            ;;
        -v  | --valgrind )
            __valgrind="yes"
            __debug_tool="valgrind"
            ;;
        -shuffle | --shuffle )
            __shuffle="--gtest_shuffle"
            ;;
        -seed | --seed | --gtest_random_seed )
            shift
            __seed=$1
            __test_flags="${__test_flags} --seed ${__seed}"
            ;;
        -f  | --run-until-failure )
            __run_until_failure="yes"
            ;;
        --sanitizer )
            __mme_test_exe_base=$MME_DEBUG_SANITIZER_BUILD/bin/mme_test_runner_
            ;;
        -r  | --release )
            __mme_test_exe_base=$MME_RELEASE_BUILD/bin/mme_test_runner
            __ld_lib=$BUILD_ROOT_RELEASE
            ;;
        -i  | --iterations )
            shift
            __num_iterations=$1
            ;;
        -l  | --list-tests )
            __print_tests="yes"
            ;;
        -s  | --specific-test )
            shift
            __filter=$1
            __negative_filter=""
            ;;
        -a  | --disable-additional-tests )
            shift
            __negative_filter=$__negative_filter":"$1
            ;;
        -c  | --chip-type )
            shift
            case "`echo "$1" | tr '[:upper:]' '[:lower:]'`" in
            gaudi | h3 )
                __filter="*MmeUT*:*Gaudi*:*GAUDI*"
                __negative_filter="*Common*:*Gaudi2*:*GAUDI2*:*Gaudi3*:*GAUDI3*"
                __chip_type="gaudi"
            ;;
            goya2 | h5 )
                __filter="*MmeUT*:*Goya2*:*GOYA2*"
                __negative_filter=""
                __chip_type="goya2"
            ;;
            gaudi2 | h6 )
                __filter="*MmeUT*:*Common*:*Gaudi2*:*GAUDI2*"
                __negative_filter=""
                __chip_type="gaudi2"
            ;;
            gaudi3 | h9 )
                __filter="*MmeUT*:*Common*:*Gaudi3*:*GAUDI3*"
                __negative_filter=""
                __chip_type="gaudi3"
            ;;
            esac
            ;;
        -o  | --check-roi )
            __test_flags="${__test_flags} --check_roi"
            ;;
        -d  | --disabled-test )
            __en_disable="--gtest_also_run_disabled_tests"
            ;;
        -x  | --xml )
            shift
            __xml="--gtest_output=xml:"$1
            ;;
        --no-color )
            __color="no"
            __test_flags="${__test_flags} --no-color"
            ;;
        --test_on_chip )
            __test_flags="${__test_flags} --chip_test"
            ;;
        --skip_sim_test )
            __test_flags="${__test_flags} --drop_sim_test"
            ;;
        --mme_limit )
            shift
            __test_flags="${__test_flags} --mme_limit $1"
            ;;
        --sanity )
            __test_suit="sanity"
            ;;
        --regression )
            __test_suit="regression"
            ;;
        --unit_tests )
            __test_suit="unit_tests"
            ;;
        --networks )
            __test_suit="networks"
            ;;
        -h  | --help )
            usage $__scriptname
            return 0
            ;;
        *)
            echo "The parameter $1 is not allowed"
            usage $__scriptname
            return 1 # error
            ;;
        esac
        shift
    done

    if [ -z ${__chip_type} ]; then
        __chip_type="gaudi goya2 gaudi2 gaudi3"
    fi

    for _type in ${__chip_type}
    do
        if [ "$_type" == "gaudi2" ]; then
            # filter test suit -
            case $__test_suit in
                sanity )
                    __filter="*MmeUT*:*MMEGaudi2Sanity*:*MMEGaudi2BGemm*"
                    ;;
                regression )
                    __filter="*MmeUT*:*MMEGaudi2Sanity*:*MMEGaudi2BGemm*:*MMEGaudi2Conv*:*Regression*:*Reduction*:*SBReuse.*:*MMEGaudi2VlsiTests*:*MMEGaudi2DedwFp8Tests*:*MMEGaudi2DedwFp8Tests*:*MMEGaudi2Dedw2xTests*"
                    ;;
		unit_tests )
                    __filter="*MmeUT*"
		    ;;
                *)
                    ;;
            esac
        fi
        if [ "$_type" == "gaudi" ]; then
            # If no filetr is set and no test suit and chosen, use all
            if [ "$__test_suit" == "" ]; then
                if [ "$__filter" == "" ]; then
                    __test_suit="all"
                fi
            fi

            case $__test_suit in
                sanity )
                    __filter="*MmeUT*:*MMEGaudiSanityTests*"
                    ;;
                regression )
                    __filter="*MmeUT*:*MMEGaudiRegressionTests*"
                    ;;
                networks )
                    __filter="*MMEGaudiNetworkTests*"
                    ;;
		unit_tests )
                    __filter="*MmeUT*"
		    ;;
                all)
                    __filter=-"*MMEGaudiOtherTests*"
                    ;;
            esac
        fi

        # change mme_test_exe according to chip type
        __mme_test_exe=${__mme_test_exe_base}_${_type}

        _verify_exists_file "$__mme_test_exe" $__mme_test_exe

        if [ -n "$__print_tests" ]; then
            ${__mme_test_exe} --gtest_list_tests --gtest_color=$__color --gtest_filter=$__filter":-"$__negative_filter
            return $?
        fi

        local __gtest_flags="${__shuffle} --gtest_random_seed=$__seed --gtest_repeat=$__num_iterations $__en_disable --gtest_color=$__color --gtest_filter=$__filter:-$__negative_filter $__xml"
        if [ -n "$__run_until_failure" ]; then
            while (set -x; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${__ld_lib}; ${__debug_tool} ${__mme_test_exe} ${__gtest_flags} ${__test_flags}); do :; done
        else
            (set -x; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${__ld_lib}; ${__debug_tool} ${__mme_test_exe} ${__gtest_flags} ${__test_flags})
        fi
        __ret=$? || ${__ret}
    done
    # return error code of the test
    return $__ret
}

run_habana_py_test()
{
    local __habana_py_test_exe="$__python_cmd -m pytest"

    local __scriptname=$(__get_func_name)
    local __print_tests=""
    local __filter=""
    local __pdb=""
    local __xml=""
    local __failures=""
    local __seed=""
    local __profiler="0"
    local __rand="--random-order"
    local __marker="-m  \"not daily and not asic_only\""
    local __color=""
    local __skip_measurements=""
    local __skip_execution=""
    local __export_measurements=""
    local __renew_measurements=""
    local __spdlog_all="4"
    local __spdlog_habana_py="2"
    local __dir="$HABANA_PY_TESTS/goya_tests/"

    # parameter while-loop
    while [ -n "$1" ];
    do
        case $1 in
        -c  | --chip )
            shift
            __dir=$HABANA_PY_TESTS/$1_tests/
            ;;
        -l  | --list-tests )
            __print_tests="yes"
            ;;
        -s  | --specific-test )
            shift
            __filter="-k \"$1\""
            ;;
        -a | --marker )
            shift
            __marker="-m \"$1\""
            ;;
        -m  | --maxfail )
            shift
            __failures="--maxfail=$1"
            ;;
        -p  | --pdb )
            shift
            __pdb="--pdb"
            ;;
        -seed  | --seed )
            shift
            __seed="--random-order-seed=$1"
            ;;
        -prof )
            __profiler="1"
            ;;
        -x  | --xml )
            shift
            __xml="--junit-xml=$1"
            ;;
        -nr  | --rand_disable )
            shift
            __rand=""
            ;;
        -sm  | --skip_measurements )
            __skip_measurements="--skip_measurements"
            ;;
        -sx  | --skip_execution )
            __skip_execution="--skip_execution"
            ;;
        -em  | --export_measurements )
            __export_measurements="--export_measurements"
            ;;
        -rm | --renew_measurements )
            __renew_measurements=true
            # For the pytest command the meaning is same as when --export_measurements was given, so we store as same variable
            __export_measurements="--export_measurements"
            ;;
        --no-color )
            __color="--color=no"
            ;;
        -spdlog )
            shift
            __spdlog_all=$1
            __spdlog_habana_py=$1
            ;;
        -h  | --help )
            usage $__scriptname
            return 0
            ;;
        *)
            echo "The parameter $1 is not allowed"
            usage $__scriptname
            return 1 # error
            ;;
        esac
        shift
    done

    # check if colliding arguments given
    if [[ -n "$__skip_measurements"  &&  -n "$__export_measurements" ]]; then
        echo "tests can't run with skip_measurements and one of export_measurements, renew_measurements"
        usage $__scriptname
        return 1 # error
    fi
    # create dyn range files directory according to arguments
    if [ "$__renew_measurements" ]; then
        # delete dyn range files dir and its contents, re-create it empty
        if [ -d $HABANA_PY_DYN_RANGE_FILES ]; then
            rm -rf $HABANA_PY_DYN_RANGE_FILES
        fi
        mkdir -p $HABANA_PY_DYN_RANGE_FILES
    elif [[ "$__export_measurements" || "$__skip_measurements" ]]; then
        # create dyn range files dir if doesn't exist, and pull its content from artifactory
        if [ ! -d $HABANA_PY_DYN_RANGE_FILES ]; then
            mkdir -p $HABANA_PY_DYN_RANGE_FILES
            # TODO - pull dyn range files from artifactory
            echo "Need to pull dynamic range files from artifactory to HABANA_PY_DYN_RANGE_FILES"
        fi
    fi

    if [ -n "$__print_tests" ]; then
        ${__habana_py_test_exe} --collect-only
        return $?
    fi

    (set -x; eval HABANA_PROFILE=${__profiler} LOG_LEVEL_ALL=${__spdlog_all} LOG_LEVEL_HABANAPY=${__spdlog_habana_py} \
    ${__habana_py_test_exe} $__dir -v $__failures $__filter $__xml $__pdb $__seed $__rand $__color ${__marker} \
    $__skip_measurements $__skip_execution $__export_measurements)

    # return error code of the test
    return $?
}

build_engines_fw ()
{
    SECONDS=0

    _verify_exists_dir "$ENGINES_FW_ROOT" $ENGINES_FW_ROOT

    local __scriptname=$(__get_func_name)

    local __jobs=$NUMBER_OF_JOBS
    local __color="ON"
    local __debug="yes"
    local __release=""
    local __all=""
    local __coral="no"
    local __asic="yes"
    local __bfmso="yes"
    local __fw_tests="yes"
    local __stm_trace_level="0"
    local __configure=""
    local __org_configure=""
    local __build_res="0"
    local __verbose=""
    local __sanitize="NO"
    local __chip_type="all"
    local __recursive=""

    if [ $OS == "red" ]; then
        echo "Detected RHEL 8.6. Build ARC FW binaries for Asic skipped by default."
        __asic=""
    fi

    # parameter while-loop
    while [ -n "$1" ];
    do
        case $1 in
        -a  | --build-all )
            __all="yes"
            ;;
        -p  | --build-coral )
            __coral="yes"
            ;;
        -noasic  | --build-noasic-fw )
            __asic=""
            ;;
        -nobfm  | --build-nobfm-so )
            __bfmso=""
            ;;
        -notests  | --build-no-tests )
            __fw_tests=""
            ;;
        -t  | --build-fw-tests )
            __fw_tests="yes"
            ;;
        -stm  | --stm-trace )
            shift
            __stm_trace_level=$1
            ;;
        -c  | --configure )
            __org_configure="yes"
            __configure="yes"
            ;;
        -C  | --chip-type )
            shift
            __chip_type=$1
            ;;
        -r  | --release )
            __debug=""
            __release="yes"
            ;;
        --recursive )
            __recursive="yes"
            ;;
        -j  | --jobs )
            shift
            __jobs=$1
            ;;
        -s  | --sanitize )
            __sanitize="ON"
            ;;
        -v  | --verbose )
            __verbose="VERBOSE=1"
            ;;
        -h  | --help )
            usage $__scriptname
            return 0
            ;;
        *)
            echo "The parameter $1 is not allowed"
            usage $__scriptname
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

    export SNPSLMD_LICENSE_FILE=27020@licsrv03.kfs.habana-labs.com:27020@licsrv01.kfs.habana-labs.com
    export LM_LICENSE_FILE=27020@licsrv03.kfs.habana-labs.com:27020@licsrv01.kfs.habana-labs.com

    if [ -n "$__recursive" ]; then
        local __release_par=""
        local __configure_par=""
        local __jobs_par=""

        if [ -n "$__configure" ]; then
            __configure_par="-c"
        fi

        if [ -n "$__release" ]; then
            __release_par="-r"
        fi

        if [ -n "$__all" ]; then
            __release_par="-a"
        fi

        __jobs_par="-j $__jobs"

        echo "Building pre-requisite packages for Engines FW"

        __common_build_dependency -m "engines_fw" $__configure_par $__release_par $__jobs_par
        if [ $? -ne 0 ]; then
            echo "Failed to build dependency packages"
            return 1
        fi
    fi

    if [ -n "$__asic" ]; then
        _verify_exists_dir "/tools" /tools/synopsys
        if [ $? -ne 0 ]; then
            echo "Make sure /tools is mounted on our machine, talk to IT"
            return 1
        fi
    fi

    if [ -n "$__debug" ]; then
        echo -e "Building in debug mode"
        if [ -n "$__configure" ]; then
            rm -rf $ENGINES_FW_DEBUG_BUILD
        fi

        mkdir -p $ENGINES_FW_DEBUG_BUILD
        _verify_exists_dir "$ENGINES_FW_DEBUG_BUILD" $ENGINES_FW_DEBUG_BUILD
        pushd $ENGINES_FW_DEBUG_BUILD

        if [ -n "$__bfmso" ]; then
            (set -x; cmake \
                -DCMAKE_BUILD_TYPE="Debug" \
                -DENABLE_STM_TRACING=$__stm_trace_level \
                -DBUILD_ENV=noec \
                -DARC_FW_CHIP_TYPE=$__chip_type \
                -DUSE_CORAL_CYCLIC_MODE=$__coral \
                -DCMAKE_COLOR_MAKEFILE=$__color \
                -DSANITIZE_ON=$__sanitize \
                $ENGINES_FW_ROOT/coral_wrapper)
            __build_res=$?
            if [ $__build_res -eq 0 ]; then
                make $__verbose -j$__jobs
                __build_res=$?
            fi
        fi
        if [ $__build_res -eq 0 ] && [ -n "$__asic" ]; then
            mkdir -p asic
            _verify_exists_dir "asic" asic
            pushd asic
            (set -x; cmake \
                -DCMAKE_BUILD_TYPE="Debug" \
                -DENABLE_STM_TRACING=$__stm_trace_level \
                -DARC_FW_CHIP_TYPE=$__chip_type \
                -DCMAKE_COLOR_MAKEFILE=$__color \
                -DARC_DEBUG=1 \
                $ENGINES_FW_ROOT)
            __build_res=$?
            if [ $__build_res -eq 0 ]; then
                # TODO: remove hardcoding of verbose value
                make VERBOSE=1 -j$__jobs
                __build_res=$?
            fi
            popd
        fi
        if [ $__build_res -eq 0 ] && [ -n "$__fw_tests" ]; then
            mkdir -p fw_tests
            _verify_exists_dir "fw_tests" fw_tests
            pushd fw_tests
            (set -x; cmake \
                -DCMAKE_BUILD_TYPE="Debug" \
                -DCMAKE_COLOR_MAKEFILE=$__color \
                -DSANITIZE_ON=$__sanitize \
                -DARC_FW_CHIP_TYPE=$__chip_type \
                $ENGINES_FW_ROOT/fw_tests)
            __build_res=$?
            if [ $__build_res -eq 0 ]; then
                make $__verbose -j$__jobs
                __build_res=$?
            fi
            popd
        fi
        popd
        if [ $__build_res -ne 0 ]; then
            return $__build_res
        fi
    fi

    if [ -n "$__release" ]; then
        echo "Building in release mode"
        if [ -n "$__configure" ]; then
            rm -rf $ENGINES_FW_RELEASE_BUILD
        fi

        mkdir -p $ENGINES_FW_RELEASE_BUILD
        _verify_exists_dir "$ENGINES_FW_RELEASE_BUILD" $ENGINES_FW_RELEASE_BUILD
        pushd $ENGINES_FW_RELEASE_BUILD

        if [ -n "$__bfmso" ]; then
            mkdir -p coral
            _verify_exists_dir "coral" coral
            pushd coral
            (set -x; cmake \
                -DCMAKE_BUILD_TYPE="Release" \
                -DENABLE_STM_TRACING=$__stm_trace_level \
                -DBUILD_ENV=noec \
                -DARC_FW_CHIP_TYPE=$__chip_type \
                -DUSE_CORAL_CYCLIC_MODE=$__coral \
                -DCMAKE_COLOR_MAKEFILE=$__color \
                -DSANITIZE_ON=$__sanitize \
                $ENGINES_FW_ROOT/coral_wrapper)
            __build_res=$?
            if [ $__build_res -eq 0 ]; then
                make $__verbose -j$__jobs
                __build_res=$?
            fi
            popd
        fi
        if [ $__build_res -eq 0 ] && [ -n "$__asic" ]; then
            mkdir -p asic
            _verify_exists_dir "asic" asic
            pushd asic
            (set -x; cmake \
                -DCMAKE_BUILD_TYPE="Release" \
                -DENABLE_STM_TRACING=$__stm_trace_level \
                -DARC_FW_CHIP_TYPE=$__chip_type \
                -DCMAKE_COLOR_MAKEFILE=$__color \
                $ENGINES_FW_ROOT)
            __build_res=$?
            if [ $__build_res -eq 0 ]; then
                # TODO: remove hardcoding of verbose value
                make VERBOSE=1 -j$__jobs
                __build_res=$?
            fi
            popd
        fi
        if [ $__build_res -eq 0 ] && [ -n "$__fw_tests" ]; then
            mkdir -p fw_tests
            _verify_exists_dir "fw_tests" fw_tests
            pushd fw_tests
            (set -x; cmake \
                -DCMAKE_BUILD_TYPE="Release" \
                -DCMAKE_COLOR_MAKEFILE=$__color \
                -DSANITIZE_ON=$__sanitize \
                -DARC_FW_CHIP_TYPE=$__chip_type \
                $ENGINES_FW_ROOT/fw_tests)
            __build_res=$?
            if [ $__build_res -eq 0 ]; then
                make $__verbose -j$__jobs
                __build_res=$?
            fi
            popd
        fi
        popd
        if [ $__build_res -ne 0 ]; then
            return $__build_res
        fi
    fi

    printf "\nElapsed time: %02u:%02u:%02u \n\n" $(($SECONDS / 3600)) $((($SECONDS / 60) % 60)) $(($SECONDS % 60))
    return 0
}

build_habana_tfrt()
{
    _verify_exists_dir "$SYNAPSE_ROOT" $SYNAPSE_ROOT
    _verify_exists_dir "$HABANA_TFRT_ROOT" $HABANA_TFRT_ROOT

    local __scriptname=$(__get_func_name)

    local __jobs=${NUMBER_OF_JOBS}
    local __configure=""
    local __all=""
    local __debug="yes"
    local __release=""

    # parameter while-loop
    while [ -n "$1" ];
    do
        case $1 in
        -j  | --jobs )
            shift
            __jobs=$1
            ;;
        -c  | --configure )
            __configure="yes"
            ;;
        -a  | --build-all )
            __all="yes"
            ;;
        -r  | --release )
            __debug=""
            __release="yes"
            ;;
        -h  | --help )
            usage $__scriptname
            return 0
            ;;
        *)
            echo "The parameter $1 is not allowed"
            usage $__scriptname
            return 1 # error
            ;;
        esac
        shift
    done

    if [ -n "$__all" ]; then
        __debug="yes"
        __release="yes"
    fi

    if [ -n "$__debug" ]; then
        echo -e "Building in Debug mode"
        if [ ! -d $HABANA_TFRT_DEBUG_BUILD ]; then
            __configure="yes"
        fi

        if [ -n "$__configure" ]; then
            if [ -d $HABANA_TFRT_DEBUG_BUILD ]; then
                rm -rf $HABANA_TFRT_DEBUG_BUILD
            fi
            mkdir -p $HABANA_TFRT_DEBUG_BUILD
        fi

        _verify_exists_dir "$HABANA_TFRT_DEBUG_BUILD" $HABANA_TFRT_DEBUG_BUILD
        _verify_exists_dir "${SYNAPSE_DEBUG_BUILD}/lib" ${SYNAPSE_DEBUG_BUILD}/lib

        pushd $HABANA_TFRT_ROOT
        make $__verbose -j$__jobs DEBUG=1 DEST=$HABANA_TFRT_DEBUG_BUILD/ \
            HABANA_LIB=${SYNAPSE_DEBUG_BUILD}/lib HABANA_LIB_INCLUDE=${SYNAPSE_ROOT}
        __build_res=$?
        popd

        if [ $__build_res -ne 0 ]; then
            return $__build_res
        fi

        cp -fs $HABANA_TFRT_DEBUG_BUILD/*.so $BUILD_ROOT_DEBUG
        if [ -z "$__all" ]; then
            cp -fs $HABANA_TFRT_DEBUG_BUILD/*.so $BUILD_ROOT_LATEST
        fi
    fi

    if [ -n "$__release" ]; then
                echo -e "Building in Release mode"
        if [ ! -d $HABANA_TFRT_RELEASE_BUILD ]; then
            __configure="yes"
        fi

        if [ -n "$__configure" ]; then
            if [ -d $HABANA_TFRT_RELEASE_BUILD ]; then
                rm -rf $HABANA_TFRT_RELEASE_BUILD
            fi
            mkdir -p $HABANA_TFRT_RELEASE_BUILD
        fi

        _verify_exists_dir "$HABANA_TFRT_RELEASE_BUILD" $HABANA_TFRT_RELEASE_BUILD
        _verify_exists_dir "${SYNAPSE_RELEASE_BUILD}/lib" ${SYNAPSE_RELEASE_BUILD}/lib

        pushd $HABANA_TFRT_ROOT
        make $__verbose -j$__jobs DEST=$HABANA_TFRT_RELEASE_BUILD/ \
            HABANA_LIB=${SYNAPSE_RELEASE_BUILD}/lib HABANA_LIB_INCLUDE=${SYNAPSE_ROOT}
        __build_res=$?
        popd

        if [ $__build_res -ne 0 ]; then
            return $__build_res
        fi

        cp -fs $HABANA_TFRT_RELEASE_BUILD/*.so $BUILD_ROOT_RELEASE
        if [ -z "$__all" ]; then
            cp -fs $HABANA_TFRT_RELEASE_BUILD/*.so $BUILD_ROOT_LATEST
        fi
    fi

    printf "\nElapsed time: %02u:%02u:%02u \n\n" $(($SECONDS / 3600)) $((($SECONDS / 60) % 60)) $(($SECONDS % 60))
    return 0
}

build_scal ()
{
    SECONDS=0

    _verify_exists_dir "$SCAL_ROOT" $SCAL_ROOT
    : "${SCAL_DEBUG_BUILD:?Need to set SCAL_DEBUG_BUILD to the build folder}"
    : "${SCAL_RELEASE_BUILD:?Need to set SCAL_RELEASE_BUILD to the build folder}"

    local __scriptname=$(__get_func_name)

    local __jobs=$NUMBER_OF_JOBS
    local __color="ON"
    local __debug="yes"
    local __release=""
    local __all=""
    local __configure=""
    local __org_configure=""
    local __build_res=""
    local __sanitize="NO"
    local __tsanitize="NO"
    local __valgrind="NO"
    local __verbose=""
    local __scal_tests="ON"
    local __recursive=""

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
        --recursive )
            __recursive="yes"
            ;;
        -j  | --jobs )
            shift
            __jobs=$1
            ;;
        -s  | --sanitize )
            __sanitize="ON"
            ;;
        -ts | --tsanitize )
            __tsanitize="ON"
            ;;
        -v  | --verbose )
            __verbose="VERBOSE=1"
            ;;
        -V  | --valgrind  )
            __valgrind="ON"
            ;;
        -l  | --scal_only  )
            __scal_tests="OFF"
            ;;
        -h  | --help )
            usage $__scriptname
            return 0
            ;;
        *)
            echo "The parameter $1 is not allowed"
            usage $__scriptname
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

    if [ -n "$__recursive" ]; then
        local __release_par=""
        local __configure_par=""
        local __jobs_par=""

        if [ -n "$__configure" ]; then
            __configure_par="-c"
        fi

        if [ -n "$__release" ]; then
            __release_par="-r"
        fi

        if [ -n "$__all" ]; then
            __release_par="-a"
        fi

        __jobs_par="-j $__jobs"

        echo "Building pre-requisite packages for Synapse"

        __common_build_dependency -m "scal" $__configure_par $__release_par $__jobs_par
        if [ $? -ne 0 ]; then
            echo "Failed to build dependency packages"
            return 1
        fi
    fi

    if [ -n "$__debug" ]; then
        echo -e "Building in debug mode"
        local __folder=$SCAL_DEBUG_BUILD
        if [[ $__sanitize = "ON" || $__tsanitize = "ON" ]]; then
            echo -e "building with sanitizer in folder $SCAL_DEBUG_SANITIZER_BUILD"
            __folder=$SCAL_DEBUG_SANITIZER_BUILD
        fi
        if [ ! -d $__folder ]; then
            __configure="yes"
        fi

        if [ -n "$__configure" ]; then
            if [ -d $__folder ]; then
                rm -rf $__folder
            fi
            mkdir -p $__folder
        fi

        _verify_exists_dir "$__folder" $__folder
        pushd $__folder
            (set -x; cmake \
            -DCMAKE_BUILD_TYPE="Debug" \
            -DCMAKE_INSTALL_PREFIX=$__folder/install \
            -DCMAKE_COLOR_MAKEFILE=$__color \
            -DSANITIZE_ON=$__sanitize \
            -DTSANITIZE_ON=$__tsanitize \
            -DVALGRIND_ON=$__valgrind \
            -DTESTS_ENABLED=$__scal_tests \
            $SCAL_ROOT)
        make $__verbose -j$__jobs && make install
        __build_res=$?
        popd
        if [ $__build_res -ne 0 ]; then
            return $__build_res
        fi
        cp -r $__folder/install/* $__folder
        _copy_build_products $__folder
    fi

    __configure=$__org_configure

    if [ -n "$__release" ]; then
        echo "Building in release mode"
        if [ ! -d $SCAL_RELEASE_BUILD ]; then
            __configure="yes"
        fi

        if [ -n "$__configure" ]; then
            if [ -d $SCAL_RELEASE_BUILD ]; then
                rm -rf $SCAL_RELEASE_BUILD
            fi
            mkdir -p $SCAL_RELEASE_BUILD
        fi

        _verify_exists_dir "$SCAL_RELEASE_BUILD" $SCAL_RELEASE_BUILD
        pushd $SCAL_RELEASE_BUILD
        (set -x; cmake \
            -DCMAKE_BUILD_TYPE="Release" \
            -DCMAKE_INSTALL_PREFIX=$SCAL_RELEASE_BUILD/install \
            -DCMAKE_COLOR_MAKEFILE=$__color \
            -DSANITIZE_ON=$__sanitize \
            -DTSANITIZE_ON=$__tsanitize \
            -DVALGRIND_ON=$__valgrind \
            -DTESTS_ENABLED=$__scal_tests \
            $SCAL_ROOT)
        make $__verbose -j$__jobs && make install
        __build_res=$?
        popd
        if [ $__build_res -ne 0 ]; then
            return $__build_res
        fi
        cp -r $SCAL_RELEASE_BUILD/install/* $SCAL_RELEASE_BUILD
        _copy_build_products $SCAL_RELEASE_BUILD -r
    fi

    printf "\nElapsed time: %02u:%02u:%02u \n\n" $(($SECONDS / 3600)) $((($SECONDS / 60) % 60)) $(($SECONDS % 60))
    return 0
}

run_scal_test()
{
    local __scaltest_exe=$SCAL_DEBUG_BUILD/bin/scal_tests
    local __ld_lib=$BUILD_ROOT_DEBUG

    local __scriptname=$(__get_func_name)
    local __print_tests=""
    local __num_iterations=1
    local __gdb=""
    local __valgrind=""
    local __filter="*"
    local __gdbserver=""
    local __xml=""
    local __spdlog="4"
    local __profiler="0"
    local __en_disable=""
    local __sudo=" "
    local __run_until_failure=""
    local __shuffle=""
    local __seed=0
    local __color="yes"
    local __debug_tool=""
    local __ret=0
    local __reset_terminal=""
    local __debug="yes"
    local __release=""
    local __scal_device_type="0"

    # parameter while-loop
    while [ -n "$1" ];
    do
        case $1 in
        -g  | --gdbserver )
            __gdbserver="yes"
            __debug_tool="gdbserver localhost:2345"
            ;;
        -gdb )
            __gdb="yes"
            __debug_tool="gdb --args"
            ;;
        -v  | --valgrind )
            __valgrind="yes"
            __debug_tool="valgrind"
            ;;
        -spdlog )
            shift
            __spdlog=$1
            ;;
        -sudo )
            __sudo="sudo LD_LIBRARY_PATH=$LD_LIBRARY_PATH -E "
            ;;
        -shuffle )
            __shuffle="--gtest_shuffle"
            ;;
        -seed | --gtest_random_seed )
            shift
            __seed=$1
            __shuffle="--gtest_shuffle"
            ;;
        -prof )
            __profiler="1"
            ;;
        --sanitizer )
            __scaltest_exe=$SCAL_DEBUG_SANITIZER_BUILD/bin/scal_tests
            ;;
        -f  | --run-until-failure )
            __run_until_failure="yes"
            ;;
        -r  | --release )
            __scaltest_exe=$SCAL_RELEASE_BUILD/bin/scal_tests
            __ld_lib=$BUILD_ROOT_RELEASE
            __debug=""
            __release="yes"
            ;;
        -i  | --iterations )
            shift
            __num_iterations=$1
            ;;
        -l  | --list-tests )
            __print_tests="yes"
            ;;
        -c  | --chip-type )
            shift
            case $1 in
            gaudi2 )
                __scal_device_type="0"
            ;;
            gaudi3 )
                __scal_device_type="1"
            ;;
            esac
            ;;
        -s  | --specific-test )
            shift
            __filter=$1
            __negative_filter=""
            ;;
        -a  | --disable-additional-tests )
            shift
            __negative_filter=$__negative_filter":"$1
            ;;
        -d  | --disabled-test )
            __en_disable="--gtest_also_run_disabled_tests"
            ;;
        -x  | --xml )
            shift
            __xml="--gtest_output=xml:"$1
            ;;
        --reset-terminal )
            __reset_terminal="yes"
            ;;
        --no-color )
            __color="no"
            ;;
        -h  | --help )
            usage $__scriptname
            return 0
            ;;
        *)
            echo "The parameter $1 is not allowed"
            usage $__scriptname
            return 1 # error
            ;;
        esac
        shift
    done

    echo -e "Running scal tests"

    _verify_exists_file "$__scaltest_exe" $__scaltest_exe

    if [ -n "$__print_tests" ]; then
        ${__scaltest_exe} --gtest_list_tests --gtest_color=$__color --gtest_filter=$__filter":-"$__negative_filter
        return $?
    fi

    if [ -n "$__run_until_failure" ]; then
        while (set -x; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${__ld_lib}; export LOG_LEVEL_ALL=${__spdlog}; export HABANA_PROFILE=${__profiler}; export SCAL_DEVICE_TYPE=${__scal_device_type}; ${__sudo} ${__scaltest_exe} $__en_disable --gtest_color=$__color ${__shuffle} --gtest_random_seed=$__seed --gtest_filter=$__filter $__xml); do :; done
    else
        (set -x; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${__ld_lib}; export LOG_LEVEL_ALL=${__spdlog}; export HABANA_PROFILE=${__profiler}; export SCAL_DEVICE_TYPE=${__scal_device_type}; ${__sudo} ${__debug_tool} ${__scaltest_exe} ${__shuffle} --gtest_random_seed=$__seed --gtest_repeat=$__num_iterations $__en_disable --gtest_color=$__color --gtest_filter=$__filter":-"$__negative_filter $__xml)
    fi
    __ret=$?

    # reset terminal to clear the colors that gtest leaves behind
    if [ -n "$__reset_terminal" ]; then
        _reset_terminal
    fi

    # return error code of the test
    return $__ret
}

run_engines_fw_test()
{
    local __engines_fw_test_exe=$ENGINES_FW_DEBUG_BUILD/fw_tests/
    local __ld_lib=$BUILD_ROOT_DEBUG

    local __scriptname=$(__get_func_name)
    local __print_tests=""
    local __num_iterations=1
    local __gdb=""
    local __valgrind=""
    local __filter="*"
    local __gdbserver=""
    local __xml=""
    local __nic_loopback=""
    local __spdlog="4"
    local __profiler="0"
    local __en_disable=""
    local  __break_on_first_failure=""
    local __sudo=" "
    local __run_until_failure=""
    local __shuffle=""
    local __seed=0
    local __color="yes"
    local __debug_tool=""
    local __ret=0
    local __reset_terminal=""
    local __debug="yes"
    local __release=""
    local __chip_type="gaudi2"
    local __fw_tests_file="fw_tests"
    local __scal_bin_path=$ENGINES_FW_DEBUG_BUILD
    local __shard_index=0
    local __total_shards=1

    # parameter while-loop
    while [ -n "$1" ];
    do
        case $1 in
        -g  | --gdbserver )
            __gdbserver="yes"
            __debug_tool="gdbserver localhost:2345"
            ;;
        -gdb )
            __gdb="yes"
            __debug_tool="gdb --args"
            ;;
        -v  | --valgrind )
            __valgrind="yes"
            __debug_tool="valgrind"
            ;;
        -spdlog )
            shift
            __spdlog=$1
            ;;
        -sudo )
            __sudo="sudo LD_LIBRARY_PATH=$LD_LIBRARY_PATH -E "
            ;;
        -shuffle )
            __shuffle="--gtest_shuffle"
            ;;
        -seed | --gtest_random_seed )
            shift
            __seed=$1
            __shuffle="--gtest_shuffle"
            ;;
        -prof )
            __profiler="1"
            ;;
        -f  | --run-until-failure )
            __run_until_failure="yes"
            ;;
        -r  | --release )
            __engines_fw_test_exe=$ENGINES_FW_RELEASE_BUILD/fw_tests/
            __scal_bin_path=$ENGINES_FW_RELEASE_BUILD
            __ld_lib=$BUILD_ROOT_RELEASE
            __debug=""
            __release="yes"
            ;;
        -C  | --chip-type )
            shift
            __chip_type=$1
            ;;
        -i  | --iterations )
            shift
            __num_iterations=$1
            ;;
        -l  | --list-tests )
            __print_tests="yes"
            ;;
        -s  | --specific-test )
            shift
            __filter=$1
            __negative_filter=""
            ;;
        -a  | --disable-additional-tests )
            shift
            __negative_filter=$__negative_filter":"$1
            ;;
        -d  | --disabled-test )
            __en_disable="--gtest_also_run_disabled_tests"
            ;;
        -b  | --break-on-first-failure )
            __break_on_first_failure="--gtest_break_on_failure"
            ;;
        -x  | --xml )
            shift
            __xml="--gtest_output=xml:"$1
            ;;
        -n  | --nic-loopback )
            __nic_loopback="yes"
            ;;
        --reset-terminal )
            __reset_terminal="yes"
            ;;
        --no-color )
            __color="no"
            ;;
        --shard-index )
            shift
            __shard_index=$1
            ;;
        --total-shards )
            shift
            __total_shards=$1
            ;;
        -h  | --help )
            usage $__scriptname
            return 0
            ;;
        *)
            echo "The parameter $1 is not allowed"
            usage $__scriptname
            return 1 # error
            ;;
        esac
        shift
    done

    echo -e "Running runtime firmware (engines_fw) tests for:" $__chip_type

    export HABANA_SCAL_BIN_PATH=${__scal_bin_path}

    if [ $__chip_type = "gaudi3" ]; then
        __fw_tests_file="g3_fw_tests"
    fi

    __engines_fw_test_exe="$__engines_fw_test_exe$__fw_tests_file"
    _verify_exists_file "$__engines_fw_test_exe" $__engines_fw_test_exe

    if [ -n "$__print_tests" ]; then
        ${__engines_fw_test_exe} --gtest_list_tests --gtest_color=$__color --gtest_filter=$__filter":-"$__negative_filter
        return $?
    fi

    if [ -n "$__nic_loopback" ]; then
        echo "Enable NIC loopback mode cn"
        $sudo echo 0xffffff | sudo tee /sys/kernel/debug/habanalabs_cn/hl_cn0/nic_mac_loopback
    fi

    if [ -n "$__run_until_failure" ]; then
        while (set -x; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${__ld_lib}; export LOG_LEVEL_ALL=${__spdlog}; export HABANA_PROFILE=${__profiler}; export GTEST_TOTAL_SHARDS=${__total_shards}; export GTEST_SHARD_INDEX=${__shard_index}; ${__sudo} ${__engines_fw_test_exe} $__en_disable --gtest_color=$__color ${__shuffle} --gtest_random_seed=$__seed --gtest_filter=$__filter $__xml); do :; done
    else
        (set -x; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${__ld_lib}; export LOG_LEVEL_ALL=${__spdlog}; export HABANA_PROFILE=${__profiler}; export GTEST_TOTAL_SHARDS=${__total_shards}; export GTEST_SHARD_INDEX=${__shard_index}; ${__sudo} ${__debug_tool} ${__engines_fw_test_exe} ${__shuffle} --gtest_random_seed=$__seed --gtest_repeat=$__num_iterations $__en_disable  $__break_on_first_failure --gtest_color=$__color --gtest_filter=$__filter":-"$__negative_filter $__xml)
    fi
    __ret=$?

    # reset terminal to clear the colors that gtest leaves behind
    if [ -n "$__reset_terminal" ]; then
        _reset_terminal
    fi

    if [ -n "$__nic_loopback" ]; then
        echo "Disable NIC loopback mode"
        $sudo echo 0 | sudo tee /sys/kernel/debug/habanalabs_cn/hl_cn0/nic_mac_loopback
    fi
    # return error code of the test
    return $__ret
}

run_demos_tests()
{
    local __demos_test_exe="$__python_cmd -m pytest"
    local __scriptname=$(__get_func_name)
    local __print_tests=""
    local __filter=""
    local __pdb=""
    local __xml=""
    local __failures=""
    local __seed=""
    local __profiler="0"
    local __rand_disable=""
    local __color=""
    local __capture=""
    local __spdlog_all="4"
    local __spdlog_habana_py="2"
    local __dir="$DEMOS_ROOT/gaudi/"
    local __test_status=0
    local __num_threads_conf=`nproc --all`
    local __marker=""
    local __pytest_marks="-m CI"

    # parameter while-loop
    while [ -n "$1" ];
    do
        case $1 in
        -l  | --list-tests )
            __print_tests="yes"
            ;;
        -s  | --specific-test )
            shift
            __filter="-k $1"
            ;;
        -m  | --maxfail )
            shift
            __failures="--maxfail=$1"
            ;;
        -a | --marker )
            shift
            __marker="-m \"$1\""
            ;;
        -p  | --pdb )
            shift
            __pdb="--pdb"
            ;;
        -seed  | --seed )
            shift
            __seed="--random-order-seed=$1"
            ;;
        -prof )
            __profiler="1"
            ;;
        -x  | --xml )
            shift
            __xml="--junit-xml=$1"
            ;;
        -nr  | --rand_disable )
            shift
            __rand_disable="--random-order-bucket=none"
            ;;
        --no-color )
            __color="--color=no"
            ;;
        --no-capture )
            __capture="--capture=no"
            ;;
        -spdlog )
            shift
            __spdlog_all=$1
            __spdlog_habana_py=$1
            ;;
        -j )
            shift
            __num_threads_conf=$1
            ;;
        -h  | --help )
            usage $__scriptname
            return 0
            ;;
        *)
            echo "The parameter $1 is not allowed"
            usage $__scriptname
            return 1 # error
            ;;
        esac
        shift
    done

    if [ -n "$__print_tests" ]; then
        ${__demos_test_exe} ${__dir} ${__pytest_marks} --collect-only
        return $?
    fi

    # TODO: remove the ignore flag from both execute and list, and
    # move dir to root folder of demos [SW-22179]

    pushd ${__dir}
    (set -x; eval NUM_OF_THREADS_CONF=${__num_threads_conf} HABANA_PROFILE=${__profiler} LOG_LEVEL_ALL=${__spdlog_all} \
    ${__demos_test_exe} ${__pytest_marks} $__failures $__filter $__xml $__pdb $__seed $__rand_disable $__color \
    $__capture $__marker)
    __test_status=$?
    popd

    # return error code of the tests
    return ${__test_status}
}

build_gaudi_demo()
{
    SECONDS=0

    _verify_exists_dir "$DEMOS_ROOT" $DEMOS_ROOT
    #: "${RESNET_DEBUG_BUILD_GAUDI:?Need to set RESNET_DEBUG_BUILD_GAUDI to the build folder}"
    #: "${RESENT_RELEASE_BUILD_GAUDI:?Need to set RESNET_RELEASE_BUILD_GAUDI to the build folder}"

    local __scriptname=$(__get_func_name)

    local __jobs=$NUMBER_OF_JOBS
    local __color="ON"
    local __debug="yes"
    local __release=""
    local __all=""
    local __configure=""
    local __org_configure=""
    local __build_res=""

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
        -j  | --jobs )
            shift
            __jobs=$1
            ;;
        -h  | --help )
            usage $__scriptname
            return 0
            ;;
        *)
            echo "The parameter $1 is not allowed"
            usage $__scriptname
            return 1 # error
            ;;
        esac
        shift
    done

    if [ -n "$__all" ]; then
        __debug="yes"
        __release="yes"
    fi

    if [ -n "$__debug" ]; then
        echo -e "Building in debug mode"

        if [ ! -d $RESNET_DEBUG_BUILD_GAUDI ]; then
            __configure="yes"
        fi

        if [ -n "$__configure" ]; then
            if [ -d $RESENT_DEBUG_BUILD_GAUDI ]; then
                rm -rf $RESNET_DEBUG_BUILD_GAUDI
            fi
            mkdir -p $RESNET_DEBUG_BUILD_GAUDI
        fi

        _verify_exists_dir "$RESENT_DEBUG_BUILD_GAUDI" $RESNET_DEBUG_BUILD_GAUDI

        pushd $RESNET_DEBUG_BUILD_GAUDI
            (set -x; cmake \
            -DCMAKE_BUILD_TYPE="Debug" \
            -DCMAKE_INSTALL_PREFIX=$HOME/.local \
            -DCMAKE_COLOR_MAKEFILE=$__color \
            $DEMOS_ROOT/gaudi/resnet_training_app)
        make $__verbose -j$__jobs
        __build_res=$?
        popd

        if [ $__build_res -ne 0 ]; then
            return $__build_res
        fi

    fi

    __configure=$__org_configure

    if [ -n "$__release" ]; then
        echo "Building in release mode"
        if [ ! -d $RESNET_RELEASE_BUILD_GAUDI ]; then
            __configure="yes"
        fi

        if [ -n "$__configure" ]; then
            if [ -d $RESNET_RELEASE_BUILD_GAUDI ]; then
                rm -rf $RESNET_RELEASE_BUILD_GAUDI
            fi
            mkdir -p $RESNET_RELEASE_BUILD_GAUDI
        fi

        _verify_exists_dir "$RESNET_RELEASE_BUILD_GAUDI" $RESNET_RELEASE_BUILD_GAUDI

        pushd $RESNET_RELEASE_BUILD_GAUDI
        (set -x; cmake \
            -DCMAKE_BUILD_TYPE="Release" \
            -DCMAKE_INSTALL_PREFIX=$HOME/.local \
            -DCMAKE_COLOR_MAKEFILE=$__color \
            $DEMOS_ROOT/gaudi/resnet_training_app)
        make $__verbose -j$__jobs
        __build_res=$?
        popd

        if [ $__build_res -ne 0 ]; then
            return $__build_res
        fi

    fi

    printf "\nElapsed time: %02u:%02u:%02u \n\n" $(($SECONDS / 3600)) $((($SECONDS / 60) % 60)) $(($SECONDS % 60))
    return 0
}

function build_synapse_and_dep ()
{
    build_synapse --recursive "$@"
}

run_gemm_benchmarks_test()
{
    if command -v python3.6 >/dev/null 2>&1; then
        __python_cmd="python3.6"
    elif command -v python3 >/dev/null 2>&1; then
        __python_cmd="python3"
    else
        echo "No python3 installation found. Please install and run script again."
        return 1
    fi

    $__python_cmd $SYNAPSE_ROOT/tests/benchmarks_tests/gaudi/gemm_benchmarks_test.py $*
    return $?
}

run_models_tests()
{
    if command -v python3.6 >/dev/null 2>&1; then
        __python_cmd="python3.6"
    elif command -v python3 >/dev/null 2>&1; then
        __python_cmd="python3"
    else
        echo "No python3 installation found. Please install and run again."
        return 1
    fi

    if [ -z "$SYNAPSE_ROOT" ]
    then
        echo "SYNAPSE_ROOT path is not defined"
        return 1
    fi

    local __models_tests_py_test_exe="$__python_cmd $SYNAPSE_ROOT/tests/json_tests/models_tests/mpm_cl.py"
    local __models_tests_args="$*"
    ${__models_tests_py_test_exe} $__models_tests_args

    return $?
}

models_file_editor()
{
    if command -v python3.6 >/dev/null 2>&1; then
        __python_cmd="python3.6"
    elif command -v python3 >/dev/null 2>&1; then
        __python_cmd="python3"
    else
        echo "No python3 installation found. Please install and run again."
        return 1
    fi

    if [ -z "$SYNAPSE_ROOT" ]
    then
        echo "SYNAPSE_ROOT path is not defined"
        return 1
    fi

    local __mpm_editor_py_test_exe="$__python_cmd $SYNAPSE_ROOT/tests/json_tests/models_tests/mpm-editor.py"
    local __mpm_editor_args="$*"
    ${__mpm_editor_py_test_exe} $__mpm_editor_args

    return $?
}

run_from_json()
{
    if [ -z "$SYNAPSE_ROOT" ]
    then
        echo "SYNAPSE_ROOT path is not defined"
        return 1
    fi

    local __json_runner_py_test_exe="$__python_cmd $SYNAPSE_ROOT/scripts/json_runner.py"
    local __json_runner_args="$*"
    ${__json_runner_py_test_exe} $__json_runner_args

    return $?
}

synrec()
{
    if [ -z "$SYNAPSE_ROOT" ]
    then
        echo "SYNAPSE_ROOT path is not defined"
        return 1
    fi

    local __synrec_py="$__python_cmd $SYNAPSE_ROOT/scripts/synrec.py"
    local __synrec_args="$*"
    ${__synrec_py} ${__synrec_args}

    return $?
}

run_perf_test()
{
    if [ -z "$SYNAPSE_ROOT" ]
    then
        echo "SYNAPSE_ROOT path is not defined"
        return 1
    fi

    local __perf_test_py="$__python_cmd $SYNAPSE_ROOT/scripts/perf-test.py"
    local __perf_test_args="$*"
    ${__perf_test_py} ${__perf_test_args}

    return $?
}

build_synapse_mlir()
{
    SECONDS=0

    _verify_exists_dir "$SYNAPSE_MLIR_ROOT" $SYNAPSE_MLIR_ROOT
    : "${SYNAPSE_MLIR_DEBUG_BUILD:?Need to set SYNAPSE_MLIR_DEBUG_BUILD to the build folder}"
    : "${SYNAPSE_MLIR_RELEASE_BUILD:?Need to set SYNAPSE_MLIR_RELEASE_BUILD to the build folder}"

    local __scriptname=$(__get_func_name)

    local __jobs=$NUMBER_OF_JOBS
    local __color="ON"
    local __debug="yes"
    local __release=""
    local __all=""
    local __configure=""
    local __org_configure=""
    local __other_options=""
    local __build_res=""
    local __sanitize=""
    local __tidy=""
    local __valgrind=""
    local __verbose=""
    local __gen_docs="OFF"
    local __doc_mode="pdf"
    local __synapse_mlir_tests="ON"
    # take cmake from habana path, to allow usage of newer versions of cmake than OS default
    local __cmake=$CMAKE_BUILD/bin/cmake

    # parameter while-loop
    # uncomment relveant paramters when they are supported
    while [ -n "$1" ];
    do
        case $1 in

        #-a  | --build-all )
        #    __all="yes"
        #    ;;
        #--no-color )
        #    __color="NO"
        #    ;;
        -c  | --configure )
            __org_configure="yes"
            __configure="yes"
            ;;
        -r  | --release )
            __debug=""
            __release="yes"
            ;;
        #-j  | --jobs )
        #    shift
        #    __jobs=$1
        #    ;;
        #-s  | --sanitize )
        #    __sanitize="ON"
        #    ;;
        #-v  | --verbose )
        #    __verbose="VERBOSE=1"
        #    ;;
        #-V  | --valgrind  )
        #    __valgrind="ON"
        #    ;;
        -l  | --synapse_mlir_only  )
            __synapse_mlir_tests="OFF"
            ;;
        #-y  | --tidy )
        #    __tidy="yes"
        #    ;;

        -h  | --help )
            usage $__scriptname
            return 0
            ;;
        *)
            echo "The parameter $1 is not allowed"
            usage $__scriptname
            return 1 # error
            ;;
        esac
        shift
    done

    if [ ! -f $__cmake ]; then
        _build_cmake
    else
        echo "Found CMake binary."
    fi

    echo "build_types"
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


    echo -e "clang tidy"
    CLANG_TIDY_DEFINE="-DCLANG_TIDY=OFF"
    if [ ! -z "$__tidy" ]; then
        CLANG_TIDY_DEFINE="-DCLANG_TIDY=ON"
    fi

    if [ -n "$__debug" ]; then

        echo -e "Building in debug mode"
        local __folder=$SYNAPSE_MLIR_DEBUG_BUILD
        if [ $__sanitize = "ON" ]; then
            __folder=$SYNAPSE_MLIR_DEBUG_SANITIZER_BUILD
        fi

        if [ ! -d $__folder ]; then
            __configure="yes"
        fi

        if [ -n "$__configure" ]; then
            if [ -d $__folder ]; then
                rm -rf $__folder
            fi
            mkdir -p $__folder
        fi

        _verify_exists_dir "$__folder" $__folder
        pushd $__folder
            (set -x; $__cmake \
            -DCMAKE_BUILD_TYPE="Debug" \
            -DCMAKE_INSTALL_PREFIX=$HOME/.local \
            -DCMAKE_COLOR_MAKEFILE=$__color \
            -DSANITIZE_ON=$__sanitize \
            -DVALGRIND_ON=$__valgrind \
            -DTESTS_ENABLED=$__synapse_mlir_tests \
            $CLANG_TIDY_DEFINE \
            $__other_options \
            $SYNAPSE_MLIR_ROOT/mlir)
        make $__verbose -j$__jobs
        __build_res=$?
        popd
        if [ $__build_res -ne 0 ]; then
            return $__build_res
        fi
        _copy_build_products $__folder

    fi

    __configure=$__org_configure

    if [ -n "$__release" ]; then
        echo "Building in release mode"
        if [ ! -d $SYNAPSE_MLIR_RELEASE_BUILD ]; then
            __configure="yes"
        fi

        if [ -n "$__configure" ]; then
            if [ -d $SYNAPSE_MLIR_RELEASE_BUILD ]; then
                rm -rf $SYNAPSE_MLIR_RELEASE_BUILD
            fi
            mkdir -p $SYNAPSE_MLIR_RELEASE_BUILD
        fi

        _verify_exists_dir "$SYNAPSE_MLIR_RELEASE_BUILD" $SYNAPSE_MLIR_RELEASE_BUILD
        pushd $SYNAPSE_MLIR_RELEASE_BUILD
        (set -x; $__cmake \
            -DCMAKE_BUILD_TYPE="Release" \
            -DCMAKE_INSTALL_PREFIX=$HOME/.local \
            -DCMAKE_COLOR_MAKEFILE=$__color \
            -DSANITIZE_ON=$__sanitize \
            -DVALGRIND_ON=$__valgrind \
            -DTESTS_ENABLED=$__synapse_mlir_tests \
            $CLANG_TIDY_DEFINE \
            $__other_options \
            $SYNAPSE_MLIR_ROOT/mlir)
        make $__verbose -j$__jobs
        __build_res=$?
        popd
        if [ $__build_res -ne 0 ]; then
            return $__build_res
        fi
        _copy_build_products $SYNAPSE_MLIR_RELEASE_BUILD -r


    fi

    printf "\nElapsed time: %02u:%02u:%02u \n\n" $(($SECONDS / 3600)) $((($SECONDS / 60) % 60)) $(($SECONDS % 60))
    return 0
}

check_syn_singleton_interface()
{
    __check_interface_version 'synapse' 'include/internal/syn_singleton_interface.hpp' 'SYNAPSE_SINGLETON_INTERFACE_VERSION'
    # 0 for success, 1 for failure
    return $?
}

run_synapse_mlir_graph_test()
{
    # uncomment parameters when they are supported
    local __scriptname=$(__get_func_name)
    local __print_tests=""
    #local __num_iterations=1;
    local __shuffle=""
    #local __seed=0
    local __color="yes"
    local __user_lit_filter=""
    local __user_lit_filter_out=""
    local __lit_test_filter=""
    local __lit_test_filter_out=""
    local __lit_user_params=""       # custom params passed to lit tool
    local __lit_test_folder="$SYNAPSE_MLIR_RELEASE_BUILD" # TODO SW-65896 change according to build type
    local __test_xml=""
    # take cmake from habana path, to allow usage of newer versions of cmake than OS default
    local __cmake=$CMAKE_BUILD/bin/cmake

    # parameter while-loop
    while [ -n "$1" ];
    do
        case $1 in
        # -i  | --iterations )
        #     shift
        #     __num_iterations=$1
        #    ;;
        -r  )
            # currently has no effect, only for CI needs
            __lit_test_folder="$SYNAPSE_MLIR_RELEASE_BUILD"
            ;;
        -l  | --list-tests )
            __print_tests="yes"
            ;;
        -s  | --specific-test )
            shift
            __user_lit_filter=$1
            __lit_test_filter="--filter \"($__user_lit_filter)\""
            ;;
        -a | --specific-filter-out )
            shift
            __user_lit_filter_out=$1
            __lit_test_filter_out="--filter-out \"($__user_lit_filter_out)\""
            ;;
        -h  | --help )
            usage $__scriptname
            return 0
            ;;
        -shuffle )
            __shuffle="--shuffle"
            ;;
        # -seed | --gtest_random_seed )
        #     shift
        #     __seed=$1
        #     __shuffle="--gtest_shuffle"
        #    ;;
        --no-color )
            __color="no"
            ;;
        -x  | --xml )
        shift
        __test_xml="--xunit-xml-output $1/synapse_mlir.xml"
            ;;
        *)
            echo "The parameter $1 is not allowed"
            usage $__scriptname
            return 1 # error
            ;;
        esac
        shift
    done

    if [ ! -f $__cmake ]; then
        _build_cmake
    else
        echo "Found CMake binary."
    fi

    # set the env var used to pass arguments to lit tool
    # a - output both passing and failing tests
    LIT_OPTS="$__lit_test_filter $__lit_test_filter_out -a"

    if [ -n "$__print_tests" ]; then
        LIT_OPTS="$LIT_OPTS --show-tests"
    fi

    echo "Running synapse_mlir graph tests"
    pushd $__lit_test_folder;
    LIT_OPTS="$LIT_OPTS" $CMAKE_BUILD/bin/cmake --build . -j$__jobs --target check-synapse-mlir-graph;
    popd

   # return error code of the test
    return $?
}

models_tests_ci()
{(
    set -x
    env

    TEST_NAME="${1:-$(date +"%Y%m%d-%H%M%S")}"
    PUBLISH="${2:-"OFF"}"
    DB_NAME="${3:-"synapse"}"
    DB_BRANCH="${4:-"master"}"
    DEVICE="${5:-"gaudi"}"
    MODELS_FOLDER="${6:-""}"
    JOB_NAME="${7:-"ALL"}"

    echo "TEST_NAME: ${TEST_NAME}"
    echo "PUBLISH: ${PUBLISH}"
    echo "DB_NAME: ${DB_NAME}"
    echo "DB_BRANCH: ${DB_BRANCH}"
    echo "DEVICE: ${DEVICE}"
    echo "MODELS_FOLDER: ${MODELS_FOLDER}"
    echo "JOB_NAME: ${JOB_NAME}"

    if [[ ! -z "${MODELS_FOLDER}" ]]; then
        MODELS_FOLDER_ARG="--models-folder ${MODELS_FOLDER}/models"
    else
        MODELS_FOLDER_ARG=""
    fi

    if [ "${PUBLISH}" == "POST_COMMIT" ]; then
        echo "publish post commit results"
        PUBLISH_ARG="--publish-results"
    else
        PUBLISH_ARG=""
    fi

    if [[ "ALL" == "${JOB_NAME}" ]]; then
        JOB="--job post-submit"
    elif [[ "PROMOTION" == "${JOB_NAME}" ]]; then
        JOB="--job promotion"
    elif [ ! -z "${JOB_NAME}" ]; then
        JOB="--job ${JOB_NAME}"
    fi

    local __models_tests_ci_py_test_exe="$SYNAPSE_ROOT/tests/json_tests/models_tests/models_tests_ci.py"
    local __models_tests_perf_args="--test-type perf --test-name ${TEST_NAME} ${MODELS_FOLDER_ARG} ${JOB} --database ${DB_NAME} --branch ${DB_BRANCH} --chip-type ${DEVICE} ${PUBLISH_ARG}"

    ( ${__models_tests_ci_py_test_exe} $__models_tests_perf_args )
    execution_sts=$?

    return ${execution_sts}
)}

models_tests_eager_ci()
{(
    set -x
    env

    TEST_NAME="${1:-$(date +"%Y%m%d-%H%M%S")}"
    PUBLISH="${2:-"OFF"}"
    DB_NAME="${3:-"synapse"}"
    DB_BRANCH="${4:-"master"}"
    DEVICE="${5:-"gaudi2"}"
    MODELS_FOLDER="${6:-""}"
    JOB_NAME="${7:-"eager"}"

    echo "TEST_NAME: ${TEST_NAME}"
    echo "PUBLISH: ${PUBLISH}"
    echo "DB_NAME: ${DB_NAME}"
    echo "DB_BRANCH: ${DB_BRANCH}"
    echo "DEVICE: ${DEVICE}"
    echo "MODELS_FOLDER: ${MODELS_FOLDER}"
    echo "JOB_NAME: ${JOB_NAME}"

    if [[ ! -z "${MODELS_FOLDER}" ]]; then
        MODELS_FOLDER_ARG="--models-folder ${MODELS_FOLDER}/models"
    else
        MODELS_FOLDER_ARG=""
    fi

    if [ "${PUBLISH}" == "POST_COMMIT" ]; then
        echo "publish post commit results"
        PUBLISH_ARG="--publish-results"
    else
        PUBLISH_ARG=""
    fi

    if [[ "ALL" == "${JOB_NAME}" ]]; then
        JOB="--job post-submit"
    elif [[ "PROMOTION" == "${JOB_NAME}" ]]; then
        JOB="--job promotion"
    elif [ ! -z "${JOB_NAME}" ]; then
        JOB="--job ${JOB_NAME}"
    fi

    local __models_tests_ci_py_test_exe="$SYNAPSE_ROOT/tests/json_tests/models_tests/models_tests_ci.py"
    local __models_tests_perf_args="--test-type perf_eager --test-name ${TEST_NAME} ${MODELS_FOLDER_ARG} ${JOB} --database ${DB_NAME} --branch ${DB_BRANCH} --chip-type ${DEVICE} ${PUBLISH_ARG}"

    ( ${__models_tests_ci_py_test_exe} $__models_tests_perf_args )
    execution_sts=$?

    return ${execution_sts}
)}

models_tests_compile_ci()
{(
    set -x
    env

    TEST_NAME="${1:-$(date +"%Y%m%d-%H%M%S")}"
    DEVICE="${2:-"gaudi"}"
    MODELS_FOLDER="${3:-""}"

    echo "TEST_NAME: ${TEST_NAME}"
    echo "DEVICE: ${DEVICE}"
    echo "MODELS_FOLDER: ${MODELS_FOLDER}"

    if [[ ! -z "${MODELS_FOLDER}" ]]; then
        MODELS_FOLDER_ARG="--models-folder ${MODELS_FOLDER}/models"
    else
        MODELS_FOLDER_ARG=""
    fi

    local __models_tests_ci_py_test_exe="$SYNAPSE_ROOT/tests/json_tests/models_tests/models_tests_ci.py"
    local __models_tests_compilation_args="--test-type compilation --test-name ${TEST_NAME} ${MODELS_FOLDER_ARG} --chip-type ${DEVICE}"

    ( ${__models_tests_ci_py_test_exe} $__models_tests_compilation_args )
    execution_sts=$?

    return ${execution_sts}
)}

models_tests_accuracy_ci()
{(
    set -x
    env

    TEST_NAME=$1
    DEVICE=$2
    MODELS_FOLDER=$3

    echo "TEST_NAME: ${TEST_NAME}"
    echo "DEVICE: ${DEVICE}"
    echo "MODELS_FOLDER: ${MODELS_FOLDER}"

    if [[ ! -z "${MODELS_FOLDER}" ]]; then
        MODELS_FOLDER_ARG="--models-folder ${MODELS_FOLDER}/models"
    else
        MODELS_FOLDER_ARG=""
    fi

    local __models_tests_ci_py_test_exe="$SYNAPSE_ROOT/tests/json_tests/models_tests/models_tests_ci.py"
    local __models_tests_accuracy_args="--test-type accuracy --job accuracy --test-name ${TEST_NAME} ${MODELS_FOLDER_ARG} --chip-type ${DEVICE}"

    ( ${__models_tests_ci_py_test_exe} $__models_tests_accuracy_args )

    return ${execution_sts}
)}

run_synrec_tests()
{
    if [ -z "$SYNAPSE_ROOT" ]
    then
        echo "SYNAPSE_ROOT path is not defined"
        return 1
    fi

    local __pytest_cmd=pytest
    hash $__pytest_cmd 2>/dev/null || { echo >&2 "Running synrec tests requires $__pytest_cmd to be installed"; return 1; }

    local __synrec_tests_py_exe="$__pytest_cmd $SYNAPSE_ROOT/tests/record_playback_tests/test_record_playback.py"
    local __synrec_tests_args="$*"
    ${__synrec_tests_py_exe} $__synrec_tests_args

    return $?
}

models_tests_job()
{
    if [ -z "$SYNAPSE_ROOT" ]
    then
        echo "SYNAPSE_ROOT path is not defined"
        return 1
    fi

    local __models_tests_ci_exe="$SYNAPSE_ROOT/tests/json_tests/models_tests/models_tests_ci.py"
    local __models_tests_ci_args="$*"
    ${__models_tests_ci_exe} $__models_tests_ci_args

    return $?
}

consistency_tests_ci()
{
    DEVICE="${1:-""}"
    ITERS="${2:-"10"}"

    run_bert_cmd="run_from_json -r -c ${DEVICE} --consistency_check --test_iter ${ITERS} -m pt_bert_mlperf_2layers"
    ( ENABLE_EXPERIMENTAL_FLAGS=true ENABLE_ADD_CACHE_WARMUP=0 ALIGN_BPT_FCD_STRIDE_TO_CACHELINE_MODE=1 ${run_bert_cmd} ) || __ret=$?

    run_resnet_cmd="run_from_json -r -c ${DEVICE} --consistency_check --test_iter ${ITERS} -m syn_resnet50_full_fwd_bwd_bf16"
    ( ${run_resnet_cmd} ) || __ret=$((${__ret} | $?))

    # ( ENABLE_EXPERIMENTAL_FLAGS=true EW_RADIUS=4 ENABLE_HABANANORM_FOR_BN=1 ENABLE_EXPERIMENTAL_PATTERNS_FUSION=1 ${run_renet_cmd} ) || __ret=$((${__ret} | $?))

    return ${__ret}
}

declare SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source "${SCRIPT_DIR}/install_iwyu.sh"
source "${SCRIPT_DIR}/run_iwyu.sh"
