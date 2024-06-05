#!/bin/bash
#
# Copyright (C) 2016-2021 HabanaLabs, Ltd.
# All Rights Reserved.
#
# Unauthorized copying of this file, via any medium is strictly prohibited.
# Proprietary and confidential.
# Author: Oded Gabbay <ogabbay@habana.ai>
#

# shellcheck disable=SC2155

export __sudo="/usr/bin/sudo"
export NUMBER_OF_CORES=`grep -c ^processor /proc/cpuinfo`
export NUMBER_OF_JOBS=`expr $NUMBER_OF_CORES`
(return 0 2>/dev/null) && __sourced=1 || __sourced=0
if [ -f /etc/os-release ]; then
    export OS=$(awk -F= '$1=="NAME" { print tolower($2) ;}' /etc/os-release | sed 's/\"//g' | cut -f1 -d" ")
elif [ -f /etc/redhat-release ]; then
    export OS=$(awk '{print tolower($1)}' /etc/redhat-release)
else
    echo "Error: Unable to fetch the OS details!"
    [ $__sourced -eq 1 ] && return 1 || exit 1
fi
if [[ $OS == "red" ]]; then
    export OS="rhel"
fi
if [[ $OS == "debian" ]]; then
    export OS_VERSION=$(cat /etc/debian_version)
elif [[ $OS == "centos" || $OS == "rhel" || $OS == "almalinux" ]]; then
    export OS_VERSION=$(cat /etc/redhat-release | tr -dc '0-9.' | cut -d '.' -f1)
else
    export OS_VERSION=$(source /etc/os-release && echo -n $VERSION_ID)
fi

__get_python_ver()
{
    if [[ ($OS == 'ubuntu' && $OS_VERSION == '20.04') || \
        ( $OS == 'ubuntu' && $OS_VERSION == '22.04' && `command -v python3.8` ) || \
        ( $OS == 'debian' && $OS_VERSION == '10.10' && `command -v python3.8` ) || \
        ( $OS == 'debian' && $OS_VERSION == '11.6' && `command -v python3.8` ) || \
        ( $OS == 'tencentos' && $OS_VERSION == '3.1' ) || \
        ( $OS == 'rhel' && $OS_VERSION == '8' ) || \
        ( $OS == 'centos' && ($OS_VERSION == '8' || ($OS_VERSION == '7' && $(python3.8 -V) = *'3.8'* ) || $OS_VERSION == '6')) || \
        ( $OS == 'amazon' && $(python3 -V) = *'3.8'* ) || \
        ( $OS == 'ubuntu' && $OS_VERSION == '18.04' && $(python3 -V) = *'3.8'* ) || \
        ( $OS == 'almalinux' && $OS_VERSION == '8' ) \
    ]]; then
        __python_ver="3.8"
    elif [[ $OS == 'rhel' ]] && [[ $OS_VERSION == '9' ]]; then
        __python_ver="3.10"
    elif [[ $OS == 'ubuntu' ]] && [[ $OS_VERSION == '22.04' ]]; then
        __python_ver="3.10"
    elif [[ $OS == 'ubuntu' ]] && [[ $OS_VERSION == '18.04' ]] && [[ `python3 -V` = *'3.7'* ]]; then
        __python_ver="3.7"  # For Ubuntu 18.04 training might be updated python version 3.6 -> 3.7
    elif [[ ( $OS == 'amazon' ) || \
         ( $OS == 'rhel' && $OS_VERSION == '7' ) \
    ]]; then
        __python_ver="3.7"
    elif [[ $OS == 'debian' ]] && [[ $OS_VERSION == '11.2' ]] && [[ `command -v python3.6` ]]; then
        __python_ver="3.6"
    else
        __python_ver="3.6"
    fi

    export __python_cmd="python${__python_ver}"
    export __pip_cmd="pip${__python_ver}"
}

__get_python_ver

SCRIPT_VER="3.39"
BOOST_VER=1_63_0

if [ ! -d $HABANA_LOGS ]; then
    echo "Creating logs dir: $HABANA_LOGS"
    mkdir -p $HABANA_LOGS
fi

__get_func_name()
{
    if [ -n "$ZSH_NAME" ]; then
        echo ${funcstack[2]}
    else
        echo ${FUNCNAME[1]}
    fi
}

__yesno()
{
    __ansyn=""
    echo
    _bail_out_fn=_default_bail_out
    while true; do
        echo -n "$* (y/n)? "
        read x

        case $x in
            [yY]) __ansyn='y';break;;
            [nN]) __ansys='n';do_bail_out;break;;
        esac
    done
}

_yesno()
{
    case $CONFIRM in
        y) __yesno $*;;
        *) __ansyn='y';;
    esac
}

_fatal_msg()
{
    echo
    echo
    echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    echo "+ ERROR!!!: $* +"
    echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    echo
    echo
}

_verify_exists_command()
{
    local _cmd=$1
    command -v "$_cmd" >/dev/null
}

_verify_exists_dir()
{
    # note: if #1 is empty, this still works unless the message string is a dir(unlikely)
    local _msg=$1
    local _dirname=$2

    if [ -z "$_dirname" ];then
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

_reset_terminal()
{
    # Reset color and clean screen
    echo -e '\033[2J\033[u'
}

tgt_reboot()
{
    _yesno 'are you REALLY ready to reboot the target?'
    case $__ansyn in
    y)
        sudo reboot
        ;;
    *) ;;
    esac
}

#######################################
# Check that variable is set and not empty.
# Parameters:
#   $1: Variable name
# Returns:
#   0 if variable is set and not empty
#   1 othervise.
#######################################
function _verify_exists_variable() {
    local var_name=${1}
    local var_value="${!var_name// /}"  # Get the value of the variable with indirect expansion

    if [[ -z ${var_value} ]]; then
        _fatal_msg "Variable '$var_name' is not set or is empty"
        return 1
    fi
}

#######################################
# Fetch Git LFS project.
# Globals:
#   GERRIT_PROJECT
#   GERRIT_PATCHSET_REVISION
# Parameters:
#   $1: LFS project
#   $2: Target folder
# Returns:
#   0 if the fetch was successful.
#   1 if there was an error.
#######################################
function fetch_git_lfs() {
    local _lfs_project=${1// /}
    local _lfs_dir=${2}
    local _res=0

    echo "=== Fetch LFS project '${_lfs_project}' to folder '${_lfs_dir}'"

    _verify_exists_variable _lfs_project || return 1
    _verify_exists_variable _lfs_dir || return 1
    _verify_exists_variable GERRIT_PROJECT || return 1
    _verify_exists_dir "Target LFS ${_lfs_project} folder does not exist" ${_lfs_dir}

    # For LFS project checkout GERRIT_PATCHSET_REVISION if set (CI) and master_next if GERRIT_PATCHSET_REVISION is not set (Promotion)
    # For any project other than LFS one - take master branch
    local _lfs_branch=$([[ ${GERRIT_PROJECT^^} == ${_lfs_project^^} ]] && echo ${GERRIT_PATCHSET_REVISION-master_next} || echo 'master')
    echo "[INFO] Fetch LFS project '${_lfs_project}' branch '${_lfs_branch}' to folder '${_lfs_dir}' (GERRIT_PROJECT='${GERRIT_PROJECT}', GERRIT_PATCHSET_REVISION='${GERRIT_PATCHSET_REVISION-unset}')"

    pushd ${_lfs_dir}
    git fetch $(git remote) ${_lfs_branch}
    _res=$?
    if [[ ${_res} -eq 0 ]]; then
        git checkout ${_lfs_branch}
        _res=$?
    fi
    if [[ ${_res} -eq 0 ]]; then
        git lfs pull
        _res=$?
    fi
    if [[ ${_res} -ne 0 ]]; then
        _fatal_msg "${0}: Failed to fetch LFS project '${_lfs_project}' branch '${_lfs_branch}' to folder '${_lfs_dir}'"
    fi
    popd
    return ${_res}
}


if [ -f ${QUAL_ROOT}/.ci/scripts/qual.sh ];then
    source ${QUAL_ROOT}/.ci/scripts/qual.sh
fi

if [ -f ${HABANAQA_CORE_ROOT}/habanaqa_core_test_suit.sh ];then
    source ${HABANAQA_CORE_ROOT}/habanaqa_core_test_suit.sh
fi

if [ -f ${HABANA_QA_HCCL_ROOT}/habana_qa_functions.sh ];then
    source ${HABANA_QA_HCCL_ROOT}/habana_qa_functions.sh
fi

if [ -f ${HABANAQA_SYNAPSE_FEATURES_ROOT}/.ci/scripts/pytenet_functions.sh ];then
    source ${HABANAQA_SYNAPSE_FEATURES_ROOT}/.ci/scripts/pytenet_functions.sh
fi

if [ -f ${HABANAQA_SYNAPSE_FEATURES_ROOT}/.ci/scripts/test.sh ];then
    source ${HABANAQA_SYNAPSE_FEATURES_ROOT}/.ci/scripts/test.sh
fi

if [ -f ${HOROVOD_ROOT}/.ci/scripts/horovod-fork.sh ];then
    source ${HOROVOD_ROOT}/.ci/scripts/horovod-fork.sh
fi

if [ -f ${TF_TESTS_ROOT}/.ci/scripts/test.sh ];then
    source ${TF_TESTS_ROOT}/.ci/scripts/test.sh
fi

if [ -f ${SYNAPSE_ROOT}/.ci/scripts/synapse.sh ];then
    source ${SYNAPSE_ROOT}/.ci/scripts/synapse.sh
    source ${SYNAPSE_ROOT}/.ci/scripts/eager_monitor.sh
fi

if [ -f ${SYNREC_ROOT}/.ci/scripts/synrec.sh ];then
    source ${SYNREC_ROOT}/.ci/scripts/synrec.sh
fi

if [ -r ${HCL_ROOT}/.ci/scripts/hcl.sh ];then
    source ${HCL_ROOT}/.ci/scripts/hcl.sh
fi

if [ -f ${MEDIA_ROOT}/.ci/scripts/media.sh ];then
    source ${MEDIA_ROOT}/.ci/scripts/media.sh
fi

if [ -f ${AEON_ROOT}/.ci/scripts/build.sh ];then
    source ${AEON_ROOT}/.ci/scripts/build.sh
fi

if [ -f ${HLTHUNK_ROOT}/.ci/scripts/build.sh ];then
    source ${HLTHUNK_ROOT}/.ci/scripts/build.sh
fi

if [ -f ${HLTHUNK_ROOT}/.ci/scripts/test.sh ];then
    source ${HLTHUNK_ROOT}/.ci/scripts/test.sh
fi

if [ -f ${HABANALABS_ROOT}/.ci/scripts/build.sh ];then
    source ${HABANALABS_ROOT}/.ci/scripts/build.sh
fi

if [ -f ${HABANALABS_ROOT}/scripts/lkd_functions.sh ];then
    source ${HABANALABS_ROOT}/scripts/lkd_functions.sh
fi

if [ -f ${NIC_KMD_ROOT}/.ci/scripts/build.sh ];then
    source ${NIC_KMD_ROOT}/.ci/scripts/build.sh
fi

if [ -f ${RDMA_CORE_ROOT}/.ci/scripts/build.sh ];then
    source ${RDMA_CORE_ROOT}/.ci/scripts/build.sh
fi

if [ -f ${RDMA_CORE_ROOT}/.ci/scripts/test.sh ];then
    source ${RDMA_CORE_ROOT}/.ci/scripts/test.sh
fi

if [ -f ${TPCSIM_ROOT}/.ci/scripts/build.sh ];then
    source ${TPCSIM_ROOT}/.ci/scripts/build.sh
fi

if [ -f ${TPC_LLVM_ROOT}/../.ci/scripts/build.sh ];then
    source ${TPC_LLVM_ROOT}/../.ci/scripts/build.sh
fi

if [ -f ${TPC_LLVM_ROOT}/../.ci/scripts/test.sh ];then
    source ${TPC_LLVM_ROOT}/../.ci/scripts/test.sh
fi

if [ -f ${TPC_LLVM_ROOT}/../.ci/scripts/compiler_explorer.sh ];then
    source ${TPC_LLVM_ROOT}/../.ci/scripts/compiler_explorer.sh
fi

if [ -f ${TPC_FUSER_ROOT}/.ci/scripts/build.sh ];then
    source ${TPC_FUSER_ROOT}/.ci/scripts/build.sh
fi

if [ -f ${TPC_FUSER_ROOT}/.ci/scripts/test.sh ];then
    source ${TPC_FUSER_ROOT}/.ci/scripts/test.sh
fi

if [ -f ${TPC_SCALAR_ROOT}/.ci/scripts/build.sh ];then
    source ${TPC_SCALAR_ROOT}/.ci/scripts/build.sh
fi

if [ -f ${TPC_SCALAR_ROOT}/.ci/scripts/test.sh ];then
    source ${TPC_SCALAR_ROOT}/.ci/scripts/test.sh
fi

if [ -f ${SYNAPSE_UTILS_ROOT}/.ci/scripts/build.sh ];then
    source ${SYNAPSE_UTILS_ROOT}/.ci/scripts/build.sh
fi

if [ -f ${SYNAPSE_UTILS_ROOT}/.ci/scripts/test.sh ];then
    source ${SYNAPSE_UTILS_ROOT}/.ci/scripts/test.sh
fi

if [ -f ${FUNC_SIM5_ROOT}/.ci/scripts/func_sim.sh ];then
    source ${FUNC_SIM5_ROOT}/.ci/scripts/func_sim.sh
fi

if [ -f ${SWTOOLS_SDK_ROOT}/.ci/scripts/build.sh ];then
    source ${SWTOOLS_SDK_ROOT}/.ci/scripts/build.sh
fi

if [ -f ${SWTOOLS_SDK_ROOT}/.ci/scripts/test.sh ];then
    source ${SWTOOLS_SDK_ROOT}/.ci/scripts/test.sh
fi

if [ -f ${SYNAPSE_PROFILER_ROOT}/.ci/scripts/build.sh ];then
    source ${SYNAPSE_PROFILER_ROOT}/.ci/scripts/build.sh
fi

if [ -f ${SYNAPSE_PROFILER_ROOT}/.ci/scripts/test.sh ];then
    source ${SYNAPSE_PROFILER_ROOT}/.ci/scripts/test.sh
fi

if [ -f ${HABANA_REGS_CLI_ROOT}/.ci/scripts/build.sh ];then
    source ${HABANA_REGS_CLI_ROOT}/.ci/scripts/build.sh
fi

if [ -f ${HABANA_REGS_CLI_ROOT}/.ci/scripts/test.sh ];then
    source ${HABANA_REGS_CLI_ROOT}/.ci/scripts/test.sh
fi

if [ -f ${HL_TRACE_VIEWER_ROOT}/.ci/scripts/build.sh ];then
    source ${HL_TRACE_VIEWER_ROOT}/.ci/scripts/build.sh
fi

if [ -f ${CORAL_SIM_ROOT}/.ci/scripts/coral_sim.sh ];then
    source ${CORAL_SIM_ROOT}/.ci/scripts/coral_sim.sh
fi

if [ -f ${SPARTA_ROOT}/.ci/scripts/sparta_ci.sh ];then
    source ${SPARTA_ROOT}/.ci/scripts/sparta_ci.sh
fi

if [ -f ${MODEL_GARDEN_ROOT}/.ci/scripts/test.sh ];then
    source ${MODEL_GARDEN_ROOT}/.ci/scripts/test.sh
fi

if [ -f ${MLPERF_INFERENCE_ROOT}/code/functions.sh ]; then
    source ${MLPERF_INFERENCE_ROOT}/code/functions.sh
fi

# As per SW-95797, Moved .CI folder from Pytorch-fork root
# to pytorch-integration root (https://gerrit.habana-labs.com/#/c/235004/).
# Hence replaced root from pytorch-fork to pytorch-integration in below path.

if [ -f ${PYTORCH_MODULES_ROOT_PATH}/.ci/scripts/build.sh ];then
    source ${PYTORCH_MODULES_ROOT_PATH}/.ci/scripts/build.sh
fi

if [ -f ${PYTORCH_TESTS_ROOT}/.ci/scripts/test.sh ];then
    source ${PYTORCH_TESTS_ROOT}/.ci/scripts/test.sh
fi

if [ -f ${PYTORCH_TESTS_ROOT}/tests/torch_training_tests/utils/tox_scripts/pytorch_tox_tests.sh ];then
    source ${PYTORCH_TESTS_ROOT}/tests/torch_training_tests/utils/tox_scripts/pytorch_tox_tests.sh
fi

if [ -f ${KINETO_ROOT}/.ci/scripts/test.sh ]; then
    source ${KINETO_ROOT}/.ci/scripts/test.sh
fi

if [ -f ${HABANAQA_FRAMEWORK_FEATURES_TESTS_ROOT}/.ci/scripts/test.sh ];then
    source ${HABANAQA_FRAMEWORK_FEATURES_TESTS_ROOT}/.ci/scripts/test.sh
fi

if [ -f ${HABANAQA_MEDIA_ROOT}/.ci/scripts/test.sh ];then
    source ${HABANAQA_MEDIA_ROOT}/.ci/scripts/test.sh
fi

if [ -f ${HABANA_QUANTIZATION_TOOLKIT_ROOT}/.ci/scripts/build.sh ];then
    source ${HABANA_QUANTIZATION_TOOLKIT_ROOT}/.ci/scripts/build.sh
fi

if [ -f ${HABANA_QUANTIZATION_TOOLKIT_ROOT}/.ci/scripts/test.sh ];then
    source ${HABANA_QUANTIZATION_TOOLKIT_ROOT}/.ci/scripts/test.sh
fi

if [ -f ${TPC_KERNELS_ROOT}/.ci/scripts/build.sh ];then
    source ${TPC_KERNELS_ROOT}/.ci/scripts/build.sh
fi

if [ -f ${TPC_KERNELS_ROOT}/.ci/scripts/test.sh ];then
    source ${TPC_KERNELS_ROOT}/.ci/scripts/test.sh
fi

if [ -f ${COMPLEX_GUID_LIB_ROOT}/.ci/scripts/build.sh ];then
    source ${COMPLEX_GUID_LIB_ROOT}/.ci/scripts/build.sh
fi

if [ -f ${COMPLEX_GUID_LIB_ROOT}/.ci/scripts/test.sh ];then
    source ${COMPLEX_GUID_LIB_ROOT}/.ci/scripts/test.sh
fi

if [ -f ${TPC_SCALAR_KERNELS_ROOT}/.ci/scripts/build.sh ];then
    source ${TPC_SCALAR_KERNELS_ROOT}/.ci/scripts/build.sh
fi

if [ -f ${TPC_SCALAR_KERNELS_ROOT}/.ci/scripts/test.sh ];then
    source ${TPC_SCALAR_KERNELS_ROOT}/.ci/scripts/test.sh
fi

if [ -f ${DEEPSPEED_EXAMPLES_FORK_ROOT}/.ci/scripts/test.sh ];then
    source ${DEEPSPEED_EXAMPLES_FORK_ROOT}/.ci/scripts/test.sh
fi

if [ -f ${DEEPSPEED_FORK_ROOT}/.ci/scripts/build.sh ];then
    source ${DEEPSPEED_FORK_ROOT}/.ci/scripts/build.sh
fi

if [ -f ${DEEPSPEED_FORK_ROOT}/.ci/scripts/test.sh ];then
    source ${DEEPSPEED_FORK_ROOT}/.ci/scripts/test.sh
fi

if [ -f ${SIVAL_ROOT}/.ci/scripts/build_host.sh ];then
    source ${SIVAL_ROOT}/.ci/scripts/build_host.sh
fi

if [ -f ${SIVAL_ROOT}/.ci/scripts/test_host.sh ];then
    source ${SIVAL_ROOT}/.ci/scripts/test_host.sh
fi

if [ -f ${SIVAL_ROOT}/.ci/scripts/build_embedded.sh ];then
    source ${SIVAL_ROOT}/.ci/scripts/build_embedded.sh
fi

if [ -f ${SIVAL2_ROOT}/.ci/scripts/build_host2.sh ];then
    source ${SIVAL2_ROOT}/.ci/scripts/build_host2.sh
fi

if [ -f ${SIVAL2_ROOT}/.ci/scripts/test_host2.sh ];then
    source ${SIVAL2_ROOT}/.ci/scripts/test_host2.sh
fi

if [ -f ${SIVAL2_ROOT}/.ci/scripts/build_embedded.sh ];then
    source ${SIVAL2_ROOT}/.ci/scripts/build_embedded.sh
fi

if [ -f ${TF_MODULES_ROOT}/.ci/scripts/tensorflow-training.sh ];then
    source ${TF_MODULES_ROOT}/.ci/scripts/tensorflow-training.sh
fi

if [ -f ${TF_MLIR_ROOT}/.ci/scripts/build.sh ];then
    source ${TF_MLIR_ROOT}/.ci/scripts/build.sh
fi

if [ -f ${TF_MLIR_ROOT}/.ci/scripts/test.sh ];then
    source ${TF_MLIR_ROOT}/.ci/scripts/test.sh
fi

if [ -f ${JAX_ROOT}/.ci/scripts/build.sh ];then
    source ${JAX_ROOT}/.ci/scripts/build.sh
fi

if [ -f ${JAX_ROOT}/.ci/scripts/test.sh ];then
    source ${JAX_ROOT}/.ci/scripts/test.sh
fi

if [ -f ${SYNAPSE_GC_TPC_TESTS_ROOT}/.ci/scripts/test.sh ];then
    source ${SYNAPSE_GC_TPC_TESTS_ROOT}/.ci/scripts/test.sh
fi

if [ -f ${MULTINODE_TESTS_ROOT}/.ci/scripts/test.sh ];then
    source ${MULTINODE_TESTS_ROOT}/.ci/scripts/test.sh
fi

if [ -f ${HABANA_QA_SYNAPSE_ROOT}/habana_qa_functions.sh ];then
    source ${HABANA_QA_SYNAPSE_ROOT}/habana_qa_functions.sh
fi

if [ -f ${HABANA_QA_INFERENCE_ROOT}/habana_qa_functions.sh ];then
    source ${HABANA_QA_INFERENCE_ROOT}/habana_qa_functions.sh
fi

if [ -f ${GPU_MIGRATION_ROOT_PATH}/.ci/scripts/test.sh ];then
    source ${GPU_MIGRATION_ROOT_PATH}/.ci/scripts/test.sh
fi

if [ -f ${GPU_MIGRATION_ROOT_PATH}/.ci/scripts/build.sh ];then
    source ${GPU_MIGRATION_ROOT_PATH}/.ci/scripts/build.sh
fi

habana_help()
{
    echo -e "\n**** This script is part of HabanaLabs internal build system - Ver. $SCRIPT_VER ****"
    echo -e "\n- Add the following lines to the end of your ~/.bashrc file:"
    echo -e '\nsource $HOME/path/to/repo/software/projects/habana_scripts/habana_env'
    echo -e "\n- The following is a list of available functions in habana_functions.sh"
    echo -e "build_npu_stack             -    Build all the components of the NPU stack"
    echo -e "build_framework_integration_docs -    Build documentation for ML frameworks integration modules"
    echo -e "build_sbs_ci_test           -    Build side-by-side CI test"
    echo -e "run_auto_index_test         -    Run auto index space mapping test"
    echo -e "run_habana_py_qa_test       -    Run habana_py qa test (default = debug)"
    echo -e "run_resnet_sbs_ci_test      -    Run resnet SBS CI test"
    echo -e "manage_sandbox              -    Control sandbox operations"
    echo -e "habana_mxnet_clone          -    Clones mxnet into a new folder and setup the branches"
    echo -e "core_dump_config            -    Allow core dump save and set their default location to $CORE_DUMPS"
    echo -e "print_core_dump_bt          -    Print core dump backtrace"
    echo -e "run_habana_regs_cli         -    Run cli utility to read/write registers from userspace"
    echo -e "get_docker_command          -    Print docker run command use in CI process"
    echo -e ""

    # Help qual group functions
    qual_functions_help

    # Help embedded group functions
    embedded_functions_help

    # Help simulation group functions
    coral_sim_functions_help
    func_sim_functions_help
    tpc_sim_functions_help

    # Help drivers group functions
    lkd_build_help
    nic_kmd_build_help
    hlthunk_build_help
    hlthunk_tests_help
    lkd_functions_help

    # Help synapse group functions
    synapse_functions_help
    synapse_utils_functions_help

    # Help HCL group messages
    type hcl_functions_help &>/dev/null && hcl_functions_help

    # Help media group functions
    media_functions_help

    # Help tensorflow-training group functions
    tensorflow_training_functions_help

    # Help tensorflow-training-tests group functions
    tensorflow_training_tests_functions_help

    # Help PyTorch group functions
    pytorch_functions_help

    # Help sival group functions
    if [ -f ${SIVAL_ROOT}/.ci/scripts/test_host.sh ];then
        sival_host_functions_help
        sival_embedded_functions_help
        sival_tests_functions_help
    fi

    if [ -f ${SIVAL2_ROOT}/.ci/scripts/test_host2.sh ];then
        sival2_host_functions_help
        sival2_tests_functions_help
    fi


    # Help TPC group functions
    tpc_functions_help

    # Help tpc-llvm group functions
    tpc_llvm_build_help
    tpc_llvm_tests_help
    tpc_compiler_explorer_help

    # Help tpc-fuser group functions
    tpc_fuser_build_help
    tpc_fuser_tests_help

    #Help tpc-kernels group functions
    tpc_kernels_build_help
    tpc_kernels_test_help

    # Help complex-guid group functions
    complex_guid_build_help
    complex_guid_test_help

    # Help SWTools group functions
    swtools_build_help
    swtools_test_help

    # Help habana_regs_cli group functions
    regs_cli_build_help
    regs_cli_test_help

    # Help trace_viewer group functions
    trace_viewer_build_help

    # Help horovod-fork group functions
    horovod_fork_functions_help

    # Help pytenet group functions
    pytenet_functions_help

    habana_common_help

    # Help SW Tools Sdk group functions
    swtools_sdk_build_help
    swtools_sdk_tests_help

    # Help synapse utils- shared layer group functions
    synapse_utils_sl_build_help
    synapse_utils_sl_test_help

    echo -e ""
    echo -e "Number of jobs = $NUMBER_OF_JOBS"

    return 0
}

usage()
{
    if [ $1 == "build_framework_integration_docs"  ]; then
        echo -e "\nusage: $1 [options]\n"

        echo -e "options:\n"
        echo -e "       --no-color             Disable colors in output"
        echo -e "  -c,  --configure            Configure before build"
        echo -e "  -C,  --clean                Clean the build directory"
        echo -e "  -v,  --verbose              Build with verbose"
        echo -e "  -j,  --jobs <val>           Overwrite number of jobs"
        echo -e "  -h,  --help                 Print this help"
    fi

    if [ $1 == "build_npu_stack" ]; then
        echo -e "\nusage: $1 [options]\n"

        echo -e "options:\n"
        echo -e "  -c,  --configure            Configure before build"
        echo -e "  -a,  --build-all            Build both debug and release build"
        echo -e "  -f,  --force-clean          wipe build directory, clean the code, init and sync repo"
        echo -e "  -r,  --release              Build only release build"
        echo -e "  -l,  --lib-only             Build only lib files (without tests for applicable components)"
        echo -e "  -j,  --jobs <val>           Overwrite number of jobs"
        echo -e "  -d,  --docker_mode          Skip build LKD in docker"
        echo -e "  -x,  --override_config      Use json file to override modules config"
        echo -e "  -h,  --help                 Prints this help"
    fi

    if [ $1 == "run_auto_index_test" ]; then
        echo -e "\nusage: $1 [options]\n"
        echo -e "options:\n"
        echo -e "  -f,  TEST_FILTER                    Test filter, default all: Run all tests"
        echo -e "  -c,  TEST_CATEGORY                  'ic' - auto index space check, 'io' - auto index space overload, default: ic"
        echo -e "  -t,  NUM                            Number of tests execution per process, default: 1"
        echo -e "  -x,  PATH                           Output XML file to PATH - available in ST mode only"
    fi

    if [ $1 == "run_habana_py_qa_test" ]; then
        echo -e "\nusage: $1 [options]\n"
        echo -e "options:\n"
        echo -e "  -l,  --list-tests                   List the available tests"
        echo -e "  -s,  --specific-test TEST           Run TEST"
        echo -e "  -m,  --maxfail NUM                  Stop after NUM failures"
        echo -e "  -p,  --pdb                          Run the app under pdb (python GDB)"
        echo -e "       --no-color                     Disable colors in output"
        echo -e "  -h,  --help                         Prints this help"
        echo -e "  -x,  --xml PATH                     Output XML file to PATH - available in ST mode only"
    fi

    if [ $1 == "manage_sandbox" ]; then
        echo -e "\nusage: $1 [options]\n"
        echo -e "options:\n"
        echo -e "  -s,  --sandbox SANDBOX      Name of Sandbox to use"
        echo -e "  -o,  --op OPERATION         Operation to perform. Supported operations are:"
        echo -e "       create                 Create new sandbox and clone all needed source files"
        echo -e "       delete                 Delete sandbox source and binary files"
        echo -e "       list                   Show all available sandboxes"
        echo -e "       set                    Set environment to compile files for the specified sandbox"
        echo -e "  -r,  --root_dir             Root directory to hold source and binary directories (default: \$HOME)"
        echo -e "  -c,  --commit_tag TAG       Commit tag to use in repo; can be branch, ID, tag (default: 'master')"
        echo -e "                              This tag must be available in all repositories in the manifest file"
        echo -e "  -u,  --url                  URL of the manifest (default: 'ssh://gerrit:29418/software-repo')"
        echo -e "  -m,  --manifest             Manifest file (default: 'default.xml')"
        echo -e "  -p,  --local-path <path>    Use local path as a sandbox source instead of cloning from URL"
        echo -e "  -j,  --jobs <val>           Overwrite number of jobs"
        echo -e "  -h,  --help                 Prints this help"
    fi

    if [ $1 == "build_sbs_ci_test" ]; then
        echo -e "\nusage: $1 [options]\n"

        echo -e "options:\n"
        echo -e "  -a,  --build-all            Build both debug and release build"
        echo -e "       --no-color             Disable colors in output"
        echo -e "  -c,  --configure            Configure before build"
        echo -e "  -r,  --release              Build only release build"
        echo -e "  -h,  --help                 Prints this help"
    fi

    if [ $1 == "print_core_dump_bt" ]; then
        echo -e "\nusage: $1 [options]\n"
        echo -e "options:\n"
        echo -e " -d, --core_dump              Path to core dump file"
        echo -e " -h,  --help                  Prints this help"
    fi

    if [ $1 == "install_requirements_habana_py" ]; then
        echo -e "\nPass one or more pip requirements.txt files, such as (the order doesn't matter):"
        echo -e "requirements-gc.txt, requirements-test.txt, requirements-demo.txt, requirements-tools.txt"
        echo -e "These files will be used by pip install.\n"
        echo -e "If no argument is passed, the default is to install the requirements.txt file.\n"
        echo -e "usage:"
        echo -e "\tinstall_requirements_habana_py -h, --help    Prints this help"
        echo -e "\tinstall_requirements_habana_py [requirements-gc.txt [requirements-test.txt [requirements-demo.txt [requirements-tools.txt]]]]\n"
    fi

    if [ $1 == "install_requirements" ]; then
        echo -e "\nPass one or more pip requirements.txt files, such as (the order doesn't matter):"
        echo -e "requirements-synapse.txt, requirements-test.txt, requirements-demo.txt, requirements-tools.txt"
        echo -e "These files will be used by pip install.\n"
        echo -e "If no argument is passed, the default is to install the requirements.txt file.\n"
        echo -e "usage:"
        echo -e "\tinstall_requirements -h, --help    Prints this help"
        echo -e "\tinstall_requirements [requirements-synapse.txt [requirements-test.txt [requirements-demo.txt [requirements-tools.txt]]]]\n"
    fi

    if [ $1 == "get_docker_command" ]; then
        echo -e "\nusage: $1 [options]\n";
        echo -e "options:\n";
        echo -e "  -b,  --build-dir            Build root directory (default: \$BUILD_ROOT)";
        echo -e "  -c,  --build-cache          Use DOCKER_BUILD_CACHE_VOLUME as build cache";
        echo -e "  -d,  --workdir              Docker container workdir";
        echo -e "  -e,  --extra-params         Additional docker run parameters";
        echo -e "  -i,  --image-name           Name of docker image to use (default: ";
        echo -e "                              artifactory-kfs.habana-labs.com/devops-docker-local/habana-builder:ubuntu20.04)";
        echo -e "  --init-run                  Allows to execute commands as root inside container. Container won't be removed automatically.";
        echo -e "  -m,  --weka-mount           Add weka mount point to docker container. [example: -m '/mnt/weka/data:/mnt/weka/data:ro']";
        echo -e "  -n,  --container-name       Custom container name. (default: randomly generated name)";
        echo -e "  -p,  --ipc                  Set ipc configuration";
        echo -e "  -t,  --tmpfs                Set tmpfs configuration";
        echo -e "  -u,  --skip-usr-mounts      Skips mounting of /etc/passwd, shadow and group. Needed to run docker on local VM";
        echo -e "  -w,  --workspace            Path to workspace (default: \$HOME)";
        echo -e "  -s,  --source-dir           Path to source dir (default: \$HABANA_NPU_STACK_PATH)";
        echo -e "  -x,  --execute-command      Command to execute inside docker container";
        echo -e "  -v,  --volume               mount points list declared as string [example: '<local_path1>:<docker_path1> <local_path2>:<docker_path2>']";
    fi

    # Help for qual group related functions
    _verify_exists_command qual_usage && qual_usage $1

    # Help for embedded group related functions
    _verify_exists_command embedded_usage && embedded_usage $1

    # Help for simulation group related functions
    _verify_exists_command coral_sim_usage && coral_sim_usage $1
    _verify_exists_command func_sim_usage && func_sim_usage $1
    _verify_exists_command tpc_sim_usage && tpc_sim_usage $1

    # Help for synapse group related functions
    _verify_exists_command synapse_usage && synapse_usage $1
    _verify_exists_command synapse_utils_usage && synapse_utils_usage $1

    # Help for HCL group related functions
    type hcl_usage &>/dev/null && hcl_usage $1

    # Help for tensorflow-training group related functions
    _verify_exists_command tensorflow_training_usage && tensorflow_training_usage $1

    # Help for pytorch related  functions
    _verify_exists_command pytorch_usage && pytorch_usage $1

    # Help for sival group related functions
    if [ -f ${SIVAL_ROOT}/.ci/scripts/build_host.sh ];then
        sival_host_build_usage $1
        sival_embedded_build_usage $1
        sival_tests_usage $1
    fi

    if [ -f ${SIVAL2_ROOT}/.ci/scripts/build_host2.sh ];then
        sival2_host_build_usage $1
        sival2_tests_usage $1
    fi

    # Help for tpc-llvm related  functions
    _verify_exists_command tpc_llvm_build_usage && tpc_llvm_build_usage $1
    _verify_exists_command tpc_llvm_test_usage && tpc_llvm_test_usage $1
    _verify_exists_command tpc_compiler_explorer_usage && tpc_compiler_explorer_usage $1

    # Help for tpc-fuser related  functions
    _verify_exists_command tpc_fuser_build_usage && tpc_fuser_build_usage $1
    _verify_exists_command tpc_fuser_test_usage && tpc_fuser_test_usage $1

    # Help for tpc-kernels related functions
    _verify_exists_command tpc_kernels_build_usage && tpc_kernels_build_usage $1
    _verify_exists_command tpc_kernels_test_usage && tpc_kernels_test_usage $1

    # Help for complex-guid related functions
    _verify_exists_command complex_guid_build_usage && complex_guid_build_usage $1
    _verify_exists_command complex_guid_test_usage && complex_guid_test_usage $1

    # Help for horovod-fork group related functions
    _verify_exists_command horovod_fork_usage && horovod_fork_usage $1

    # Help for pytenet related  functions
    _verify_exists_command pytenet_usage && pytenet_usage $1

    # Help for scalar build related  functions
    _verify_exists_command build_scalar_usage && build_scalar_usage $1

    # Help for scalar test related  functions
    _verify_exists_command test_scalar_usage && test_scalar_usage $1

    # Help for synapse utils- shared layer group related functions
    _verify_exists_command synapse_utils_sl_build_usage && synapse_utils_sl_build_usage $1
    _verify_exists_command synapse_utils_sl_test_usage && synapse_utils_sl_test_usage $1

    # Help for tpc scalar kernels related functions
    _verify_exists_command tpc_scalar_kernels_build_usage && tpc_scalar_kernels_build_usage $1
    _verify_exists_command tpc_scalar_kernels_test_usage && tpc_scalar_kernels_test_usage $1

    echo -e ""
    _verify_exists_command habana_common_usage && habana_common_usage $1
    return 0
}

__abs_to_rel_symlink()
{
    local link=$1
    local link_target=$(realpath "$link")
    local link_parent_dir=$(realpath "$(dirname "$link")")
    local relative_target_path=$(realpath --relative-to="$link_parent_dir" "$link_target")

    ln -snf "$relative_target_path" "$link" || { echo "failed creating symlink $link->$relative_target_path"; return 1; }
    echo "changed $link target from $link_target to $relative_target_path"
    return 0
}

__create_relative_build_symlinks()
{
    local src=$(realpath "$1")
    local dst_build_root=$(realpath "$2")
    local dst_build_latest=$(realpath "$3")

    # for each file in src dir:
    #     1. construct paths in which the link will be created
    #     2. construct relative paths for the link targets
    #     3. create symlinks
    for target in "$src"/*; do
        local file_name=$(basename "$target")
        local build_root_link_path="$dst_build_root/$file_name"
        local repo_build_link_path="$dst_build_latest/$file_name"
        local relative_build_root_target_path=$(realpath --relative-to="$dst_build_root" "$target")
        local relative_repo_build_target_path=$(realpath --relative-to="$dst_build_latest" "$target")

        ln -snf "$relative_build_root_target_path" "$build_root_link_path" ||
        { echo "failed creating build root symlink <$build_root_link_path->$relative_build_root_target_path>"; return 1; }

        ln -snf "$relative_repo_build_target_path" "$repo_build_link_path" ||
        { echo "failed creating repo build symlink <$repo_build_link_path->$relative_repo_build_target_path>"; return 2; }
    done
    return 0
}

__find_and_convert_abs_build_symlinks()
{
    export -f __abs_to_rel_symlink
    find "$BUILD_ROOT" -type l -lname '/*' -exec bash -c '__abs_to_rel_symlink "$0"' {} \;
}

_copy_build_products()
{
    local __dir=$BUILD_ROOT_DEBUG
    local __path=$1
    mkdir -p "$BUILD_ROOT_LATEST"


    if  [ -n "$1" ]; then
        if [[ "$2" == "-r" ]]; then
            __dir=$BUILD_ROOT_RELEASE
        fi
        mkdir -p $__dir
    fi

    #Check Ubuntu version
    local _Var=$(lsb_release -r)
    local _WhichUbuntu=$(cut -f2 <<< "$_Var")

    #echo 'Ubunto version is ' $_WhichUbuntu
    _verify_exists_dir "$BUILD_ROOT" "$BUILD_ROOT"

    if  [ -n "$1" ]; then
        local src=$__path/lib/
        local dst_debug_or_release_path=$__dir
        __create_relative_build_symlinks "$src" "$dst_debug_or_release_path" "$BUILD_ROOT_LATEST"
    fi

    if [ "$_WhichUbuntu" == "20.04" ] && [ "$BUILD_ROOT" = /home_local/"$USER"/builds ]
    then
        # convert absolute symlinks on home_local to be relative
        __find_and_convert_abs_build_symlinks

        # Makefile, version.cpp (rsynched), and compile_commands.json below (created on nfs build),
        # enable clangd indexing on machines without access to home_local.

        rsync --archive --verbose --human-readable --prune-empty-dirs --include '*/'\
                                                                      --include '*.so*'\
                                                                      --include '*.ko'\
                                                                      --include '*_tests'\
                                                                      --include '/**tpc_mlir**/bin/**'\
                                                                      --include '/**tpc_llvm**/bin/**'\
                                                                      --include '/**synapse_profiler**/bin/**'\
                                                                      --include 'hl-prof-config'\
                                                                      --include 'synprof_configuration'\
                                                                      --include 'synprof_parser'\
                                                                      --include 'shim_ctl'\
                                                                      --include 'Makefile'\
                                                                      --include 'version.cpp'\
                                                                      --exclude '*'\
                                                                      "/home_local/$USER/builds/" "/home/$USER/builds"

        local __src_compile_cmds_json="$__path"/compile_commands.json
        if [ -f "$__src_compile_cmds_json" ]; then
            local __dst_compile_cmds_json=${__src_compile_cmds_json/home_local/home}
            sed 's#/home_local/#/home/#g' "$(realpath "$__src_compile_cmds_json")" > "$__dst_compile_cmds_json"
        fi
    fi
}

core_dump_config()
{
    echo "Setting unlimited core dump size"
    ulimit -c "${CORE_DUMP_SIZE}"
    echo "Creating core dumps dir: $CORE_DUMPS"
    mkdir -p $CORE_DUMPS
    echo "Setting core dumps location $CORE_DUMPS"
    sudo sysctl -w kernel.core_pattern=$CORE_DUMPS/$CORE_DUMP_PREFIX.%E.%p.%h.%t
}

print_core_dump_bt()
{
    local __scriptname=$(__get_func_name)
    local __core_dump=''
    local __binary=''

    # parameter while-loop
    while [ -n "$1" ];
    do
        case $1 in
        -d  | --core_dump )
            shift
            __core_dump=$1
            ;;
        -h  | --help )
            usage ${__scriptname}
            return 0
            ;;
        *)
            echo "The parameter $1 is not allowed"
            usage ${__scriptname}
            return 1 # error
            ;;
        esac
        shift
    done
    if [ -z "${__core_dump}" ]; then
        echo "Please provide core dump file!"
        usage ${__scriptname}
        return 1
    fi
    __binary=$(echo "${__core_dump}" | cut -d '.' -f 2 | sed 's/\!/\//g')
    gdb ${__binary} "${__core_dump}" -ex bt -ex q
}

# returns 0 if running in a Python virtual environment, 1 otherwise
__running_in_venv()
{
    $__python_cmd - <<EOF
import sys
native = (hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix))
exit(not native)
EOF
}

install_numpy()
{
    install_numpy_cmd=($__pip_cmd install numpy==1.23.5)
    if ! __running_in_venv; then
        install_numpy_cmd+=(--user)
    fi
    "${install_numpy_cmd[@]}"
}

install_pyparsing()
{
    install_pyparsing_cmd=($__pip_cmd install pyparsing==2.4.7)
    if ! __running_in_venv; then
        install_pyparsing_cmd+=(--user)
    fi
    "${install_pyparsing_cmd[@]}"
}

install_cython()
{
    local __install_cython_cmd=""
    local __cython_version=$1
    if [ -z ${__cython_version} ]; then
        __install_cython_cmd="${__pip_cmd} install cython"
    else
        __install_cython_cmd="${__pip_cmd} install cython==${__cython_version}"
    fi

    if ! __running_in_venv; then
        __install_cython_cmd="${__install_cython_cmd} --user"
    fi
    (set -x;${__install_cython_cmd})
}

# Pass one or more pip requirements.txt files, such as:
# requirements-synapse.txt, requirements-test.txt, requirements-demo.txt, requirements-tools.txt
# If no argument is passed, the default is to install the requirements.txt file.
install_requirements_habana_py()
{
    local __scriptname=$(__get_func_name)
    local files=()

    while [ -n "$1" ]; do
        local param=$1
        case $param in
            -h | --help)
                usage $__scriptname
                return 1
                ;;
            *)

            # This method will be called after files are at their new location. Hence:
            # except requirements-tools.txt which is not part of habana_py, all others "are there"
            if [ -f $HABANA_PY_ROOT/.ci/requirements/$param ]; then
                files+=($HABANA_PY_ROOT/.ci/requirements/$param)
            elif [ -f $AUTOMATION_ROOT/ci/$param ]; then
                files+=($AUTOMATION_ROOT/ci/$param)
            else
                echo "ERROR: unknown parameter \"$param\""
                usage $__scriptname
                return 1
            fi
            ;;
        esac
        shift
    done

    if [ -z "$files" ]; then
        files="$HABANA_PY_ROOT/.ci/requirements/requirements.txt"
    fi

    $__pip_cmd uninstall -y hb-tensorflow tensorflow tensorflow-cpu tensorflow-gpu tensorboard tf-estimator-nightly
    $__sudo -H $__pip_cmd uninstall -y hb-tensorflow tensorflow tensorflow-cpu tensorflow-gpu tensorboard tf-estimator-nightly
    install_numpy
    install_pyparsing
    for requirement_file in "${files[@]}"; do
        echo "Using ${requirement_file}"
        cmd=($__pip_cmd install -r ${requirement_file})
        if ! __running_in_venv; then
            cmd+=(--user)
        fi
        "${cmd[@]}"
        if [ $? -ne 0 ]; then
            echo "Error, failed to install ${requirement_file}"
            return 1
        fi
    done
}

#### OLD function => To be removed on next commit ###
#
# Pass one or more pip requirements.txt files, such as:
# requirements-synapse.txt, requirements-test.txt, requirements-demo.txt, requirements-tools.txt
# If no argument is passed, the default is to install the requirements.txt file.
install_requirements()
{
    local __scriptname=$(__get_func_name)
    local files=()

    while [ -n "$1" ]; do
        local param=$1
        case $param in
            -h | --help)
                usage $__scriptname
                return 1
                ;;
            *)

            if [ -f $AUTOMATION_ROOT/ci/$param ]; then
                files+=($AUTOMATION_ROOT/ci/$param)
            elif [ -f $SYNAPSE_ROOT/.ci/requirements/$param ]; then
                files+=($SYNAPSE_ROOT/.ci/requirements/$param)
            else
                echo "ERROR: unknown parameter \"$param\""
                usage $__scriptname
                return 1
            fi
            ;;
        esac
        shift
    done

    if [ -z "$files" ]; then
        files="$AUTOMATION_ROOT/ci/requirements.txt"
    fi

    $__pip_cmd uninstall -y hb-tensorflow tensorflow tensorflow-cpu tensorflow-gpu tensorboard tf-estimator-nightly
    $__sudo -H $__pip_cmd uninstall -y hb-tensorflow tensorflow tensorflow-cpu tensorflow-gpu tensorboard tf-estimator-nightly
    install_numpy
    install_pyparsing
    for requirement_file in "${files[@]}"; do
        echo "Using ${requirement_file}"
        cmd=($__pip_cmd install -r ${requirement_file})
        if ! __running_in_venv; then
            cmd+=(--user)
        fi
        "${cmd[@]}"
        if [ $? -ne 0 ]; then
            echo "Error, failed to install ${requirement_file}"
            return 1
        fi
    done
}

install_requirements_inference() {
    cmd=($__pip_cmd install -r $AUTOMATION_ROOT/ci/requirements-inference.txt)
    if ! __running_in_venv; then
        cmd+=(--user)
    fi
    "${cmd[@]}"
}

install_requirements_training()
{
    $__pip_cmd uninstall -y hb-tensorflow tensorflow tensorflow-cpu tensorflow-gpu tensorboard tf-estimator-nightly keras keras-nightly
    $__sudo -H $__pip_cmd uninstall -y hb-tensorflow tensorflow tensorflow-cpu tensorflow-gpu tensorboard tf-estimator-nightly keras keras-nightly
    install_numpy
    cmd=($__pip_cmd install -r $AUTOMATION_ROOT/ci/requirements-training.txt)
    if ! __running_in_venv; then
        cmd+=(--user)
    fi
    "${cmd[@]}"
}

install_requirements_llvm()
{
    cmd=($__pip_cmd install -r $AUTOMATION_ROOT/ci/requirements-llvm.txt)
    if ! __running_in_venv; then
        cmd+=(--user)
    fi
    "${cmd[@]}"
}

install_requirements_tpc_kernel_unroll_pipeline()
{
    cmd=($__pip_cmd install -r $AUTOMATION_ROOT/ci/requirements_tpc_kernels_unroll_pipeline.txt)
    if ! __running_in_venv; then
        cmd+=(--user)
    fi
    "${cmd[@]}"
}

install_requirements_elastic()
{
    cmd=($__python_cmd -m pip install -r $AUTOMATION_ROOT/ci/requirements-elastic.txt --user -q --disable-pip-version-check)
    "${cmd[@]}"
}

__check_tf_dev_py_deps()
{
    install_requirements_training
}

__check_llvm_dev_py_deps()
{
    install_requirements_llvm
}

__check_openmpi()
{
    echo "warning: __check_openmpi command is deprecated. Please use install_openmpi instead."
    install_openmpi
}

verify_openmpi()
{
    if [ -z "$MPI_ROOT" ]; then
        export MPI_ROOT=/usr/local/share/openmpi
        echo "warning: MPI_ROOT not set; setting to: $MPI_ROOT"
    fi

    local _which_mpirun=`which mpirun`

    if [ -z "$_which_mpirun" ]; then
        echo "warning: Open MPI is not installed on your system."
        echo "         Please run: install_openmpi"
        return -1
    fi

    local _mpirun_path="${MPI_ROOT}/bin/mpirun"

    if [ -n "$_which_mpirun" ]; then
        if [[ "$_which_mpirun" != ${MPI_ROOT}/bin/* ]]; then
            echo "warning: mpirun installed to a non-standard path: $_which_mpirun"
            echo "         Where: MPI_ROOT = $MPI_ROOT"
            echo "         Please run: install_openmpi"
            return -1
        fi
    fi

    local _mpi_version=`$_which_mpirun --version | grep 'mpirun (Open MPI)' | grep -oE '[^ ]+$'`

    if [[ "$_mpi_version" != "$SUPPORTED_MPI_VERSION" ]]; then
        echo "warning: An existing installation of Open MPI found at $MPI_ROOT"
        echo "         is not in currently supported version."
        echo "         Supported version is: $SUPPORTED_MPI_VERSION"
        echo "         Installed version is: $_mpi_version"
        echo "         Please run: install_openmpi"
        return -1
    fi
}

install_openmpi()
{
    if [ -z "$MPI_ROOT" ]; then
        export MPI_ROOT=/usr/local/share/openmpi
        echo "warning: MPI_ROOT not set; setting to: $MPI_ROOT"
    fi

    local _mpirun_path="${MPI_ROOT}/bin/mpirun"
    local _which_mpirun=`which mpirun`
    local _remove_mpi_root='false'

    if [ -n "$_which_mpirun" ]; then
        if [[ "$_which_mpirun" != ${MPI_ROOT}/bin/* ]]; then
            echo "warning: mpirun installed to a non-standard path: $_which_mpirun"
            echo "         Where: MPI_ROOT = $MPI_ROOT"
            echo "Removing any existing apt-based installations of Open MPI."
            sudo apt-get purge -y -q --auto-remove openmpi-bin libopenmpi-dev
        fi
    else
        echo "warning: mpirun not found using command: which mpirun"
        local _remove_mpi_root='true'
    fi

    if [ -f "$_mpirun_path" ]; then
        local _mpi_version=`$_which_mpirun --version | grep 'mpirun (Open MPI)' | grep -oE '[^ ]+$'`

        if [[ "$_mpi_version" != "$SUPPORTED_MPI_VERSION" ]]; then
            echo "warning: An existing installation of Open MPI found at $MPI_ROOT"
            echo "         is not in currently supported version."
            echo "         Supported version is: $SUPPORTED_MPI_VERSION"
            echo "         Installed version is: $_mpi_version"
            local _remove_mpi_root='true'
        fi
    else
        echo "warning: mpirun not found at: $_mpirun_path"
        local _remove_mpi_root='true'
    fi

    if [ "$_remove_mpi_root" == 'true' ]; then
        echo "Removing the Open MPI from: $MPI_ROOT"
        sudo rm -rf "$MPI_ROOT"
    fi

    if [ ! -d "$MPI_ROOT" ]; then
        echo "Installing Open MPI $SUPPORTED_MPI_VERSION to: $MPI_ROOT"
        local VER=$SUPPORTED_MPI_VERSION
        wget -O /tmp/openmpi-$VER.tar.gz https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-$VER.tar.gz && \
            tar zxf /tmp/openmpi-$VER.tar.gz -C /tmp && \
            pushd /tmp/openmpi-$VER && \
            ./configure --prefix="$MPI_ROOT" && \
            make -j `nproc` && \
            sudo make install && \
            sudo mv /tmp/openmpi-$VER/LICENSE "$MPI_ROOT" &&  \
            rm -rf /tmp/openmpi-$VER* && \
            popd && \
            sudo /sbin/ldconfig
    fi

    if [[ "$PATH" != *"${MPI_ROOT}/bin"* ]]; then
        echo "Adding ${MPI_ROOT}/bin to PATH."
        export PATH="${MPI_ROOT}/bin:$PATH"
    fi
}

__check_mandatory_pkgs()
{
    if [[ $OS == "centos" || $OS == "rhel" || $OS == "amazon" || $OS == "tencentos" ]]; then
        echo "installing mandatory packages is not supported in rpm"
    elif [ -n "$DOCKER_CI_ENV" ]; then
        echo "installing mandatory packages is done in Docker pre made images"
    elif [ -n "$CI_ENV" ]; then
        echo "installing mandatory packages is done in Puppet CI images"
    else
        local _package
        local _required_pacakge_list=""
        local _failed_pacakge_list=""

        _required_pacakge_list+="g++ "
        _required_pacakge_list+="build-essential "
        _required_pacakge_list+="libelf-dev "
        _required_pacakge_list+="cmake "
        _required_pacakge_list+="libgmp3-dev "
        _required_pacakge_list+="autotools-dev "
        _required_pacakge_list+="automake "
        _required_pacakge_list+="texinfo "
        if [[ $OS == 'ubuntu' && ($OS_VERSION == '20.04' || $OS_VERSION == '22.04') ]]; then
            _required_pacakge_list+="python3-dev "
        else
            _required_pacakge_list+="python-dev "
        fi
        _required_pacakge_list+="libxml2-dev "
        _required_pacakge_list+="libxslt1-dev "
        _required_pacakge_list+="libprotobuf-dev "
        _required_pacakge_list+="protobuf-compiler "
        _required_pacakge_list+="ccache "
        _required_pacakge_list+="libboost-dev "
        _required_pacakge_list+="libboost-program-options-dev "
        _required_pacakge_list+="libboost-filesystem-dev "
        _required_pacakge_list+="libboost-regex-dev "
        _required_pacakge_list+="libboost-random-dev "
        _required_pacakge_list+="uuid-dev "
        if ! [[ $OS == 'ubuntu' && $OS_VERSION == '22.04' ]]; then
            _required_pacakge_list+="clang-6.0 "
        fi
        _required_pacakge_list+="libcmocka-dev "
        _required_pacakge_list+="pkg-config "
        _required_pacakge_list+="zip "
        _required_pacakge_list+="zlib1g-dev "
        _required_pacakge_list+="unzip "
        _required_pacakge_list+="bc "
        _required_pacakge_list+="bison "
        _required_pacakge_list+="flex "
        _required_pacakge_list+="sshpass "
        _required_pacakge_list+="libsox-dev "
        _required_pacakge_list+="libopencv-core-dev "
        _required_pacakge_list+="libopencv-imgproc-dev "
        _required_pacakge_list+="libopencv-highgui-dev "
        _required_pacakge_list+="libcurl4-gnutls-dev "
        _required_pacakge_list+="libzmq3-dev "
        _required_pacakge_list+="gdb "
        _required_pacakge_list+="python3-pip "
        _required_pacakge_list+="libyaml-cpp-dev "
        _required_pacakge_list+="valgrind "
        _required_pacakge_list+="device-tree-compiler "
        _required_pacakge_list+="openssl "
        _required_pacakge_list+="xxd "
        if ! [[ $OS == 'ubuntu' && $OS_VERSION == '22.04' ]]; then
            _required_pacakge_list+="clang-10 "
        fi
        _required_pacakge_list+="yasm "
        _required_pacakge_list+="maven "
        _required_pacakge_list+="libssl-dev "
        _required_pacakge_list+="ninja-build "
        _required_pacakge_list+="libgl1-mesa-dev "
        _required_pacakge_list+="python3-venv "
        _required_pacakge_list+="nasm "
        _required_pacakge_list+="libncurses5 "
        _required_pacakge_list+="libncurses5-dev "
        _required_pacakge_list+="google-perftools "
        _required_pacakge_list+="numactl "
        _required_pacakge_list+="yarn "
        _required_pacakge_list+="jq "
        _required_pacakge_list+="libnuma-dev "
        _required_pacakge_list+="libgtest-dev "
        # required for building performant PyTorch fork from 2.2 onwards
        _required_pacakge_list+="libmkl-dev "

        if [ -n "$ZSH_NAME" ]; then
            setopt shwordsplit
        fi
        for _package in $_required_pacakge_list; do
            if [ $(dpkg-query -W -f='${Status}' $_package 2>/dev/null | grep -c "ok installed") -eq 0 ]; then
                sudo apt install -y $_package
                if [ $? -ne 0 ]; then
                    _failed_pacakge_list="$_failed_pacakge_list $_package"
                fi
            fi
        done
        if [ -n "$ZSH_NAME" ]; then
            unsetopt shwordsplit
        fi
    fi

    if [ -n "$_failed_pacakge_list" ]; then
        echo
        echo "Warning $(__get_func_name) failed installing the package(s): $_failed_pacakge_list"
        echo
    fi

    return 0
}

_extract_and_rebuild_boost ()
{
    local __build_res=""

    if [ ! -e $FUNC_SIM_BOOST_BUILD/boost/stage/lib/libboost_atomic.a ] || [ -n "$__build_boost" ] ; then
        if [ -d $FUNC_SIM_BOOST_BUILD ]; then
            rm -rf $FUNC_SIM_BOOST_BUILD
        fi
        mkdir -p $FUNC_SIM_BOOST_BUILD
        pushd $FUNC_SIM_BOOST_BUILD
        echo -e "Extracting Boost"
        cp -r $THIRD_PARTIES_ROOT/boost boost
        cd boost
        ./bootstrap.sh --without-libraries=python
        # SW-5680 the CPLUS include path is to solve debian specific compilation issue
        CPLUS_INCLUDE_PATH=/usr/local/include/python3.6m ./b2 link=static cxxflags=-fPIC cflags=-fPIC -j$__jobs
        __build_res=$?
        popd
        if [ $__build_res -ne 0 ]; then
            echo "Failed to build Boost !!! Exiting..." >&2
            return $__build_res
        fi
    fi
}

build_framework_integration_docs ()
{
    SECONDS=0

    _verify_exists_dir "$FW_DOC_ROOT" $FW_DOC_ROOT
    : "${FW_DOC_BUILD:?Need to set FW_DOC_BUILD to the build folder}"

    local __scriptname=$(__get_func_name)

    local __build_res=0
    local __clean=""
    local __color="yes"
    local __configure=""
    local __jobs=$NUMBER_OF_JOBS
    local __verbose=0

    # parameter while-loop
    while [ -n "$1" ];
    do
        case $1 in
        -a  | --build-all | \
        -r  | --release )
            # All builds are the same for documentation
            # Those flags kept for CI compatibility
            ;;
        -C  | --clean )
            __clean="yes"
            ;;
        -c  | --configure )
            __configure="yes"
            ;;
        --no-color )
            __color="no"
            ;;
        -j  | --jobs )
            shift
            __jobs=$1
            ;;
        -v  | --verbose )
            __verbose=1
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

    if [ -n "$__clean" ]; then
        rm -rf $FW_DOC_BUILD
    fi

    if [ ! -d $FW_DOC_BUILD ]; then
        __configure="yes"
    fi

    if [ -n "$__configure" ]; then
        __check_mandatory_pkgs
        if [ $? -ne 0 ]; then
            return 1
        fi

        if [ -d $FW_DOC_BUILD ]; then
            rm -rf $FW_DOC_BUILD
        fi

        mkdir -p $FW_DOC_BUILD
        (
            set -x
            cd $FW_DOC_BUILD
            cmake -G "Unix Makefiles" -DCMAKE_VERBOSE_MAKEFILE=$__verbose -DCMAKE_COLOR_MAKEFILE=$__color $FW_DOC_ROOT
        )
    fi

    _verify_exists_dir "$FW_DOC_BUILD" $FW_DOC_BUILD
    pushd $FW_DOC_BUILD
    cmake --build . -- V=$__verbose -j$__jobs
    __build_res=$?
    popd
    if [ $__build_res -ne 0 ]; then
        return $__build_res
    fi

    printf "\nElapsed time: %02u:%02u:%02u \n\n" $(($SECONDS / 3600)) $((($SECONDS / 60) % 60)) $(($SECONDS % 60))
    return 0
}

__find_all_simulations()
{
    local ret=1 # errror
    local proc_pid

    if [ -d /sys/class/habanalabs ]; then
        for proc_pid in $(find /proc -maxdepth 1 -name "[0-9]*"); do
            local output=$(ls -l ${proc_pid}/fd 2>/dev/null | grep "/dev/hlv")
            if [ -n "${output}" ]; then
                local vdevice_file=$(echo $output | awk '{print $NF}' | sort -u)
                local lkd_id="${vdevice_file##*"/dev/hlv"}"
                local sim_id=`expr $lkd_id - 200`
                local pid="${proc_pid#/proc/}"
                local dev_name=$(head -n 1 "/sys/class/habanalabs/hlv$lkd_id/device_name" | cut -d'_' -f1)

                printf '/dev/accel/accel%s\t%s\t\t%s\n' $sim_id $dev_name $pid

                ret=0
            fi
        done
    fi

    if [ $ret -ne 0 ]; then
        echo "There are no active simulations"
    fi

    return $ret
}

__kill_sim_by_dev()
{
    local device=$1
    local number="${device##*"hl"}"
    local lkd_id=`expr $number + 200`

    if [ -d /sys/class/habanalabs/hlv$lkd_id ]; then
        local pid=$(__find_all_simulations | grep $1 | awk '{print $3}')
        kill -9 $pid >> /dev/null

        echo "Device $1 was killed"
        return 0
    else
        echo "Device $1 wasn't killed since it wasn't found"
        return 1 # error
    fi
}

__install_bazel()
{
    local __bazel_ver="0.25.2"
    local __install_res=0

    local __bazel_info=$(bazel version 2> /dev/null)
    local __current_bazel_ver=""
    local __bazel_installed=${?}

    if [ ${__bazel_installed} -eq 0 ]; then
        __current_bazel_ver=$(echo ${__bazel_info} | awk 'NR==1{print $3}')
    fi

    if [ ${__bazel_installed} -ne 0 ] || [ "${__bazel_ver}" != "${__current_bazel_ver}" ]; then
        __check_mandatory_pkgs
        if [ $? -ne 0 ]; then
            return 1
        fi
    else
        echo "Bazel ${__bazel_ver} already installed "
    fi
    return ${__install_res}
}

run_auto_index_test()
{
    local __exe_name="run_auto_index_test.py"
    local __auto_index_test_dir="$TPC_KERNELS_ROOT/scripts/"
    local __python_ver="python3 "

    local __scriptname=$(__get_func_name)
    local __test_filter=""
    local __test_category=""
    local __tests_per_process=""
    local __xml_output_path=""

    # parameter while-loop
    while [ -n "$1" ];
    do
        case $1 in
        -f | --filter )
            shift
            __test_filter="-f $1"
            ;;
        -c | --category )
            shift
            __test_category="-c $1"
            ;;
        -t | --test_per_process )
            shift
            __tests_per_process="-t $1"
            ;;
        -x | --xml_output )
            shift
            __xml_output_path="-x $1"
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

    local __auto_index_test_exe=$__python_ver$__auto_index_test_dir$__exe_name

    (set -x;${__auto_index_test_exe} $__test_filter $__test_category $__tests_per_process $__xml_output_path)

    # return error code of the test
    return $?
}

run_habana_py_qa_test()
{
    local __habana_py_qa_test_exe="python -m pytest $HABANA_PY_QA_ROOT"

    local __scriptname=$(__get_func_name)
    local __print_tests=""
    local __filter=""
    local __pdb=""
    local __xml="--junit-xml=qa_test_res.xml"
    local __failures=""
    local __color=""

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
        -p  | --pdb )
            shift
            __pdb="--pdb"
            ;;
        -x  | --xml )
            shift
            __xml="--junit-xml=$1"
            ;;
        --no-color )
            __color="--color=no"
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
        ${__habana_py_qa_test_exe} --collect-only
        return $?
    fi

    (set -x;${__habana_py_qa_test_exe} -q $__failures $__filter $__pdb $__xml $__color -n auto >> /dev/null)

    # return error code of the test
    return $?
}

habana_mxnet_clone()
{
    sudo rm -rf $HABANA_MXNET_ROOT
    git clone git@192.168.0.22:mxnet $HABANA_MXNET_ROOT --recursive
    pushd $HABANA_MXNET_ROOT
    git checkout habana
    git submodule sync
    cd nnvm
    git fetch
    git checkout habana
    git submodule sync
    cd ..
    git submodule foreach -q --recursive 'echo $path ; git fetch --all ; git checkout $($([ -f ../.gitmodules ]) git config -f ../.gitmodules submodule.$name.branch || $([ -f ../../.gitmodules ]) git config -f ../../.gitmodules submodule.$name.branch || echo master)'
    popd
}

function parse_git_branch {
    git branch --no-color 2> /dev/null | sed -e '/^[^*]/d' -e 's/* \(.*\)/(\1) /'
}

manage_sandbox()
{
    local __scriptname=$(__get_func_name)

    local __jobs=$NUMBER_OF_JOBS
    local __sandbox=""
    local __root_dir="$HOME"
    local __commit_tag="master"
    local __manifest_url="$HABANA_REPO_URL"
    local __manifest_file="default.xml"
    local __op=""
    local __src_dir=""
    local __bin_dir=""
    local __log_dir=""
    local __local_path=""
    local __local_path_excludes="buildroot-external/output"
    local __src_root_dir=""
    local __full_src_dir=""
    local __full_bin_dir=""
    local __full_log_dir=""

    # parameter while-loop
    while [ -n "$1" ];
    do
        case $1 in
        -s  | --sandbox )
            shift
            __sandbox=$1
            ;;
        -r  | --root_dir )
            shift
            __root_dir=$1
            ;;
        -c  | --commit_tag )
            shift
            __commit_tag=$1
            ;;
        -u  | --url )
            shift
            __manifest_url=$1
            ;;
        -p  | --local-path )
            shift
            __local_path=$1
            if [ -z "$__local_path" ]; then
                echo "Error, '-p' requires an argument"
                usage $__scriptname
                return 1 # error
            fi
            ;;
        -m  | --manifest )
            shift
            __manifest_file=$1
            ;;
        -o  | --op )
            shift
            __op=$1
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

    if [ "$__op" != "list" ]; then
        if [ -z "$__sandbox" ]; then
            echo "Error, sandbox name can't be blank"
            return 1
        fi
    fi

    __src_dir=npu-stack_$__sandbox
    __bin_dir=builds_$__sandbox
    __log_dir=.habana_logs/$__sandbox/
    __src_root_dir=$__root_dir/trees
    __full_src_dir=$__src_root_dir/$__src_dir
    __full_bin_dir=$__root_dir/builds/$__bin_dir
    __full_log_dir=$HOME/$__log_dir

    case $__op in
    create )
        if [ -z "$__commit_tag" ]; then
            echo "Error, commit tag can't be blank"
            return 1
        fi

        if [ -d $__full_src_dir ]; then
            echo "Error, can't create sandbox '$__sandbox' - already exists"
            return 1
        fi

        if [ -n "$__local_path" ]; then
            if [ ! -d $__local_path ]; then
                echo "Error, can't copy from '$__local_path' - does not exist"
                return 1
            fi

            echo -e "Note: '-p' flag specified - ignoring any of '-c' '-u' '-m' flags...\n"
            echo "Creating sandbox '$__sandbox' from local source directory '$__local_path'"
        else
            echo "Creating sandbox '$__sandbox' with URL '$__manifest_url' and manifest '$__manifest_file' using commit tag '$__commit_tag'"
        fi

        mkdir -p $__full_src_dir
        if [ $? -ne 0 ]; then
            echo "Error, can't create sandbox '$__sandbox' - failed to create directory"
            return 1
        fi

        if [ -n "$__local_path" ]; then
            sudo rsync -a --delete --info=progress2 --exclude="$__local_path_excludes" $__local_path/ $__full_src_dir
            if [ $? -ne 0 ]; then
                echo "Error, can't create sandbox '$__sandbox' - failed to copy directory"
                return 1
            fi
        else
            pushd $__full_src_dir

            repo init -u $__manifest_url -m $__manifest_file -b $__commit_tag
            if [ $? -ne 0 ]; then
                echo "Error, failed to execute repo init for git ID '$__commit_tag'"
                popd
                return 1
            fi

            repo sync -j$__jobs
            if [ $? -ne 0 ]; then
                echo "Error, failed to execute sync init"
                popd
                return 1
            fi

            popd

        fi

        echo
        echo "Sandbox '$__sandbox' was created in root directory '$__root_dir'"
        ;;
    delete )
        echo "The following directories are part of sandbox '$__sandbox' will be deleted:"
        echo "    $__full_src_dir"
        echo "    $__full_bin_dir"
        echo "    $__full_log_dir"
        echo

        __yesno "are you REALLY sure you want to delete sandbox '$__sandbox'?"
        case $__ansyn in
        y )
            echo "Deleting sandbox '$__sandbox'. This may takes a few minutes ..."
            \rm -fr $__full_src_dir $__full_bin_dir $__full_log_dir
            echo "Sandbox '$__sandbox' was deleted"
            ;;
        * )
            # do nothing
            ;;
        esac
        ;;
    list )
        echo "The following sandboxes were found:"
        ls $__src_root_dir | grep npu-stack | cut -c11- | sed -e /^$/d | sed 's/^/ \* /'
        ;;
    set )
        if [ ! -d $__full_src_dir ]; then
            echo "Error, can't set sandbox '$__sandbox' - source directory doesn't exists"
            return 1
        fi

        SET_ABSOLUTE_HABANA_ENV=1 source $__full_src_dir/automation/habana_scripts/habana_env `dirname $__full_src_dir` $__full_src_dir $__full_bin_dir
        export HABANA_LOGS=$__full_log_dir
        if [ ! -d $__full_log_dir ]; then
            mkdir -p $__full_log_dir
        fi
        PS1="\[\e[0;35m[$__sandbox] \e[m\]$PS1"
        echo "Sandbox '$__sandbox' was set"
        ;;
    * )
        echo "Error, unsupported sandbox operation '$__op'"
        usage $__scriptname
        return 1
    esac
}

__common_build_dependency()
{
    local __jobs=$NUMBER_OF_JOBS
    local __release=""
    local __all=""
    local __configure=""
    local __lib_only=""
    local __module=""
    local __release_par=""
    local __configure_par=""
    local __lib_only_par=""
    local __module_par=""
    local __docker_mode=""
    local __override_config=""
    local __override_config_par=""
    local __run_in_containers=""
    local __docker_run_cmd=""
    local __build_res=0

    # parameter while-loop
    while [ -n "$1" ];
    do
        case $1 in
        -a  | --build-all )
            __all="yes"
            ;;
        -r  | --release )
            __release="yes"
            ;;
        -c  | --configure )
            __configure="yes"
            ;;
        -l  | --lib_only )
            __lib_only="yes"
            ;;
        -d  | --docker_mode )
            __docker_mode="yes"
            ;;
        -x  | --override-config )
            shift
            __override_config=$1
            ;;
        -m  | --module )
            shift
            __module=$1
            ;;
        -t  | --run_in_containers )
            __run_in_containers="yes"
            ;;
        -j  | --jobs )
            shift
            __jobs=$1
            ;;
        *)
            echo "The parameter $1 is not allowed in $(__get_func_name)"
            return 1 # error
            ;;
        esac
        shift
    done

    if [ -n "$__configure" ]; then
        __configure_par="-c"
    fi

    if [ -n "$__release" ]; then
        __release_par="-r"
    fi

    if [ -n "$__all" ]; then
        __release_par="-a"
    fi

    if [ -n "$__lib_only" ]; then
        __lib_only_par="-l"
    fi

    if [ -n "$__module" ]; then
        __module_par="-m $__module"
    fi

    if [ -n "$__override_config" ]; then
        __override_config_par="-x $__override_config"
    fi

    # Since PT bridge depends on other PT packages;
    # by default, we don't want PT to be built in this mode.
    # If one wants to build it, he should build it explicitly.
    local build_commands=$($HABANA_SCRIPTS_FOLDER/package_dependency.py $__module_par $__configure_par $__release_par $__lib_only_par $__override_config_par | grep -v build_pytorch_modules)
    if [ $? -ne 0 ]; then
        echo "Error, failed to get the dependency commands"
        return 1
    fi

    if [[ -z "$build_commands" ]]; then
        echo "No build pre-requisite required for $__module"
        return 0
    fi

    local number_of_build_command_lines=$(wc -l <<< "$build_commands")
    if [[ $number_of_build_command_lines -gt 1 ]]; then
        echo "The following build commands will be executed:"
        echo "$build_commands" | while read -r __current_build; do
            if [[ -n "$__docker_mode" ]] && [[ $__current_build == *"build_and_insmod_habanalabs"* ]]   ; then
               continue
            fi
            if [[ -n "$__run_in_containers" ]] && [[ $__current_build == *"build_habana_tf_modules"* ]]   ; then
                # Tensorflow-training component does not support build inside container. Skipping.
                continue
            fi
            echo "# $__current_build"
        done
    fi

    if [[ -n "$__run_in_containers" ]]; then
            local __build_cache_mount_par=""
            local __docker_name_par=""
            local __ipc_par=""
            local __tmpfs_par=""
            local __weka_mount_par=""
            local __workspace_par=""
            local __src_dir_par=""
            local __extra_params=""
            local __user_name=$(whoami)
            local __dev_run=""
            local __dev_docker_name="artifactory-kfs.habana-labs.com/devops-docker-local/habana-builder:ubuntu20.04-${__user_name}"

            if [[ -n "$BUILD_CACHE_MOUNT" ]]; then
                __build_cache_mount_par="-c $BUILD_CACHE_MOUNT"
            fi
            if [[ -n "$WORKDIR" ]]; then
                __workdir_par="-d $WORKDIR"
            fi
            if [[ -n "$DOCKER_NAME" ]]; then
                __docker_name_par="-i $DOCKER_NAME"
                __dev_docker_name="$DOCKER_NAME-$__user_name"
            fi
            if [[ -n "$IPC_PARAM" ]]; then
                __ipc_par="-p $IPC_PARAM"
            fi
            if [[ -n "$TPMFS_PARAM" ]]; then
                __tmpfs_par="-t $TPMFS_PARAM"
            fi
            if [[ -n "$WEKA_PARAM" ]]; then
                __weka_mount_par="-m $WEKA_PARAM"
            fi
            if [[ -n "$WORKSPACE" ]]; then
                __workspace_par="-w $WORKSPACE"
            fi
            if [[ "$CI" == "true" ]]; then
                __src_dir_par="-s $HABANA_SOFTWARE_STACK"
            else
                # Domain accounts logins are not present in /etc/passwd. This causes docker container to not recognize
                # user running build_npu_stack on local vm. Effect of that is lack of possibility to run commands with sudo.
                # To workaround this issue we create user inside docker container basing on provided image and add to
                # /etc/sudors file. After that image is commited and used for compilation of build_npu_stack components.
                __dev_run_par="-u"
                local __init_container="$__user_name-init"
                docker rm $__init_container
                local __cmd="get_docker_command $__docker_name_par -n $__init_container $__dev_run_par --init-run"
                local __init_cmd=$($__cmd)
                __init_cmd="$__init_cmd bash -c 'useradd --uid $(id -u) ${__user_name}; echo \"${__user_name} ALL=(ALL) NOPASSWD: ALL\" >> /etc/sudoers'"
                $__init_cmd
                docker rmi $__dev_docker_name
                docker commit $__init_container $__dev_docker_name
                docker rm $__init_container
                sudo chown $(id -u):$(id -g) -R ~/.venv
                __docker_name_par="-i $__dev_docker_name"
            fi
            if [[ -n "$EXTRA_PARAMS" ]]; then
                __extra_params="-e ${EXTRA_PARAMS}"
            fi

            local __cmd="get_docker_command ${__extra_params} ${__build_cache_mount_par} ${__workdir_par} ${__docker_name_par} ${__ipc_par} ${__tmpfs_par} ${__workspace_par} ${__src_dir_par} ${__dev_run_par}"
            __docker_run_cmd=$($__cmd)
        fi

    echo "$build_commands" | while read -r __current_build; do
        if [[ -n "$__docker_mode" ]] && [[ $__current_build == *"build_and_insmod_habanalabs"* ]]   ; then
            continue
        fi
        if [[ -n "$__run_in_containers" ]] && [[ $__current_build == *"build_habana_tf_modules"* ]]   ; then
            echo 'Tensorflow-training component does not support build inside container. Skipping.'
            continue
        fi

        local __full_build_cmd="$__docker_run_cmd $__current_build -j $__jobs"
        echo "#--------------------------------------------------------------------"
        echo "# Running the following build command: $__full_build_cmd"
        echo "#--------------------------------------------------------------------"

        $__full_build_cmd
        __build_res=$?

        # in case of an error, try to (re)build from scratch this package
        if [ $__build_res -ne 0 ]; then
            if [ -z "$__configure_par" ]; then
                $__full_build_cmd -c
                __build_res=$?
            fi
        fi

        if [ $__build_res -ne 0 ]; then
            echo "'$__full_build_cmd' failed with error code : $__build_res"
            return $__build_res
        fi
    done
}

build_npu_stack()
{
    local __scriptname=$(__get_func_name)
    local __jobs=$NUMBER_OF_JOBS
    local __release=""
    local __all=""
    local __force_clean=""
    local __configure=""
    local __release_par=""
    local __configure_par=""
    local __jobs_par=""
    local __docker_par=""
    local __docker_mode=""
    local __override_config=""
    local __override_config_par=""
    local __lib_only=""
    local __lib_only_par=""
    local __run_in_containers=""
    local __run_in_containers_par=""
    local __build_res=0

    # parameter while-loop
    while [ -n "$1" ];
    do
        case $1 in
        -a  | --build-all )
            __all="yes"
            ;;
        -f  | --force-clean )
            __force_clean="yes"
            ;;
        -r  | --release )
            __release="yes"
            ;;
        -c  | --configure )
            __configure="yes"
            ;;
        -l  | --lib-only )
            __lib_only="yes"
            ;;
        -x  | --override-config )
            shift
            __override_config=$1
            ;;
        -d  | --docker_mode )
            __docker_mode="yes"
            ;;
        -t  | --run_in_containers )
            __run_in_containers="yes"
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

    if [ -n "$__force_clean" ]; then
        read -r -p "Are you sure you want to force clean the repo ??!! [Y/n]: " response
        response=${response,,} # tolower
        if [[ $response =~ ^(yes|y) ]]; then
            if [ -z "$HABANA_SOFTWARE_STACK" ]; then
                echo "Error : HABANA_SOFTWARE_STACK wasn't set"
                return 1
            fi
            rm -rf "$BUILD_ROOT"
            rm -rf "$HABANA_SOFTWARE_STACK"
            mkdir -p "$HABANA_SOFTWARE_STACK"
            cd $HABANA_SOFTWARE_STACK
            repo init -u $HABANA_REPO_URL
            repo sync
        fi
    fi

    if [ -n "$__configure" ]; then
        __configure_par="-c"
    fi

    if [ -n "$__release" ]; then
        __release_par="-r"
    fi

    if [ -n "$__all" ]; then
        __release_par="-a"
    fi

    if [ -n "$__docker_mode" ]; then
        __docker_par="-d"
    fi

    if [ -n "$__override_config" ]; then
        __override_config_par="-x $__override_config"
    fi

    if [ -n "$__lib_only" ]; then
        __lib_only_par="-l"
    fi

    if [ -n "$__run_in_containers" ]; then
        __run_in_containers_par="-t"
    fi

    __jobs_par="-j $__jobs"

    __common_build_dependency $__configure_par $__release_par $__jobs_par $__docker_par $__override_config_par $__lib_only_par $__run_in_containers_par
}

set_tpcsim_mode()
{
  SECONDS=0

    _verify_exists_dir "$BUILD_ROOT_LATEST" $BUILD_ROOT_LATEST
    _verify_exists_dir "$TPCSIM_RELEASE_BUILD" $TPCSIM_RELEASE_BUILD
    _verify_exists_dir "$TPCSIM_DEBUG_BUILD" $TPCSIM_DEBUG_BUILD

    local __scriptname=$(__get_func_name)

    local __jobs=$NUMBER_OF_JOBS
    local __is_debug="yes"

    # parameter while-loop
    while [ -n "$1" ];
    do
        case $1 in
        -r  | --release )
            __is_debug=""
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

    pushd $BUILD_ROOT_LATEST

    if [ -n "$__is_debug" ]; then
        echo -e "Setting tpcsim in debug mode"
        ln -sfn $TPCSIM_DEBUG_BUILD/libtpcsim_shared.so libtpcsim_shared.so
    fi

    if [ -z "$__is_debug" ]; then
        echo "Setting tpcsim in release mode"
        ln -sfn $TPCSIM_RELEASE_BUILD/libtpcsim_shared.so libtpcsim_shared.so
    fi

    popd

    printf "\nElapsed time: %02u:%02u:%02u \n\n" $(($SECONDS / 3600)) $((($SECONDS / 60) % 60)) $(($SECONDS % 60))
    return 0
}


build_sbs_ci_test()
{
    SECONDS=0

    _verify_exists_dir "$SBS_ROOT" $SBS_ROOT
    #: "${SBS_DEBUG_BUILD_GAUDI:?Need to set SBS_DEBUG_BUILD_GAUDI to the build folder}"
    #: "${SBS_RELEASE_BUILD_GAUDI:?Need to set SBS_RELEASE_BUILD_GAUDI to the build folder}"

    local __scriptname=$(__get_func_name)

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

        if [ ! -d $SBS_DEBUG_BUILD_GAUDI ]; then
            __configure="yes"
        fi

        if [ -n "$__configure" ]; then
            if [ -d $SBS_DEBUG_BUILD_GAUDI ]; then
                rm -rf $SBS_DEBUG_BUILD_GAUDI
            fi
            mkdir -p $SBS_DEBUG_BUILD_GAUDI
        fi

        _verify_exists_dir "$SBS_DEBUG_BUILD_GAUDI" $SBS_DEBUG_BUILD_GAUDI

        pushd $SBS_DEBUG_BUILD_GAUDI
            (set -x; cmake \
            -DCMAKE_BUILD_TYPE="Debug" \
            -DCMAKE_INSTALL_PREFIX=$HOME/.local \
            -DCMAKE_COLOR_MAKEFILE=$__color \
            $SBS_ROOT)
        make $__verbose
        __build_res=$?
        popd

        if [ $__build_res -ne 0 ]; then
            return $__build_res
        fi
    fi

    __configure=$__org_configure

    if [ -n "$__release" ]; then
        echo "Building in release mode"
        if [ ! -d $SBS_RELEASE_BUILD_GAUDI ]; then
            __configure="yes"
        fi

        if [ -n "$__configure" ]; then
            if [ -d $SBS_RELEASE_BUILD_GAUDI ]; then
                rm -rf $SBS_RELEASE_BUILD_GAUDI
            fi
            mkdir -p $SBS_RELEASE_BUILD_GAUDI
        fi

        _verify_exists_dir "$SBS_RELEASE_BUILD_GAUDI" $SBS_RELEASE_BUILD_GAUDI

        pushd $SBS_RELEASE_BUILD_GAUDI
        (set -x; cmake \
            -DCMAKE_BUILD_TYPE="Release" \
            -DCMAKE_INSTALL_PREFIX=$HOME/.local \
            -DCMAKE_COLOR_MAKEFILE=$__color \
            $SBS_ROOT)
        make $__verbose
        __build_res=$?
        popd

        if [ $__build_res -ne 0 ]; then
            return $__build_res
        fi

    fi
    printf "\nElapsed time: %02u:%02u:%02u \n\n" $(($SECONDS / 3600)) $((($SECONDS / 60) % 60)) $(($SECONDS % 60))
    return 0
}

run_resnet_sbs_ci_test()
{
    local __executable=$SBS_RELEASE_BUILD_GAUDI/gaudi_resnet50_training_sbs
    local __scriptname=$(__get_func_name)
    local __orig_args="$@"
    local __build_res=""

    _verify_exists_file "$__executable" $__executable

    ${__executable} ${__orig_args}
    __build_res=$?

    if [ ! -s ${WORKSPACE}/failed_tensor.log ]; then
        printf "\ntest passed - removing swout files\n\n"
        rm -rf ${HABANA_LOGS}/*SWOUT*.npy
    fi

    # return error code of the test
    return $__build_res
}

run_habana_regs_cli()
{
    local __executable=$HABANA_REGS_CLI_BUILD/habana_regs_cli

    _verify_exists_file "$__executable" $__executable

    (set -x; sudo ${__executable} $@)

    # return error code of the test
    return $?
}

collect_runtime_logs()
{
    set +e # ignoring errors due to IT-40227
    lsmod > "${HABANA_LOGS}"/lsmod.log
    lsof > "${HABANA_LOGS}"/lsof.log
    ps -elf > "${HABANA_LOGS}"/ps_-elf.log
    dmesg -T > "${HABANA_LOGS}"/dmesg_-T.log
    set -e
}

# The following function invocation must be the last line in this script, please add new functions above this segment

if [ -n "$1" ]; then
    declare -i __ret=0
    $1 ${@:2} || { __ret=$?; exit $__ret; }
fi


get_docker_command()
{
    local __image_name="artifactory-kfs.habana-labs.com/devops-docker-local/habana-builder:ubuntu20.04"
    local __extra_params=""
    local __ipc_param=""
    local __workdir=$PWD
    local __tmpfs_param=""
    local __build_cache_mount=""
    local __workspace=$HOME
    local __source_dir=$HABANA_NPU_STACK_PATH
    local __build_dir=$BUILD_ROOT
    local __docker_cmd=""
    local __cmd=""
    local __dst_source_dir=$HOME/repos
    local __weka_mount_point_param=""
    local __mount_point_param=""
    local __mount_point=""
    local __container_name=""
    local __user_params="--user $(id -u):$(id -g)"
    local __container_rm="--rm"
    local __user_mounts="
        -v /etc/group:/etc/group:ro \
        -v /etc/passwd:/etc/passwd:ro \
        -v /etc/shadow:/etc/shadow:ro \
        "

    while [ -n "$1" ];
    do
        case $1 in
        -b  | --build-dir )
            shift
            __build_dir=$1
            ;;
        -c  | --build-cache )
            shift
            __build_cache_mount=$1
            ;;
        -d  | --workdir )
            shift
            __workdir=$1
            ;;
        -e  | --extra-params )
            shift
            __extra_params=$1
            ;;
        -i  | --image-name )
            shift
            __image_name=$1
            ;;
        -m  | --weka-mount )
            shift
            __weka_mount_point_param="-v $1"
            ;;
        -n  | --container-name )
            shift
            __container_name="--name $1"
            ;;
        -p  | --ipc )
            shift
            __ipc_param="--ipc=$1"
            ;;
        -t  | --tmpfs )
            shift
            __tmpfs_param="--tmpfs=$1"
            ;;
        -w  | --workspace )
            shift
            __workspace=$1
            ;;
        -s  | --source-dir )
            shift
            __source_dir=$1
            __dst_source_dir=$__source_dir
            ;;
        -u | --skip-usr-mounts )
            __user_mounts=""
            ;;
        -x  | --execute-command )
            shift
            __cmd=$1
            ;;
        -v  | --volume )
            shift
            for __mount_point in ${1}; do
                if [ -n "$__mount_point" ] && [ "${__mount_point:0:1}" != "-" ]; then
                    __mount_point_param+="-v ${__mount_point} "
                fi
            done
            ;;
        --init-run )
            __user_params=""
            __container_rm=""
            __user_mounts=""
            ;;
        *)
            echo "The parameter $1 is not allowed in $(__get_func_name)"
            return 1 # error
            ;;
        esac
        shift
    done

    asic_write_premission_demand=$(if [ "${HW_TYPE}" == "asic" ]; then echo "rw"; else echo "ro"; fi);
    __docker_cmd="docker run -t ${__container_rm} --privileged --network=host ${__ipc_param} \
        ${__user_params} --workdir=${__workdir} \
        --ulimit memlock=-1:-1 \
        ${__tmpfs_param} \
        ${__build_cache_mount} \
        ${__extra_params} \
        ${__weka_mount_point_param} \
        ${__mount_point_param} \
        -v /dev:/dev \
        -v /sys/kernel/debug:/sys/kernel/debug \
        ${__user_mounts} \
        -v /etc/environment:/etc/environment:ro \
        -v /etc/localtime:/etc/localtime:ro \
        -v /software:/software:ro \
        -v /git_lfs/data:/git_lfs/data:ro \
        -v /software/data/pytorch/cache:/software/data/pytorch/cache \
        -v /software/ci:/software/ci \
        -v /qa:/qa:ro \
        -v /qa/frameworks/Events/training-logs:/qa/frameworks/Events/training-logs \
        -v /qa/gdn_qa:/qa/gdn_qa \
        -v /qa/qa_logs:/qa/qa_logs \
        -v /tools:/tools:ro \
        -v /synopsys:/synopsys:ro \
        -v /usr/bin/hl-smi:/usr/bin/hl-smi:${asic_write_premission_demand} \
        -v ${HOME}/.ssh:${HOME}/.ssh:ro \
        -v ${__workspace}:${__workspace} \
        -v ${__source_dir}:${__dst_source_dir} \
        -v ${__build_dir}:${__build_dir} \
        -v ${__workspace}/habana_logs:${__workspace}/.habana_logs \
        -e HOME=${__workspace} \
        -e HABANA_NO_VENV \
        ${__container_name} \
        ${__image_name} \
        ${__cmd}"

    echo $__docker_cmd
}