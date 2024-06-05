#!/bin/bash
#
# Copyright (C) 2022 HabanaLabs, Ltd.
# All Rights Reserved.
#
# Unauthorized copying of this file, via any medium is strictly prohibited.
# Proprietary and confidential.
# Author: Shai Kedem <skedem@habana.ai>
#

#
# You should run this script from a git root dir
#

#
function run_iwyu()
(
    local -r _prog_name="${FUNCNAME}"

    function _usage()
    {
        echo -e "\n"
        echo -e "Usage: ${_prog_name} [options]"
        echo -e "       ${_prog_name} [-v] [-c] [-d] [--build-cmd <build_...>] < <--stash> | <--head <num--commits-back> > | <--commits <sha1>,<sha2> > >"
        echo -e "       ${_prog_name} [-v] [ --nobuild [ --before-file <file-name> ] [ --after-file <file-name> ] ]"
        echo -e "\n"
        echo -e "Examples:"
        echo -e "         ${_prog_name} --clean --stash"
        echo -e "         ${_prog_name} --head 1"
        echo -e "         ${_prog_name} --commits ce8a6679,866a60f6"
        echo -e "         ${_prog_name} --debug --head 1"
        echo -e "         ${_prog_name} --build-cmd build_mme --head 1"
        echo -e "         ${_prog_name} --nobuild --before-file /tmp/before.out --after-file /tmp/after.out"

        echo -e "\noptions:"
        echo -e "  -v,  --verbose               Print verbose messages for debug"
        echo -e "       --nobuild               Do not perform build command before and after the change (by default will build)"
        echo -e "       --build-cmd <build_...> When build, run this build command, default is build_hcl"
        echo -e "  -c,  --clean                 When build, perform clean build before and after the change"
        echo -e "  -d,  --debug                 When build, use debug mode build. By default will use release"
        echo -e "  -s,  --stash                 When build, perform git stash to build before and unstash to build after (final state is unstashed)"
        echo -e "       --head <num>            When build, perform git checkout to head~<num> to build before and then checkout to HEAD to build after (change must be committed)"
        echo -e "       --commits <#,#>         When build, perform git checkout to 1st commit hash to build before and then checkout to 2nd commit hash to build after (changes must be committed)"
        echo -e "       --before-file <#>       Override name of temp file to store build with IWYU output before change. In case --nobuild is used, use this file as input for before change"
        echo -e "       --after-file <#>        Override name of temp file to store build with IWYU output after change. In case --nobuild is used, use this file as input for after change"
        echo -e "  -h,  --help                  Prints this help"
        echo -e ""
    }

    local verbose_output=false

    # Colors
    function _set_colors()
    {
        local -r colors="local -r Black='\033[0;30m' ; local -r Red='\033[0;31m' ; local -r Green='\033[0;32m'; local -r Yellow='\033[0;33m'; local -r Blue='\033[0;34m'; local -r Purple='\033[0;35m'; local -r Cyan='\033[0;36m'; local -r White='\033[0;37m'; local -r BBlack='\033[1;30m'; local -r BRed='\033[1;31m'; local -r BGreen='\033[1;32m'; local -r BYellow='\033[1;33m'; local -r BBlue='\033[1;34m'; local -r BPurple='\033[1;35m'; local -r BCyan='\033[1;36m'; local -r BWhite='\033[1;37m'; local -r UBlack='\033[4;30m'; local -r URed='\033[4;31m'; local -r UGreen='\033[4;32m'; local -r UYellow='\033[4;33m'; local -r UBlue='\033[4;34m'; local -r UPurple='\033[4;35m'; local -r UCyan='\033[4;36m'; local -r UWhite='\033[4;37m'; local -r On_Black='\033[40m'; local -r On_Red='\033[41m'; local -r On_Green='\033[42m'; local -r On_Yellow='\033[43m'; local -r On_Blue='\033[44m'; local -r On_Purple='\033[45m'; local -r On_Cyan='\033[46m'; local -r On_White='\033[47m'; local -r IBlack='\033[0;90m'; local -r IRed='\033[0;91m'; local -r IGreen='\033[0;92m'; local -r IYellow='\033[0;93m'; local -r IBlue='\033[0;94m'; local -r IPurple='\033[0;95m'; local -r ICyan='\033[0;96m'; local -r IWhite='\033[0;97m'; local -r BIBlack='\033[1;90m'; local -r BIRed='\033[1;91m'; local -r BIGreen='\033[1;92m'; local -r BIYellow='\033[1;93m'; local -r BIBlue='\033[1;94m'; local -r BIPurple='\033[1;95m'; local -r BICyan='\033[1;96m'; local -r BIWhite='\033[1;97m'; local -r On_IBlack='\033[0;100m'; local -r On_IRed='\033[0;101m'; local -r On_IGreen='\033[0;102m'; local -r On_IYellow='\033[0;103m'; local -r On_IBlue='\033[0;104m'; local -r On_IPurple='\033[0;105m'; local -r On_ICyan='\033[0;106m'; local -r On_IWhite='\033[0;107m'; local -r NC='\033[0m'"
        echo "${colors}"
    }

    function _print_error()
    {
        eval $( _set_colors )
        local -r str="$1"

        echo -e "${BIRed}${str}${NC}"
    }

    function _print_verbose()
    {
        eval $( _set_colors )

        if "${verbose_output}" ; then
            local -r str="$1"
            echo -e "${White}${str}${NC}"
        fi
    }

    function _print_verbose_ok()
    {
        eval $( _set_colors )

        if "${verbose_output}" ; then
            local -r str="$1"
            echo -e "${IGreen}${str}${NC}"
        fi
    }

    function _print_info()
    {
        eval $( _set_colors )
        local -r str="$1"

        echo -e "${White}${str}${NC}"
    }

    function _print_info_ok()
    {
        eval $( _set_colors )
        local -r str="$1"

        echo -e "${IGreen}${str}${NC}"
    }

    function _print_warning()
    {
        eval $( _set_colors )
        local -r str="$1"

        echo -e "${IYellow}${str}${NC}"
    }

    local -r log_file="/tmp/${_prog_name}.$$"
    exec &> >(tee -a "${log_file}")

    local iwyu_before_changes_output_file="/tmp/iwyu_before.out"  # default tmp build output file before change
    local iwyu_after_changes_output_file="/tmp/iwyu_after.out"    # default tmp build output file after change
    local build=true  # by default will call build cmd
    local build_release="-r"  # by default use release mode build cmd
    local clean=false
    local stash=false
    local commits=""  # 2 sha values separated by comma/space
    local head="" # num of commits to switch HEAD~<num>
    local build_cmd="build_hcl" # default build command

    #
    # Read command line options
    local -r args=$( getopt -l "verbose,stash,nobuild,debug,clean,head:,commits:,before-file:,after-file:,build-cmd:,help" -o "vdch" -- "$@" 2>/dev/null ||  echo error  )
    [[ "${args}" == *error ]] && { _print_error "Invalid options!"; _usage; return 1; }
    eval set -- "${args}"

    while [ $# -ge 1 ]; do
        case "$1" in
            --)
                # No more options left.
                shift
                break
                ;;
            -v|--verbose)
                verbose_output=true
                ;;
            -d|--debug)
                build_release=""
                ;;
            --nobuild)
                build=false
                ;;
            -c|--clean)
                clean=true
                ;;
            -s|--stash)
                stash=true
                ;;
            --head)
                head="$2"
                shift
                ;;
            --commits)
                commits="$2"
                shift
                ;;
            --before-file)
                iwyu_before_changes_output_file="$2"
                shift
                ;;
            --after-file)
                iwyu_after_changes_output_file="$2"
                shift
                ;;
            --build-cmd)
                build_cmd="$2"
                shift
                ;;
            -h|--help)
                _usage
                exit 0
                ;;
            *)
                # invalid options
                _usage
                return 1
                ;;
        esac
        shift
    done

    local -r iwyu_start_error_prefix=" should add these lines:"
    local -r iwyu_remove_error_prefix=" should remove these lines:"
    local -r iwyu_end_error_prefix="---"
    local -r iwyu_full_list_prefix="The full include-list for "
    local -r iwyu_correct_prefix="has correct #includes"

    #
    # Validate we run from hcl root dir
    env | egrep -q HABANA_SOFTWARE_STACK || { _print_error "Habana env not set"; return 1; }
    # source "${HABANA_SOFTWARE_STACK}/automation/habana_scripts/habana_env" || { _print_error "Habana env not set"; return 1; }
    local -r cwd=$( pwd )
    local -r git_root_dir=$( git rev-parse --show-toplevel )
    [ "${cwd}" ==  "${git_root_dir}" ] || { _print_error "You must be in git root dir to run this script"; return 1; }

    #
    # Validate colordiff tool is installed
    _print_info "Updating & Installing colordiff tool"
    colordiff --version  &>/dev/null    # check if its already installed
    if [ $? -ne  0 ]; then
        sudo apt-get update -y  &>/dev/null && sudo apt-get install -y colordiff  &>/dev/null || { _print_error "Cannot install colordiff tool"; return 1; }
        colordiff --version  &>/dev/null ||  { _print_error "colordiff tool not installed"; return 1; }
    fi

    # Validate mutual exclusive options
    ! ${build} && ${clean} && { _usage; _print_error "clean only valid with build"; return 1; }
    ! ${build} && ${stash} && { _usage;  _print_error "stash only valid with build"; return 1; }
    ! ${build} && [ -n "${head}" ] && { _usage; _print_error "git head only valid with build"; return 1; }
    ! ${build} && [ -n "${commits}" ] && { _usage; _print_error "git commits only valid with build"; return 1; }

    ${build} && ${stash} &&  [[ -n "${head}" || -n "${commits}" ]] && { _usage; _print_error "Cannot do build with stash and git checkout"; return 1; }
    ${build} && [[ -n "${head}" && -n "${commits}" ]] && { _usage; _print_error "Cannot do build with git checkout head and commits options"; return 1; }
    ${build} && ! ${stash} && [[ -z "${head}" && -z "${commits}" ]] && { _usage; _print_error "Cannot do build without stash/head/commits options option"; return 1; }

    #
    # validate current git branch name (required to restore checkout branch)
    local head_git_branch_name=""
    if [[ -n "${head}" || -n "${commits}" ]]; then
        head_git_branch_name=$( git rev-parse --abbrev-ref HEAD )
        [ -z "${head_git_branch_name}" ] && { _print_error "Cannot read HEAD branch name, exiting"; return 1; }
        _print_info "Detected git HEAD branch name to be ${head_git_branch_name}"
    fi

    #
    # Validate head parameter
    local -i head_num=0
    if [ -n "${head}" ]; then
        head_num=${head} &>/dev/null # try to convert to num, ignore errors
        [ ${head_num} -lt 1 ] && { _usage; _print_error "--head must be followed by a positive integer"; return 1; }
    fi

    #
    # Validate commits hashes parameter
    local commit_before=""
    local commit_after=""
    if [ -n "${commits}" ]; then
        IFS=',' read -ra hashes <<< "${commits}"
        local -ir hashes_count=${#hashes[@]}
        [ ${hashes_count} -ne 2 ] && { _usage; _print_error "--commits should contain 2 commit hashes separated by a comma, e.g. ce8a6679,866a60f6"; return 1; }
        commit_before="${hashes[0]}"
        commit_after="${hashes[1]}"
        git branch "${head_git_branch_name}"  -a --contains "${commit_before}" &>/dev/null || { _usage; _print_error "${commit_before} is not a valid commit hash in current branch ${head_git_branch_name}, --commits must be followed by two commit hashes separated by comma"; return 1; }
        git branch "${head_git_branch_name}"  -a --contains "${commit_after}" &>/dev/null || { _usage; _print_error "${commit_after} is not a valid commit hash in current branch ${head_git_branch_name}, --commits must be followed by two commit hashes separated by comma"; return 1; }
    fi

    # Valid IWYU input files before/after, whether we build or not
    [ -z "${iwyu_before_changes_output_file}" ] && { _usage; _print_error "Before change file name cannot be empty"; return 1; }
    realpath -q "${iwyu_before_changes_output_file}" &>/dev/null || { _usage; _print_error "Must provide valid before change file name string" ; return 1; }
    [ -z "${iwyu_after_changes_output_file}" ] && { _usage; _print_error "After change file name cannot be empty"; return 1; }
    realpath -q "${iwyu_after_changes_output_file}" &>/dev/null || { _usage; _print_error "Must provide valid after change file name string" ; return 1; }

    #
    # get IWYU output before and after the change
    #
    if ${build}; then
        local -r iwyu_path_local="/home/linuxbrew/.linuxbrew/bin"
        local build_clean=""
        ${clean} && build_clean="-c"

        # check build cmd is working
        [[ -z "${SYNAPSE_ROOT}" || ! -d "${SYNAPSE_ROOT}" ]] && { _print_error "synapse root dir ${SYNAPSE_ROOT} not set or valid"; return 1; }
        source "${SYNAPSE_ROOT}/.ci/scripts/synapse.sh" || { _print_error "Cannot source synapse scripts file"; return 1; }
        "${build_cmd}" --help &>/dev/null ||  { _print_error "${build_cmd} not working"; return 1; }

        # check IWYU is working
        _print_info "Checking IWYU tool is installed"
        ${iwyu_path_local}/include-what-you-use --version || { _print_error "IWYU not working, exiting"; return 1; }

        # stash case - we first revert (push) and build, then we restore the change and build again
        if ${stash}; then
            _print_info "Doing git stash push"
            git stash push -m "${_prog_name} temporary stash" &>/dev/null || { _print_error "Cannot stash files, exiting"; return 1; }
            _print_info "Running build before change..."
            IWYU="YES" IWYU_PATH="${iwyu_path_local}" "${build_cmd}" "${build_clean}" "${build_release}" &>"${iwyu_before_changes_output_file}" || { _print_warning "${build_cmd} did not end successfully after stash push"; }
            _print_info "Doing git stash pop"
            git stash pop &>/dev/null || { _print_error "Cannot unstash files, exiting"; return 1; }
            _print_info "Running build after change..."
            IWYU="YES" IWYU_PATH="${iwyu_path_local}" "${build_cmd}" "${build_clean}" "${build_release}" &>"${iwyu_after_changes_output_file}" || { _print_warning "${build_cmd} did not end successfully after stash pop"; }
        elif [ -n "${head}" ]; then
            _print_info "Doing git checkout HEAD~${head_num} ..."
            git checkout "HEAD~${head_num}" &>/dev/null || { _print_error "Cannot checkout HEAD~${head_num}, exiting"; return 1; }
            _print_info "Running build before change..."
            IWYU="YES" IWYU_PATH="${iwyu_path_local}" "${build_cmd}" "${build_clean}" "${build_release}" &>"${iwyu_before_changes_output_file}" || { _print_warning "${build_cmd} did not end successfully after git checkout before"; }
            _print_info "Doing git checkout ${head_git_branch_name} ..."
            git checkout "${head_git_branch_name}" &>/dev/null || { _print_error "Cannot checkout back to ${head_git_branch_name}, exiting"; return 1; }
            _print_info "Running build after change..."
            IWYU="YES" IWYU_PATH="${iwyu_path_local}" "${build_cmd}" "${build_clean}" "${build_release}" &>"${iwyu_after_changes_output_file}" || { _print_warning "${build_cmd} did not end successfully after git checkout after"; }
        elif [ -n "${commits}" ]; then
            _print_info "Doing git checkout ${commit_before} ..."
            git checkout "${commit_before}" &>/dev/null || { _print_error "Cannot checkout ${commit_before}, exiting"; return 1; }
            _print_info "Running build before change..."
            IWYU="YES" IWYU_PATH="${iwyu_path_local}" "${build_cmd}" "${build_clean}" "${build_release}" &>"${iwyu_before_changes_output_file}" || { _print_warning "${build_cmd} did not end successfully after git checkout before"; }
            _print_info "Doing git checkout ${commit_after} ..."
            git checkout "${commit_after}" &>/dev/null || { _print_error "Cannot checkout ${commit_after}, exiting"; return 1; }
            _print_info "Running build after change..."
            IWYU="YES" IWYU_PATH="${iwyu_path_local}" "${build_cmd}" "${build_clean}" "${build_release}" &>"${iwyu_after_changes_output_file}" || { _print_warning "${build_cmd} did not end successfully after git checkout after"; }
            _print_info "Doing git checkout ${head_git_branch_name} ..."
            git checkout "${head_git_branch_name}" &>/dev/null || { _print_error "Cannot checkout back to ${head_git_branch_name}, exiting"; return 1; }
        else
            _print_error "Unsupported operation"
            return 1
        fi
    fi

    #
    # Validate we have valid IWYU readable input files before/after, wether we build or not
    [ ! -r "${iwyu_before_changes_output_file}" ] && { _print_error "Before change file ${iwyu_before_changes_output_file} cannot be read"; return 1; }
    [ ! -r "${iwyu_after_changes_output_file}" ] && { _print_error "After changes file ${iwyu_after_changes_output_file} cannot be read"; return 1; }

    #
    # extract only  IWYU errors related lines
    local -r iwyu_before_error_lines=$( sed -n "/^.*${iwyu_start_error_prefix}/, /^${iwyu_end_error_prefix}/p" "${iwyu_before_changes_output_file}" )
    local -r iwyu_after_error_lines=$( sed -n "/^.*${iwyu_start_error_prefix}/, /^${iwyu_end_error_prefix}/p" "${iwyu_after_changes_output_file}" )

    if [ -z "${iwyu_before_error_lines}" ]; then
        _print_error "No IWYU output lines before change, check file ${iwyu_before_changes_output_file}"
        return 1
    fi

    if [ -z "${iwyu_after_error_lines}" ]; then
        _print_error "No IWYU output lines after change, check file ${iwyu_after_changes_output_file}"
        return 1
    fi

    #
    # extract only correct files related lines
    local -r iwyu_before_correct_lines=$( sed -n "/^.*${iwyu_correct_prefix}.*$/p" "${iwyu_before_changes_output_file}" )
    local -r iwyu_after_correct_lines=$( sed -n "/^.*${iwyu_correct_prefix}.*$/p" "${iwyu_after_changes_output_file}" )

    #
    # Get list of file names reported by IWYU before and after change
    local -ar bfiles=$( echo "${iwyu_before_error_lines}" | sed -n "s/^\(.*\)\(${iwyu_start_error_prefix}\)/\1/p" | sort -u )
    local -ar afiles=$( echo "${iwyu_after_error_lines}" | sed -n "s/^\(.*\)\(${iwyu_start_error_prefix}\)/\1/p" | sort -u )

    # check arrays not empty
    local -ir before_files_len=${#bfiles[@]}
    if [ ${before_files_len} -eq 0 ]; then
        _print_error "No IWYU files detected before change, check file ${iwyu_before_changes_output_file}"
        return 1
    fi

    local -ir after_files_len=${#afiles[@]}
    if [ ${after_files_len} -eq 0 ]; then
        _print_error "No IWYU files detected after change, check file ${iwyu_after_changes_output_file}"
        return 1
    fi

    #
    # build a list of files that were processed
    local checked_files=""

    _print_info "Start to process IWYU diff between builds"

    local -i count_files=0
    local -ir print_count=10

    _print_info "Checking before change files"

    #
    # check for each files before if it has changed IWYU error content
    for bf in ${bfiles[@]}; do
        _print_verbose "Processing file ${bf}"
        let count_files++
        [ $(( ${count_files} % ${print_count} )) -eq 0 ] && { _print_info "Processed ${count_files} files..."; }
        local escaped=$( echo "${bf}" | sed 's/\//\\\//g' )
        local before_content=$( echo "${iwyu_before_error_lines}" | sed -n "/^${escaped}.*${iwyu_start_error_prefix}/, /^${iwyu_end_error_prefix}/p" )
        if [ -z "${before_content}" ]; then
            _print_warning "No IWYU output lines before change for file ${bf}, check file ${iwyu_before_changes_output_file}"
        else
            local after_content=$( echo "${iwyu_after_error_lines}" | sed -n "/^${escaped}.*${iwyu_start_error_prefix}/, /^${iwyu_end_error_prefix}/p" )
            if [ -n "${after_content}" ]; then
                # there is content before and after
                checked_files="${bf} ${checked_files}"

                local multiple_reports=false
                # There might be more than once occurrence of the IWYU report per file - we can only process single occurrence
                local -i before_repeats=$( echo "${before_content}"  | egrep -c "${iwyu_full_list_prefix}" )
                if [ ${before_repeats} -eq 0 ]; then
                    _print_warning "Error processing IWYU output lines before change for file ${bf}, check file ${iwyu_before_changes_output_file}"
                    continue
                elif [ ${before_repeats} -ge 2 ]; then
                    _print_verbose "IWYU output lines before change for file ${bf} has multiple sections, check file ${iwyu_before_changes_output_file}"
                    multiple_reports=true
                fi

                # There might be more than once occurrence of the IWYU report per file - we can only process single occurrence
                local -i after_repeats=$( echo "${after_content}"  | egrep -c "${iwyu_full_list_prefix}" )
                if [ ${after_repeats} -eq 0 ]; then
                    _print_warning "Error processing IWYU output lines after change for file ${bf}, check file ${iwyu_after_changes_output_file}"
                    continue
                elif [ ${after_repeats} -ge 2 ]; then
                    _print_verbose "IWYU output lines after change for file ${bf} has multiple sections, check file ${iwyu_after_changes_output_file}"
                    multiple_reports=true
                fi

                local diff=""
                local -i diff_res=0

                # remove text after C++ comments //, sort lines
                local before_add_section=$( echo "${before_content}" | sed "/^${escaped}.*${iwyu_start_error_prefix}/,/^${escaped}.*${iwyu_remove_error_prefix}/!d" | sed  's/^\(.*\)\/\/.*$/\1\/\//g' | sort -u )
                local before_remove_section=$( echo "${before_content}" | sed "/^${escaped}.*${iwyu_remove_error_prefix}/,/^${iwyu_full_list_prefix}.*/!d" | sed  's/^\(.*\)\/\/.*$/\1\/\//g' | sort -u )
                local before_full_section=$( echo "${before_content}" | sed "/^${iwyu_full_list_prefix}.*/,/^${iwyu_full_list_prefix}/!d" | sed  's/^\(.*\)\/\/.*$/\1\/\//g' | sort -u )
                # local before_sorted_section="${before_add_section}\n${before_remove_section}\n${before_full_section}"
                local before_sorted_section="${before_add_section}\n${before_remove_section}\n"

                local after_add_section=$( echo "${after_content}" | sed "/^${escaped}.*${iwyu_start_error_prefix}/,/^${escaped}.*${iwyu_remove_error_prefix}/!d" | sed  's/^\(.*\)\/\/.*$/\1\/\//g' | sort -u )
                local after_remove_section=$( echo "${after_content}" | sed "/^${escaped}.*${iwyu_remove_error_prefix}/,/^${iwyu_full_list_prefix}.*/!d" | sed  's/^\(.*\)\/\/.*$/\1\/\//g' | sort -u )
                local after_full_section=$( echo "${after_content}" | sed "/^${iwyu_full_list_prefix}.*/,/^${iwyu_full_list_prefix}/!d" | sed  's/^\(.*\)\/\/.*$/\1\/\//g' | sort -u )
                # local after_sorted_section="${after_add_section}\n${after_remove_section}\n${after_full_section}"
                local after_sorted_section="${after_add_section}\n${after_remove_section}\n"

                # diff the sorted sections of the output
                diff=$( colordiff --width=170 --ignore-tab-expansion --ignore-blank-lines --ignore-all-space --ignore-trailing-space --side-by-side <( echo -n "${before_sorted_section}" ) <( echo -n "${after_sorted_section}" ) 2>/dev/null  )
                diff_res=$?

                if [ ${diff_res} -ne 0 ]; then  # not identical
                    _print_warning "File ${bf} IWYU output is different after change:"
                    _print_info "============================================================================================"
                    _print_info "${diff}"
                    _print_info "============================================================================================"
                    _print_info ""
                    if ${multiple_reports}; then
                        _print_warning "IWYU output lines before or after change for file ${bf} has multiple sections, check file ${iwyu_before_changes_output_file} & ${iwyu_after_changes_output_file} again"
                    fi
                else
                    _print_verbose_ok "File ${bf} has the same IWYU output after change"
                fi
            else
                _print_verbose "File ${bf} does not appear in IWYU after change"
                was_corrected=$( egrep -q "${b}" <<< "${iwyu_after_correct_lines}" )
                if [ -n "${was_corrected}" ]; then
                    _print_verbose_ok "File ${bf} was corrected in IWYU after change"
                else
                    # normally would not happen, unless file was deleted
                    _print_warning "File ${bf} does not appear in IWYU after change, check IWYU output in case it was not deleted after change"
                fi
                # TODO" check if it was corrected
            fi
        fi
    done

    #
    # check for each files after if it has changed IWYU error content, if it was not processed before

    local new_files=""

    _print_verbose "checked_files=${checked_files}"

    _print_info "Checking after change files"

    count_files=0

    for af in ${afiles[@]}; do

        let count_files++
        [ $(( ${count_files} % ${print_count} )) -eq 0 ] && { _print_info "Processed ${count_files} files..."; }

        local re_file="\\${af}\\b"
        if [[ ${checked_files} =~ ${re_file} ]]; then
            # file was already process in before section
            _print_verbose "${af} was processed before"
            continue
        fi
        _print_verbose_ok "Processing after file ${af}"

        local escaped=$( echo "${af}" | sed 's/\//\\\//g' )
        local after_content=$( echo "${iwyu_after_error_lines}" | sed -n "/^${escaped}.*${iwyu_start_error_prefix}/, /^${iwyu_end_error_prefix}/p" )

        new_files="${af} ${new_files}"

        _print_warning "File ${af} WING output was not before change or was correct, check report:"
        _print_info "============================================================================================"
        _print_info "${after_content}"
        _print_info "============================================================================================"
        _print_info ""

    done

    _print_info_ok "Done, log was also saved in ${log_file}"

)
