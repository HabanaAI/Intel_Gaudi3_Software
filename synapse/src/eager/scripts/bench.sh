#!/usr/bin/env bash

__eager_networks=( bert_ft bert_p1 bert_p2 lamma7B resnet unet2d unet3d )

function bench_large_help()
{
    echo "run bert and resnet per layer benchmarks"
    echo ""
    echo -e "Usage:"
    echo -e "\t--sanity         only run graph 0 from each network as a sanity check"
    echo -e "\t--graph_mode     run in GRAPH-MODE instead of the default EAGER-MODE"
    echo -e "\t--by_engine      dump post graph and include in summary csv"
    echo -e "\t--summarize      postprocess only"
    echo -e "\t--suffix SUFFIX  optional per-test folder suffix"
    echo -e "\t--gaudi[2|3]     test with gaudi2 or gaudi3 (Cannot use both; defaults to 2 if unspecified)"
    echo ""
    echo -e "Selection of target:"
    echo -e "\t--all                                compile and run all networks"
    echo -e "\t--compile                            compile all networks"
    echo -e "\t--run                                run all networks"
    echo -e "\t--select <compile|run>_<network>     regex pattern matching supported networks to compile or run"
    echo -e "\t                                     eg. \"--select '.*_unet.*'\" to compile and run unet[2,3]d."
    echo -e "\tSupported networks: ${__eager_networks[*]}"
}

__sanity=""
__graph_mode=false
__by_engine=false
__summarize=false
__suffix=""

__use_gaudi2=false
__use_gaudi3=false

__selected_pattern=""

while [ $# -gt 0 ]; do
    case "$1" in
        --sanity                ) __sanity="-g 0" ;;
        --graph_mode            ) __graph_mode=true ;;
        --by_engine             ) __by_engine=true ;;
        --summarize             ) __summarize=true ;;
        --suffix                )
            if [ -n "$2" ]; then
                shift && __suffix="$1"
            else
                echo "error: --suffix requires an extra argument" 1>&2;
                return 1;
            fi
            ;;

        --gaudi2                ) __use_gaudi2=true ;;
        --gaudi3                ) __use_gaudi3=true ;;

        --all     ) __selected_pattern="^.*$" ;;
        --compile ) __selected_pattern="${__selected_pattern:+${__selected_pattern}|}compile_.*" ;;
        --run     ) __selected_pattern="${__selected_pattern:+${__selected_pattern}|}run_.*" ;;
        --select  )
            if [ -n "$2" ]; then
                shift && __selected_pattern="${__selected_pattern:+${__selected_pattern}|}$1"
            else
                echo "error: --select requires an extra argument" 1>&2;
                return 1;
            fi
            ;;

        -h | --help             ) bench_large_help; return 0;;
        *                       )
            echo "error: unknown flag \"""$1""\" (See --help)" 2>&1;
            return 1
            ;;
    esac
    shift
done

# if neither is set by the user, then default to gaudi2
if [[ ${__use_gaudi2} == false ]] && [[ ${__use_gaudi3} == false ]]; then
    __use_gaudi2=true
elif [[ ${__use_gaudi2} == true ]] && [[ ${__use_gaudi3} == true ]]; then
    # TODO: temporarily dissallow both since they'd overwrite each other
    echo "error: cannot set both --use_gaudi2 and --use_gaudi3" 2>&1;
    return 1
fi

__tcmalloc_path=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4
if [ ! -f "${__tcmalloc_path}" ] ; then
    sudo apt install -y libtcmalloc-minimal4
    if [ ! -f "${__tcmalloc_path}" ] ; then
        echo "Missing tcmalloc, please install it and add a symbolic link from /usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4"
        return 1
    fi
fi

function do_bench {
    local cond="$1"
    local compile_bench="$2"
    local name="$3"
    if [[ $4 == /* ]] ; then
        local full_json="$4"    # assume already is a full path
    else
        local full_json="/git_lfs/data/synapse/tests/eager/benchmark_models/$4"
    fi
    local device_type=$5

    if [ "${compile_bench}" = true ] ; then
        if [ "${__graph_mode}" = true ] ; then
            local iters=100
        else
            local iters=1000
        fi
        local cmd="run_from_json -j ${full_json} -c ${device_type} -r --quiet --measure_syntime --keep_going ${__sanity} --test_iters=${iters}"
    else
        if [ "${__graph_mode}" = true ] ; then
            local iters=10
        else
            local iters=10
        fi
        local cmd="run_from_json -j ${full_json} -c ${device_type} -r --quiet --measure_syntime --keep_going ${__sanity} --test_iters=1 -i ${iters} --run --time_measurement profiler"
    fi

    if [ "${__graph_mode}" = true ] ; then
        name="${name}"_gm
        cmd="${cmd} --compilation_mode graph"
        local compilation_mode="graph"
    else
        cmd="${cmd} --compilation_mode eager"
        local compilation_mode="eager"
    fi
    name="${name}${__suffix}"

    if [ "${cond}" = true ] ; then
        echo "Running ${name}..."
        mkdir -p latest_rundir/"${name}"
        (
            [ "${__graph_mode}" = false ] && export ENABLE_EXPERIMENTAL_FLAGS=1 && export FORCE_EAGER=1
            cd latest_rundir/"${name}" || exit 1

            if [ "${__by_engine}" = true ] ; then
                (
                    export DUMP_POST_GRAPHS=./post_graph.json
                    LD_PRELOAD="${__tcmalloc_path}" EAGER_NOPLOT=true run_from_json -j "${full_json}" --compilation_mode "${compilation_mode}" -c "${device_type}" -r --quiet --measure_syntime --keep_going "${__sanity}" --test_iters=1 -i 1
                )
            fi

            LD_PRELOAD="${__tcmalloc_path}" EAGER_NOPLOT=true ${cmd} |& tee log.txt
        )
    fi

    local stats_fn="latest_rundir/${name}/stats.latest.json"
    if [[ ("${cond}" == "true" || "${__summarize}" == "true") && -f "${stats_fn}" ]]; then
        echo -e "\nRunning stats2csv.py for ${name}...."
        if [ "${__by_engine}" = false ]; then
            python3 ./stats2csv.py -s "${stats_fn}" --pre_graph_file "${full_json}"
        else
            python3 ./stats2csv.py -s "${stats_fn}" --pre_graph_file "${full_json}" --post_graph_file latest_rundir/"${name}"/post_graph.json
        fi
    fi
}

echo ""
echo "====================================================="
echo "The following targets will be executed in this order:"
for network in "${__eager_networks[@]}"; do
    for mode in compile run; do
        [[ -n "${__selected_pattern}" && "${mode}_${network}" =~ ${__selected_pattern} ]] && echo "- ${mode}_${network}"
    done
done
echo "====================================================="
echo ""

for network in "${__eager_networks[@]}"; do
    for mode in compile run; do
        if [[ -n "${__selected_pattern}" && "${mode}_${network}" =~ ${__selected_pattern} ]]; then
            param_cond=true
        else
            param_cond=false
        fi
        if [[ ${mode} == "compile" ]]; then
            param_is_compile=true
        else
            param_is_compile=false
        fi
        param_result_dir="${network}_${mode}_bench"
        param_pregraph="latest.${network}.json"

        [[ ${__use_gaudi2} == true ]] && do_bench "${param_cond}" "${param_is_compile}" "${param_result_dir}" "${param_pregraph}" gaudi2
        [[ ${__use_gaudi3} == true ]] && do_bench "${param_cond}" "${param_is_compile}" "${param_result_dir}" "${param_pregraph}" gaudi3
    done
done
