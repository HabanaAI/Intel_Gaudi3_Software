#!/bin/bash

RED_COLOR='\033[0;31m'
NO_COLOR='\033[0m'

eager_monitor()
{
    local _xml_file=""
    while [ -n "$1" ]; do
        case $1 in
            --xml | -x)
                if [ ! -n "$2" ]; then
                    printf "${RED_COLOR}Error: xml file is missing${NO_COLOR}\n";
                    return 1;
                fi;
                _xml_file=$2;
                shift
            ;;
            esac;
            shift;
        done
    run_cmd="$SYNAPSE_ROOT/scripts/eager_kibana.py --output_xml ${_xml_file}"
    (set -x
    ( ${run_cmd} ))

    return 0
}