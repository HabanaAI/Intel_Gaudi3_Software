#!/bin/bash

# This file must be used with "source bin/activate" *from bash*
# you cannot run it directly
if [ "${BASH_SOURCE-}" = "$0" ]; then
  echo "You must source this script: $ source $0" >&2
  exit 33
fi

# Function to display usage information
usage() {
    echo "Usage: source habana_set_env_val.sh /path/to/code_dir /path/to/build_root"
    echo "  code_dir        Path to the open source code directory"
    echo "  build_root      Path to the build root directory"
}

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Error: Exactly two arguments are required." >&2
    usage
    return 1
fi

# Assign command-line arguments to variables
code_folder=$(readlink -f "$1")
build_folder=$(readlink -f "$2")

# Check if the provided directories exist
if [ ! -d "$code_folder" ]; then
    echo "Error: Code directory does not exist: $code_folder" >&2
    usage
    return 1
fi

if [ ! -d "$build_folder" ]; then
    echo "Error: Build root directory does not exist: $build_folder" >&2
    usage
    return 1
fi

# Set the path to the habana_env script
habanaEnvScript="$code_folder/automation/habana_scripts/habana_env"

# Display the paths
echo "Code folder path is: $code_folder"
echo "Build folder path is: $build_folder"

# Export PS1 for custom prompt
export PS1="\e[36m[HABANA BUILD ENV]\e[0m $PS1"

# Export the environment variable
export SET_ABSOLUTE_HABANA_ENV=1

# Source the habana_env script
source "$habanaEnvScript" "$(readlink -f "$code_folder")" "$(readlink -f "$code_folder")" "$(readlink -f "$build_folder")"
source $SWTOOLS_SDK_ROOT/.ci/scripts/build.sh
source $SYNAPSE_ROOT/.ci/scripts/synapse.sh
