#!/bin/bash

set -e

DRY_RUN=""
if [ -n "${HABANA_DRY_RUN}" ]; then
    DRY_RUN="--dry-run --verbose"
fi

if [ -z $1 ]; then
    _sw_stack="$HOME/trees/npu-stack"
else
    _sw_stack=$1
fi

pushd $_sw_stack/synapse
patch -f -p1 -i .cd/patches/external_synapse_package.patch --reject-file=synapse.rej --no-backup-if-mismatch $DRY_RUN
popd
