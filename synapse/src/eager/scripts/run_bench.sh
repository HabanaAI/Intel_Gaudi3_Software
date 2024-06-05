#!/usr/bin/env bash

( # subshell since we run it with '. run_bench.sh'
    __username=rrichman # cannot user $USER since it's used on both vm and 511
    __share_location="/software/users/${__username}/share/omer/"

    if true; then # remember to change to 'false' when running on 511
        (                                                                      . bench.sh --by_engine --run                |& tee log.run.em.txt    )
        (                                                                      . bench.sh --by_engine --run --graph_mode   |& tee log.run.gm.txt    )
        cp -r log.*.txt latest_rundir ${__share_location}
    else
        (                                                                      . bench.sh --by_engine --compile                |& tee log.compile.em.txt    )
        ( ENABLE_EXPERIMENTAL_FLAGS=1 ENABLE_COMPLEX_GUID_LIB_IN_EAGER=0       . bench.sh --by_engine --compile --suffix _nc   |& tee log.compile.em.nc.txt )
        sudo cp -r log.*.txt latest_rundir ${__share_location}
        sudo chown -R ${__username}:Software-SG ${__share_location}
    fi

    sudo cp summary.py ${__share_location} &&
    cd ${__share_location} && sudo python3 summary.py --em_features nosm cp du nc
    sudo chown -R ${__username}:Software-SG ${__share_location}
)
