#!/bin/tcsh

source $ENV_NAME/mme_verif/mme_sim/scripts/set_build_env

if( "${1}" == "clean" ) then
    /bin/rm -rf CMakeCache.txt
endif

if( -f CMakeCache.txt ) then
    make all -j 12
    set exit_status = $status
else
    cmake -DCMAKE_BUILD_TYPE=Release -DUSE_TPCSIM=no -DUSE_ARMCP=no $VERIF/mme_verif/mme_sim
    make clean all -j 12
    set exit_status = $status
endif

exit $exit_status

