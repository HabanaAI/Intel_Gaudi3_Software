#/bin/tcsh
rm -rf CMakeCache.txt
source $VERIF/mme_verif/mme_sim/scripts/set_build_env.sh
cmake -DUSE_TPCSIM=no -DUSE_ARMCP=no $VERIF/mme_verif/mme_sim
make clean all -j 12
