#!/usr/bin/env bash

(
  __eager_networks=( bert_ft bert_p1 bert_p2 lamma7B resnet unet2d unet3d )

  __tcmalloc_path=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4
  if [ ! -f "${__tcmalloc_path}" ]; then
    sudo apt install -y libtcmalloc-minimal4
    if [ ! -f "${__tcmalloc_path}" ]; then
      echo "Missing tcmalloc, please install it and add a symbolic link from ${__tcmalloc_path}"
      return 1
    fi
  fi

  export ENABLE_EXPERIMENTAL_FLAGS=1
  export FORCE_EAGER=1
  export LD_PRELOAD="${__tcmalloc_path}"

  for network in "${__eager_networks[@]}"; do
    time run_from_json --compilation_mode eager --keep_going --mt_perf --test_iters 20 -c gaudi2 -j "/git_lfs/data/synapse/tests/eager/benchmark_models/latest.${network}.json" -r |& tee "par.$network.em.txt"
  done
)

# Collect results from all par_<network>[_<optional_feat>].txt log files,
# throw away the bot and top 10% and print out avg results per network per feature.
python3 par_summarize.py