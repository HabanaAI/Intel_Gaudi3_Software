#!/bin/bash

TARGET_SCRIPT="run_bench.sh"
if [[ -n "$1" ]]; then
  TARGET_SCRIPT="$1"
fi

# Check if target script exists
if [[ ! -f "$TARGET_SCRIPT" ]]; then
  echo "Error: Script '$TARGET_SCRIPT' does not exist."
  exit 1
fi

cleanup() {
    #restore
    min_freq=$(cat /sys/devices/system/cpu/cpu32/cpufreq/cpuinfo_min_freq)
    max_freq=$(cat /sys/devices/system/cpu/cpu32/cpufreq/cpuinfo_max_freq)
    echo "$min_freq" | sudo tee /sys/devices/system/cpu/cpu{32,96}/cpufreq/scaling_min_freq
    echo "$max_freq" | sudo tee /sys/devices/system/cpu/cpu{32,96}/cpufreq/scaling_max_freq
}

trap cleanup EXIT

max_freq=$(cat /sys/devices/system/cpu/cpu32/cpufreq/cpuinfo_max_freq)
echo "$max_freq" | sudo tee /sys/devices/system/cpu/cpu{32,96}/cpufreq/scaling_{max,min}_freq
curenv=$(declare -p -x)
sudo nice --20 sudo -u labuser taskset --cpu-list 32 bash -ic "eval '$curenv' && . $TARGET_SCRIPT"