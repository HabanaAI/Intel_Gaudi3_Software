import os
import argparse
import syn_infra
import logging
from typing import Dict, List, Tuple
from enum import Enum

LOG = syn_infra.config_logger("syn_tests", logging.INFO)

def parse_args():
    dep_parser = argparse.ArgumentParser(add_help=False)
    executable = dep_parser.add_mutually_exclusive_group()
    executable.add_argument(
        "-r", "--release", action="store_true", help="Run release binary"
    )
    executable.add_argument(
        "--sanitizer", action="store_true", help="Run synapse sanitizer build tests"
    )

    dep_args, _ = dep_parser.parse_known_args()

    parser = argparse.ArgumentParser(parents=[dep_parser])
    parser.add_argument(
        "-spdlog",
        "--spdlog",
        type=int,
        choices=range(7),
        help="0 - TRACE, 1 - DEBUG, 2 - INFO, 3 - WARNING, 4 - ERROR, 5 - CRITICAL, 6 - OFF",
    )
    parser.add_argument(
        "-shuffle",
        "--shuffle",
        action="store_true",
        help="Randomize tests order",
    )
    parser.add_argument(
        "-seed",
        "--seed",
        type=int,
        help="Tests order seed",
        default=0,
    )
    parser.add_argument(
        "-prof",
        "--prof",
        action="store_true",
        help="Capture synapse profiler trace",
    )
    parser.add_argument(
        "-i",
        "--iterations",
        type=int,
        help="Run iterations",
        default=1,
    )
    parser.add_argument(
        "-l",
        "--list-tests",
        action="store_true",
        help="List the available tests",
    )
    parser.add_argument(
        "-f",
        "--run-until-failure",
        action="store_true",
        help="Run tests until first failure",
    )
    parser.add_argument(
        "-mr",
        "--media-recipes",
        action="store_true",
        help="Dumps recipes for media in local directory",
    )
    parser.add_argument("-s", "--specific-test",
                        help="Run only the specified test")
    parser.add_argument(
        "-a",
        "--skip-additional-tests",
        help="Run all tests without the specified test and any default skipped tests",
    )
    parser.add_argument(
        "-c",
        "--chip_type",
        choices=list(syn_infra.CHIP_TYPES.keys()),
        help="Select chip type",
    )
    parser.add_argument(
        "-n",
        "--num_of_devices",
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "-d",
        "--disabled-test",
        action="store_true",
        help="Also run disabled tests",
    )
    parser.add_argument("-x", "--xml", help="Output XML file path")
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colors in output",
    )
    parser.add_argument(
        "--death-test",
        action="store_true",
        help="Run RT device failure tests",
    )
    parser.add_argument(
        "--eager",
        action="store_true",
        help="Run only eager mode tests",
    )
    parser.add_argument(
        "--test-packages",
        nargs="+",
        help=f"Run test packages",
    )
    parser.add_argument(
        "--ex-packages",
        nargs="+",
        help=f"Exclude test packages",
    )
    syn_build = (
        os.getenv("SYNAPSE_RELEASE_BUILD")
        if dep_args.release
        else os.getenv("SYNAPSE_DEBUG_SANITIZER_BUILD")
        if dep_args.sanitizer
        else os.getenv("SYNAPSE_DEBUG_BUILD")
    )
    parser.add_argument(
        "--syn-tests-bin",
        help="Full path to synapse tests binary",
        default=os.path.join(syn_build, "bin", "syn_tests")
        if syn_build
        else None,
    )
    parser.add_argument(
        "--rename-report-file",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "-g",
        "--gdbserver" ,
        action="store_true",
        help="Run the app with gdbserver"
    )
    parser.add_argument(
        "-gdb",
        "--gdb" ,
        action="store_true",
        help="Run the app under gdb"
    )
    parser.add_argument(
        "-v",
        "--valgrind" ,
        action="store_true",
        help="Run the app under valgrind"
    )
    parser.add_argument(
        "-m",
        "--mode" ,
        choices=list(["all", "asic", "asic-ci", "sim", "daily", "postCommit"]),
        help="Run the app in a specific mode: all/asic/sim/daily/postCommit"
    )
    args, unknown = parser.parse_known_args()
    print("Ignored arguments: " + str(unknown))
    if args.xml and args.rename_report_file:
        args.xml = f'{args.xml.replace("xml", "syn")}.xml'
    return args


def build_env(args) -> Tuple[Dict[str, str], List[str]]:
    curr_env: Dict[str, str] = {}

    if args.spdlog is not None:
        curr_env["LOG_LEVEL_ALL"] = str(args.spdlog)

    if args.prof:
        curr_env["HABANA_PROFILE"] = "1"

    curr_env["MEDIA_RECIPES"] = "true" if args.media_recipes else "false"

    __debug_tool = ""
    __debug_param = ""

    if args.gdbserver:
        __debug_tool  = "gdbserver localhost:2345"
        __debug_param = "gdbserver"
    if args.gdb:
        if __debug_tool != "":
            print("Cannot support \"gdb\" arg while \"" + __debug_param + "\" is set\n")
            exit(0)
        __debug_tool  = "gdb --args"
        __debug_param = "gdb"
    if args.valgrind:
        if __debug_tool != "":
            print("Cannot support \"valgrind\" arg while \"" + __debug_param + "\" is set\n")
            exit(0)
        __debug_tool  = "valgrind"
        __debug_param = "valgrind"

    cmd_args: List[str] = [__debug_tool + " " + args.syn_tests_bin]

    if args.chip_type:
        cmd_args += ["--device-type", args.chip_type]

    # Included packages
    packagesSpecified = False
    package_list = "--test-packages"

    if args.test_packages:
        package_list += f'{" " + " ".join(args.test_packages)}'
        packagesSpecified = True
    elif args.death_test:
        # Defined so in case of an empty list of test-packages, it will not execute all but just this package
        package_list += f'{" DEATH"}'
        packagesSpecified = True

    if args.mode:
        if args.mode == "postCommit":
            package_list += f'{" SIM"}'
        else:
            if args.mode.upper() not in package_list:
                package_list += f'{" "+args.mode.upper()}'
        packagesSpecified = True

    if (packagesSpecified):
        cmd_args.append(f"{package_list}")

    # Excluded packages
    isExcludedPackages = False
    excluded_package_list = "--ex-packages"

    if not args.death_test:
        excluded_package_list += f'{" DEATH "}'
        isExcludedPackages = True

    if args.ex_packages:
        excluded_package_list += f'{" ".join(args.ex_packages)}'
        isExcludedPackages = True

    if isExcludedPackages:
        cmd_args.append(f"{excluded_package_list}")

    # Filtered tests
    isFilterNeeded = False
    filter         = "--gtest_filter="

    if args.specific_test:
        filter += f'{args.specific_test}'
        isFilterNeeded = True

    if (isFilterNeeded):
        cmd_args.append(f"{filter}")

    cmd_args.append(f"--gtest_random_seed={args.seed}")

    if args.xml:
        cmd_args.append(f'--gtest_output="xml:{args.xml}"')
    if args.iterations:
        cmd_args.append(f"--gtest_repeat={args.iterations}")
    if args.shuffle:
        cmd_args.append(f"--gtest_shuffle")
    if args.run_until_failure:
        cmd_args.append(f"--gtest_break_on_failure")
    if args.disabled_test:
        cmd_args.append(f"--gtest_also_run_disabled_tests")
    if args.list_tests:
        cmd_args.append(f"--gtest_list_tests")
    cmd_args.append(f'--gtest_color={"no" if args.no_color else "yes"}')

    return cmd_args, curr_env


def run(cmd_args: List[str], curr_env: Dict[str, str]):
    LOG.info(f"Running command:")
    LOG.info(
        " ".join(f"{k}={v}" for k, v in curr_env.items()) +
        " " + " ".join(cmd_args)
    )
    return syn_infra.run_external(cmd_args, curr_env)

if __name__ == "__main__":
    args = parse_args()
    cmd_args, curr_env = build_env(args)

    exit(run(cmd_args, curr_env))
