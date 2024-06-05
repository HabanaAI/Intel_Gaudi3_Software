#!/usr/bin/env python
import argparse
import logging
import os
import syn_infra as syn
from typing import Dict, List, Tuple


DEFAULT_DATA_COMPARATOR_CONFIG_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "default_data_comparator_config.json"
)
LOG = syn.config_logger("syn_tests", logging.INFO)

COMPILATION_MODE = {0: "graph", 1: "eager", 2: "any"}

PACKAGES_IDS = {
    0: "TEST_PACKAGE_DEFAULT",
    1: "TEST_PACKAGE_CONV_PACKING",
    2: "TEST_PACKAGE_AUTOGEN",
    3: "TEST_PACKAGE_NODE_OPERATIONS",
    4: "TEST_PACKAGE_RT_API",
    5: "TEST_PACKAGE_BROADCAST",
    6: "TEST_PACKAGE_COMPARE_TEST",
    7: "TEST_PACKAGE_GEMM",
    8: "TEST_PACKAGE_DMA",
    9: "TEST_PACKAGE_TRANSPOSE",
    10: "TEST_PACKAGE_CONVOLUTION",
    11: "TEST_PACKAGE_SRAM_SLICING",
    12: "TEST_PACKAGE_DSD",
    13: "TEST_PACKAGE_EAGER",
}

GROUP_IDS = {
    0: [0, 10],
    1: [6, 2],
    2: [8, 9, 11],
    3: [3, 7, 4],
    4: [1, 5],
    5: [12],
    6: [13],
}

CHIP_FILTERS_INC = {
    "gaudi": ["SynTraining", "gaudi1", "GAUDI", "SynGaudi"],
    "gaudi2": [
        "SynTraining",
        "gaudi2",
        "SynScal",
        "GAUDI",
        "GAUDI2",
        "SynGaudi",
        "SynGaudi2",
    ],
    "gaudi3": ["SynGaudi", "SynTraining", "gaudi3", "SynScal", "GAUDI"],
    "goya2": ["SynCommon", "SynAPITests", "SynGoya2", "GOYA2", "SynGreco", "GRECO"],
    "greco": ["GOYA2", "GRECO", "SynGoya2"],
}
CHIP_FILTERS_EXC = {
    "greco": [],
    "gaudi": ["SynGaudi2", "GAUDI2", "gaudi2", "gaudi3"],
    "gaudi2": ["gaudi1", "gaudi3", "eventCreateDestroy"],
    "gaudi3": ["gaudi1", "gaudi2", "full_fwd_bwd", "SynGaudi*ASIC", "ASIC*SynGaudi"], # yikes
    "goya": [
        "SynCommon",
        "SynGaudi",
        "GAUDI",
        "SynGoya2",
        "GOYA2",
        "SynGreco",
        "GRECO",
    ],
}

FRONT_MODE_FILTERS = {"asic": ["ASIC"], "asic-ci": ["ASIC_CI"]}

BACK_MODE_FILTERS = {
    "asic": ["_ASIC"],
    "asic-ci": ["_ASIC_CI"],
    "daily": ["_DAILY"],
    "postCommit": ["_L2"],
}

GLOBAL_MODE_FILTERS_INC = {
    "all": [
        "SynGaudi",
        "SynAPITests",
    ],
    "asic": ["SynAPITests"],
    "asic-ci": ["SynAPITests"],
    "sim": ["SynCommon", "SynAPITests"],
    "daily": ["_DAILY"],
    "postCommit": ["SynGaudi", "SynAPITests"],
}

GLOBAL_MODE_FILTERS_EXC = {
    "all": ["test_hcl", "SynGaudiNCCL", "DEATH_TEST", "MultipleThreads"],
    "asic": ["_DAILY", "test_hcl", "SynGaudiNCCL", "DEATH_TEST"],
    "asic-ci": ["_DAILY","test_hcl", "SynGaudiNCCL", "DEATH_TEST","_L2"],
    "sim": ["ASIC","ASIC_CI", "_DAILY", "_L2", "test_hcl", "SynGaudiNCCL", "DEATH_TEST"],
    "daily": ["test_hcl", "SynGaudiNCCL", "DEATH_TEST"],
    "postCommit": ["test_hcl", "SynGaudiNCCL", "DEATH_TEST"],
}


def get_group_id_info():
    ret = ""
    for k, v in GROUP_IDS.items():
        ret += f"{k}: {[PACKAGES_IDS.get(i) for i in v]}, "
    return ret


def build_filter(args) -> str:
    if args.media_recipes:  # yuck
        return "*ResnetSynGoya2Tests.resnet50_int8_full_fixed_goya2*:*ResnetSynGrecoTests.resnet50_int8_full_fixed_goya2*"
    inc_list = []
    mode_inc = [v for v in GLOBAL_MODE_FILTERS_INC.get(args.mode)]
    chip_inc = (
        [v for v in CHIP_FILTERS_INC.get(args.chip_type)] if args.chip_type else []
    )
    back_mode_inc = BACK_MODE_FILTERS.get(args.mode)
    if back_mode_inc:
        for m in back_mode_inc:
            for c in chip_inc:
                inc_list.append(f"{c}*{m}")
            for mi in mode_inc:
                inc_list.append(f"{mi}*{m}")
    front_mode_inc = FRONT_MODE_FILTERS.get(args.mode)
    if front_mode_inc:
        for m in front_mode_inc:
            for c in chip_inc:
                inc_list.append(f"{m}*{c}")
            for mi in mode_inc:
                inc_list.append(f"{m}*{mi}")
    if not inc_list:
        inc_list = mode_inc + chip_inc
    chip_exc = (
        [v for v in CHIP_FILTERS_EXC.get(args.chip_type)] if args.chip_type else []
    )
    exc_list = chip_exc + GLOBAL_MODE_FILTERS_EXC.get(args.mode)
    if args.skip_additional_tests:
        exc_list += [args.skip_additional_tests]
    if args.chip_type == "gaudi3" and "ASIC" in exc_list:  # yuck
        exc_list.remove("ASIC")
    if args.mode == "daily":  # yuck
        inc_list = GLOBAL_MODE_FILTERS_INC.get(args.mode)
    inc_list = [f"*{v}*" for v in inc_list]
    exc_list = [f"*{v}*" for v in exc_list]

    ret = (
        f'{":".join(inc_list)}:-{":".join(exc_list)}'
        if inc_list
        else f'-{":".join(exc_list)}'
    )
    return ret


def build_env(args) -> Tuple[Dict[str, str], List[str]]:
    curr_env: Dict[str, str] = {}

    if args.sanitizer:
        curr_env["ASAN_OPTIONS"] = "use_sigaltstack=0"

    if args.spdlog is not None:
        curr_env["LOG_LEVEL_ALL"] = str(args.spdlog)

    if args.prof:
        curr_env["HABANA_PROFILE"] = "1"

    curr_env["MEDIA_RECIPES"] = "true" if args.media_recipes else "false"

    cmd_args: List[str] = [args.syn_tests_bin]

    if args.chip_type:
        cmd_args += ["--device-type", args.chip_type]

    if args.eager:
        cmd_args += ["--compilation-mode", "eager"]

    if args.compilation_mode:
        cmd_args += ["--compilation-mode", COMPILATION_MODE.get(args.compilation_mode)]

    packages = (
        args.test_packages or GROUP_IDS.get(args.test_group_id)
        if args.test_group_id is not None
        else None
    )

    if (
        os.environ.get("TESTS_PACKAGES") is not None
        and os.environ.get("TESTS_PACKAGES") != ""
    ):
        LOG.info(f"using test packages from TESTS_PACKAGES env var")
        packages = os.environ.get("TESTS_PACKAGES").split(",")

    if packages:
        cmd_args += ["--test-packages", " ".join([str(v) for v in packages])]

    if args.specific_test:
        chip_exc = (
            [f"*{v}*" for v in CHIP_FILTERS_EXC.get(args.chip_type)]
            if args.chip_type
            else []
        )
        filter = f'{args.specific_test}'
        if args.skip_additional_tests:
            filter += f':-{args.skip_additional_tests}'
    else:
        filter = build_filter(args)

    cmd_args.append(f"--gtest_filter={filter}")
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
        " ".join(f"{k}={v}" for k, v in curr_env.items()) + " " + " ".join(cmd_args)
    )
    return syn.run_external(cmd_args, curr_env)


def main(args):
    cmd_args, curr_env = build_env(args)
    return run(cmd_args, curr_env)


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
        default=4,
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
    parser.add_argument("-s", "--specific-test", help="Run only the specified test")
    parser.add_argument(
        "-a",
        "--skip-additional-tests",
        help="Run all tests without the specified test and any default skipped tests",
    )
    parser.add_argument(
        "-c",
        "--chip_type",
        choices=list(syn.CHIP_TYPES.keys()),
        help="Select chip type",
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=list(GLOBAL_MODE_FILTERS_INC.keys()),
        help=" Run only testsfrom selected mode, default: all",
        default="all",
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
    compilation = parser.add_mutually_exclusive_group()
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
    compilation.add_argument(
        "--compilation-mode",
        type=int,
        choices=COMPILATION_MODE.keys(),
        help=f"Select compilation mode: {COMPILATION_MODE}",
        default=2
    )
    packages = parser.add_mutually_exclusive_group()
    packages.add_argument(
        "--test-packages",
        nargs="+",
        choices=PACKAGES_IDS.keys(),
        help=f"Run test packages: {PACKAGES_IDS}",
    )
    packages.add_argument(
        "--test-group-id",
        type=int,
        choices=GROUP_IDS.keys(),
        help=f"Run group of test-packages: {get_group_id_info()}",
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
        help="Full path to gc_platform_tests binary",
        default=os.path.join(syn_build, "bin", "gc_platform_tests")
        if syn_build
        else None,
    )
    parser.add_argument(
        "--rename-report-file",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    if args.xml and args.rename_report_file:
        args.xml = f'{args.xml.replace("xml", "gc")}.xml'
    if args.chip_type == "goya2":
        args.chip_type = "greco"
    return args


if __name__ == "__main__":
    args = parse_args()
    exit(main(args))
