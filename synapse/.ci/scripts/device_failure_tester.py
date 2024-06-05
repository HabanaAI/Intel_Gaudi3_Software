#!/usr/bin/env python
import os
import os.path
import subprocess
import time
from collections import namedtuple
from pathlib import Path
from subprocess import PIPE, run
from typing import List, Tuple

TIMEOUT_SEC = 300
SLEEP_SEC = 2
PREFIX = "   * "
STATUS = "/sys/class/accel/accel0/device/status"
DEVICE = "/sys/class/accel/accel0/device/device_type"
LOG_PATH = os.environ['HABANA_LOGS'] + "/device_failure_tester.log"

# struct-like object with 3 fields:
# dev_name_by_file  - device name as it appears in the opened file
# dev_name_by_test  - device name to pass as an argument for run_synapse_test
# tests_list_info   - list of supported death tests (test name, is old tests-infra)
DevInfo = namedtuple('DevTestInfo', ['dev_name_by_file', 'dev_name_by_test', 'tests_list_info'])

# lists of tests per device
TESTS_G1: List[str] = [
    # ("DeviceFailureTests.DEATH_TEST_cs_timeout"),
    ("DeviceFailureTests.DEATH_TEST_cs_timeout_death"),
    ("DeviceFailureTests.DEATH_TEST_cs_timeout_busy_wait_death"),
    ("DfaDevCrashTests.DEATH_TEST_razwiAndMmuPageFault"),
    ("DfaDevCrashTests.DEATH_TEST_razwiOnly"),
    # 'DEATH_TEST_cs_timeout_acquire_after_reset' and 'acquire_device' are bundle test
    # The first get the device into a reset state and the second is trying to acquire
    # the device after it is recovered
    ("DeviceFailureTests.DEATH_TEST_cs_timeout_acquire_after_reset"),
    ("SynAPITest.device_acquire"),
    ("DeviceFailureTests.DEATH_TEST_assert_async_during_launch"),
    ("DeviceFailureTests.DEATH_TEST_undefined_op_code"),
    ("SynCanaryProtectionTest.DEATH_TEST_check_dc_canary_protection")
]

TESTS_G2: List[str] = [
    ("DfaDevCrashTests.DEATH_TEST_gaudi2_LBW_Razwi"),
    ("DfaDevCrashTests.DEATH_TEST_gaudi2_LBW_PageFault"),
    ("DfaDevCrashTests.DEATH_TEST_gaudi2_MemCopy_Razwi"),
    ("DfaDevCrashTests.DEATH_TEST_gaudi2_MemCopy_PageFault"),
    ("DeviceFailureTests.DEATH_TEST_undefined_op_code")
]

# list of object for each device
ALL_DEV_INFO: List[DevInfo] = [
    DevInfo("GAUDI", "gaudi", TESTS_G1),
    DevInfo("GAUDI HL2000M", "gaudi", TESTS_G1),
    DevInfo("GAUDI2", "gaudi2", TESTS_G2),
    DevInfo("GAUDI3", "gaudi3", [])
]

SUPPORTED_DEVICES: List[str] = [x.dev_name_by_file for x in ALL_DEV_INFO]


# function:
# wait until device is operational.
# return True when device is operational, or False on timeout
def wait_for_device() -> bool:
    expiration_time = time.time() + TIMEOUT_SEC
    print(PREFIX + "Waiting for device.")

    while time.time() < expiration_time:
        if not os.path.exists(DEVICE):
            time.sleep(SLEEP_SEC)  # sleep to avoid tight loop
            continue
        print(PREFIX + "Device is operational.")
        return True

    print(PREFIX + f"Error: Timeout reached ({TIMEOUT_SEC} seconds) while waiting for device. Exiting.")
    return False


# function:
# validate that the found device is supported
# if so, return the matching tuple ('dev_struct'). If not supported, return None
def get_device_info():
    try:
        with open(DEVICE, "r") as file:
            name_in_file = file.readline().strip()  # get device name
            for dev_struct in ALL_DEV_INFO:
                if dev_struct.dev_name_by_file == name_in_file:
                    print(f"* Found device: {name_in_file}")
                    return dev_struct
            print(f"* Error: Device '{name_in_file}' is unsupported.")
            print(f"* Supported devices are: {SUPPORTED_DEVICES}")
    except Exception as e:
        print(e)
    return None


# function:
# run a given test (in release mode) on a given device
# log and print output and errors
# assumes this device is operational
def run_test(dev_name: str, test_name: str) -> int:
    # format command to run
    # NOTICE: running test in release

    cmd = f"source $SYNAPSE_ROOT/.ci/scripts/synapse.sh && run_synapse_test -r -c {dev_name} --death-test -s {test_name}"
    print(PREFIX + f"Running command:\t {cmd}")
    shell_cmd = ["bash", "-c", cmd]

    print(PREFIX + "======= Test output: =======")
    process = subprocess.Popen(shell_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    stdout_total = ""
    while process.stdout.readable():
        line = process.stdout.readline()
        if not line.strip():
            break
        stdout_total += line
        print("\t" + line, end='')
    log_file.write(stdout_total)

    return_code = process.wait()
    if return_code != 0:
        stderr_str = process.stderr.read()
        print(PREFIX + "======= Test stderr: =======")
        print("\t" + stderr_str.replace('\n', '\n\t'))
        print(PREFIX + f"Failed: {test}")
        print(PREFIX + f"Return code: {return_code}")
        log_file.write(stderr_str)

    return return_code


# main script:
# run all death tests supported on current device.
# exit codes: 0 if all tests passed, 1 if any test failed, -1 if device wasn't ready within timeout
with open(LOG_PATH, 'w+') as log_file:
    # Check device is supported
    dev_info = get_device_info()
    if dev_info is None:
        exit(-1)

    # print list of selected tests
    tests: List[str] = dev_info.tests_list_info
    total: int = len(tests)
    print(f"* Running {total} death tests:")
    for test in tests:
        print(f"  - {test}")
    print("")

    # run all tests that are supported on this device
    passed: int = 0
    count: int = 0  # used to log number of the currently running test
    failed_tests: List[str] = []
    for test in tests:
        count += 1
        print(f"======= Test [{count}/{total}]: {test} =======")

        # wait until device is operational again
        if not wait_for_device():
            exit(-1)
        # run test on device
        test_result = run_test(dev_info.dev_name_by_test, test)
        if test_result == 0:
            passed += 1
        else:
            failed_tests.append(test)
        print("\n")  # 2 empty lines

    failed_count = len(failed_tests)
    if failed_count == 0:  # if list of failed tests isn't empty
        print(f"* Success. All {total} tests passed.")
    else:
        print(f"* Failure. {failed_count} out of {total} tests failed:")
        for test in failed_tests:
            print(f"\t- {test}")

    # Exit with 1 if some failed. If all passed, exit with 0
    exit(failed_count != 0)
