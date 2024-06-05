import subprocess
import os
import re
import sys
import pytest
import argparse
from datetime import datetime

def check_result(expected_result, actual_result, regression_threshold, improvement_threshold, test_id):
    result = {'reference': expected_result, 'error_message': None}
    expected_result = float(expected_result)
    regression_range = expected_result * regression_threshold / 100
    improvement_range = expected_result * improvement_threshold / 100
    try:
        diff_percentage = ((actual_result - expected_result) / expected_result) * 100
    except:
        diff_percentage = 'N/A'

    if actual_result > expected_result - regression_range:
        result['status'] = 'Pass'
        if actual_result > expected_result + improvement_range:
            result['status'] = 'Improve'

    else:
        result['status'] = 'Fail'
        result['error_message'] = 'Error in test {}: '      \
                                  'expected result: {}, '   \
                                  'actual result: {}, '     \
                                  'margin percentage: {}, ' \
                                  'diff percentage: {:.3f}' \
                                  .format(test_id,
                                          expected_result,
                                          actual_result,
                                          regression_threshold,
                                          diff_percentage)
    result['diff_percentage'] = diff_percentage
    return result


def parse_bgemm_results(out, bgemm_data):
    result = dict()
    result.update(bgemm_data)
    for line in out:
        if line.find('microseconds') != -1:
            result['run_time'] = float(line.split()[-2])
        if line.find('TOPS/second:') != -1:
            result['tops'] = float(line.split()[-1])
    return result


def run_command(command, path = None ,ignore_return_status = False, wait=True):
    p = subprocess.Popen([command], stdout=subprocess.PIPE, shell=True, stderr=subprocess.PIPE ,cwd=path)
    if not wait:
        return
    out, err = p.communicate()
    print(out.decode('utf-8'))
    if (p.returncode and not ignore_return_status):
        print(err.decode('utf-8'))
        print(out.decode('utf-8'))
        print('error exist status: {}'.format(p.returncode))
        raise RuntimeError
    return out.decode('utf-8').splitlines()


def set_env(var_name, value):
    if 'transpose' not in var_name.lower():
        os.environ[var_name] = str(value)
    elif 'transpose' in var_name.lower():
        if '0' == value:
            print('name: {}'.format(var_name))
            os.unsetenv(var_name)
        else:
            os.environ[var_name] = str(value)
    else:
        raise ValueError('unsupported case: name: {} = {}'.format(var_name, value))


def run_bgemm(path, bgemm_params):
    for key, value in bgemm_params.items():
        if key not in ['TEST_TYPE', 'TEST_ID', 'NUM_OF_RUNS']:
            set_env(key, value)
    test_type = 'BF16' if bgemm_params['TEST_TYPE'] == 'bf16' else 'Float'
    cmd = 'HABANA_PROFILE=1 ./bgemm_test --gtest_filter=GaudiExampleInfraBGemm.bgemm{}_benchmark'.format(test_type)
    res = run_command(cmd, path)
    return parse_bgemm_results(res, bgemm_params)


def compile_test(dev_env, path):
    compile_command = 'make'
    if dev_env:
        compile_command = '{} dev'.format(compile_command)
    print('compilation command line: {}'.format(compile_command))
    run_command(compile_command, path)

class TestMicroBenchmarksExample:
    is_initialized = False

    @pytest.fixture(autouse=True)
    def setup_class(self, request):
        if self.is_initialized is False:
            self._setup_class(request)

    @classmethod
    def _setup_class(cls, request):
        cls.is_initialized = True
        dt_string = datetime.now().strftime('%d_%m_%Y_%H_%M_%S')
        output_dir = request.config.getoption('output')

        cls.bgemm_res_file = os.path.join(output_dir, 'bgemm_res_file_{}.csv'.format(dt_string))
        cls.bgemm_reference_file = os.path.join(output_dir, 'bgemm_reference_file_{}.csv'.format(dt_string))
        cls.bgemm_percentage = request.config.getoption('threshold')
        cls.bgemm_improvement_percentage = request.config.getoption('improvement')
        cls.repeats = request.config.getoption('repeats')
        cls.skip_validation = request.config.getoption('skip_validation')
        cls.generate_csv = request.config.getoption('gen')
        cls.generate_reference = request.config.getoption('gen_ref')
        cls.specific_tests = request.config.getoption('tests')
        cls.device = request.config.getoption('device')
        set_env('MICRO_BENCHMARK_DEVICE', cls.device)
        print('Current run id is {}'.format(dt_string))
        if os.getenv('HABANA_SOFTWARE_STACK', None):
            dev_env = True
            cls.bgemm_path = os.path.join(os.getenv('HABANA_SOFTWARE_STACK'),
                                         'examples',
                                         'gaudi',
                                         'micro-benchmarks',
                                         'bgemm')
        else:
            dev_env = False
            cls.bgemm_path = os.path.join(os.getenv('HOME'),
                                         'habanalabs',
                                         'examples',
                                         'gaudi',
                                         'micro-benchmarks',
                                         'bgemm')
        compile_test(dev_env, cls.bgemm_path)
        if cls.generate_csv:
            with open(cls.bgemm_res_file, 'w') as fh:
                fh.write('Test ID, '
                         'BGEMM_M, '
                         'BGEMM_K, '
                         'BGEMM_N, '
                         'BGEMM_C '
                         'BGEMM_B '
                         'BGEMM_SKIP_VALIDITY, '
                         'NUM_OF_ITER, '
                         'BGEMM_NUM_NODES, '
                         'BGEMM_INPUT_MEAN, '
                         'BGEMM_INPUT_STD, '
                         'TEST_TYPE, '
                         'BGEMM_TRANSPOSE_A, '
                         'BGEMM_TRANSPOSE_B, '
                         'Topology Execution Time[ms], '
                         'TOPS/second, '
                         'Expected results [TOPS/second], '
                         'Status, '
                         'Diff (TOPS/second Vs. Expected) [%]\n')
        if cls.generate_reference:
            with open(cls.bgemm_reference_file, 'w') as fh:
                fh.write('Test ID, '
                         'BGEMM_M, '
                         'BGEMM_K, '
                         'BGEMM_N, '
                         'BGEMM_C '
                         'BGEMM_B '
                         'BGEMM_SKIP_VALIDITY, '
                         'NUM_OF_ITER, '
                         'BGEMM_NUM_NODES, '
                         'BGEMM_INPUT_MEAN, '
                         'BGEMM_INPUT_STD, '
                         'TEST_TYPE, '
                         'BGEMM_TRANSPOSE_A, '
                         'BGEMM_TRANSPOSE_B, '
                         'EXPECTED_RESULT, '
                         'NUM_OF_RUNS\n')

    @pytest.mark.bgemm
    def test_bgemm(self, bgemm_data, record_property):
        if self.specific_tests and int(bgemm_data['TEST_ID']) not in self.specific_tests:
            pytest.skip()
        num_of_runs = self.repeats if self.repeats != -1 else int(bgemm_data['NUM_OF_RUNS'])
        results = []
        for test_counter in range(num_of_runs):
            try:
                bgemm_res = run_bgemm(self.bgemm_path, bgemm_data)
            except RuntimeError as e:
                if test_counter + 1 == num_of_runs:
                    raise e
                else:
                    continue
            results.append(float(bgemm_res['tops']))
            status_results = check_result(bgemm_data['EXPECTED_RESULT'],
                                          max(results),
                                          self.bgemm_percentage,
                                          self.bgemm_improvement_percentage,
                                          bgemm_data['TEST_ID'])
            if not self.generate_reference and (self.skip_validation or status_results['status'] in ['Pass', 'Improve']):
                print('test pass after {} times'.format(test_counter + 1))
                break
        if results:
            expected_result = float(bgemm_data['EXPECTED_RESULT'])
            valid_result = expected_result * (1 - self.bgemm_percentage / 100)
            record_property("tops_per_second", max(results))
            record_property("expected_result", expected_result)
            record_property("valid_result", valid_result)
        if self.generate_csv:
            with open(self.bgemm_res_file, 'a') as fh:
                fh.write('{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n'
                         .format(bgemm_data['TEST_ID'],
                                 bgemm_res['BGEMM_M'],
                                 bgemm_res['BGEMM_K'],
                                 bgemm_res['BGEMM_N'],
                                 bgemm_res['BGEMM_C'],
                                 bgemm_res['BGEMM_B'],
                                 bgemm_res['BGEMM_SKIP_VALIDITY'],
                                 bgemm_res['NUM_OF_ITER'],
                                 bgemm_res['BGEMM_NUM_NODES'],
                                 bgemm_res['BGEMM_INPUT_MEAN'],
                                 bgemm_res['BGEMM_INPUT_STD'],
                                 bgemm_res['TEST_TYPE'],
                                 bgemm_res['BGEMM_TRANSPOSE_A'],
                                 bgemm_res['BGEMM_TRANSPOSE_B'],
                                 bgemm_res['run_time'],
                                 bgemm_res['tops'],
                                 bgemm_data['EXPECTED_RESULT'],
                                 status_results['status'],
                                 status_results['diff_percentage']))
        if self.generate_reference:
            index = (5 * len(results)) // 100
            # result is maximum TOPS, but ignore 5% outliers
            result = sorted(results)[-index - 1]
            if result > float(bgemm_res['EXPECTED_RESULT']) or self.skip_validation:
                expected_result = result
            else:
                expected_result = bgemm_res['EXPECTED_RESULT']
            with open(self.bgemm_reference_file, 'a') as fh:
                fh.write('{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n'
                         .format(bgemm_data['TEST_ID'],
                                 bgemm_res['BGEMM_M'],
                                 bgemm_res['BGEMM_K'],
                                 bgemm_res['BGEMM_N'],
                                 bgemm_res['BGEMM_C'],
                                 bgemm_res['BGEMM_B'],
                                 bgemm_res['BGEMM_SKIP_VALIDITY'],
                                 bgemm_res['NUM_OF_ITER'],
                                 bgemm_res['BGEMM_NUM_NODES'],
                                 bgemm_res['BGEMM_INPUT_MEAN'],
                                 bgemm_res['BGEMM_INPUT_STD'],
                                 bgemm_res['TEST_TYPE'],
                                 bgemm_res['BGEMM_TRANSPOSE_A'],
                                 bgemm_res['BGEMM_TRANSPOSE_B'],
                                 expected_result,
                                 bgemm_data['NUM_OF_RUNS']))
        if not self.skip_validation and status_results['status'] == 'Fail':
            raise ValueError(status_results['error_message'])

def parse_args():
    parser = argparse.ArgumentParser(description="Run Gemm benchmark tests")
    parser.add_argument('--pytest-help', help='print pytest help', action='store_true', dest='pytest_help')
    parser.add_argument('--all', help='Run all tests (override stop on first failure)', action='store_true', dest='all')
    parser.add_argument('--threshold', help='Regression threshold in percent (default=5)', type=int, dest='threshold', default=5)
    parser.add_argument('--improvement', help='Improvement threshold in percent (default=5)', type=int, dest='improvement', default=5)
    parser.add_argument('--repeats', help='Override maximum repeats until test pass', type=int, dest='repeats', default=-1)
    parser.add_argument('--generate', help='Generate result file (csv), Also run all test', action='store_true', dest='gen')
    parser.add_argument('--generate-reference', help='Generate new refernce file update only imrovement, to override combine with --skip-validation', action='store_true', dest='gen_ref')
    parser.add_argument('--skip-validation', help='Skip validation', action='store_true', dest='skip_validation')
    parser.add_argument('--random-order', help='Run the tests in random order', action='store_true', dest='rand')
    parser.add_argument('--tests', help='Run specific tests', type=int, nargs='+', dest='tests', default=[])
    parser.add_argument('--params-path', help='Gemm params file (default bgemm_params.csv)', dest='params_path', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bgemm_params.csv'))
    parser.add_argument('--output-dir', help='Path to directory for output file (default ./)', dest='output', default=os.path.abspath('./'))
    parser.add_argument('--device', help='Device type', dest='device', choices=['gaudi', 'gaudi2'], default='gaudi')
    return parser.parse_known_args()

def get_pytest_args(args, unknown_args):
    pytest_args = unknown_args.copy()
    if args.pytest_help:
        pytest_args.append('-h')

    if args.gen:
        pytest_args.append('--generate')
        args.all = True

    if not args.all:
        pytest_args.append('--exitfirst')

    if args.gen_ref:
        pytest_args.append('--generate_reference')
        args.rand = False

    if args.skip_validation:
        pytest_args.append('--skip_validation')

    if not args.rand:
        pytest_args.append('--random-order-bucket=none')

    if args.tests != []:
        pytest_args.append('--tests')
        pytest_args += [str(test) for test in args.tests]

    pytest_args.append('--threshold')
    pytest_args.append(str(args.threshold))

    pytest_args.append('--improvement')
    pytest_args.append(str(args.improvement))

    pytest_args.append('--repeats')
    pytest_args.append(str(args.repeats))

    pytest_args.append('--params_path')
    pytest_args.append(str(args.params_path))

    pytest_args.append('--output_dir')
    pytest_args.append(str(args.output))

    pytest_args.append('--device')
    pytest_args.append(str(args.device))

    return ' '.join(pytest_args)

def get_params_from_file(params_file):
    first_line = True
    list_of_params = list()
    indices = {
        'TEST_ID' : 0,
        'BGEMM_M': 1,
        'BGEMM_K': 2,
        'BGEMM_N': 3,
        'BGEMM_C': 4,
        'BGEMM_B': 5,
        'BGEMM_SKIP_VALIDITY': 6,
        'NUM_OF_ITER': 7,
        'BGEMM_NUM_NODES': 8,
        'BGEMM_INPUT_MEAN': 9,
        'BGEMM_INPUT_STD': 10,
        'TEST_TYPE': 11,
        'BGEMM_TRANSPOSE_A': 12,
        'BGEMM_TRANSPOSE_B': 13,
        'EXPECTED_RESULT': 14,
        'NUM_OF_RUNS' : 15
    }
    with open(params_file) as fh:
        for line in fh:
            if first_line:
                first_line = False
                continue
            params = {}
            tmp_line = line.split(',')

            for key, index in indices.items():
                value = tmp_line[index].strip().rstrip()
                params[key] = value
            list_of_params.append(params)

    return list_of_params

def id_func(bgemm_data):
    return 'ID({})-M({})-K({})-N({})-C({})-B({})-TYPE({})'.format(bgemm_data['TEST_ID'],
                                                      bgemm_data['BGEMM_M'],
                                                      bgemm_data['BGEMM_K'],
                                                      bgemm_data['BGEMM_N'],
                                                      bgemm_data['BGEMM_C'],
                                                      bgemm_data['BGEMM_B'],
                                                      bgemm_data['TEST_TYPE'])

def pytest_generate_tests(metafunc):
    # This is called for every test. Only get/set command line arguments
    # if the argument is specified in the list of test "fixturenames".
    params_file = metafunc.config.getoption("params_path")
    if 'bgemm_data' in metafunc.fixturenames and params_file is not None:
        metafunc.parametrize('bgemm_data', get_params_from_file(params_file), ids=id_func)

if __name__ == '__main__':
    args, unknown_args = parse_args()
    pytest_args = get_pytest_args(args, unknown_args)
    additional_args = '--tb=line'

    run_command('mkdir -p ~/.habana')
    run_command('hl-prof-config -e off -{} -i -host=off'.format(args.device))
    try:
        print('python -m pytest {} {} {}'.format(os.path.abspath(__file__), pytest_args, additional_args))
        os.system('python -m pytest {} {} {}'.format(os.path.abspath(__file__), pytest_args, additional_args))
    finally:
        run_command('hl-prof-config -e off -{}'.format(args.device))
