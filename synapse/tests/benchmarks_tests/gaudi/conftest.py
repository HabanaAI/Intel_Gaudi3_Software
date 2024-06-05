
def pytest_addoption(parser):
    parser.addoption('--threshold', type=int, dest='threshold')
    parser.addoption('--improvement', type=int, dest='improvement')
    parser.addoption('--repeats', type=int, dest='repeats')
    parser.addoption('--generate', action='store_true', dest='gen')
    parser.addoption('--generate_reference', action='store_true', dest='gen_ref')
    parser.addoption('--skip_validation', action='store_true', dest='skip_validation')
    parser.addoption('--tests', type=int, nargs='+', dest='tests', default=[])
    parser.addoption('--params_path', dest='params_path')
    parser.addoption('--output_dir', dest='output')
    parser.addoption('--device', dest='device')


