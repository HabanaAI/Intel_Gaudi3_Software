import glob
import re
from collections import defaultdict
import scipy
from scipy import stats
import tabulate
import numpy as np

def main():
    results = defaultdict(dict)

    for fn in glob.glob('par.*.txt'):
        network_name, feature_name = re.match('par\.(.*?)\.(.*).txt', fn).groups()
        assert network_name not in results or feature_name not in results[network_name]

        res = defaultdict(list)
        with open(fn) as f:
            for line in f:
                tmp = re.match('.*total ([a-zA-Z_0-9]+) phase took ([0-9]+) micro waited ([0-9]+) micro', line)
                if tmp is not None:
                    res[tmp.group(1)] += [int(tmp.group(2))]
                    res[tmp.group(1) + ' waited'] += [int(tmp.group(3))]
                    continue

                tmp = re.match('.*total ([a-zA-Z_0-9]+) phase took ([0-9]+) micro', line)
                if tmp is not None:
                    res[tmp.group(1)] += [int(tmp.group(2))]
                    continue

        # trim the leftmost and rightmost 10%
        results[network_name][feature_name] = {k: int(np.round(scipy.stats.trim_mean(vs, 0.1)) if len(vs) >= 10 else 0) for k, vs in res.items()}

    for network_name, features in results.items():
        feature_names, measurement_data = map(list, zip(*sorted(features.items())))

        measurement_names = measurement_data[0].keys()
        assert all(m.keys() == measurement_names for m in measurement_data)

        headers = [network_name] + feature_names
        content = zip(*([measurement_names] + [list(m.values()) for m in measurement_data]))
        print(tabulate.tabulate(content, headers=headers))
        print()


if __name__ == '__main__':
    main()
