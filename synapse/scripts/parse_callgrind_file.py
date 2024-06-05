#!/usr/bin/env python3

import argparse
import re
import sys
from typing import Dict, List, Optional, Tuple


def parse_callgrind_file(
    file_path: str, fns_pat: str
) -> Tuple[List[str], Dict[str, List[int]]]:
    """Find functions by regex pattern within a callgrind result file and collect cumulative stats"""  # noqa: E501

    keys: List[str] = []
    res: Dict[str, List[int]] = {}

    # TODO: use .* instead of fns_pat so that it doesn't mess the regex up,
    #       and then re-parse the name vs fn_pat.
    line_pat = re.compile(Rf"(c)?fn=(?:\(([0-9]*)\))?\s*({fns_pat})?")

    id_map: Dict[int, str] = {}
    curr_func = None
    with open(file_path) as f:
        for line_nr, line in enumerate(f, 1):
            line = line.strip()
            if line.startswith("#"):
                continue
            if not line:
                curr_func = None

            if line.startswith("events: "):
                new_keys = [x.lower() for x in line.split()[1:]]
                assert keys == [] or keys == new_keys, "inconsistent keys"
                keys = new_keys
                # res = {fn: [0 for _ in range(len(keys))] for fn in fns}
                continue

            tmp = re.match(line_pat, line)
            if tmp is not None:
                c, alias, name = tmp.groups()

                if name is not None:
                    if alias is not None:
                        alias_int = int(alias)
                        assert alias_int not in id_map, f"{alias_int=}"
                        id_map[alias_int] = name
                        # print(f'set at {line_nr}: {alias_int} -> {name}')
                else:
                    if alias is not None:
                        new_name = id_map.get(int(alias), None)
                        if new_name is not None:
                            name = new_name

                if c is None:
                    if name is not None:
                        assert curr_func is None
                        curr_func = name
                    if curr_func is not None:
                        # print(f'set at {line_nr}')
                        pass

            elif curr_func and line[0] in "0123456789+-*":
                vals = line.split()[1:]
                assert curr_func is not None
                if curr_func not in res:
                    res[curr_func] = [0 for _ in keys]
                else:
                    assert len(vals) <= len(res[curr_func])
                    for i, v in enumerate(vals):
                        res[curr_func][i] += int(v)

    return keys, res


def adjust_function_name(fn, short_name: bool) -> str:
    return fn.split("(", 1)[0] if short_name else fn


def do_simple_parse(callgrind_res: str, func_regex_pattern: str, short_names: bool):
    keys, per_func_results = parse_callgrind_file(callgrind_res, func_regex_pattern)

    try:
        import tabulate

        print(
            tabulate.tabulate(
                sorted(
                    [
                        [adjust_function_name(k, short_names), *(f"{v:,}" for v in vs)]
                        for k, vs in per_func_results.items()
                    ]
                ),
                headers=["fn"] + keys,
            )
        )
    except ImportError:
        print(
            keys,
            {
                adjust_function_name(k, short_names): v
                for k, v in per_func_results.items()
            },
        )
        print("Note: 'pip install tabulate' for a nicer print", file=sys.stderr)


def do_diff_parse(
    callgrind_ref: str, callgrind_res, func_regex_pattern: str, short_names: bool
):
    """Parse the two callgrind files, collecting the cumulative results,
    for the functions matching the filter and show the diff of res on top of ref."""

    ref_keys, ref_per_func_results = parse_callgrind_file(
        callgrind_ref, func_regex_pattern
    )
    res_keys, res_per_func_results = parse_callgrind_file(
        callgrind_res, func_regex_pattern
    )

    try:
        import termcolor

        def my_colored(s: str, font_color: str, bg_color: Optional[str] = None):
            return termcolor.colored(s, font_color, bg_color)

        have_colors = True
    except ImportError:

        def my_colored(s: str, font_color: str, bg_color: Optional[str] = None):
            return s

        have_colors = False

    # TODO: this is so much cleaner if we can import pandas...
    def prep_diff_data(ref_keys, ref_data, res_keys, res_data):
        common_keys = set(ref_keys) & set(res_keys)

        def collect_common_data(
            keys: List[str], data: Dict[str, List[int]]
        ) -> Dict[str, Dict[str, int]]:
            res: Dict[str, Dict[str, int]] = {}
            for fn, vs in data.items():
                res[fn] = {}
                for k, v in zip(keys, vs):
                    if k in common_keys:
                        res[fn][k] = v
            return res

        ref = collect_common_data(ref_keys, ref_data)
        res = collect_common_data(res_keys, res_data)

        all_fns = sorted(set(ref.keys()) | set(res.keys()))

        # done to preserve the order
        keys = [k for k in ref_keys if k in common_keys]

        data = {}
        for fn in all_fns:
            data[fn] = []
            in_ref = fn in ref
            in_res = fn in res
            for k in keys:
                res_val = int(res.get(fn, {k: "0"})[k])
                ref_val = int(ref.get(fn, {k: "0"})[k])
                v = res_val - ref_val

                # only apply colors to the diffs if the fn is found in both ref and res
                if in_res and in_ref:
                    if ref_val != 0:
                        # print as percent rounded to 3 decimals
                        proc = f"({int(1e5 * v / ref_val) / 1e3}%)"
                    else:
                        proc = ""
                    data[fn] += [
                        my_colored(f"{v:,} {proc}", "green" if v <= 0 else "red", None)
                    ]
                else:
                    data[fn] += [f"{v:,}"]
        return keys, data

    keys, data = prep_diff_data(
        ref_keys, ref_per_func_results, res_keys, res_per_func_results
    )

    try:
        import tabulate

        print(
            tabulate.tabulate(
                sorted(
                    [
                        [adjust_function_name(k, short_names), *vs]
                        for k, vs in data.items()
                    ]
                ),
                headers=["fn"] + keys,
            )
        )

        if have_colors:
            print()
            print("Notes:")
            print("\tGreen      - No regression")
            print("\tRef        - Regression")
            print("\tNeutral    - Function only found in one of the files")
            print()
        else:
            print("'pip install termcolor' For colored diff", file=sys.stderr)

    except ImportError:
        print(keys, {adjust_function_name(k, short_names): v for k, v in data.items()})
        print("Note: 'pip install tabulate' for a nicer print", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        "Find functions by regex pattern within a callgrind result file and show cumulative results or diff vs a second file"  # noqa: E501
    )
    parser.add_argument("-f", "--callgrind_res", required=True)
    parser.add_argument(
        "-d", "--callgrind_ref", help="Diff the callgrind res file against this file"
    )
    parser.add_argument(
        "-r",
        "--func_regex_pattern",
        default="|".join(
            (
                "main",
                "dl(?:open|sym)",
                "syn[A-Z][a-zA-Z0-9_]*",
                ".*addNode",
                "malloc",
                "realloc",
                "free",
                "sbrk",
                "mmap",
                "new",
            )
        ),
    )
    parser.add_argument(
        "-s",
        "--short_names",
        action="store_true",
        help="Do not show the function arguments",
    )
    args = parser.parse_args()

    if args.callgrind_ref is None:
        do_simple_parse(args.callgrind_res, args.func_regex_pattern, args.short_names)
    else:
        do_diff_parse(
            args.callgrind_ref,
            args.callgrind_res,
            args.func_regex_pattern,
            args.short_names,
        )


if __name__ == "__main__":
    main()
