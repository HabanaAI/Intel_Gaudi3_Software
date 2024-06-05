#!/usr/bin/env python
import argparse
import csv
import os
import re
from itertools import groupby
from typing import Any, Dict, List, Set, Tuple


def gen_fieldnames(features: List[str]) -> List[str]:
    """Generate a list of all supported fieldnames (cols).
    Unused ones will be eliminated later."""

    res = [
        "network",
        "graph",
        "compile min em",
        *(f"compile min em {ft}" for ft in features),
        #    'compile min gm',
        *(f"compile min {ft} overhead %" for ft in features),
        "compile mean trim 1% em",
        *(f"compile mean trim 1% em {ft}" for ft in features),
        #    'compile mean trim 1% gm',
        *(f"compile mean trim 1% {ft} overhead %" for ft in features),
        "runtime min em",
        *(f"runtime min em {ft}" for ft in features),
        "runtime min gm",
        *(f"runtime min {ft} overhead %" for ft in features),
        "runtime mean trim 1% em",
        *(f"runtime mean trim 1% em {ft}" for ft in features),
        "runtime mean trim 1% gm",
        *(f"runtime mean trim 1% {ft} overhead %" for ft in features),
        "pre node #",
        "pre nodes",
    ]
    for x in ["em", "gm"] + features:
        res += [
            f"post node # {x}",
            f"post nodes {x}",
            f"logical guid # {x}",
            f"logical guids {x}",
            f"mme guid # {x}",
            f"mme guids {x}",
            f"tpc guid # {x}",
            f"tpc guids {x}",
            f"dma guid # {x}",
            f"dma guids {x}",
        ]
    return res


def prep_summary(
    features: List[str], fieldnames: List[str], name: str
) -> List[Dict[str, str]]:
    def read(fn):
        try:
            with open(fn) as f:
                return list(csv.DictReader(f))
        except FileNotFoundError:
            # Not having a file is ok, any other exception is a problem.
            return []

    def read_path(title):
        return read(f"latest_rundir/{title}/stats.latest.json.csv")

    c_em = read_path(f"{name}_compile_bench")
    # c_gm    = read_path(f'{name}_compile_bench_gm')
    c_em_ft = [read_path(f"{name}_compile_bench_{ft}") for ft in features]
    r_em = read_path(f"{name}_run_bench")
    r_gm = read_path(f"{name}_run_bench_gm")
    r_em_ft = [read_path(f"{name}_run_bench_{ft}") for ft in features]

    def col(dictreader, col_name):
        return [row[col_name] for row in dictreader]

    # collect since not all graphs might be presnet
    graphs: Set[Any] = set().union(
        set(col(c_em, "graph")),
        # set(col(c_gm, 'graph')),
        *(set(col(ft, "graph")) for ft in c_em_ft),
        set(col(r_em, "graph")),
        set(col(r_gm, "graph")),
        *(set(col(ft, "graph")) for ft in r_em_ft),
    )

    res = {g: {**{f: "" for f in fieldnames}, "network": name} for g in graphs}

    def update(res, rows, mapping={}, ft=""):
        def set_if_new(src_key, dst_key):
            try:
                src_val = row[src_key]
            except KeyError:
                return
            src_val = src_val.strip().strip('"')
            assert dest[dst_key] in (
                "",
                src_val,
            ), f'{dst_key=}, {dest[dst_key]=}, {src_val=} <- {row["graph"]=}'
            dest[dst_key] = src_val

        for row in rows:
            dest = res[row["graph"]]
            set_if_new("graph", "graph")
            for k, v in mapping.items():
                dest[k] = row[v]
            set_if_new(' "pre-guid count"', "pre node #")
            set_if_new(' "pre-guids"', "pre nodes")
            set_if_new(' "post-guid count"', "post node #" + ft)
            set_if_new(' "post-guids"', "post nodes" + ft)
            set_if_new(' "Logical guid count"', "logical guid #" + ft)
            set_if_new(' "Logical guids"', "logical guids" + ft)
            set_if_new(' "MME guid count"', "mme guid #" + ft)
            set_if_new(' "MME guids"', "mme guids" + ft)
            set_if_new(' "TPC guid count"', "tpc guid #" + ft)
            set_if_new(' "TPC guids"', "tpc guids" + ft)
            set_if_new(' "DMA guid count"', "dma guid #" + ft)
            set_if_new(' "DMA guids"', "dma guids" + ft)

    update(
        res,
        c_em,
        {
            "compile min em": ' "min compile"',
            "compile mean trim 1% em": ' "mean compile trim 1%"',
        },
        " em",
    )
    # update(
    #     res,
    #     c_gm,
    #     {
    #         "compile min gm": ' "min compile"',
    #         "compile mean trim 1% gm": ' "mean compile trim 1%"',
    #     },
    #     " gm",
    # )
    update(
        res,
        r_em,
        {
            "runtime min em": ' "min runtime"',
            "runtime mean trim 1% em": ' "mean runtime trim 1%"',
        },
        " em",
    )
    update(
        res,
        r_gm,
        {
            "runtime min gm": ' "min runtime"',
            "runtime mean trim 1% gm": ' "mean runtime trim 1%"',
        },
        " gm",
    )
    for ft, c, r in zip(features, c_em_ft, r_em_ft):
        update(
            res,
            c,
            {
                f"compile min em {ft}": ' "min compile"',
                f"compile mean trim 1% em {ft}": ' "mean compile trim 1%"',
            },
            f" {ft}",
        )
        update(
            res,
            r,
            {
                f"runtime min em {ft}": ' "min runtime"',
                f"runtime mean trim 1% em {ft}": ' "mean runtime trim 1%"',
            },
            f" {ft}",
        )

    # use 4 decimal points, so that it has 2 significant digits when converted to %
    def update_overhead(row, key):
        try:
            em = float(row[f"{key} em"].strip())
        except ValueError:
            return
        if not em:
            return

        for ft in features:
            try:
                em_feature = float(row[f"{key} em {ft}"].strip())
            except ValueError:
                continue
            row[f"{key} {ft} overhead %"] = "{:.4f}".format((em_feature - em) / em)

    for row in res.values():
        update_overhead(row, "compile min")
        update_overhead(row, "compile mean trim 1%")
        update_overhead(row, "runtime min")
        update_overhead(row, "runtime mean trim 1%")

    return list(res.values())


def summarize_to_csv(fieldnames: List[str], rows: List[Dict[str, str]]) -> None:
    def remove_unused_columns(
        fieldnames: List[str], rows: List[dict]
    ) -> Tuple[List[str], List[dict]]:
        vals_to_remove = {f for f in fieldnames if not any(row[f] for row in rows)}
        new_fieldnames = [f for f in fieldnames if f not in vals_to_remove]
        new_rows = [
            {k: v for k, v in row.items() if k not in vals_to_remove} for row in rows
        ]
        return new_fieldnames, new_rows

    # per-network summary files
    if True:
        for network_name, grouped_rows in groupby(rows, lambda row: row["network"]):
            if not grouped_rows:
                continue

            filtered_fieldnames, filtered_rows = remove_unused_columns(
                fieldnames, list(grouped_rows)
            )

            with open(f"latest_rundir/{network_name}.csv", "w") as f:
                writer = csv.DictWriter(f, fieldnames=filtered_fieldnames)
                writer.writeheader()
                for row in sorted(filtered_rows, key=lambda row: int(row["graph"])):
                    writer.writerow(row)

    # total summary in a single file
    if True:
        filtered_fieldnames, filtered_rows = remove_unused_columns(fieldnames, rows)

        with open("latest_rundir/all.csv", "w") as f:
            writer = csv.DictWriter(f, fieldnames=filtered_fieldnames)
            writer.writeheader()
            for row in sorted(
                filtered_rows, key=lambda row: (row["network"], int(row["graph"]))
            ):
                writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--em_features", nargs="*", default=[])
    args = parser.parse_args()

    fieldnames = gen_fieldnames(args.em_features)

    def detected_networks():
        res = set()
        for fn in os.listdir("latest_rundir"):
            if os.path.isdir(f"latest_rundir/{fn}"):
                tmp = re.match("(.*)_(:?compile|run)_bench(:?_.*)*", fn)
                if tmp is not None:
                    res.add(tmp.group(1))
        return sorted(res)

    res = []
    for network_name in detected_networks():
        res += prep_summary(args.em_features, fieldnames, network_name)

    summarize_to_csv(fieldnames, res)


if __name__ == "__main__":
    main()
