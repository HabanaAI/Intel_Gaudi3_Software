#!/usr/bin/env python
import argparse
from collections import defaultdict, Counter
import itertools
import json
import numpy as np
from scipy import stats
import tabulate


def load_data(stats_file, pre_graph_file=None, post_graph_file=None):
    with open(stats_file) as f:
        stats_json = json.load(f)

    if pre_graph_file is None:
        pre_guids = None
    else:
        with open(pre_graph_file) as f:
            jj = json.load(f)
        pre_guids = {g["name"]: [n["guid"] for n in g["nodes"]] for g in jj["graphs"]}

        graph_num = len(jj["graphs"])
        assert len(pre_guids) == graph_num, "repeated names found?!"

    if post_graph_file is None:
        post_guids = None
        post_guids_by_engine = None
    else:
        with open(post_graph_file) as f:
            jj = json.load(f)

        graph_num = len(jj["graphs"])
        assert len(set(g["name"] for g in jj["graphs"])) == len(
            jj["graphs"]
        ), "non-unique graph names..."

        post_guids = {}
        post_guids_by_engine = {
            "None": defaultdict(list),
            "DMA": defaultdict(list),
            "MME": defaultdict(list),
            "TPC": defaultdict(list),
        }
        for g in jj["graphs"]:
            g_name = g["name"]
            exec_sorted_nodes = sorted(g["nodes"], key=lambda n: n["exec_order_idx"])

            post_guids[g_name] = [n["guid"] for n in exec_sorted_nodes]
            for n in exec_sorted_nodes:
                post_guids_by_engine[n["engine"]][g_name] += [n["guid"]]

        sanity_res = defaultdict(Counter)
        for by_eng in post_guids_by_engine.values():
            for graph, guids in by_eng.items():
                for guid in guids:
                    sanity_res[graph][guid] += 1
        sanity_ref = {graph: Counter(guids) for graph, guids in post_guids.items()}
        assert sanity_res == sanity_ref

    if pre_guids is not None and post_guids is not None:
        assert set(pre_guids) >= set(
            post_guids
        ), "expected pre-graph names to be a super-set of the post-graph names"

    return stats_json, pre_guids, post_guids, post_guids_by_engine


def summrize_stats_json_to_csv(
    stats_json,
    pre_guids,
    post_guids,
    post_guids_by_engine,
    output_csv,
    eager_node_threshold=16,
):
    with open(output_csv, "w") as f:
        header_line = [
            "graph",
            "compile iters",
            "min compile",
            "mean compile",
            "mean compile trim 1%",
            "mean compile trim 5%",
            "max compile",
            "runtime iters",
            "min runtime",
            "mean runtime",
            "mean runtime trim 1%",
            "mean runtime trim 5%",
            "max runtime",
        ]
        if pre_guids:
            header_line += ["pre-guid count", "pre-guids"]
        if post_guids:
            header_line += ["post-guid count", "post-guids"]
        if post_guids_by_engine:
            header_line += ["Logical guid count", "Logical guids"]
            header_line += ["DMA guid count", "DMA guids"]
            header_line += ["MME guid count", "MME guids"]
            header_line += ["TPC guid count", "TPC guids"]
        f.write(", ".join(('"{}"'.format(v) for v in header_line)) + "\n")

        seen_graphs = set()
        for graph_name, graph_stats in stats_json["graphs"].items():
            assert graph_name not in seen_graphs, f"{graph_name=} already seen!"
            seen_graphs.add(graph_name)

            compile_times = graph_stats.get("compileTime", [0])
            device_runtimes = list(
                itertools.chain.from_iterable(graph_stats.get("deviceRuntime", [[0]]))
            )
            line = [
                graph_name,
                len(compile_times),
                np.min(compile_times),
                int(np.round(np.mean(compile_times))),
                int(np.round(stats.trim_mean(compile_times, 0.01))),
                int(np.round(stats.trim_mean(compile_times, 0.05))),
                np.max(compile_times),
                len(device_runtimes),
                np.min(device_runtimes),
                int(np.round(np.mean(device_runtimes))),
                int(np.round(stats.trim_mean(device_runtimes, 0.01))),
                int(np.round(stats.trim_mean(device_runtimes, 0.05))),
                np.max(device_runtimes),
            ]
            if pre_guids or post_guids or post_guids_by_engine:
                if pre_guids:
                    line += [
                        len(pre_guids[graph_name]),
                        '"' + " ".join(pre_guids[graph_name]) + '"',
                    ]
                if post_guids:
                    line += [
                        len(post_guids[graph_name]),
                        '"' + " ".join(post_guids[graph_name]) + '"',
                    ]
                if post_guids_by_engine:
                    line += [
                        len(post_guids_by_engine["None"][graph_name]),
                        '"' + " ".join(post_guids_by_engine["None"][graph_name]) + '"',
                    ]
                    line += [
                        len(post_guids_by_engine["DMA"][graph_name]),
                        '"' + " ".join(post_guids_by_engine["DMA"][graph_name]) + '"',
                    ]
                    line += [
                        len(post_guids_by_engine["MME"][graph_name]),
                        '"' + " ".join(post_guids_by_engine["MME"][graph_name]) + '"',
                    ]
                    line += [
                        len(post_guids_by_engine["TPC"][graph_name]),
                        '"' + " ".join(post_guids_by_engine["TPC"][graph_name]) + '"',
                    ]
            f.write(", ".join(("{}".format(v) for v in line)) + "\n")


def print_summary(stats_json, pre_guids, eager_node_threshold=16):
    header_line = [
        "min compile",
        "mean compile",
        "mean compile trim 1%",
        "mean compile trim 5%",
        "max compile",
        "min runtime",
        "mean runtime",
        "mean runtime trim 1%",
        "mean runtime trim 5%",
        "max runtime",
    ]

    graph_totals = [0 for _ in range(10)]
    graph_upto_threshold_nodes_totals = [0 for _ in range(10)]

    graph_count = 0
    graph_count_with_nodes_below_threshold = 0
    op_count = 0
    op_count_with_nodes_below_threshold = 0

    for graph_name, graph_stats in stats_json["graphs"].items():
        compile_times = graph_stats.get("compileTime", [0])
        device_runtimes = list(
            itertools.chain.from_iterable(graph_stats.get("deviceRuntime", [[0]]))
        )
        line = [
            np.min(compile_times),
            int(np.round(np.mean(compile_times))),
            int(np.round(stats.trim_mean(compile_times, 0.01))),
            int(np.round(stats.trim_mean(compile_times, 0.05))),
            np.max(compile_times),
            np.min(device_runtimes),
            int(np.round(np.mean(device_runtimes))),
            int(np.round(stats.trim_mean(device_runtimes, 0.01))),
            int(np.round(stats.trim_mean(device_runtimes, 0.05))),
            np.max(device_runtimes),
        ]

        graph_totals = [graph_totals[i] + line[i] for i in range(10)]
        graph_count += 1

        if pre_guids:
            guid_num = len(pre_guids[graph_name])
            op_count += guid_num
            if guid_num <= eager_node_threshold:
                graph_upto_threshold_nodes_totals = [
                    graph_upto_threshold_nodes_totals[i] + line[i] for i in range(10)
                ]
                graph_count_with_nodes_below_threshold += 1
                op_count_with_nodes_below_threshold += guid_num

    res = []
    if graph_count:
        tmp = list(map(int, np.round(np.divide(graph_totals, graph_count))))
        res += [["per_graph"] + tmp]
    if op_count:
        tmp = list(map(int, np.round(np.divide(graph_totals, op_count))))
        res += [["per_node"] + tmp]
    if graph_count_with_nodes_below_threshold:
        tmp = list(
            map(
                int,
                np.round(
                    np.divide(
                        graph_upto_threshold_nodes_totals,
                        graph_count_with_nodes_below_threshold,
                    )
                ),
            )
        )
        res += [[f"per_graph_upto_{eager_node_threshold}_threshold"] + tmp]
    if op_count_with_nodes_below_threshold:
        tmp = list(
            map(
                int,
                np.round(
                    np.divide(
                        graph_upto_threshold_nodes_totals,
                        op_count_with_nodes_below_threshold,
                    )
                ),
            )
        )
        res += [[f"per_node_upto_{eager_node_threshold}_threshold"] + tmp]
    if res:
        res = [["avg of"] + header_line] + res

    res = list(zip(*res))
    print(tabulate.tabulate(res))

    if "failedGraphs" in stats_json:
        print("Nr of graphs failed to compile:", len(stats_json["failedGraphs"]))


def main():
    parser = argparse.ArgumentParser(
        "Read a stats json file generated by run_from_json measurments and summarize the results per graph in csv"
    )
    parser.add_argument(
        "-s", "--stats_file", help="stats json file to parse", required=True
    )
    parser.add_argument(
        "--pre_graph_file", help="graph json file to parse for node types"
    )
    parser.add_argument(
        "--post_graph_file", help="graph json file to parse for node types"
    )
    parser.add_argument("-o", "--output_csv", help="output csv")
    args = parser.parse_args()

    stats_json, pre_guids, post_guids, post_guids_by_engine = load_data(
        args.stats_file, args.pre_graph_file, args.post_graph_file
    )
    summrize_stats_json_to_csv(
        stats_json,
        pre_guids,
        post_guids,
        post_guids_by_engine,
        args.output_csv or (args.stats_file + ".csv"),
    )
    print_summary(stats_json, pre_guids)


if __name__ == "__main__":
    main()
