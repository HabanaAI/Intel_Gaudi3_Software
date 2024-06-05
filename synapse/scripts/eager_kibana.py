#!/usr/bin/env python3
#
#   Run  number of benchmarks to generate the eager kibana xml
#
import argparse
import dataclasses
from io import TextIOWrapper
import json
import os
from typing import Any, Callable, Dict, List, Optional
import pickle
import json_runner
import numpy as np
from scipy import stats


class Config:
    @staticmethod
    def find_json_runner_exe_path() -> str:
        build_folder = os.getenv("SYNAPSE_RELEASE_BUILD")
        assert build_folder is not None
        res = os.path.join(build_folder, "bin", "json_tests")
        assert os.path.isfile(res)
        return res

    @staticmethod
    def find_tcmalloc_path() -> str:
        res = "/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4"
        if not os.path.isfile(res):
            res = "/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4"
        assert os.path.isfile(res)
        return res

    # @staticmethod
    # def get_base_dir_old() -> str:
    #     return "/software/data/eager/models/"

    @staticmethod
    def get_base_dir_new() -> str:
        return "/git_lfs/data/synapse/tests/eager/benchmark_models/"

    @staticmethod
    def get_env() -> Dict[str, str]:
        return {
            "EAGER_NOPLOT": "true",
            "LD_PRELOAD": Config.find_tcmalloc_path(),
            "ENABLE_EXPERIMENTAL_FLAGS": "1",
            "FORCE_EAGER": "1",
        }


class MeasurmentRunner:
    def __init__(self) -> None:
        # self.base_dir_old = Config.get_base_dir_old()
        self.base_dir_new = Config.get_base_dir_new()
        self.json_exe = Config.find_json_runner_exe_path()
        self.curr_env = Config.get_env()

    def run_measure_ir(
        self,
        json_fn: str,
        fn_pat: str,
        base_dir: str,
        device: str,
        env_flags: Dict[str, str] = {},
    ) -> Dict[str, json_runner.IrReport]:
        assert device in ("gaudi2", "gaudi3")
        args = [
            self.json_exe,
            "playback",
            "--device-type",
            device,
            "--compilation-mode",
            "eager",
            "--quiet",
            "--keep-going",
            "--json-file",
            os.path.join(base_dir, json_fn),
        ]

        res = json_runner.run_measure_ir(args, {**self.curr_env, **env_flags}, fn_pat)
        return res

    def run_measure_syntime(
        self,
        json_fn: str,
        iters: int,
        base_dir: str,
        device: str,
        env_flags: Dict[str, str] = {},
    ) -> Dict[str, List[int]]:
        assert device in ("gaudi2", "gaudi3")
        args = [
            self.json_exe,
            "st_perf",
            "--device-type",
            device,
            "--compilation-mode",
            "eager",
            "--quiet",
            "--keep-going",
            "--test-iter",
            str(iters),
            "--stats-file",
            "stats.latest.json",
            "--json-file",
            os.path.join(base_dir, json_fn),
        ]

        @dataclasses.dataclass(frozen=True)
        class ArgsMock:
            stats_json_path: str = "stats.latest.json"
            measure_syntime: List[str] = dataclasses.field(default_factory=list)
            test_iters: Optional[int] = iters

        res = json_runner.run_measure_syntime(args, {**self.curr_env, **env_flags}, ArgsMock())
        return res

    @staticmethod
    def getNumFailed() -> int:
        # requires removing stats.latest.json before the run!
        # TODO: why don't we return it as part of the other measruments in the first place
        stats_fn = "stats.latest.json"
        assert os.path.isfile(stats_fn)
        with open(stats_fn, encoding="utf-8") as f:
            stats_json = json.load(f)
        if "failedGraphs" in stats_json:
            return sum(1 for v in stats_json["failedGraphs"] if v is not None)
        else:
            return 0


class ResDumper:
    def __init__(self, f: TextIOWrapper) -> None:
        self.f = f

    def dump_prefix(self):
        self.f.write('<?xml version="1.0" ?>\n')
        self.f.write("<testsuites>\n")
        self.f.write('    <testsuite name="eager_test_results">\n')

    def dump_res(self, name: str, time: int, table: str = "", **kwargs):
        props_join = " ".join(
            f'properties.{k}="{v}"' for k, v in sorted(kwargs.items())
        )
        table_str = f"table={table}" if table else ""
        self.f.write(
            f'        <testcase name="{name}" time="{time}" {table_str} {props_join} />\n'
        )

    def dump_comment(self, msg):
        self.f.write(f"        <!-- {msg} -->\n")  # TODO: no escaping

    def dump_blank(self):
        self.f.write("\n")

    def dump_suffix(self):
        self.f.write("    </testsuite>\n")
        self.f.write("</testsuites>\n")


def bench_old(
    mr: MeasurmentRunner,
    rd: ResDumper,
    json_fn: str,
    ir_filter: str,
    name: str,
    device: str,
    env_flags: Dict[str, str] = {},
):
    assert device in ("gaudi2", "gaudi3")

    res_ir = mr.run_measure_ir(json_fn, ir_filter, mr.base_dir_new, device, env_flags)
    res_ir_data = res_ir["graph_0_synGraphCompile"].results
    res_syntime = mr.run_measure_syntime(
        json_fn, 10000, mr.base_dir_new, device, env_flags
    )
    res_syntime_data = res_syntime["graph_0_synGraphCompile"]

    rd.dump_res(
        name,
        0,
        mean=int(np.round(np.mean(res_syntime_data))),
        mean_trim_1=int(np.round(stats.trim_mean(res_syntime_data, 0.01))),
        mean_trim_5=int(np.round(stats.trim_mean(res_syntime_data, 0.05))),
        max=np.max(res_syntime_data),
        min=np.min(res_syntime_data),
        cold_run_ir=res_ir_data["cold_run_ir"],
        warm_run_ir=res_ir_data["warm_run_ir"],
    )


def bench_new(
    mr: MeasurmentRunner,
    rd: ResDumper,
    json_fn: str,
    name: str,
    device: str,
    env_flags: Dict[str, str] = {},
):
    ir_filter = "synGraphCompile"

    def execute_benchmark() -> Dict[str, Any]:
        if os.path.exists("stats.latest.json"):
            os.remove("stats.latest.json")
            assert not os.path.exists("stats.latest.json")

        if True:  # normal (non-debug) path
            res_syntime = mr.run_measure_syntime(
                json_fn, 1000, mr.base_dir_new, device, env_flags
            )
            res_fail_num = (
                mr.getNumFailed()
            )  # requires removing stats.latest.json before the run
            return {
                (json_fn, ir_filter): {
                    "res_syntime": res_syntime,
                    "res_fail_num": res_fail_num,
                }
            }
        else:  # Alternative debug path with (optional) pickled data
            pickled_fn = "yummy"
            try:
                with open(pickled_fn, "rb") as f:
                    data = pickle.load(f)
            except Exception:
                data = {}

            if False:  # rerun and update pickled data
                # res_ir = mr.run_measure_ir(json_fn, ir_filter, mr.base_dir_new, device, env_flags)
                res_syntime = mr.run_measure_syntime(
                    json_fn, 1, mr.base_dir_new, device, env_flags
                )
                res_fail_num = mr.getNumFailed()
                data[(json_fn, ir_filter)] = {
                    "res_syntime": res_syntime,
                    "res_fail_num": res_fail_num,
                    # "res_ir": res_ir,
                }
                with open(pickled_fn, "wb") as f:
                    pickle.dump(data, f)
            return data

    data = execute_benchmark()

    with open(f"{mr.base_dir_new}/{json_fn}", encoding="utf-8") as f:
        j = json.load(f)

    res = data[(json_fn, ir_filter)]
    # res_ir_data = res["res_ir"]

    compile_times = {}
    for k, v in res["res_syntime"].items():
        if k.endswith("_synGraphCompile"):
            tmp = k[6:-16]
            assert str(int(tmp)) == tmp
            compile_times[int(k[6:-16])] = v

    fail_num = res["res_fail_num"]
    assert len(compile_times) + fail_num == len(j["graphs"])

    # compile_ir = {}
    # for k, v in res["res_ir"].items():
    #     if k.endswith("_synGraphCompile"):
    #         tmp = k[6:-16]
    #         assert str(int(tmp)) == tmp
    #         compile_ir[int(k[6:-16])] = v
    #         print(tmp)
    # assert len(compile_ir) == len(j["graphs"])

    # assert all(g["name"] == str(i) for i, g in enumerate(j["graphs"]))

    def summarize(func: Callable, graph_or_node: bool, sub11: bool) -> int:
        graphs = j["graphs"]
        if sub11:
            relevant_compile_times = (
                kv for kv in compile_times.items() if len(graphs[kv[0]]["nodes"]) < 11
            )
        else:
            relevant_compile_times = compile_times.items()
        if graph_or_node:  # per-graph
            flat_data = [func(vs) for k, vs in relevant_compile_times]
        else:
            # to count per-node, we consider the mean of the graph as if it appeared,
            # num of node amount of times.
            data = (
                len(graphs[k]["nodes"]) * [func(vs)] for k, vs in relevant_compile_times
            )
            flat_data = [v for lst in data for v in lst]
        return int(np.round(np.mean(flat_data))) if flat_data else 0

    def trim1(xs):
        return stats.trim_mean(xs, 0.01)

    def trim5(xs):
        return stats.trim_mean(xs, 0.05)

    rd.dump_res(
        name,
        0,
        graph_avg_min=summarize(np.min, True, False),
        graph_avg_max=summarize(np.max, True, False),
        graph_avg_mean=summarize(np.mean, True, False),
        graph_avg_mean_trim_1=summarize(trim1, True, False),
        graph_avg_mean_trim_5=summarize(trim5, True, False),
        graph_avg_sub11_min=summarize(np.min, True, True),
        graph_avg_sub11_max=summarize(np.max, True, True),
        graph_avg_sub11_mean=summarize(np.mean, True, True),
        graph_avg_sub11_mean_trim_1=summarize(trim1, True, True),
        graph_avg_sub11_mean_trim_5=summarize(trim5, True, True),
        node_avg_min=summarize(np.min, False, False),
        node_avg_max=summarize(np.max, False, False),
        node_avg_mean=summarize(np.mean, False, False),
        node_avg_mean_trim_1=summarize(trim1, False, False),
        node_avg_mean_trim_5=summarize(trim5, False, False),
        node_avg_sub11_min=summarize(np.min, False, True),
        node_avg_sub11_max=summarize(np.max, False, True),
        node_avg_sub11_mean=summarize(np.mean, False, True),
        node_avg_sub11_mean_trim_1=summarize(trim1, False, True),
        node_avg_sub11_mean_trim_5=summarize(trim5, False, True),
        fail_num=fail_num,
        # cold_run_ir=res_ir_data["cold_run_ir"],
        # warm_run_ir=res_ir_data["warm_run_ir"],
    )


def work(xml_fn):
    mr = MeasurmentRunner()
    with open(xml_fn, "w", encoding="utf-8") as f:
        rd = ResDumper(f)
        rd.dump_prefix()
        rd.dump_blank()

        rd.dump_comment("legacy format")
        rd.dump_blank()
        bench_old(
            mr,
            rd,
            "add_recipe.pre.json",
            "syn[A-Z][A-Za-z0-9_]*",
            "g2.add_recipe_compileTime",
            "gaudi2",
        )
        bench_old(
            mr,
            rd,
            "gemm_recipe.pre.json",
            "synGraphCompile",
            "g2.gemm_recipe_compileTime",
            "gaudi2",
        )
        bench_old(
            mr,
            rd,
            "add_recipe.pre.json",
            "syn[A-Z][A-Za-z0-9_]*",
            "g3.add_recipe_compileTime",
            "gaudi3",
        )
        bench_old(
            mr,
            rd,
            "gemm_recipe.pre.json",
            "synGraphCompile",
            "g3.gemm_recipe_compileTime",
            "gaudi3",
        )
        rd.dump_blank()

        rd.dump_comment("whole network avgs")
        rd.dump_blank()
        bench_new(mr, rd, "latest.bert_ft.json", "g2.bert_ft", "gaudi2")
        bench_new(mr, rd, "latest.bert_p1.json", "g2.bert_p1", "gaudi2")
        bench_new(mr, rd, "latest.bert_p2.json", "g2.bert_p2", "gaudi2")
        bench_new(mr, rd, "latest.lamma7B.json", "g2.lamma7B", "gaudi2")
        bench_new(mr, rd, "latest.resnet.json", "g2.resnet", "gaudi2")
        bench_new(mr, rd, "latest.unet2d.json", "g2.unet2d", "gaudi2")
        bench_new(mr, rd, "latest.unet3d.json", "g2.unet3d", "gaudi2")
        rd.dump_blank()
        bench_new(mr, rd, "latest.bert_ft.json", "g3.bert_ft", "gaudi3")
        bench_new(mr, rd, "latest.bert_p1.json", "g3.bert_p1", "gaudi3")
        bench_new(mr, rd, "latest.bert_p2.json", "g3.bert_p2", "gaudi3")
        bench_new(mr, rd, "latest.lamma7B.json", "g3.lamma7B", "gaudi3")
        bench_new(mr, rd, "latest.resnet.json", "g3.resnet", "gaudi3")
        bench_new(mr, rd, "latest.unet2d.json", "g3.unet2d", "gaudi3")
        bench_new(mr, rd, "latest.unet3d.json", "g3.unet3d", "gaudi3")
        rd.dump_blank()

        rd.dump_comment("whole network avgs optimization features")
        rd.dump_blank()
        bench_new(
            mr,
            rd,
            "latest.resnet.json",
            "g2.resnet",
            "gaudi2",
            {"MME_DESCRIPTORS_CACHE_SIZE": "0"},
        )
        rd.dump_blank()

        rd.dump_suffix()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-x",
        "--output_xml",
        nargs="?",
        const="eager_report.xml",
        help="xml output file",
    )
    args = parser.parse_args()

    return work(args.output_xml)


if __name__ == "__main__":
    main()
