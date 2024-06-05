#!/usr/bin/env python3

import argparse
import glob
import json
import math
import os
import time
from typing import Dict, List


class ModelInfo:
    def __init__(self, name, compile_mem_usage, run_mem_usage):
        self.name = name
        self.compile_mem_usage = compile_mem_usage
        self.run_mem_usage = run_mem_usage


def get_files(folder, postfix):
    if not folder:
        return
    yield from (
        f
        for f in glob.iglob(f"{folder}/**/*{postfix}", recursive=True)
        if os.path.isfile(f)
    )


def get_all_models(folder):
    models = {}
    for tf in get_files(folder, ".com.json"):
        models[os.path.basename(tf).replace(".com.json", "")] = tf
    return models


def get_models_infos(folder) -> List[ModelInfo]:
    ret = []
    models = get_all_models(folder)
    for k, v in models.items():
        print(f"processing model: {k}")
        with open(v) as f:
            jd = json.load(f)
            compile_data = jd.get("compile")
            if compile_data:
                compile_max_mem_usage = compile_data.get("max_memory_usage")
                compile_max_mem_usage = int(math.ceil(compile_max_mem_usage * 1e-7) * 1e7)
            else:
                compile_max_mem_usage = None
            run_data = jd.get("run")
            if run_data:
                run_max_mem_usage = run_data.get("max_memory_usage")
                run_max_mem_usage = int(math.ceil(run_max_mem_usage * 1e-7) * 1e7)
            else:
                run_max_mem_usage = None
        ret.append(ModelInfo(k, compile_max_mem_usage, run_max_mem_usage))
    return ret


def models_to_dict(models_infos: List[ModelInfo]) -> Dict[str, int]:
    ret = {}
    for mi in models_infos:
        ret[mi.name] = {}
        if mi.compile_mem_usage:
            ret[mi.name]["compile"] = mi.compile_mem_usage
        if mi.run_mem_usage:
            ret[mi.name]["run"] = mi.run_mem_usage
    return ret


def main(args):
    models_infos = get_models_infos(args.mpm_test_folder)
    model_to_mem = models_to_dict(models_infos)
    output: Dict[str, Dict[str, int]] = {}
    if args.input_file:
        with open(args.input_file) as f:
            output = json.load(f)
    for d in args.chip_types:
        if d in output:
            output[d].update(model_to_mem)
        else:
            output[d] = model_to_mem
    with open(args.output_file, "w") as f:
        json.dump(output, f, indent=4, sort_keys=True)
    print(f"output file path: {args.output_file}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--mpm-test-folder",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output-file",
        default=f"models_mem_usage-{time.strftime('%Y%m%d-%H%M%S')}.json",
    )
    parser.add_argument(
        "-i",
        "--input-file",
        help="add to existing file",
    )
    parser.add_argument(
        "-c",
        "--chip-types",
        nargs="+",
        choices=["gaudi", "gaudi2", "gaudi3"],
        help="Select chip type",
        required=True,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    exit(main(args))
