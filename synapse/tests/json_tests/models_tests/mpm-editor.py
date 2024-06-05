#!/usr/bin/env python3
import argparse
import json
import os
from typing import Dict, List

from mpm_model_list import Filter, ModelMetadata, ModelsList
from mpm_types import Devices, ModelStats


def add(
    input: str,
    output: str,
    models: List[str],
    devices: List[str],
    jobs: List[str],
    description: str,
):
    if not devices:
        raise RuntimeError(
            "adding filter to the models file requires to specify at least one device"
        )
    models_list = ModelsList()
    if input and os.path.exists(input):
        with open(input) as f:
            models_list = ModelsList.deserialize(json.load(f))
    if not models:
        models = models_list.models.keys()
    if not jobs:
        jobs = [None]
    for d in devices:
        for j in jobs:
            filter = Filter(d, j)
            for m in models:
                mi = models_list.get_info(m)
                if not mi:
                    mi = ModelMetadata()
                    models_list.add(m, mi)
                mi.add(filter, description)
    with open(output, "w") as f:
        json.dump(models_list.serialize(), f, indent=4, sort_keys=True)
    print(
        f"new models/filters were added to models file {output}, models: {models}, devices: {devices}, jobs: {jobs}"
    )


def remove(
    input: str, output: str, models: List[str], devices: List[str], jobs: List[str]
):
    with open(input) as f:
        models_list = ModelsList.deserialize(json.load(f))
    if not models:
        models = models_list.models.keys()
    if devices is None:
        for m in models:
            models_list.remove(m)
    else:
        if not jobs:
            jobs = [None]
        for d in devices:
            for j in jobs:
                filter = Filter(d, j)
                for m in models:
                    mi = models_list.get_info(m)
                    if not mi:
                        raise RuntimeError(
                            f"model: {m} is not in the models file and can not be removed"
                        )
                    mi.remove(filter)
    with open(output, "w") as f:
        json.dump(models_list.serialize(), f, indent=4, sort_keys=True)
    print(
        f"models/filters were removed from models file {output}, models: {models}, devices: {devices}, jobs: {jobs}"
    )


def update(
    input_file_path: str,
    output_file_path: str,
    update_file_path: str,
):
    if not os.path.exists(update_file_path):
        raise RuntimeError(
            f"update file path doesn't exists: {update_file_path}"
        )

    update_models_list = ModelsList()
    with open(update_file_path) as f:
        update_models_list = ModelsList.deserialize(json.load(f))

    models_list = ModelsList()
    if input_file_path and os.path.exists(input_file_path):
        with open(input_file_path) as f:
            models_list = ModelsList.deserialize(json.load(f))

    for model_name, mmd in update_models_list.models.items():
        model_metadata: ModelMetadata = models_list.models.setdefault(model_name, mmd)
        for device_type, mdi in mmd.devices.items():
            if device_type not in model_metadata.devices:
                model_metadata.devices[device_type] = mdi
            else:
                model_metadata.devices[device_type].stats = mdi.stats
            if mdi.jobs:
                model_metadata.devices[device_type].jobs = list(set(model_metadata.devices[device_type].jobs + mdi.jobs))

    with open(output_file_path, "w") as f:
        json.dump(models_list.serialize(), f, indent=4, sort_keys=True)
    print(
        f"models from {update_file_path} were added to models file {output_file_path}"
    )


def get_info(input: str) -> Dict[str, Dict[str, List[str]]]:
    with open(input) as f:
        models_list = ModelsList.deserialize(json.load(f))
    info: Dict[str, Dict[str, List[str]]] = {}
    for m, mi in models_list.models.items():
        for d, mdi in mi.devices.items():
            device = info.setdefault(d, {})
            for j in mdi.jobs:
                job = device.setdefault(j, [])
                job.append(m)
    return info


def filter_by_models(
    input: str, models: List[str], devices: List[str]
) -> Dict[str, ModelMetadata]:
    ret: Dict[str, ModelMetadata] = {}
    with open(input) as f:
        models_list = ModelsList.deserialize(json.load(f))
    for m, mi in models_list.models.items():
        if not mi or not mi.devices:
            continue
        if models and not [x for x in models if x in m]:
            continue
        for d in mi.devices.keys():
            if devices and d not in devices:
                continue
            ret[m] = mi
    return ret


def export(
    file_path: str, input: str, models: List[str], devices: List[str], jobs: List[str]
):
    filtered_models = filter_by_models(input, models, devices)
    csv_lines = []
    index = 0
    for m, mi in filtered_models.items():
        for d, v in mi.devices.items():
            if devices and d not in devices:
                continue
            if not (set(v.jobs).intersection(jobs) if jobs else v.jobs):
                continue
            csv_lines.append({"#": index, "Model": m, "Device": d, "Compile Mem": v.stats.compile.max_memory_usage, "Compile Duration": v.stats.compile.duration, "Run Mem": v.stats.run.max_memory_usage, "Run Duration": v.stats.run.duration, "Jobs": " ".join(v.jobs), "Description": mi.description})
            index += 1
    if csv_lines:
        header = csv_lines[0].keys()
        with open(file_path, "w") as f:
            f.write(f'{",".join(header)}\n')
            for l in csv_lines:
                line = [str(l.get(h, "---")) for h in header]
                f.write(f'{",".join(line)}\n')
        print(f"Export models list to CSV file: {file_path}")


def list_models_sort_by_models(
    input: str, models: List[str], devices: List[str], jobs: List[str]
):
    filtered_models = filter_by_models(input, models, devices)
    printed = False
    for m, mi in filtered_models.items():
        msg = ""
        for d, v in mi.devices.items():
            filtered_jobs = set(v.jobs).intersection(jobs) if jobs else v.jobs
            if filtered_jobs:
                msg += f"    Device: {d}, Jobs: {', '.join(filtered_jobs)}{os.linesep}"
        if msg:
            printed = True
            print(
                f"Model: {m} {os.linesep}  Description:{os.linesep}    {mi.description}{os.linesep}  Filters:{os.linesep}{msg}"
            )
    if not printed:
        print(
            f"Devices filter: {devices} with jobs filter: {jobs} produce empty models list"
        )


def list_models_sort_by_filter(input: str, models: List[str], devices: List[str], jobs: List[str]):
    info = get_info(input)
    printed = False
    for d, _jobs in info.items():
        if devices and d not in devices:
            continue
        msg = ""
        for j, _models in _jobs.items():
            if jobs and j not in jobs:
                continue
            if models:
                _models = [x for x in _models if x in models]
            msg += f"{os.linesep}  Job: {j}{os.linesep}    "
            msg += f"{os.linesep}    ".join(_models)
            msg += os.linesep
        if msg:
            printed = True
            print(f"Device: {d}{msg}{os.linesep}")
    if not printed:
        print(
            f"Devices filter: {devices} with jobs filter: {jobs} produce empty models list"
        )

def ms_to_string(time) -> str:
    seconds = time * 1e-3
    return f"{int(seconds/60)}:{int(seconds % 60)}"


def calc_execution_time(
    input: str, models: List[str], devices: List[str], jobs: List[str]
):
    filtered_models = filter_by_models(input, models, devices)
    times: Dict[str, Dict[str, (ModelStats, ModelStats)]] = {}
    for mi in filtered_models.values():
        for d, v in mi.devices.items():
            filtered_jobs = set(v.jobs).intersection(jobs) if jobs else v.jobs
            times.setdefault(d, {})
            for j in filtered_jobs:
                times[d].setdefault(j, (ModelStats(), ModelStats()))
                if v.stats.compile.duration > times[d][j][0].compile.duration:
                    times[d][j][0].compile.duration = v.stats.compile.duration
                if v.stats.run.duration > times[d][j][0].run.duration:
                    times[d][j][0].run.duration = v.stats.run.duration
                times[d][j][1].compile.duration += v.stats.compile.duration
                times[d][j][1].run.duration += v.stats.run.duration
    if not times:
        print(
            f"Devices filter: {devices} with jobs filter: {jobs} produce empty models list"
        )
    print("Execution Time:")
    for d, v in times.items():
        if not v:
            continue
        print(f"    Device: {d}")
        for j, jv in v.items():
            print("        Jobs:")
            print(f"            {j}:")
            print(f"                Max compile time: {ms_to_string(jv[0].compile.duration)} minutes")
            print(f"                Total compile time: {ms_to_string(jv[1].compile.duration)} minutes")
            print(f"                Max run time: {ms_to_string(jv[0].run.duration)} minutes")
            print(f"                Total run time: {ms_to_string(jv[1].run.duration)} minutes")
        print()


def extend_models_partial_name(models_file: str, models: List[str]) -> List[str]:
    with open(models_file) as f:
        models_list = ModelsList.deserialize(json.load(f))
    models = []
    for m in models_list.models:
        models += [m for x in args.models if x in m]
    if models:
        return list(set(models))
    return models


def main(args):
    if args.remove:
        remove(args.input, args.output, args.models, args.chip_types, args.jobs)
        exit(0)
    if args.models:
        args.models = extend_models_partial_name(args.input, args.models)
    if args.add:
        add(
            None if args.create else args.input,
            args.output,
            args.models,
            args.chip_types,
            args.jobs,
            args.description,
        )
        exit(0)
    if args.update:
        update(
            None if args.create else args.input,
            args.output,
            args.update
        )
        exit(0)
    if args.list:
        print(f"List models in file: {args.input}")
        if args.sort == "model":
            list_models_sort_by_models(
                args.input, args.models, args.chip_types, args.jobs
            )
        if args.sort == "filters":
            list_models_sort_by_filter(args.input, args.models, args.chip_types, args.jobs)
        exit(0)
    if args.exe_time:
        print(f"Calculate total execution time of models in file: {args.input}")
        calc_execution_time(
            args.input, args.models, args.chip_types, args.jobs
        )
        exit(0)
    if args.export:
        export(args.export, args.input, args.models, args.chip_types, args.jobs)
    exit(0)


def validate_args(parser: argparse.ArgumentParser):
    args = parser.parse_args()
    if (args.add or args.remove or args.update) and not args.output:
        args.output = args.input
    if args.remove and not (args.input and args.output):
        parser.error("Arguments --input and --output are required for --remove")
    if args.description:
        args.description = " ".join(args.description)
    return args


def get_default_models_file() -> str:
    paths = [
        os.path.join(
            os.environ.get("HABANA_NPU_STACK_PATH"),
            "mpm-test-data",
            "models",
            ".default.models-list.json",
        ),
        "/tmp/.mpm/models/.default.models-list.json",
    ]
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def get_jobs(models_file: str):
    if models_file is None:
        return []
    try:
        with open(models_file) as f:
            ml = ModelsList.deserialize(json.load(f))
        return list(ml.get_jobs())
    except:
        return []


def parse_args():
    dep_parser = argparse.ArgumentParser(add_help=False)
    dep_parser.add_argument(
        "-i", "--input", help="input models file", default=get_default_models_file()
    )

    dep_args, _ = dep_parser.parse_known_args()
    parser = argparse.ArgumentParser(parents=[dep_parser])

    parser.add_argument(
        "-o",
        "--output",
        help="output models file",
    )
    cmd_parser = parser.add_mutually_exclusive_group()
    cmd_parser.add_argument(
        "-a",
        "--add",
        action="store_true",
        help="Update the models file at (--input) path or create an empty models list if not set."
        "The requested models (--models) will be updated with the requested chip type (--chip-types) if given, with the requested job type (--jobs) if given"
        "and with the requested description (--description) if given.",
    )
    cmd_parser.add_argument(
        "-r",
        "--remove",
        action="store_true",
        help="Update the models file at (--input) path.\n"
        "The combined filter (--chip-types and --jobs) will be removed for all models (--models).",
    )
    cmd_parser.add_argument(
        "-u",
        "--update",
        help="File path to a models file. Update the models in (--input) and write the new list to (--output)"
    )
    cmd_parser.add_argument(
        "--exe-time",
        action="store_true",
        help="Calculate total execution time"
    )
    cmd_parser.add_argument(
        "-l",
        "--list",
        action="store_true",
        help="Show list of models with their filters list of a models file at (--input) path.\n"
        "The requested models (--models) will be listed.",
    )
    parser.add_argument(
        "--create",
        action="store_true",
        help="Create new file instead of using input file.",
    )
    parser.add_argument(
        "-c",
        "--chip-types",
        nargs="+",
        choices=[d.value for d in Devices],
        help="Select chip type",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        nargs="+",
        help=f"Filter models by job type, current options: {get_jobs(dep_args.input)}",
    )
    parser.add_argument(
        "--description",
        nargs="+",
        help="Model's description text",
    )
    parser.add_argument(
        "-s",
        "--sort",
        choices=["filters", "model"],
        default="filters",
        help="Sort list by model or filters (device and job)",
    )
    parser.add_argument("-m", "--models", nargs="+", help="Select models")
    parser.add_argument(
        "--export",
        help="Write models list to file in CSV format",
    )
    return validate_args(parser)


if __name__ == "__main__":
    args = parse_args()
    main(args)
