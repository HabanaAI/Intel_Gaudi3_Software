#!/usr/bin/env python
import argparse
import errno
import glob
import json
import os
import re
import time
from typing import List
import subprocess

PLUGINS_FOLDER = os.environ.get("HABANA_PLUGINS_LIB_PATH")
GRAPH_COMPILER_PLUGIN = os.path.join(PLUGINS_FOLDER, "libGraphCompilerPlugin.so")

compression = {
    "none": 0,
    "lz4": 1
}

validation = ["nan","inf"]

def create_folder(folder):
    try:
        os.makedirs(folder)
    except OSError as exc:
        if not (exc.errno == errno.EEXIST and os.path.isdir(folder)):
            raise RuntimeError("failed to create results folder")


def run_external_in_bash(args: List[str], venv=None, log=None, test_execution= None):
    args = [re.escape(arg) for arg in args]
    if venv:
        args = ["source", venv, "&&", "cd", os.getcwd(), "&&"] + args
    args_str = " ".join(args)
    bash_flags = '-c' if test_execution else '-ic'
    cmd = f'bash {bash_flags} "{args_str}"'
    process = subprocess.run(
        args=cmd,
        shell=True,
        env=os.environ
    )
    return process.returncode

def get_files(folder, postfix):
    if not folder:
        return
    yield from (
        f
        for f in glob.iglob(f"{folder}/**/*{postfix}", recursive=True)
        if os.path.isfile(f)
    )


def set_graphs_groups_according_to_process_id_field_if_exist(graphs, process_id_to_group):
    for graph in graphs:
        if "process_id" in graph:
            if graph["process_id"] in process_id_to_group:
                graph["group"] = process_id_to_group[graph["process_id"]]
            else:
                graph["group"] = len(process_id_to_group)
                process_id_to_group[graph["process_id"]] = graph["group"]


def split_file(args):
    with open(args.split_file) as f:
        data = json.load(f)
    os.makedirs(os.path.abspath(args.path))
    global_config = data.get("global_config", None)
    for g in data.get("graphs"):
        file_name = f'{g.get("name").replace(".graph_dumps/", "")}.{g.get("group")}.json'
        graph_data = {}
        graph_data["global_config"] = global_config
        graph_data["graphs"] = [g]
        with open(os.path.join(os.path.abspath(args.path), file_name), "w") as f:
            json.dump(graph_data, f)
    print(f"Split graphs path: {os.path.abspath(args.path)}")


def join_graph_files(args, output_file_path):
    input_files_paths = None
    if args.join_files:
        input_files_paths = args.join_files
    if args.join_folder:
        input_files_paths = get_files(args.join_folder, ".json")
    if input_files_paths is None:
        return None
    model = None
    model_name = os.path.basename(args.path)
    process_id_to_group = {}
    for file_path in input_files_paths:
        with open(file_path) as f:
            data = json.load(f)
        if "graphs" in data:
            set_graphs_groups_according_to_process_id_field_if_exist(data["graphs"], process_id_to_group)
            if model is None:
                model = data
                model["name"] = model_name
            else:
                model["graphs"] = model.get("graphs") + data.get("graphs")
    with open(output_file_path, "w") as f:
        json.dump(model, f, sort_keys=True)
    return output_file_path


def verify_local_file_path(file_path):
    path = file_path
    while (not os.path.exists(path)):
        path = os.path.dirname(path)
    try:
        subprocess.check_output(['df', '-l', path])
    except:
        raise RuntimeError(f"The path {file_path} is not on local folder, output destination must be on local folder (i.e. /tmp/<path>)")


def validate_file_path(file_path, overwrite):
    if os.path.exists(file_path):
        if overwrite:
            print(
                f"Destination path is already exists, deleting the requested destination file: {file_path}.")
            os.remove(file_path)
        else:
            raise RuntimeError(f"The requested file path: {file_path} already exists, run with '--overwrite' to force delete, aborting.")


def main(args):
    if not os.path.exists(GRAPH_COMPILER_PLUGIN):
        print(
            f"Couldn't find libGraphCompilerPlugin.so at: {GRAPH_COMPILER_PLUGIN}. Please fix your env and try again, aborting.")
        exit(-1)
    conf_file_path = f'/tmp/synrec_conf-{time.strftime("%H%M%S_%d%m%y")}.{os.getpid()}.json' if args.conf is None else args.conf
    conf_data = dict()
    values = dict()
    graph = dict()
    tensors = dict()
    plugin = {"enable": True, "lib": "libGraphCompilerPlugin.so",
              "name": "GraphCompiler"}
    graphPath = os.path.abspath(args.path)
    tensorsPath = os.path.abspath(args.path)
    if args.record_tensors_data:
        verify_local_file_path(tensorsPath)
    if args.split or args.pass_name or args.split_ranks:
        if args.split:
            values["split_type"] = {"value": "graph"}
        if args.split_ranks:
            values["split_type"] = {"value": "rank"}
        create_folder(args.path)
    else:
        graphPath += ".json"
        tensorsPath += ".db"
        graph["split"] = {"value": False}
        tensors["split"] = {"value": False}
        validate_file_path(graphPath, args.overwrite)
        validate_file_path(tensorsPath, args.overwrite)

    if args.split_file:
        split_file(args)
        return 0
    joined_file_path = join_graph_files(args, graphPath)
    if joined_file_path is not None:
        print(f"Joined graphs path: {joined_file_path}")
        return 0
    graph["path"] = {"value": graphPath}
    pre = True if "pre" in args.graph_states else False
    post = True if "post" in args.graph_states else False
    passes = True if "passes" in args.graph_states or args.pass_name else False
    graph["type"] = {}
    graph["type"]["pre"] = {"value": pre}
    graph["type"]["post"] = {"value": post}
    graph["type"]["passes"] = {"enable": {"value": passes}}
    if args.pass_name:
        graph["type"]["passes"]["filter"] = {"value": args.pass_name}
    if args.record_tensors_data:
        tensors["path"] = {"value": tensorsPath}
    if args.min_iter is not None:
        tensors["min_iter"] = {"value": args.min_iter}
    if args.max_iter is not None:
        tensors["max_iter"] = {"value": args.max_iter}
    if args.compression:
        tensors["compression"] = {"value": compression.get(args.compression)}
    if args.graphs:
        values["graphs_filter"] = {"value": args.graphs}
    tensors["validation"] = {"value": validation}
    tensors["rec_failures"] = {"value": args.rec_failures}
    tensors["stop_on_failure"] = {"value": args.stop_on_failure}
    tensors["const"] = {"value": args.const}

    if args.last_iters:
        tensors["last_iters"] = {"value": args.last_iters}
    if args.filter_by_elements:
        tensors["filter_by_elements"] = {"value": args.filter_by_elements}
    if args.ignore_errors:
        values["ignore_errors"] = True
    if args.single_process:
        values["ranks"] = [int(0)]
    if args.ranks:
        values["ranks"] = [int(i) for i in args.ranks]

    values["graph"] = graph
    values["tensors"] = tensors
    plugin["values"] = values
    conf_data["Plugins"] = [plugin]

    with open(conf_file_path, "w") as f:
        json.dump(conf_data, f, indent=4, sort_keys=True)

    if args.conf is None:
        cmd = [f"HABANA_PROF_CONFIG={conf_file_path}"] + args.cmd
        status = run_external_in_bash(args=cmd,
                                      venv=args.venv,
                                      test_execution=args.test)
    else:
        print(f"Configuration file path: {os.path.abspath(conf_file_path)}")
        print(f"To record <cmd> run: HABANA_PROF_CONFIG={os.path.abspath(conf_file_path)} <cmd>")
        return 0

    if not args.keep_conf:
        os.remove(conf_file_path)
    else:
        print(f"Configuration file path: {os.path.abspath(conf_file_path)}")

    if os.path.exists(graphPath):
        print(f"Record graph path: {graphPath}")
    else:
        print(f"Error: Failed to record graph")
    if args.record_tensors_data:
        if os.path.exists(tensorsPath):
            print(f"Record tensors path: {tensorsPath}")
        else:
            print(f"Error: Failed to record tensors")

    return status

def set_synrec_init_state(enable):
    if enable:
        os.environ["SYNREC"] = "1"
    else:
        os.environ["SYNREC"] = "0"
    os.environ["SYNREC_INIT"] = "1"

def parse_args():
    parser = argparse.ArgumentParser(
        add_help=True, usage=f'{os.path.basename(__file__)} <args> -- <cmd>')
    parser.add_argument(
        '-p', '--path', help='File/Folder path, extension is added in case --split is not set', default=f'synrec-{time.strftime("%Y%m%d_%H%M%S")}'
    )
    parser.add_argument(
        "-t", "--tensors-data", dest="record_tensors_data", action="store_true", help="Record persistent tensors data",
    )
    parser.add_argument(
        "--const", action="store_true", help="Skip persistent tensors, record only const tensors data",
    )
    split_group = parser.add_mutually_exclusive_group()
    split_group.add_argument(
        "-s", "--split", action="store_true", help="Split synapse graphs/tensors to multiple files (graph per file)",
    )
    split_group.add_argument(
        "--split-ranks", action="store_true", help="Split synapse graphs/tensors to multiple files (rank per file)",
    )
    parser.add_argument(
        "--min-iter", type=int, help="Run iteration to start tensors data record",
    )
    parser.add_argument(
        "--max-iter", type=int, help="Run iteration to stop tensors data record",
    )
    parser.add_argument(
        "--last-iters", type=int, help="Keep last N iterations of tensors data record",
    )
    parser.add_argument(
        "--filter-by-elements", type=int, help="Record only const tensors with specified number of elements",
    )
    parser.add_argument(
        "--metadata", action="store_true", help="Record only tensors metadata (without tensor data)",
    )
    parser.add_argument(
        "--ignore-errors", action="store_true", help="Force record in case of recoverable errors",
    )
    parser.add_argument(
        "-w", "--wait-for-record", action="store_true", help="Do not start recording from the beginning of the model."
        " In this mode, one can use calls to start/stop recording API as in:"
        " pytorch-integration/python_packages/habana_frameworks/torch/utils/synrec_helper.py",
    )
    parser.add_argument(
        "cmd",
        nargs=argparse.REMAINDER,
        help="The command to record"
    )
    parser.add_argument(
        "-k", "--keep-conf", action="store_true", help="Keep configuration file",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="In case the recording destination path already exists, delete it before the run",
    )
    parser.add_argument(
        "--graph-states",
        nargs="+",
        choices=["pre", "post", "passes"],
        help="Select graph state/s to be serialized, pre graph is selected by default",
    )
    parser.add_argument(
        '--pass-name', help='Record the specified pass (single pass). If set, --graph-types passes is enabled and --split is set automatically',
    )
    parser.add_argument(
        "--compression",
        choices=compression.keys(),
        default="lz4",
        help="Select data compression type",
    )
    parser.add_argument(
        "--venv",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--conf", help="Generate config file to run with HABANA_PROF_CONFIG=<config_file_path>, user command is not recorded",
    )
    parser.add_argument(
        "--graphs",
        nargs="+",
        help="List of graphs names (or partial names) to record, other graphs will be skipped",
    )
    parser.add_argument(
        "--rec-failures",
        action="store_true",
        help="Validate output tensors and record only the failed graphs",
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Fail synLaunch in case of failure in output tensors validation",
    )
    join_group = parser.add_mutually_exclusive_group()
    join_group.add_argument(
        "--join-files",
        nargs="+",
        help="Join multiple Json graphs files to single json file (model), user command is not recorded",
    )
    join_group.add_argument(
        "--join-folder",
        help="Join multiple Json graphs files from a folder to single json file (model), user command is not recorded",
    )
    join_group.add_argument(
        "--split-file",
        help="Split single model file to multiple single graph files, user command is not recorded",
    )
    parser.add_argument(
        # hidden flag. should be turned on when synrec is executed from tests
        "--test",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    process_filter_group = parser.add_mutually_exclusive_group()
    process_filter_group.add_argument(
        "--single-process", action="store_true", help="Record only the first process",
    )
    process_filter_group.add_argument(
        "--ranks",
        type=int,
        nargs="+",
        help="Record only the specified rank IDs",
    )
    args = parser.parse_args()
    set_synrec_init_state(not args.wait_for_record)
    args.cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
    if not args.cmd and args.conf is None and args.join_files is None and args.join_folder is None and args.split_file is None:
        parser.error('Nothing to do, missing command to record')
    if args.metadata:
        if not args.record_tensors_data:
            parser.error('Argument --metadata is relevant only if --tensors-data is set')
        if args.min_iter is not None or args.max_iter is not None:
            parser.error('Argument --metadata is not allowed with --min-iter or --max-iter')
        args.min_iter = -1
        args.max_iter = -1
    if not args.graph_states:
        args.graph_states = ["passes"] if args.pass_name else ["pre"]
    if args.graph_states and len(args.graph_states) > 1 and not args.split:
        parser.error('Multiple graph states recording is allowed only with --split')
    if args.last_iters and args.last_iters <= 0:
        parser.error('--last-iters must be positive integer')
    if not args.venv:
        supported_venvs = ["QNPU_PATH", "VIRTUAL_ENV", "CONDA_PREFIX"]
        for venv in supported_venvs:
            venv_folder = os.environ.get(venv, None)
            if venv_folder:
                args.venv = f'{venv_folder}/bin/activate'
                break
    if args.venv:
        print(f"Recording from venv: {args.venv}")
    return args


if __name__ == "__main__":
    args = parse_args()
    exit(main(args))
