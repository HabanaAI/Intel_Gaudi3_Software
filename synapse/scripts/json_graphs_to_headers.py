#!/usr/bin/env python3
import os
import json
import syn_infra as syn

SYNAPSE_ROOT = os.environ.get("SYNAPSE_ROOT")
GRAPHS_FOLDER = os.path.join(SYNAPSE_ROOT, "tests", "gc_tests", "platform_tests", "graphs")
GRAPHS_FILE_PATH = os.path.join(SYNAPSE_ROOT, "tests", "gc_tests", "platform_tests", "graphs.h")

def main(graph_folder, graph_file):
    graphs_str = {}
    graphs_files = sorted(syn.get_files(graph_folder, ".json"))
    for g in graphs_files:
        with open(g) as f:
            data = json.load(f)
            name = os.path.basename(g).replace(".json", "")
            if "global_config" in data:
                del data["global_config"]
            graphs_str[name] = str(data).replace("'",'"').replace("True", "true").replace("False", "false")
    header = "#pragma once\nnamespace json_graphs\n{\n"
    for k, v in graphs_str.items():
        header += f'static const char {k}[] = R"({v})";\n'
    header += "}\n"

    prev_header =  None
    if os.path.exists(graph_file):
        with open(graph_file) as f:
            prev_header = f.read()
    if prev_header != header:
        with open(graph_file, "w") as f:
            f.write(header)
    return 0

if __name__ == "__main__":
    exit(main(GRAPHS_FOLDER, GRAPHS_FILE_PATH))
