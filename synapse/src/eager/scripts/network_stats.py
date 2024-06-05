import binascii
import csv
import json
import os
import sys

# Collect statistics from json file of a network and export them into CSV file with same base name

if len(sys.argv) != 2:
    print("Usage: python $HOME/trees/npu-stack/synapse/src/eager/scripts/network_stats.py <name>.json")
    exit(1)

with open(sys.argv[1], 'r') as json_file:
    source_data = json.load(json_file)
    graphs = source_data["graphs"]
    graph_stats = dict()

    graph_idx = 0
    for g in graphs:
        graph_stats[graph_idx] = {
            "nodes_nr": len(g["nodes"]),
            "guids": ', '.join([n["guid"] for n in g["nodes"]]),
            "params": binascii.crc32(str([n["params"] for n in g["nodes"]]).encode('utf-8'))}
        graph_idx += 1

    base_name, extension = os.path.splitext(sys.argv[1])
    csv_file_name = base_name + '.csv'
    with open(csv_file_name, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["graph index", "number of nodes", "node GUIDs", "params CRC"])
        for graph_idx, stats in graph_stats.items():
            csv_writer.writerow([graph_idx, stats["nodes_nr"], stats["guids"], stats["params"]])
        print("{} graphs processed in {}".format(len(graphs), csv_file_name))
