import json
import os
import sys

# Extract one or multiple sub-graphs from a given network into new file with same base name extended by graph index boundaries

params_nr = len(sys.argv)
if params_nr != 3 and params_nr != 4:
    print("Usage: python $HOME/trees/npu-stack/synapse/src/eager/scripts/sub_graph_extractor.py <name>.json <from index> <to index (exclusive and optional)>")
    exit(1)
start_idx = int(sys.argv[2])
end_idx = int(sys.argv[3]) if params_nr == 4 else start_idx
assert start_idx <= end_idx

with open(sys.argv[1], 'r') as json_file:
    source_data = json.load(json_file)
    sections_to_move = {
        "global_config": source_data["global_config"],
        "graphs": []
    }
    graphs = source_data["graphs"]
    graph_idx = 0
    for i in range(start_idx, end_idx + 1):
        graph = graphs[i]
        graph["name"] = str(graph_idx)
        sections_to_move["graphs"].append(graph)
        graph_idx += 1

base_name, extension = os.path.splitext(sys.argv[1])
dest_json_file_name = "{}_{}_{}.json".format(base_name, start_idx, end_idx)
with open(dest_json_file_name, 'w') as new_file:
    json.dump(sections_to_move, new_file, indent=1)
