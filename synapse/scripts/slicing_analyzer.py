#!/usr/bin/env python
import argparse
import json
import shutil
import syn_infra as syn
import time
import os
import logging
import csv
from zipfile import ZipFile

LOG = syn.config_logger("slicing_analyzer", logging.INFO)
num_of_digits_after_point=3

class ScopedFolder:
    def __init__(self, folder_path):
        LOG.debug(f"Create ScopedFolder to {folder_path}")
        self.folder_path = folder_path
    def __del__(self):
        LOG.debug(f"ScopedFolder removing: {self.folder_path}")
        shutil.rmtree(self.folder_path)

def save_list_to_csv_file(l, file_path):
    with open(file_path, "w") as o_file:
        writer = csv.writer(o_file)
        for row in l:
            writer.writerow(row)

def save_to_csv(data, file_path):
    LOG.info(f"Save data to {file_path}")
    # Assumes elements in data have the same keys
    header = data[0].keys()
    data_to_list = [list(header)]
    for e in data:
        e_to_list = []
        for k in header:
            e_to_list.append(e[k])
        data_to_list.append(e_to_list)
    save_list_to_csv_file(data_to_list, file_path)

def calculate_change(ref, res):
    return ((ref - res) / ref) * 100.0

def add_column_of_calculated_changes_between_col_a_to_col_b(analyzed_mme_origin_nodes, col_a, col_b, name_of_new_column):
    LOG.info(f"Add column {name_of_new_column}")
    for mme_origin_node in analyzed_mme_origin_nodes:
        if (col_a in mme_origin_node and col_b in mme_origin_node):
            mme_origin_node[name_of_new_column] = round(calculate_change(float(mme_origin_node[col_a]), float(mme_origin_node[col_b])),num_of_digits_after_point)

def add_comparison_columns(analyzed_mme_origin_nodes):
    LOG.info("Add comparison columns")
    add_column_of_calculated_changes_between_col_a_to_col_b(analyzed_mme_origin_nodes, "MME Origin Expected (usec)", "Sum Of slices 'MME Expected Compute (usec)'", "MME Expected Compute Gain (%)")

def group_by(ungrouped_list, key):
    m = {}
    for e in ungrouped_list:
        if e[key] is None:
            continue
        if e[key] not in m:
            m[e[key]] = [e]
        else:
            m[e[key]].append(e)
    return m

def create_analyzed_mme_origin_nodes(trace_analyzer):
    # Returns: analyzed_mme_origin_nodes = [{"MME Origin Node": x, "MME Origin Expected (usec)":y, "Sum Of slices 'MME Expected Compute (usec)'" : z},...,{}]
    LOG.info(f"Create analyzed mme origin nodes table")
    analzed_mme_nodes = [node for node in trace_analyzer["analyzer_metadata"] if node["MME Origin Node"] != ""]
    nodes_grouped = group_by(analzed_mme_nodes, "MME Origin Node")
    if nodes_grouped == {}:
        return {}

    analyzed_mme_origin_nodes = []
    for mme_origin_node, grouped_nodes in nodes_grouped.items():
        analyzed_mme_origin_node = {"MME Origin Node": mme_origin_node, "MME Origin Expected (usec)": float(grouped_nodes[0]["MME Origin Expected (usec)"])}
        sliced_sum_expected_compute = 0.0
        for node in grouped_nodes:
            sliced_sum_expected_compute += float(node["MME Expected Compute (usec)"])
        analyzed_mme_origin_node["Sum Of slices 'MME Expected Compute (usec)'"] = round(sliced_sum_expected_compute,num_of_digits_after_point)
        analyzed_mme_origin_nodes.append(analyzed_mme_origin_node)
    return analyzed_mme_origin_nodes

def get_analyzer_file_path_from_hltv(hltv_flie, extraction_path):
    LOG.info("Getting trace analyzer json file from hltv file")
    with ZipFile(hltv_flie) as zip_file:
        zip_file.extract("views/trace_analyzer.json", path=extraction_path)
    return ScopedFolder(extraction_path), os.path.join(extraction_path,"views","trace_analyzer.json")

def main(args):
    if (args.trace_analyzer_json is None):
        extraction_path = f'/tmp/hltv_extraction-{time.strftime("%H%M%S_%d%m%y")}'
        extraction_path_scoped_folder, trace_analyzer_file_path = get_analyzer_file_path_from_hltv(args.hltv_file, extraction_path)
    else:
        trace_analyzer_file_path = args.trace_analyzer_json
    LOG.info(f"Loading json data of {trace_analyzer_file_path}")
    trace_analyzer_file = open(trace_analyzer_file_path, "r")
    trace_analyzer = json.load(trace_analyzer_file)
    analyzed_mme_origin_nodes = create_analyzed_mme_origin_nodes(trace_analyzer)
    if (analyzed_mme_origin_nodes != {}):
        add_comparison_columns(analyzed_mme_origin_nodes)
        analyzed_mme_origin_nodes.sort(key=lambda mme_origin_node:mme_origin_node["MME Expected Compute Gain (%)"])
        save_to_csv(analyzed_mme_origin_nodes, "mme_origin_nodes_analysis.csv")

def parse_args():
    parser = argparse.ArgumentParser(add_help=True)
    files_test_group = parser.add_mutually_exclusive_group()
    files_test_group.add_argument("-f", "--hltv-file", help="An hltv profling file")
    files_test_group.add_argument("-j", "--trace_analyzer_json", help="A trace_anaylzer.json file")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    exit(main(args))