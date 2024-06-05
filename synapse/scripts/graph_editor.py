#!/usr/bin/env python
import argparse
import copy
import json
import logging
import os
from typing import List

import json_runner as run
import syn_infra as syn
from graph_editor_types import (GraphsLoader, Node, get_json_data,
                                get_node_tensors_names)

LOG = syn.config_logger("graph_editor", logging.INFO)


def nodes_from_names(names, all_nodes) -> "list[Node]":
    ret: list[Node] = []
    for n in names:
        ret.append(Node(n, all_nodes))
    return ret


def search_forward(root: Node, radius, all_nodes) -> "list[Node]":
    if radius == 0:
        return []
    nodes = nodes_from_names(root.consumers, all_nodes)
    ret: list[Node] = nodes_from_names(root.consumers, all_nodes)
    for n in nodes:
        consumers = search_forward(n, radius - 1, all_nodes)
        if consumers:
            ret += consumers
    return ret


def search_backward(root: Node, radius, all_nodes) -> "list[Node]":
    if radius == 0:
        return []
    nodes = nodes_from_names(root.producers, all_nodes)
    ret: list[Node] = nodes_from_names(root.producers, all_nodes)
    for n in nodes:
        producers = search_backward(n, radius - 1, all_nodes)
        if producers:
            ret += producers
    return ret


def rmv_not_valid_blocking_nodes(graph):
    graph_nodes_names_set = set(n.name for n in graph)
    for node in graph:
        new_blocking_nodes = [
            node_name
            for node_name in node.data["blocking_nodes"]
            if (node_name in graph_nodes_names_set)
        ]
        node.data["blocking_nodes"] = new_blocking_nodes


def get_nodes(graph_data, nodes: List[str], guids: List[str]):
    return (
        nodes or (
            [n["name"] for n in graph_data["nodes"]
            if n["guid"] in args.guids] if guids
            else [n["name"] for n in graph_data["nodes"]]
        )
    )


def generate_graphs(args) -> List[dict]:
    graphs: List[dict] = []
    json_data = get_json_data(args.json_file)
    graph_indices = range(len(json_data["graphs"])) if args.graph_index is None else [args.graph_index]
    for i in graph_indices:
        nodes = get_nodes(json_data["graphs"][i], args.nodes, args.guids)
        for n in nodes:
            graphs.append(generate_graph(json_data, [n], n.replace("/", "_"), i, None, i, args))
    return graphs


def generate_graph(json_data, nodes, postfix, index, bundle_index, graph_index ,args):
    graph = json_data["graphs"][graph_index]
    nodes_list = graph["nodes"]
    nodes_dict = dict()
    for n in nodes_list:
        nodes_dict[n["name"]] = n
    sub_graph = []
    for n in nodes:
        root = Node(n, nodes_dict)
        sub_graph += [root]
        if not args.bwd:
            sub_graph += search_forward(root, args.radius, nodes_dict)
        if not args.fwd:
            sub_graph += search_backward(root, args.radius, nodes_dict)
    rmv_not_valid_blocking_nodes(sub_graph)

    tensors_list = graph["tensors"]
    all_output_tensors_names_origin_graph = set(output_tensor_name for node in nodes_list for output_tensor_name in node["output_tensors"]) - set(input_tensor_name for node in nodes_list for input_tensor_name in node["input_tensors"])
    tensors_dict = dict()
    for t in tensors_list:
        tensors_dict[t["name"]] = t
    all_input_tensors = dict()
    all_output_tensors = dict()
    all_nodes = dict()
    for n in sub_graph:
        n.data["graph_index"] = index
        all_nodes[n.name] = n.data
        it, ot = get_node_tensors_names(n.data)
        for t in it:
            if t:
                all_input_tensors[t] = tensors_dict[t]
        for t in ot:
            if t:
                all_output_tensors[t] = tensors_dict[t]
    persistent_tensors = {
        k: all_input_tensors[k]
        for k in set(all_input_tensors) - set(all_output_tensors)
    }
    persistent_tensors.update(
        {
            k: all_output_tensors[k]
            for k in set(all_output_tensors) - set(all_input_tensors) if k not in all_output_tensors_names_origin_graph
        }
    )
    for v in persistent_tensors.values():
        v["persistent"] = True
    all_tensors = all_input_tensors
    all_tensors.update(all_output_tensors)
    all_tensors.update(persistent_tensors)

    tensors_names_ordered_list = [t.get("name") for t in tensors_list]
    tensors_names_to_index_in_ordered_list= {tensor_name: tensors_names_ordered_list.index(tensor_name) for tensor_name in tensors_names_ordered_list}
    all_tensors_as_list = list(all_tensors.values())

    # It was noticed that a different order of the tensors list can cause to different
    # execution order in the post_graph. Therefore the order of the tensors in the new graph
    # will be maintained.
    all_tensors_as_list.sort(key=lambda n: tensors_names_to_index_in_ordered_list[n.get("name")],reverse=False)
    graph_name = f"{graph['name']}.{postfix}"
    edited_graph= dict()
    edited_graph["attributes"] = graph.get("attributes")
    edited_graph["group"] = graph.get("group")
    edited_graph["name"] = graph_name
    edited_graph["nodes"] = list(all_nodes.values())
    edited_graph["tensors"] = all_tensors_as_list
    for t in edited_graph["tensors"]:
        t["graph_index"] = index
    if bundle_index is not None:
        edited_graph["origin_bundle_index"] = bundle_index
    LOG.info(f"Done creating graph: {graph_name}")
    return edited_graph


def get_bundle_origin_nodes_names(bundle_index, pre_graph, post_graph) -> "list[str]":
    post_graph_nodes_list = post_graph["nodes"]
    pre_graph_nodes_list = pre_graph["nodes"]
    origin_nodes_ids_lists = [
        n.get("origin_nodes")
        for n in post_graph_nodes_list
        if (n.get("bundle_index") == int(bundle_index))
    ]
    origin_nodes_ids_set = set(n for list in origin_nodes_ids_lists for n in list)
    origin_nodes_names = [
        n.get("name")
        for n in pre_graph_nodes_list
        if n.get("id") in origin_nodes_ids_set
    ]
    return origin_nodes_names


def bundle_generator(post_graph):
    bundles = list(
        set(
            n.get("bundle_index")
            for n in post_graph["nodes"]
            if n.get("bundle_index") is not None
        )
    )
    bundles.sort()
    for bundle in bundles:
        yield bundle


def get_biggest_node_id(pre_graph):
    node_id_biggest = pre_graph["nodes"][0]["id"]
    for n in pre_graph["nodes"]:
        if (n["id"]>node_id_biggest):
            node_id_biggest=n["id"]
    node_id_biggest += 1
    return node_id_biggest


def generate_bundles_graphs(copy_original_post_graphs,post_graphs,pre_graphs, pre_graphs_file_path,args):
    all_graphs_bundles_graphs=[]
    for copy_original_post_graph,post_graph, pre_graph in zip(copy_original_post_graphs,post_graphs,pre_graphs):
        node_id_biggest = get_biggest_node_id(pre_graph) #is used in fix mode
        origin_graph_index = pre_graph.get("origin_graph_index") if (pre_graph.get("origin_graph_index") != None) else pre_graph["nodes"][0].get("graph_index")
        LOG.debug(f'Generate pre graphs to bundles of origin post graph index {origin_graph_index}')
        bundles_graphs = {}
        bundles_graphs["bundles_graphs"] = generate_bundles_pre_graphs_from_graph(copy_original_post_graph,post_graph, pre_graph, pre_graphs_file_path, node_id_biggest, args)
        if (len(bundles_graphs["bundles_graphs"])==0):
            LOG.info(f'Origin post graph index {origin_graph_index} bundles pre graphs were not created')
        bundles_graphs["origin_graph_index"] = origin_graph_index

        all_graphs_bundles_graphs.append(bundles_graphs)
    return all_graphs_bundles_graphs


def get_copy_tensor_by_name(graph, tensor_name):
    copy_of_tensor={}
    for tensor in graph["tensors"]:
        if (tensor["name"] == tensor_name):
            copy_of_tensor = copy.deepcopy(tensor)
            break
    return copy_of_tensor


def change_tensor_to_be_persistent_without_section(tensor, tensor_name):
    tensor["persistent"] = True
    tensor["rmw_section"] = False
    tensor["name"] = tensor_name
    if ("user_mem_offset" in tensor):
        del tensor["user_mem_offset"]
    if ("user_mem_section_index" in tensor):
        del tensor["user_mem_section_index"]
    return tensor


def change_output_memcopy_tensor_to_fit_to_graph(tensor, memcopy_output_tensor_name):
    return change_tensor_to_be_persistent_without_section(tensor, memcopy_output_tensor_name)


def change_input_memcopy_tensor_to_fit_to_graph(tensor, memcopy_input_tensor_name):
    return change_tensor_to_be_persistent_without_section(tensor, memcopy_input_tensor_name)


def find_first_index_of_tensor_in_list_by_its_name(list_of_tensors, tensor_name):
    for k,tensor in enumerate(list_of_tensors):
        if (tensor["name"] == tensor_name):
            return k
    # couldn't found this tensor name.
    return -1


def add_memcpy_node_to_tensor(tensor_name, bundle_pre_graph, graph_index, node_id_biggest):
    memcopy_input_tensor_name = tensor_name + "_before_memcpy"
    memcpy_node= { "blocking_nodes": [],"graph_index": graph_index,"guid": "memcpy","id": node_id_biggest,"input_layouts": [],"input_tensors": [memcopy_input_tensor_name],"name": "memcpy_"+str(node_id_biggest),"output_layouts": [],"output_tensors": [tensor_name],"params": []}
    copy_of_tensor = get_copy_tensor_by_name(bundle_pre_graph, tensor_name)
    if (copy_of_tensor != {}):
        bundle_pre_graph["nodes"].append(memcpy_node)
        bundle_pre_graph["tensors"].insert(find_first_index_of_tensor_in_list_by_its_name(bundle_pre_graph["tensors"],tensor_name),change_input_memcopy_tensor_to_fit_to_graph(copy_of_tensor, memcopy_input_tensor_name))


def get_tensor_producer_name(graph, tensor_name):
    for node in graph["nodes"]:
        if tensor_name in node["output_tensors"]:
            return node["name"]
    return ""


def add_memcpy_node_from_tensor(tensor_name, bundle_pre_graph, graph_index, node_id_biggest):
    # Creates a memcopy node from tensor names tensor_name to new copied tensor
    memcopy_output_tensor_name = tensor_name + "_memcpy"
    memcpy_node= { "blocking_nodes": [],"graph_index": graph_index,"guid": "memcpy","id": node_id_biggest,"input_layouts": [],"input_tensors": [tensor_name],"name": "memcpy_"+str(node_id_biggest),"output_layouts": [],"output_tensors": [memcopy_output_tensor_name],"params": []}
    copy_of_tensor = get_copy_tensor_by_name(bundle_pre_graph, tensor_name)
    if (copy_of_tensor != {}):
        bundle_pre_graph["nodes"].append(memcpy_node)
        # it's essential to add the memcopy output tensor right after its input because it was noticed
        # that the exec order is affected by the tensors order.
        bundle_pre_graph["tensors"].insert(find_first_index_of_tensor_in_list_by_its_name(bundle_pre_graph["tensors"],tensor_name)+1,change_output_memcopy_tensor_to_fit_to_graph(copy_of_tensor, memcopy_output_tensor_name))

def get_inputs_tensors_names_from_graph(graph):
    tensors_names = set(tensor["name"] for tensor in graph["tensors"])
    names_of_tensors_that_output_of_some_node = set(tensor_name for node in graph["nodes"] for tensor_name in node["output_tensors"])
    return tensors_names - names_of_tensors_that_output_of_some_node


def check_bundle_pre_graph_persistents_diffs_to_pre_graph_and_add_memcopy_instead(pre_graph, bundle_pre_graph, graph_index, node_id_biggest, check_only_inputs_of_bundle_pre_graph=False):
    tensors_names_of_pre_graph = set(tensor["name"] for tensor in pre_graph["tensors"])
    persistent_tensors_names_of_pre_graph = set(tensor["name"] for tensor in pre_graph["tensors"] if (tensor["persistent"]))
    bundle_pre_graph_inputs_names = get_inputs_tensors_names_from_graph(bundle_pre_graph)
    if (check_only_inputs_of_bundle_pre_graph):
        persistent_tensors_names_of_bundle_pre_graph_and_not_in_pre_graph = set(tensor["name"] for tensor in bundle_pre_graph["tensors"] if (("DATA_TENSOR" in tensor["type"]) and (tensor["name"] in tensors_names_of_pre_graph) and (tensor["name"] in bundle_pre_graph_inputs_names) and tensor["persistent"] and (tensor["name"] not in persistent_tensors_names_of_pre_graph)))
    else:
        persistent_tensors_names_of_bundle_pre_graph_and_not_in_pre_graph = set(tensor["name"] for tensor in bundle_pre_graph["tensors"] if (("DATA_TENSOR" in tensor["type"]) and (tensor["name"] in tensors_names_of_pre_graph) and (tensor["name"] not in bundle_pre_graph_inputs_names) and tensor["persistent"] and (tensor["name"] not in persistent_tensors_names_of_pre_graph)))

    persistent_tensors_names_of_bundle_pre_graph_and_not_in_pre_graph_list = list(persistent_tensors_names_of_bundle_pre_graph_and_not_in_pre_graph)
    persistent_tensors_names_of_bundle_pre_graph_and_not_in_pre_graph_list.sort()


    for tensor in bundle_pre_graph["tensors"]:
        if (tensor["name"] in persistent_tensors_names_of_bundle_pre_graph_and_not_in_pre_graph_list):
            tensor["persistent"]= False
    for tensor_name in persistent_tensors_names_of_bundle_pre_graph_and_not_in_pre_graph_list:
        if (check_only_inputs_of_bundle_pre_graph):
            add_memcpy_node_to_tensor(tensor_name, bundle_pre_graph, graph_index, node_id_biggest)
        else:
            add_memcpy_node_from_tensor(tensor_name, bundle_pre_graph, graph_index, node_id_biggest)
        node_id_biggest+=1
    return node_id_biggest


def add_memcopy_for_tensors_in_bundle_pre_graph_that_are_being_consumed_outside_of_bundle_in_origin_post_graph(copy_original_post_graph, bundle_pre_graph,graph_index, node_id_biggest):
    bundle_pre_graph_tensors_names = set(tensor["name"] for tensor in bundle_pre_graph["tensors"] if (not tensor["persistent"]))
    for node in copy_original_post_graph["nodes"]:
        if node.get("bundle_index")!=bundle_pre_graph["origin_bundle_index"]:
            consumed_tensors_names_set = set(node["input_tensors"]) & bundle_pre_graph_tensors_names
            if len(consumed_tensors_names_set)>0:
                consumed_tensors_names_list = list(consumed_tensors_names_set)
                consumed_tensors_names_list.sort()
                for consumed_tensor_name in consumed_tensors_names_list:
                    add_memcpy_node_from_tensor(consumed_tensor_name, bundle_pre_graph, graph_index, node_id_biggest)
                    node_id_biggest+=1

                    #
                bundle_pre_graph_tensors_names -= consumed_tensors_names_set
                if (len(bundle_pre_graph_tensors_names)==0):
                    break
    return node_id_biggest


def fix_bundle_pre_graph(pre_graph,bundle_pre_graph, origin_nodes_names, graph_index, node_id_biggest, copy_original_post_graph):
    node_id_biggest = add_memcopy_for_tensors_in_bundle_pre_graph_that_are_being_consumed_outside_of_bundle_in_origin_post_graph(copy_original_post_graph, bundle_pre_graph,graph_index, node_id_biggest)
    node_id_biggest = check_bundle_pre_graph_persistents_diffs_to_pre_graph_and_add_memcopy_instead(pre_graph, bundle_pre_graph, graph_index, node_id_biggest, False)
    node_id_biggest = check_bundle_pre_graph_persistents_diffs_to_pre_graph_and_add_memcopy_instead(pre_graph, bundle_pre_graph, graph_index, node_id_biggest, True)
    return node_id_biggest


def generate_bundles_pre_graphs_from_graph(copy_original_post_graph,post_graph, pre_graph, pre_graphs_file_path, node_id_biggest,args):
    graphs = []
    for bundle in bundle_generator(post_graph):
        origin_nodes_names = get_bundle_origin_nodes_names(
            bundle,
            pre_graph,
            post_graph,
        )
        if len(origin_nodes_names) > 0:
            bundle_pre_graph = generate_graph(get_json_data(args.json_file), origin_nodes_names, f"bundle_{str(bundle)}.edit", len(graphs), bundle, pre_graph["nodes"][0].get("graph_index"), args)
            LOG.debug(f'Check if {bundle_pre_graph["name"]} need fixes')
            node_id_biggest_after_fix = fix_bundle_pre_graph(pre_graph,bundle_pre_graph, origin_nodes_names, len(graphs), node_id_biggest, copy_original_post_graph)
            if (node_id_biggest_after_fix > node_id_biggest):
                # memcopy nodes were added to the graph
                bundle_pre_graph["config"] = {"ENABLE_REMOVE_REDUNDANT_MEMCPY": "False"}

            graphs.append(bundle_pre_graph)
    return graphs

def save_bundles_graphs_per_each_original_graph(all_origin_graphs_bundles_graphs, pre_graphs_file_path, args):
    for bundles_graphs in all_origin_graphs_bundles_graphs:
        if (len(bundles_graphs["bundles_graphs"])>0):
            bundles_graphs["bundles_graphs_path"] = save_graphs(bundles_graphs["bundles_graphs"], pre_graphs_file_path, f'.bundles_graphs_of_origin_graph_index_{str(bundles_graphs["origin_graph_index"])}',args)
        else:
            bundles_graphs["bundles_graphs_path"] = ""
    return all_origin_graphs_bundles_graphs


def save_graphs(graphs, pre_graphs_path, post_fix_name, args):
    json_data = get_json_data(pre_graphs_path)
    json_data["graphs"] = graphs
    if "name" not in json_data:
        json_data["name"] = pre_graphs_path.split('/')[-1].split('.json')[0]
    if (post_fix_name is None):
        json_data["name"] += ".edit"
    else:
        json_data["name"] += post_fix_name

    output = (
        args.output if args.output else f'{os.path.basename(json_data["name"])}.json'
    )
    with open(output, "w") as f:
        json.dump(json_data, f, indent=4, sort_keys=False)
    LOG.info(f"processed file: {os.path.abspath(output)}")
    return os.path.abspath(output)


def get_all_nodes_tensors_names(graph):
    tensors_names = set()
    for node in graph["nodes"]:
        it, ot = get_node_tensors_names(node)
        tensors_names |= it | ot
    return tensors_names


def filter_from_graph_nodes_and_tensors_that_not_in_bundle_indexes(graph, bundle_indexes, quiet_mode=True):
    bundle_indexes = [int(bi) for bi in bundle_indexes]
    if (not quiet_mode):
        LOG.debug(f'Filter graph to contain only nodes and tensors of these bundles: {bundle_indexes}')
    graph["nodes"] = list(
        filter(lambda n: n.get("bundle_index") in bundle_indexes, graph["nodes"])
    )
    tensors_names = get_all_nodes_tensors_names(graph) #be careful they are not equal to graph["tensors"]
    graph["tensors"] = list(
        filter(lambda n: n.get("name") in tensors_names, graph["tensors"])
    )
    return graph


def generate_bundle_graph_from_original_post_graph(origin_post_graph_path,origin_post_graph_index,  bundle):
    json_data = get_json_data(origin_post_graph_path)
    if (len(json_data["graphs"])==1):
        graph = json_data["graphs"][0]
    else:
        graph = json_data["graphs"][origin_post_graph_index]
    bundle_nodes = [n for n in graph["nodes"] if (n.get("bundle_index") == int(bundle))]
    graph["nodes"] = bundle_nodes
    tensors_names = get_all_nodes_tensors_names(graph)
    graph["tensors"] = list(
        filter(lambda n: n.get("name") in tensors_names, graph["tensors"])
    )
    return graph


def create_bundles_graphs_from_original_post_graph(origin_post_graph, origin_post_graph_path, origin_post_graph_index, args):
    bundle_to_graph = {}
    for bundle in bundle_generator(origin_post_graph):
        bundle_to_graph[bundle] = generate_bundle_graph_from_original_post_graph(origin_post_graph_path, origin_post_graph_index, bundle)
    return bundle_to_graph


def split_graph_to_its_bundles(graph):
    graphs = []
    for bundle in bundle_generator(graph):
        graphs.append(filter_from_graph_nodes_and_tensors_that_not_in_bundle_indexes(copy.deepcopy(graph), [int(bundle)], quiet_mode=True))
        graphs[len(graphs)-1]["bundle_index"] = int(bundle)
    return graphs


def compare_dictionary_fields(d1, d2, dictionary_type, field_name):
    # Assumes d1,d2 contaings name and field_name as keys
    log_failure_prefix = f"Mismatch between {dictionary_type}s: {d1.get('name')} and {d2.get('name')}"
    if (d1.get(field_name) != d2.get(field_name)):
        return False, log_failure_prefix + f" because of different {field_name}: {d1.get(field_name)} and {d2.get(field_name)} respectively"
    return True,""


def compare_tensors(t1, t2):
    status, log_failures = compare_dictionary_fields(t1, t2, "tensor","dtype")
    if (status == False):
        return status, log_failures
    status, log_failures = compare_dictionary_fields(t1, t2, "tensor","max_shape")
    if (status == False):
        return status, log_failures
    status, log_failures = compare_dictionary_fields(t1, t2, "tensor","min_shape")
    return status, log_failures


def compare_tensors_lists(
    graph_one_tensors_name_to_data,
    tensors_names1,
    graph_two_tensors_name_to_data,
    tensors_names2
):
    for t1, t2 in zip(tensors_names1, tensors_names2):
        status, log_failures = compare_tensors(graph_one_tensors_name_to_data[t1], graph_two_tensors_name_to_data[t2])
        if (status == False):
            return status, log_failures
    return True,""


def compare_nodes(
    graph_one_tensors_name_to_data, node1, graph_two_tensors_name_to_data, node2
):
    # node comparison method is not perfect. can be corner cases especially with fused nodes cases.

    #fused nodes guids are not to be compared because fused guids contain indexes.
    if (not node1["guid"].startswith("fused") or not node2["guid"].startswith("fused")):
        status, log_failures = compare_dictionary_fields(node1, node2, "node","guid")
        if (status == False):
            return status, log_failures

    if (len(node1["input_tensors"]) != len(node2["input_tensors"])):
        return False, f"Mismatch between nodes: {node1.get('name')} and {node2.get('name')} different input_tensors length"
    if (len(node1["output_tensors"]) != len(node2["output_tensors"])):
        return False, f"Mismatch between nodes: {node1.get('name')} and {node2.get('name')} different output_tensors length"

    status, log_failures = compare_tensors_lists(graph_one_tensors_name_to_data, node1["input_tensors"], graph_two_tensors_name_to_data, node2["input_tensors"])
    if (status == False):
        return status, log_failures
    status, log_failures = compare_tensors_lists(graph_one_tensors_name_to_data, node1["output_tensors"], graph_two_tensors_name_to_data, node2["output_tensors"])
    return status, log_failures


def compare_bundles(bundle_graph_one, bundle_graph_two):
    # bundle_graph_one, bundle_graph_two - each one of them should contain only the nodes of the bundle and at least all the tensors of these nodes.
    if len(bundle_graph_one["nodes"]) != len(bundle_graph_two["nodes"]):
        return False, f"Mismatch in the number of nodes between the bundles"

    graph_one_node_list = bundle_graph_one["nodes"]
    graph_two_node_list = bundle_graph_two["nodes"]
    graph_one_node_list.sort(key=lambda n: n.get("exec_order_idx"))
    graph_two_node_list.sort(key=lambda n: n.get("exec_order_idx"))
    graph_one_tensors_name_to_data = {
        t.get("name"): t for t in bundle_graph_one["tensors"]
    }
    graph_two_tensors_name_to_data = {
        t.get("name"): t for t in bundle_graph_two["tensors"]
    }
    for node1, node2 in zip(graph_one_node_list, graph_two_node_list):
        status, log_failures = compare_nodes(graph_one_tensors_name_to_data, node1, graph_two_tensors_name_to_data, node2)
        if (status == False):
            return status, log_failures
    return True, ""


def get_bundles_post_graphs_in_map_way(bundles_pre_graphs_path, args):
    LOG.debug(f'Compile bundles pre graphs')
    bundles_post_graphs_scoped_file = GraphsLoader.record_and_save_graphs("post", bundles_pre_graphs_path, args.chip_type, None, args.venv, LOG)
    LOG.debug(f'Compile bundles pre graphs finished successfully')
    bundles_post_graphs = get_json_data(bundles_post_graphs_scoped_file.file_path).get("graphs")
    bundles_pre_graphs = get_json_data(bundles_pre_graphs_path).get("graphs")
    if (len(bundles_pre_graphs) != len(bundles_post_graphs)):
        LOG.error(f'Something went wrong in bundles pre graphs compilation')

    bundle_index_to_post_graph = {}
    for bundle_pre_graph,bundle_post_graph in zip(bundles_pre_graphs,bundles_post_graphs):
        bundle_index_to_post_graph[bundle_pre_graph["origin_bundle_index"]] = bundle_post_graph
    return bundle_index_to_post_graph


def print_to_log_mismatches(post_graph_of_claimed_bundle_pre_graph_splitted_to_bundles):
    for bundle in post_graph_of_claimed_bundle_pre_graph_splitted_to_bundles:
        # TBD: Add bundle_index to the log message.
        LOG.info(f"{bundle['log_failures']}")


def validate_bundles_pre_graphs(
    bundles_pre_graphs_path, origin_graph_index, origin_post_graph,origin_post_graph_path, args
):
    bundle_index_to_post_graph_of_claimed_bundle_pre_graph = get_bundles_post_graphs_in_map_way(bundles_pre_graphs_path, args)

    LOG.debug(f'Split origin post graph to bundles')
    bundle_index_to_origin_post_graph_bundle = create_bundles_graphs_from_original_post_graph(
        origin_post_graph,origin_post_graph_path,origin_graph_index, args
    )
    LOG.debug(f'Split origin post graph to bundles finished successfully')

    if len(bundle_index_to_origin_post_graph_bundle) != len(bundle_index_to_post_graph_of_claimed_bundle_pre_graph):
        LOG.info("There is a mismatch between the number of bundles graphs")
        return -1

    validation_process_status=0
    LOG.debug(f'Compare each graph in the splitted origin post graph with its counterpart in the compiled bundles pre graphs')
    for bundle_index in bundle_index_to_post_graph_of_claimed_bundle_pre_graph.keys():
        origin_post_graph_bundle = bundle_index_to_origin_post_graph_bundle[bundle_index]
        post_graph_of_claimed_bundle_pre_graph = bundle_index_to_post_graph_of_claimed_bundle_pre_graph[bundle_index]

        # It can be that the bundle post graph (the post of the claimed pre graph bundle which was created by graph_editor) contains non bundles or more than one bundle.
        # If one of its bundles is equal to the origin bundle then the validation succeed.
        post_graph_of_claimed_bundle_pre_graph_splitted_to_bundles = split_graph_to_its_bundles(post_graph_of_claimed_bundle_pre_graph)
        flag=False
        for bundle in post_graph_of_claimed_bundle_pre_graph_splitted_to_bundles:
            status, bundle["log_failures"] = compare_bundles(bundle, origin_post_graph_bundle)
            flag = flag or status
            if flag:
                LOG.info(f'Origin graph index: {str(origin_graph_index)}, bundle {bundle_index} validation succeeded')
                break
        if not flag:
            LOG.info(f'Origin graph index: {str(origin_graph_index)}, bundle {bundle_index} validation failed due to the following mismatches:')
            print_to_log_mismatches(post_graph_of_claimed_bundle_pre_graph_splitted_to_bundles)
            validation_process_status = -1
    return validation_process_status

def find_all_bundles_indexes(post_graph):
    return set( n.get("bundle_index") for n in post_graph["nodes"] if n.get("bundle_index") is not None)


def check_if_bundle_index_is_not_unique(unique_bundles_indexes, bi, bundles_indexes_to_nodes_list, tensors_list):
    for unique_bi in unique_bundles_indexes:
        bi_graph = {"nodes": bundles_indexes_to_nodes_list[bi], "tensors": tensors_list}
        unique_bi_graph ={"nodes": bundles_indexes_to_nodes_list[unique_bi], "tensors": tensors_list}
        status, _ = compare_bundles(bi_graph, unique_bi_graph)
        if (status == True):
            LOG.info(f'bundle index {bi} was found equal to bundle index {unique_bi} which already in the unique list')
            return True
    return False


def edit_each_post_graph_to_contain_only_unique_bundles(post_graphs, args):
    # Work process:
    # For each post_graph in post_graphs the function will iterate over its bundles and filter out non unique bundles.
    # Uniqueness of bundle is determined with this API : compare_bundles(bundle_one, bundle_two).
    for post_graph in post_graphs:
        if (not check_if_there_are_bundles_in_the_graph(post_graph)):
            continue
        origin_post_graph_index = post_graph.get("origin_graph_index") if (post_graph.get("origin_graph_index") != None) else post_graph["nodes"][0].get("graph_index")
        LOG.info(f'Create unique bundles list for origin post graph index {str(origin_post_graph_index)}')
        unique_bundles_indexes = []
        tensors_list = post_graph["tensors"]
        bundles_indexes = find_all_bundles_indexes(post_graph)
        bundles_indexes_to_nodes_list = {}
        for bi in bundles_indexes:
            bundles_indexes_to_nodes_list[bi] = [n for n in post_graph["nodes"] if n.get("bundle_index")==bi]
            if (len(unique_bundles_indexes) == 0):
                unique_bundles_indexes.append(bi)
            else:
                #bi_graph = {"nodes": bundles_indexes_to_nodes_list[bi], "tensors": tensors_list}
                if (not check_if_bundle_index_is_not_unique(unique_bundles_indexes, bi, bundles_indexes_to_nodes_list, tensors_list)):
                    unique_bundles_indexes.append(bi)
        LOG.info(f'Unique bundles of post graph index {args.origin_graph_index if args.origin_graph_index else str(post_graph["nodes"][0]["graph_index"])} are: {unique_bundles_indexes}')
        post_graph = filter_from_graph_nodes_and_tensors_that_not_in_bundle_indexes(post_graph, unique_bundles_indexes, False)
    return post_graphs


def check_if_there_are_bundles_in_the_graph(graph):
    for n in graph["nodes"]:
        if (n.get("bundle_index")!=None):
            return True
    return False


def bundle_reproduction_tool_logic(args):
    # How the bundle-tool reproduce a bundle pre graph:
    # 1) Determine which bundles to reproduce (according to data given by the user) and the relevant graphs.
    # 2) Create a pre graph for each bundle from the first step, with the help of the origin_nodes tracking mechanism (implemented in gc).
    # 3) Fix bundle-pre-graphs according to specific fixes that revealed experimentally.
    # 4) If asked, it validates the created bundle pre graphs
    if (args.no_record == False):
        LOG.info(f"Start of pre and post graphs recordings")
    #TO-DO: pre_graphs and post_graphs should be recorded together if possible (saving time).
    graphs = GraphsLoader(args, LOG)
    post_graphs = graphs.post_graphs
    pre_graphs = graphs.pre_graphs
    if (args.no_record == False):
        LOG.info(f"End of pre and post graphs recordings")

    if (len(post_graphs)!=len(pre_graphs)):
        raise RuntimeError("The number of post graphs and pre graphs is not match")

    LOG.info(f"Bundle tool - json: {args.json_file}, bundles: {args.bundles if args.bundles else 'all bundles'}{(', graph index: '+str(args.origin_graph_index)) if args.origin_graph_index is not None else ''}")
    # copy of the original post graph will be used for bundles pre graphs fixes.
    copy_original_post_graphs = copy.deepcopy(post_graphs)

    if args.bundles:
        # In this case len(post_graphs) == 1.
        # Filter post_graph to contain only nodes and tensors of those bundles
        post_graphs[0] = filter_from_graph_nodes_and_tensors_that_not_in_bundle_indexes(post_graphs[0], args.bundles, False)

    if args.no_unique == False:
        LOG.info(f'Chosen mode: Unique bundles only')
        # Unique mode - means that bundle_tool should reproduce the pre graphs of the unique bundles only.
        # How it is done: the post_graph will be filtered to include only unique bundles (means only nodes & tensors of those bundles).
        # Then the script continues to operate as usual.
        if (args.bundles and len(args.bundles)>1) or args.all_bundles:
            post_graphs = edit_each_post_graph_to_contain_only_unique_bundles(post_graphs, args)
    else:
        LOG.debug(f'Chosen mode: Non unique bundles')

    LOG.debug(f'Generate bundles pre graphs')
    all_origin_graphs_bundles_graphs = generate_bundles_graphs(copy_original_post_graphs,post_graphs,pre_graphs,args.json_file, args)
    # all_origin_graphs_bundles_graphs is of this form: {{"bundles_graphs": {graph1, ..., graphk}, "origin_graph_index": int} , {..}, ...}
    save_bundles_graphs_per_each_original_graph(all_origin_graphs_bundles_graphs, args.json_file, args)

    validation_status=0
    if args.validate:
        # check if bundles pre graphs were created
        empty_graphs = [len(bundles_graphs["bundles_graphs"])==0 for bundles_graphs in all_origin_graphs_bundles_graphs]
        if not all(empty_graphs):
            LOG.info(f'Start bundles validation:')
            for post_graph, bundles_graphs in zip(post_graphs,all_origin_graphs_bundles_graphs):
                if (len(bundles_graphs["bundles_graphs"])==0):
                    continue
                LOG.info(f'Validate bundles of origin graph index: {str(bundles_graphs["origin_graph_index"])}')
                validation_status |= validate_bundles_pre_graphs(
                    bundles_graphs["bundles_graphs_path"], bundles_graphs["origin_graph_index"], post_graph,args.post_graph, args
                )
    return validation_status, [origin_graph_bundles_graphs["bundles_graphs_path"] for origin_graph_bundles_graphs in all_origin_graphs_bundles_graphs]


def main(args):
    if args.bundles or args.all_bundles:
        bundle_reproduction_tool_logic(args)
    elif args.single_node or args.guids:
        graphs = generate_graphs(args)
        save_graphs(graphs, args.json_file, None, args)
    else:
        json_data = get_json_data(args.json_file)
        graph_index = 0 if args.graph_index is None else args.graph_index
        graphs = [generate_graph(json_data, args.nodes, "edit", 0, None, graph_index, args)]
        save_graphs(graphs, args.json_file, None,args)


def check_forbidden_combinations(args, parser):
    if args.all_bundles or args.bundles:
        if not (args.chip_type or args.no_record):
            parser.error("Argument --chip-type is required to generate bundles")
        if args.validate and not args.chip_type:
            parser.error("Argument --chip-type is required in bundles validation mode")
        if (args.bundles and args.graph_index == None):
            parser.error("Argument --graph_index is required when specific bundles index are mentioned")
        if args.no_record and (args.json_file is None or args.post_graph is None):
            parser.error("Argument --no-record requires pre graphs and post graphs")
    else:
        if args.graph_index is None and args.single_node is None:
            parser.error("Argument --graph_index is required when editing the graph")


def get_venv_if_any():
    supported_venvs = ["QNPU_PATH", "VIRTUAL_ENV", "CONDA_PREFIX"]
    for venv in supported_venvs:
        venv_folder = os.environ.get(venv, None)
        if venv_folder:
            return f'{venv_folder}/bin/activate'
    return None


def parse_args(user_args=None):
    parser = argparse.ArgumentParser(add_help=True, description="When compiling graphs release binaries are being used.")
    models = list(run.get_supported_models(run.MODELS_TESTS_PATH, None, True))
    graphs_input = parser.add_mutually_exclusive_group(required=True)
    graphs_input.add_argument("-j", "--json-file", help="Pre graph json file path")
    graphs_input.add_argument("-m", "--model", choices=models, help="A model to work on " + ", ".join(models), metavar="",)
    parser.add_argument("-o", "--output", help="Edited pre graph json file path")
    parser.add_argument("-g", "--graph-index", type=int, help="Graph index", default=None)
    parser.add_argument("-r", "--radius", type=int, help="Search radius", default=0)
    parser.add_argument("--single-node", action="store_true", help="Generates single node graphs for all --nodes, if --nodes not set use all nodes.")
    nodes_filter = parser.add_mutually_exclusive_group()
    nodes_filter.add_argument("-n", "--nodes", nargs="+", help="Nodes to include")
    nodes_filter.add_argument("--guids", nargs="+", help="Filter nodes by guid")
    bundle_group = parser.add_mutually_exclusive_group()
    bundle_group.add_argument(
        "--bundles",
        nargs="+",
        help="Generates chosen bundles pre graphs, works on one graph only.",
    )
    bundle_group.add_argument(
        "--all-bundles",
        action="store_true",
        help="Generates all bundles pre graphs.",
    )
    parser.add_argument(
        "--post-graph",
        help="Post graph json file path, used when --bundles --all-bundles is set."
        "Relevant only in no_record mode",
    )
    parser.add_argument(
        "-v",
        "--validate",
        action="store_true",
        help="Validates the produced bundles graphs.",
    )
    parser.add_argument(
        "--no-unique",
        action="store_true",
        help="Produce all bundles pre graphs (if not set only unique bundles pre graphs are created).",
    )
    dir_group = parser.add_mutually_exclusive_group()
    dir_group.add_argument("--fwd", action="store_true", help="Search only forward")
    dir_group.add_argument("--bwd", action="store_true", help="Search only backward")
    parser.add_argument(
        "-c",
        "--chip-type",
        choices=list(syn.CHIP_TYPES.keys()),
        help="Select chip type",
    )
    parser.add_argument(
        "--venv",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-record",
        action="store_true",
        help="Bundle tool doesn't record the pre and post graphs.",
        default=False
    )
    args = parser.parse_args(args=user_args)
    check_forbidden_combinations(args, parser)
    if not args.venv:
        args.venv = get_venv_if_any()
    if args.model:
        models = run.get_all_models(run.MODELS_TESTS_PATH, True)
        args.json_file = models.get(args.model)
    return args


if __name__ == "__main__":
    args = parse_args()
    exit(main(args))


###### Api definition - used by other tools ######

def get_graph_bundles_json(json_file: str, bundles: List[int], graph_index: int, chip_type: str) -> str:
    """
    json_file: json path\\
    bundles: list of bundles indexes\\
    graph_index: the index of the graph in the json file\\
    chip_type: device\\
    return: json file path of bundles pre graphs
    """
    # arrange bundle reproduction tool args
    argv = ["--json-file", json_file, "--graph-index", str(graph_index), "--chip-type", chip_type, "-v", "--bundles"] + [str(b) for b in bundles]

    # call bundle reproduction tool
    args = parse_args(argv)
    validation_status, bundle_graphs_paths = bundle_reproduction_tool_logic(args)

    # check validation and path
    if (len(bundle_graphs_paths) != 1):
        raise RuntimeError("More than one json has been created even though there is only one origin graph")
    if (validation_status != 0):
        LOG.error(f'Bundles validation failed')
    if (bundle_graphs_paths[0] == ""):
        exit("No bundles pre graphs created")

    return bundle_graphs_paths[0], validation_status
