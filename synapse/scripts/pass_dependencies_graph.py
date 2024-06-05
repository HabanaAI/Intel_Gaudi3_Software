import argparse
import graphviz
import syn_infra as syn
import re
import logging

LOG = syn.config_logger("pass_dependencies_graph_creator", logging.INFO)

graphviz.__version__, graphviz.version()

DUMMY_IN_SUFFIX = "dummyIn"
DUMMY_OUT_SUFFIX = "dummyOut"
SUBGRAPH_NAME_PREFIX = "cluster"
EDGE_MIN_LEN = "2"

frame_colors = ["orange", "blue", "green", "purple", "red"]


PASS = "Pass"
GROUP = "Group"


class PassDependenciesLogParser:
    """
    This object parses the log prints of pass dependencies graph.

    Example of such log prints:
    [PASS_MANAGER          ][trace][tid:13EDF3] Print passes/groups and their dependencies
    [PASS_MANAGER          ][trace][tid:13EDF3] Pass: pass12 belongs to group: group1 depends on 0 registered passes/groups
    [PASS_MANAGER          ][trace][tid:13EDF3] Pass: pass21 belongs to group: group2 depends on 0 registered passes/groups
    [PASS_MANAGER          ][trace][tid:13EDF3] Pass: pass22 belongs to group: group2 depends on 0 registered passes/groups
    [PASS_MANAGER          ][trace][tid:13EDF3] Group: group3 belongs to group: group2 depends on 0 registered passes/groups
    [PASS_MANAGER          ][trace][tid:13EDF3] Pass: pass31 belongs to group: group3 depends on 0 registered passes/groups
    [PASS_MANAGER          ][trace][tid:13EDF3] Group: group1 depends on 2 registered passes/groups
    [PASS_MANAGER          ][trace][tid:13EDF3] group2
    [PASS_MANAGER          ][trace][tid:13EDF3] pass22
    [PASS_MANAGER          ][trace][tid:13EDF3] Pass: pass11 belongs to group: group1 depends on 1 registered passes/groups
    [PASS_MANAGER          ][trace][tid:13EDF3] pass31
    [PASS_MANAGER          ][trace][tid:13EDF3] Group: group2 depends on 0 registered passes/groups
    [PASS_MANAGER          ][trace][tid:13EDF3] Print passes/groups and their dependencies finished

    Assumptions:
    - Each pass mentioned in the prints belongs to exactly one group."""

    def __init__(self, log_file):
        self.log_file = log_file
        self.counter_dependencies = 0
        self.current_pass_or_group = None
        self.parsing_started = False
        # start parsing:
        self.parse_log_to_graph_object()

    def handle_start_parsing(self, pattern):
        """parse line that contains the headline of all passes dependencies"""
        self.pass_or_group_to_dependencies: dict[str, set[str]] = {}
        self.group_to_its_elements: dict[str, set[str]] = {}
        self.passes = set()
        self.groups = set()
        self.parsing_started = True

    def handle_pass_or_group_headline_with_group_parent_pattern(self, pattern):
        """ parse line that contains element's info including its group """
        pass_or_group, pass_or_group_name, parent_group_name ,num_of_dependencies = pattern.groups()
        if parent_group_name not in self.group_to_its_elements:
            self.group_to_its_elements[parent_group_name] = set()
        self.group_to_its_elements[parent_group_name].add(pass_or_group_name)
        self.parse_info_of_pass_or_group(pass_or_group, pass_or_group_name, num_of_dependencies)

    def handle_pass_or_group_headline_without_group_parent_pattern(self, pattern):
        """parse line that contains element's info without its group"""
        pass_or_group, pass_or_group_name, num_of_dependencies = pattern.groups()
        self.parse_info_of_pass_or_group(pass_or_group, pass_or_group_name, num_of_dependencies)


    def parse_info_of_pass_or_group(self, pass_or_group, pass_or_group_name, num_of_dependencies):
        if (PASS in pass_or_group):
            self.passes.add(pass_or_group_name)
        else:
            self.groups.add(pass_or_group_name)
        self.counter_dependencies = int(num_of_dependencies)
        self.current_pass_or_group = pass_or_group_name

    def handle_add_dependency(self, pattern):
        """parse line that contains dependency of self.current_pass_or_group"""
        dependency, = pattern.groups()
        if self.current_pass_or_group not in self.pass_or_group_to_dependencies:
            self.pass_or_group_to_dependencies[self.current_pass_or_group] = set()
        self.pass_or_group_to_dependencies[self.current_pass_or_group].add(dependency)
        self.counter_dependencies -= 1

    def parse_log_to_graph_object(self):
        LOG.info(f"start parsing the log")
        pass_manager_scheme_pattern = re.compile(r'.*\[PASS_MANAGER.*\s*\] (.*)')
        start_parsing_pattern = re.compile(r'.* passes/groups .* dependencies')
        pass_or_group_headline_with_group_parent_pattern = re.compile(r'.* (Pass:|Group:) (.*) belongs to group: (.*) depends on (\d+)')
        pass_or_group_headline_without_group_parent_pattern = re.compile(r'.* (Pass:|Group:) (.*) depends on (\d+)')
        end_parsing_pattern = re.compile(r'.* passes/groups .* dependencies finished')
        with open(self.log_file) as infile:
            for line_count, line in enumerate(infile):
                try:
                    if pass_manager_scheme_pattern.search(line):
                        if end_parsing_pattern.search(line):
                            res = end_parsing_pattern.search(line)
                            break
                        elif start_parsing_pattern.search(line):
                            res = start_parsing_pattern.search(line)
                            self.handle_start_parsing(res)
                            continue
                        if self.parsing_started:
                            if pass_or_group_headline_with_group_parent_pattern.search(line):
                                res = pass_or_group_headline_with_group_parent_pattern.search(line)
                                self.handle_pass_or_group_headline_with_group_parent_pattern(res)
                            elif pass_or_group_headline_without_group_parent_pattern.search(line):
                                res = pass_or_group_headline_without_group_parent_pattern.search(line)
                                self.handle_pass_or_group_headline_without_group_parent_pattern(res)
                            elif self.counter_dependencies > 0:
                                res = pass_manager_scheme_pattern.search(line)
                                self.handle_add_dependency(res)
                except Exception as e:
                    LOG.error(
                        "problem occrued in parsing of log file, in line {}".format(
                            line_count
                        )
                    )
                    LOG.error(line)
                    raise e

        LOG.info(f"parsing finished")


class Graph:
    """a wrapper class for the graph of graphviz"""

    def __init__(self, name):
        self.name = name
        self.initialize_concrete_graph_name()
        self.concrete_object = graphviz.Digraph(name=self.concrete_graph_name)
        self.concrete_object.attr(compound="true")

    def initialize_concrete_graph_name(self):
        self.concrete_graph_name = self.name

    def add_node(self, node_name):
        self.concrete_object.node(node_name)

    def add_sub_graph(self, sub_graph):
        self.concrete_object.subgraph(sub_graph.concrete_object)

    def add_color(self, color):
        self.concrete_object.attr(color=color)

    def add_edge(self, tail, head, ltail=None, lhead=None, style=""):
        self.concrete_object.edge(tail, head, ltail=ltail, lhead=lhead, minlen=EDGE_MIN_LEN, style=style)


class SubGraph(Graph):
    def __init__(self, name):
        super().__init__(name)
        self.concrete_object.attr(label=self.name)
        self.add_dummy_nodes()
        self.add_dummy_edge()

    def add_dummy_edge(self):
        self.add_edge(self.dummy_in_name, self.dummy_out_name, style="invis")

    def add_dummy_node(self, node_name):
        self.concrete_object.node(node_name, shape="point", color="white")

    def add_dummy_nodes(self):
        self.dummy_in_name = self.name + DUMMY_IN_SUFFIX
        self.dummy_out_name = self.name + DUMMY_OUT_SUFFIX
        self.add_dummy_node(self.dummy_in_name)
        self.add_dummy_node(self.dummy_out_name)

    def initialize_concrete_graph_name(self):
        # subgraph name must contain 'cluster' as prefix, otherwise it won't be shown in the dumped graph.
        self.concrete_graph_name = SUBGRAPH_NAME_PREFIX + self.name

    def add_edges_to_dummies(self, node_name):
        self.add_edge(self.dummy_in_name, node_name, style="invis")
        self.add_edge(node_name, self.dummy_out_name, style="invis")

    def add_node(self, node_name):
        super().add_node(node_name)
        self.add_edges_to_dummies(node_name)

    def add_sub_graph(self, sub_graph):
        self.concrete_object.subgraph(sub_graph.concrete_object)
        # add edges to dummies :
        # logic is different from add_edges_to_dummies() because: a. there is no common nodes between the edges,
        # b. there is logical tail/head in these edges.
        self.add_edge(self.dummy_in_name, sub_graph.dummy_in_name, ltail= None, lhead=sub_graph.concrete_graph_name, style="invis")
        self.add_edge(sub_graph.dummy_out_name, self.dummy_out_name, ltail=sub_graph.concrete_graph_name, lhead=None, style="invis")


class PassDependenciesGraph:
    def __init__(self, parsed_data : PassDependenciesLogParser, no_tred=False, name="clusterG"):
        self.sub_graphs_map: dict[str, Graph] = {}
        self.sub_graphs_from_second_layer = set()
        self.nodes = set()
        self.name: str = name
        self.no_tred = no_tred
        self.parsed_data = parsed_data
        self.dg = Graph(name=self.name)
        self.sub_graphs_names_to_graphs: dict[str, Graph] = {}

    def initialize_sub_graphs_from_graph_lines(self):
        for group_name in self.parsed_data.groups:
            self.sub_graphs_names_to_graphs[group_name] = SubGraph(group_name)

    def create_nodes(self):
        for group_name in self.parsed_data.group_to_its_elements.keys():
            sub_graph = self.sub_graphs_names_to_graphs[group_name]
            elements = self.parsed_data.group_to_its_elements[group_name]
            # elements = passes or groups
            for element_name in elements:
                if element_name in self.parsed_data.passes:
                    sub_graph.add_node(element_name)

    def get_head_node(self, node_name):
        lhead = None
        if node_name in self.parsed_data.groups:
            sub_graph = self.sub_graphs_names_to_graphs[node_name]
            lhead = sub_graph.concrete_graph_name
            node_name = sub_graph.dummy_in_name
        return node_name, lhead

    def get_tail_node(self, node_name):
        ltail = None
        if node_name in self.parsed_data.groups:
            sub_graph = self.sub_graphs_names_to_graphs[node_name]
            ltail = sub_graph.concrete_graph_name
            node_name = sub_graph.dummy_out_name
        return node_name, ltail

    def create_edges(self):
        for element_name in self.parsed_data.pass_or_group_to_dependencies.keys():
            head_name, lhead = self.get_head_node(element_name)
            element_dependencies = self.parsed_data.pass_or_group_to_dependencies[element_name]
            for element_dependency in element_dependencies:
                # element_name depends on element_dependency,
                # means that element_dependency needs to be executed before element_name.
                # This dependency will be shown in the graph this way: element_dependency ---> element_name
                tail_name, ltail = self.get_tail_node(element_dependency)
                self.dg.add_edge(tail_name, head_name, ltail, lhead)

    def add_sub_graphs_of_g(self, g: Graph, depth=0):
        # DFS
        if g.name not in self.parsed_data.group_to_its_elements.keys():
            return
        for element_name in self.parsed_data.group_to_its_elements[g.name]:
            if element_name in self.parsed_data.groups:
                sub_graph = self.sub_graphs_names_to_graphs[element_name]
                self.add_sub_graphs_of_g(sub_graph, depth + 1)
                sub_graph.add_color(frame_colors[depth % len(frame_colors)])
                g.add_sub_graph(sub_graph)

    def get_dg_sub_graphs(self):
        dg_sub_graphs = set()
        all_groups_elements = set()
        for group_elements in self.parsed_data.group_to_its_elements.values():
            all_groups_elements |= group_elements
        for group_name in self.parsed_data.groups:
            if group_name not in all_groups_elements:
                dg_sub_graphs.add(group_name)
        return dg_sub_graphs

    def add_sub_graphs(self):
        dg_sub_graphs_names = self.get_dg_sub_graphs()
        for dg_sub_graph_name in dg_sub_graphs_names:
            dg_sub_graph: SubGraph = self.sub_graphs_names_to_graphs[dg_sub_graph_name]
            self.add_sub_graphs_of_g(dg_sub_graph, 1)
            self.dg.add_sub_graph(dg_sub_graph)

    def do_transitive_reduction(self):
        cmd = [
            "tred",
            self.name + ".gv",
            "|",
            "dot",
            "-T",
            "pdf",
            ">",
            self.name + ".pdf",
        ]
        syn.run_external_in_bash(cmd)

    def create_graph(self) -> graphviz.Digraph:
        """implements the creation logic of pass dependencies graph"""

        # creation logic:
        # 1. initialize all subgraphs objects.
        # 2. add subgraphs' nodes (create_nodes()).
        # 3. add subgraphs to their parents graphs (done with dfs scan)
        # 4. add edges (works because nodes names are uniques).

        self.initialize_sub_graphs_from_graph_lines()
        LOG.info(f"create nodes")
        # nodes creation stage split to 2 different stages:
        # 1. the creation of nodes that are passes, that occurs in create_node().
        # 2. the creation of nodes that are groups (subgraphs) that occurs in add_sub_graphs().
        self.create_nodes()
        self.add_sub_graphs()

        LOG.info(f"create edges")
        self.create_edges()

        if self.no_tred:
            LOG.info(f"dumping graph")
            self.dg.concrete_object.render()
        else:
            LOG.info(f"executing transitive reduction and dumping the graph")
            self.dg.concrete_object.save()
            self.do_transitive_reduction()


def create_pass_dependencies_graph(parsed_data: PassDependenciesLogParser, args):
    LOG.info(f"create the pass dependencies graph")
    gw = PassDependenciesGraph(parsed_data, args.no_tred)
    gw.create_graph()


def create_and_dump_pass_dependencies_graph(args):
    pass_dependencies_parsed_data = PassDependenciesLogParser(args.log_file)
    create_pass_dependencies_graph(pass_dependencies_parsed_data, args)


def main(args):
    create_and_dump_pass_dependencies_graph(args)


def parse_args():
    parser = argparse.ArgumentParser(
        add_help=True,
        description="This tool creates a pdf of pass dependencies graph. It assumes graphviz and re packages are installed.\nIf you encounter any error, please open a ticket and assign to tmagdaci.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    files_test_group = parser.add_mutually_exclusive_group()
    files_test_group.add_argument("-f", "--log-file", help="A graph_compiler.log file")
    parser.add_argument(
        "--no-tred",
        action="store_true",
        help="Dump the pass dependencies graph without transitive reduction.",
        default=False,
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    exit(main(args))
