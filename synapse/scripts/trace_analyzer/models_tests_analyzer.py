import os
import re
import logging

from typing import List
from dataclasses import dataclass
from collections import defaultdict
import multiprocessing

from trace_analyzer_utils import calc_ratio, chunks
# original_stderr = sys.__stderr__
# sys.stderr = open('/dev/null', 'w')
import pandas as pd

# sys.stderr = original_stderr

from trace_analyzer import Trace
from column import Column, FINAL_COLUMNS_ORDER
from config import (
    RUN_1_DIR,
    RUN_2_DIR,
    TRACE_DIR,
    LOG_DIR,
    LOG_SUFFIX,
    OUTPUT_SUFFIX_RUN_1,
    OUTPUT_SUFFIX_RUN_2,
    MULTIPROCESSING_NUM_PROCESSES
)
logging.getLogger('pandas').setLevel(logging.NOTSET)


@dataclass
class GraphInfo:
    """
    Holds tnformation on one graph from run_nodels_tests run
    each field holds list of 2 values, one for each run_models_tests run
    """
    graph_index: int
    hw_csv_path: list
    nodes_csv_path: list
    hltv_path: list
    hltv_size: list
    trace_analyser_file_path: list
    trace_analyser_exists: list
    compare_file_path: str
    compare_file_exists: bool

@dataclass
class ModelInfo:
    """
    Holds tnformation on one model from run_nodels_tests run
    each field holds 2 values, one for each run_models_tests run
    """
    name: str
    pathes: tuple
    trace_pathes: tuple
    log_pathes: tuple
    graphs_count: int
    graphs_info: dict

class ModelsTestsAnalyzer:
    """
    analyse profiler output from a run_models_tests run
    anyalyse run_models_test run that was exexuted with:
        - config_compare_file/config_compare_values option
        - hl-prof-config -phase=enq -g 1-255
    analyse all the graphs' traces and compare the analysion between two run_models_tests runs
    For each graph it save report in file system under <model>/trace/hw
    It save comparison repoorts under new 'compare' directory
    analyse
        root_dir - path for directory which contains run_models_tests output
        output_dir - path for output
        nodes - optional list of nodes to analyse. if None - all nodes are analysed
        overwrite_analyze - overwrite analysion. false by default
        overwrite_compare - overwrite comparison repoorts. false by default
        model - optional analyse only one model
        graph_index - optional analyse only one graph
        nodes - optional list of nodes to analyse
        output_dir - optional output directory (by default under <model>/trace/hw)
    """
    def __init__(self,
                 root_dir: str,
                 overwrite_analyze: bool = False,
                 overwrite_compare: bool = False,
                 model: str = None,
                 graph_index: int = None,
                 nodes: list = None,
                 output_dir: str = None):
        self.root_dir = root_dir
        self.overwrite_analyze = overwrite_analyze
        self.overwrite_compare = overwrite_compare
        self.run1_dir = os.path.join(self.root_dir,RUN_1_DIR) # partial_false_dir
        self.run2_dir = os.path.join(self.root_dir,RUN_2_DIR) # partial_true_dir
        self.model = model
        self.graph_index = graph_index
        self.nodes = nodes
        self.output_dir = output_dir
        self.models_info = {}
        self.models_errors = defaultdict(list)
        self.compare_dir = os.path.join(self.root_dir, 'compare')
        self._validate_root_dir()
        self._calculate_models_info()
        self._calculate_to_analyse()
        self._analyse()
        self._compare()
        if not self.model:
            self.analyse_compare()



    def _validate_root_dir(self):
        """
        validation for root_dir input param
        """
        if not isinstance(self.root_dir, str):
            raise ValueError(f"root_dir should be str not {type(self.root_dir)}")
        if not os.path.exists(self.root_dir):
            raise ValueError(f"root_dir not exist in file system. {self.root_dir}")
        if not os.path.isdir(self.root_dir):
            raise ValueError(f"{self.root_dir} is not directory")
        if not os.path.exists(self.run1_dir) or not os.path.isdir(self.run1_dir):
            raise ValueError(f"invalid partial_false_dir {self.run1_dir}")
        if not os.path.exists(self.run2_dir) or not os.path.isdir(self.run2_dir):
            raise ValueError(f"invalid partial_true_dir {self.run2_dir}")


    def calculate_graphs_info(self,
                              trace_dir_path: str,
                              run_id: int,
                              graphs_info: List[GraphInfo],
                              errors: List[str]):
        """
        Updates graphs_info list with information of all graphs of one model from one run of run_models_tests
        """
        for filename in os.listdir(trace_dir_path):
            match = re.match(r'(.*)_hw_prof_accel(\d)_(\d+)\.csv$',filename)
            if match:
                card_id = match.group(2)
                model = match.group(1)
                graph_index = match.group(3)
                graph_index_int = int(graph_index)
                base_file_name = f"{model}_hw_prof_accel{card_id}_{graph_index}"
                suffix = OUTPUT_SUFFIX_RUN_1 if run_id ==0 else OUTPUT_SUFFIX_RUN_2
                trace_analyser_file_name = f"trace_analyser_{model}_{int(graph_index)}_{suffix}.csv"
                trace_analyser_file_path = os.path.join(trace_dir_path,trace_analyser_file_name)
                trace_analyser_exists = os.path.exists(trace_analyser_file_path)
                nodes_csv_file_name = f"{base_file_name}_analyzed_nodes.csv"
                nodes_csv_file_path = os.path.join(trace_dir_path,nodes_csv_file_name)
                hltv_file_name = f"{base_file_name}.hltv"
                hltv_file_path = os.path.join(trace_dir_path,hltv_file_name)
                hw_csv_file_path = os.path.join(trace_dir_path,filename)
                if not os.path.exists(nodes_csv_file_path):
                    errors.append(f"graph {graph_index}: nodes csv not exist for run_id:{run_id}")
                    continue
                if not os.path.exists(hw_csv_file_path):
                    errors.append(f"graph {graph_index}: hw csv not exist for run_id:{run_id}")
                    continue
                if not os.path.exists(hltv_file_path):
                    errors.append(f"graph {graph_index}: hltv not exist for run_id:{run_id}")
                hltv_size =  (os.path.getsize(hltv_file_path) /1024) / 1024 if os.path.exists(hltv_file_path) else 0
                compare_file_path = os.path.join(self.compare_dir,model,f"trace_compare_{model}_{graph_index_int}.csv")
                compare_file_exists = os.path.exists(compare_file_path)
                if run_id == 0:
                    graphs_info[graph_index_int] = GraphInfo(graph_index=graph_index_int,
                                                            hw_csv_path=[hw_csv_file_path],
                                                            nodes_csv_path=[nodes_csv_file_path],
                                                            hltv_path=[hltv_file_path],
                                                            hltv_size=[hltv_size],
                                                            trace_analyser_file_path=[trace_analyser_file_path],
                                                            trace_analyser_exists=[trace_analyser_exists],
                                                            compare_file_path=compare_file_path,
                                                            compare_file_exists=compare_file_exists)
                if run_id == 1:
                    if graph_index_int not in  graphs_info:
                        errors.append(f"graph {graph_index}: only in run_id:{run_id}")
                        continue
                    graphs_info[graph_index_int].hw_csv_path.append(hw_csv_file_path)
                    graphs_info[graph_index_int].nodes_csv_path.append(nodes_csv_file_path)
                    graphs_info[graph_index_int].hltv_path.append(hltv_file_path)
                    graphs_info[graph_index_int].hltv_size.append(hltv_size)
                    graphs_info[graph_index_int].trace_analyser_file_path.append(trace_analyser_file_path)
                    graphs_info[graph_index_int].trace_analyser_exists.append(trace_analyser_exists)
        if run_id == 1:
            for k,v in dict(graphs_info).items():
                if len(v.hw_csv_path) != 2 or len(v.nodes_csv_path) != 2:
                    errors.append(f"graph {k}: csv not in true")
                    del graphs_info[k]


    def _calculate_models_info(self):
        """
        Updates self.models_info with infromation taken from run_models_tests run
        self.models_info is dictionary - {model_name: ModelInfo}
        """
        false_models = os.listdir(self.run1_dir)
        true_models = os.listdir(self.run2_dir)
        for model in false_models:
            graphs_info = {}
            if not model in true_models:
                self.models_errors[model].append("not in both dirs")
                continue
            model_path_false = os.path.join(self.run1_dir, model)
            model_path_true = os.path.join(self.run2_dir, model)
            if not os.path.isdir(model_path_false) or not os.path.isdir(model_path_true):
                self.models_errors[model].append("not exist in both dirs")
                continue
            trace_dir_path_false = os.path.join(model_path_false,TRACE_DIR)
            trace_dir_path_true = os.path.join(model_path_true,TRACE_DIR)
            if not os.path.exists(trace_dir_path_false) or not os.path.exists(trace_dir_path_true):
                self.models_errors[model].append(f"traces dir not exist in both dirs.{trace_dir_path_false},{trace_dir_path_true}")
                continue
            log_path_false = os.path.join(model_path_false,LOG_DIR,f"{model}{LOG_SUFFIX}")
            log_path_true = os.path.join(model_path_true,LOG_DIR,f"{model}{LOG_SUFFIX}")
            if not os.path.exists(log_path_false):
                log_path_false = None
            if not os.path.exists(log_path_true):
                log_path_true = None
            graphs_info = {}
            trace_dirs = [trace_dir_path_false,trace_dir_path_true]
            for i in range(2):
                self.calculate_graphs_info(trace_dir_path=trace_dirs[i],
                                            run_id=i,
                                            graphs_info=graphs_info,
                                            errors = self.models_errors[model])

            self.models_info[model] = ModelInfo(name=model,
                                                pathes=(model_path_false,model_path_true),
                                                trace_pathes = (trace_dir_path_false,trace_dir_path_true),
                                                log_pathes= (log_path_false, log_path_true),
                                                graphs_info=graphs_info,
                                                graphs_count=len(graphs_info)
                                                )


    def _calculate_to_analyse(self, to_compare=False):
        """
        Prepares to_analyse list that hold all models and graphs that need to handle
        to_compare - indicates if the handling is analysion or comparison
        """
        to_analyse = []
        if self.model:
            if self.model not in self.models_info:
                print(f"There is no info for {self.model}")
                return to_analyse
            if self.graph_index:
                if self.graph_index not in self.models_info[self.model].graphs_info.keys():
                    print(self.models_info[self.model].graphs_info.keys())
                    print(f"There is no info for {self.model} graph:{self.graph_index}")
                    return to_analyse
                graphs =[self.graph_index]
            else:
                graphs = list(self.models_info[self.model].graphs_info.keys())
                print(f"self.models_info[self.model].graphs_info.keys:{self.models_info[self.model].graphs_info.keys}")
            if not to_compare and not self.overwrite_analyze:
                graphs = [graph for graph in graphs if not self.models_info[self.model].graphs_info[graph].trace_analyser_exists]
            if to_compare and not self.overwrite_compare:
                graphs = [graph for graph in graphs if not self.models_info[self.model].graphs_info[graph].compare_file_exists]
            if graphs:
                to_analyse.append((self.model,graphs))
        else:
            for model_name in self.models_info:
                graphs = list(self.models_info[model_name].graphs_info.keys())
                if not to_compare and not self.overwrite_analyze:
                    graphs = [graph for graph in graphs if not self.models_info[model_name].graphs_info[graph].trace_analyser_exists]
                if to_compare and not self.overwrite_compare:
                    graphs = [graph for graph in graphs if not self.models_info[model_name].graphs_info[graph].compare_file_exists]
                if graphs:
                    to_analyse.append((model_name,graphs))
        return to_analyse

    def _compare(self):
        """
        Creats reports with comparison between the 2 runs of run_models_tests
        """
        to_analyse = self._calculate_to_analyse(to_compare=True)
        print(f"start compare:{len(to_analyse)} models")
        os.makedirs(self.compare_dir, exist_ok=True)
        for model, graphs in to_analyse:
            print(f"start compare:{model}, {len(graphs)} graphs")
            compare_model_dir = os.path.join(self.compare_dir, model)
            os.makedirs(compare_model_dir, exist_ok=True)
            for graph_index in graphs:
                analyser_files = self.models_info[model].graphs_info[graph_index].trace_analyser_file_path
                df1 = pd.read_csv(analyser_files[0], usecols=lambda x: x != 0) if os.path.exists(analyser_files[0]) else None
                df2 = pd.read_csv(analyser_files[1], usecols=lambda x: x != 0) if os.path.exists(analyser_files[1]) else None
                if df1 is not None and df2 is not None and not df1.empty and not df2.empty:
                    merged_df = pd.merge(df1, df2, on=[Column.NODE_NAME,'Op Type'], how='inner')
                    if not merged_df.empty:
                        only_in_1_df = df1[~df1[Column.NODE_NAME].isin(df2[Column.NODE_NAME])]
                        only_in_2_df = df2[~df2[Column.NODE_NAME].isin(df1[Column.NODE_NAME])]
                        pk_columns = list(merged_df.columns[:2])
                        other_columns = list(merged_df.columns[2:])
                        column_in_one_df = len(other_columns)//2
                        new_order = [*pk_columns]
                        for i in range(column_in_one_df):
                            new_order.append(other_columns[i])
                            new_order.append(other_columns[column_in_one_df + i])
                        merged_df = merged_df[new_order]

                        ratio_columns = merged_df.apply(self.calculate_ratio,axis=1)
                        merged_df = pd.concat([merged_df,ratio_columns],axis=1)
                        output_file = os.path.join(compare_model_dir,f"trace_compare_{model}_{graph_index}.csv")
                        output_file_only_in_1 = os.path.join(compare_model_dir,f"{model}_{graph_index}_only_in_f.csv")
                        output_file_only_in_2 = os.path.join(compare_model_dir,f"{model}_{graph_index}_only_in_t.csv")
                        merged_df = merged_df[FINAL_COLUMNS_ORDER]
                        merged_df.to_csv(output_file, index=True)
                        only_in_1_df.to_csv(output_file_only_in_1, index=True)
                        only_in_2_df.to_csv(output_file_only_in_2, index=True)


    def _analyse(self):
        """
        Create report for each graph's trace
        use multiprocessing to speed up the work
        """
        to_analyse = self._calculate_to_analyse(to_compare=False)
        if not to_analyse:
            print(f"no new trace to analyze")
        else:
            print(f"start analyze {len(to_analyse)} moedls")
            num_model_for_process = len(to_analyse)//MULTIPROCESSING_NUM_PROCESSES
            print(f"num_model_for_process{num_model_for_process}")
            processes = []
            for i,sub_to_analyze in enumerate(chunks(to_analyse, MULTIPROCESSING_NUM_PROCESSES)):
                if len(sub_to_analyze):
                    print(f"starting process {i}:{len(sub_to_analyze)}")
                    process = multiprocessing.Process(target=self.analyze_one_model, args=(i,sub_to_analyze))
                    processes.append(process)
                    process.start()
            for process in processes:
                process.join()

    def analyze_one_model(self,task_id,to_analyze):
        """
        Analyse all graphs of one model
        """
        try:
            print(f"{task_id}: start for {len(to_analyze)} models")
            for model,graphs in to_analyze:
                print(f"{task_id}: Analyzing {model} ({len(graphs)} graphs)")
                for graph_index in graphs:
                    print(f"{task_id}: Analyzing {model} graph {graph_index}")
                    trace1 = Trace(model=model,
                                graph_index=graph_index,
                                nodes = self.nodes,
                                output_dir=self.models_info[model].trace_pathes[0] if not self.output_dir else self.output_dir,
                                nodes_csv_path=self.models_info[model].graphs_info[graph_index].nodes_csv_path[0],
                                hw_events_csv_path=self.models_info[model].graphs_info[graph_index].hw_csv_path[0],
                                hltv_path=self.models_info[model].graphs_info[graph_index].hltv_path[0],
                                hltv_size=self.models_info[model].graphs_info[graph_index].hltv_size[0],
                                log_file_path=self.models_info[model].log_pathes[0],
                                suffix=OUTPUT_SUFFIX_RUN_1,
                                overwrite=self.overwrite_analyze)
                    trace2 = Trace(model=model,
                                graph_index=graph_index,
                                nodes = self.nodes,
                                output_dir=self.models_info[model].trace_pathes[1] if not self.output_dir else self.output_dir,
                                nodes_csv_path=self.models_info[model].graphs_info[graph_index].nodes_csv_path[1],
                                hw_events_csv_path=self.models_info[model].graphs_info[graph_index].hw_csv_path[1],
                                hltv_path=self.models_info[model].graphs_info[graph_index].hltv_path[1],
                                hltv_size=self.models_info[model].graphs_info[graph_index].hltv_size[1],
                                log_file_path=self.models_info[model].log_pathes[1],
                                suffix=OUTPUT_SUFFIX_RUN_2,
                                overwrite=self.overwrite_analyze)
                print(f"{task_id}: end for {model} ({len(graphs)} graphs)")
        except Exception as e:
            print(f"error: {e} task_id:{task_id}")


    def analyse_compare(self):
        """
        Prepare final report with Comparison between the 2 runs on all models
        output:
        false_partials_handled.csv - all nodes that were handled by partial write pass but no issue was detected in trace
        parials_exists.csv - all nodes that were not handled but partial issue was detected in trace
        """
        models = os.listdir(self.compare_dir)
        print(f"start analyze comparation of {len(models)} models")
        for model in models:
            model_dir = os.path.join(self.compare_dir,model)
            if (not self.model or self.model == model) and os.path.isdir(model_dir):
                filtered_dfs = []
                compare_files = [f for f in os.listdir(model_dir) if f.startswith('trace_compare')]
                for compare_file in compare_files:
                    match = re.match(r'trace_compare_.*_(\d*).csv', compare_file)
                    if match:
                        graph_index = match.group(1)
                    else:
                        print(f"error matching graph index:{compare_file}")
                    compare_graph_path = os.path.join(model_dir,compare_file)
                    df = pd.read_csv(compare_graph_path, sep=',')
                    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
                    if not df.empty:
                        filtered_df = df[df[ Column.GOOD_HANDLING] == False].copy()
                        filtered_df['graph_index'] = graph_index
                        filtered_df['model'] = model
                        cols = list(filtered_df.columns)
                        cols.insert(1, cols.pop(cols.index('graph_index')))
                        cols.insert(1, cols.pop(cols.index('model')))
                        filtered_df = filtered_df[cols]
                        filtered_dfs.append(filtered_df)
                if filtered_dfs:
                    final_df = pd.concat(filtered_dfs, ignore_index=True)
                    if not final_df.empty:
                        final_partials_handled_df = final_df[final_df[Column.PARTIAL_ISSUE_HANDLED] == True]
                        final_partials_exists_df = final_df[final_df[Column.PARTIAL_ISSUE_EXISTS] == True]

                        final_df = final_df.dropna(subset=[Column.DURATION_TILL_END_WRITE_RATIO])
                        final_partials_handled_df = final_partials_handled_df.dropna(subset=[Column.DURATION_TILL_END_WRITE_RATIO])
                        final_partials_exists_df = final_partials_exists_df.dropna(subset=[Column.DURATION_TILL_END_WRITE_RATIO])
                        final_df.to_csv(os.path.join(model_dir,'final.csv'))
                        final_partials_handled_df.to_csv(os.path.join(model_dir,'partials_handled.csv'))
                        final_partials_exists_df.to_csv(os.path.join(model_dir,'parials_exists.csv'))
        if not self.model:
            sum_dfs = []
            for model in os.listdir(self.compare_dir):
                # print(f"model:{model}")
                model_dir = os.path.join(self.compare_dir,model)
                if os.path.isdir(model_dir):
                    sum_path = os.path.join(model_dir,'partials_handled.csv')
                    if os.path.exists(sum_path):
                        sum_df = pd.read_csv(sum_path)
                        if not sum_df.empty:
                        # print(f"\tmodel:{model} {len(sum_df)} nodes false partials handled")
                            sum_dfs.append(sum_df)
            if sum_dfs:
                sum_all_df = pd.concat(sum_dfs, ignore_index=True)
                positive_df = sum_all_df[sum_all_df[Column.DURATION_TILL_END_WRITE_RATIO] > 0].sort_values(by=[Column.DURATION_TILL_END_WRITE_RATIO, 'Node Name'], ascending=[False, False])
                negative_df= sum_all_df[sum_all_df[Column.DURATION_TILL_END_WRITE_RATIO] <= 0].sort_values(by=[Column.DURATION_TILL_END_WRITE_RATIO, 'Node Name'], ascending=[False, False])
                negative_df_2= sum_all_df[sum_all_df[Column.DURATION_TILL_END_WRITE_RATIO] <= 0].sort_values(by=['Node Name',Column.DURATION_TILL_END_WRITE_RATIO], ascending=[False, False])
                # Concatenate the sorted partitions
                sum_all_df_sorted = pd.concat([positive_df, negative_df])
                sum_all_df_sorted = sum_all_df_sorted.loc[:, ~sum_all_df_sorted.columns.str.contains('^Unnamed')]
                negative_df_2 = negative_df_2.loc[:, ~negative_df_2.columns.str.contains('^Unnamed')]
                sum_all_df_sorted.to_csv(os.path.join(self.compare_dir,'false_partials_handled.csv'))
                negative_df_2.to_csv(os.path.join(self.compare_dir,'false_partials_handled_negetive.csv'))
            sum_dfs = []
            for model in os.listdir(self.compare_dir):
                model_dir = os.path.join(self.compare_dir,model)
                if os.path.isdir(model_dir):
                    sum_path = os.path.join(model_dir,'parials_exists.csv')
                    if os.path.exists(sum_path):
                        sum_df = pd.read_csv(sum_path)
                        if not sum_df.empty:
                            # print(f"\tmodel:{model} {len(sum_df)} nodes false partials not handled")
                            sum_dfs.append(sum_df)
            if sum_dfs:
                sum_all_df = pd.concat(sum_dfs, ignore_index=True)
                sum_all_df = sum_all_df.loc[:, ~sum_all_df.columns.str.contains('^Unnamed')]
                sum_all_df.to_csv(os.path.join(self.compare_dir,'partials_not_handled.csv'))




    def calculate_ratio(self, row):
        column_names = row.index

        backpressure_pair_columns = [col for col in column_names if col.startswith(Column.BACKPRESSURE_COUNT)]
        backpressure_diff = row[backpressure_pair_columns[1]] - row[backpressure_pair_columns[0]]


        partials_pair_columns = [col for col in column_names if col.startswith(Column.PARTIALS_COUNT)]
        partial_diff = row[partials_pair_columns[1]] - row[partials_pair_columns[0]]

        pair_columns = [col for col in column_names if col.startswith(Column.MEM_WRITE_COUNT)]
        mem_write_diff = row[pair_columns[1]] - row[pair_columns[0]]


        pair_columns = [col for col in column_names if col.startswith('duration_with_write')]


        duration_till_end_write_ratio = calc_ratio(row[pair_columns[1]],row[pair_columns[0]])

        duration_max_engine_pair_columns = [col for col in column_names if col.startswith(Column.DURATION_MAX_ENGINE)]

        duration_max_engine_ratio = calc_ratio(row[duration_max_engine_pair_columns[1]],row[duration_max_engine_pair_columns[0]])


        parallel_engine_pair_columns = [col for col in column_names if col.startswith('Parallel Engines')]
        is_cache_warmup_pair_columns = [col for col in column_names if col.startswith(Column.IS_CACHE_WARM_UP)]

        is_partial_issue_identified = False
        is_partial_issue = False
        if (row[parallel_engine_pair_columns[1]] == 16 and row[parallel_engine_pair_columns[1]] < row[parallel_engine_pair_columns[0]]) or  row[is_cache_warmup_pair_columns[1]] is True:
            is_partial_issue_identified = True

        if row[partials_pair_columns[0]] > 0 and row[backpressure_pair_columns[0]] > 0:
            is_partial_issue = True

        is_good_handling = is_partial_issue_identified == is_partial_issue

        return pd.Series({Column.BACKPRESSURE_DIFF: backpressure_diff, Column.PARTIAL_DIFF: partial_diff,
                          Column.MEMWRITE_DIFF: mem_write_diff,
                          Column.DURATION_TILL_END_WRITE_RATIO: duration_till_end_write_ratio,
                          Column.PARTIAL_ISSUE_EXISTS: is_partial_issue,Column.PARTIAL_ISSUE_HANDLED: is_partial_issue_identified,
                          Column.GOOD_HANDLING: is_good_handling, Column.DURATION_MAX_ENGINE_RATIO: duration_max_engine_ratio})


if __name__ == "__main__":
    pass
    # path = "/home/ykopfstein/qnpu/1.15.0-319/src/synapse/8_5/run"
    # m = ModelsTestsAnalyzer(path,overwrite_analyze=False,overwrite_compare=True)
    # m.analyse_compare()
    # m = ModelsTestsAnalyzer(path,overwrite_analyze=True,overwrite_compare=False,model='pt_mid_journey_1x',graph_index=7,nodes=['fusedTPCNode_5_303'],output_dir='/home/ykopfstein/qnpu/1.15.0-319/src/synapse/8_5/temp')
