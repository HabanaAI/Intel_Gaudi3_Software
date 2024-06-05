import argparse
from models_tests_analyzer import ModelsTestsAnalyzer


"""
Overview:
    This tool analyses the output of run_model_tests that was executed with --config_compare_file or
    --config_compare_values options.
    it analsys the traces, compare the results between the two runs and outpus final report on the results

example of useage:
    python trace_analyzer/main.py <models_tests_output_directory>
    this execution will create new directory  <models_tests_output_directory>/compare that holds comparation report
    for all models + final report that summaries all

current usage:
    it compares time performance between running with and without execution of partials writes handaling
    fore each node on each run we gather the below information:
    1. the running time of the node
       a. Duration of running
       b. Duration from satrt of running till the last TPC_END_OF_KERNEL_MESSAGE_WRITE event
       c. Duration from start of cache warm node (if exists) till the last TPC_END_OF_KERNEL_MESSAGE_WRITE event
       d. if node was "partial handled" information regarding the reason for handling (from log)
    2. Whether 'Back pressure from memory to TPC write queue on store tensors' hw counter has exceeded a threshold
       during node's running time
    3. Whether 'Outgoing HBM partial writes'  hw counter has exceeded a threshold during node's running time
    4. Whether "partials handling" was done on this node:
        a. cache warmup nodes were added
        b. working engines number was reduced

    The runs comparison outputs reports regarding:
    1. Nodes with partials issue (acoording hw counters) that are not handled
    2. Nodes that are handled but have no partials issue (acoording hw counters)

Notes:
    1. it uses pandas. (pip install pandas)
    2. configuration for run_models_tests:
        a. The input hw traces requeried to be in csv format + hltv (hl-prof-config --invoc hltv,csv)
        b. hw traces should be diveded for graphs (hl-prof-config -phase=enq -g 1-255)
        c. for each graph's trace the tool require this information:
            - hw_prof_accel csv
            - analyzed_nodes csv
            - graph_compiler.log
        d. LOG_LEVEL_LB_PARTIALS=0

Output:
    1. For each graphs trace a csv report on graph's nodes (i.e trace_analyser_pt_wav2vec2_8x_g2_34_t.csv)
    2. Under 'compare' directory for each graph a csv with a comparison between the two runs
    (i.e.trace_compare_pt_wav2vec2_8x_g2_41.csv)
    3. reports that summeries all nodes of all models:
        a. partials_not_handled.csv - Nodes with partials issue (acoording hw counters) that are not handled
        b. false_partials_handled.csv - Nodes that are handled but have no partials issue (acoording hw counters)


Components:
    1.log_analyzer.py:
    2.trace_analyzer.py:
    3.models_tests_analyzer.pycd




"""





def validate_args(args):
    if args.graph_index and not args.model:
        raise ValueError("graph index without model is invalid")

def main(args):
    ModelsTestsAnalyzer(root_dir=args.working_dir,
                        overwrite_analyze=args.overwrite_analyze,
                        overwrite_compare=args.overwrite_compare,
                        model=args.model,
                        graph_index=args.graph_index,
                        nodes=args.nodes,
                        output_dir=args.output_dir)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'working_dir',
        help="working dir with models_tests output"
        )

    parser.add_argument(
        "-o",
        "--output-dir",
        nargs='?',
        help='directory to put the analysed traces output. if not given the output is the directory of each hw trace'
    )

    parser.add_argument(
        "--overwrite-analyze",
        action='store_true',
        help="overwrite the trace analysis.",
    )

    parser.add_argument(
        "--overwrite-compare",
        action='store_true',
        help="overwrite the two runs comparation",
    )

    parser.add_argument(
        "-m",
        "--model",
        nargs='?',
        help="analyse only one model",
    )

    parser.add_argument(
        "-g",
        "--graph-index",
        nargs='?',
        type=int,
        help="analyse only one graph (model should be given too)",
    )
    parser.add_argument(
        "-n",
        "--nodes",
        nargs='+',
        help="analyse only specific nodes",
    )
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()
    print(args)
    validate_args(args)
    main(args)