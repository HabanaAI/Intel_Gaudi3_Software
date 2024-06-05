import os
import time
import logging
import subprocess

from typing import List, Optional
from dataclasses import dataclass, field

import pandas as pd

from log_analyzer import LogAnalyzer
from column import Column, TRACE_ANALYZER_OUTPUT_COLUMNS
from config import (
    BACK_PRESSURE_THRESHOLD,
    PARTIAL_WRITES_THRESHOLD,
    BACK_PRESSURE_EVENT_NAME,
    PARTIAL_WRITES_EVENT_NAME,
    OUTGOING_MEM_WRITE_REQUESTS_EVENT_NAME,
    END_OF_WRITE_EVENT_DESCIPTION,
    TPC_SPU_START_EVENT_DESCRIPTION,
    TPC_VALUE,
    LOG_INFO_PARTIAL_INFO_KEY,
    LOG_INFO_CACHE_WARMUP_NODE_NAMES,
    TPC_SPU_START_TO_SPU_HALT_EVENT_DESCRIPTION
)
logging.getLogger('pandas').setLevel(logging.NOTSET)


@dataclass
class Trace:
    """
    Analyse trace of one graph according to profiler outputs + graph compiler's log.
    outputs csv report

    input parameters:
        model - model name
        graph_index -
        output_dir -
        log_file_path - optional path for graph compiler log
        hltv_path - currently optional - just to put the path in output report
        hltv_size - optional just to put the path in output report
        nodes_csv_path - path for analyzed_nodes csv (profiler's output)
        hw_events_csv_path - node for hw csv (profiler's output)
        overwrite - indicates if to overwrite previous report. false by default
        nodes - optional - list of nodes to analyse
        extract_from_hltv - currently should be always false

        output:
            new csv file under output_dir ("trace_analyser_{self.model}_{self.graph_index}_{self.suffix}.csv")
    """
    model: str
    graph_index: int
    output_dir: str
    log_file_path: str = None
    hltv_path: str = None
    hltv_size: int = None
    nodes_csv_path: str = None
    hw_events_csv_path: str = None
    overwrite: bool = False
    nodes: Optional[List[str]] = None
    suffix: Optional[str] = ''
    nodes: Optional[List[str]] = None
    extract_from_hltv: bool = False
    nodes_df: pd.DataFrame = field(init=False)
    tpc_nodes_df: pd.DataFrame = field(init=False)
    hw_events_df: pd.DataFrame = field(init=False)
    trace_analysed_path: str = field(init=False)
    log_info: dict = field(init=False)

    def __post_init__(self):
        """
        analyse csv on init
        """
        self._validate_input_files()
        self.log_info =  LogAnalyzer(self.log_file_path).get_nodes_log_info() if self.log_file_path else None
        suffix = f"_{self.suffix}" if self.suffix else ''
        self.trace_analysed_path = os.path.join(self.output_dir,f"trace_analyser_{self.model}_{self.graph_index}{suffix}.csv")
        if self.overwrite:
            if os.path.exists(self.trace_analysed_path):
                os.remove(self.trace_analysed_path)

        if not os.path.exists(self.trace_analysed_path):
            columns_to_read = [Column.NAME, Column.ENGINE, Column.TIMESTAMPUSEC,Column.HW_EVENT_DESCRIPTION, Column.SPMU_VALUE, Column.EVENT_TYPE]
            self.hw_events_df = pd.read_csv(self.hw_events_csv_path, sep=',', low_memory=False, usecols=columns_to_read)
            columns_to_read = [Column.UNIT, Column.UNIQUE_NODE_ID, Column.NODE_NAME , Column.OP_TYPE,
                               Column.PARALLEL_ENGINES, Column.START_TIME_OF_NODE , Column.END_TIME_OF_NODE,
                               Column.DURATION_US]
            self.nodes_df = pd.read_csv(self.nodes_csv_path, sep=',',  usecols=columns_to_read)
            self._preprocess_dfs()
            if len(self.tpc_nodes_df):
                self.analyse()
            else:
                print(f"0 nodes to analyze for {self.model},graph {self.graph_index}. csv path:{self.nodes_csv_path}")
    def __str__(self):
        return(f"model:{self.model},graph:{self.graph_index},nodes_csv_path:{self.nodes_csv_path},hw_csv_path:{self.hw_events_csv_path}")

    def _validate_input_files(self):
        if self.extract_from_hltv and self.hltv_path:
            if self.hw_events_csv_path or self.nodes_csv_path:
                raise ValueError("only hltv file is needed")
            if not os.path.exists(self.hltv_path):
                 raise ValueError(f"{self.hltv_path} not exists")
            self._extract_csv_from_hltv()
        else:
            if not self.hw_events_csv_path or not self.nodes_csv_path:
                raise ValueError("both csvs (hw, analysed nodes ) are needed")
            if not os.path.exists(self.hw_events_csv_path):
                raise ValueError(f"{self.hw_events_csv_path} not exists")
            if not os.path.exists(self.nodes_csv_path):
                raise ValueError(f"{self.nodes_csv_path} not exists")
        if self.log_file_path:
            if not os.path.exists(self.log_file_path):
                raise ValueError(f"{self.log_file_path} not existsis")


    def _extract_csv_from_hltv(self):
        """
        extract needed csv files from hltv by using profiler's bin
        # TODO currently _extract_csv_from_hltv not working. check with profiler team
        """
        root_dir = os.path.dirname(self.hltv_path)
        file_name = os.path.basename(self.hltv_path)
        file_name_no_ext, _ = os.path.splitext(file_name)
        graph_index = file_name_no_ext.split('_')[-1]

        self.hw_events_csv_path = os.path.join(root_dir,f"{file_name_no_ext}.csv")
        self.nodes_csv_path = os.path.join(root_dir,f"{file_name_no_ext}_analyzed_nodes.csv")

        if not os.path.exists(self.hw_events_csv_path) or not os.path.exists(self.nodes_csv_path):
            cmd = f"unzip -o {self.hltv_path};"
            cmd += f"$SYNAPSE_PROFILER_RELEASE_BUILD/bin/synprof_parser  {root_dir}/prof-data/bin/pt_mid_journey_8x_hw_prof_accel0_{graph_index}.bin "
            cmd += f"-gaudi3 -conf {root_dir}/prof-data/config/prof_config.json "
            cmd +=f"-post_graph_dir {root_dir}/prof-data/post-graph/ -dbg_dir {root_dir}/prof-data/debug-info/ -csv;ls {root_dir}"
            print(f"cmd:{cmd}")
            result = subprocess.run(cmd, cwd=root_dir, shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"error extracting from hltv:{result.returncode},{result.stderr}")

        if not os.path.exists(self.hw_events_csv_path):
            print(f"error, failed to generate {self.hw_events_csv_path}")
        if not os.path.exists(self.nodes_csv_path):
            print(f"error, failed to generate {self.nodes_csv_path}")

    def _preprocess_dfs(self):
        """
        preprocess input data
        """
        # self.hw_events_df[Column.TIMESTAMPUSEC] = pd.to_datetime(self.hw_events_df[Column.TIMESTAMPUSEC], unit='us')
        self.hw_events_df.sort_values(by=Column.TIMESTAMPUSEC, inplace=True)
        self.tpc_nodes_df = self.nodes_df[self.nodes_df[Column.UNIT] == TPC_VALUE]
        # self.tpc_nodes_df[Column.START_TIME_OF_NODE] = pd.to_datetime(self.tpc_nodes_df[Column.START_TIME_OF_NODE], unit='us')
        # self.tpc_nodes_df[Column.END_TIME_OF_NODE] = pd.to_datetime(self.tpc_nodes_df[Column.END_TIME_OF_NODE], unit='us')
        self.tpc_nodes_df = self.tpc_nodes_df.sort_values(by=Column.START_TIME_OF_NODE).copy()
        self.tpc_nodes_df.reset_index(drop=True, inplace=True)
        log_columns = self.tpc_nodes_df.apply(self.calculate_log_info, axis=1)
        self.tpc_nodes_df = pd.concat([self.tpc_nodes_df,log_columns],axis=1)

        if self.nodes:
            # filter tpc_nodes_df to hold only the analysed nodes
            print(f"analysis is done only for {self.nodes}")
            filtered_nodes_df = self.tpc_nodes_df[self.tpc_nodes_df[Column.NODE_NAME].isin(self.nodes)]
            filtered_nodes_with_cache_warm_up_df = filtered_nodes_df[filtered_nodes_df[Column.CACHE_WARM_UP_NAME] != False]
            if not filtered_nodes_with_cache_warm_up_df.empty:
                cache_warm_up_nodes_lists = filtered_nodes_with_cache_warm_up_df[Column.CACHE_WARM_UP_NAME].to_list()
                cache_warm_up_nodes_list = []
                for sublist in cache_warm_up_nodes_lists:
                    cache_warm_up_nodes_list.extend(sublist)
                filtered_cache_warm_up_df = self.tpc_nodes_df[self.tpc_nodes_df[Column.NODE_NAME].isin(cache_warm_up_nodes_list)]
                self.tpc_nodes_df = pd.concat([filtered_nodes_df, filtered_cache_warm_up_df], ignore_index=True)
                print(f"analysis is done also for {cache_warm_up_nodes_list}")
            else:
                self.tpc_nodes_df = filtered_nodes_df


    ### functions that works on dataframe's row
    def calculate_log_info(self, row):
        """
        Take Node's related info from log and add it to new columns.
        """
        is_cache_warm_up = False
        cache_warm_up_names = False
        real_start_time = row[Column.START_TIME_OF_NODE]
        log = None
        if self.log_info and row[Column.NODE_NAME] in self.log_info:
            if LOG_INFO_PARTIAL_INFO_KEY in self.log_info[row[Column.NODE_NAME]]:
                log = self.log_info[row[Column.NODE_NAME]][LOG_INFO_PARTIAL_INFO_KEY]
            if LOG_INFO_CACHE_WARMUP_NODE_NAMES in self.log_info[row[Column.NODE_NAME]]:
                is_cache_warm_up = True
                cache_warm_up_names = self.log_info[row[Column.NODE_NAME]][LOG_INFO_CACHE_WARMUP_NODE_NAMES]
                for warmup_name in cache_warm_up_names:
                    real_start_time = min(real_start_time, self.tpc_nodes_df[self.tpc_nodes_df[Column.NODE_NAME] == warmup_name][Column.START_TIME_OF_NODE].iloc[0])
        return pd.Series({Column.IS_CACHE_WARM_UP: is_cache_warm_up,
                          Column.CACHE_WARM_UP_NAME: cache_warm_up_names,
                          Column.REAL_START_TIME: real_start_time,
                          Column.LOG: log})


    def calculate_counter_value(self, row,hw_event_df, do_count, debug=False):
        """
        gets row from nodes dataframe and hw_event_df that holds rows of only one event.
        caclulates information regarding the HW events during the running time of this specific row
        """
        counter_val = 0
        start_time = row[Column.START_TIME_OF_NODE]
        end_time = row[Column.END_TIME_OF_NODE]
        # print(f"strat:{start_time}")
        # print(f"end:{end_time}")
        if len(hw_event_df) > 0:
            start_idx = hw_event_df[hw_event_df[Column.TIMESTAMPUSEC] < start_time].index.max()
            end_idx = hw_event_df[hw_event_df[Column.PREV_SAMPLE_TIME] > end_time].index.min()
            if pd.isna(end_idx):
                end_idx = hw_event_df.index.max()
            if pd.isna(start_idx):
                start_idx = hw_event_df.index.min()

            filtered_df = hw_event_df.loc[start_idx:end_idx]
            # if debug:
            #     print(f"{start_idx}:{end_idx}")
            #     filtered_df.to_csv("check_bp0.csv")
            #     hw_event_df.to_csv("check_hw.csv")
            filtered_df = filtered_df[((filtered_df[Column.TIMESTAMPUSEC] >= start_time) & (filtered_df[Column.TIMESTAMPUSEC] <= end_time)) |
                                    ((filtered_df[Column.PREV_SAMPLE_TIME] < end_time) & (filtered_df[Column.TIMESTAMPUSEC] > end_time))]
            # if debug:
            #     filtered_df.to_csv("check_bp.csv")
            if do_count:
                counter_val = len(filtered_df)
            else:
                counter_val = filtered_df[Column.SPMU_VALUE].sum()
        return counter_val

    def get_times_for_one_engine(self, group):
        """
        function for groupby.
        group holds all the events that related to one node on one engine
        returns start time, end time and duration of the node on this engine
        """
        sorted_group = group.sort_values(Column.TIMESTAMPUSEC)
        start = sorted_group.loc[sorted_group[Column.HW_EVENT_DESCRIPTION] == TPC_SPU_START_EVENT_DESCRIPTION,
                                 Column.TIMESTAMPUSEC].iloc[0] if (sorted_group[Column.HW_EVENT_DESCRIPTION] == TPC_SPU_START_EVENT_DESCRIPTION).any() else None
        end = sorted_group.loc[sorted_group[Column.HW_EVENT_DESCRIPTION] == END_OF_WRITE_EVENT_DESCIPTION,
                            Column.TIMESTAMPUSEC].iloc[-1] if (sorted_group[Column.HW_EVENT_DESCRIPTION] == END_OF_WRITE_EVENT_DESCIPTION).any() else None
        if not start:
            start = sorted_group.loc[sorted_group[Column.HW_EVENT_DESCRIPTION] == TPC_SPU_START_TO_SPU_HALT_EVENT_DESCRIPTION,
                            Column.TIMESTAMPUSEC].iloc[0] if (sorted_group[Column.HW_EVENT_DESCRIPTION] == TPC_SPU_START_TO_SPU_HALT_EVENT_DESCRIPTION).any() else 0
        if not end:
            end = sorted_group.loc[sorted_group[Column.HW_EVENT_DESCRIPTION] == TPC_SPU_START_TO_SPU_HALT_EVENT_DESCRIPTION,
                            Column.TIMESTAMPUSEC].iloc[-1] if (sorted_group[Column.HW_EVENT_DESCRIPTION] == TPC_SPU_START_TO_SPU_HALT_EVENT_DESCRIPTION).any() else 0
        if not end or not start:
            start = 0
            end = 0
        return pd.Series({'start_engine': start, 'end engine': end, 'duration_engine': end-start})

    def calculate_duration_info(self, row, hw_event_df):
        """
        This function is called for each row of modes df.
        it returns:
        1. running duration on engine that worked the longest on the node - Column.END_OF_WRITE_MAX
        2. time of the latest END_OF_WRITE_EVENT taken from all engines - Column.DURATION_MAX_ENGINE
        """
        latest_end_write = None
        max_engine_duration = None
        filtered_node_all_events_df = hw_event_df[hw_event_df[Column.NAME] == row[Column.NODE_NAME]]
        if not filtered_node_all_events_df.empty:
            grouped_engines_df = filtered_node_all_events_df.groupby(Column.ENGINE).apply(self.get_times_for_one_engine)
            max_engine_duration = grouped_engines_df['duration_engine'].max()
            latest_end_write = grouped_engines_df['end engine'].max()
            if not latest_end_write:
                latest_end_write = row[Column.END_TIME_OF_NODE]
        return pd.Series({Column.END_OF_WRITE_MAX: latest_end_write, Column.DURATION_MAX_ENGINE: max_engine_duration})

    ####
    def get_hw_event_df(self, event_name, threshold = None):
        """
        return a filtered df that hold only one he event type
        threshold - optional to filter also according smpu value by threshold
        for each event - save information regarding the previous event of this type from the same engine
        """
        filtered_df = self.hw_events_df[self.hw_events_df[Column.NAME] == event_name].copy()
        filtered_df.dropna(axis=1, how='all', inplace=True)
        if len(filtered_df) > 0:
            filtered_df.sort_values(by=Column.TIMESTAMPUSEC, inplace=True, ascending=True)
            filtered_df[Column.PREV_SAMPLE_VALUE] = filtered_df.groupby(Column.ENGINE)[Column.SPMU_VALUE].shift().fillna(value=0)
            filtered_df[Column.PREV_SAMPLE_TIME] = filtered_df.groupby(Column.ENGINE)[Column.TIMESTAMPUSEC].shift().fillna(value=pd.NaT)
            filtered_df[Column.SPMU_VALUE] = filtered_df[Column.SPMU_VALUE].astype(float)
            filtered_df[Column.PREV_SAMPLE_VALUE] = filtered_df[Column.PREV_SAMPLE_VALUE].astype(float)
            # filtered_df.to_csv(os.path.join(self.output_dir,f"hw_{event_name}.csv"), index=False)
        if threshold:
            if Column.SPMU_VALUE in filtered_df.columns:
                filtered_df = filtered_df[filtered_df[Column.SPMU_VALUE] >= threshold]

        return filtered_df

    def calculate_tpc_nodes_partial_writes_detected(self):
        """
        Add information redaring partial writes for each node in nodes df.
        - count the high backpressure samples that were sampled during node's running duration
        - count the high partial writes samples that were sampled during node's running duration
        - count the mem writes that were done during node's running duration
        """
        high_back_pressure_events_df = self.get_hw_event_df(BACK_PRESSURE_EVENT_NAME, BACK_PRESSURE_THRESHOLD)
        high_partial_writes_events_df = self.get_hw_event_df(PARTIAL_WRITES_EVENT_NAME, PARTIAL_WRITES_THRESHOLD)
        memwrite_events_df = self.get_hw_event_df(OUTGOING_MEM_WRITE_REQUESTS_EVENT_NAME)


        self.tpc_nodes_df[Column.BACKPRESSURE_COUNT] = self.tpc_nodes_df.apply(self.calculate_counter_value,
                                                                          axis=1,
                                                                          hw_event_df=high_back_pressure_events_df,
                                                                          do_count=True,
                                                                          debug=False)

        self.tpc_nodes_df[Column.PARTIALS_COUNT] = self.tpc_nodes_df.apply(self.calculate_counter_value,
                                                                      axis=1,
                                                                      hw_event_df=high_partial_writes_events_df,
                                                                      do_count=True)
        self.tpc_nodes_df[Column.MEM_WRITE_COUNT] = self.tpc_nodes_df.apply(self.calculate_counter_value,
                                                                      axis=1,
                                                                      hw_event_df=memwrite_events_df,
                                                                      do_count=False)


    def calculate_tpc_nodes_running_time(self):
        """
        Add information redaring running duration for each node in nodes df
        """
        hw_events_write_events_df = self.hw_events_df[(self.hw_events_df[Column.HW_EVENT_DESCRIPTION] == END_OF_WRITE_EVENT_DESCIPTION) |
                                                      (self.hw_events_df[Column.HW_EVENT_DESCRIPTION] == TPC_SPU_START_EVENT_DESCRIPTION) |
                                                      (self.hw_events_df[Column.HW_EVENT_DESCRIPTION] == TPC_SPU_START_TO_SPU_HALT_EVENT_DESCRIPTION)]

        # start = time.time()
        duration_columns = self.tpc_nodes_df.apply(self.calculate_duration_info,
                                                        axis=1,
                                                        hw_event_df=hw_events_write_events_df)

        self.tpc_nodes_df = pd.concat([self.tpc_nodes_df,duration_columns],axis=1)
        self.tpc_nodes_df[Column.DURATION_FULL] = (self.tpc_nodes_df[Column.END_OF_WRITE_MAX] - self.tpc_nodes_df[Column.REAL_START_TIME]).where(self.tpc_nodes_df[Column.IS_CACHE_WARM_UP],self.tpc_nodes_df["end_of_write_max"] - self.tpc_nodes_df[Column.START_TIME_OF_NODE])
        self.tpc_nodes_df[Column.DURATION_WITH_WRITE] = (self.tpc_nodes_df[Column.END_OF_WRITE_MAX] - self.tpc_nodes_df[Column.START_TIME_OF_NODE])
        self.tpc_nodes_df[Column.EXTRA_START_DURATION] = self.tpc_nodes_df[Column.START_TIME_OF_NODE] - self.tpc_nodes_df[Column.REAL_START_TIME]
        self.tpc_nodes_df[Column.EXTRA_END_DURATION] = self.tpc_nodes_df[Column.END_OF_WRITE_MAX] - self.tpc_nodes_df[Column.END_TIME_OF_NODE]
        self.tpc_nodes_df[Column.DURATION] = self.tpc_nodes_df[Column.END_TIME_OF_NODE] - self.tpc_nodes_df[Column.START_TIME_OF_NODE]

    def uodate_partial_writes_info(self):
        """
        update node df with all needed partial writes info
        """

        # start = time.time()
        self.calculate_tpc_nodes_partial_writes_detected()
        # end = time.time()
        # print(f"{self.model},{self.graph_index}:calculate_tpc_nodes_partial_writes_detected: {end-start}")
        # start = time.time()
        self.calculate_tpc_nodes_running_time()
        # end = time.time()
        # print(f"{self.model},{self.graph_index}:calculate_tpc_nodes_running_time: {end-start}")
        self.tpc_nodes_df[Column.HLTV] = self.hltv_path
        self.tpc_nodes_df[Column.HLTV_SIZE] = self.hltv_size
        self.tpc_nodes_df[Column.NODE_CSV_PATH] = self.nodes_csv_path
        self.tpc_nodes_df[Column.HW_EVENTS_CSV_PATH] = self.hw_events_csv_path
        self.tpc_nodes_df[Column.LOG_FILE_PATH] = self.log_file_path
        self.tpc_nodes_df[Column.TRACE_ANALYSED_PATH] = self.trace_analysed_path

        returned_df = self.tpc_nodes_df[TRACE_ANALYZER_OUTPUT_COLUMNS]
        if self.suffix:
            renamed_columns = [f"{col_name}__{self.suffix}" if col_name not in [Column.NODE_NAME,Column.OP_TYPE] else col_name for col_name in TRACE_ANALYZER_OUTPUT_COLUMNS ]
            returned_df.columns = renamed_columns
        return returned_df

    def analyse(self):
        df = self.uodate_partial_writes_info()
        if len(df) > 0:
            df.to_csv(self.trace_analysed_path, index=False)
        else:
            print(f"No output for:{self.nodes_csv_path}")

    def get_trace_analysed_df(self):
        return pd.read_csv(self.trace_analysed_path,usecols=lambda x: x != 0) if os.path.exists(self.trace_analysed_path) else None


if __name__ == "__main__":
    pass