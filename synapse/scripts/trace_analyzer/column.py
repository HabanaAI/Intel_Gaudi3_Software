class Column:
    # analyzed_nodes columns
    UNIT = 'Unit'
    UNIQUE_NODE_ID = 'Unique node id'
    NODE_NAME = 'Node Name'
    OP_TYPE = 'Op Type'
    START_TIME_OF_NODE = 'Start time of node'
    END_TIME_OF_NODE = 'End time of node'
    DURATION_US = 'Duration (us)'
    ACTIVE_US = 'Active (us)'
    MIN_SINGLE_ENGINE_US = 'Min single engine (us)'
    MAX_SINGLE_ENGINE_US = 'Max single engine (us)'
    DIFF_MAX_MIN_SINGLE_ENGINE_US = 'Diff max min single engine (us)'
    AVG_SINGLE_ENGINE_US = 'Avg single engine (us)'
    BUNDLE_INDEX = 'Bundle index'
    AVG_SINGLE_ENGINE_ACTIVE_US = 'Avg single engine active (us)'
    VARIANCE_SINGLE_ENGINE_US = 'Variance single engine (us²)'
    NUM_ACTIVATIONS = 'Num activations'
    PARALLEL_ENGINES = 'Parallel Engines'
    TPC_INPUT_BW_UTILIZATION_ = 'TPC Input BW Utilization (%)'
    TPC_OUTPUT_BW_UTILIZATION_ = 'TPC Output BW Utilization (%)'
    INPUT_VECTOR_UTILIZATION_RATIO = 'Input Vector Utilization (ratio)'
    ALL_INPUTS_IN_SRAM = 'All inputs in sram'
    INPUT_SHAPES = 'Input shapes'
    INPUT_DATATYPE = 'Input Datatype'
    INPUT_PLACEMENT = 'Input placement'
    OUTPUT_SHAPES = 'Output shapes'
    OUTPUT_DATATYPE = 'Output Datatype'
    OUTPUT_PLACEMENT = 'Output placement'
    UNEXPOSED_TPC_US = 'Unexposed tpc (us)'
    EXPOSED_TPC_US = 'Exposed tpc (us)'
    UNEXPOSED_MME_US = 'Unexposed mme (us)'
    EXPOSED_MME_US = 'Exposed mme (us)'
    EXPOSED_DMA_US = 'Exposed dma (us)'
    MME_COMPUTE_UTILIZATION_ = 'MME Compute Utilization (%)'
    MME_EXPECTED_VS_ACTUAL_ = 'MME expected vs actual (%)'
    MME_ORIGIN_NODE = 'MME Origin Node'
    MME_ORIGIN_EXPECTED_US = 'MME Origin Expected (us)'
    MME_NODE_STRATEGY = 'MME Node Strategy'
    UNSLICED_MME_DURATION_US = 'Unsliced MME duration (us)'
    BUNDLE_DURATION_US = 'Bundle duration (us)'
    COST_MODEL = 'Cost Model'
    INPUT_TENSORS = 'Input Tensors'
    OUTPUT_TENSORS = 'Output Tensors'
    INPUT_CACHE_ALLOCATION = 'Input cache allocation'
    OUTPUT_CACHE_ALLOCATION = 'Output cache allocation'
    UNSLICED_MME_ACTIVE_TIME_US = 'Unsliced MME Active Time (us)'
    TPC_DURATION_UNTIL_MSG_WRITE_US = 'TPC Duration until MSG_WRITE (us)'
    MME_COMPUTE_TIME_US = 'MME Compute time (us)'
    MME_READONLY_TIME_US = 'MME Read-only time (us)'
    MME_WRITEONLY_TIME_US = 'MME Write-only time (us)'
    OUTPUT_TENSOR_REDUCTION = 'Output Tensor Reduction'
    TPC_PARALLEL_EXECUTION_ = 'TPC Parallel Execution (%)'

    # HW columns
    NAME = 'Name'
    TRACE_SOURCE = 'Trace Source'
    ENGINE = 'Engine'
    EVENT_TYPE = 'Event type'
    TIMESTAMPUSEC = 'Timestamp[usec]'
    RECIPEHANDLE = 'recipeHandle'
    STREAMHANDLE = 'streamHandle'
    RECIPE_ID = 'Recipe ID'
    RECIPE_NAME = 'Recipe name'
    CONTEXT_ID = 'Context ID'
    DEVICE = 'Device'
    OPERATION = 'Operation'
    DATA_TYPE = 'Data Type'
    HW_EVENT_PORT = 'HW Event port'
    HW_EVENT_DESCRIPTION = 'HW Event description'
    SPMU_VALUE = 'SPMU value'
    BMON_MIN = 'BMON min'
    BMON_AVG = 'BMON avg'
    BMON_MAX = 'BMON max'
    BMON_MATCHES = 'BMON matches'
    BMON_CONFIGURED_CYCLES = 'BMON configured cycles'
    NIC_PACKETS_AMOUNT = 'NIC Packets amount'
    MISC = 'Misc'

    # compare columns
    PARALLEL_ENGINES__T = 'Parallel Engines__t'
    DURATION_WITH_WRITE__T = 'duration_with_write__t'
    BACKPRESSURE_COUNT__T = 'backpressure_count__t'
    PARTIALS_COUNT__T = 'partials_count__t'
    MEM_WRITE_COUNT__T = 'mem_write_count__t'
    IS_CACHE_WARM_UP__T = 'is_cache_warm_up__t'
    CACHE_WARM_UP_NAME__T = 'cache_warm_up_name__t'
    DURATION__T = 'duration__t'
    DURATION_FULL__T = 'duration_full__t'
    EXTRA_START_DURATION__T = 'extra_start_duration__t'
    EXTRA_END_DURATION__T = 'extra_end_duration__t'
    LOG__T = 'log__t'
    HLTV__T = 'hltv__t'
    HLTV_SIZE__T = 'hltv_size__t'
    NODE_CSV_PATH__T = 'node_csv_path__t'
    HW_EVENTS_CSV_PATH__T = 'hw_events_csv_path__t'
    LOG_FILE_PATH__T = 'log_file_path__t'
    TRACE_ANALYSED_PATH__T = 'trace_analysed_path__t'
    DURATION_MAX_ENGINE__T = "duration max engine__t"

    PARALLEL_ENGINES__F = 'Parallel Engines__f'
    DURATION_WITH_WRITE__F = 'duration_with_write__f'
    BACKPRESSURE_COUNT__F = 'backpressure_count__f'
    PARTIALS_COUNT__F = 'partials_count__f'
    MEM_WRITE_COUNT__F = 'mem_write_count__f'
    IS_CACHE_WARM_UP__F = 'is_cache_warm_up__f'
    CACHE_WARM_UP_NAME__F = 'cache_warm_up_name__f'
    DURATION__F = 'duration__f'
    DURATION_FULL__F = 'duration_full__f'
    EXTRA_START_DURATION__F = 'extra_start_duration__f'
    EXTRA_END_DURATION__F = 'extra_end_duration__f'
    LOG__F = 'log__f'
    HLTV__F = 'hltv__f'
    HLTV_SIZE__F = 'hltv_size__f'
    NODE_CSV_PATH__F = 'node_csv_path__f'
    HW_EVENTS_CSV_PATH__F = 'hw_events_csv_path__f'
    LOG_FILE_PATH__F = 'log_file_path__f'
    TRACE_ANALYSED_PATH__F = 'trace_analysed_path__f'
    DURATION_MAX_ENGINE__F = "duration max engine__f"

    PREV_SAMPLE_TIME = 'prev_sample_time'
    PREV_SAMPLE_VALUE = 'prev_sample_value'
    IS_CACHE_WARM_UP = 'is_cache_warm_up'
    CACHE_WARM_UP_NAME = 'cache_warm_up_name'
    REAL_START_TIME = 'real_start_time'
    BACKPRESSURE_COUNT = 'backpressure_count'
    PARTIALS_COUNT = 'partials_count'
    MEM_WRITE_COUNT = 'mem_write_count'
    LOG = 'log'
    DURATION_FULL = 'duration_full'
    DURATION_WITH_WRITE = 'duration_with_write'
    EXTRA_START_DURATION = 'extra_start_duration'
    EXTRA_END_DURATION = 'extra_end_duration'
    DURATION = 'duration'
    END_OF_WRITE_MAX = 'end_of_write_max'
    HLTV = 'hltv'
    HLTV_SIZE = 'hltv_size'
    NODE_CSV_PATH = 'node_csv_path'
    HW_EVENTS_CSV_PATH = 'hw_events_csv_path'
    LOG_FILE_PATH = 'log_file_path'
    TRACE_ANALYSED_PATH = 'trace_analysed_path'
    DURATION_TILL_END_WRITE_RATIO = 'duration till end write ratio'
    PARTIAL_ISSUE_EXISTS = 'partial issue exists'
    PARTIAL_ISSUE_HANDLED = 'partial issue handled'
    BACKPRESSURE_DIFF = 'backpressure diff'
    PARTIAL_DIFF = 'partial diff'
    MEMWRITE_DIFF = 'memwrite diff'
    GOOD_HANDLING = 'good_handling'
    DURATION_MAX_ENGINE = "duration max engine"
    DURATION_MAX_ENGINE_RATIO = "duration max engine ratio"

FINAL_COLUMNS_ORDER = [
    Column.NODE_NAME, Column.OP_TYPE, Column.DURATION_MAX_ENGINE__F,Column.DURATION_MAX_ENGINE__T,
    Column.DURATION_MAX_ENGINE_RATIO, Column.PARTIAL_ISSUE_EXISTS, Column.PARTIAL_ISSUE_HANDLED,
    Column.LOG__T, Column.BACKPRESSURE_COUNT__F, Column.BACKPRESSURE_COUNT__T, Column.PARTIALS_COUNT__F,
    Column.PARTIALS_COUNT__T, Column.PARALLEL_ENGINES__F, Column.PARALLEL_ENGINES__T, Column.MEM_WRITE_COUNT__F,
    Column.MEM_WRITE_COUNT__T, Column.IS_CACHE_WARM_UP__F, Column.IS_CACHE_WARM_UP__T, Column.CACHE_WARM_UP_NAME__F,
    Column.CACHE_WARM_UP_NAME__T, Column.DURATION_WITH_WRITE__F, Column.DURATION_WITH_WRITE__T,
    Column.DURATION_TILL_END_WRITE_RATIO,Column.DURATION__F, Column.DURATION__T, Column.DURATION_FULL__F,
    Column.DURATION_FULL__T, Column.EXTRA_START_DURATION__F, Column.EXTRA_START_DURATION__T, Column.EXTRA_END_DURATION__F,
    Column.EXTRA_END_DURATION__T, Column.LOG__F, Column.BACKPRESSURE_DIFF, Column.PARTIAL_DIFF,
    Column.MEMWRITE_DIFF, Column.HLTV__F, Column.HLTV__T, Column.HLTV_SIZE__F, Column.HLTV_SIZE__T,
    Column.GOOD_HANDLING, Column.NODE_CSV_PATH__F, Column.HW_EVENTS_CSV_PATH__F, Column.LOG_FILE_PATH__F,
    Column.TRACE_ANALYSED_PATH__F, Column.NODE_CSV_PATH__T, Column.HW_EVENTS_CSV_PATH__T,
    Column.LOG_FILE_PATH__T, Column.TRACE_ANALYSED_PATH__T
    ]

TRACE_ANALYZER_OUTPUT_COLUMNS = [
    Column.NODE_NAME,
    Column.OP_TYPE,
    Column.PARALLEL_ENGINES,
    Column.DURATION_WITH_WRITE,
    Column.BACKPRESSURE_COUNT,
    Column.PARTIALS_COUNT,
    Column.MEM_WRITE_COUNT,
    Column.IS_CACHE_WARM_UP,
    Column.CACHE_WARM_UP_NAME,
    Column.DURATION,
    Column.DURATION_FULL,
    Column.EXTRA_START_DURATION,
    Column.EXTRA_END_DURATION,
    Column.LOG,
    Column.HLTV,
    Column.HLTV_SIZE,
    Column.NODE_CSV_PATH,
    Column.HW_EVENTS_CSV_PATH,
    Column.LOG_FILE_PATH,
    Column.TRACE_ANALYSED_PATH,
    Column.DURATION_MAX_ENGINE
]