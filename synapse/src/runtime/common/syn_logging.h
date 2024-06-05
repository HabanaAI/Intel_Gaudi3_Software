#ifndef _SYN_LOGGING_H_
#define _SYN_LOGGING_H_

#define SYNAPSE_LOG_SINK_FILE    "synapse_log.txt"
#define SYNAPSE_LOG_SINK_FILE_RT "synapse_runtime.log"

#define LOG_SIZE       10 * 1024 * 1024
#define LOG_SIZE_RT    100 * 1024 * 1024
#define LOG_AMOUNT     5
#define LOG_AMOUNT_RT  1
#define USER_LOG_LEVEL 2  // spdlog::level::info
#define ENFORCE_TRACE_MODE_LOGGING

#define GRAPH_COMPILER_SEPARATE_LOG_FILE      "graph_compiler.log"
#define GRAPH_COMPILER_PERF_SEPARATE_LOG_FILE "graph_compiler_perf.log"
#define EVENT_TRIGGER_SEPARATE_LOG_FILE       "event_triggered_logger.log"
#define SYNAPSE_CS_PARSER_SEPARATE_LOG_FILE   "command_submission_parser.log"

#define PERFORMANCE_MEASURMENTS_COLLECT_FILE "perf_measure.log"
#define PERFORMANCE_LOG_SIZE                 50 * 1000 * 1000
#define PERFORMANCE_LOG_AMOUNT               0

#define RECIPE_STATS_COLLECT_FILE "recipe_stats.log"
#define RECIPE_STATS_LOG_SIZE     50 * 1000 * 1000
#define RECIPE_STATS_LOG_AMOUNT   0

#include "log_manager.h"

#endif
