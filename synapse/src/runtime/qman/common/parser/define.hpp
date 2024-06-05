#pragma once

#include "graph_compiler/sync/sync_types.h"

#include <stddef.h>
#include <limits>

// We use INFO level for the Parser-Print-Log, to be able to print all logs, in case of a failure
// Meaning, there might be a device reset which may lead to process kill,
// and we would like to be able to get the cause, even if we cannot get the full parsing
#define LOG_GCP_VERBOSE(msg, ...) LOG_INFO(SYN_CS_PARSER, msg, ##__VA_ARGS__)
//
#define LOG_GCP_WARNING(msg, ...) LOG_ERR(SYN_CS_PARSER, "WARN " msg, ##__VA_ARGS__)
#define LOG_GCP_FAILURE(msg, ...) LOG_CRITICAL(SYN_CS_PARSER, "ERR " msg, ##__VA_ARGS__)

// #define DONT_PARSE_CONTROL_BLOCK

namespace common
{
enum eParsingDefinitions
{
    PARSING_DEFINITION_REGULAR,
    PARSING_DEFINITION_ON_HOST_LOWER_CP,
    PARSING_DEFINITION_ON_HOST_UPPER_CP
};

// QMANs blocks (start)
//
enum eQmanType
{
    QMAN_TYPE_DMA,
    QMAN_TYPE_PDMA = QMAN_TYPE_DMA,
    QMAN_TYPE_MME,
    QMAN_TYPE_TPC,
    QMAN_TYPE_DDMA,
    QMAN_TYPE_ROT
};

enum eCpParsingState
{
    CP_PARSING_STATE_INVALID,
    CP_PARSING_STATE_FENCE_CLEAR,
    CP_PARSING_STATE_FENCE_SET,
    CP_PARSING_STATE_ARB_REQUEST,
    CP_PARSING_STATE_CP_DMAS,
    // Work-Completion is only allowed by ARB-Master, preceding ARB-Release and following CP-DMAs states
    CP_PARSING_STATE_WORK_COMPLETION,
    // ARB-Release is internaly handled during CP_PARSING_STATE_CP_DMAS state
    CP_PARSING_STATE_ARB_RELEASE,
    CP_PARSING_STATE_COMPLETED,
    // Lower-CP normal state
    CP_PARSING_STATE_BASIC_COMMANDS
};

static const uint16_t INVALID_MONITOR_ID  = std::numeric_limits<uint16_t>::max();
static const uint16_t INVALID_FENCE_INDEX = std::numeric_limits<uint16_t>::max();
}  // namespace common
