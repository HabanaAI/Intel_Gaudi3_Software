#pragma once
#include <cstdint>
#include <hl_gcfg/hlgcfg_item.hpp>

#include "infra/habana_global_conf_common.h"

// Common:
extern GlobalConfUint64    GCFG_STAGED_SUBMISSION_NODES_PER_STAGE; // todo: can be uint32
extern GlobalConfFloat     GCFG_STAGED_SUBMISSION_NODES_PER_STAGE_INCREMENT_FACTOR;
extern GlobalConfUint64    GCFG_DSD_STAGED_SUBMISSION_NODES_PER_STAGE; // todo: can be uint32
extern GlobalConfFloat     GCFG_DSD_STAGED_SUBMISSION_NODES_PER_STAGE_INCREMENT_FACTOR;
extern GlobalConfBool      GCFG_STAGED_SUBMISSION_NODE_EXE_VALIDATION;
extern GlobalConfBool      GCFG_RUNTIME_DUMP_RECIPE;
extern GlobalConfBool      GCFG_RUNTIME_SKIP_RECIPE_VALIDATION;
extern GlobalConfBool      GCFG_INIT_HCCL_ON_ACQUIRE;
extern GlobalConfBool      GCFG_LOG_LAUNCH_INFO_UPON_FAILURE;
extern GlobalConfUint64    GCFG_TERMINATE_SYNAPSE_UPON_DFA;
extern GlobalConfUint64    GCFG_DFA_READ_REG_MODE;
extern GlobalConfBool      GCFG_DFA_COLLECT_CCB;
extern GlobalConfBool      GCFG_DBG_ENFORCE_NEW_SCRATCHPAD_SECTION_ADDRESS;
extern GlobalConfUint64    GCFG_DBG_ENFORCE_NUM_OF_NEW_SECTIONS_GROUP_ADDRESSES;
extern GlobalConfBool      GCFG_ENABLE_MAPPING_IN_STREAM_COPY;
extern GlobalConfUint64    GCFG_MAX_WAIT_TIME_FOR_MAPPING_IN_STREAM_COPY;
extern GlobalConfUint64    GCFG_POOL_MAPPING_SIZE_IN_STREAM_COPY;
extern GlobalConfBool      GCFG_ENABLE_POOL_MAPPING_WAIT_IN_STREAM_COPY;
extern GlobalConfBool      GCFG_DFA_ON_SIGNAL;
extern GlobalConfUint64    GCFG_HOST_CYCLIC_BUFFER_SIZE;
extern GlobalConfUint64    GCFG_HOST_CYCLIC_BUFFER_CHUNKS_AMOUNT;
extern GlobalConfBool      GCFG_ENABLE_SAMPLING_HOST_CYCLIC_BUFFER_WATERMARK;
extern GlobalConfBool      GCFG_DFA_SAVE_RECIPE;

// Gaudi:
extern GlobalConfUint64    GCFG_RECIPE_CACHE_SIZE;
extern GlobalConfUint64    GCFG_RECIPE_CACHE_BLOCK_SIZE;
extern GlobalConfUint64    GCFG_STREAM_COMPUTE_REDUCE_FACTOR;
extern GlobalConfUint64    GCFG_STREAM_COMPUTE_DATACHUNK_SINGLE_CHUNK_SIZE_UPPER_CP;
extern GlobalConfUint64    GCFG_STREAM_COMPUTE_DATACHUNK_CACHE_AMOUNT_UPPER_CP;
extern GlobalConfUint64    GCFG_STREAM_COMPUTE_DATACHUNK_SINGLE_CHUNK_SIZE_LOWER_CP;
extern GlobalConfUint64    GCFG_STREAM_COMPUTE_DATACHUNK_CACHE_AMOUNT_LOWER_CP;
extern GlobalConfUint64    GCFG_STREAM_COMPUTE_ARC_DATACHUNK_SINGLE_CHUNK_SIZE_LOWER_CP;
extern GlobalConfUint64    GCFG_STREAM_COMPUTE_ARC_DATACHUNK_CACHE_AMOUNT_LOWER_CP;
extern GlobalConfUint64    GCFG_STREAM_COPY_UP_DATACHUNK_CACHE_AMOUNT_POOL_SMALL;
extern GlobalConfUint64    GCFG_STREAM_COPY_UP_DATACHUNK_CACHE_AMOUNT_POOL_MEDIUM;
extern GlobalConfUint64    GCFG_STREAM_COPY_UP_DATACHUNK_CACHE_AMOUNT_POOL_LARGE;
extern GlobalConfUint64    GCFG_STREAM_COPY_DOWN_SYNAPSE_DATACHUNK_CACHE_AMOUNT_POOL_SMALL;
extern GlobalConfUint64    GCFG_STREAM_COPY_DOWN_SYNAPSE_DATACHUNK_CACHE_AMOUNT_POOL_MEDIUM;
extern GlobalConfUint64    GCFG_STREAM_COPY_DOWN_DATACHUNK_CACHE_AMOUNT_POOL_SMALL;
extern GlobalConfUint64    GCFG_STREAM_COPY_DOWN_DATACHUNK_CACHE_AMOUNT_POOL_MEDIUM;
extern GlobalConfUint64    GCFG_STREAM_COPY_DOWN_DATACHUNK_CACHE_AMOUNT_POOL_LARGE;
extern GlobalConfUint64    GCFG_DATACHUNK_LOOP_NUM_RETRIES;
extern GlobalConfUint64    GCFG_NUM_OF_CSDC_TO_CHK;
extern GlobalConfUint64    GCFG_WCM_REPORT_AMOUNT;
extern GlobalConfUint64    GCFG_WCM_QUERIER_REPORT_AMOUNT;
extern GlobalConfBool      GCFG_CHECK_SECTION_OVERLAP;
extern GlobalConfString    GCFG_SCAL_CONFIG_FILE_PATH;
extern GlobalConfUint64    GCFG_NUM_OF_DCS_PER_CANARY_PROTECTION_HEADER;
extern GlobalConfBool      GCFG_PARSE_EACH_COMPUTE_CS;
extern GlobalConfBool      GCFG_ENABLE_WIDE_BUCKET;
extern GlobalConfBool      GCFG_DISABLE_SYNAPSE_HUGE_PAGES;
extern GlobalConfUint64    GCFG_NUM_OF_USER_STREAM_EVENTS;

// Gaudi 2:
extern GlobalConfUint64    GCFG_SCAL_RECIPE_LAUNCHER_DEBUG_MODE;