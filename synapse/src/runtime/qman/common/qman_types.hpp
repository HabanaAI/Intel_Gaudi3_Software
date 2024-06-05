#pragma once

#include <stdint.h>
#include <limits>
#include <vector>
#include <deque>

struct MemoryInfo
{
    uint64_t m_memSize;
    uint64_t m_memAddr;
};

typedef std::vector<MemoryInfo> MemoryParamsVector;

struct DeviceRecipeMemory
{
    MemoryInfo         m_workspace;
    MemoryParamsVector m_tensors;
};

enum TrainingQueue
{
    TRAINING_QUEUE_DMA_UP,
    TRAINING_QUEUE_DMA_DOWN_USER,
    TRAINING_QUEUE_COMPUTE_0,
    TRAINING_QUEUE_COMPUTE_1,
    TRAINING_QUEUE_COLLECTIVE_0,
    TRAINING_QUEUE_COLLECTIVE_1,
    TRAINING_QUEUE_COLLECTIVE_2,
    TRAINING_QUEUE_DMA_DOWN_SYNAPSE,
    TRAINING_QUEUE_DEV_TO_DEV_SYNAPSE,
    TRAINING_QUEUE_NUM
};

enum TrainingRetCode
{
    TRAINING_RET_CODE_SUCCESS,
    TRAINING_RET_CODE_INVALID_REQUEST,
    TRAINING_RET_CODE_NUM_OF_SIGNALS_LIMIT,
    TRAINING_RET_CODE_NO_CHANGE,
    TRAINING_RET_CODE_FULLY_USED,
    TRAINING_RET_CODE_MAPPED_BUFFER_FULLY_USED,
    TRAINING_RET_CODE_CB_INITIALIZATION_FAILURE,
    TRAINING_RET_CODE_FAIL,
    TRAINING_RET_CODE_AFTER_SO_RESET
};

typedef uint32_t PhysicalQueuesId;

static const uint64_t INVALID_HANDLE_VALUE = std::numeric_limits<uint64_t>::max();

static const uint64_t INVALID_DEVICE_ADDR = std::numeric_limits<uint64_t>::max();

static const uint64_t MANDATORY_KERNEL_BITS_ALIGNMENT = 13;  // 13-bits alignment is mandatory for HW functionality
static const uint64_t MANDATORY_KERNEL_ALIGNMENT      = (1 << MANDATORY_KERNEL_BITS_ALIGNMENT);

// A zero workspace address indicates that there is no need to patch the ORIGINAL recipe
static const uint64_t INITIAL_WORKSPACE_ADDRESS = 0;

class DataChunk;
typedef std::deque<DataChunk*> DataChunksDB;

#define SYNAPSE_WAIT_FOR_CS_DEFAULT_TIMEOUT 1000000  // 1 second in us
#define SYNAPSE_WAIT_FOR_QUERY_TIMEOUT      1        // 1 us
