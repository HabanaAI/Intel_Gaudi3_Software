// This file is intended for INNER definitions between Habana's SW components

#pragma once

#include "compiler_types.h"
#include "synapse_common_types.h"
#include <cstdint>
#include <vector>

struct synDeviceLimitationInfo
{
    bool    fp32Limited;
};

enum DataType
{
    DATA_TYPE_INPUT,
    DATA_TYPE_OUTPUT,
    DATA_TYPE_INTERMEDIATE
};

// PdmaDirCtx enum should match the values in FW sched_arc_pdma_dir_t
enum class PdmaDirCtx : uint8_t
{
    UP,
    DOWN,
    DEV2DEV,
    NUM
};

inline constexpr std::string_view internalStreamDirName[(uint8_t)PdmaDirCtx::NUM] {"D2H", "H2D", "D2D"};

enum internalStreamType
{
    INTERNAL_STREAM_TYPE_DMA_UP,
    INTERNAL_STREAM_TYPE_DMA_UP_PROFILER,
    INTERNAL_STREAM_TYPE_DEV_TO_DEV,
    INTERNAL_STREAM_TYPE_DMA_DOWN_USER,
    INTERNAL_STREAM_TYPE_DMA_DOWN_SYNAPSE,
    INTERNAL_STREAM_TYPE_COMPUTE,
    INTERNAL_STREAM_TYPE_COLLECTIVE_NETWORK,
    INTERNAL_STREAM_TYPE_NUM
};

// Todo consolidate internalStreamTypeName with internalStreamTypeToString
inline constexpr std::string_view internalStreamTypeName[(uint8_t)INTERNAL_STREAM_TYPE_NUM] {
    "pdma_rx",
    "pdma_rx_profiler",
    "pdma_device2device",
    "pdma_tx",
    "pdma_tx_commands",
    "compute",
    "collective",
};

inline const char* internalStreamTypeToString(internalStreamType queueType)
{
    switch (queueType)
    {
        case INTERNAL_STREAM_TYPE_DMA_UP:
            return "INTERNAL_STREAM_TYPE_DMA_UP";
        case INTERNAL_STREAM_TYPE_DMA_UP_PROFILER:
            return "INTERNAL_STREAM_TYPE_DMA_UP_PROFILER";
        case INTERNAL_STREAM_TYPE_DEV_TO_DEV:
            return "INTERNAL_STREAM_TYPE_DEV_TO_DEV";
        case INTERNAL_STREAM_TYPE_DMA_DOWN_USER:
            return "INTERNAL_STREAM_TYPE_DMA_DOWN_USER";
        case INTERNAL_STREAM_TYPE_DMA_DOWN_SYNAPSE:
            return "INTERNAL_STREAM_TYPE_DMA_DOWN_SYNAPSE";
        case INTERNAL_STREAM_TYPE_COMPUTE:
            return "INTERNAL_STREAM_TYPE_COMPUTE";
        case INTERNAL_STREAM_TYPE_COLLECTIVE_NETWORK:
            return "INTERNAL_STREAM_TYPE_COLLECTIVE_NETWORK";

        default:
            return "Unknown queueType";
    }
}

#define INVALID_DMA_QUEUE_CB_INDEX (0xFFFF)

typedef enum internalDmaDir
{
    MEMCOPY_HOST_TO_DRAM,
    MEMCOPY_HOST_TO_SRAM,
    MEMCOPY_DRAM_TO_SRAM,
    MEMCOPY_SRAM_TO_DRAM,
    MEMCOPY_SRAM_TO_HOST,
    MEMCOPY_DRAM_TO_HOST,
    MEMCOPY_DRAM_TO_DRAM,
    MEMCOPY_SRAM_TO_SRAM,
    MEMCOPY_MAX_ENUM
} internalDmaDir;

inline internalDmaDir directionConversion(synDmaDir code)
{
    switch (code)
    {
        case HOST_TO_DRAM:
            return MEMCOPY_HOST_TO_DRAM;
        case DRAM_TO_HOST:
            return MEMCOPY_DRAM_TO_HOST;
        case DRAM_TO_DRAM:
            return MEMCOPY_DRAM_TO_DRAM;
        default:
            // just in case no code matches
            return MEMCOPY_MAX_ENUM;
    }
}

typedef enum EngArcBufferAddrBase
{
    PATCHING_ADDR_BASE,
    EXECUTE_ADDR_BASE,
    DYNAMIC_ADDR_BASE,
    NOP_KERNEL_ADDR_BASE = 7  // Agreeable value with the Firmware for indicating the NOP-Kernel address
} EngArcBufferAddrBase;

// We take the following numbers from the SCAL json config file (scal/configs/default.json)
// TODO - Should find a different location
static const unsigned GAUDI2_FIRST_AVAILABLE_SYNC_OBJECT_FOR_GC = 392;
static const unsigned GAUDI2_FIRST_AVAILABLE_MONITOR_FOR_GC     = 1266;
static const unsigned GAUDI2_NUM_SYNC_OBJECTS_FOR_GC            = 632;
static const unsigned GAUDI2_NUM_MONITORS_FOR_GC                = 270;

// List of reserved memory IDs
static const uint64_t MEMORY_ID_RESERVED_FOR_WORKSPACE      = 0;
static const uint64_t MEMORY_ID_RESERVED_FOR_PROGRAM_DATA   = 1;
static const uint64_t MEMORY_ID_RESERVED_FOR_PROGRAM        = 2;
static const uint64_t MEMORY_ID_RESERVED_FOR_ASSERT_ASYNC   = 3;
static const uint64_t MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR = 4;
constexpr auto        SKIP_PREDICATE                        = 31;
constexpr auto        RUN_PREDICATE                         = 0;

typedef std::vector<internalMemcopyParamEntry> internalMemcopyParams;
