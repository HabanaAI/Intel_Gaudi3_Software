#pragma once

#include "define_synapse_common.hpp"

#define IS_VALID_QUEUE_DIR(queueDir) (queueDir < (uint8_t)PdmaDirCtx::NUM)

#define IS_VALID_QUEUE_TYPE(queueType) (queueType < INTERNAL_STREAM_TYPE_NUM)

#define QUEUE_INDEX_MAX 8
#define IS_VALID_QUEUE_INDEX(queueIndex) (queueIndex < QUEUE_INDEX_MAX)

// The user is expected to validate queueDir, queueType and queueIndex fields before usage
#define GET_QUEUE_NAME(queueDir, queueType, queueIndex)                                                                \
    fmt::format("{} {} {}", internalStreamDirName[queueDir], internalStreamTypeName[queueType], queueIndex)

struct ContextEncoding
{
    static constexpr uint8_t DIR_OFFSET = 0;
    static constexpr uint8_t DIR_MASK   = 3;

    static constexpr uint8_t TYPE_OFFSET = 2;
    static constexpr uint8_t TYPE_MASK   = 7;

    static constexpr uint8_t STREAM_OFFSET = 5;
    static constexpr uint8_t STREAM_MASK   = 7;
};