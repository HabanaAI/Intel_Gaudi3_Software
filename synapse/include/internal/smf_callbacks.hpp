#pragma once
#include <stdint.h>

struct smf_callbacks_t
{
    void (*notifyStrideUpdate)(uint32_t nodeIdx, const char* tensorName, uint8_t stride_idx, uint64_t stride_value);
    void (*notifyOffsetUpdate)(uint32_t nodeIdx, const char* tensorName, uint64_t offset);
};
