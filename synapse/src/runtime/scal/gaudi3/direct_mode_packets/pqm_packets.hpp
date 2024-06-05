#pragma once

#include "runtime/scal/common/infra/scal_types.hpp"

#include "gaudi3/gaudi3_pqm_packets.h"

#include <stdint.h>

#define MAX_VALID_FENCE_ID (3)
#define INVALID_FENCE_ID   (MAX_VALID_FENCE_ID + 1)

namespace pqm
{
    // At the moment, supporting a PDMA with exactly one WR-Completion
    struct LinPdma
    {
        static void build(uint8_t*           pktBuffer,
                          uint64_t           src,
                          uint64_t           dst,
                          uint32_t           size,
                          bool               bMemset,
                          PdmaDir            direction,
                          // When breaking a memcopy command, we will want to use a barrier
                          LinPdmaBarrierMode barrierMode,
                          uint64_t           barrierAddress,
                          uint32_t           barrierData,
                          // Fence mechanism enabled => configure fence to wait for a given fenceTarget
                          uint32_t           fenceDecVal,
                          uint32_t           fenceTarget,
                          // fenceId could be 0-3, while 4 means fence is disabled
                          uint32_t           fenceId);

        static uint64_t getSize(bool useBarrier);

        static uint64_t dump(const uint8_t* pktBuffer);
    };

    struct Fence
    {
        using pktType = pqm_packet_fence;

        static void build(void*    pktBuffer,
                          uint8_t  fenceId,
                          uint32_t fenceDecVal,
                          uint32_t fenceTarget);

        static constexpr uint64_t getSize() {return sizeof(pktType);}

        static uint64_t dump(const void* pktBuffer);
    };

    struct MsgLong
    {
        using pktType = pqm_packet_msg_long;

        static void build(void*    pktBuffer,
                          uint32_t val,
                          uint64_t address);

        static constexpr uint64_t getSize() {return sizeof(pktType);}

        static uint64_t dump(const void* pktBuffer);
    };

    struct MsgShort
    {
        using pktType = pqm_packet_msg_short;

        static void build(void*    pktBuffer,
                          uint32_t val,
                          uint8_t  baseIndex,
                          uint16_t elementOffset);

        static constexpr uint64_t getSize() {return sizeof(pktType);}

        static uint64_t dump(const void* pktBuffer);
    };

    struct Nop
    {
        using pktType = pqm_packet_nop;

        static void build(void* pktBuffer);

        static constexpr uint64_t getSize() {return sizeof(pktType);}

        static uint64_t dump(const void* pktBuffer);
    };

    struct ChWreg32
    {
        using pktType = pqm_packet_ch_wreg32;

        static void build(void* pktBuffer, uint32_t regOffset, uint32_t value);

        static constexpr uint64_t getSize() { return sizeof(pktType); }

        static uint64_t dump(const void* pktBuffer);
    };

    uint64_t dumpPqmPacket(const uint8_t* pktBuffer);
} // namespace pqm
