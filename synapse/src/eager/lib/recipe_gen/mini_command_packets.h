#pragma once

// std includes
#include <cstdint>

// The arc commands and qman packets do not change too much between Gaudi2 and Gaudi3.
// So instead of duplicating code we translate the chip specific commands
// structures into Eager internal structures, allowing for as much code
// sharing as possible.
// We also do not attempt to condense the structures using bit structus as the current
// sizes are still small and do not exceed a CPU word size.
// Those structures are not complete and only include the currently used\required fields
// For Eager chip agnostic code.
// struct fields explanations can be found in the underlying chip structures headers.

namespace eager_mode
{
enum class PacketId
{
    WREG_32,
    WREG_BULK,
    FENCE,
    NOP,
    CB_LIST,
    WREG_64_LONG,
    MAX_ID
};

enum class EngineArcCommandId
{
    LIST_SIZE,
    NOP,
    WD_FENCE_AND_EXE,
    SCHED_DMA,
    STATIC_DESC_V2,
    COUNT
};

namespace mini_qman_packets
{
struct Wreg32
{
    uint16_t regOffset;
};

struct WregBulk
{
    uint16_t size64;
    uint16_t regOffset;
};

struct Wreg64Long
{
    uint8_t  dwEnable;
    bool     rel;
    uint16_t dregOffset;
};

}  // namespace mini_qman_packets

namespace mini_ecb_packets
{
struct Nop
{
    uint32_t padding;
    bool     switchCq;
    bool     yield;
    uint8_t  dmaCompletion;
};

struct StaticDescV2
{
    static constexpr uint8_t CPU_ID_ALL = 0xFF;
    uint32_t                 addrOffset;
    uint16_t                 size;
    uint8_t                  cpuIndex;
    uint8_t                  addrIndex;
    bool                     yield;
};

struct ListSize
{
    uint32_t size;
    bool     topologyStart;
    bool     yield;
};

struct Fence
{
    bool    yield;
    uint8_t dmaCompletion;
    uint8_t wdCtxtId;
};

struct SchedDma
{
    uint32_t addrOffset;
    uint16_t size;
    uint16_t gcCtxtOffset;
    uint8_t  addrIndex;
    bool     yield;
};

}  // namespace mini_ecb_packets

}  // namespace eager_mode