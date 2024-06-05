#pragma once

#include <cstdint>
#include "struct_fw_packets.hpp"

template<class Tfw>
struct EngEcbNopPkt
{
    using pktType = typename Tfw::eng_arc_cmd_nop_t;

    static void build(void* pktBuffer, bool yield, uint32_t dma_completion, uint32_t switch_cq, uint32_t padding);

    static constexpr uint64_t getSize() { return sizeof(pktType); };
};

template<class Tfw>
struct EngEcbSizePkt
{
    using pktType = typename Tfw::eng_arc_cmd_list_size_t;

    static void build(void* pktBuffer, bool yield, uint32_t dma_completion, bool topologyStart, uint32_t list_size);

    static constexpr uint64_t getSize() { return sizeof(pktType); };
};

template<class Tfw>
struct EngStaticDescPkt
{
    using pktType = typename Tfw::eng_arc_cmd_static_desc_v2_t;

    static void build(void* pktBuffer, bool yield, uint32_t cpu_index, uint32_t size, uint32_t addr_offset, uint32_t addr_index);

    static constexpr uint64_t getSize() { return sizeof(pktType); };
};
