#include <cstring>
#include <cstdint>

#include "scal_internal/pkt_macros.hpp"

/********************** EngEcbNopPkt ***********************/
template<class Tfw>
void EngEcbNopPkt<Tfw>::build(void* pktBuffer,
                           bool     yield,
                           uint32_t dma_completion,
                           uint32_t switch_cq,
                           uint32_t padding)
{
    pktType& pkt = *(reinterpret_cast<pktType*>(pktBuffer));
    memset(&pkt, 0, sizeof(pktType));

    pkt.cmd_type       = Tfw::ECB_CMD_NOP;
    pkt.yield          = yield;
    pkt.dma_completion = dma_completion;
    pkt.switch_cq      = switch_cq;
    pkt.padding        = padding;
}

template struct EngEcbNopPkt<G2Packets>;
template struct EngEcbNopPkt<G3Packets>;

/********************** EngEcbSizePkt ***********************/
template<class Tfw>
void EngEcbSizePkt<Tfw>::build(void* pktBuffer,
                            bool     yield,
                            uint32_t dma_completion,
                            bool     topologyStart,
                            uint32_t list_size)
{
    pktType& pkt = *(reinterpret_cast<pktType*>(pktBuffer));
    memset(&pkt, 0, sizeof(pktType));

    pkt.cmd_type = Tfw::ECB_CMD_LIST_SIZE;
    // Yield ARC control to the other list (s/d) after execution
    pkt.yield = (uint32_t)yield;
    // start of new topology, fw can reset prev_sob_id etc. when this flag is set to 1
    pkt.topology_start = topologyStart;
    // Total size of list in bytes; for FW management of double buffer
    // The size includes the size of this command as well.
    pkt.list_size = list_size;
}

template struct EngEcbSizePkt<G2Packets>;
template struct EngEcbSizePkt<G3Packets>;


/********************** EngStaticDescPkt ***********************/
template<class Tfw>
void EngStaticDescPkt<Tfw>::build(void* pktBuffer, bool yield, uint32_t cpu_index,
                                  uint32_t size, uint32_t addr_offset, uint32_t addr_index)
{
    pktType& pkt = *(reinterpret_cast<pktType*>(pktBuffer));
    memset(&pkt, 0, sizeof(pktType));

    pkt.cmd_type = Tfw::ECB_CMD_STATIC_DESC_V2;
    // Yield ARC control to the other list (s/d) after execution
    pkt.yield = yield;
    // ARC CPU ID as defined in the common header  CPU_ID_xxx_QMAN_ARCx in specs/gaudi2/arc/gaudi2_arc_common_packets.h
    // cpu_index = CPU_ID_ALL command is processed by all engine ARCs
    // cpu_index = CPU_ID_INVALID command is ignored by all engine ARCs
    pkt.cpu_index = cpu_index;
    // transfer size in bytes  (13 bits extended to 21 bits)
    pkt.size = size;
    // Recipe base address register index to be used to generate target address of 64 bits
    pkt.addr_index = addr_index;
    // 32bit address offset
    pkt.addr_offset = addr_offset;
}

template struct EngStaticDescPkt<G2Packets>;
template struct EngStaticDescPkt<G3Packets>;
