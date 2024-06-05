#include "upper_cp_qman_info.hpp"
#include "synapse_runtime_logging.h"
#include "gaudi/gaudi_packets.h"

using namespace gaudi;

bool UpperCpQmanInfo::parseArbPoint()
{
    static const uint32_t packetSize = 2;
    if (!validatePacketSize(packetSize))
    {
        return false;
    }

    packet_arb_point* pPacket      = (packet_arb_point*)m_hostAddress;
    uint32_t*         pHostAddress = (uint32_t*)m_hostAddress;

    m_isArbRelease = pPacket->rls;

    LOG_GCP_VERBOSE("{}{}: Arb-Point packet (0x{:x} : [{:#x}]): Is-Release {} Priority {}",
                    common::UpperCpQmanInfo::getIndentation(),
                    getPacketIndexDesc(),
                    m_hostAddress,
                    fmt::join(&pHostAddress[0], &pHostAddress[2], " "),
                    m_isArbRelease,
                    pPacket->priority);

    parseControlBlock(true);

    updateNextBuffer(packetSize);

    return true;
}

bool UpperCpQmanInfo::parseCpDma()
{
    static const uint32_t packetSize = 4;
    if (!validatePacketSize(packetSize))
    {
        return false;
    }

    packet_cp_dma* pPacket      = (packet_cp_dma*)m_hostAddress;
    uint32_t*      pHostAddress = (uint32_t*)m_hostAddress;

    m_lowerCpBufferHandle = pPacket->src_addr;
    m_lowerCpBufferSize   = pPacket->tsize;

    LOG_GCP_VERBOSE("{}{}: CP-DMA packet (0x{:x} : 0x{:x} 0x{:x} 0x{:x} 0x{:x}): address 0x{:x} size 0x{:x}",
                    common::UpperCpQmanInfo::getIndentation(),
                    getPacketIndexDesc(),
                    m_hostAddress,
                    (uint64_t)(pHostAddress[0]),
                    (uint64_t)(pHostAddress[1]),
                    (uint64_t)(pHostAddress[2]),
                    (uint64_t)(pHostAddress[3]),
                    m_lowerCpBufferHandle,
                    m_lowerCpBufferSize);

    parseControlBlock(true);

    updateNextBuffer(packetSize);

    return true;
}