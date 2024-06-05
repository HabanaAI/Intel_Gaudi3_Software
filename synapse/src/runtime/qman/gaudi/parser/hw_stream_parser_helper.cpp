#include "hw_stream_parser_helper.hpp"

#include "qman_cp_info.hpp"

#include "defenders.h"
#include "synapse_runtime_logging.h"

#include "gaudi/gaudi_packets.h"

#include "drm/habanalabs_accel.h"

using namespace gaudi;

uint32_t HwStreamParserHelper::getQmansIndex()
{
    if (m_hwQueueId < GAUDI_QUEUE_ID_CPU_PQ)
    {
        return (m_hwQueueId + m_queuePhysicalOffset) / UPPER_CPS_PER_QMAN;
    }

    // else
    return (m_hwQueueId + m_queuePhysicalOffset - 1) / UPPER_CPS_PER_QMAN;
}

// CPs are arranged upon QMANs' order
// Each has (UPPER_CPS_PER_QMAN + 1) CP instances

// Retrieves CP-Index for the upper CP.
uint32_t HwStreamParserHelper::getUpperCpIndex()
{
    return getQmansIndex() * CPS_PER_QMAN + m_queuePhysicalOffset;
}

uint32_t HwStreamParserHelper::getLowerCpIndex()
{
    return getQmansIndex() * CPS_PER_QMAN + UPPER_CPS_PER_QMAN;
}

bool HwStreamParserHelper::isCurrentUpperCpPacketCpDma() const
{
    return (m_upperCpInfo.getCurrentPacketId() == PACKET_CP_DMA);
}

bool HwStreamParserHelper::isCurrentUpperCpPacketArbPoint() const
{
    return (m_upperCpInfo.getCurrentPacketId() == PACKET_ARB_POINT);
}

uint64_t HwStreamParserHelper::getExpectedPacketForFenceClearState()
{
    return PACKET_MSG_LONG;
}

uint64_t HwStreamParserHelper::getExpectedPacketForFenceSetState()
{
    return PACKET_FENCE;
}

uint64_t HwStreamParserHelper::getExpectedPacketForArbRequestState()
{
    return PACKET_ARB_POINT;
}
