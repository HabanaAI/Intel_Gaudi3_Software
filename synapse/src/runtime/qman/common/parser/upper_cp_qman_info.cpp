#include "upper_cp_qman_info.hpp"

#include "define.hpp"

#include "synapse_runtime_logging.h"

#include <sstream>

using namespace common;

bool UpperCpQmanInfo::getLowerCpBufferHandleAndSize(uint64_t& handle, uint64_t& size)
{
    handle = m_lowerCpBufferHandle;
    size   = m_lowerCpBufferSize;

    return true;
}

bool UpperCpQmanInfo::checkFenceClearPacket(uint64_t expectedAddress, uint16_t expectedFenceValue) const
{
    if (m_currentPacketAddressField != expectedAddress)
    {
        LOG_GCP_FAILURE("Fence packet is wrongly defined (address 0x{:x} expected 0x{:x})",
                        m_currentPacketAddressField,
                        expectedAddress);

        return false;
    }

    if (m_currentPacketValueField != 1)
    {
        LOG_GCP_FAILURE("Fence packet is wrongly defined (value {} (expected {})",
                        m_currentPacketAddressField,
                        expectedFenceValue);

        return false;
    }

    return true;
}

std::string UpperCpQmanInfo::getPacketIndexDesc() const
{
    return std::to_string(m_packetIndex);
}

bool UpperCpQmanInfo::isArbRelease() const
{
    return m_isArbRelease;
}