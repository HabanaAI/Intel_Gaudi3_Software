#include "command_buffer_packet_generator.hpp"

#include "runtime/qman/gaudi/command_buffer_packet_generator.hpp"

#include "generate_packet.hpp"
#include "synapse_runtime_logging.h"

using namespace generic;

#define VERIFY_IS_NULL_POINTER(pointer, name)                                                                          \
    if (pointer == nullptr)                                                                                            \
    {                                                                                                                  \
        LOG_WARN(SYN_STREAM, "{}: got null pointer for {} ", HLLOG_FUNC, name);                                        \
        return synFail;                                                                                                \
    }

CommandBufferPktGenerator::CommandBufferPktGenerator() {}

synStatus CommandBufferPktGenerator::_generatePacket(char*&                    pPacket,
                                                     uint64_t&                 bufferSize,
                                                     CommonPktGen::BasePacket& packetGen,
                                                     bool                      shouldIncrementPointer) const
{
    uint64_t packetSize = packetGen.getPacketSize();

    if (pPacket == nullptr)
    {
        bufferSize = packetSize;
        pPacket    = new char[bufferSize];
    }
    else if (packetSize > bufferSize)
    {
        LOG_ERR(SYN_STREAM, "{}: Invalid allocated buffer-size {} packet-size {}", HLLOG_FUNC, bufferSize, packetSize);
        return synFail;
    }

    char* pTmpPacket = pPacket;  // As generate packet increments the pointer
    packetGen.generatePacket(pPacket);

    if (!shouldIncrementPointer)
    {
        pPacket = pTmpPacket;
    }

    return synSuccess;
}

synStatus CommandBufferPktGenerator::generateArbitratorDisableConfigCommand(char*& pPackets, uint32_t qmanId) const
{
    return synUnsupported;
}

synStatus
CommandBufferPktGenerator::generateMasterArbitratorConfigCommand(char*&                           pPackets,
                                                                 generic::masterSlavesArbitration masterSlaveIds,
                                                                 bool                             isByPriority) const
{
    return synUnsupported;
}

synStatus CommandBufferPktGenerator::generateArbitrationCommand(
    char*&                        pPacket,
    uint64_t&                     packetSize,
    bool                          priorityRelease,
    generic::eArbitrationPriority priority /* = ARB_PRIORITY_NORMAL */) const
{
    return synUnsupported;
}

synStatus CommandBufferPktGenerator::generateSlaveArbitratorConfigCommand(char*&   pPackets,
                                                                          uint32_t slaveId,
                                                                          uint32_t uSlaveQmanId,
                                                                          uint32_t uMasterQmanId) const
{
    return synUnsupported;
}

CommandBufferPktGenerator* CommandBufferPktGenerator::getCommandBufferPktGenerator(synDeviceType deviceType)
{
    CommandBufferPktGenerator* pCommandBufferPktGenerator = nullptr;
    switch (deviceType)
    {
        case synDeviceGaudi:
        {
            pCommandBufferPktGenerator = gaudi::CommandBufferPktGenerator::getInstance();
            break;
        }
        default:
        {
            HB_ASSERT(false, "Illegal device type {}", deviceType);
        }
    }

    return pCommandBufferPktGenerator;
}
