#include "stream_master_helper.hpp"

#include "command_buffer.hpp"
#include "data_chunk/data_chunk.hpp"
#include "defenders.h"
#include "synapse_runtime_logging.h"
#include "utils.h"

#include "runtime/qman/gaudi/command_buffer_packet_generator.hpp"
#include "runtime/common/recipe/device_agnostic_recipe_info.hpp"

StreamMasterHelper::StreamMasterHelper(synDeviceType deviceType, uint32_t physicalQueueOffset)
: m_deviceType(deviceType), m_physicalQueueOffset(physicalQueueOffset)
{
    switch (m_deviceType)
    {
        case synDeviceGaudi:
        {
            m_packetGenerator = gaudi::CommandBufferPktGenerator::getInstance();
            break;
        }
        default:
        {
            HB_ASSERT(false, "Illegal device type {}", m_deviceType);
        }
    }
}

StreamMasterHelper::~StreamMasterHelper()
{
    if (m_pCommandBuffer != nullptr)
    {
        synStatus status = CommandBufferMap::GetInstance()->RemoveCommandBuffer(m_pCommandBuffer);
        if (status != synSuccess)
        {
            LOG_CRITICAL(SYN_STREAM, "{}: Failed to remove Stream-Master command buffer", HLLOG_FUNC);
        }

        m_pCommandBuffer = nullptr;
    }
    if (m_pFenceClearCommandBuffer != nullptr)
    {
        synStatus status = CommandBufferMap::GetInstance()->RemoveCommandBuffer(m_pFenceClearCommandBuffer);
        if (status != synSuccess)
        {
            LOG_CRITICAL(SYN_STREAM, "{}: Failed to remove clear fence Stream-Master command buffer", HLLOG_FUNC);
        }

        m_pFenceClearCommandBuffer = nullptr;
    }
    if (m_pFenceCommandBuffer != nullptr)
    {
        synStatus status = CommandBufferMap::GetInstance()->RemoveCommandBuffer(m_pFenceCommandBuffer);
        if (status != synSuccess)
        {
            LOG_CRITICAL(SYN_STREAM, "{}: Failed to remove fence Stream-Master command buffer", HLLOG_FUNC);
        }

        m_pFenceCommandBuffer = nullptr;
    }
}

bool StreamMasterHelper::createStreamMasterJobBuffer(uint64_t arbMasterBaseQmanId)
{
    bool                 operStatus           = true;
    synStatus            status               = synSuccess;
    static const uint8_t fenceClearPacketSize = m_packetGenerator->getFenceClearPacketCommandSize();
    static const uint8_t fenceSetPacketSize   = m_packetGenerator->getFenceSetPacketCommandSize();
    uint64_t             arbMasterStreamId    = arbMasterBaseQmanId + m_physicalQueueOffset;

    // Add commands buffer:
    // TODO use map buffer and unmap buffer later
    char* pFenceClearBuffer = nullptr;
    char* pFenceBuffer      = nullptr;
    if (GCFG_ENABLE_STAGED_SUBMISSION.value())
    {
        status = CommandBufferMap::GetInstance()->AddCommandBufferUpdateOccupancyAndMap(fenceClearPacketSize,
                                                                                        &m_pFenceClearCommandBuffer,
                                                                                        pFenceClearBuffer);
        if (status != synSuccess)
        {
            LOG_ERR(SYN_STREAM, "{}: Failed to create fence clear command buffer", HLLOG_FUNC);
            return false;
        }

        status = CommandBufferMap::GetInstance()->AddCommandBufferUpdateOccupancyAndMap(fenceSetPacketSize,
                                                                                        &m_pFenceCommandBuffer,
                                                                                        pFenceBuffer);
        if (status != synSuccess)
        {
            // Todo return allocated resources
            LOG_ERR(SYN_STREAM, "{}: Failed to create fence command buffer", HLLOG_FUNC);
            return false;
        }
    }
    else
    {
        status = CommandBufferMap::GetInstance()->AddCommandBufferUpdateOccupancyAndMap(fenceClearPacketSize +
                                                                                            fenceSetPacketSize,
                                                                                        &m_pCommandBuffer,
                                                                                        pFenceClearBuffer);
        if (status != synSuccess)
        {
            LOG_ERR(SYN_STREAM, "{}: Failed to create command buffer", HLLOG_FUNC);
            return false;
        }
        pFenceBuffer = pFenceClearBuffer + fenceClearPacketSize;
    }

    do
    {
        // FENCE-Clear to arbitration-master stream
        uint64_t tmpFenceClearPacketSize = fenceClearPacketSize;  // sadly required...
        status =
            m_packetGenerator->generateFenceClearCommand(pFenceClearBuffer, tmpFenceClearPacketSize, arbMasterStreamId);
        if (status != synSuccess)
        {
            LOG_ERR(SYN_STREAM, "{}: Failed to generate Fence-Clear", HLLOG_FUNC);
            operStatus = false;
            break;
        }
        // FENCE-Raise on current stream (stream-master)
        uint64_t tmpFenceSetPacketSize = fenceSetPacketSize;  // sadly required...
        status                         = m_packetGenerator->generateFenceCommand(pFenceBuffer, tmpFenceSetPacketSize);
        if (status != synSuccess)
        {
            LOG_ERR(SYN_STREAM, "{}: Failed to generate Fence-Raise", HLLOG_FUNC);
            operStatus = false;
            break;
        }
    } while (false);

    if (!operStatus)
    {
        // clear according to GCFG_ENABLE_STAGED_SUBMISSION
        if (m_pCommandBuffer != nullptr)
        {
            status = CommandBufferMap::GetInstance()->RemoveCommandBuffer(m_pCommandBuffer);
            if (status != synSuccess)
            {
                LOG_CRITICAL(SYN_STREAM, "{}: Failed to remove command buffer", HLLOG_FUNC);
            }
            else
            {
                m_pCommandBuffer = nullptr;
            }
        }
        if (m_pFenceClearCommandBuffer != nullptr)
        {
            status = CommandBufferMap::GetInstance()->RemoveCommandBuffer(m_pFenceClearCommandBuffer);
            if (status != synSuccess)
            {
                LOG_CRITICAL(SYN_STREAM, "{}: Failed to remove command buffer", HLLOG_FUNC);
            }
            else
            {
                m_pFenceClearCommandBuffer = nullptr;
            }
        }
        if (m_pFenceCommandBuffer != nullptr)
        {
            status = CommandBufferMap::GetInstance()->RemoveCommandBuffer(m_pFenceCommandBuffer);
            if (status != synSuccess)
            {
                LOG_CRITICAL(SYN_STREAM, "{}: Failed to remove command buffer", HLLOG_FUNC);
            }
            else
            {
                m_pFenceCommandBuffer = nullptr;
            }
        }
    }

    return operStatus;
}

uint32_t StreamMasterHelper::getStreamMasterBufferSize() const
{
    return m_pCommandBuffer->GetOccupiedSize();
}

uint64_t StreamMasterHelper::getStreamMasterBufferHandle() const
{
    return m_pCommandBuffer->GetCbHandle();
}

uint64_t StreamMasterHelper::getStreamMasterBufferHostAddress() const
{
    return (uint64_t)m_pCommandBuffer->GetBufferMap();
}

uint32_t StreamMasterHelper::getStreamMasterFenceBufferSize() const
{
    return m_pFenceCommandBuffer->GetOccupiedSize();
}

uint64_t StreamMasterHelper::getStreamMasterFenceBufferHandle() const
{
    return m_pFenceCommandBuffer->GetCbHandle();
}

uint64_t StreamMasterHelper::getStreamMasterFenceHostAddress() const
{
    return (uint64_t)m_pFenceCommandBuffer->GetBufferMap();
}

uint32_t StreamMasterHelper::getStreamMasterFenceClearBufferSize() const
{
    return m_pFenceClearCommandBuffer->GetOccupiedSize();
}

uint64_t StreamMasterHelper::getStreamMasterFenceClearBufferHandle() const
{
    return m_pFenceClearCommandBuffer->GetCbHandle();
}

uint64_t StreamMasterHelper::getStreamMasterFenceClearHostAddress() const
{
    return (uint64_t)m_pFenceClearCommandBuffer->GetBufferMap();
}
