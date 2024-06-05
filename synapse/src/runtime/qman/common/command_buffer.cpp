/*****************************************************************************
 * Copyright (C) 2016 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 * Authors:
 * Tzachi Cohen <tcohen@gaudilabs.com>
 * Oded Gabbay <ogabbay@gaudilabs.com>
 ******************************************************************************
 */

// #include "synapse_common_types.h"
#ifndef _WIN32
#include <cstring>

#include "command_buffer.hpp"

#include "event_triggered_logger.hpp"
#include "runtime/common/osal/osal.hpp"
#include "syn_singleton.hpp"

#include "synapse_runtime_logging.h"

#include "drm/habanalabs_accel.h"

using namespace std;

const uint64_t CommandBuffer::m_cpFetchSize = 8;

// This is fine, as this single instance MUST be created at the SINGLE process that exists in our setup.
// Otherwise, needs to add a mutex_singleton and create the class on the GetInstance method (using that mutex).
CommandBufferMap* CommandBufferMap::m_pInstance = new CommandBufferMap();

synStatus CommandBufferMap::AddCommandBufferUpdateOccupancyAndMap(unsigned        commandBufferSize,
                                                                  CommandBuffer** ppCommandBuffer,
                                                                  char*&          pCommandBufferData,
                                                                  bool            isForceMmuMapped)
{
    synStatus status = synSuccess;
    status           = AddCommandBuffer(commandBufferSize, ppCommandBuffer, isForceMmuMapped);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_CS, "{}: Failed to create command buffer", HLLOG_FUNC);
        return status;
    }
    CommandBuffer* pCommandBuffer = *ppCommandBuffer;
    status                        = pCommandBuffer->UpdateOccupiedSize(commandBufferSize);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_CS, "{}: Failed to update occupied cb-mapped buffer-size", HLLOG_FUNC);
        if (CommandBufferMap::GetInstance()->RemoveCommandBuffer(pCommandBuffer) != synSuccess)
        {
            LOG_CRITICAL(SYN_CS, "{}: Failed to remove command buffer", HLLOG_FUNC);
            return status;
        }
        ppCommandBuffer = nullptr;
    }
    status = pCommandBuffer->MapBuffer();
    if (status != synSuccess)
    {
        LOG_ERR(SYN_CS, "{}: Failed to map buffer", HLLOG_FUNC);
        if (CommandBufferMap::GetInstance()->RemoveCommandBuffer(pCommandBuffer) != synSuccess)
        {
            LOG_CRITICAL(SYN_CS, "{}: Failed to remove command buffer", HLLOG_FUNC);
            return status;
        }
        ppCommandBuffer = nullptr;
    }

    pCommandBufferData = (char*)pCommandBuffer->GetBufferMap();
    return status;
}

synStatus CommandBufferMap::AddCommandBuffer(unsigned        commandBufferSize,
                                             CommandBuffer** ppCommandBuffer,
                                             bool            isForceMmuMapped)
{
    CommandBuffer* pCommandBuffer = nullptr;

    if (ppCommandBuffer == nullptr)
    {
        LOG_ERR(SYN_CS, "Failed due to null pointer (ppCommandBuffer)");
        return synInvalidArgument;
    }

    synDeviceInfo devInfo {};
    OSAL::getInstance().GetDeviceInfo(devInfo);
    // Mmu mapped command buffers are for internal queues
    // while CB mapped are for external queues (External Queue = the driver can detect completion )
    if ((devInfo.deviceType == synDeviceGaudi2) || (isForceMmuMapped))
    {
        pCommandBuffer = new MmuMappedCommandBuffer;
    }
    else
    {
        pCommandBuffer = new ExternalCommandBuffer;
    }

    if (pCommandBuffer == nullptr)
    {
        LOG_ERR(SYN_CS, "Failed to allocate command buffer on host");
        return synOutOfHostMemory;
    }

    synStatus status = pCommandBuffer->InitializeCommandBuffer(commandBufferSize);
    if (status == synSuccess)
    {

        status = pCommandBuffer->MapBuffer();
        if (status != synSuccess)
        {
            LOG_ERR(SYN_CS, "Failure on command buffer mapping");

            if (pCommandBuffer->DestroyCommandBuffer() != synSuccess)
            {
                LOG_ERR(SYN_CS, "Failed to destroy command buffer, upon create failure");
            }

            delete pCommandBuffer;
            return synFail;
        }
    }
    else
    {
        LOG_ERR(SYN_CS, "Failed to initialize command buffer");

        delete pCommandBuffer;
        return status;
    }

    {
        std::unique_lock<std::mutex> mlock(m_commandBufferDBmutex);

        if (m_commandBufferDB.insert(std::make_pair(pCommandBuffer, pCommandBuffer)).second)
        {
            LOG_TRACE(SYN_CS,
                      "{}: Command buffer successfully created and mapped. Key 0x{:x} added",
                      HLLOG_FUNC,
                      (uint64_t)pCommandBuffer);

            *ppCommandBuffer = pCommandBuffer;

            return synSuccess;
        }
        else
        {
            LOG_WARN(SYN_CS, "{}: Key 0x{:x} already found in map", HLLOG_FUNC, (uint64_t)pCommandBuffer);
            return synFail;
        }
    }
}

synStatus CommandBufferMap::RemoveCommandBuffer(CommandBuffer* pCommandBuffer)
{
    HB_ASSERT_PTR(pCommandBuffer);
    LOG_TRACE(SYN_CS, "{}: Key {}", HLLOG_FUNC, (void*)pCommandBuffer);

    std::unique_lock<std::mutex> lock(m_commandBufferDBmutex);

    auto itr = m_commandBufferDB.find(pCommandBuffer);

    if (itr == m_commandBufferDB.end())
    {
        return synFail;
    }

    synStatus status = _destroyCommandBuffer(itr->second);
    if (status != synSuccess)
    {
        return status;
    }

    unsigned mapSize = m_commandBufferDB.size();
    _remove(itr);

    if (m_commandBufferDB.size() == mapSize)
    {
        return synFail;
    }
    else
    {
        return synSuccess;
    }
}

synStatus CommandBufferMap::Clear(uint32_t& numOfElemCleared)
{
    std::unique_lock<std::mutex> lock(m_commandBufferDBmutex);
    uint32_t                     commandBufferMapSize = m_commandBufferDB.size();
    synStatus                    status(synSuccess);

    auto itr = m_commandBufferDB.begin();
    while (itr != m_commandBufferDB.end())
    {
        void*          key            = itr->first;
        CommandBuffer* pCommandBuffer = itr->second;

        if (_destroyCommandBuffer(pCommandBuffer) != synSuccess)
        {
            LOG_ERR(SYN_CS, "{}: Failed to destroy command-buffer (key = {})", HLLOG_FUNC, key);
            // We want to delete anyway, and the delete is part of the _destroy operation which failed, so...
            delete pCommandBuffer;
            status = synFail;
        }
        else
        {
            numOfElemCleared++;
        }

        itr = _remove(itr);
    }

    LOG_TRACE(SYN_CS, "{} destroyed {} elements out of {}", HLLOG_FUNC, numOfElemCleared, commandBufferMapSize);
    return status;
}

uint32_t CommandBufferMap::MapSize()
{
    uint32_t mapSize;
    std::unique_lock<std::mutex> mlock(m_commandBufferDBmutex);
    mapSize = m_commandBufferDB.size();

    return mapSize;
}

CommandBufferMap::CommandBufferMappingIter CommandBufferMap::_remove(CommandBufferMappingIter& commandBufferItr)
{
    void*                    cmdBufferKey = commandBufferItr->first;
    CommandBufferMappingIter cmdBuffItr   = m_commandBufferDB.erase(commandBufferItr);
    return cmdBuffItr;
}

synStatus CommandBufferMap::_destroyCommandBuffer(CommandBuffer* pCommandBuffer)
{
    synStatus status = pCommandBuffer->UnmapBuffer();
    if (status != synSuccess)
    {
        LOG_ERR(SYN_CS, "Failed to unmap command-buffer");
        return status;
    }

    status = pCommandBuffer->DestroyCommandBuffer();
    if (status != synSuccess)
    {
        LOG_ERR(SYN_CS, "Failed to destroy command-buffer");
        return status;
    }

    delete pCommandBuffer;

    return status;
}

// ----------------------------------------------------------------------- //

CommandBuffer::CommandBuffer()
: m_cbHandle(0),
  m_occupiedSize(0),
  m_totalSize(0),
  m_bufferMap(NULL),
  m_isCbCreated(false),
  m_isBufferMapped(false),
  m_pageSize((unsigned)OSAL::getInstance().getPageSize())
{
}

synStatus CommandBuffer::ClearCB()
{
    if (m_totalSize == 0)
    {
        LOG_ERR(SYN_CS, "Command buffer size is 0");
        return synFail;
    }

    m_occupiedSize = 0;
    return synSuccess;
}

void CommandBuffer::SetQueueIndex(uint32_t queueIndex)
{
    m_queueIndex.set(queueIndex);
}

synStatus CommandBuffer::UpdateOccupiedSize(uint64_t additionalUsedSize)
{
    LOG_TRACE(SYN_CS, "{} , occupiedSize {} additionalUsedSize {}", HLLOG_FUNC, m_occupiedSize, additionalUsedSize);

    if (additionalUsedSize % m_cpFetchSize != 0)
    {
        LOG_ERR(SYN_CS, "Invalid additional occupied size ({}), must be divisible by 8", additionalUsedSize);
        return synInvalidArgument;
    }

    if (m_occupiedSize + additionalUsedSize > m_totalSize)
    {
        return synCbFull;
    }

    m_occupiedSize += additionalUsedSize;

    return synSuccess;
}

synStatus CommandBuffer::SetBufferToCB(const void* pBuffer, unsigned cb_size, uint64_t* bufferOffset /*=nullptr*/)
{
    LOG_TRACE(SYN_CS, "{} , pBuffer {} cb_size {}", HLLOG_FUNC, pBuffer, cb_size);

    if (cb_size % m_cpFetchSize != 0)
    {
        LOG_ERR(SYN_CS, "Invalid cb size ({}), must be divisible by 8", cb_size);
        return synInvalidArgument;
    }

    if ((uint64_t)m_occupiedSize + cb_size > (uint64_t)m_totalSize)
    {
        return synCbFull;
    }

    memcpy(m_bufferMap + m_occupiedSize, pBuffer, cb_size);
    if (bufferOffset != nullptr)
    {
        *bufferOffset = (uint64_t)m_occupiedSize;
    }

    m_occupiedSize += cb_size;

    return synSuccess;
}

synStatus CommandBuffer::GetPacketFromCB(void* pPacket, unsigned packet_size, unsigned pkt_offset)
{
    if (packet_size % m_cpFetchSize != 0)
    {
        LOG_ERR(SYN_CS, "Invalid packet size ({}), must be divisible by 8", packet_size);
        return synInvalidArgument;
    }

    if ((pkt_offset > m_occupiedSize) || (pkt_offset + packet_size > m_occupiedSize))
    {
        LOG_ERR(SYN_CS, "Invalid packet to copy from CB at offset {} with packet size {}", pkt_offset, packet_size);
        return synInvalidArgument;
    }

    synStatus status = MapBuffer();
    if (status != synSuccess)
    {
        return status;
    }

    memcpy(pPacket, m_bufferMap + pkt_offset, packet_size);

    return synSuccess;
}

synStatus CommandBuffer::FillCBChunk(hl_cs_chunk& args, uint32_t queueOffset)
{
    uint32_t  queueIndex = 0;
    synStatus status     = GetQueueIndex(queueIndex);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_CS, "{}: Queue index was not set", HLLOG_FUNC);
        return synFail;
    }

    LOG_TRACE(SYN_CS,
              "    Fill CB handle 0x{:x}, size {}, queue index {} queueOffset {} (final index {}) isBufferMapped {}",
              m_cbHandle,
              m_occupiedSize,
              queueIndex,
              queueOffset,
              queueIndex + queueOffset,
              m_isBufferMapped);
    args.cb_handle   = m_cbHandle;
    args.cb_size     = m_occupiedSize;
    args.queue_index = queueIndex + queueOffset;

    return synSuccess;
}

unsigned char* CommandBuffer::GetBufferMap() const
{
    return m_bufferMap;
}

uint32_t CommandBuffer::GetOccupiedSize() const
{
    return m_occupiedSize;
}

uint64_t CommandBuffer::GetCbHandle() const
{
    return m_cbHandle;
}

uint32_t CommandBuffer::GetQueueIndex() const
{
    if (!m_queueIndex.is_set())
    {
        return INVALID_QUEUE_INDEX;
    }

    return m_queueIndex.value();
}

synStatus CommandBuffer::GetQueueIndex(uint32_t& queueIndex) const
{
    if (!m_queueIndex.is_set())
    {
        return synFail;
    }

    queueIndex = m_queueIndex.value();
    return synSuccess;
}

synStatus ExternalCommandBuffer::InitializeCommandBuffer(uint32_t size /*= c_defaultCbSize*/)
{
    synStatus status            = synSuccess;
    uint32_t  commandBufferSize = size;

    if (m_isCbCreated)
    {
        LOG_WARN(SYN_CS, "Command buffer already initialized");
        return synObjectAlreadyInitialized;
    }

    if (size == 0)
    {
        LOG_ERR(SYN_CS, "zero CB size is not allowed");
        return synInvalidArgument;
    }

    if (commandBufferSize % m_pageSize != 0)
    {
        commandBufferSize += (m_pageSize - commandBufferSize % m_pageSize);

        if (commandBufferSize > HL_MAX_CB_SIZE)
        {
            commandBufferSize = HL_MAX_CB_SIZE;
        }
    }

    uint64_t handle;
    int      ret;
    int      fd = OSAL::getInstance().getFd();

    ret = hlthunk_request_command_buffer(fd, commandBufferSize, &handle);

    if (ret < 0)
    {
        LOG_ERR(SYN_CS, "{}: hlthunk_request_command_buffer size {} error {} errno {}",
                HLLOG_FUNC, commandBufferSize, ret, errno);
        _SYN_SINGLETON_INTERNAL->notifyHlthunkFailure(DfaErrorCode::requestCommandBufferFailed);
        status = ((errno == ENODEV) ? synDeviceReset : synFailedToInitializeCb);
    }

    if (status)
    {
        ETL_TRIGGER(EVENT_LOGGER_TRIGGER_TYPE_CS_ORDER);

        LOG_ERR(SYN_CS, "Can not create command buffer, ioctl failed errno {}", errno);
        return (status == synDeviceReset) ? synDeviceReset : synFailedToInitializeCb;
    }

    m_cbHandle     = handle;
    m_totalSize    = commandBufferSize;
    m_occupiedSize = 0;

    m_isCbCreated = true;

    return synSuccess;
}

synStatus ExternalCommandBuffer::DestroyCommandBuffer()
{
    synStatus status = synSuccess;
    if (!m_isCbCreated)
    {
        LOG_ERR(SYN_CS, "Failed to destroy - command buffer is not created");
        return synObjectNotInitialized;
    }

    int fd = OSAL::getInstance().getFd();
    if (fd == -1) return synFailedToFreeCb;

    int ret = hlthunk_destroy_command_buffer(fd, m_cbHandle);

    if (ret < 0)
    {
        LOG_ERR(SYN_CS, "{}: hlthunk_destroy_command_buffer m_cbHandle 0x{:x} error {} errno {}",
                HLLOG_FUNC, m_cbHandle, ret, errno);
        _SYN_SINGLETON_INTERNAL->notifyHlthunkFailure(DfaErrorCode::destroyCommandBufferFailed);
        status = ((errno == ENODEV) ? synDeviceReset : synFailedToInitializeCb);
    }

    if (status)
    {
        ETL_TRIGGER(EVENT_LOGGER_TRIGGER_TYPE_CS_ORDER);

        LOG_ERR(SYN_CS, "Failed to free command buffer, ioctl failed");
        return (status == synDeviceReset) ? synDeviceReset : synFailedToFreeCb;
    }

    m_totalSize   = 0;
    m_isCbCreated = false;
    return synSuccess;
}

synStatus ExternalCommandBuffer::MapBuffer()
{
    if (m_bufferMap)
    {
        LOG_DEBUG(SYN_CS, "Try to map already mapped command buffer {}", (void*)this);
        return synSuccess;
    }

    if (m_totalSize == 0)
    {
        LOG_ERR(SYN_CS, "Failure on command buffer mapping because not initialized");
        return synObjectNotInitialized;
    }

    m_bufferMap = (unsigned char*)OSAL::getInstance().mapMem(m_totalSize, m_cbHandle);
    if (m_bufferMap == NULL)
    {
        LOG_ERR(SYN_CS, "Failure on command buffer mapping");
        return synFailedToMapCb;
    }

    m_isBufferMapped = true;
    return synSuccess;
}

synStatus ExternalCommandBuffer::UnmapBuffer()
{
    if (!m_isBufferMapped)
    {
        LOG_ERR(SYN_CS, "{}: Command buffer is not mapped", HLLOG_FUNC);
        return synSuccess;
    }

    int status = OSAL::getInstance().unmapMem(m_bufferMap, m_totalSize);
    if (status)
    {
        LOG_ERR(SYN_CS, "Failed to unmap CB");
        return synFailedToUnmapCb;
    }

    m_bufferMap      = NULL;
    m_isBufferMapped = false;
    return synSuccess;
}

synStatus MmuMappedCommandBuffer::InitializeCommandBuffer(uint32_t size /*= c_defaultCbSize*/)
{
    uint32_t commandBufferSize = size;

    if (m_isCbCreated)
    {
        LOG_WARN(SYN_CS, "Command buffer already initialized");
        return synObjectAlreadyInitialized;
    }

    if (size == 0)
    {
        LOG_ERR(SYN_CS, "zero CB size is not allowed");
        return synInvalidArgument;
    }

    if (commandBufferSize % m_pageSize != 0)
    {
        commandBufferSize += (m_pageSize - commandBufferSize % m_pageSize);
    }

    synDeviceInfo devInfo {};
    OSAL::getInstance().GetDeviceInfo(devInfo);

    synStatus status = _SYN_SINGLETON_INTERNAL->allocateDeviceMemory(devInfo.deviceId,
                                                                     commandBufferSize,
                                                                     synMemFlags::synMemHost,
                                                                     (void**)&m_bufferMap,
                                                                     0,
                                                                     &m_cbHandle);
    if (status != synSuccess)
    {
        ETL_TRIGGER(EVENT_LOGGER_TRIGGER_TYPE_CS_ORDER);

        LOG_ERR(SYN_CS, "Can not create command buffer, allocation failed");
        return (status == synDeviceReset) ? synDeviceReset : synFailedToInitializeCb;
    }

    m_totalSize    = commandBufferSize;
    m_occupiedSize = 0;

    m_isCbCreated    = true;
    m_isBufferMapped = true;

    return synSuccess;
}

synStatus MmuMappedCommandBuffer::DestroyCommandBuffer()
{
    if (!m_isCbCreated)
    {
        LOG_ERR(SYN_CS, "Failed to destroy - command buffer is not created");
        return synObjectNotInitialized;
    }

    synDeviceInfo devInfo {};
    OSAL::getInstance().GetDeviceInfo(devInfo);

    synStatus status =
        _SYN_SINGLETON_INTERNAL->deallocateDeviceMemory(devInfo.deviceId, (void*)m_bufferMap, synMemFlags::synMemHost);

    if (status != synSuccess)
    {
        ETL_TRIGGER(EVENT_LOGGER_TRIGGER_TYPE_CS_ORDER);

        LOG_ERR(SYN_CS, "Failed to free command buffer, ioctl failed");
        return (status == synDeviceReset) ? synDeviceReset : synFailedToFreeCb;
    }

    m_totalSize      = 0;
    m_isCbCreated    = false;
    m_isBufferMapped = false;
    return synSuccess;
}

synStatus MmuMappedCommandBuffer::MapBuffer()
{
    return synSuccess;
}

synStatus MmuMappedCommandBuffer::UnmapBuffer()
{
    return synSuccess;
}

synStatus MmuMappedCommandBuffer::FillCBChunk(hl_cs_chunk& args, uint32_t queueOffset)
{
    CommandBuffer::FillCBChunk(args, queueOffset);
    args.cs_chunk_flags = HL_CS_CHUNK_FLAGS_USER_ALLOC_CB;

    return synSuccess;
}

#endif  //_WIN32
