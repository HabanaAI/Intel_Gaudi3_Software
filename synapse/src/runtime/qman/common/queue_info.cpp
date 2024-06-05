#include "queue_info.hpp"
#include "defs.h"
#include "synapse_runtime_logging.h"

#include <limits>

const uint64_t INVALID_CS_HANDLE = std::numeric_limits<uint64_t>::max();

QueueInfo::QueueInfo(uint64_t queueId, uint32_t physicalQueueOffset, PhysicalQueuesId physicalQueuesId, uint32_t flags)
: m_queueId(queueId), m_physicalQueueOffset(physicalQueueOffset), m_physicalQueuesId(physicalQueuesId)
{
    m_internalWaitHandle.handle = INVALID_CS_HANDLE;
}

TrainingRetCode QueueInfo::updateStatus(const InternalWaitHandle& waitForEventHandle, const std::string& desc)
{
    operationCompletion(false, waitForEventHandle.handle, true, false, desc);

    return TRAINING_RET_CODE_SUCCESS;
}

TrainingRetCode QueueInfo::getLastWaitHandles(InternalWaitHandlesVector& lastWaitHandles) const
{
    std::lock_guard<std::mutex> guard(m_mutex);
    const InternalWaitHandle&   waitHandle = m_internalWaitHandle;
    if (waitHandle.handle != INVALID_CS_HANDLE)
    {
        lastWaitHandles.push_back(waitHandle);
    }

    return TRAINING_RET_CODE_SUCCESS;
}

uint32_t QueueInfo::getPhysicalQueueOffset() const
{
    return m_physicalQueueOffset;
}

uint64_t QueueInfo::getQueueId() const
{
    return m_queueId;
}

const PhysicalQueuesId& QueueInfo::getPhysicalQueuesId() const
{
    return m_physicalQueuesId;
}

void QueueInfo::operationCompletion(bool               isSignalOperation,
                                    uint64_t           operationSequenceId,
                                    bool               updateWaitHandle,
                                    bool               debugCSMode,
                                    const std::string& desc)
{
    std::lock_guard<std::mutex> guard(m_mutex);

    if (updateWaitHandle)
    {
        if (operationSequenceId == INVALID_CS_HANDLE ||
            (operationSequenceId < m_internalWaitHandle.handle && m_internalWaitHandle.handle != INVALID_CS_HANDLE))
        {
            LOG_DEBUG(SYN_STREAM,
                      "{}: ignore update handle 0x{:x}, seqId is 0x{:x} desc {}",
                      HLLOG_FUNC,
                      m_internalWaitHandle.handle,
                      operationSequenceId,
                      desc);
        }
        else
        {
            LOG_TRACE(SYN_STREAM,
                      "{}: Update handle 0x{:x} -> 0x{:x} desc {}",
                      HLLOG_FUNC,
                      m_internalWaitHandle.handle,
                      operationSequenceId,
                      desc);
            m_internalWaitHandle.handle = operationSequenceId;
        }
    }

    if (debugCSMode)
    {
        CsInfo type;
        type.threadId                  = (uint64_t)pthread_self();
        type.isSignal                  = isSignalOperation;
        m_csTypes[operationSequenceId] = type;
    }
}

void QueueInfo::printCSDB() const
{
    if (!LOG_LEVEL_AT_LEAST_INFO(SYN_STREAM)) return;

    LOG_INFO(SYN_STREAM, "{}:", HLLOG_FUNC);

    std::lock_guard<std::mutex> guard(m_mutex);
    for (auto it = m_csTypes.cbegin(); it != m_csTypes.cend(); ++it)
    {
        LOG_INFO(SYN_STREAM, "seq {}, threadId {}, isSignal {}", it->first, it->second.threadId, it->second.isSignal);
    }
}