#include "queue_handle_container.hpp"
#include <algorithm>
#include <defenders.h>
#include "log_manager.h"
#include "runtime/common/recipe/recipe_handle_impl.hpp"
#include "queue_compute_qman.hpp"
#include "defs.h"

bool QueueInterfaceContainer::addStreamHandle(QueueInterface* pQueueInterface)
{
    CHECK_POINTER(SYN_STREAM, pQueueInterface, "Stream-handle", false);
    auto streamHandlesEndItr = m_streamHandles.end();
    auto streamHandlesItr    = std::find(m_streamHandles.begin(), streamHandlesEndItr, pQueueInterface);
    if (streamHandlesItr != streamHandlesEndItr)
    {
        LOG_DEBUG(SYN_STREAM,
                  "{}: Stream-handle already found on database (0x{:x})",
                  HLLOG_FUNC,
                  (uint64_t)pQueueInterface);
        return true;
    }

    m_streamHandles.push_back(pQueueInterface);

    return true;
}

bool QueueInterfaceContainer::removeStreamHandle(QueueInterface* pQueueInterface)
{
    CHECK_POINTER(SYN_STREAM, pQueueInterface, "Stream-handle", false);

    auto streamHandlesEndItr = m_streamHandles.end();
    auto streamHandlesItr    = std::find(m_streamHandles.begin(), streamHandlesEndItr, pQueueInterface);
    if (streamHandlesItr == streamHandlesEndItr)
    {
        LOG_DEBUG(SYN_STREAM,
                  "{}: Stream-handle not found on database (0x{:x})",
                  HLLOG_FUNC,
                  (uint64_t)pQueueInterface);
        return true;
    }

    CHECK_POINTER(SYN_STREAM, pQueueInterface, "pStream", false);
    if (isComputeStreamValid(pQueueInterface, pQueueInterface->getBasicQueueInfo()))
    {
        static_cast<QueueComputeQman*>(pQueueInterface)->notifyAllRecipeRemoval();
    }

    m_streamHandles.erase(std::find(m_streamHandles.begin(), m_streamHandles.end(), pQueueInterface));

    return true;
}

void QueueInterfaceContainer::notifyRecipeRemoval(InternalRecipeHandle& rRecipeHandle)
{
    for (auto const& rpQueueInterface : m_streamHandles)
    {
        if (isComputeStreamValid(rpQueueInterface, rpQueueInterface->getBasicQueueInfo()))
        {
            static_cast<QueueComputeQman*>(rpQueueInterface)->notifyRecipeRemoval(rRecipeHandle);
        }
    }
}

void QueueInterfaceContainer::notifyAllRecipeRemoval()
{
    for (auto const& rpQueueInterface : m_streamHandles)
    {
        if (isComputeStreamValid(rpQueueInterface, rpQueueInterface->getBasicQueueInfo()))
        {
            static_cast<QueueComputeQman*>(rpQueueInterface)->notifyAllRecipeRemoval();
        }
    }
}

synStatus QueueInterfaceContainer::getDeviceTotalStreamMappedMemory(uint64_t& totalStreamMappedMemorySize) const
{
    synStatus status            = synSuccess;
    totalStreamMappedMemorySize = 0;
    for (auto const& rpQueueInterface : m_streamHandles)
    {
        uint64_t currentResult = 0;

        if (static_cast<QueueBaseQman*>(rpQueueInterface)->getMappedMemorySize(currentResult) != synSuccess)
        {
            LOG_ERR(SYN_STREAM,
                    "{}: Can not retrieve stream mapped memory for stream {}",
                    HLLOG_FUNC,
                    rpQueueInterface);
            status = synFail;
        }
        totalStreamMappedMemorySize += currentResult;
    }
    LOG_DEBUG(SYN_STREAM,
              "{}: Calculated total stream mapped memory size (0x{:x})",
              HLLOG_FUNC,
              totalStreamMappedMemorySize);
    return status;
}

bool QueueInterfaceContainer::isComputeStreamValid(const QueueInterface* pQueueInterface,
                                                   const BasicQueueInfo& rBasicQueueInfo)
{
    if (rBasicQueueInfo.queueType == INTERNAL_STREAM_TYPE_COMPUTE)
    {
        HB_ASSERT((pQueueInterface != nullptr),
                  "StreamHandles DB contained null-pointer as compute pStream during recipe-handle removal");
        return true;
    }

    return false;
}