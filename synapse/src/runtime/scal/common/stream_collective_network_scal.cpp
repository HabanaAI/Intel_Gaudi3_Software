#include "stream_collective_network_scal.hpp"
#include "hcl_public_streams.h"
#include "hcl/hcl_api_wrapper.h"
#include "defs.h"
#include "defenders.h"
#include "runtime/scal/common/scal_event.hpp"

QueueCollectiveNetworkScal::QueueCollectiveNetworkScal(const BasicQueueInfo& rBasicQueueInfo,
                                                       hclApiWrapper&        rHclApiWrapper)
: QueueBaseScal(rBasicQueueInfo), m_rHclApiWrapper(rHclApiWrapper), m_pHclStreamHandle(nullptr)
{
}

synStatus QueueCollectiveNetworkScal::createHclStream()
{
    if (m_pHclStreamHandle != nullptr)
    {
        LOG_ERR(SYN_STREAM, "HCL stream was already created {:#x}", TO64(m_pHclStreamHandle));
        return synFail;
    }
    synStatus status = m_rHclApiWrapper.createStream(m_pHclStreamHandle);
    if (status == synSuccess)
    {
        m_physicalQueueOffset = hcl::getStreamID(m_pHclStreamHandle);
        LOG_DEBUG(SYN_STREAM,
                  "HCL stream was created {:#x} with StreamID = {}",
                  TO64(m_pHclStreamHandle),
                  m_physicalQueueOffset);
    }
    else
    {
        LOG_ERR(SYN_STREAM, "HCL stream was not created status {}", status);
    }
    return status;
}

synStatus QueueCollectiveNetworkScal::destroyHclStream()
{
    synStatus status;
    if (m_pHclStreamHandle != nullptr)
    {
        status = m_rHclApiWrapper.destroyStream(m_pHclStreamHandle);
        if (status == synSuccess)
        {
            LOG_DEBUG(SYN_STREAM, "HCL stream was destroyed {:#x}", TO64(m_pHclStreamHandle));
            m_pHclStreamHandle = nullptr;
        }
        else
        {
            LOG_ERR(SYN_STREAM, "HCL stream was not destroyed {:#x} status {}", TO64(m_pHclStreamHandle), status);
        }
    }
    else
    {
        LOG_DEBUG(SYN_STREAM, "HCL stream was already destroyed");
        status = synSuccess;
    }
    return status;
}

synStatus QueueCollectiveNetworkScal::getMappedMemorySize(uint64_t& mappedMemorySize) const
{
    mappedMemorySize = 0;
    return synSuccess;
}

synStatus QueueCollectiveNetworkScal::eventRecord(EventInterface& rEventInterface, synStreamHandle streamHandle)
{
    LOG_DEBUG(SYN_STREAM, "{} Stream {}", HLLOG_FUNC, m_basicQueueInfo.getDescription());
#ifdef DISABLE_SYNC_ON_DEV
    synchronizeStream(streamHandle);
    return synSuccess;
#endif

    ScalEvent* pScalEvent = dynamic_cast<ScalEvent*>(&rEventInterface);
    CHECK_POINTER(SYN_STREAM, pScalEvent, "pScalEvent", synInvalidArgument);
    pScalEvent->clearState();
    pScalEvent->pStreamIfScal = this;
    return m_rHclApiWrapper.eventRecord(pScalEvent, m_pHclStreamHandle);
}

synStatus QueueCollectiveNetworkScal::eventWait(const EventInterface& rEventInterface,
                                                const unsigned int    flags,
                                                synStreamHandle       streamHandle)
{
    LOG_DEBUG(SYN_STREAM, "{} Stream {}", HLLOG_FUNC, m_basicQueueInfo.getDescription());

#ifdef DISABLE_SYNC_ON_DEV
    return synSuccess;
#endif

    const ScalEvent* pScalEvent = dynamic_cast<const ScalEvent*>(&rEventInterface);
    CHECK_POINTER(SYN_STREAM, pScalEvent, "pScalEvent", synInvalidArgument);

    // defend against overriding the event while we're working on it
    ScalEvent scalEvent = *pScalEvent;
    const ScalLongSyncObject& rLongSo = scalEvent.longSo;

    LOG_INFO(SYN_STREAM,
             "{} on stream collective, longso index {} value {} (isOnHclStream {})",
             HLLOG_FUNC,
             rLongSo.m_index,
             rLongSo.m_targetValue,
             scalEvent.isOnHclStream());

    return m_rHclApiWrapper.streamWaitEvent(m_pHclStreamHandle, &scalEvent);
}

synStatus QueueCollectiveNetworkScal::eventQuery(const EventInterface& rEventInterface)
{
    const ScalEvent* pScalEvent = dynamic_cast<const ScalEvent*>(&rEventInterface);
    CHECK_POINTER(SYN_STREAM, pScalEvent, "pScalEvent", synInvalidArgument);
    // defend against overriding the event while we're working on it
    ScalEvent scalEvent = *pScalEvent;
    HB_ASSERT(scalEvent.isOnHclStream(), "Invalid source stream");
    return m_rHclApiWrapper.eventQuery(&scalEvent);
}

synStatus QueueCollectiveNetworkScal::eventSynchronize(const EventInterface& rEventInterface)
{
    const ScalEvent* pScalEvent = dynamic_cast<const ScalEvent*>(&rEventInterface);
    CHECK_POINTER(SYN_STREAM, pScalEvent, "pScalEvent", synInvalidArgument);
    // defend against overriding the event while we're working on it
    ScalEvent scalEvent = *pScalEvent;
    HB_ASSERT(scalEvent.isOnHclStream(), "Invalid source stream");
    return m_rHclApiWrapper.eventSynchronize(&scalEvent);
}

synStatus QueueCollectiveNetworkScal::query()
{
    return m_rHclApiWrapper.streamQuery(m_pHclStreamHandle);
}

synStatus QueueCollectiveNetworkScal::synchronize(synStreamHandle streamHandle, bool isUserRequest)
{
    return m_rHclApiWrapper.synchronizeStream(m_pHclStreamHandle);
}

void QueueCollectiveNetworkScal::dfaInfo(DfaReq dfaReq, uint64_t csSeq)
{
    LOG_INFO(SYN_DEV_FAIL, "--- network stream, no stream info ---");
}

synStatus QueueCollectiveNetworkScal::getDynamicShapesTensorInfoArray(synRecipeHandle             recipeHandle,
                                                                      std::vector<tensor_info_t>& tensorInfoArray) const
{
    LOG_ERR(SYN_API, "Unsupported stream type for getLastTensorArray: {}", getBasicQueueInfo().queueType);
    return synFail;
}