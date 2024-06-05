#include "hcl_api_wrapper.h"
#include "synapse_common_types.h"
#include "log_manager.h"
#include "runtime/scal/common/scal_event.hpp"

synStatus hclApiWrapper::provideScal(scal_handle_t scal)
{
    try
    {
        m_hclPublicStream = hcl::HclPublicStreamsFactory::createHclPublicStreams(scal);
        return synSuccess;
    }
    catch (const hcl::NotImplementedApiException& e)
    {
        return synFail;
    }
}

synStatus hclApiWrapper::createStream(hcl::hclStreamHandle& streamHandle)
{
    try
    {
        streamHandle = m_hclPublicStream->createStream();
        return synSuccess;
    }
    catch (const hcl::NotImplementedApiException& e)
    {
        LOG_ERR(SYN_API, "CreateStrem for HCL not supported. {}", e.what());
        return synUnsupported;
    }
    catch (const std::exception& e)
    {
        LOG_ERR(SYN_API, "CreateStrem for HCL failed for {}", e.what());
        return synFail;
    }
}

synStatus hclApiWrapper::destroyStream(const hcl::hclStreamHandle& streamHandle) const
{
    try
    {
        m_hclPublicStream->destroyStream(streamHandle);
        return synSuccess;
    }
    catch (const hcl::NotImplementedApiException& e)
    {
        return synUnsupported;
    }
    catch (const std::exception& e)
    {
        return synFail;
    }
}

synStatus hclApiWrapper::eventRecord(ScalEvent* scalEvent, const hcl::hclStreamHandle streamHandle) const
{
    synStatus status = synSuccess;
    try
    {
        if (scalEvent->collectTime)
        {
            scalEvent->hclSyncInfo = m_hclPublicStream->eventRecord(streamHandle,
                                                                    scalEvent->collectTime,
                                                                    scalEvent->timestampBuff->cbOffsetInFd,
                                                                    scalEvent->timestampBuff->indexInCbOffsetInFd);
        }
        else
        {
            scalEvent->hclSyncInfo = m_hclPublicStream->eventRecord(streamHandle);
        }

        scalEvent->longSo.m_index       = scalEvent->hclSyncInfo.long_so_index;
        scalEvent->longSo.m_targetValue = scalEvent->hclSyncInfo.targetValue;

        scalEvent->setOnHclStream();

        LOG_DEBUG(SYN_STREAM,
                  "HCL {} longSOIdx 0x{:x}, target {}",
                  HLLOG_FUNC,
                  scalEvent->hclSyncInfo.long_so_index,
                  scalEvent->hclSyncInfo.targetValue);

        return status;
    }
    catch (const hcl::NotImplementedApiException& e)
    {
        status = synUnsupported;
    }
    catch (const std::exception& e)
    {
        status = synFail;
    }
    LOG_ERR(SYN_API, "{}: eventRecord failed with status {}", HLLOG_FUNC, status);
    return synFail;
}

synStatus hclApiWrapper::streamWaitEvent(const hcl::hclStreamHandle streamHandle, const ScalEvent* scalEvent) const
{
    synStatus     status = synSuccess;
    hcl::syncInfo params;

    if (scalEvent->isOnHclStream())
    {
        params = scalEvent->hclSyncInfo;
    }
    else
    {
        params.long_so_index = scalEvent->longSo.m_index;
        params.targetValue   = scalEvent->longSo.m_targetValue;
        params.cp_handle     = nullptr;
    }

    try
    {
        m_hclPublicStream->streamWaitEvent(streamHandle, params);

        return synSuccess;
    }
    catch (const hcl::NotImplementedApiException& e)
    {
        status = synUnsupported;
    }
    catch (const std::exception& e)
    {
        status = synFail;
    }

    LOG_ERR(SYN_API, "{}: eventRecord failed with status {}", HLLOG_FUNC, status);
    return status;
}

synStatus hclApiWrapper::streamQuery(const hcl::hclStreamHandle streamHandle) const
{
    synStatus status = synSuccess;

    try
    {
        if (m_hclPublicStream->streamQuery(streamHandle))
        {
            return synSuccess;
        }
        else
        {
            return synBusy;
        }
    }
    catch (const std::exception& e)
    {
        status = synFail;
    }

    LOG_ERR(SYN_API, "{}: streamQuery failed with status {}", HLLOG_FUNC, status);
    return status;
}

synStatus hclApiWrapper::eventQuery(const ScalEvent* scalEvent) const
{
    synStatus status = synSuccess;

    try
    {
        if (m_hclPublicStream->eventQuery(scalEvent->hclSyncInfo))
        {
            return synSuccess;
        }
        else
        {
            return synBusy;
        }
    }
    catch (const std::exception& e)
    {
        status = synFail;
    }

    LOG_ERR(SYN_API, "{}: eventQuery failed with status {}", HLLOG_FUNC, status);
    return status;
}

synStatus hclApiWrapper::synchronizeStream(const hcl::hclStreamHandle streamHandle) const
{
    synStatus status = synSuccess;

    try
    {
        m_hclPublicStream->streamSynchronize(streamHandle);
        return synSuccess;
    }
    catch (const std::exception& e)
    {
        status = synDeviceReset;
    }

    LOG_ERR(SYN_API, "{}: synchronizeStream failed with status {}", HLLOG_FUNC, status);
    return status;
}

synStatus hclApiWrapper::eventSynchronize(const ScalEvent* scalEvent) const
{
    synStatus status = synSuccess;

    try
    {
        m_hclPublicStream->eventSynchronize(scalEvent->hclSyncInfo);
        return synSuccess;
    }
    catch (const std::exception& e)
    {
        status = synDeviceReset;
    }

    LOG_ERR(SYN_API, "{}: eventSynchronize failed with status {}", HLLOG_FUNC, status);
    return status;
}

synStatus hclApiWrapper::checkHclFailure(DfaStatus dfaStatus, void (*logFunc)(int, const char*), hcl::HclPublicStreams::DfaLogPhase options) const
{
    synStatus status = synSuccess;

    try
    {
        m_hclPublicStream->DFA(dfaStatus, logFunc, options);
        return synSuccess;
    }
    catch (const std::exception& e)
    {
        status = synFail;
    }

    LOG_ERR(SYN_API, "{}: checkHclFailure failed with status {}", HLLOG_FUNC, status);
    return status;
}