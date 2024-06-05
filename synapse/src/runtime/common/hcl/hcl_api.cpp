#include "hcl_api.hpp"
#include "syn_singleton.hpp"
#include "profiler_api.hpp"

synStatus synGenerateApiId(uint8_t& rApiId)
{
    LOG_SYN_API();
    const synStatus status = _SYN_SINGLETON_INTERNAL->generateApiId(rApiId);
    if (status == synSuccess)
    {
        ProfilerApi::setHostProfilerApiId(rApiId);
    }
    return status;
}

synStatus synStreamSyncHCLStreamHandle(synStreamHandle streamHandle)
{
    LOG_SYN_API("streamHandle {:#x}", TO64(streamHandle));
    return _SYN_SINGLETON_INTERNAL->syncHCLStreamHandle(streamHandle);
}

synStatus synStreamIsInitialized(synStreamHandle streamHandle, bool& rIsInitialized)
{
    LOG_SYN_API("streamHandle {:#x}", TO64(streamHandle));
    return _SYN_SINGLETON_INTERNAL->isStreamInitialized(streamHandle, rIsInitialized);
}

synStatus synStreamFlushWaitsOnCollectiveHandle(synStreamHandle streamHandle)
{
    LOG_SYN_API("streamHandle {:#x}", TO64(streamHandle));
    return _SYN_SINGLETON_INTERNAL->flushWaitsOnCollectiveStream(streamHandle);
}

uint32_t synStreamGetPhysicalQueueOffset(synStreamHandle streamHandle)
{
    LOG_SYN_API("streamHandle {:#x}", TO64(streamHandle));
    return _SYN_SINGLETON_INTERNAL->getNetworkStreamPhysicalQueueOffset(streamHandle);
}

hcl::hclStreamHandle synStreamGetHclStreamHandle(synStreamHandle streamHandle)
{
    LOG_SYN_API("streamHandle {:#x}", TO64(streamHandle));
    return _SYN_SINGLETON_INTERNAL->getNetworkStreamHclStreamHandle(streamHandle);
}

uint64_t hclNotifyFailure(DfaErrorCode dfaErrorCode, uint64_t options)
{
    return hclNotifyFailureV2(dfaErrorCode, options, "");
}

uint64_t hclNotifyFailureV2(DfaErrorCode dfaErrorCode, uint64_t options, std::string msg)
{
    LOG_ERR(SYN_API, "HCL notified about a Dfa error with code {:x} option {:x} msg {}", dfaErrorCode, options, msg);

    DfaExtraInfo dfaExtraInfo = {
        .extraInfo = DfaExtraInfo::DfaExtraInfoMsg {.msg = msg}};

    _SYN_SINGLETON_INTERNAL->notifyHlthunkFailure(dfaErrorCode, dfaExtraInfo);
    return 0;
}
