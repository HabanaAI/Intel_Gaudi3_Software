#include "api.h"
#include "api_calls_counter.hpp"
#include "syn_logging.h"
#include "syn_singleton.hpp"
#include "synapse_api.h"

synStatus SYN_API_CALL synInitialize()
{
    synStatus status = synSuccess;

    try
    {
        LOG_SYN_API();
        (void) ApiCounterRegistry::getInstance();

        status = synSingleton::initializeInstance();
#if 0  // skip the following condition until implemented in ComplexGuid
        if (status == synSuccess && _SYN_SINGLETON_->supportsComplexGuid() != synSuccess)
        {
            synDestroy();
            return synFail;
        }
#endif
    }
    catch (...)
    {
        return handleException(__FUNCTION__);
    }

    return status;
}

synStatus SYN_API_CALL synDestroy()
{
    synStatus status = synSuccess;

    try
    {
        LOG_SYN_API();

        status = synSingleton::destroyInstance();
    }
    catch (...)
    {
        return handleException(__FUNCTION__);
    }

    return status;
}

synStatus SYN_API_CALL synProfilerQueryRequiredMemory(const synDeviceId deviceId, uint32_t* bytesRequired)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API("deviceId = {} bytesRequired {} ", deviceId, bytesRequired);
    status = _SYN_SINGLETON_->profilerQueryRequiredMemory(deviceId, bytesRequired);
    API_EXIT_STATUS_TIMED(status, synProfilerQueryRequiredMemoryP);
}

synStatus SYN_API_CALL synProfilerSetUserBuffer(const synDeviceId deviceId, void* userBuffer)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API("deviceId = {} userBuffer {}", deviceId, userBuffer);
    status = _SYN_SINGLETON_->profilerSetUserBuffer(deviceId, userBuffer);
    API_EXIT_STATUS_TIMED(status, synProfilerSetUserBufferP);
}

synStatus SYN_API_CALL synProfilerStart(synTraceType type, const synDeviceId deviceId)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API("deviceId {} type {}", deviceId, type);
    status = _SYN_SINGLETON_->profilerStart(type, deviceId);
    API_EXIT_STATUS_TIMED(status, synProfilerStartP);
}

synStatus SYN_API_CALL synProfilerStop(synTraceType type, const synDeviceId deviceId)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API("deviceId {} type {}", deviceId, type);
    status = _SYN_SINGLETON_->profilerStop(type, deviceId);
    API_EXIT_STATUS_TIMED(status, synProfilerStopP);
}

synStatus SYN_API_CALL synProfilerGetTrace(synTraceType      type,
                                           const synDeviceId deviceId,
                                           synTraceFormat    format,
                                           void*             buffer,
                                           size_t*           size,
                                           size_t*           numEntries)
{
    API_ENTRY_STATUS_TIMED()
    LOG_SYN_API("deviceId {} type {} format {}", deviceId, type, format);
    status = _SYN_SINGLETON_->profilerGetTrace(type, deviceId, format, buffer, size, numEntries);
    API_EXIT_STATUS_TIMED(status, synProfilerGetTraceP);
}

synStatus SYN_API_CALL synProfilerGetCurrentTimeNS(uint64_t* nanoTime)
{
    API_ENTRY_STATUS_TIMED()
    if (nanoTime == nullptr)
    {
        return synInvalidArgument;
    }

    *nanoTime = _SYN_SINGLETON_->profilerGetCurrentTimeNs();
    API_EXIT_STATUS_TIMED(synSuccess, synProfilerGetCurrentTimeNSP);
}

synStatus SYN_API_CALL synProfilerAddCustomMeasurement(const char* description, uint64_t nanoTime)
{
    API_ENTRY_STATUS_TIMED()
    status = _SYN_SINGLETON_->profileInternalFunction(description, nanoTime);
    API_EXIT_STATUS_TIMED(status, synProfilerAddCustomMeasurementP);
}

synStatus SYN_API_CALL synProfilerAddCustomMeasurementArgs(const char* description, uint64_t profStartTime, const char** args, size_t argsSize)
{
    API_ENTRY_STATUS_TIMED()
    status = _SYN_SINGLETON_->profileInternalFunctionWithArgs(description, profStartTime, args, argsSize);
    API_EXIT_STATUS_TIMED(status, synProfilerAddCustomMeasurementArgsP);
}

synStatus SYN_API_CALL synProfilerAddCustomMeasurementArgsAndThread(const char* description, uint64_t profStartTime, const char** args, size_t argsSize, const char* threadName)
{
    API_ENTRY_STATUS_TIMED()
    status = _SYN_SINGLETON_->profileInternalFunctionWithArgsAndThread(description, profStartTime, args, argsSize, threadName);
    API_EXIT_STATUS_TIMED(status, synProfilerAddCustomMeasurementArgsP);
}

synStatus SYN_API_CALL synProfilerAddCustomEvent(const char* description, uint64_t profStartTime, uint64_t endTime, const char** args, size_t argsSize)
{
    API_ENTRY_STATUS_TIMED()
    status = _SYN_SINGLETON_->profilerAddCustomEvent(description, profStartTime, endTime, args, argsSize);
    API_EXIT_STATUS_TIMED(status, synProfilerAddCustomEventP);
}
