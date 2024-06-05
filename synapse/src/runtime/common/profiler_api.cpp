#include "profiler_api.hpp"
#include "syn_singleton.hpp"

uint64_t internalProfilerGetCurrentTimeNs()
{
    return synSingleton::getInstance()->profilerGetCurrentTimeNs();
}

synStatus internalProfileInternalFunction(const char* description, uint64_t nanoTime)
{
    return synSingleton::getInstance()->profileInternalFunction(description, nanoTime);
}

synStatus ProfilerApi::setHostProfilerApiIdInternal(uint8_t apiId)
{
    synTraceEventArg apiId_arg;
    apiId_arg.key       = API_ID_KEY.c_str();
    apiId_arg.type      = synTraceEventArg::TYPE_UINT64;
    apiId_arg.value.u64 = apiId;

    SynKeyValArgVector synArgVector;
    synArgVector.push_back(apiId_arg);

    return synSingleton::getInstance()->setHostProfilerArg(synArgVector);
}

const std::string ProfilerApi::API_ID_KEY {"API_ID"};