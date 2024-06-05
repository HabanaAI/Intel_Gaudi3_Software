#pragma once

#include "habana_global_conf.h"
#include "synapse_common_types.h"

uint64_t  internalProfilerGetCurrentTimeNs();
synStatus internalProfileInternalFunction(const char* description, uint64_t nanoTime);

#define PROFILER_COLLECT_TIME()                                                                                        \
    uint64_t profStart = 0;                                                                                            \
    if (GCFG_ENABLE_PROFILER.value() && GCFG_ENABLE_SYN_LAUNCH_PROFILER_ANNOTATIONS.value())                           \
    {                                                                                                                  \
        profStart = internalProfilerGetCurrentTimeNs();                                                                \
    }

#define PROFILER_MEASURE_TIME(desc)                                                                                    \
    if (GCFG_ENABLE_PROFILER.value() && GCFG_ENABLE_SYN_LAUNCH_PROFILER_ANNOTATIONS.value())                           \
    {                                                                                                                  \
        internalProfileInternalFunction(desc, profStart);                                                              \
    }

#define PROFILER_MEASURE_TIME(desc)                                                                                    \
    if (GCFG_ENABLE_PROFILER.value() && GCFG_ENABLE_SYN_LAUNCH_PROFILER_ANNOTATIONS.value())                           \
    {                                                                                                                  \
        internalProfileInternalFunction(desc, profStart);                                                              \
    }

class ProfilerApi
{
public:
    static inline synStatus setHostProfilerApiId(uint8_t apiId)
    {
        synStatus status;
        if (GCFG_ENABLE_PROFILER.value())
        {
            status = setHostProfilerApiIdInternal(apiId);
        }
        else
        {
            status = synSuccess;
        }
        return status;
    }

private:
    using SynKeyValArgVector = std::vector<synTraceEventArg>;

    static synStatus setHostProfilerApiIdInternal(uint8_t apiId);

    static const std::string API_ID_KEY;
};
