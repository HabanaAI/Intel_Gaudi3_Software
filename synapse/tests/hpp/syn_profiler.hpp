#pragma once

#include "syn_object.hpp"
#include "synapse_common_types.h"

#include <cstddef>
#include <memory>
#include <utility>

namespace syn
{
class Profiler : public SynObject<uint32_t>
{
public:
    void start() { SYN_CHECK(synProfilerStart(m_traceType, handle())); }

    void stop() { SYN_CHECK(synProfilerStop(m_traceType, handle())); }

    // TODO: temp api, need a nicer return type
    std::pair<std::size_t, std::unique_ptr<char[]>> getEvents(synTraceFormat traceFormat) const
    {
        size_t size         = 0;
        size_t numOfEntries = 0;
        SYN_CHECK(synProfilerGetTrace(m_traceType, handle(), traceFormat, nullptr, &size, &numOfEntries));
        auto buffer = std::make_unique<char[]>(size);
        SYN_CHECK(synProfilerGetTrace(m_traceType, handle(), traceFormat, buffer.get(), &size, &numOfEntries));
        return {numOfEntries, std::move(buffer)};
    }

    uint64_t getTimeInNanoseconds() const
    {
        uint64_t time = 0;
        SYN_CHECK(synProfilerGetCurrentTimeNS(&time));
        return time;
    }

    void addCustomMeasurement(const std::string& description, uint64_t timeInNs)
    {
        SYN_CHECK(synProfilerAddCustomMeasurement(description.c_str(), timeInNs));
    }

private:
    Profiler(const std::shared_ptr<uint32_t>& handle, synTraceType traceType)
    : SynObject(handle), m_traceType(traceType)
    {
    }

    synTraceType m_traceType;

    friend class Device;  // Device class requires access to Profiler private constructor
};
}  // namespace syn