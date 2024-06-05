#pragma once

#include "syn_object.hpp"

namespace syn
{
class Event;
using Events = std::vector<Event>;

class Event : public SynObject<synEventHandle>
{
public:
    Event() = default;

    uint64_t getElapsedTime(const Event& other) const
    {
        uint64_t elapseTime = 0;
        SYN_CHECK(synEventElapsedTime(&elapseTime, handle(), other.handle()));
        return elapseTime;
    }

    synStatus tryGetElapsedTime(const Event& other, uint64_t& elapseTime) const
    {
        return synEventElapsedTime(&elapseTime, handle(), other.handle());
    }

    synStatus query() const { return synEventQuery(handle()); }

    void synchronize(const Event& other) const { SYN_CHECK(synEventSynchronize(handle())); }

private:
    Event(std::shared_ptr<synEventHandle> handle) : SynObject(handle) {}

    friend class Device;  // Device class requires access to Event private constructor
};
}  // namespace syn