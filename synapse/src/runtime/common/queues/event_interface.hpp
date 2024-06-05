#pragma once
#include "synapse_types.h"

class EventInterface
{
public:
    enum WaitMode : uint8_t
    {
        unset   = 0,
        not_yet = 1,
        waited  = 2
    };
    virtual ~EventInterface()                = default;
    virtual std::string toString() const     = 0;

    static synStatus eventElapsedTime(uint64_t*             pNanoSeconds,
                                      const EventInterface* eventHandleStart,
                                      const EventInterface* eventHandleEnd);

    enum WaitMode  getWaitMode() const { return m_waitMode;}
    void setWaitMode(enum WaitMode  wait_mode) const {m_waitMode = wait_mode;}
protected:
    mutable enum WaitMode      m_waitMode = WaitMode::unset;
private:
    virtual synStatus getTime(uint64_t& nanoseconds, bool start) const = 0;
};