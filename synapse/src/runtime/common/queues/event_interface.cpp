#include "event_interface.hpp"
#include "defs.h"
/*
***************************************************************************************************
*   @brief Returns the time between events.
*          It gets the time of each event and calculates the difference
*   @param pNanoseconds     - return value of when this event was finished.
*          eventHandleStart - start event
*          eventHandleEnd   - end event
*   @return synStatus
***************************************************************************************************
*/
synStatus EventInterface::eventElapsedTime(uint64_t*             pNanoSeconds,
                                           const EventInterface* eventHandleStart,
                                           const EventInterface* eventHandleEnd)
{
    HB_ASSERT_PTR(pNanoSeconds);
    HB_ASSERT_PTR(eventHandleStart);
    LOG_TRACE(SYN_STREAM,
              "event elapsed time event {} to {}",
              eventHandleStart->toString(),
              eventHandleEnd ? eventHandleEnd->toString() : "");

    uint64_t  startTime;
    synStatus status = eventHandleStart->getTime(startTime, true);
    if (status != synSuccess)
    {
        return status;
    }
    if (eventHandleEnd == nullptr)
    {
        *pNanoSeconds = startTime;
        return synSuccess;
    }
    uint64_t endTime;
    status = eventHandleEnd->getTime(endTime, false);
    if (status != synSuccess)
    {
        return status;
    }

    *pNanoSeconds = endTime - startTime;
    LOG_TRACE(SYN_STREAM,
              "event elapsed time event {} to {} elapsed time is {} ns",
              eventHandleStart->toString(),
              eventHandleEnd ? eventHandleEnd->toString() : "",
              *pNanoSeconds);

    eventHandleStart->setWaitMode(EventInterface::WaitMode::waited);
    eventHandleEnd->setWaitMode(EventInterface::WaitMode::waited);
    return synSuccess;
}
