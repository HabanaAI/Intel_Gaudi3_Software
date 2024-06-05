#include "single_execution_owner.hpp"
#include "defs.h"

#include <limits>

const uint64_t SingleExecutionOwner::INVALID_OWNERSHIP_ID         = std::numeric_limits<uint64_t>::max();
const uint64_t SingleExecutionOwner::ABORT_OPERATION_OWNERSHIP_ID = SingleExecutionOwner::INVALID_OWNERSHIP_ID - 1;
const uint64_t SingleExecutionOwner::FORCE_OPERATION_OWNERSHIP_ID = SingleExecutionOwner::ABORT_OPERATION_OWNERSHIP_ID - 1;

static const uint64_t MINIMAL_RESERVED_OWNERSHIP_ID = SingleExecutionOwner::FORCE_OPERATION_OWNERSHIP_ID;

SingleExecutionOwner::SingleExecutionOwner() : m_isOwned(false), m_ownershipId(INVALID_OWNERSHIP_ID) {}

bool SingleExecutionOwner::takeOwnership(uint64_t ownershipId)
{
    // Invalid ID - up for the user to use a valid input
    // No reason for the overhead of Error-Handling
    HB_ASSERT((ownershipId <= ABORT_OPERATION_OWNERSHIP_ID), "takeOwnership: Invalid ownershipId");

    if ((ownershipId == m_ownershipId) &&
        (ownershipId != FORCE_OPERATION_OWNERSHIP_ID))
    {
        return false;
    }

    // Under mutex - take ownership
    std::unique_lock<std::mutex> mutex(m_mutex);

    // In case more than one thread awaits and are notified at once,
    //      one will get the lock and the ownership and then unlock the mutex
    //      the others will wait on that mutex and when released will see that it had been owned again,
    //          and return to wait using the cond-var
    while (m_isOwned)
    {
        m_conditionVariable.wait(mutex);
    }

    // In case no need to an ownership - operation completed - then just return gracefuly (w/o ownership)
    if ((ownershipId == m_ownershipId) &&
        (ownershipId != FORCE_OPERATION_OWNERSHIP_ID))
    {
        // No need to take ownership - nothing changed
        return false;
    }

    m_isOwned = true;
    return true;
}

void SingleExecutionOwner::releaseOwnership(uint64_t newOwnershipId)
{
    HB_ASSERT((newOwnershipId != INVALID_OWNERSHIP_ID), "takeOwnership: Invalid newOwnershipId");

    std::unique_lock<std::mutex> mutex(m_mutex);

    if (newOwnershipId != ABORT_OPERATION_OWNERSHIP_ID)
    {
        m_ownershipId = newOwnershipId;
    }

    m_isOwned = false;

    m_conditionVariable.notify_all();
}
