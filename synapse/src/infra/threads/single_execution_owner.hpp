#pragma once

#include <condition_variable>
#include <mutex>

// This class allows the user to request ownership over some execution,
// to allow (for example) that only one thread will execute it

// Usage example (ownershipId is an ID that uniquely describes the current execution-request):
//
// bool isExecutedSuccessfully = m_pSingleExecutionOwner->takeOwnership(ownershipId);
// if (!isExecutedSuccessfully)
// {
//     isExecutedSuccessfully = execution_function(ownershipId);
// }
// m_pSingleExecutionOwner->releaseOwnership(ownershipId);


class SingleExecutionOwner
{
    public:
        static const uint64_t INVALID_OWNERSHIP_ID;
        static const uint64_t FORCE_OPERATION_OWNERSHIP_ID;
        static const uint64_t ABORT_OPERATION_OWNERSHIP_ID;

        SingleExecutionOwner();

        ~SingleExecutionOwner() = default;

        // Will try to take ownership, only in case ownershipId parameter is the different than m_ownershipId
        // INVALID_OWNERSHIP_ID is forbidden
        //
        // Returns status - is ownership taken
        //
        // Motivation - In case one thread completed relevant-execution, one may want to "carry-on" and not re-execute
        bool takeOwnership(uint64_t ownershipId);

        // Will release ownership
        // m_ownershipId will be upadted according to the given parameter (unless it is equal to INVALID_OWNERSHIP_ID)
        void releaseOwnership(uint64_t newOwnershipId);


    private:
        // At the moment supporting single execution at a time (using boolean and not counter)
        bool                        m_isOwned;
        uint64_t                    m_ownershipId;

        std::mutex                  m_mutex;
        std::condition_variable     m_conditionVariable;
};