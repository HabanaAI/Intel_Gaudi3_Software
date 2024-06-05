#pragma once

#include <mutex>
#include <map>
#include <unordered_map>

#include "synapse_types.h"
#include "runtime/common/common_types.hpp"
#include "runtime/qman/common/qman_types.hpp"

class TrainingQueueInfo;

class QueueInfo
{
public:
    QueueInfo(uint64_t queueId, uint32_t physicalQueueOffset, PhysicalQueuesId physicalQueuesId, const uint32_t flags);

    QueueInfo(const QueueInfo& queueInfo) = delete;
    virtual QueueInfo& operator=(const QueueInfo&) = delete;

    TrainingRetCode updateStatus(const InternalWaitHandle& waitForEventHandle, const std::string& desc);

    TrainingRetCode getLastWaitHandles(InternalWaitHandlesVector& lastWaitHandles) const;

    uint32_t getPhysicalQueueOffset() const;

    uint64_t getQueueId() const;

    const PhysicalQueuesId& getPhysicalQueuesId() const;

    void operationCompletion(bool               isSignalOperation,
                             uint64_t           operationSequenceId,
                             bool               updateWaitHandle,
                             bool               debugCSMode,
                             const std::string& desc);

    bool isLastOperationSignaled(uint64_t& lastOperationSeqId);

    void printCSDB() const;

private:
    uint64_t           m_queueId;
    uint32_t           m_physicalQueueOffset;
    PhysicalQueuesId   m_physicalQueuesId;
    InternalWaitHandle m_internalWaitHandle;

    mutable std::mutex m_mutex;

    struct CsInfo
    {
        uint64_t threadId;
        bool     isSignal;
    };
    std::map<uint64_t, CsInfo> m_csTypes;
};
