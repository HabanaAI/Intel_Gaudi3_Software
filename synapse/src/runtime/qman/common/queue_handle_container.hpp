#pragma once

#include <cstdint>
#include <deque>

#include "synapse_common_types.h"

class QueueInterface;
struct BasicQueueInfo;
struct InternalRecipeHandle;

class QueueInterfaceContainer
{
public:
    QueueInterfaceContainer() {}
    virtual ~QueueInterfaceContainer() {}

    bool addStreamHandle(QueueInterface* pQueueInterface);
    bool removeStreamHandle(QueueInterface* pQueueInterface);

    void      notifyRecipeRemoval(InternalRecipeHandle& rRecipeHandle);
    void      notifyAllRecipeRemoval();
    synStatus getDeviceTotalStreamMappedMemory(uint64_t& totalStreamMappedMemorySize) const;

    void                         clearAll() { m_streamHandles.clear(); }
    std::deque<QueueInterface*>& getQueueInterfaces() { return m_streamHandles; }

private:
    static bool isComputeStreamValid(const QueueInterface* pQueueInterface, const BasicQueueInfo& rBasicQueueInfo);

    std::deque<QueueInterface*> m_streamHandles;
};
