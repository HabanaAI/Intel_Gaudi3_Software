/*
.
1) This class is used for Compute and Memcopy operations
.
.
2) Usage of DCs (DC = Data Chunks) (execution operation) -
    a) Memcopy operation:
    We need to acquire CB DCs per execution (CB = Command Buffer)
.
    b) Compute operation:
    We need to acquire MMU DCs per launch and CB DCs statically (done during static processing, and using the CB DC
allocator of the Recipe-Singleton
.
.
3) Release of DCs due to a CS DC release -
    a) Memcopy operation:
    We need to clear the CB DCs
.
b) Compute operation:
    We need to clear the MMU DCs
    [The CB DCs are static, as defined above]
*/
#pragma once

#include "synapse_api_types.h"
#include "synapse_common_types.h"
#include "define_synapse_common.hpp"
#include "runtime/qman/common/qman_types.hpp"
#include "runtime/common/queues/queue_base.hpp"

class Stream;
class IDynamicInfoProcessor;
class MemoryManager;
class CommandSubmissionDataChunks;
class PhysicalQueuesManager;
class DeviceRecipeDownloaderInterface;

struct basicRecipeInfo;
struct tensor_info_t;
class EventInterface;

class PhysicalQueuesManagerInterface;

class QueueBaseQman : public QueueBase
{
public:
    QueueBaseQman(const BasicQueueInfo&           rBasicQueueInfo,
                  uint32_t                        physicalQueueOffset,
                  synDeviceType                   deviceType,
                  PhysicalQueuesManagerInterface* pPhysicalStreamsManager);

    virtual ~QueueBaseQman() = default;

    virtual uint32_t getPhysicalQueueOffset() const override { return m_physicalQueueOffset; }

    virtual synStatus
    eventWait(const EventInterface& rEventInterface, const unsigned int flags, synStreamHandle streamHandle) override;

    virtual synStatus query() override;

protected:
    synStatus performStreamsSynchronization(QueueInterface& rPrecedingStreamInterface, bool isUser);

    const uint32_t m_physicalQueueOffset;

    const synDeviceType m_deviceType;

    PhysicalQueuesManagerInterface* m_pPhysicalStreamsManager;
};
