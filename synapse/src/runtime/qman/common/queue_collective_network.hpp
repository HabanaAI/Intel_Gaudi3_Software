#pragma once

#include "queue_base_qman.hpp"

class QueueCollectiveNetwork : public QueueBaseQman
{
public:
    QueueCollectiveNetwork(const BasicQueueInfo&           rBasicQueueInfo,
                           uint32_t                        physicalQueueOffset,
                           synDeviceType                   deviceType,
                           PhysicalQueuesManagerInterface* pPhysicalStreamsManager);

    virtual ~QueueCollectiveNetwork();

    virtual synStatus getMappedMemorySize(uint64_t& mappedMemorySize) const override;

    virtual synStatus eventRecord(EventInterface& rEventInterface, synStreamHandle streamHandle) override;

    virtual synStatus
    eventWait(const EventInterface& rEventInterface, const unsigned int flags, synStreamHandle streamHandle) override;

    virtual synStatus synchronize(synStreamHandle streamHandle, bool isUserRequest) override;

    virtual synStatus memcopy(internalMemcopyParams& memcpyParams,
                              const internalDmaDir   direction,
                              bool                   isUserRequest,
                              QueueInterface*        pPreviousStream,
                              const uint64_t         overrideMemsetVal,
                              bool                   inspectCopiedContent,
                              SpRecipeProgramBuffer* pRecipeProgramBuffer,
                              uint8_t                apiId) override
    {
        return synFail;
    };

    virtual synStatus launch(const synLaunchTensorInfoExt* launchTensorsInfo,
                             uint32_t                      launchTensorsAmount,
                             uint64_t                      workspaceAddress,
                             InternalRecipeHandle*         pRecipeHandle,
                             uint64_t                      assertAsyncMappedAddress,
                             uint32_t                      flags,
                             EventWithMappedTensorDB&      events,
                             uint8_t                       apiId) override
    {
        return synFail;
    };

    virtual void finalize() override {};

    virtual void dfaInfo(DfaReq dfaReq, uint64_t csSeq) override;

    virtual synStatus getDynamicShapesTensorInfoArray(synRecipeHandle             recipeHandle,
                                                      std::vector<tensor_info_t>& tensorInfoArray) const override;
};
