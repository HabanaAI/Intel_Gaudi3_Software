#pragma once

#include "stream_base_scal.hpp"

class hclApiWrapper;

class QueueCollectiveNetworkScal : public QueueBaseScal
{
public:
    QueueCollectiveNetworkScal(const BasicQueueInfo& rBasicQueueInfo, hclApiWrapper& rHclApiWrapper);

    virtual ~QueueCollectiveNetworkScal() = default;

    virtual synStatus createHclStream() override;

    virtual synStatus destroyHclStream() override;

    virtual hcl::hclStreamHandle getHclStreamHandle() const override { return m_pHclStreamHandle; }

    virtual synStatus getMappedMemorySize(uint64_t& mappedMemorySize) const override;

    virtual synStatus eventRecord(EventInterface& rEventInterfac, synStreamHandle streamHandle) override;

    virtual synStatus
    eventWait(const EventInterface& rEventInterface, const unsigned int flags, synStreamHandle streamHandle) override;

    virtual synStatus eventQuery(const EventInterface& rEventInterface) override;

    virtual synStatus eventSynchronize(const EventInterface& rEventInterface) override;

    virtual synStatus query() override;

    virtual synStatus synchronize(synStreamHandle streamHandle, bool isUserRequest) override;

    virtual uint32_t getPhysicalQueueOffset() const override { return m_physicalQueueOffset; }

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

    void dfaInfo(DfaReq dfaReq, uint64_t csSeq) override;

    virtual synStatus getDynamicShapesTensorInfoArray(synRecipeHandle             recipeHandle,
                                                      std::vector<tensor_info_t>& tensorInfoArray) const override;

private:
    hclApiWrapper& m_rHclApiWrapper;

    hcl::hclStreamHandle m_pHclStreamHandle;

    uint32_t m_physicalQueueOffset;
};
