#pragma once

#include "stream_base_scal.hpp"

class DevMemoryAllocInterface;

class QueueCopyScal : public QueueBaseScalCommon
{
public:
    QueueCopyScal(const BasicQueueInfo&    rBasicQueueInfo,
                  ScalStreamCopyInterface* pScalStream,
                  DevMemoryAllocInterface& rDevMemoryAlloc);

    virtual ~QueueCopyScal() = default;

    virtual synStatus getMappedMemorySize(uint64_t& mappedMemorySize) const override;

    virtual synStatus eventRecord(EventInterface& rEventInterface, synStreamHandle streamHandle) override;

    virtual synStatus
    eventWait(const EventInterface& rEventInterface, const unsigned int flags, synStreamHandle streamHandle) override;

    virtual synStatus query() override;

    virtual synStatus synchronize(synStreamHandle streamHandle, bool isUserRequest) override;

    synStatus memcopy(internalMemcopyParams& memcpyParams,
                      internalDmaDir         direction,
                      bool                   isUserRequest,
                      QueueInterface*        pPreviousStream,
                      const uint64_t         overrideMemsetVal,
                      bool                   inspectCopiedContent,
                      SpRecipeProgramBuffer* pRecipeProgramBuffer,
                      uint8_t                apiId) override;

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
    }

    virtual synStatus getDynamicShapesTensorInfoArray(synRecipeHandle             recipeHandle,
                                                      std::vector<tensor_info_t>& tensorInfoArray) const override;

private:
    static internalStreamType getInternalStreamType(internalDmaDir dir);

    virtual std::set<ScalStreamCopyInterface*> dfaGetQueueScalStreams() override { return { m_scalStream }; }

    DevMemoryAllocInterface& m_rDevMemoryAlloc;
};
