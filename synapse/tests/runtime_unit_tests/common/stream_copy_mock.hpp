#pragma once

#include "runtime/common/queues/queue_interface.hpp"
#include "runtime/common/queues/basic_queue_info.hpp"
#include <gtest/gtest.h>

class QueueMock : public QueueInterface
{
    FRIEND_TEST(UTStreamTest, check_synchronize_with_wait);

public:
    QueueMock();

    virtual ~QueueMock() override = default;

    virtual const BasicQueueInfo& getBasicQueueInfo() const override { return m_basicQueueInfo; }

    virtual uint32_t getPhysicalQueueOffset() const override { return 0; }

    virtual synStatus createHclStream() override { return synUnsupported; };

    virtual synStatus destroyHclStream() override { return synUnsupported; };

    virtual hcl::hclStreamHandle getHclStreamHandle() const override { return nullptr; }

    virtual synStatus getMappedMemorySize(uint64_t& mappedMemorySize) const override { return synSuccess; }

    virtual synStatus eventRecord(EventInterface& rEventInterface, synStreamHandle streamHandle) override
    {
        m_recordCounter++;
        return synSuccess;
    }

    virtual synStatus
    eventWait(const EventInterface& rEventInterface, const unsigned int flags, synStreamHandle streamHandle) override
    {
        m_waitCounter++;
        return synSuccess;
    }

    virtual synStatus query() override
    {
        m_queryCounter++;
        return synSuccess;
    }

    virtual synStatus synchronize(synStreamHandle streamHandle, bool isUserRequest) override
    {
        m_syncCounter++;
        return synSuccess;
    }

    virtual synStatus memcopy(internalMemcopyParams& memcpyParams,
                              const internalDmaDir   direction,
                              bool                   isUserRequest,
                              QueueInterface*        pPreviousStream,
                              const uint64_t         overrideMemsetVal,
                              bool                   inspectCopiedContent,
                              SpRecipeProgramBuffer* pRecipeProgramBuffer,
                              uint8_t                apiId) override
    {
        m_copyCounter++;
        m_lastMemcpyParams     = memcpyParams;
        m_lastDirection        = direction;
        m_lastIsUserRequest    = isUserRequest;
        m_pPreviousStream      = pPreviousStream;
        return synSuccess;
    }

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

    virtual void finalize() override {}
    virtual void dfaInfo(DfaReq dfaReq, uint64_t csSeq) override {}

    virtual synStatus getDynamicShapesTensorInfoArray(synRecipeHandle             recipeHandle,
                                                      std::vector<tensor_info_t>& tensorInfoArray) const override
    {
        return synFail;
    }

    const BasicQueueInfo        m_basicQueueInfo;
    uint64_t                    m_recordCounter;
    uint64_t                    m_waitCounter;
    uint64_t                    m_queryCounter;
    uint64_t                    m_syncCounter;
    uint64_t                    m_copyCounter;
    internalMemcopyParams       m_lastMemcpyParams;
    internalDmaDir              m_lastDirection;
    bool                        m_lastIsUserRequest;
    QueueInterface*             m_pPreviousStream;
};
