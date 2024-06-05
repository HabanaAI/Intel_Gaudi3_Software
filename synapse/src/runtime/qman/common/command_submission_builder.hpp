#pragma once

#include "synapse_types.h"
#include <deque>
#include <memory>

class CommandSubmissionBuilder
{
public:
    static const CommandSubmissionBuilder* getInstance();

    virtual ~CommandSubmissionBuilder();

    synStatus createAndAddBufferToCb(std::deque<void*>    hostBuffersPerCB,
                                     std::deque<uint64_t> commandBuffersSize,
                                     std::deque<uint32_t> queueIds,
                                     synCommandBuffer*    pSynCommandBuffers,
                                     uint32_t             numOfCb,
                                     uint16_t             cbOffset = 0) const;

    synStatus createAndAddHangCommandBuffer(synCommandBuffer*& pSynCB,
                                            uint32_t           cbIndex,
                                            const void*        pBuffer,
                                            uint64_t           commandBufferSize,
                                            uint32_t           queueId,
                                            bool               isForceMmuMapped = false) const;

    synStatus createSynCommandBuffer(synCommandBuffer* pSynCB,
                                     uint32_t          queueId,
                                     uint64_t          commandBufferSize) const;

    synStatus setBufferOnCb(synCommandBuffer& synCb,
                            unsigned          cbSize,
                            const void*       pBuffer,
                            uint64_t*         bufferOffset = nullptr) const;

    // Destroys a Syn Command-buffer (synCB is not deleted, only the CB it points to)
    synStatus destroyAllSynCommandBuffers(synCommandBuffer* synCBs, uint32_t numOfSynCBs) const;
    synStatus destroySynCommandBuffer(synCommandBuffer& synCB) const;

    void createInternalQueuesCB(CommandSubmission& commandSubmission, uint32_t numOfInternalQueues) const;

    bool addPqCmdForInternalQueue(CommandSubmission& commandSubmission,
                                  uint32_t           pqIndex,
                                  uint32_t           queueId,
                                  uint32_t           pqSize,
                                  uint64_t           pqAddress) const;

    synStatus destroyCmdSubmissionSynCBs(CommandSubmission& commandSubmission) const;

    void buildShellCommandSubmission(CommandSubmission*& pCommandSubmission,
                                     uint64_t            numOfPqsForInternalQueues,
                                     uint64_t            numOfPqsForExternalQueues) const;

private:
    CommandSubmissionBuilder();

    synStatus _createAndAddHangCommandBuffer(synCommandBuffer*& pSynCB,
                                             uint32_t           cbIndex,
                                             const void*        pBuffer,
                                             uint64_t           commandBufferSize,
                                             uint32_t           queueId,
                                             bool               isForceMmuMapped = false) const;

    synStatus _createSynCommandBuffer(synCommandBuffer*& pSynCB,
                                      uint32_t           queueId,
                                      uint64_t           commandBufferSize,
                                      bool               isForceMmuMapped = false) const;

    synStatus
    _setBufferOnCb(synCommandBuffer& synCb, unsigned cbSize, const void* pBuffer, uint64_t* bufferOffset) const;

    synStatus _destroyAllSynCommandBuffers(synCommandBuffer* synCBs, uint32_t numOfSynCBs) const;

    synStatus _destroySynCommandBuffer(synCommandBuffer& synCB) const;

    static std::shared_ptr<CommandSubmissionBuilder> m_pInstance;
};