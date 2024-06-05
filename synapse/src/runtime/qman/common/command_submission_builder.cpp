#include "command_submission_builder.hpp"

#include "command_buffer.hpp"
#include "command_submission.hpp"
#include "synapse_runtime_logging.h"
#include "profiler_api.hpp"

#define VERIFY_IS_NULL_POINTER(pointer, name)                                                                          \
    if (pointer == nullptr)                                                                                            \
    {                                                                                                                  \
        LOG_WARN(SYN_CS, "{}: got null pointer for {} ", HLLOG_FUNC, name);                                            \
        return synInvalidArgument;                                                                                     \
    }

std::shared_ptr<CommandSubmissionBuilder> CommandSubmissionBuilder::m_pInstance = nullptr;

const CommandSubmissionBuilder* CommandSubmissionBuilder::getInstance()
{
    if (m_pInstance == nullptr)
    {
        m_pInstance.reset(new CommandSubmissionBuilder());
    }

    return m_pInstance.get();
}

CommandSubmissionBuilder::CommandSubmissionBuilder() {}

CommandSubmissionBuilder::~CommandSubmissionBuilder() {}

synStatus CommandSubmissionBuilder::createAndAddBufferToCb(std::deque<void*>    hostBuffersPerCB,
                                                           std::deque<uint64_t> commandBuffersSize,
                                                           std::deque<uint32_t> queueIds,
                                                           synCommandBuffer*    pSynCommandBuffers,
                                                           uint32_t             numOfCb,
                                                           uint16_t             cbOffset /* = 0 */) const
{
    if (numOfCb != hostBuffersPerCB.size())
    {
        LOG_ERR(SYN_CS,
                "{}: Size-mismatch numOfCb {} hostBuffersPerCB {}",
                HLLOG_FUNC,
                numOfCb,
                hostBuffersPerCB.size());
        return synInvalidArgument;
    }
    if (numOfCb != commandBuffersSize.size())
    {
        LOG_ERR(SYN_CS,
                "{}: Size-mismatch numOfCb {} commandBuffersSize {}",
                HLLOG_FUNC,
                numOfCb,
                commandBuffersSize.size());
        return synInvalidArgument;
    }
    if (numOfCb != queueIds.size())
    {
        LOG_ERR(SYN_CS, "{}: Size-mismatch numOfCb {} queueIds {}", HLLOG_FUNC, numOfCb, queueIds.size());
        return synInvalidArgument;
    }

    synStatus status      = synSuccess;
    uint32_t  cbIndex     = cbOffset;
    uint32_t  lastCbIndex = cbOffset + numOfCb;

    synCommandBuffer* pCurrSynCb = &pSynCommandBuffers[cbIndex];

    auto hostBufferIter        = hostBuffersPerCB.begin();
    auto commandBufferSizeIter = commandBuffersSize.begin();
    auto queueIdsIter          = queueIds.begin();

    for (; cbIndex < lastCbIndex; cbIndex++, pCurrSynCb++, hostBufferIter++, commandBufferSizeIter++, queueIdsIter++)
    {
        if (*commandBufferSizeIter == 0)
        {
            LOG_ERR(SYN_CS, "Zero command buffer size for CB {}", cbIndex);
            return synFail;
        }

        LOG_DEBUG(SYN_CS, "CB num {} is the size of buffer {}", cbIndex, *commandBufferSizeIter);

        status = _createAndAddHangCommandBuffer(pCurrSynCb,
                                                cbIndex,
                                                *hostBufferIter,
                                                *commandBufferSizeIter,
                                                *queueIdsIter);
        if (status != synSuccess)
        {
            break;
        }
    }

    if (status != synSuccess)
    {
        _destroyAllSynCommandBuffers(pSynCommandBuffers, cbIndex);
    }

    return status;
}

synStatus CommandSubmissionBuilder::createAndAddHangCommandBuffer(synCommandBuffer*& pSynCB,
                                                                  uint32_t           cbIndex,
                                                                  const void*        pBuffer,
                                                                  uint64_t           commandBufferSize,
                                                                  uint32_t           queueId,
                                                                  bool               isForceMmuMapped) const
{
    return _createAndAddHangCommandBuffer(pSynCB,
                                          cbIndex,
                                          pBuffer,
                                          commandBufferSize,
                                          queueId,
                                          isForceMmuMapped);
}

synStatus CommandSubmissionBuilder::createSynCommandBuffer(synCommandBuffer* pSynCB,
                                                           uint32_t          queueId,
                                                           uint64_t          commandBufferSize) const
{
    return _createSynCommandBuffer(pSynCB, queueId, commandBufferSize);
}

synStatus CommandSubmissionBuilder::setBufferOnCb(synCommandBuffer& synCb,
                                                  unsigned          cbSize,
                                                  const void*       pBuffer,
                                                  uint64_t*         bufferOffset /*= nullptr*/) const
{
    return _setBufferOnCb(synCb, cbSize, pBuffer, bufferOffset);
}

synStatus CommandSubmissionBuilder::destroyAllSynCommandBuffers(synCommandBuffer* synCBs, uint32_t numOfSynCBs) const
{
    return _destroyAllSynCommandBuffers(synCBs, numOfSynCBs);
}

synStatus CommandSubmissionBuilder::destroySynCommandBuffer(synCommandBuffer& synCB) const
{
    return _destroySynCommandBuffer(synCB);
}

void CommandSubmissionBuilder::createInternalQueuesCB(CommandSubmission& commandSubmission,
                                                      uint32_t           numOfInternalQueues) const
{
    commandSubmission.setNumExecuteInternalQueue(numOfInternalQueues);
    synInternalQueue* intQueueCb = new synInternalQueue[numOfInternalQueues];
    memset(intQueueCb, 0, sizeof(synInternalQueue) * numOfInternalQueues);
    commandSubmission.setExecuteInternalQueueCb(intQueueCb);
}

bool CommandSubmissionBuilder::addPqCmdForInternalQueue(CommandSubmission& commandSubmission,
                                                        uint32_t           pqIndex,
                                                        uint32_t           queueId,
                                                        uint32_t           pqSize,
                                                        uint64_t           pqAddress) const
{
    return commandSubmission.setExecuteIntQueueCbIndex(pqIndex, queueId, pqSize, pqAddress);
}

synStatus CommandSubmissionBuilder::destroyCmdSubmissionSynCBs(CommandSubmission& commandSubmission) const
{
    synStatus status = synSuccess;
    std::unique_lock<std::mutex> guard(commandSubmission.getMutex());

    uint32_t          numOfSynCbs = commandSubmission.getNumExecuteExternalQueue();
    synCommandBuffer* pSynCbs     = commandSubmission.getExecuteExternalQueueCb();
    if ((numOfSynCbs != 0) && (pSynCbs != nullptr))
    {
        synCommandBuffer* pCurrSynCB = pSynCbs;
        for (uint32_t cbIndex = 0; cbIndex < commandSubmission.getNumExecuteExternalQueue(); cbIndex++, pCurrSynCB++)
        {
            status = _destroySynCommandBuffer(*pCurrSynCB);
            if (status != synSuccess)
            {
                status = synFail;
            }
        }

        delete[] pSynCbs;
        pSynCbs = nullptr;
        commandSubmission.setExecuteExternalQueueCb(nullptr);
        commandSubmission.setNumExecuteExternalQueue(0);
    }
    else
    {
        LOG_TRACE(SYN_CS, "{}: No Execute CB on CS", HLLOG_FUNC);
    }

    return status;
}

void CommandSubmissionBuilder::buildShellCommandSubmission(CommandSubmission*& pCommandSubmission,
                                                           uint64_t            numOfPqsForInternalQueues,
                                                           uint64_t            numOfPqsForExternalQueues) const
{
    PROFILER_COLLECT_TIME()

    pCommandSubmission = new CommandSubmission(numOfPqsForExternalQueues != 0);

    const CommandSubmissionBuilder* pCmdSubmissionBuilder = CommandSubmissionBuilder::getInstance();

    CommandSubmission& commandSubmission = *pCommandSubmission;

    if (numOfPqsForExternalQueues != 0)
    {
        synCommandBuffer* pSynCommandBuffers = new synCommandBuffer[numOfPqsForExternalQueues];
        commandSubmission.setExecuteExternalQueueCb(pSynCommandBuffers);
    }
    commandSubmission.setNumExecuteExternalQueue(numOfPqsForExternalQueues);

    pCmdSubmissionBuilder->createInternalQueuesCB(commandSubmission, numOfPqsForInternalQueues);

    if (GCFG_ENABLE_SYN_LAUNCH_PROFILER_ANNOTATIONS.value())
    {
        char desc[50] = {};
        snprintf(desc,
                 sizeof(desc),
                 "%s numOfPqs=%lu,%lu",
                 "buildShellCS",
                 numOfPqsForInternalQueues,
                 numOfPqsForExternalQueues);
        PROFILER_MEASURE_TIME(desc)
    }
}

synStatus CommandSubmissionBuilder::_createAndAddHangCommandBuffer(synCommandBuffer*& pSynCB,
                                                                   uint32_t           cbIndex,
                                                                   const void*        pBuffer,
                                                                   uint64_t           commandBufferSize,
                                                                   uint32_t           queueId,
                                                                   bool               isForceMmuMapped) const
{
    VERIFY_IS_NULL_POINTER(pBuffer, "buffer");
    VERIFY_IS_NULL_POINTER(pSynCB, "Syn CB");

    if (commandBufferSize == 0)
    {
        LOG_ERR(SYN_CS, "Zero command buffer size for CB {}", cbIndex);
        return synFail;
    }

    // Create a command buffer
    synStatus status = _createSynCommandBuffer(pSynCB, queueId, commandBufferSize, isForceMmuMapped);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_CS, "Can not create command buffer {}", cbIndex);
        return status;
    }

    // Hang the buffer on the cb
    status = _setBufferOnCb(*pSynCB, commandBufferSize, (void*)pBuffer, nullptr);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_CS, "Failed to set buffer on {} command buffer", cbIndex);

        status = _destroySynCommandBuffer(*pSynCB);
        if (status != synSuccess)
        {
            LOG_ERR(SYN_CS, "Failed to unset buffer of command-buffer {}", cbIndex);
        }
    }

    return status;
}

synStatus CommandSubmissionBuilder::_createSynCommandBuffer(synCommandBuffer*& pSynCB,
                                                            uint32_t           queueId,
                                                            uint64_t           commandBufferSize,
                                                            bool               isForceMmuMapped) const
{
    VERIFY_IS_NULL_POINTER(pSynCB, "pointer to syn-command-buffer");

    CommandBuffer* pBuffer = nullptr;
    synStatus      status =
        CommandBufferMap::GetInstance()->AddCommandBuffer(commandBufferSize, &pBuffer, isForceMmuMapped);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_CS, "Can not create command buffer");
        return status;
    }

    pBuffer->SetQueueIndex(queueId);

    LOG_DEBUG(SYN_CS, "Created command buffer successfully");
    *pSynCB = reinterpret_cast<synCommandBuffer>(pBuffer);

    return synSuccess;
}

synStatus CommandSubmissionBuilder::_setBufferOnCb(synCommandBuffer& synCB,
                                                   unsigned          cbSize,
                                                   const void*       pBuffer,
                                                   uint64_t*         bufferOffset) const
{
    VERIFY_IS_NULL_POINTER(synCB, "syn-command-buffer");

    CommandBuffer* pCommandBuffer = reinterpret_cast<CommandBuffer*>(synCB);

    return pCommandBuffer->SetBufferToCB(pBuffer, cbSize, bufferOffset);
}

synStatus CommandSubmissionBuilder::_destroyAllSynCommandBuffers(synCommandBuffer* synCBs, uint32_t numOfSynCBs) const
{
    synStatus         status     = synSuccess;
    synCommandBuffer* pCurrSynCB = synCBs;

    if (numOfSynCBs > 0)
    {
        VERIFY_IS_NULL_POINTER(pCurrSynCB, "syn-command-buffer");
    }

    for (uint32_t cbIndex = 0; cbIndex < numOfSynCBs; cbIndex++, pCurrSynCB++)
    {
        if (_destroySynCommandBuffer(*pCurrSynCB) != synSuccess)
        {
            LOG_ERR(SYN_CS, "{}: Failed to destroy syn-command-buffer {}", HLLOG_FUNC, cbIndex);
            status = synFail;
        }
    }

    return status;
}

synStatus CommandSubmissionBuilder::_destroySynCommandBuffer(synCommandBuffer& synCB) const
{
    VERIFY_IS_NULL_POINTER(synCB, "syn-command-buffer");

    CommandBuffer* pCommandBuffer = reinterpret_cast<CommandBuffer*>(synCB);

    return CommandBufferMap::GetInstance()->RemoveCommandBuffer(pCommandBuffer);
}
