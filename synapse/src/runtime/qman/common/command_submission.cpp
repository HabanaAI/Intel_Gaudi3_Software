#include "command_submission.hpp"

#include "command_buffer.hpp"
#include "event_triggered_logger.hpp"
#include "string.h"

#include "synapse_runtime_logging.h"

#include "drm/habanalabs_accel.h"
#include "runtime/common/osal/osal.hpp"
#include "syn_singleton.hpp"
#include "profiler_api.hpp"

OffsetAndSize::OffsetAndSize()
{
    offset = 0;
    size   = 0;
}

StagedInfo::StagedInfo()
{
    isFirstSubmission = false;
    isLastSubmission  = false;
    hasWork           = false;
};

const unsigned MAX_SUBMISSION_PARTS = 2;

void StagedInfo::addDataChunkInfo(unsigned dcIndex, uint64_t offset, uint32_t size)
{
    HB_ASSERT(dcIndex < offsetSizeInDc.size(), "addDataChunkInfo invalid access");
    offsetSizeInDc[dcIndex].offset = offset;
    offsetSizeInDc[dcIndex].size   = size;
    if (size > 0)
    {
        hasWork = true;
    }
}

// This is a static method, so we will not need to include the habanalabs at the header, for it
static void addSinglePrimeQueueEntryIntoCsChunks(const PrimeQueueEntry& primeQueueEntry,
                                                 hl_cs_chunk*&          pCurrentChunkArgs,
                                                 uint32_t               queueOffset)
{
    pCurrentChunkArgs->cb_size     = primeQueueEntry.size;
    pCurrentChunkArgs->queue_index = primeQueueEntry.queueIndex + queueOffset;
    pCurrentChunkArgs->cb_handle   = primeQueueEntry.address;


    LOG_TRACE(SYN_CS,
              "    Fill PQ entry: handle 0x{:x}, size 0x{:x}, queue index {} queueOffset {}",
              pCurrentChunkArgs->cb_handle,
              pCurrentChunkArgs->cb_size,
              pCurrentChunkArgs->queue_index,
              queueOffset);

    pCurrentChunkArgs++;
}

static void addPrimeQueueEntriesIntoCsChunks(const PrimeQueueEntries& primeQueueEntries,
                                             hl_cs_chunk*&            pCurrentChunkArgs,
                                             uint32_t                 queueOffset)
{
    PrimeQueueEntriesConstIterator primeQueueEntriesIter    = primeQueueEntries.begin();
    PrimeQueueEntriesConstIterator primeQueueEntriesEndIter = primeQueueEntries.end();

    for (; primeQueueEntriesIter < primeQueueEntriesEndIter; primeQueueEntriesIter++)
    {
        addSinglePrimeQueueEntryIntoCsChunks(*primeQueueEntriesIter, pCurrentChunkArgs, queueOffset);
    }
}

CommandSubmission::CommandSubmission(bool isExternalCbRequired)
: m_executeExternalQueueCb(nullptr),
  m_executeInternalQueueCb(nullptr),
  m_numExecuteExternalCbs(0),
  m_numExecuteInternalCbs(0),
  m_firstStageCSHandle(0),
  m_calledFrom(nullptr),
  m_encapsHandleId(SIG_HANDLE_INVALID),
  m_isExternalCbRequired(isExternalCbRequired)
{
}

CommandSubmission::~CommandSubmission()
{
    if (m_executeExternalQueueCb != nullptr)
    {
        delete[] m_executeExternalQueueCb;
        m_executeExternalQueueCb = nullptr;
    }

    if (m_executeInternalQueueCb != nullptr)
    {
        delete[] m_executeInternalQueueCb;
        m_executeInternalQueueCb = nullptr;
    }
}

void CommandSubmission::addPrimeQueueEntry(ePrimeQueueEntryType primeQueueType,
                                           uint32_t             queueIndex,
                                           uint32_t             size,
                                           uint64_t             address)
{
    PrimeQueueEntries& pqEntries = _getPqEntries(primeQueueType);

    PrimeQueueEntry singlePqEntry = {queueIndex, size, address};

    pqEntries.push_back(singlePqEntry);
}

void CommandSubmission::clearPrimeQueueEntries(ePrimeQueueEntryType primeQueueType)
{
    PrimeQueueEntries& pqEntries = _getPqEntries(primeQueueType);

    pqEntries.clear();
}

void CommandSubmission::copyPrimeQueueEntries(ePrimeQueueEntryType primeQueueType, PrimeQueueEntries& primeQueueEntries)
{
    PrimeQueueEntries& pqEntries = _getPqEntries(primeQueueType);

    for (auto singlePqEntry : primeQueueEntries)
    {
        pqEntries.push_back(singlePqEntry);
    }
}

void CommandSubmission::getPrimeQueueEntries(ePrimeQueueEntryType      primeQueueType,
                                             const PrimeQueueEntries*& pprimeQueueEntries) const
{
    pprimeQueueEntries = &(_getPqEntries(primeQueueType));
}

uint32_t CommandSubmission::getPrimeQueueEntriesAmount(ePrimeQueueEntryType primeQueueType) const
{
    const PrimeQueueEntries& primeQueueEntries = _getPqEntries(primeQueueType);

    return primeQueueEntries.size();
}

synCommandBuffer* CommandSubmission::getExecuteExternalQueueCb() const
{
    return m_executeExternalQueueCb;
}

void CommandSubmission::setExecuteExternalQueueCb(synCommandBuffer* extQueueCb)
{
    m_executeExternalQueueCb = extQueueCb;
}

void CommandSubmission::copyExtQueueCb(synCommandBuffer* extQueueCb, uint32_t numOfCb)
{
    if (m_executeExternalQueueCb != nullptr)
    {
        LOG_WARN(SYN_CS, "{}: Overriding existing command buffers", HLLOG_FUNC);
        delete[] m_executeExternalQueueCb;
    }
    m_executeExternalQueueCb = new synCommandBuffer[numOfCb];
    memcpy(m_executeExternalQueueCb, extQueueCb, numOfCb * sizeof(synCommandBuffer));
}

const synInternalQueue* CommandSubmission::getExecuteInternalQueueCb() const
{
    return m_executeInternalQueueCb;
}

bool CommandSubmission::setExecuteIntQueueCbIndex(uint32_t index, uint32_t queueIndex, uint32_t size, uint64_t address)
{
    if (index >= m_numExecuteInternalCbs)
    {
        LOG_ERR(SYN_CS,
                "{}: Cannot add pqIndex {} due to array limitation of {}",
                HLLOG_FUNC,
                index,
                m_numExecuteInternalCbs);
        return false;
    }

    m_executeInternalQueueCb[index].queueIndex = queueIndex;
    m_executeInternalQueueCb[index].size       = size;
    m_executeInternalQueueCb[index].address    = address;

    return true;
}

void CommandSubmission::setExecuteInternalQueueCb(synInternalQueue* intQueueCb)
{
    m_executeInternalQueueCb = intQueueCb;
}

void CommandSubmission::copyIntQueueCb(synInternalQueue* intQueueCb, uint32_t numOfInternal)
{
    m_executeInternalQueueCb = new synInternalQueue[numOfInternal];
    memcpy(m_executeInternalQueueCb, intQueueCb, numOfInternal * sizeof(synInternalQueue));
}

uint32_t CommandSubmission::getNumExecuteExternalQueue() const
{
    return m_numExecuteExternalCbs;
}

void CommandSubmission::setNumExecuteExternalQueue(uint32_t numExtQueue)
{
    m_numExecuteExternalCbs = numExtQueue;
}

void CommandSubmission::setNumExecuteInternalQueue(uint32_t numExtQueue)
{
    m_numExecuteInternalCbs = numExtQueue;
}

uint32_t CommandSubmission::getNumExecuteInternalQueue() const
{
    return m_numExecuteInternalCbs;
}

std::mutex& CommandSubmission::getMutex()
{
    return m_commandSubmissionManagerMutex;
}

void CommandSubmission::CommandSubmission::dump() const
{
    LOG_INFO(SYN_CS, "m_executeInternalPqEntries.size() {}", m_executeInternalPqEntries.size());
    LOG_INFO(SYN_CS, "m_executeExternalPqEntries.size() {}", m_executeExternalPqEntries.size());
    LOG_INFO(SYN_CS, "m_executeExternalQueueCb {:x}", TO64(m_executeExternalQueueCb));
    LOG_INFO(SYN_CS, "m_executeInternalQueueCb {:x}", TO64(m_executeInternalQueueCb));

    for (uint32_t numExecuteInternalCbs = 0; numExecuteInternalCbs < m_numExecuteInternalCbs; numExecuteInternalCbs++)
    {
        synInternalQueue& executeInternalQueueCb = *m_executeInternalQueueCb;
        LOG_INFO(SYN_CS,
                 "m_executeInternalQueueCb {} queueIndex {} size {} address {:x}",
                 numExecuteInternalCbs,
                 executeInternalQueueCb.queueIndex,
                 executeInternalQueueCb.size,
                 TO64(executeInternalQueueCb.address));
    }

    LOG_INFO(SYN_CS, "m_numExecuteExternalCbs {}", m_numExecuteExternalCbs);
    LOG_INFO(SYN_CS, "m_numExecuteInternalCbs {}", m_numExecuteInternalCbs);

    LOG_INFO(SYN_CS, "m_firstStageCSHandle {}", m_firstStageCSHandle);

    LOG_INFO(SYN_CS, "m_calledFrom {:x}", TO64(m_calledFrom));
    LOG_INFO(SYN_CS, "m_encapsHandleId {}", m_encapsHandleId);

    LOG_INFO(SYN_CS, "m_isExternalCbRequired {}", m_isExternalCbRequired);
}

synStatus CommandSubmission::prepareForSubmission(void*&            pExecuteChunkArgs,
                                                  uint32_t&         executeChunksAmount,
                                                  bool              isRequireExternalChunk,
                                                  uint32_t          queueOffset,
                                                  const StagedInfo* pStagedInfo)
{
    synStatus status = synSuccess;
    bool isNotStaged = pStagedInfo == nullptr;

    status = _createExecuteChunks(pExecuteChunkArgs, executeChunksAmount, isRequireExternalChunk, pStagedInfo, queueOffset);
    if (status != synSuccess)
    {
        return status;
    }

    if (executeChunksAmount == 0)
    {
        delete[](hl_cs_chunk*) pExecuteChunkArgs;

        LOG_ERR(SYN_CS, "number of CB must be greater than 0");
        return synInvalidArgument;
    }

    // In case of a CS bigger than HL_MAX_JOBS_PER_CS we need to break it into two seperate CS as of LKD limitation.
    // In case of staged submission is not enabled we can't break it into 2, on this case fail the operation
    if (executeChunksAmount > HL_MAX_JOBS_PER_CS * MAX_SUBMISSION_PARTS ||
        (isNotStaged && (executeChunksAmount > HL_MAX_JOBS_PER_CS)))
    {
        delete[](hl_cs_chunk*) pExecuteChunkArgs;

        ETL_TRIGGER(EVENT_LOGGER_TRIGGER_TYPE_CS_ORDER);
        LOG_ERR(SYN_CS,
                "Failed to submit command buffer, too many chunks {}, max chunks supported is {}",
                executeChunksAmount,
                (uint64_t)HL_MAX_JOBS_PER_CS);

        return synCommandSubmissionFailure;
    }

    return synSuccess;
}

// Could be striped into a file that handles all hl-thunk calls
synStatus CommandSubmission::submitCommandBuffers(uint64_t*            csHandle,
                                                  uint64_t*            mappedBuff,
                                                  uint32_t             queueOffset,
                                                  const StagedInfo*    pStagedInfo,
                                                  globalStatPointsEnum point)
{
    LOG_TRACE(SYN_CS, "{} : called from {}", HLLOG_FUNC, getCalledFrom());

    synStatus status = synSuccess;

    void*    pExecuteChunkArgs   = nullptr;
    uint32_t executeChunksAmount = 0;

    synDeviceType deviceType             = OSAL::getInstance().getDeviceType();
    bool          isRequireExternalChunk = (deviceType == synDeviceGaudi);

    status = prepareForSubmission(pExecuteChunkArgs,
                                  executeChunksAmount,
                                  isRequireExternalChunk,
                                  queueOffset,
                                  pStagedInfo);
    if (status != synSuccess)
    {
        return status;
    }

    bool isExecuteChunkRequireBreak = executeChunksAmount > HL_MAX_JOBS_PER_CS;
    unsigned csBreakSize = isExecuteChunkRequireBreak ? MAX_SUBMISSION_PARTS : 1;

    struct hlthunk_cs_in  inArgs[MAX_SUBMISSION_PARTS];
    struct hlthunk_cs_out outArgs[MAX_SUBMISSION_PARTS];
    std::memset(&inArgs, 0, sizeof(hlthunk_cs_in) * csBreakSize);
    std::memset(&outArgs, 0, sizeof(hlthunk_cs_out) * csBreakSize);

    inArgs[0].chunks_restore = nullptr;
    inArgs[0].num_chunks_restore = 0;
    inArgs[0].chunks_execute = (void*)reinterpret_cast<__u64>(pExecuteChunkArgs);
    if (isExecuteChunkRequireBreak)
    {
        LOG_STG_INFO("CS seq {} is bigger than {}, breaking the CS into two CS's", m_firstStageCSHandle, (uint64_t)HL_MAX_JOBS_PER_CS);
        inArgs[0].num_chunks_execute = HL_MAX_JOBS_PER_CS;

        inArgs[1].chunks_restore = nullptr;
        inArgs[1].num_chunks_restore = 0;
        inArgs[1].chunks_execute = (void*)(reinterpret_cast<__u64>(pExecuteChunkArgs) + HL_MAX_JOBS_PER_CS);
        inArgs[1].num_chunks_execute = executeChunksAmount - HL_MAX_JOBS_PER_CS;

        if (pStagedInfo != nullptr && pStagedInfo->isFirstSubmission)
        {
            inArgs[0].flags |= HL_CS_FLAGS_STAGED_SUBMISSION_FIRST;
            inArgs[1].flags &= !HL_CS_FLAGS_STAGED_SUBMISSION_FIRST;
        }
        if (pStagedInfo != nullptr && pStagedInfo->isLastSubmission)
        {
            inArgs[0].flags &= !HL_CS_FLAGS_STAGED_SUBMISSION_LAST;
            inArgs[1].flags |= HL_CS_FLAGS_STAGED_SUBMISSION_LAST;
        }
    }
    else
    {
        inArgs[0].num_chunks_execute = executeChunksAmount;

        if (pStagedInfo != nullptr)
        {
            if (pStagedInfo->isFirstSubmission)
            {
                inArgs[0].flags |= HL_CS_FLAGS_STAGED_SUBMISSION_FIRST;
            }
            if (pStagedInfo->isLastSubmission)
            {
                inArgs[0].flags |= HL_CS_FLAGS_STAGED_SUBMISSION_LAST;
            }
        }
        else
        {
            inArgs[0].flags |= HL_CS_FLAGS_STAGED_SUBMISSION_FIRST;
            inArgs[0].flags |= HL_CS_FLAGS_STAGED_SUBMISSION_LAST;
        }
    }

    int ret;
    int fd = OSAL::getInstance().getFd();
    STAT_GLBL_START(commandSubmition);

#if 0  // keeping for now, for easy debug if needed
    for(int j = 0; j < csBreakSize; j++)
    {
        for(int i = 0; i < inArgs[j].num_chunks_execute; i++)
        {
            hl_cs_chunk* curr = (hl_cs_chunk*)(inArgs[j].chunks_execute);
            LOG_STG_INFO("{:x} handle 0x{:x} size 0x{:x}", i, curr[i].cb_handle, curr[i].cb_size);
        }
    }
#endif

    // In case of a CS bigger than HL_MAX_JOBS_PER_CS we iterate csBreakSize times and submit each CS independently
    for (unsigned i = 0; i < csBreakSize; i++)
    {
        bool isFirstSubmission  = ((inArgs[i].flags) & HL_CS_FLAGS_STAGED_SUBMISSION_FIRST) != 0;
        bool isLastSubmission   = ((inArgs[i].flags) & HL_CS_FLAGS_STAGED_SUBMISSION_LAST) != 0;

        if ((pStagedInfo == nullptr) || (isFirstSubmission && isLastSubmission))
        {
            if (m_encapsHandleId != SIG_HANDLE_INVALID)
            {
                inArgs[i].flags |= HL_CS_FLAGS_STAGED_SUBMISSION | HL_CS_FLAGS_ENCAP_SIGNALS;
                ret = hlthunk_staged_command_submission_encaps_signals(fd, m_encapsHandleId, &inArgs[i], &outArgs[i]);
            }
            else
            {
                ret = hlthunk_command_submission(fd, &inArgs[i], &outArgs[i]);
            }
            if (ret != 0)
            {
                LOG_ERR(SYN_CS, "{}: hlthunk_command_submission failed {} errno {}", HLLOG_FUNC, ret, errno);
                _SYN_SINGLETON_INTERNAL->notifyHlthunkFailure(DfaErrorCode::commandSubmissionFailed);
                return synFail;
            }
        }
        else
        {
            inArgs[i].flags |= HL_CS_FLAGS_STAGED_SUBMISSION;
            if (isFirstSubmission)
            {
                m_firstStageCSHandle = 0;
            }

            PROFILER_COLLECT_TIME()
            if (m_encapsHandleId != SIG_HANDLE_INVALID && isFirstSubmission)
            {
                inArgs[i].flags |= HL_CS_FLAGS_ENCAP_SIGNALS;
                ret = hlthunk_staged_command_submission_encaps_signals(fd, m_encapsHandleId, &inArgs[i], &outArgs[i]);
            }
            else
            {
                ret = hlthunk_staged_command_submission(fd, m_firstStageCSHandle, &inArgs[i], &outArgs[i]);
            }
            if (ret != 0)
            {
                LOG_ERR(SYN_CS, "{}: hlthunk_staged_command_submission failed {} errno {}", HLLOG_FUNC, ret, errno);
                _SYN_SINGLETON_INTERNAL->notifyHlthunkFailure(DfaErrorCode::stagedCsFailed);
                return synFail;
            }
            PROFILER_MEASURE_TIME("hlthunkStagedCS")
        }
    }

    if (point != globalStatPointsEnum::colLast || IS_VTUNE_ENABLED == 1)
    {
        STAT_GLBL_COLLECT_TIME(commandSubmition, point);
    }
#if 0
    for (unsigned i = 0; i < csBreakSize; i++)
    {
        LOG_STG_INFO("Submitted with seq {}, staged_enabled? {} flags 0x{:x} ret_val {} seq 0x{:x} first {} last {}",
                     m_firstStageCSHandle, m_staged.enable, inArgs[i].flags, ret, outArgs[i].seq, m_staged.isFirstSubmission, m_staged.isLastSubmission);
    }
#endif
    for (unsigned i = 0; i < csBreakSize; i++)
    {
        if (outArgs[i].status != HL_CS_STATUS_SUCCESS)
        {
            ETL_TRIGGER(EVENT_LOGGER_TRIGGER_TYPE_CS_ORDER);
            LOG_ERR(SYN_CS, "Failed to submit command buffer, ioctl failed");
            status = synCommandSubmissionFailure;
            break;
        }
    }

    if (status == synSuccess)
    {
        *csHandle = (uint64_t)outArgs[0].seq;
        delete[](hl_cs_chunk*) pExecuteChunkArgs;

        LOG_DEBUG(SYN_CS, "{}: hlthunk_staged_command_submission with handle {}", __FUNCTION__, *csHandle);
    }

    return status;
}

PrimeQueueEntries& CommandSubmission::_getPqEntries(ePrimeQueueEntryType primeQueueType)
{
    switch (primeQueueType)
    {
        case PQ_ENTRY_TYPE_INTERNAL_EXECUTION:
            return m_executeInternalPqEntries;

        case PQ_ENTRY_TYPE_EXTERNAL_EXECUTION:
            return m_executeExternalPqEntries;
    }

    return m_executeInternalPqEntries;
}

const PrimeQueueEntries& CommandSubmission::_getPqEntries(ePrimeQueueEntryType primeQueueType) const
{
    switch (primeQueueType)
    {
        case PQ_ENTRY_TYPE_INTERNAL_EXECUTION:
            return m_executeInternalPqEntries;

        case PQ_ENTRY_TYPE_EXTERNAL_EXECUTION:
            return m_executeExternalPqEntries;
    }

    return m_executeInternalPqEntries;
}

void CommandSubmission::setCalledFrom(const char* msg)
{
    m_calledFrom = msg;
}

const char* CommandSubmission::getCalledFrom()
{
    return ((m_calledFrom != nullptr) ? m_calledFrom : "");
}

synStatus CommandSubmission::_createExecuteChunks(void*&            pExecuteChunkArgs,
                                                  uint32_t&         executeChunksAmount,
                                                  bool              isRequireExternalChunk,
                                                  const StagedInfo* pStagedInfo,
                                                  uint32_t          queueOffset)
{
    synCommandBuffer* pCurrentExecuteSynCBs = getExecuteExternalQueueCb();

    uint32_t executeExternalQueuesCbSize = getNumExecuteExternalQueue();
    uint32_t executeInternalQueuesCbSize = getNumExecuteInternalQueue();

    uint32_t executeExternalPqEntryAmount = getPrimeQueueEntriesAmount(PQ_ENTRY_TYPE_EXTERNAL_EXECUTION);
    uint32_t executeInternalPqEntryAmount = getPrimeQueueEntriesAmount(PQ_ENTRY_TYPE_INTERNAL_EXECUTION);

    uint32_t executeCbSize = executeExternalQueuesCbSize + executeInternalQueuesCbSize + executeInternalPqEntryAmount +
                             executeExternalPqEntryAmount;

    pExecuteChunkArgs = nullptr;

    uint16_t numCBs   = 0;
    bool     isStaged = (pStagedInfo != nullptr);
    if (executeCbSize > 0)
    {
        if ((isRequireExternalChunk) && (executeExternalPqEntryAmount == 0) && (executeExternalQueuesCbSize == 0))
        {
            LOG_ERR(SYN_CS, "number of execution's external CBs must be greater than 0 if there is any execute ones");
            return synInvalidArgument;
        }

        if ((m_isExternalCbRequired) && ((executeInternalQueuesCbSize + executeExternalQueuesCbSize) > 0) &&
            (pCurrentExecuteSynCBs == nullptr))
        {
            LOG_ERR(SYN_CS, "Execute CBs are empty");
            return synInvalidArgument;
        }

        pExecuteChunkArgs = new hl_cs_chunk[executeCbSize];
        memset(pExecuteChunkArgs, 0, sizeof(hl_cs_chunk) * executeCbSize);

        hl_cs_chunk*             pCurrentExecuteChunkArgs = (hl_cs_chunk*)pExecuteChunkArgs;
        const PrimeQueueEntries* pprimeQueueEntries       = nullptr;

        LOG_TRACE(SYN_CS, "{}: Fill {} execute external CBs", HLLOG_FUNC, executeExternalQueuesCbSize);
        for (unsigned i = 0; i < executeExternalQueuesCbSize /*m_numExecuteExternalCbs*/;
             i++, pCurrentExecuteChunkArgs++, pCurrentExecuteSynCBs++ /*m_executeExternalQueueCb*/)
        {
            synCommandBuffer& currentSynCB  = *pCurrentExecuteSynCBs;
            CommandBuffer*    commandBuffer = reinterpret_cast<CommandBuffer*>(currentSynCB);

            if (commandBuffer == nullptr)
            {
                LOG_CRITICAL(SYN_CS, "External command buffer (index {}) should not be NULL", i);
                HB_ASSERT(0, "command buffer should not be NULL");
                delete[](hl_cs_chunk*) pExecuteChunkArgs;
                return synFail;
            }

            synStatus status = commandBuffer->FillCBChunk(*pCurrentExecuteChunkArgs /*hl_cs_chunk[i]*/, queueOffset);
            if (status != synSuccess)
            {
                LOG_ERR(SYN_CS, "Failed to fill Execute's CB-Chunk {}", i);
                delete[](hl_cs_chunk*) pExecuteChunkArgs;
                return synFail;
            }
        }

        if (!isStaged)
        {
            getPrimeQueueEntries(PQ_ENTRY_TYPE_EXTERNAL_EXECUTION, pprimeQueueEntries);
            HB_ASSERT((pprimeQueueEntries != nullptr), "Got nullptr for Execute-external PQ-entries DB");
            numCBs++;
            addPrimeQueueEntriesIntoCsChunks(*pprimeQueueEntries, pCurrentExecuteChunkArgs, queueOffset);
        }
        else
        {
            if (pStagedInfo->isFirstSubmission)
            {
                getPrimeQueueEntries(PQ_ENTRY_TYPE_EXTERNAL_EXECUTION, pprimeQueueEntries);
                HB_ASSERT(pprimeQueueEntries != nullptr, "Got nullptr for Execute-external PQ-entries DB");
                HB_ASSERT(pprimeQueueEntries->size() == 2, "Prime Queue Entries size must be 2. Fence clear + fence");
                numCBs++;
                addSinglePrimeQueueEntryIntoCsChunks(pprimeQueueEntries->at(0), pCurrentExecuteChunkArgs, queueOffset);
            }
            if (pStagedInfo->isLastSubmission)
            {
                getPrimeQueueEntries(PQ_ENTRY_TYPE_EXTERNAL_EXECUTION, pprimeQueueEntries);
                HB_ASSERT(pprimeQueueEntries != nullptr, "Got nullptr for Execute-external PQ-entries DB");
                HB_ASSERT(pprimeQueueEntries->size() == 2, "Prime Queue Entries size must be 2. Fence clear + fence");
                numCBs++;
                addSinglePrimeQueueEntryIntoCsChunks(pprimeQueueEntries->at(1), pCurrentExecuteChunkArgs, queueOffset);
            }
        }

        LOG_TRACE(SYN_CS, "{}: Fill {} internal CBs", HLLOG_FUNC, executeInternalQueuesCbSize);
        const synInternalQueue* pCurrentSynInternalQueueCb = getExecuteInternalQueueCb();
        for (unsigned i = 0; i < executeInternalQueuesCbSize; i++, pCurrentSynInternalQueueCb++)
        {
            if (!isStaged)
            {
                pCurrentExecuteChunkArgs->cb_size     = pCurrentSynInternalQueueCb->size;
                pCurrentExecuteChunkArgs->queue_index = pCurrentSynInternalQueueCb->queueIndex + queueOffset;
                pCurrentExecuteChunkArgs->cb_handle   = pCurrentSynInternalQueueCb->address;
            }
            else if (pStagedInfo->offsetSizeInDc[i].size != 0)
            {
                pCurrentExecuteChunkArgs->cb_size =
                    pStagedInfo->offsetSizeInDc[i].size;  //  pCurrentSynInternalQueueCb->size;
                pCurrentExecuteChunkArgs->queue_index = pCurrentSynInternalQueueCb->queueIndex + queueOffset;
                pCurrentExecuteChunkArgs->cb_handle =
                    pCurrentSynInternalQueueCb->address + pStagedInfo->offsetSizeInDc[i].offset;
            }
            else
            {
                continue;
            }
            LOG_TRACE(SYN_CS,
                      "    Fill CB 0x{:x} cb 0x{:x} handle 0x{:x}, size 0x{:x}, queue index {}",
                      i,
                      numCBs,
                      pCurrentExecuteChunkArgs->cb_handle,
                      pCurrentExecuteChunkArgs->cb_size,
                      pCurrentExecuteChunkArgs->queue_index);
            numCBs++;
            pCurrentExecuteChunkArgs++;
        }

        getPrimeQueueEntries(PQ_ENTRY_TYPE_INTERNAL_EXECUTION, pprimeQueueEntries);
        HB_ASSERT((pprimeQueueEntries != nullptr), "Got nullptr for Execute-internal PQ-entries DB");
        addPrimeQueueEntriesIntoCsChunks(*pprimeQueueEntries, pCurrentExecuteChunkArgs, queueOffset);
    }

    executeChunksAmount = isStaged ? numCBs : executeCbSize;
    return synSuccess;
}