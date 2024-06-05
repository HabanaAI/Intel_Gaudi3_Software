#include "queue_base_qman_wcm.hpp"

#include "defenders.h"
#include "event_triggered_logger.hpp"
#include "habana_global_conf_runtime.h"
#include "habana_global_conf.h"
#include "internal/hccl_internal.h"
#include "physical_queues_manager.hpp"
#include "queue_info.hpp"
#include "runtime/qman/common/address_range_mapper.hpp"
#include "runtime/qman/common/command_buffer.hpp"
#include "runtime/qman/common/command_submission_data_chunks.hpp"
#include "runtime/qman/common/command_submission.hpp"
#include "runtime/qman/common/data_chunk/data_chunk.hpp"
#include "runtime/qman/common/inflight_cs_parser.hpp"
#include "runtime/qman/common/qman_event.hpp"
#include "runtime/qman/common/wcm/work_completion_manager_interface.hpp"
#include "runtime/qman/gaudi/parser/define.hpp"
#include "runtime/qman/gaudi/recipe_parser.hpp"
#include "syn_singleton.hpp"
#include "types_exception.h"

#include <memory>

// Stream WCM CS query timeout in seconds. 0 duration means blocking query
#define STREAM_WCM_CS_QUERY_TIMEOUT 120;

void addPqEntiesInfoToParserHelper(InflightCsParserHelper&  parserHelper,
                                   const CommandSubmission* pCommandSubmission,
                                   ePrimeQueueEntryType     pqEntryType)
{
    HB_ASSERT_PTR(pCommandSubmission);

    const PrimeQueueEntries* pPrimeQueueEntries;
    pCommandSubmission->getPrimeQueueEntries(pqEntryType, pPrimeQueueEntries);

    switch (pqEntryType)
    {
        case PQ_ENTRY_TYPE_INTERNAL_EXECUTION:
        case PQ_ENTRY_TYPE_EXTERNAL_EXECUTION:
            for (auto singlePqEntry : *pPrimeQueueEntries)
            {
                parserHelper.addParserPqEntry(singlePqEntry.queueIndex,
                                              singlePqEntry.address,
                                              singlePqEntry.size);
            }
            break;
    }
}

void addPqEntiesInfoToParserHelper(InflightCsParserHelper& parserHelper, const CommandSubmission* pCommandSubmission)
{
    addPqEntiesInfoToParserHelper(parserHelper, pCommandSubmission, PQ_ENTRY_TYPE_INTERNAL_EXECUTION);
    addPqEntiesInfoToParserHelper(parserHelper, pCommandSubmission, PQ_ENTRY_TYPE_EXTERNAL_EXECUTION);
}

QueueBaseQmanWcm::QueueBaseQmanWcm(const BasicQueueInfo&           rBasicQueueInfo,
                                   uint32_t                        physicalQueueOffset,
                                   synDeviceType                   deviceType,
                                   PhysicalQueuesManagerInterface* pPhysicalStreamsManager,
                                   WorkCompletionManagerInterface& rWorkCompletionManager)
: QueueBaseQman(rBasicQueueInfo, physicalQueueOffset, deviceType, pPhysicalStreamsManager),
  m_rWorkCompletionManager(rWorkCompletionManager),
  m_waitForAllCsdcs(false),
  m_waitForInternalRecipeHandle(nullptr)
{
    if (pPhysicalStreamsManager == nullptr)
    {
        throw SynapseException("QueueBaseQmanWcm: initialized with nullptr TSM");
    }
}

// Updates upon GCFGs tuneables - Is not in use due to SW-69371
void QueueBaseQmanWcm::updateDataChunkCacheSize(uint64_t& maximalCacheAmount,
                                                uint64_t& minimalCacheAmount,
                                                uint64_t& maximalFreeCacheAmount,
                                                bool      ignoreInitialCacheSizeConfiguration)
{
    // Data-Chunks-Cache Breathing | Minimal-Cache-Size | Cache capabilities
    // ----------------------------|--------------------|-----------------------------------------------------------
    //           Allowed           |      Allowed       | Cache breaths from defined minimal cache-size
    //         Not Allowed         |      Allowed       | Cache-size can only grow (from pre-defined minimal-size)
    //           Allowed           |    Not Allowed     | Cache breaths from zero cache-size
    //         Not Allowed         |    Not Allowed     | Cache-size is fixed, upon pre-defined max-size
    // * For Compute we will ignore the Minimal-Cache-Size configuration
    // No limitation on free cache-elements (cache cannot shrink)
    maximalFreeCacheAmount = maximalCacheAmount;

    if (!ignoreInitialCacheSizeConfiguration)
    {
        // Initial Cache-size
        minimalCacheAmount = 0;
    }
}

synStatus QueueBaseQmanWcm::eventRecord(EventInterface& rEventInterface, synStreamHandle streamHandle)
{
    QmanEvent&      rEventHandle    = dynamic_cast<QmanEvent&>(rEventInterface);
    TrainingRetCode trainingRetCode = m_pPhysicalStreamsManager->signalEvent(m_basicQueueInfo, rEventHandle);
    if ((trainingRetCode != TRAINING_RET_CODE_SUCCESS) && (trainingRetCode != TRAINING_RET_CODE_NO_CHANGE))
    {
        LOG_ERR(SYN_STREAM, "{}: Operation failed on stream {}", HLLOG_FUNC, m_basicQueueInfo.getDescription());

        return synFailedToSubmitWorkload;
    }

#ifndef _POWER_PC_
    rEventHandle.handleRequest = hcclEventHandle();
#endif

    return synSuccess;
}

synStatus QueueBaseQmanWcm::synchronize(synStreamHandle streamHandle, bool isUserRequest)
{
    if (m_pPhysicalStreamsManager == nullptr)
    {
        LOG_ERR(SYN_STREAM, "{}: no instance of TSM", HLLOG_FUNC);
        return synUnsupported;
    }

    InternalWaitHandlesVector streamWaitHandles;
    TrainingRetCode retCode = m_pPhysicalStreamsManager->getLastWaitHandles(m_basicQueueInfo, streamWaitHandles);
    if (retCode != TRAINING_RET_CODE_SUCCESS)
    {
        LOG_ERR(SYN_STREAM, "{}: Can not get wait-handle of {}", HLLOG_FUNC, m_basicQueueInfo.getDescription());
        return synFail;
    }

    ETL_PRE_OPERATION_NEW_ID(logId, EVENT_LOGGER_LOG_TYPE_CS_ORDER);

    synStatus status =
        _SYN_SINGLETON_INTERNAL->waitAndReleaseStreamHandles(streamWaitHandles, SYNAPSE_WAIT_FOR_CS_DEFAULT_TIMEOUT);

    ETL_ADD_LOG_T_DEBUG(EVENT_LOGGER_LOG_TYPE_CS_ORDER,
                        logId,
                        SYN_STREAM,
                        "{}: Synchronized {}",
                        HLLOG_FUNC,
                        m_basicQueueInfo.getDescription());

    return status;
}

void QueueBaseQmanWcm::_waitForWCM(bool queryAllCsdcs, InternalRecipeHandle* pRecipeHandle)
{
    const uint64_t               csQuerytimeoutSec = STREAM_WCM_CS_QUERY_TIMEOUT;
    std::unique_lock<std::mutex> mutex_cv(m_condMutex);
    const uint64_t               csInflight = getCsDcDataBaseSize();
    if (csInflight == 0)
    {
        return;
    }
    if (pRecipeHandle != nullptr && !isRecipeHasInflightCsdc(pRecipeHandle))
    {
        return;
    }
    m_cvFlag                      = false;
    m_waitForAllCsdcs             = queryAllCsdcs;
    m_waitForInternalRecipeHandle = pRecipeHandle;
    LOG_DEBUG(SYN_STREAM,
              "{}: 0x{:x} waiting queryAllCsdcs {} recipeId 0x{:x} csInflight {}",
              HLLOG_FUNC,
              TO64(this),
              queryAllCsdcs,
              TO64(pRecipeHandle),
              csInflight);
    STAT_GLBL_START(wcmObserverWaitDuration);
    bool waitSuccess;
    if (csQuerytimeoutSec == 0)
    {
        m_condVar.wait(mutex_cv, [&] { return m_cvFlag; });
        waitSuccess = true;
    }
    else
    {
        waitSuccess = m_condVar.wait_for(mutex_cv, std::chrono::seconds(csQuerytimeoutSec), [&] { return m_cvFlag; });
    }
    STAT_GLBL_COLLECT_TIME(wcmObserverWaitDuration, globalStatPointsEnum::wcmObserverWaitDuration);
    if (waitSuccess)
    {
        LOG_INFO(SYN_STREAM, "{}: 0x{:x} waiting done successfully ", HLLOG_FUNC, TO64(this));
    }
    else
    {
        LOG_CRITICAL(SYN_STREAM, "{}: 0x{:x} waiting done failure", HLLOG_FUNC, TO64(this));
    }
}

void QueueBaseQmanWcm::_wcmReleaseThreadIfNeeded()
{
    std::unique_lock<std::mutex> mutex_cv(m_condMutex);
    LOG_DEBUG(SYN_STREAM, "{}", HLLOG_FUNC);
    if (m_waitForAllCsdcs && getCsDcDataBaseSize() != 0)
    {
        LOG_DEBUG(SYN_STREAM, "{}, ret", HLLOG_FUNC);
        return;
    }

    if (m_waitForInternalRecipeHandle != nullptr)
    {
        if (isRecipeHasInflightCsdc(m_waitForInternalRecipeHandle))
        {
            LOG_DEBUG(SYN_STREAM, "{}, ret, recipe destroy", HLLOG_FUNC);
            return;
        }
        else
        {
            m_waitForInternalRecipeHandle = nullptr;
        }
    }
    m_cvFlag          = true;
    m_waitForAllCsdcs = false;
    m_condVar.notify_one();
}

void QueueBaseQmanWcm::finalize()
{
    _waitForWCM(true /* waitForAllCsdcs*/);
}

uint64_t QueueBaseQmanWcm::getCsDcDataBaseSize() const
{
    STAT_GLBL_START(wcmObserverDbMutexDuration);
    std::unique_lock<std::mutex> mtx(m_DBMutex);
    STAT_GLBL_COLLECT_TIME(wcmObserverDbMutexDuration, globalStatPointsEnum::wcmObserverDbMutexDuration);
    return m_csDataChunksDb.size();
}

bool QueueBaseQmanWcm::parseCsDc(uint64_t suspectedCsHandle)
{
    std::unique_lock<std::mutex> mtx(m_DBMutex);

    CommandSubmissionDataChunks* csdc = nullptr;

    if (suspectedCsHandle == 0)  // just take the oldest one
    {
        if (m_csDataChunksDb.empty())  // if empty, nothing to do
        {
            LOG_INFO(SYN_DEV_FAIL, "csdsDb empty");
            return true;
        }

        std::string msg = "csSeq in csdc: ";
        csdc            = m_csDataChunksDb.front();
        auto all        = csdc->getAllWaitForEventHandles();

        for (auto seq : all)
        {
            msg += fmt::format("{} ", seq);
        }
        LOG_INFO(SYN_DEV_FAIL, "given csHandle is 0, using oldest csdc: {}", msg);
    }
    else
    {
        bool found = false;
        for (auto pCsDc : m_csDataChunksDb)
        {
            auto        all = pCsDc->getAllWaitForEventHandles();
            std::string msg = "csSeq in csdc: ";
            for (auto seq : all)
            {
                if (seq == suspectedCsHandle)
                {
                    found = true;
                    csdc  = pCsDc;
                    LOG_INFO(SYN_DEV_FAIL, "Found cs {} in csdc {:x}", seq, TO64(pCsDc));
                    break;
                }
            }
            if (found) break;
        }
        if (!found)
        {
            LOG_INFO(SYN_DEV_FAIL, "cs {} not found", suspectedCsHandle);
        }
    }

    if (csdc)
    {
        LOG_INFO(SYN_DEV_FAIL, "Logging csdc {:x} to file {}", TO64(csdc), SYNAPSE_CS_PARSER_SEPARATE_LOG_FILE);
        if (!_parseSingleCommandSubmission(csdc))
        {
            return false;
        }
    }
    return true;
}

bool QueueBaseQmanWcm::_parseSingleCommandSubmission(CommandSubmissionDataChunks* pCsDc) const
{
    LOG_ERR(SYN_CS_PARSER, "======================= parsing csdc {:x} =======================", TO64(pCsDc));

    AddressRangeMapper     addressRangeMap;
    InflightCsParserHelper parserHelper(&addressRangeMap, m_physicalQueueOffset, pCsDc->isActivate());

    if (!_addDataChunksMapping(addressRangeMap, *pCsDc))
    {
        LOG_ERR(SYN_STREAM, "Failed to add CS-DC mapping");
        return false;
    }

    const CommandSubmission* pCommandSubmission = pCsDc->getCommandSubmissionInstance();
    if (pCommandSubmission == nullptr)
    {
        LOG_DEBUG(SYN_STREAM, "{}: No Command-Submission in CS DC", HLLOG_FUNC);
        return true;
    }

    if (pCsDc->isEnqueue())
    {
        pCsDc->updateWithActivatePqEntries(parserHelper);
    }

    // Get info from PQ-Entries
    addPqEntiesInfoToParserHelper(parserHelper, pCommandSubmission);

    // Get info from synCommandBuffer ("Externals")
    const synCommandBuffer* pSynCommandBuffer = pCommandSubmission->getExecuteExternalQueueCb();
    uint32_t                numOfExternalCbs  = pCommandSubmission->getNumExecuteExternalQueue();
    for (uint32_t i = 0; i < numOfExternalCbs; i++, pSynCommandBuffer++)
    {
        const synCommandBuffer& currentSynCB   = *pSynCommandBuffer;
        const CommandBuffer*    pCommandBuffer = reinterpret_cast<const CommandBuffer*>(currentSynCB);

        parserHelper.addParserPqEntry(pCommandBuffer->GetQueueIndex(),
                                      pCommandBuffer->GetCbHandle(),
                                      pCommandBuffer->GetOccupiedSize());
    }

    // Get info from synInternalQueue ("Internals")
    const synInternalQueue* pInternalQueue   = pCommandSubmission->getExecuteInternalQueueCb();
    uint32_t                numOfInternalCbs = pCommandSubmission->getNumExecuteInternalQueue();
    for (uint32_t i = 0; i < numOfInternalCbs; i++, pInternalQueue++)
    {
        parserHelper.addParserPqEntry(pInternalQueue->queueIndex, pInternalQueue->address, pInternalQueue->size);
    }

    std::unique_ptr<InflightCsParser> pCsParser;
    if (m_deviceType == synDeviceGaudi)
    {
        pCsParser = std::make_unique<gaudi::InflightCsParser>();
    }

    CHECK_POINTER(SYN_STREAM, pCsParser, "CS parser", false);

    bool status = pCsParser->parse(parserHelper);
    // In any case we would like to know about status
    LOG_GCP_VERBOSE("Parsing status - {}. Command-Submission CS-Handles:", status ? "Pass" : "Fail");

    const std::deque<uint64_t>& csHandles = pCsDc->getAllWaitForEventHandles();

    uint32_t i = 0;
    for (auto singleHandle : csHandles)
    {
        LOG_GCP_VERBOSE("Handle-{}: {}", i, singleHandle);
        i++;
    }

    return status;
}

void QueueBaseQmanWcm::_addCsdcToDb(CommandSubmissionDataChunks* pCsDataChunks)
{
    std::unique_lock<std::mutex> mutexDb(m_DBMutex, std::defer_lock);
    STAT_GLBL_START(streamDbMutexDuration);
    mutexDb.lock();
    STAT_GLBL_COLLECT_TIME(streamDbMutexDuration, globalStatPointsEnum::streamDbMutexDuration);

    STAT_GLBL_COLLECT(m_csDataChunksDb.size(), csdcDbSize);

    m_csDataChunksDb.push_back(pCsDataChunks);

    const spQueueInfo&     pQueueInfo         = m_basicQueueInfo.pQueueInfo;
    const PhysicalQueuesId physicalQueuesId   = pQueueInfo->getPhysicalQueuesId();
    uint64_t               waitForEventHandle = 0;
    const bool             operationStatus    = pCsDataChunks->getWaitForEventHandle(waitForEventHandle, false);
    HB_ASSERT(operationStatus == true, "{}: Failed to get wait-for-event handle", __FUNCTION__);
    m_rWorkCompletionManager.addCs(physicalQueuesId, this, waitForEventHandle);
}

bool QueueBaseQmanWcm::_addDataChunksMapping(AddressRangeMapper&          addressRangeMap,
                                             CommandSubmissionDataChunks& csDc) const
{
    if (m_deviceType == synDeviceGaudi)
    {
        _addStaticMapping(addressRangeMap);
    }

    for (auto singleUpperCpDc : csDc.getCpDmaDataChunks())
    {
        uint64_t handle       = singleUpperCpDc->getHandle();
        uint64_t mappedBuffer = (uint64_t)singleUpperCpDc->getChunkBuffer();
        uint64_t bufferSize   = singleUpperCpDc->getUsedSize();

        if (!addressRangeMap.addMapping(handle, mappedBuffer, bufferSize, AddressRangeMapper::ARM_MAPPING_TYPE_RANGE))
        {
            return false;
        }
    }

    for (auto singleLowerCpDc : csDc.getCommandsBufferDataChunks())
    {
        uint64_t handle       = singleLowerCpDc->getHandle();
        uint64_t mappedBuffer = (uint64_t)singleLowerCpDc->getChunkBuffer();
        uint64_t bufferSize   = singleLowerCpDc->getUsedSize();

        if (!addressRangeMap.addMapping(handle, mappedBuffer, bufferSize, AddressRangeMapper::ARM_MAPPING_TYPE_RANGE))
        {
            return false;
        }
    }

    csDc.getProgramCodeBlocksMapping().updateMappingsOf(addressRangeMap);

    return true;
}

void QueueBaseQmanWcm::dfaLogCsDescription(const uint64_t csSeq, DfaReq dfaReq)
{
    std::unique_lock<std::mutex> lock(m_DBMutex);

    LOG_TRACE(SYN_DEV_FAIL, "stream has {} csdc-s", m_csDataChunksDb.size());
    if (m_csDataChunksDb.empty())
    {
        return;
    }

    bool errorCsOnly = dfaReq == DfaReq::ERR_WORK;

    for (auto csIter : m_csDataChunksDb)
    {
        if (csIter->containsHandle(csSeq) || (csSeq == 0))
        {
            std::string csList   = csIter->dfaGetCsList();
            const int   logLevel = errorCsOnly ? SPDLOG_LEVEL_ERROR : SPDLOG_LEVEL_INFO;
            SYN_LOG_TYPE(SYN_DEV_FAIL, logLevel, "cs: {}", csList);

            _dfaLogCsDcInfo(csIter, logLevel, errorCsOnly);
            if (errorCsOnly) break;  // if only the error one, break out
        }
    }
}

void QueueBaseQmanWcm::dfaInfo(DfaReq dfaReq, uint64_t csSeq)
{
    switch (dfaReq)
    {
        case DfaReq::STREAM_INFO:
        {
            LOG_TRACE(SYN_DEV_FAIL, "stream {:x} {}", TO64(this), m_basicQueueInfo.getDescription());
            break;
        }
        case DfaReq::PARSE_CSDC:
        {
            LOG_TRACE(SYN_DEV_FAIL,
                      "------ checking stream {:x} {} ------",
                      TO64(this),
                      getBasicQueueInfo().getDescription());
            parseCsDc(csSeq);
            break;
        }
        case DfaReq::ALL_WORK:
        case DfaReq::ERR_WORK:
        {
            std::string desc = m_basicQueueInfo.getDescription();
            LOG_TRACE(SYN_DEV_FAIL, "------ checking stream {:x} {} for csSeq {} ------", TO64(this), desc, csSeq);
            dfaLogCsDescription(csSeq, dfaReq);
            break;
        }
        case DfaReq::SCAL_STREAM:
            break;  // we should never get here
    }               // switch
}
