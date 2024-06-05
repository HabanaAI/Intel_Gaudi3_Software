#include "command_submission_data_chunks.hpp"

#include "command_buffer.hpp"
#include "command_submission_builder.hpp"
#include "command_submission.hpp"
#include "data_chunk/data_chunk.hpp"
#include "data_chunk/data_chunks_cache.hpp"
#include "runtime/qman/common/inflight_cs_parser.hpp"
#include "synapse_runtime_logging.h"
#include "types_exception.h"
#include "types.h"
#include "utils.h"

#include "runtime/qman/common/device_recipe_addresses_generator_interface.hpp"
#include "runtime/common/common_types.hpp"

#include "runtime/qman/common/recipe_cache_manager.hpp"
#include "runtime/qman/common/qman_types.hpp"

#include "platform/gaudi/utils.hpp"

#include "event_triggered_logger.hpp"
#include <unistd.h>

const uint32_t INVALID_SOBJ_ADDRESS = UINT32_MAX;
bool           findHostBuffer(uint64_t                      mappedBaseAddress,
                              const upHostRangeMappingDB&   hostRangeMappingDB,
                              uint64_t                      hostRangeMappingDbSize,
                              const HostRangeMappingVector& additionalHostBuffersMapping,
                              uint64_t&                     hostBaseBufferAddress)
{
    for (uint64_t i = 0; i < hostRangeMappingDbSize; i++)
    {
        if ((mappedBaseAddress >= hostRangeMappingDB[i].mappedAddress) &&
            (mappedBaseAddress < hostRangeMappingDB[i].mappedAddress + hostRangeMappingDB[i].bufferSize))
        {
            hostBaseBufferAddress =
                hostRangeMappingDB[i].hostAddress + (mappedBaseAddress - hostRangeMappingDB[i].mappedAddress);
            return true;
        }
    }

    for (auto singleMapping : additionalHostBuffersMapping)
    {
        if ((mappedBaseAddress >= singleMapping.mappedAddress) &&
            (mappedBaseAddress < singleMapping.mappedAddress + singleMapping.bufferSize))
        {
            hostBaseBufferAddress = singleMapping.hostAddress + (mappedBaseAddress - singleMapping.mappedAddress);
            return true;
        }
    }

    return false;
}

void waitForever()
{
    LOG_CRITICAL(SYN_CS, "Wait forever");
    ETL_TRIGGER(EVENT_LOGGER_TRIGGER_TYPE_CHECK_OPCODES);
    while (1)
    {
        sleep(10);
    }
}

#define WAIT_FOREVER(returnUponFailure) \
    if (!returnUponFailure)             \
        waitForever();                  \
    return false;

CommandSubmissionDataChunks::CommandSubmissionDataChunks(eCommandSumissionDataChunkType usageType,
                                                         synDeviceType                  deviceType,
                                                         size_t                         csDcMappingDbSize)
: m_deviceType(deviceType),
  m_recipeId(INVALID_RECIPE_ID),
  m_pRecipe(nullptr),
  m_pRecipeHandle(nullptr),
  m_pCommandSubmission(nullptr),
  m_csDcExecutionType(CS_DC_EXECUTION_TYPE_NOT_READY),
  m_usageType(usageType),
  m_stage(EXECUTION_STAGE_LAST),
  m_isCopiedToDc(false),
  m_spRecipeProgramBuffer(nullptr),
  m_prgDataSubTypeAllocationId(RecipeCacheManager::INVALID_ALLOCATION_ID)
{
    m_cpDmaDataChunksMappingDatabase = cpDmaDataChunksMappingDB(csDcMappingDbSize, cpDmaSingleEngineInfo());
}

CommandSubmissionDataChunks::CommandSubmissionDataChunks(eCommandSumissionDataChunkType usageType,
                                                         synDeviceType                  deviceType,
                                                         size_t                         csDcMappingDbSize,
                                                         uint64_t                       recipeId,
                                                         const recipe_t&                rRecipe,
                                                         InternalRecipeHandle*          pRecipeHandle,
                                                         eExecutionStage                stage,
                                                         uint64_t                       maxSectionId,
                                                         uint64_t                       numOfSectionsToPatch)
: m_deviceType(deviceType),
  m_recipeId(recipeId),
  m_pRecipe(&rRecipe),
  m_pRecipeHandle(pRecipeHandle),
  m_executionHandle(CS_DC_INVALID_EXECUTION_HANDLE),
  m_pCommandSubmission(nullptr),
  m_prgCodeHandle(INVALID_HANDLE_VALUE),
  m_prgDataHandle(INVALID_HANDLE_VALUE),
  m_scratchpadHandle(INVALID_HANDLE_VALUE),
  m_sfgSobjLowPartAddress(INVALID_SOBJ_ADDRESS),
  m_csDcExecutionType(CS_DC_EXECUTION_TYPE_NOT_READY),
  m_usageType(usageType),
  m_stage(stage),
  m_isCopiedToDc(false),
  m_spRecipeProgramBuffer(nullptr),
  m_prgDataSubTypeAllocationId(RecipeCacheManager::INVALID_ALLOCATION_ID)
{
    if (usageType == CS_DC_TYPE_COMPUTE)
    {
        bool status = m_hostAddressPatchingOnDataChunksInfo.initialize(maxSectionId, numOfSectionsToPatch);
        if (!status)
        {
            LOG_ERR(SYN_CS, "{}: Failed to initialize patching-info DB", HLLOG_FUNC);
            throw SynapseException("CommandSubmissionDataChunks: Failed to initialize host-patching-information");
        }
    }
    m_cpDmaDataChunksMappingDatabase = cpDmaDataChunksMappingDB(csDcMappingDbSize, cpDmaSingleEngineInfo());
}
CommandSubmissionDataChunks::~CommandSubmissionDataChunks()
{
    if (m_pCommandSubmission != nullptr)
    {
        this->destroyCommandSubmission();
    }
}

void CommandSubmissionDataChunks::addSingleProgramBlobsDataChunk(DataChunk* singleDataChunk)
{
    m_commandsBuffersChunksDatabase.push_back(singleDataChunk);
}

void CommandSubmissionDataChunks::addProgramBlobsDataChunks(const commandsDataChunksDB& dataChunks)
{
    for (auto singleDataChunk : dataChunks)
    {
        m_commandsBuffersChunksDatabase.push_back(singleDataChunk);
    }
}

void CommandSubmissionDataChunks::addCpDmaDataChunks(cpDmaDataChunksDB cpDmaDataChunks)
{
    for (auto singleDataChunk : cpDmaDataChunks)
    {
        m_cpDmaChunksDatabase.push_back(singleDataChunk);
    }
}

void CommandSubmissionDataChunks::setRecipeProgramBuffer(SpRecipeProgramBuffer spRecipeProgramBuffer)
{
    m_spRecipeProgramBuffer = spRecipeProgramBuffer;
}

const SpRecipeProgramBuffer CommandSubmissionDataChunks::getRecipeProgramBuffer() const
{
    return m_spRecipeProgramBuffer;
}

void CommandSubmissionDataChunks::addSingleCpDmaDataChunksMapping(uint32_t engineId,
                                                                  uint64_t hostAddress,
                                                                  uint64_t deviceVirtualAddress,
                                                                  uint64_t size)
{
    cpDmaSingleDataChunkInfo cpDmaDataChunkInfo = {0};
    cpDmaDataChunkInfo.hostAddress              = hostAddress;
    cpDmaDataChunkInfo.deviceVirtualAddress     = deviceVirtualAddress;
    cpDmaDataChunkInfo.size                     = size;
    if (engineId >= m_cpDmaDataChunksMappingDatabase.size())
    {
        m_cpDmaDataChunksMappingDatabase.resize(engineId + 1);
    }
    m_cpDmaDataChunksMappingDatabase[engineId].push_back(cpDmaDataChunkInfo);

    LOG_TRACE(SYN_CS, "{}: engineId {} hostAddress 0x{:x} size 0x{:x}", HLLOG_FUNC, engineId, hostAddress, size);
}

bool CommandSubmissionDataChunks::setCommandSubmissionInstance(CommandSubmission* pCommandSubmission)
{
    if (m_pCommandSubmission != nullptr)
    {
        LOG_ERR(SYN_CS, "{}: Command-Submission is already set", HLLOG_FUNC);
        return false;
    }

    if (pCommandSubmission == nullptr)
    {
        LOG_ERR(SYN_CS, "{}: Got null-pointer for Command-Submission parameter", HLLOG_FUNC);
        return false;
    }

    m_pCommandSubmission = pCommandSubmission;
    return true;
}

CommandSubmission* CommandSubmissionDataChunks::getCommandSubmissionInstance()
{
    return m_pCommandSubmission;
}

const CommandSubmission* CommandSubmissionDataChunks::getCommandSubmissionInstance() const
{
    return m_pCommandSubmission;
}

void CommandSubmissionDataChunks::addWaitForEventHandle(uint64_t waitForEventHandle)
{
    LOG_TRACE(SYN_CS,
              "{}: {:#x} New waitForEvent handle {:#x}, recipeID is {:#x}",
              HLLOG_FUNC,
              TO64(this),
              waitForEventHandle,
              getRecipeId());
    m_waitForEventHandle.push_back(waitForEventHandle);
}

bool CommandSubmissionDataChunks::popWaitForEventHandle(uint64_t waitForEventHandle)
{
    std::deque<uint64_t>::iterator iter;
    for (iter = m_waitForEventHandle.begin(); iter != m_waitForEventHandle.end(); ++iter)
    {
        if (*iter == waitForEventHandle)
        {
            break;
        }
    }

    if (iter == m_waitForEventHandle.end())
    {
        LOG_WARN(SYN_CS,
                 "{}: failed to find waitForEvent handle {:#x}, recipeID is {:#x}",
                 HLLOG_FUNC,
                 waitForEventHandle,
                 getRecipeId());
        return false;
    }

    m_waitForEventHandle.erase(iter);
    return true;
}

bool CommandSubmissionDataChunks::getWaitForEventHandle(uint64_t& waitForEventHandle, bool isOldset) const
{
    if (m_waitForEventHandle.size() == 0)
    {
        LOG_ERR(SYN_CS, "{}: No waitForEvent handle had been set", HLLOG_FUNC);
        return false;
    }

    if (isOldset)
    {
        waitForEventHandle = m_waitForEventHandle.front();
    }
    else
    {
        waitForEventHandle = m_waitForEventHandle.back();
    }

    return true;
}

const std::deque<uint64_t>& CommandSubmissionDataChunks::getAllWaitForEventHandles() const
{
    return m_waitForEventHandle;
}

bool CommandSubmissionDataChunks::isWaitForEventHandleSet() const
{
    return (m_waitForEventHandle.size() != 0);
}

bool CommandSubmissionDataChunks::destroyCommandSubmission()
{
    if (m_pCommandSubmission == nullptr)
    {
        LOG_DEBUG(SYN_CS, "{}: Command-Submission is not set", HLLOG_FUNC);
        return true;
    }

    synStatus status = CommandSubmissionBuilder::getInstance()->destroyCmdSubmissionSynCBs(*m_pCommandSubmission);
    if (status != synSuccess)
    {
        LOG_CRITICAL(SYN_CS, "{}: Failed to destroy Command-Submission", HLLOG_FUNC);
        return false;
    }

    delete m_pCommandSubmission;
    return true;
}

const cpDmaSingleEngineInfo& CommandSubmissionDataChunks::getSingleEngineCpDmaMapping(uint32_t engineId) const
{
    return m_cpDmaDataChunksMappingDatabase.at(engineId);
}

commandsDataChunksDB& CommandSubmissionDataChunks::getCommandsBufferDataChunks()
{
    return m_commandsBuffersChunksDatabase;
}

const commandsDataChunksDB& CommandSubmissionDataChunks::getCommandsBufferDataChunks() const
{
    return m_commandsBuffersChunksDatabase;
}

void CommandSubmissionDataChunks::clearCommandsBufferDataChunks()
{
    m_commandsBuffersChunksDatabase.clear();
}

const cpDmaDataChunksMappingDB& CommandSubmissionDataChunks::getBlobsCpDmaMapping() const
{
    return m_cpDmaDataChunksMappingDatabase;
}

cpDmaDataChunksDB& CommandSubmissionDataChunks::getCpDmaDataChunks()
{
    return m_cpDmaChunksDatabase;
}

const cpDmaDataChunksDB& CommandSubmissionDataChunks::getCpDmaDataChunks() const
{
    return m_cpDmaChunksDatabase;
}

uint64_t CommandSubmissionDataChunks::getRecipeId() const
{
    return m_recipeId;
}

void CommandSubmissionDataChunks::setExecutionHandle(uint64_t executionHandle)
{
    m_executionHandle = executionHandle;
}

uint64_t CommandSubmissionDataChunks::getExecutionHandle() const
{
    return m_executionHandle;
}

eCommandSumissionDataChunkType CommandSubmissionDataChunks::getUsageType() const
{
    return m_usageType;
}

uint64_t CommandSubmissionDataChunks::getProgramCodeHandle() const
{
    return m_prgCodeHandle;
}

void CommandSubmissionDataChunks::setProgramCodeHandle(uint64_t programCodeHandle)
{
    m_prgCodeHandle = programCodeHandle;
}

void CommandSubmissionDataChunks::getProgramDataHandle(uint64_t& prgDataHandle) const
{
    prgDataHandle = m_prgDataHandle;
}

void CommandSubmissionDataChunks::setProgramDataHandle(uint64_t prgDataHandle)
{
    m_prgDataHandle = prgDataHandle;
}

bool CommandSubmissionDataChunks::isNewProgramDataHandle(uint64_t newProgramDataHandle) const
{
    return (m_prgDataHandle != newProgramDataHandle);
}

bool CommandSubmissionDataChunks::isNewSobjAddress(uint32_t newSobjAddress) const
{
    return (m_sfgSobjLowPartAddress != newSobjAddress);
}

void CommandSubmissionDataChunks::setSobjAddress(uint32_t sobjAddress)
{
    m_sfgSobjLowPartAddress = sobjAddress;
}

void CommandSubmissionDataChunks::getSobjAddress(uint32_t& sobjAddress) const
{
    sobjAddress = m_sfgSobjLowPartAddress;
}

void CommandSubmissionDataChunks::getScratchpadHandle(uint64_t& scratchpadHandle) const
{
    scratchpadHandle = m_scratchpadHandle;
}

void CommandSubmissionDataChunks::setScratchpadHandle(uint64_t scratchpadHandle)
{
    m_scratchpadHandle = scratchpadHandle;
}

bool CommandSubmissionDataChunks::isNewScratchpadHandle(uint64_t newScratchpadHandle) const
{
    return (m_scratchpadHandle != newScratchpadHandle);
}

eCsDcExecutionType CommandSubmissionDataChunks::getCsDcExecutionType() const
{
    return m_csDcExecutionType;
}

void CommandSubmissionDataChunks::setCsDcExecutionType(eCsDcExecutionType csDcExecutionType)
{
    m_csDcExecutionType = csDcExecutionType;
}

void CommandSubmissionDataChunks::setDataChunksStagesInfo(const std::vector<uint32_t>&   stagesNodes,
                                                          const std::vector<StagedInfo>& stagesInfo)
{
    m_stagesNodes = std::move(stagesNodes);
    m_stagesInfo  = std::move(stagesInfo);
}

const std::vector<uint32_t>& CommandSubmissionDataChunks::getDataChunksStagesNodes()
{
    return m_stagesNodes;
}

const std::vector<StagedInfo>& CommandSubmissionDataChunks::getDataChunksStagesInfo()
{
    return m_stagesInfo;
}

patching::HostAddressPatchingInformation& CommandSubmissionDataChunks::getHostAddressInDcInformation()
{
    return m_hostAddressPatchingOnDataChunksInfo;
}

void CommandSubmissionDataChunks::hostRangeMappingReserveSize(uint64_t size)
{
    m_hostRangeMapping.reserve(size);
}

void CommandSubmissionDataChunks::hostRangeMappingAddEntry(uint64_t hostAddress,
                                                           uint64_t mappedAddress,
                                                           uint64_t bufferSize)
{
    m_hostRangeMapping.push_back(
        {.hostAddress = hostAddress, .mappedAddress = mappedAddress, .bufferSize = bufferSize});
}

const HostRangeMappingVector& CommandSubmissionDataChunks::getHostRangeMapping() const
{
    return m_hostRangeMapping;
}

bool checkForCsUndefinedOpcode(const CommandSubmissionDataChunks* pCommandSubmissionDataChunks,
                               const CommandSubmission*           pCommandSubmissionInput,
                               uint32_t                           amountOfEnginesInArbGroup,
                               bool                               returnUponFailure)
{
    if (pCommandSubmissionDataChunks->getDeviceType() != synDeviceGaudi)
    {
        return true;
    }

    std::array<unsigned, QUEUE_ID_SIZE_MAX> enginesNumOfArbCommands;

    // For when we do not have a CS to define the engine-id
    unsigned genericEngineNumOfArbCommands = 0;

    const cpDmaDataChunksDB&    upperCpDataChunks = pCommandSubmissionDataChunks->getCpDmaDataChunks();
    const commandsDataChunksDB& lowerCpDataChunks = pCommandSubmissionDataChunks->getCommandsBufferDataChunks();

    const std::deque<uint64_t> waitForEventHandles = pCommandSubmissionDataChunks->getAllWaitForEventHandles();

    ETL_ADD_LOG_NEW_ID_ERR(EVENT_LOGGER_LOG_TYPE_CHECK_OPCODES,
                           logId,
                           SYN_CS,
                           "Recipe 0x{:x} Num of event-handlers {}",
                           pCommandSubmissionDataChunks->getRecipeId(),
                           waitForEventHandles.size());

    uint32_t i = 0;
    for (auto singleHandle : waitForEventHandles)
    {
        ETL_ERR(EVENT_LOGGER_LOG_TYPE_CHECK_OPCODES, SYN_CS, "waitForEventHandle ({}): {}", i, singleHandle);

        i++;
    }

    uint64_t             lowerCpDataChunkSize   = lowerCpDataChunks.size();
    uint64_t             upperCpDataChunkSize   = upperCpDataChunks.size();
    uint64_t             hostRangeMappingDbSize = lowerCpDataChunkSize + upperCpDataChunkSize;
    upHostRangeMappingDB hostRangeMapping(new HostRangeMapping[hostRangeMappingDbSize]);

    // Create mapping information host-address<->mapped-address, according to DCs info
    i = 0;
    for (auto singleDataChunk : lowerCpDataChunks)
    {
        hostRangeMapping[i].hostAddress   = (uint64_t)singleDataChunk->getChunkBuffer();
        hostRangeMapping[i].mappedAddress = singleDataChunk->getHandle();
        hostRangeMapping[i].bufferSize    = singleDataChunk->getUsedSize();
        i++;
    }
    for (auto singleDataChunk : upperCpDataChunks)
    {
        hostRangeMapping[i].hostAddress   = (uint64_t)singleDataChunk->getChunkBuffer();
        hostRangeMapping[i].mappedAddress = singleDataChunk->getHandle();
        hostRangeMapping[i].bufferSize    = singleDataChunk->getUsedSize();
        i++;
    }

    const HostRangeMappingVector& additionalHostBuffersMapping = pCommandSubmissionDataChunks->getHostRangeMapping();

    const uint64_t gigabyte         = 1024 * 1024 * 1024;
    const uint64_t minDeviceAddress = 0x20000000;  // From gaudi Hal-Reader
    const uint64_t maxDeviceAddress = minDeviceAddress + (64 * gigabyte);

    uint64_t minRecipeCacheAddress = std::numeric_limits<uint64_t>::max();
    uint64_t maxRecipeCacheAddress = 0;

    struct CpDmaDeviceRange
    {
        uint64_t deviceAddress;
        uint64_t size;
    };

    std::vector<CpDmaDeviceRange> deviceRanges;

    // Create PrimeQueueEntries database
    //
    // For Compute we should go over the Upper-CP, for Memcopy over the Lower-CP (the Upper-CP DB is empty)
    // TODO - add PQ-Entries to the CS itself, and use them, instead...
    PrimeQueueEntries        pqEntries;
    const PrimeQueueEntries* pprimeQueueEntries;
    const CommandSubmission* pCommandSubmission = pCommandSubmissionDataChunks->getCommandSubmissionInstance();

    i = 0;
    if (pCommandSubmission == nullptr)
    {
        pCommandSubmission = pCommandSubmissionInput;
    }

    if (pCommandSubmission == nullptr)
    {
        LOG_TRACE(SYN_CS, "Using Data-Chunks buffers");
        if (upperCpDataChunkSize != 0)
        {
            pqEntries.resize(upperCpDataChunkSize);
            for (auto singleDataChunk : upperCpDataChunks)
            {
                pqEntries[i].queueIndex = QUEUE_ID_SIZE_MAX;
                pqEntries[i].address    = singleDataChunk->getHandle();
                pqEntries[i].size       = singleDataChunk->getUsedSize();
                i++;
            }
        }
        else
        {
            pqEntries.resize(lowerCpDataChunkSize);
            for (auto singleDataChunk : lowerCpDataChunks)
            {
                pqEntries[i].queueIndex = QUEUE_ID_SIZE_MAX;
                pqEntries[i].address    = singleDataChunk->getHandle();
                pqEntries[i].size       = singleDataChunk->getUsedSize();
                i++;
            }
        }
    }
    else if (upperCpDataChunkSize != 0)
    {
        uint32_t executeInternalQueuesCbSize = pCommandSubmission->getNumExecuteInternalQueue();

        uint32_t executeInternalPqEntriesAmount =
            pCommandSubmission->getPrimeQueueEntriesAmount(PQ_ENTRY_TYPE_INTERNAL_EXECUTION);

        const synInternalQueue* pCurrentSynInternalQueueCb = pCommandSubmission->getExecuteInternalQueueCb();
        pqEntries.resize(executeInternalQueuesCbSize + executeInternalPqEntriesAmount);
        for (unsigned entryIdx = 0; entryIdx < executeInternalQueuesCbSize; entryIdx++, pCurrentSynInternalQueueCb++)
        {
            pqEntries[entryIdx].queueIndex = pCurrentSynInternalQueueCb->queueIndex;
            pqEntries[entryIdx].address    = pCurrentSynInternalQueueCb->address;
            pqEntries[entryIdx].size       = pCurrentSynInternalQueueCb->size;
        }

        pCommandSubmission->getPrimeQueueEntries(PQ_ENTRY_TYPE_INTERNAL_EXECUTION, pprimeQueueEntries);
        for (auto singlePqEntry : *pprimeQueueEntries)
        {
            pqEntries[i].queueIndex = singlePqEntry.queueIndex;
            pqEntries[i].address    = singlePqEntry.address;
            pqEntries[i].size       = singlePqEntry.size;
            i++;
        }
    }
    else
    {
        uint32_t queueIndex = 0;

        uint32_t executeExternalQueuesCbSize = pCommandSubmission->getNumExecuteExternalQueue();

        uint32_t executeExternalPqEntriesAmount =
            pCommandSubmission->getPrimeQueueEntriesAmount(PQ_ENTRY_TYPE_EXTERNAL_EXECUTION);

        synCommandBuffer* pCurrentExecuteSynCBs = pCommandSubmission->getExecuteExternalQueueCb();
        pqEntries.resize(executeExternalQueuesCbSize + executeExternalPqEntriesAmount);
        for (unsigned entryIdx = 0; entryIdx < executeExternalQueuesCbSize; entryIdx++, pCurrentExecuteSynCBs++)
        {
            synCommandBuffer& currentSynCB   = *pCurrentExecuteSynCBs;
            CommandBuffer*    pCommandBuffer = reinterpret_cast<CommandBuffer*>(currentSynCB);

            pCommandBuffer->GetQueueIndex(queueIndex);

            pqEntries[entryIdx].queueIndex = queueIndex;
            pqEntries[entryIdx].address    = pCommandBuffer->GetCbHandle();
            pqEntries[entryIdx].size       = pCommandBuffer->GetOccupiedSize();
        }

        pCommandSubmission->getPrimeQueueEntries(PQ_ENTRY_TYPE_EXTERNAL_EXECUTION, pprimeQueueEntries);
        for (auto singlePqEntry : *pprimeQueueEntries)
        {
            pqEntries[i].queueIndex = singlePqEntry.queueIndex;
            pqEntries[i].address    = singlePqEntry.address;
            pqEntries[i].size       = singlePqEntry.size;
            i++;
        }
    }

    ETL_ADD_LOG_SET_ID_ERR(EVENT_LOGGER_LOG_TYPE_CHECK_OPCODES,
                           logId,
                           SYN_CS,
                           "Num of PQ entries {}",
                           pqEntries.size());

    uint16_t numOfEnginesWithArbPackets = 0;

    bool shouldLog      = false;
    bool isFailureFound = false;
    do
    {
        ePacketValidationLoggingMode loggingMode = PKT_VAIDATION_LOGGING_MODE_DISABLED;

        numOfEnginesWithArbPackets = 0;

        i = 0;
        for (unsigned k = 0; k < QUEUE_ID_SIZE_MAX; k++)
        {
            enginesNumOfArbCommands[k] = 0;
        }

        if (shouldLog)
        {
            loggingMode = PKT_VAIDATION_LOGGING_MODE_ENABLED;
        }

        for (auto singlePqEntry : pqEntries)
        {
            if (shouldLog)
            {
                ETL_ERR(EVENT_LOGGER_LOG_TYPE_CHECK_OPCODES,
                        SYN_CS,
                        "PQ entry {} queue-index {}",
                        i,
                        singlePqEntry.queueIndex);
            }

            // To be used in case there are DCs but no CS supplied. For verifying that each DC will have full
            // ARB-wrapping
            if (pCommandSubmission == nullptr)
            {
                genericEngineNumOfArbCommands = 0;
            }

            const void* buffer     = (const void*)singlePqEntry.address;
            uint64_t    bufferSize = singlePqEntry.size;

            uint64_t pqHostBufferAddress = 0;

            bool isPqHostBufferFound = findHostBuffer((uint64_t)buffer,
                                                      hostRangeMapping,
                                                      hostRangeMappingDbSize,
                                                      additionalHostBuffersMapping,
                                                      pqHostBufferAddress);
            if (!isPqHostBufferFound)
            {
                if (!shouldLog)
                {
                    isFailureFound = true;
                    break;
                }
                else
                {
                    ETL_CRITICAL(EVENT_LOGGER_LOG_TYPE_CHECK_OPCODES, SYN_CS, "PQ blobs-buffer {} is OOB", i);

                    WAIT_FOREVER(returnUponFailure);
                }
            }

            const uint32_t* pCurrentPacket = (const uint32_t*)pqHostBufferAddress;
            int64_t         leftBufferSize = bufferSize;

            unsigned* pNumOfArbCommands = &genericEngineNumOfArbCommands;
            if ((pCommandSubmission != nullptr) && (singlePqEntry.queueIndex < QUEUE_ID_SIZE_MAX))
            {
                pNumOfArbCommands = &enginesNumOfArbCommands[singlePqEntry.queueIndex];

                if (*pNumOfArbCommands == 2)
                {
                    // Just to make sure
                    *pNumOfArbCommands = 0;
                }
            }

            while (leftBufferSize > 0)
            {
                unsigned&      numOfArbCommands = *pNumOfArbCommands;
                utilPacketType packetType       = UTIL_PKT_TYPE_OTHER;
                uint64_t       pktBufferAddress = 0;
                uint64_t       pktBufferSize    = 0;

                if (!gaudi::isValidPacket(pCurrentPacket,
                                          leftBufferSize,
                                          packetType,
                                          pktBufferAddress,
                                          pktBufferSize,
                                          loggingMode))
                {
                    if (!shouldLog)
                    {
                        isFailureFound = true;
                        break;
                    }
                    else
                    {
                        ETL_CRITICAL(EVENT_LOGGER_LOG_TYPE_CHECK_OPCODES,
                                     SYN_CS,
                                     "Invalid OPCODE found in upper-CP blobs' buffer index {}",
                                     i);

                        WAIT_FOREVER(returnUponFailure);
                    }
                }

                if ((packetType == UTIL_PKT_TYPE_CP_DMA) || (packetType == UTIL_PKT_TYPE_LDMA))
                {
                    if ((pktBufferAddress <= maxDeviceAddress) && (pktBufferAddress >= minDeviceAddress))
                    {  // In the device
                        minRecipeCacheAddress = std::min(minRecipeCacheAddress, pktBufferAddress);
                        maxRecipeCacheAddress = std::max(maxRecipeCacheAddress, pktBufferAddress + pktBufferSize);

                        if (packetType == UTIL_PKT_TYPE_CP_DMA)
                        {
                            deviceRanges.push_back({.deviceAddress = pktBufferAddress, .size = pktBufferSize});
                        }
                    }
                    else if ((packetType == UTIL_PKT_TYPE_LDMA) && (additionalHostBuffersMapping.size() == 0))
                    {  // Indicates a LDMA packet of a buffer which contains data instead of packets
                        continue;
                    }
                    else
                    {  // In the host
                        uint64_t hostBaseBufferAddress = 0;

                        bool isHostBufferFound = findHostBuffer(pktBufferAddress,
                                                                hostRangeMapping,
                                                                hostRangeMappingDbSize,
                                                                additionalHostBuffersMapping,
                                                                hostBaseBufferAddress);
                        HB_ASSERT(isHostBufferFound, "Lower-CP blobs-buffer {} is OOB", i);

                        const void* commandsBuffer = (const void*)hostBaseBufferAddress;

                        if (!gaudi::checkForUndefinedOpcode(commandsBuffer, pktBufferSize, loggingMode))
                        {
                            const std::string packetName = (packetType == UTIL_PKT_TYPE_CP_DMA) ? "CP-DMA" : "LIN-DMA";

                            if (!shouldLog)
                            {
                                isFailureFound = true;
                                break;
                            }
                            else
                            {
                                ETL_ADD_LOG_NEW_ID_ERR(
                                    EVENT_LOGGER_LOG_TYPE_CHECK_OPCODES,
                                    etlLogId,
                                    SYN_CS,
                                    "Invalid OPCODE found in lower-CP blobs' buffer {} index {}. "
                                    "Pointed buffer: Host-Address 0x{:x} Mapped-Address 0x{:x} size 0x{:x}",
                                    packetName,
                                    i,
                                    hostBaseBufferAddress,
                                    pktBufferAddress,
                                    pktBufferSize);

                                ETL_PRE_OPERATION_SET_ID(etlLogId, EVENT_LOGGER_LOG_TYPE_CHECK_OPCODES);
                                ETL_PRINT_BUFFER(EVENT_LOGGER_LOG_TYPE_CHECK_OPCODES,
                                                 etlLogId,
                                                 (uint32_t*)commandsBuffer,
                                                 pktBufferSize,
                                                 MAX_VARIABLE_PARAMS);

                                WAIT_FOREVER(returnUponFailure);
                            }
                        }
                    }
                }
                else if (packetType == UTIL_PKT_TYPE_ARB_CLEAR)
                {
                    LOG_DEBUG(SYN_CS, "Arb clear for queue-index {}", singlePqEntry.queueIndex);

                    if (numOfArbCommands == 2)
                    {
                        numOfArbCommands = 0;
                    }

                    if (numOfArbCommands != 1)
                    {
                        if (!shouldLog)
                        {
                            isFailureFound = true;
                            break;
                        }
                        else
                        {
                            // Might be single clear and multiple set

                            ETL_ERR(EVENT_LOGGER_LOG_TYPE_CHECK_OPCODES,
                                    SYN_CS,
                                    "Unexpected multiple ARB-Clear packets for queue-index {}",
                                    singlePqEntry.queueIndex);

                            WAIT_FOREVER(returnUponFailure);
                        }
                    }

                    numOfArbCommands++;
                }
                else if (packetType == UTIL_PKT_TYPE_ARB_SET)
                {
                    LOG_DEBUG(SYN_CS, "Arb set for queue-index {}", singlePqEntry.queueIndex);

                    if (numOfArbCommands == 2)
                    {
                        numOfArbCommands = 0;
                    }

                    if (numOfArbCommands != 0)
                    {
                        if (!shouldLog)
                        {
                            isFailureFound = true;
                            break;
                        }
                        else
                        {
                            ETL_ERR(EVENT_LOGGER_LOG_TYPE_CHECK_OPCODES,
                                    SYN_CS,
                                    "Unexpected double ARB-Set packets for queue-index {}",
                                    singlePqEntry.queueIndex);

                            WAIT_FOREVER(returnUponFailure);
                        }
                    }
                    else
                    {
                        numOfEnginesWithArbPackets++;
                    }

                    numOfArbCommands++;
                }
            }

            if (leftBufferSize < 0)
            {
                if (unlikely(shouldLog))
                {
                    ETL_ERR(EVENT_LOGGER_LOG_TYPE_CHECK_OPCODES, SYN_CS, "CS Buffer size does not fit");
                }

                isFailureFound = true;
            }

            i++;
        }

        for (unsigned k = 0; k < QUEUE_ID_SIZE_MAX; k++)
        {
            if (enginesNumOfArbCommands[k] == 1)
            {
                if (!shouldLog)
                {
                    isFailureFound = true;
                    break;
                }
                else
                {
                    ETL_ERR(EVENT_LOGGER_LOG_TYPE_CHECK_OPCODES,
                            SYN_CS,
                            "Missing ARB-Clear following ARB-Request for queue-index {}",
                            k);

                    WAIT_FOREVER(returnUponFailure);
                }
            }
        }

        if ((isFailureFound) && (!shouldLog))
        {
            shouldLog = true;
        }
        else
        {
            shouldLog = false;
        }
    } while (shouldLog);

    if ((amountOfEnginesInArbGroup != 0) && (amountOfEnginesInArbGroup != numOfEnginesWithArbPackets))
    {
        ETL_ERR(EVENT_LOGGER_LOG_TYPE_CHECK_OPCODES,
                SYN_CS,
                "Invalid amount of engines with ARB packets: expected {} actual {}",
                amountOfEnginesInArbGroup,
                numOfEnginesWithArbPackets);

        WAIT_FOREVER(returnUponFailure);
    }

    LOG_ERR(SYN_CS,
            "Recipe-ID 0x{:x} minRecipeCacheAddress {} maxRecipeCacheAddress {}",
            pCommandSubmissionDataChunks->getRecipeId(),
            minRecipeCacheAddress,
            maxRecipeCacheAddress);

    return !isFailureFound;
}

bool CommandSubmissionDataChunks::containsHandle(uint64_t csSeq)
{
    return std::find(m_waitForEventHandle.begin(), m_waitForEventHandle.end(), csSeq) != m_waitForEventHandle.end();
}

std::string CommandSubmissionDataChunks::dfaGetCsList()
{
    std:: string s;
    for (const auto& cs : m_waitForEventHandle)
    {
        s += std::to_string(cs) + " ";
    }
    return s;
};

void CommandSubmissionDataChunks::setProgramCodeBlocksMappingInCache(const AddressRangeMapper& programCodeBlocksMapping)
{
    m_programCodeBlocksMapping = programCodeBlocksMapping;
}

bool CommandSubmissionDataChunks::setProgramCodeBlocksMappingInWorkspace(uint64_t                 handle,
                                                                         uint64_t                 mappedAddress,
                                                                         uint64_t                 size,
                                                                         eAddressRangeMappingType mappingType)
{
    m_programCodeBlocksMapping.clear();

    return m_programCodeBlocksMapping.addMapping(handle, mappedAddress, size, mappingType);
}

const AddressRangeMapper& CommandSubmissionDataChunks::getProgramCodeBlocksMapping() const
{
    return m_programCodeBlocksMapping;
}

void CommandSubmissionDataChunks::updateWithActivatePqEntries(InflightCsParserHelper& csParserHelper)
{
    HB_ASSERT_PTR(m_pRecipe);

    uint32_t     activateJobsAmount  = m_pRecipe->activate_jobs_nr;
    const job_t* activateJobs        = m_pRecipe->activate_jobs;
    const job_t* pCurrentActivateJob = activateJobs;

    const blob_t* blobs = m_pRecipe->blobs;
    for (uint32_t i = 0; i < activateJobsAmount; i++, pCurrentActivateJob++)
    {
        const program_t& currentProgram = m_pRecipe->programs[pCurrentActivateJob->program_idx];
        uint32_t         hwStreamId     = pCurrentActivateJob->engine_id;

        uint32_t        programLength     = currentProgram.program_length;
        const uint64_t* blobsIndices      = currentProgram.blob_indices;
        const uint64_t* pCurrentBlobIndex = blobsIndices;

        for (uint32_t j = 0; j < programLength; j++, pCurrentBlobIndex++)
        {
            const blob_t& currentBlob = blobs[*pCurrentBlobIndex];
            csParserHelper.addParserPqEntry(hwStreamId, ((uint64_t) (currentBlob.data)), currentBlob.size);
        }
    }
}