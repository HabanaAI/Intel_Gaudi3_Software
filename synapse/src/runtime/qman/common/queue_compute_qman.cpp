#include "queue_compute_qman.hpp"

#include "defenders.h"
#include "global_statistics.hpp"
#include "habana_global_conf_runtime.h"
#include "hlthunk.h"
#include "physical_queues_manager.hpp"
#include "profiler_api.hpp"
#include "runtime/common/common_types.hpp"
#include "runtime/common/queues/queue_compute_utils.hpp"
#include "runtime/common/recipe/recipe_logger.hpp"
#include "runtime/common/recipe/recipe_manager.hpp"
#include "runtime/common/recipe/recipe_utils.hpp"
#include "runtime/qman/common/address_range_mapper.hpp"
#include "runtime/qman/common/arb_master_helper.hpp"
#include "runtime/qman/common/data_chunk/data_chunk.hpp"
#include "runtime/qman/common/device_recipe_addresses_generator_interface.hpp"
#include "runtime/qman/common/device_recipe_downloader_container.hpp"
#include "runtime/qman/common/device_recipe_downloader_interface.hpp"
#include "runtime/qman/common/device_recipe_downloader.hpp"
#include "runtime/qman/common/dynamic_info_processor.hpp"
#include "runtime/qman/common/qman_event.hpp"
#include "runtime/qman/common/queue_info.hpp"
#include "runtime/qman/common/recipe_cache_manager.hpp"
#include "runtime/qman/common/recipe_program_buffer.hpp"
#include "runtime/qman/common/recipe_static_information.hpp"
#include "runtime/qman/common/stream_dc_downloader.hpp"
#include "runtime/qman/common/stream_master_helper.hpp"
#include "runtime/qman/common/wcm/work_completion_manager.hpp"
#include "types_exception.h"

#include <cstdint>
#include <threads/single_execution_owner.hpp>
#include <sys/mman.h>
#include <sys/signal.h>
#include <execinfo.h>
#include "queue_copy_qman.hpp"

typedef std::shared_ptr<QueueInfo> spQueueInfo;

const uint32_t QueueComputeQman::numOfElementsToRelease = 1;

#define VALIDATE_PROTECTED_OPERATION(func, ...)                                                                        \
    {                                                                                                                  \
        STAT_GLBL_START(streamDbMutexDuration);                                                                        \
        std::unique_lock<std::mutex> lock(m_DBMutex);                                                                  \
        STAT_GLBL_COLLECT_TIME(streamDbMutexDuration, globalStatPointsEnum::streamDbMutexDuration);                    \
        func(__VA_ARGS__);                                                                                             \
    }

typedef void (*Handler)(int signum);
static Handler prevSigHandle = SIG_IGN;

const uint64_t REF_COUNT_PER_CS_STAGE = 1;
const uint64_t REF_COUNT_ACTIVATE     = REF_COUNT_PER_CS_STAGE;
const uint64_t REF_COUNT_ENQUEUE      = REF_COUNT_PER_CS_STAGE;

static void dcMprotectSignalHandler(int sig, siginfo_t* si, void* unused)
{
    LOG_ERR(SYN_DATA_CHUNK, "Got SIGSEGV at address: {} logging backtrace", si->si_addr);
    int                   numOfFrames;
    static unsigned const BUFFER_SIZE(100);
    void*                 buffer[BUFFER_SIZE];
    char**                strings;
    numOfFrames = backtrace(buffer, BUFFER_SIZE);

    strings = backtrace_symbols(buffer, numOfFrames);
    if (strings == nullptr)
    {
        perror("backtrace_symbols");
        exit(EXIT_FAILURE);
    }

    for (int frame = 0; frame < numOfFrames; frame++)
    {
        LOG_INFO(SYN_DATA_CHUNK, "bt frame {} : {}", frame, strings[frame]);
    }

    free(strings);
    if (prevSigHandle != SIG_IGN && prevSigHandle != SIG_DFL)
    {
        LOG_TRACE(SYN_DATA_CHUNK, "previous SIGSEGV handler was set, calling it");
        (*prevSigHandle)(sig);
    }
    else
    {
        LOG_TRACE(SYN_DATA_CHUNK, "no previous SIGSEGV handler was set, exiting");
    }
    exit(EXIT_FAILURE);
}

IDynamicInfoProcessor* QueueComputeQman::s_pTestDynamicInfoProcessor = nullptr;

void QueueComputeQman::setDcMprotectSignalHandler()
{
    struct sigaction sa;
    sa.sa_flags = SA_SIGINFO;
    sigemptyset(&sa.sa_mask);
    sa.sa_sigaction = dcMprotectSignalHandler;
    Handler old     = signal(SIGINT, SIG_IGN);
    if (old != SIG_IGN)
    {
        prevSigHandle = old;
    }
    if (sigaction(SIGSEGV, &sa, NULL) == -1)
    {
        LOG_ERR(SYN_DATA_CHUNK, "unable to call sigaction");
    }
}

QueueComputeQman::QueueComputeQman(const BasicQueueInfo&                        rBasicQueueInfo,
                                   uint32_t                                     physicalQueueOffset,
                                   uint32_t                                     amountOfEnginesInArbGroup,
                                   bool                                         isReduced,
                                   synDeviceType                                deviceType,
                                   PhysicalQueuesManagerInterface*              pPhysicalStreamsManager,
                                   WorkCompletionManagerInterface&              rWorkCompletionManager,
                                   SubmitCommandBuffersInterface&               rSubmitter,
                                   DevMemoryAllocInterface&                     rDevMemAlloc,
                                   DeviceRecipeAddressesGeneratorInterface&     rDevRecipeAddress,
                                   DeviceRecipeDownloaderContainerInterface&    rDeviceRecipeDownloaderContainer,
                                   QueueInterface&                              rStreamCopy,
                                   std::unique_ptr<StreamMasterHelperInterface> pStreamMasterHelper)
: QueueBaseQmanWcm(rBasicQueueInfo, physicalQueueOffset, deviceType, pPhysicalStreamsManager, rWorkCompletionManager),
  m_amountOfEnginesInArbGroup(amountOfEnginesInArbGroup),
  m_rSubmitter(rSubmitter),
  m_rDevMemAlloc(rDevMemAlloc),
  m_rDevRecipeAddress(rDevRecipeAddress),
  m_rDeviceRecipeDownloaderContainer(rDeviceRecipeDownloaderContainer),
  m_rStreamCopy(rStreamCopy),
  m_memoryManager(rDevMemAlloc),
  m_csDcAllocator(m_memoryManager, isReduced),
  m_arbMasterHelper(m_deviceType),
  m_pStreamMasterHelper(std::move(pStreamMasterHelper)),
  m_streamDcDownloader(m_deviceType, m_physicalQueueOffset),
  m_isSyncWithDmaSynapseRequired(true),
  m_lastRecipeIdActivated(0),
  m_lastWsDwldPrgCodeAddress(INVALID_DEVICE_ADDR),
  m_lastWsDwldPrgCodeRecipeId(0),
  m_lastWsDwldPrgDataAddress(INVALID_DEVICE_ADDR),
  m_lastWsDwldPrgDataRecipeId(0)
{
    HB_ASSERT(m_basicQueueInfo.queueType == INTERNAL_STREAM_TYPE_COMPUTE,
              "QueueComputeQman: illegal type {}",
              m_basicQueueInfo.queueType);

    if (m_deviceType == synDeviceGaudi)
    {
        if (!m_pStreamMasterHelper->createStreamMasterJobBuffer(m_arbMasterHelper.getArbMasterQmanId()))
        {
            throw SynapseException("QueueBase: Failed to create Job buffer for Stream-Master");
        }
    }
}

QueueComputeQman::~QueueComputeQman()
{
    LOG_TRACE(SYN_STREAM, "{}", HLLOG_FUNC);

    _tryToReleaseProcessorsCsDc();
    m_csDcAllocator.updateCache();

    {
        STAT_GLBL_START(streamDbMutexDuration);
        std::unique_lock<std::mutex> mutex(m_DBMutex);
        STAT_GLBL_COLLECT_TIME(streamDbMutexDuration, globalStatPointsEnum::streamDbMutexDuration);
        _deleteAllRecipeProcessors();
        _clearCsdcDB();
    }
}

synStatus QueueComputeQman::initAllocators()
{
    return m_csDcAllocator.initAllocators();
}

void QueueComputeQman::_deleteAllRecipeProcessors()
{
    for (auto recipeProcessorItr : m_recipeProcessorsDB)
    {
        if (s_pTestDynamicInfoProcessor == nullptr)
        {
            delete recipeProcessorItr.second;
        }
    }
    m_recipeProcessorsDB.clear();
}

void QueueComputeQman::_clearCsdcDB()
{
    uint32_t csdcDbSize = m_csDataChunksDb.size();
    if (csdcDbSize != 0)
    {
        LOG_CRITICAL(SYN_STREAM, "free all CS Data-Chunk, should not happen unless device reset!");
        while (csdcDbSize != 0)
        {
            CommandSubmissionDataChunks* pCsDataChunks           = m_csDataChunksDb.front();
            const uint64_t               csDataChunksHostAddress = TO64(pCsDataChunks);

            uint64_t waitForEventHandle;
            bool     status = pCsDataChunks->getWaitForEventHandle(waitForEventHandle, true);
            HB_ASSERT(status,
                      "{}: CSDC {:#x} does not contain waitForEventHandle {:#x}",
                      __FUNCTION__,
                      TO64(pCsDataChunks),
                      waitForEventHandle);
            status = pCsDataChunks->popWaitForEventHandle(waitForEventHandle);
            HB_ASSERT(status,
                      "{}: CSDC {:#x} cannot pop waitForEventHandle {:#x}",
                      __FUNCTION__,
                      TO64(pCsDataChunks),
                      waitForEventHandle);

            if (!pCsDataChunks->isWaitForEventHandleSet())
            {
                LOG_DEBUG(SYN_STREAM, "Clear {:#x} CS Data-Chunk since it is not use", csDataChunksHostAddress);
                delete pCsDataChunks;
            }

            LOG_DEBUG(SYN_STREAM, "Pop csDataChunksHostAddress {:#x}", csDataChunksHostAddress);
            m_csDataChunksDb.pop_front();
            csdcDbSize--;
        }
    }
}

synStatus QueueComputeQman::launch(const synLaunchTensorInfoExt* launchTensorsInfo,
                                   uint32_t                      launchTensorsAmount,
                                   uint64_t                      workspaceAddress,
                                   InternalRecipeHandle*         pRecipeHandle,
                                   uint64_t                      assertAsyncMappedAddress,
                                   uint32_t                      flags,
                                   EventWithMappedTensorDB&      events,
                                   uint8_t                       apiId)
{
    STAT_GLBL_START(streamComputeLaunch);

    CHECK_POINTER(SYN_STREAM, pRecipeHandle, "pRecipeHandle", synFail);
    basicRecipeInfo& basicRecipeHandle = pRecipeHandle->basicRecipeHandle;
    recipe_t*        pRecipe           = basicRecipeHandle.recipe;
    HB_ASSERT_PTR(pRecipe);

    const uint64_t currLaunchSeqId = QueueComputeUtils::getLaunchSeqId();
    STAT_GLBL_COLLECT(currLaunchSeqId, seqId);
    STAT_GLBL_COLLECT(pRecipeHandle->recipeSeqNum, recipeId);

    PROFILER_COLLECT_TIME()

    bool isIH2DRecipe = RecipeUtils::isIH2DRecipe(pRecipe);

    LOG_DEBUG(SYN_STREAM,
              "Stream {:#x} {} launch recipe {:#x} workspaceAddress {:#x} currLaunchSeqId {} isIH2DRecipe {}",
              TO64(this),
              m_basicQueueInfo.getDescription(),
              TO64(pRecipe),
              workspaceAddress,
              currLaunchSeqId,
              isIH2DRecipe);

    if (QueueComputeUtils::isRecipeEmpty(*pRecipeHandle))
    {
        return synSuccess;
    }

    synStatus status = QueueComputeUtils::prepareLaunch(m_basicQueueInfo,
                                                        launchTensorsInfo,
                                                        launchTensorsAmount,
                                                        pRecipeHandle,
                                                        events,
                                                        flags);
    if (status != synSuccess)
    {
        return status;
    }

    DeviceRecipeDownloaderInterface* pDeviceRecipeDownloader;
    m_rDeviceRecipeDownloaderContainer.addDeviceRecipeDownloader(m_amountOfEnginesInArbGroup,
                                                                 *pRecipeHandle,
                                                                 pDeviceRecipeDownloader);
    DeviceRecipeDownloaderInterface& rDeviceRecipeDownloader = *pDeviceRecipeDownloader;

    if (launchTensorsAmount > 0)
    {
        CHECK_POINTER(SYN_STREAM, launchTensorsInfo, "launchTensorsInfo", synFail);
    }

    uint64_t refCount = getRecipeCacheReferenceCount(m_deviceType);
    uint64_t recipeId;

    uint64_t programDataHandle;
    uint64_t programCodeHandle;
    uint64_t programDataDeviceAddress;
    uint64_t programCodeDeviceAddress;

    bool programCodeInCache = false;
    bool programDataInCache = false;

    uint64_t prgDataSubTypeAllocId =
        isIH2DRecipe ? currLaunchSeqId : RecipeCacheManager::PRG_DATA_SINGULAR_ALLOCATION_ID;

    // For cache blocks
    std::vector<uint64_t> executionBlocksDeviceAddresses;
    status = m_rDevRecipeAddress.generateDeviceRecipeAddresses(*pRecipeHandle,
                                                               workspaceAddress,
                                                               refCount,
                                                               prgDataSubTypeAllocId,
                                                               recipeId,
                                                               programCodeDeviceAddress,
                                                               programDataDeviceAddress,
                                                               programCodeInCache,
                                                               programDataInCache,
                                                               programCodeHandle,
                                                               programDataHandle,
                                                               executionBlocksDeviceAddresses);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_STREAM,
                "{}: stream {} generateDeviceRecipeAddresses failed with status {}",
                HLLOG_FUNC,
                m_basicQueueInfo.getDescription(),
                status);
        return status;
    }

    status = _processRecipe(rDeviceRecipeDownloader,
                            executionBlocksDeviceAddresses,
                            programCodeHandle,
                            programCodeDeviceAddress,
                            workspaceAddress,
                            programCodeInCache);
    if (status != synSuccess)
    {
        if (programCodeInCache || programDataInCache)
        {
            m_rDevRecipeAddress.notifyDeviceRecipeAddressesAreNotUsed(recipeId, refCount, prgDataSubTypeAllocId);
        }

        LOG_ERR(SYN_STREAM,
                "{}: stream {} process-recipe failed with status {}",
                HLLOG_FUNC,
                m_basicQueueInfo.getDescription(),
                status);
        return status;
    }

    const DeviceAgnosticRecipeInfo& rDeviceAgnosticRecipeInfo = rDeviceRecipeDownloader.getDeviceAgnosticRecipeInfo();
    const RecipeStaticInfo&         rRecipeStaticInfo         = rDeviceRecipeDownloader.getRecipeStaticInfo();

    // In case of IH2D, we create a new buffer which will be deleted only after both
    // 1) The Memcopy of it to the Device is completed
    // AND
    // 2) The SMF (done as part of the DIP Host-operation) is completed
    std::shared_ptr<RecipeProgramBuffer> programDataRecipeBuffer =
        std::make_shared<RecipeProgramBuffer>(recipeId,
                                              pRecipe->program_data_blobs_buffer,
                                              pRecipe->program_data_blobs_size,
                                              false);

    // The PRG-Data host buffer is required here, as SIF is using it, prior of DWL that buffer to the device
    if (isIH2DRecipe)
    {
        char* ih2dProgramDataBuffer = new char[pRecipe->program_data_blobs_size];
        std::memcpy(ih2dProgramDataBuffer, pRecipe->program_data_blobs_buffer, pRecipe->program_data_blobs_size);

        std::shared_ptr<RecipeProgramBuffer> ih2dProgramDataRecipeBuffer =
            std::make_shared<RecipeProgramBuffer>(recipeId,
                                                  ih2dProgramDataBuffer,
                                                  pRecipe->program_data_blobs_size,
                                                  true);

        // Now the *external* programDataRecipeBuffer holds the ih2d RPB
        ih2dProgramDataRecipeBuffer.swap(programDataRecipeBuffer);

        LOG_DEBUG(SYN_STREAM, "Launch Pre-SIF Data {:#x}", (uint64_t)ih2dProgramDataBuffer);

        STAT_GLBL_START(streamComputeMutexDuration);
        std::unique_lock<std::mutex> mutex(m_mutex);
        STAT_GLBL_COLLECT_TIME(streamComputeMutexDuration, globalStatPointsEnum::streamComputeMutexDuration);

        status = _runSifPreDownload(basicRecipeHandle,
                                    recipeId,
                                    rDeviceAgnosticRecipeInfo,
                                    rRecipeStaticInfo,
                                    pRecipeHandle,
                                    launchTensorsInfo,
                                    launchTensorsAmount,
                                    (uint64_t)ih2dProgramDataBuffer);
        if (status != synSuccess)
        {
            if (programCodeInCache || programDataInCache)
            {
                m_rDevRecipeAddress.notifyDeviceRecipeAddressesAreNotUsed(recipeId, refCount, prgDataSubTypeAllocId);
            }
            return status;
        }
    }

    // Currently this DWL also performs processing, which is required even in case neither (PRG-Code nor PRG-Data) is in
    // cache
    status = _downloadBuffersToCache(rDeviceRecipeDownloader,
                                     executionBlocksDeviceAddresses,
                                     programCodeHandle,
                                     programDataHandle,
                                     programDataDeviceAddress,
                                     workspaceAddress,
                                     programDataRecipeBuffer,
                                     programCodeInCache,
                                     programDataInCache);
    if (status != synSuccess)
    {
        if (isIH2DRecipe)
        {
            releaseSifOwnership(basicRecipeHandle,
                                rDeviceAgnosticRecipeInfo,
                                rRecipeStaticInfo,
                                pRecipeHandle,
                                recipeId);
        }
        if (programCodeInCache || programDataInCache)
        {
            m_rDevRecipeAddress.notifyDeviceRecipeAddressesAreNotUsed(recipeId, refCount, prgDataSubTypeAllocId);
        }
        return status;
    }

    // SW-74708 Soften the lock on the workspace address
    STAT_GLBL_START(streamComputeMutexDuration);
    std::unique_lock<std::mutex> mutex(m_mutex);
    STAT_GLBL_COLLECT_TIME(streamComputeMutexDuration, globalStatPointsEnum::streamComputeMutexDuration);

    IDynamicInfoProcessor* pRecipeProcessor = nullptr;
    status                                  = retrieveDynamicInfoProcessor(basicRecipeHandle,
                                          rDeviceAgnosticRecipeInfo,
                                          rRecipeStaticInfo,
                                          pRecipeHandle,
                                          recipeId,
                                          pRecipeProcessor);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_STREAM,
                "{}: Cant perform launch on stream {} (retrieve processor)",
                HLLOG_FUNC,
                m_basicQueueInfo.getDescription());

        // There is no point in calling releaseSifOwnership since we cannot release ownership w/o the DIP

        if (programCodeInCache || programDataInCache)
        {
            m_rDevRecipeAddress.notifyDeviceRecipeAddressesAreNotUsed(recipeId, refCount, prgDataSubTypeAllocId);
        }
        return status;
    }

    bool isSyncWithDmaSynapseRequired = false;

    // Only if the last DWL to WS (PrgData or PrgCode had been for given recipe) and to the same address,
    // then DWL is not required (it is already at that location)
    //
    // We DO NOT check address overrun by synMemcopy!!! (no definitions about that, at the moment)

    if ((!programCodeInCache) || (!programDataInCache))
    {
        status = _downloadBuffersToWorkspace(rDeviceRecipeDownloader,
                                             isSyncWithDmaSynapseRequired,
                                             recipeId,
                                             programDataRecipeBuffer,
                                             programCodeHandle,
                                             programDataHandle,
                                             programCodeDeviceAddress,
                                             programDataDeviceAddress,
                                             programCodeInCache,
                                             programDataInCache);
        if (status != synSuccess)
        {
            if (isIH2DRecipe)
            {
                pRecipeProcessor->getDsdPatcher()->releaseOwnership();
            }

            if (programCodeInCache || programDataInCache)
            {
                m_rDevRecipeAddress.notifyDeviceRecipeAddressesAreNotUsed(recipeId, refCount, prgDataSubTypeAllocId);
            }

            return status;
        }
    }

    const IDynamicInfoProcessor* pConstRecipeProcessor = pRecipeProcessor;

    status =
        _syncPreOperation(isSyncWithDmaSynapseRequired, pConstRecipeProcessor, programCodeHandle, programDataHandle);
    if (status != synSuccess)
    {
        if (isIH2DRecipe)
        {
            pRecipeProcessor->getDsdPatcher()->releaseOwnership();
        }

        if (programCodeInCache || programDataInCache)
        {
            m_rDevRecipeAddress.notifyDeviceRecipeAddressesAreNotUsed(recipeId, refCount, prgDataSubTypeAllocId);
        }

        return status;
    }

    status = _activateAndEnqueue(launchTensorsInfo,
                                 launchTensorsAmount,
                                 basicRecipeHandle,
                                 rDeviceAgnosticRecipeInfo,
                                 rRecipeStaticInfo,
                                 recipeId,
                                 pRecipeHandle,
                                 pRecipeProcessor,
                                 workspaceAddress,
                                 programCodeDeviceAddress,
                                 programDataDeviceAddress,
                                 programCodeHandle,
                                 programDataHandle,
                                 prgDataSubTypeAllocId,
                                 assertAsyncMappedAddress,
                                 refCount,
                                 flags,
                                 programCodeInCache,
                                 programDataInCache,
                                 events);

    if (status != synSuccess)
    {
        if (isIH2DRecipe)
        {
            pRecipeProcessor->getDsdPatcher()->releaseOwnership();
        }

        if (programCodeInCache || programDataInCache)
        {
            m_rDevRecipeAddress.notifyDeviceRecipeAddressesAreNotUsed(recipeId, refCount, prgDataSubTypeAllocId);
        }

        return status;
    }

    if (isIH2DRecipe)
    {
        pRecipeProcessor->getDsdPatcher()->releaseOwnership();
    }

    STAT_GLBL_COLLECT_TIME(streamComputeLaunch, globalStatPointsEnum::streamComputeLaunch);

    PROFILER_MEASURE_TIME(("enqueueWithExternalEvents seqId:" + std::to_string(currLaunchSeqId) +
                           " recipeId:" + std::to_string(pRecipeHandle->recipeSeqNum))
                              .c_str());

    LOG_DEBUG(SYN_STREAM, "Post-Launch {:#x}", (uint64_t)programDataRecipeBuffer->getBuffer());
    return synSuccess;
}

synStatus QueueComputeQman::getMappedMemorySize(uint64_t& mappedMemorySize) const
{
    mappedMemorySize = m_memoryManager.getMappedSize();
    return synSuccess;
}

uint64_t QueueComputeQman::getRecipeCacheReferenceCount(synDeviceType deviceType)
{
    uint64_t refCount;

    switch (deviceType)
    {
        case synDeviceGaudi:
        {
            refCount = REF_COUNT_ENQUEUE;
            break;
        }
        default:
        {
            HB_ASSERT(false, "QueueComputeQman: illegal deviceType {}", deviceType);
            refCount = 0;
        }
    }

    return refCount;
}

synStatus QueueComputeQman::retrieveDynamicInfoProcessor(const basicRecipeInfo&          rBasicRecipeInfo,
                                                         const DeviceAgnosticRecipeInfo& rDeviceAgnosticRecipeInfo,
                                                         const RecipeStaticInfo&         rRecipeStaticInfo,
                                                         InternalRecipeHandle*           pRecipeHandle,
                                                         uint64_t                        recipeId,
                                                         IDynamicInfoProcessor*&         pRecipeProcessor)

{
    PROFILER_COLLECT_TIME()

    std::unique_lock<std::mutex> mutexdb(m_DBMutex, std::defer_lock);
    STAT_GLBL_START(streamDbMutexDuration);
    mutexdb.lock();
    STAT_GLBL_COLLECT_TIME(streamDbMutexDuration, globalStatPointsEnum::streamDbMutexDuration);

    auto recipeProcessorItr    = m_recipeProcessorsDB.find(pRecipeHandle);  // get dynamicInfoProcessor
    auto recipeProcessorEndItr = m_recipeProcessorsDB.end();
    if (recipeProcessorItr != recipeProcessorEndItr)
    {
        LOG_DEBUG(SYN_STREAM, "Stream {:#x} recipe {:#x} already exists", TO64(this), TO64(rBasicRecipeInfo.recipe));
        pRecipeProcessor = recipeProcessorItr->second;
        return synSuccess;
    }

    mutexdb.unlock();

    try
    {
        if (s_pTestDynamicInfoProcessor != nullptr)
        {
            pRecipeProcessor = s_pTestDynamicInfoProcessor;
        }
        else
        {
            pRecipeProcessor = new DynamicInfoProcessor(m_deviceType,
                                                        m_basicQueueInfo,
                                                        m_physicalQueueOffset,
                                                        m_amountOfEnginesInArbGroup,
                                                        recipeId,
                                                        rBasicRecipeInfo,
                                                        rDeviceAgnosticRecipeInfo,
                                                        rRecipeStaticInfo,
                                                        m_arbMasterHelper,
                                                        *m_pStreamMasterHelper.get(),
                                                        m_streamDcDownloader,
                                                        m_rSubmitter,
                                                        m_rDevMemAlloc);
        }
        VALIDATE_PROTECTED_OPERATION(_addRecipeProcessor, pRecipeProcessor, pRecipeHandle);
        LOG_DEBUG(SYN_STREAM,
                  "Stream {:#x} recipe {:#x} DynamicInfoProcessor does not exist, creating {:#x}",
                  TO64(this),
                  TO64(rBasicRecipeInfo.recipe),
                  TO64(pRecipeProcessor));
    }
    catch (SynapseException& err)
    {
        LOG_ERR(SYN_STREAM, "Cant create new (stream) recipe-processor (err - {})", err.what());
        return synFail;
    }

    PROFILER_MEASURE_TIME("retreiveRecipeProcessor")

    return synSuccess;
}

void QueueComputeQman::_addRecipeProcessor(IDynamicInfoProcessor* pRecipeProcessor, InternalRecipeHandle* pRecipeHandle)
{
    m_recipeProcessorsDB[pRecipeHandle] = pRecipeProcessor;
}

void QueueComputeQman::_popCsdc()
{
    m_csDataChunksDb.pop_front();
}

void QueueComputeQman::_clearProcessorsDb()
{
    m_recipeProcessorsDB.clear();
}

synStatus QueueComputeQman::_enqueueAndSync(const synLaunchTensorInfoExt*   launchTensorsInfo,
                                            uint32_t                        launchTensorsAmount,
                                            basicRecipeInfo&                basicRecipeHandle,
                                            const DeviceAgnosticRecipeInfo& rDeviceAgnosticRecipeInfo,
                                            const RecipeStaticInfo&         rRecipeStaticInfo,
                                            uint64_t                        recipeId,
                                            InternalRecipeHandle*           pRecipeHandle,
                                            IDynamicInfoProcessor*&         pRecipeProcessor,
                                            uint64_t                        programDataHandle,
                                            uint64_t                        programCodeHandle,
                                            uint64_t                        workspaceAddress,
                                            uint64_t                        prgDataSubTypeAllocId,
                                            uint64_t                        assertAsyncMappedAddress,
                                            uint32_t                        flags,
                                            bool                            programCodeInCache,
                                            bool                            programDataInCache,
                                            uint64_t                        programDataDeviceAddress,
                                            uint64_t                        programCodeDeviceAddr,
                                            eAnalyzeValidateStatus          analyzeValidateStatus,
                                            EventWithMappedTensorDB&        events,
                                            eCsDcProcessorStatus&           csDcProcessingStatus)
{
    PROFILER_COLLECT_TIME()

    reserve_sig_handle sigHandle;
    sigHandle.id                   = SIG_HANDLE_INVALID;
    sigHandle.sob_base_addr_offset = SIG_HANDLE_INVALID;
    uint64_t csHandle              = 0;
    STAT_GLBL_START(streamComputeEnqueue);
    synStatus status = _enqueue(launchTensorsInfo,
                                launchTensorsAmount,
                                rDeviceAgnosticRecipeInfo,
                                basicRecipeHandle,
                                rRecipeStaticInfo,
                                recipeId,
                                pRecipeHandle,
                                pRecipeProcessor,
                                programDataHandle,
                                programCodeHandle,
                                workspaceAddress,
                                prgDataSubTypeAllocId,
                                assertAsyncMappedAddress,
                                flags,
                                programCodeInCache,
                                programDataInCache,
                                programDataDeviceAddress,
                                programCodeDeviceAddr,
                                csHandle,
                                analyzeValidateStatus,
                                events,
                                sigHandle,
                                csDcProcessingStatus);

    STAT_GLBL_COLLECT_TIME(streamComputeEnqueue, globalStatPointsEnum::streamComputeEnqueue);
    PROFILER_MEASURE_TIME("_enqueueAndSync")

    if (status == synBusy)
    {
        LOG_DEBUG(SYN_STREAM,
                  "{}: Failed to perform launch on stream {}, device is busy",
                  HLLOG_FUNC,
                  m_basicQueueInfo.getDescription());
        return status;
    }

    if (status != synSuccess)
    {
        LOG_ERR(SYN_STREAM, "{}: Failed to perform launch on stream {}", HLLOG_FUNC, m_basicQueueInfo.getDescription());
        return status;
    }

    STAT_GLBL_START(streamComputeEnqueueUpdatePostExecution);
    status = _updateStreamPostExecution(m_basicQueueInfo, csHandle, "enqueue");
    if (status != synSuccess)
    {
        LOG_ERR(SYN_STREAM,
                "{}: Failed to perform post-launch stream-synchronization on stream {}",
                HLLOG_FUNC,
                m_basicQueueInfo.getDescription());
        return status;
    }

    STAT_GLBL_COLLECT_TIME(streamComputeEnqueueUpdatePostExecution,
                      globalStatPointsEnum::streamComputeEnqueueUpdatePostExecution);

    return synSuccess;
}

synStatus QueueComputeQman::_enqueue(const synLaunchTensorInfoExt*   launchTensorsInfo,
                                     uint32_t                        launchTensorsAmount,
                                     const DeviceAgnosticRecipeInfo& rDeviceAgnosticRecipeInfo,
                                     basicRecipeInfo&                basicRecipeHandle,
                                     const RecipeStaticInfo&         rRecipeStaticInfo,
                                     uint64_t                        recipeId,
                                     InternalRecipeHandle*           pRecipeHandle,
                                     IDynamicInfoProcessor*&         pRecipeProcessor,
                                     uint64_t                        programDataHandle,
                                     uint64_t                        programCodeHandle,
                                     uint64_t                        workspaceAddress,
                                     uint64_t                        prgDataSubTypeAllocId,
                                     uint64_t                        assertAsyncMappedAddress,
                                     uint32_t                        flags,
                                     bool                            programCodeInCache,
                                     bool                            programDataInCache,
                                     uint64_t                        programDataDeviceAddress,
                                     uint64_t                        programCodeDeviceAddr,
                                     uint64_t&                       csHandle,
                                     eAnalyzeValidateStatus          analyzeValidateStatus,
                                     EventWithMappedTensorDB&        events,
                                     reserve_sig_handle&             sigHandle,
                                     eCsDcProcessorStatus&           csDcProcessingStatus)
{
    HB_ASSERT((pRecipeProcessor != nullptr), "Recipe-Processor is nullptr");
    // set and validate event handles
    uint32_t numberOfExternalTensors = rDeviceAgnosticRecipeInfo.m_signalFromGraphInfo.getNumberOfExternalTensors();

    STAT_GLBL_START(streamComputeReserve);
    recipe_t* pRecipe = basicRecipeHandle.recipe;
    if ((events.size() > 0) && (numberOfExternalTensors > 0))
    {
        if (events.size() <= numberOfExternalTensors)
        {
            TrainingRetCode reserveStatus =
                m_pPhysicalStreamsManager->reserveSignalObjects(m_basicQueueInfo, numberOfExternalTensors, &sigHandle);
            if (reserveStatus != TRAINING_RET_CODE_SUCCESS)
            {
                LOG_ERR(SYN_API, "{}: Unable to reserve signal objects, status {}", HLLOG_FUNC, reserveStatus);
                return synFailedToSubmitWorkload;
            }
        }
        else
        {
            LOG_WARN(SYN_API,
                     "{}: number of external events {} exceeds number of external tensors {}, running without external "
                     "events instead",
                     HLLOG_FUNC,
                     events.size(),
                     numberOfExternalTensors);
            events.clear();
            numberOfExternalTensors = 0;
        }
    }

    STAT_GLBL_COLLECT_TIME(streamComputeReserve, globalStatPointsEnum::streamComputeReserve);

    STAT_GLBL_START(streamComputeEventManipulation);

    for (size_t eventIdx = 0; eventIdx < events.size(); eventIdx++)
    {
        QmanEvent* currentEvent = dynamic_cast<QmanEvent*>(events[eventIdx].get());
        if (currentEvent == nullptr)
        {
            LOG_ERR(SYN_API,
                    "{}: Failed to cast event handle 0x{:x} to QmanEvent",
                    HLLOG_FUNC,
                    currentEvent->getHandle());
            return synInvalidEventHandle;
        }

        if (currentEvent->getInternalRecipeHandle() != pRecipeHandle)
        {
            LOG_ERR(SYN_API,
                    "{}: Received wrong event handle {:#x} with wrong recipe ID {:#x}",
                    HLLOG_FUNC,
                    currentEvent->getHandle(),
                    recipeId);
            if (sigHandle.id != SIG_HANDLE_INVALID)
            {
                m_pPhysicalStreamsManager->unreserveSignalObjects(m_basicQueueInfo, &sigHandle);
            }
            return synInvalidEventHandle;
        }
        if (rDeviceAgnosticRecipeInfo.m_signalFromGraphInfo.getExtTensorExeOrderByExtTensorIdx(
                currentEvent->getTensorIdx()) == SignalFromGraphInfo::TENSOR_EXE_ORDER_INVALID)
        {
            LOG_ERR(SYN_API,
                    "{}: Received wrong event handle {:#x} with wrong tensor ID {:#x}, tensor name {}",
                    HLLOG_FUNC,
                    currentEvent->getHandle(),
                    currentEvent->getTensorIdx(),
                    currentEvent->getTensorName());
            if (sigHandle.id != SIG_HANDLE_INVALID)
            {
                m_pPhysicalStreamsManager->unreserveSignalObjects(m_basicQueueInfo, &sigHandle);
            }
            return synInvalidEventHandle;
        }
    }

    STAT_GLBL_COLLECT_TIME(streamComputeEventManipulation, globalStatPointsEnum::streamComputeEventManipulation);

    CommandSubmissionDataChunks* pCsDataChunks = nullptr;

    pRecipeProcessor->incrementExecutionHandle();

    uint64_t cpDmaChunksAmount;
    if (!rRecipeStaticInfo.getCpDmaChunksAmount(EXECUTION_STAGE_ENQUEUE, cpDmaChunksAmount, programCodeInCache))
    {
        LOG_ERR(SYN_STREAM,
                "{}: Failed to retrieve data-chunks amount (CP-DMA), cpDmaChunksAmount:{}",
                HLLOG_FUNC,
                cpDmaChunksAmount);
        if (sigHandle.id != SIG_HANDLE_INVALID)
        {
            m_pPhysicalStreamsManager->unreserveSignalObjects(m_basicQueueInfo, &sigHandle);
        }
        return synFail;
    }

    uint64_t commandsDataChunksAmount;
    if (!rDeviceAgnosticRecipeInfo.m_recipeStaticInfo.getProgramCommandsChunksAmount(EXECUTION_STAGE_ENQUEUE,
                                                                                     commandsDataChunksAmount))
    {
        LOG_ERR(SYN_STREAM, "{}: Failed to retrieve data-chunks amount (Commands-Buffer)", HLLOG_FUNC);
        if (sigHandle.id != SIG_HANDLE_INVALID)
        {
            m_pPhysicalStreamsManager->unreserveSignalObjects(m_basicQueueInfo, &sigHandle);
        }
        return synFail;
    }

    // 2) Acquire via DataChunksCache
    eCsDataChunkStatus retrieveStatus = retrieveCsDc(rDeviceAgnosticRecipeInfo,
                                                     *pRecipeProcessor,
                                                     recipeId,
                                                     *pRecipe,
                                                     pRecipeHandle,
                                                     cpDmaChunksAmount,
                                                     commandsDataChunksAmount,
                                                     true,
                                                     EXECUTION_STAGE_ENQUEUE,
                                                     pCsDataChunks);

    // Check if the DCs are relevant for reuse according to their handles and scratch pad
    if (retrieveStatus != CS_DATA_CHUNKS_STATUS_COMPLETED)
    {
        if (sigHandle.id != SIG_HANDLE_INVALID)
        {
            m_pPhysicalStreamsManager->unreserveSignalObjects(m_basicQueueInfo, &sigHandle);
        }

        if (retrieveStatus == CS_DATA_CHUNKS_STATUS_NOT_COMPLETED)
        {
            LOG_DEBUG(SYN_STREAM, "{}: Failed to retrieve Data-Chunks (all resources taken)", HLLOG_FUNC);
            return synAllResourcesTaken;
        }
        else
        {
            LOG_DEBUG(SYN_STREAM, "{}: Failure while trying to retrieve Data-Chunks", HLLOG_FUNC);
            return synFail;
        }
    }

    if (programCodeInCache)
    {
        pCsDataChunks->setProgramCodeBlocksMappingInCache(rRecipeStaticInfo.getProgramCodeBlocksMapping());
        pCsDataChunks->setProgramDataSubTypeAllocationId(prgDataSubTypeAllocId);
    }
    else
    {
        pCsDataChunks->setProgramCodeBlocksMappingInWorkspace(programCodeDeviceAddr,
                                                              (uint64_t)pRecipe->execution_blobs_buffer,
                                                              pRecipe->execution_blobs_buffer_size,
                                                              AddressRangeMapper::ARM_MAPPING_TYPE_RANGE);
    }

    if (isRecipeStageCsDcCpDmaReady(recipeId, EXECUTION_STAGE_ENQUEUE, programCodeHandle, *pCsDataChunks))
    {
        STAT_GLBL_COLLECT(1, streamComputePartialReuse);
        LOG_DEBUG_T(SYN_STREAM,
                    "Stream {:#x} partial reuse (same recipe, stage, and programCodeHandle handles)",
                    TO64(this));
        pCsDataChunks->setCsDcExecutionType(CS_DC_EXECUTION_TYPE_CP_DMA_READY);
    }
    else
    {
        LOG_DEBUG_T(SYN_STREAM,
                    "Stream {:#x} no reuse (different recipe, stage, and programCodeHandle handles)",
                    TO64(this));
        pCsDataChunks->setCsDcExecutionType(CS_DC_EXECUTION_TYPE_NOT_READY);
    }

    synStatus status = pRecipeProcessor->enqueue(launchTensorsInfo,
                                                 launchTensorsAmount,
                                                 pCsDataChunks,
                                                 workspaceAddress,
                                                 programDataDeviceAddress,
                                                 programCodeDeviceAddr,
                                                 programDataHandle,
                                                 programCodeInCache,
                                                 programDataInCache,
                                                 assertAsyncMappedAddress,
                                                 flags,
                                                 csHandle,
                                                 analyzeValidateStatus,
                                                 sigHandle.id,
                                                 sigHandle.sob_base_addr_offset,
                                                 csDcProcessingStatus);

    // In case flag is on, parse each packet (on Gaudi-1)
    // Otherwise, only upon failure
    if ((pCsDataChunks != nullptr) && ((GCFG_PARSE_EACH_COMPUTE_CS.value()) || (status != synSuccess)))
    {
        _parseSingleCommandSubmission(pCsDataChunks);
    }

    if (status != synSuccess)
    {
        LOG_ERR(SYN_STREAM, "{}: Failed to launch", HLLOG_FUNC);
    }
    else
    {
        spQueueInfo pQueueInfo = m_pPhysicalStreamsManager->getStreamInfo(m_basicQueueInfo);
        uint64_t    streamId   = pQueueInfo->getQueueId();

        LOG_TRACE(SYN_PROGRESS,
                  GAUDI_PROGRESS_FMT,
                  INTERNAL_STREAM_TYPE_COMPUTE,
                  pQueueInfo->getQueueId(),
                  pQueueInfo->getPhysicalQueueOffset(),
                  pQueueInfo->getPhysicalQueuesId(),
                  csHandle,
                  HLLOG_FUNC,
                  __LINE__);

        for (auto& event : events)
        {
            QmanEvent* currentEvent = dynamic_cast<QmanEvent*>(event.get());
            currentEvent->setSignalSeqId(streamId, csHandle);
        }
    }

    STAT_GLBL_START(streamComputeUnreserve);

    if (csDcProcessingStatus == CS_DC_PROCESSOR_STATUS_STORED_AND_SUBMITTED)
    {
        // retrieve the CSDC values and store them in the stream processor
        pCsDataChunks->setProgramCodeHandle(programCodeHandle);
        pCsDataChunks->setProgramDataHandle(programDataHandle);
        pCsDataChunks->setScratchpadHandle(workspaceAddress);
        pCsDataChunks->setSobjAddress(sigHandle.sob_base_addr_offset);

        pRecipeProcessor->setProgramCodeHandle(programCodeHandle);
        pRecipeProcessor->setProgramDataHandle(programDataHandle);

        if (!programCodeInCache)
        {
            pRecipeProcessor->setProgramCodeAddrInWS(programCodeHandle);
        }

        uint64_t commandSubmissionHandle = 0;
        pCsDataChunks->getWaitForEventHandle(commandSubmissionHandle, false);
        LOG_DEBUG(SYN_STREAM, "Stream {:#x} store cmd submission info ({:#x})", TO64(this), commandSubmissionHandle);
        _addCsdcToDb(pCsDataChunks);
    }
    else if (csDcProcessingStatus == CS_DC_PROCESSOR_STATUS_FAILED)
    {
        tryToReleaseCommandSubmissionDataChunk(pCsDataChunks);
    }
    if (status != synSuccess && sigHandle.id != SIG_HANDLE_INVALID)
    {
        m_pPhysicalStreamsManager->unreserveSignalObjects(m_basicQueueInfo, &sigHandle);
    }

    STAT_GLBL_COLLECT_TIME(streamComputeUnreserve, globalStatPointsEnum::streamComputeUnreserve);
    return status;
}

synStatus QueueComputeQman::_syncPreOperation(QueueInterface& rPrecedingStream)
{
    STAT_GLBL_START(streamComputeSync);

    PROFILER_COLLECT_TIME()

    synStatus status = performStreamsSynchronization(rPrecedingStream, false);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_STREAM,
                "{}: Failed to perform inner-stream's signaling on stream {}",
                HLLOG_FUNC,
                m_basicQueueInfo.getDescription());
        return status;
    }

    PROFILER_MEASURE_TIME("syncPreOperation")
    STAT_GLBL_COLLECT_TIME(streamComputeSync, globalStatPointsEnum::streamComputeSync);

    return synSuccess;
}

synStatus QueueComputeQman::_updateStreamPostExecution(const BasicQueueInfo& rBasicQueueInfo,
                                                       uint64_t              operationHandle,
                                                       const std::string&    desc)
{
    PROFILER_COLLECT_TIME()

    InternalWaitHandle waitHandle;
    waitHandle.handle = operationHandle;

    TrainingRetCode trainingRetCode =
        m_pPhysicalStreamsManager->updateStreamPostExecution(rBasicQueueInfo, waitHandle, desc);
    if (trainingRetCode != TRAINING_RET_CODE_SUCCESS)
    {
        LOG_ERR(SYN_STREAM,
                "{}: Failed to update stream {} post-execution",
                HLLOG_FUNC,
                rBasicQueueInfo.getDescription());
        return synFail;
    }

    PROFILER_MEASURE_TIME("updateStreamPostExec")

    return synSuccess;
}

eCsDataChunkStatus QueueComputeQman::retrieveCsDc(const DeviceAgnosticRecipeInfo& rDeviceAgnosticRecipeInfo,
                                                  IDynamicInfoProcessor&          rRecipeProcessor,
                                                  uint64_t                        recipeId,
                                                  const recipe_t&                 rRecipe,
                                                  InternalRecipeHandle*           pRecipeHandle,
                                                  uint64_t                        cpDmaChunksAmount,
                                                  uint64_t                        commandsDataChunksAmount,
                                                  bool                            isOldCsDcReuseAllowed,
                                                  eExecutionStage                 stage,
                                                  CommandSubmissionDataChunks*&   pCsDataChunks)
{
    PROFILER_COLLECT_TIME()
    STAT_GLBL_START(streamComputeRetrieveCsDc);

    LOG_TRACE(SYN_STREAM,
              "Stream {:#x} retrieveCsDc recipeId {:#x} cpDmaChunksAmount {} commandsDataChunksAmount {} "
              "isOldCsDcReuseAllowed {}",
              TO64(this),
              recipeId,
              cpDmaChunksAmount,
              commandsDataChunksAmount,
              isOldCsDcReuseAllowed);

    uint32_t           retryCounter = 0;
    const uint64_t     maxRetries(GCFG_DATACHUNK_LOOP_NUM_RETRIES.value());
    const uint32_t     releaseCsDataChunksThreshold = 0;
    const uint32_t     waitForWcmThreshold          = releaseCsDataChunksThreshold + 1;
    eCsDataChunkStatus status                       = CS_DATA_CHUNKS_STATUS_NOT_COMPLETED;

    for (; retryCounter < maxRetries; retryCounter++)
    {
        status = tryToRetrieveCsDc(m_deviceType,
                                   m_recipeProcessorsDB,
                                   rDeviceAgnosticRecipeInfo.m_recipeTensorInfo,
                                   rRecipeProcessor,
                                   recipeId,
                                   rRecipe,
                                   pRecipeHandle,
                                   cpDmaChunksAmount,
                                   commandsDataChunksAmount,
                                   isOldCsDcReuseAllowed,
                                   stage,
                                   retryCounter == 0,
                                   (retryCounter + 1) == maxRetries,
                                   pCsDataChunks);

        if (status != CS_DATA_CHUNKS_STATUS_NOT_COMPLETED)
        {
            break;
        }

        if (retryCounter > waitForWcmThreshold)
        {
            STAT_GLBL_START(streamComputeRetrieveCsDcWcm);
            // Todo enable this code for debug
            // Only single allocator can be dumped at the time. We choose CS_DC_ALLOCATOR_CP_DMA to be default.
            // m_allocators[CS_DC_ALLOCATOR_CP_DMA]->dumpStat();
            LOG_PERIODIC_BY_LEVEL(SYN_STREAM,
                                  SPDLOG_LEVEL_WARN,
                                  std::chrono::milliseconds(1000),
                                  10,
                                  "{} There are no available Data Chunks, wait for WCM to release resources cpDmaChunksAmount {} "
                                  "commandsDataChunksAmount {}",
                                  m_basicQueueInfo.getDescription(),
                                  cpDmaChunksAmount,
                                  commandsDataChunksAmount);
            _waitForWCM(false /* waitForAllCsdcs*/);
            STAT_GLBL_COLLECT_TIME(streamComputeRetrieveCsDcWcm, globalStatPointsEnum::streamComputeRetrieveCsDcWcm);
        }
    }

    // In any case we would like to update the DC cache
    m_csDcAllocator.updateCache();

    if (g_validateDataChunksUsage && (status == CS_DATA_CHUNKS_STATUS_COMPLETED))
    {
        cpDmaDataChunksDB&    upperCpDataChunks = pCsDataChunks->getCpDmaDataChunks();
        commandsDataChunksDB& lowerCpDataChunks = pCsDataChunks->getCommandsBufferDataChunks();

        LOG_ERR(SYN_API, "Compute-stream's Data-Chunks usage:");
        printDataChunksIds("  Upper CP", upperCpDataChunks);
        printDataChunksIds("  Lower CP", lowerCpDataChunks);
    }

    if ((GCFG_ENABLE_SYN_LAUNCH_PROFILER_ANNOTATIONS.value()) &&
        (status == CS_DATA_CHUNKS_STATUS_COMPLETED))
    {
        std::unique_lock<std::mutex> mtx(m_DBMutex);
        char                         desc[100];
        snprintf(desc,
                 sizeof(desc),
                 "%s DBSize=%ld execType = %d",
                 "retrieveCSDC",
                 m_csDataChunksDb.size(),
                 (int)pCsDataChunks->getCsDcExecutionType());
        PROFILER_MEASURE_TIME(desc)
    }

    STAT_GLBL_COLLECT_TIME(streamComputeRetrieveCsDc, globalStatPointsEnum::streamComputeRetrieveCsDc);
    return status;
}

void QueueComputeQman::notifyRecipeRemoval(InternalRecipeHandle& rRecipeHandle)
{
    LOG_DEBUG(SYN_STREAM, "{}", HLLOG_FUNC);
    {
        std::unique_lock<std::mutex> mutex(m_mutex);
        _waitForRecipeCsdcs(&rRecipeHandle, true /* erase from processors DB*/);
    }
    _waitForRecipeCopyCsdcs(rRecipeHandle.recipeSeqNum);
}

void QueueComputeQman::_waitForRecipeCopyCsdcs(uint64_t recipeId)
{
    QueueCopyQman& rStreamCopy = dynamic_cast<QueueCopyQman&>(m_rStreamCopy);
    rStreamCopy.waitForRecipeCsdcs(recipeId);
}

void QueueComputeQman::_waitForRecipeCsdcs(InternalRecipeHandle* pRecipeHandle, bool eraseFromDb)
{
    LOG_DEBUG(SYN_STREAM, "{}: calling _waitforWCM", HLLOG_FUNC);

    _waitForWCM(false /*waitForAllCsdcs*/, pRecipeHandle);

    STAT_GLBL_START(streamDbMutexDuration);
    std::unique_lock<std::mutex> mutex(m_DBMutex);
    STAT_GLBL_COLLECT_TIME(streamDbMutexDuration, globalStatPointsEnum::streamDbMutexDuration);
    auto recipeProcessorsIter = m_recipeProcessorsDB.find(pRecipeHandle);
    if (recipeProcessorsIter == m_recipeProcessorsDB.end())
    {
        LOG_DEBUG(SYN_STREAM, "{}: recipe is not valid {:#x}", HLLOG_FUNC, TO64(pRecipeHandle));
        return;
    }

    IDynamicInfoProcessor* pDynamicProcessor = recipeProcessorsIter->second;
    tryToReleaseProcessorCsDc(*pDynamicProcessor);

    delete pDynamicProcessor;
    if (eraseFromDb)
    {
        m_recipeProcessorsDB.erase(pRecipeHandle);
    }
}

bool QueueComputeQman::isRecipeHasInflightCsdc(InternalRecipeHandle* pRecipeHandle)
{
    HB_ASSERT(pRecipeHandle != nullptr, "recipeId pointer is null");
    STAT_GLBL_START(streamDbMutexDuration);
    std::unique_lock<std::mutex> mutex(m_DBMutex);
    STAT_GLBL_COLLECT_TIME(streamDbMutexDuration, globalStatPointsEnum::streamDbMutexDuration);
    auto recipeProcessorsIter = m_recipeProcessorsDB.find(pRecipeHandle);

    if (recipeProcessorsIter != m_recipeProcessorsDB.end())
    {
        return (recipeProcessorsIter->second)->isAnyInflightCsdc();
    }
    return false;  // recipeID is not valid
}

void QueueComputeQman::notifyAllRecipeRemoval()
{
    {
        std::unique_lock<std::mutex> mutex(m_mutex);
        LOG_DEBUG(SYN_STREAM, "{}", HLLOG_FUNC);

        for (auto recipeProcessorsEntry : m_recipeProcessorsDB)
        {
            _waitForRecipeCsdcs(recipeProcessorsEntry.first, false /* erase from processors DB*/);
            _waitForRecipeCopyCsdcs(recipeProcessorsEntry.first->recipeSeqNum);
        }
        VALIDATE_PROTECTED_OPERATION(_clearProcessorsDb);
    }
}

void QueueComputeQman::notifyCsCompleted(const WcmCsHandleQueue& rCsHandles, bool csFailed)
{
    for (uint64_t csHandle : rCsHandles)
    {
        _notifyCsCompleted(csHandle, csFailed);
        _wcmReleaseThreadIfNeeded();
    }
}

void QueueComputeQman::_notifyCsCompleted(uint64_t waitForEventHandle, bool csFailed)
{
    STAT_GLBL_START(streamDbMutexWcmDur);
    std::unique_lock<std::mutex> mtx(m_DBMutex);
    STAT_GLBL_COLLECT_TIME(streamDbMutexWcmDur, globalStatPointsEnum::wcmObserverDbMutexDuration);

    std::deque<CommandSubmissionDataChunks*>::const_iterator iter;
    CommandSubmissionDataChunks*                             pCsDataChunks = nullptr;
    for (iter = m_csDataChunksDb.begin(); iter != m_csDataChunksDb.end(); ++iter)
    {
        pCsDataChunks = *iter;
        if (pCsDataChunks->containsHandle(waitForEventHandle))
        {
            break;
        }
    }

    if (iter == m_csDataChunksDb.end())
    {
        LOG_CRITICAL(SYN_STREAM, "{}: Can not find waitForEventHandle {}", HLLOG_FUNC, waitForEventHandle);
        return;
    }

    const uint64_t recipeId = pCsDataChunks->getRecipeId();

    if (iter != m_csDataChunksDb.begin())
    {
        LOG_DEBUG(SYN_STREAM,
                  "{}: unordered CS completion detected waitForEventHandle {:#x} recipeId {:#x}",
                  HLLOG_FUNC,
                  waitForEventHandle,
                  recipeId);
    }

    if (csFailed)
    {
        LOG_CRITICAL(SYN_STREAM,
                     "WCM notified recipe recipeId {:#x} waitForEventHandle {:#x} failure",
                     recipeId,
                     waitForEventHandle);
    }
    else
    {
        LOG_TRACE(SYN_STREAM,
                  "WCM notified recipe recipeId {:#x} waitForEventHandle {:#x} success",
                  recipeId,
                  waitForEventHandle);
    }

    InternalRecipeHandle* pRecipeHandle      = pCsDataChunks->getRecipeHandle();
    auto                  recipeProcessorItr = m_recipeProcessorsDB.find(pRecipeHandle);
    if (recipeProcessorItr == m_recipeProcessorsDB.end())
    {
        LOG_CRITICAL(SYN_STREAM, "Failed to find DynamicInfoProcessor of recipe-ID {}", recipeId);
    }
    else
    {
        IDynamicInfoProcessor* pDynamicInfoProcessor = recipeProcessorItr->second;
        if (!pDynamicInfoProcessor->notifyCsCompleted(pCsDataChunks, waitForEventHandle, csFailed))
        {
            LOG_CRITICAL(SYN_STREAM,
                         "Notification to Stream-Recipe-Processor about CS completion of recipe-ID {} failed",
                         recipeId);
        }
    }

    bool deleteCs = false;
    if ((!pCsDataChunks->isWaitForEventHandleSet()) && (csFailed))
    {
        LOG_DEBUG(SYN_STREAM, "Release pCsDataChunks {:#x} data chunks since it is not use", TO64(pCsDataChunks));

        if (!m_csDcAllocator.releaseDataChunks(pCsDataChunks))
        {
            LOG_ERR(SYN_STREAM,
                    "{}: Failed to release pCsDataChunks {:#x} data chunks",
                    HLLOG_FUNC,
                    TO64(pCsDataChunks));
        }
        deleteCs = true;
    }

    const uint64_t prgDataSubTypeAllocationId = pCsDataChunks->getPrgDataSubTypeAllocationId();
    m_rDevRecipeAddress.notifyDeviceRecipeAddressesAreNotUsed(recipeId,
                                                              REF_COUNT_PER_CS_STAGE,
                                                              prgDataSubTypeAllocationId);

    m_csDataChunksDb.erase(iter);

    LOG_DEBUG(SYN_STREAM,
              "{} with Recipe-ID 0x{:x} CS {:#x} waitForEventHandle {:#x} csFailed {}",
              m_basicQueueInfo.getDescription(),
              recipeId,
              TO64(pCsDataChunks),
              waitForEventHandle,
              csFailed);

    if (deleteCs)
    {
        delete pCsDataChunks;
    }
}

void QueueComputeQman::_tryToReleaseProcessorsCsDc()
{
    for (auto& recipeProcessorsEntry : m_recipeProcessorsDB)
    {
        tryToReleaseProcessorCsDc(*recipeProcessorsEntry.second);
    }
}

bool QueueComputeQman::isRecipeStageCsDcCpDmaReady(uint64_t                           recipeId,
                                                   eExecutionStage                    stage,
                                                   uint64_t                           programCodeHandle,
                                                   const CommandSubmissionDataChunks& rCsDataChunks)
{
    LOG_TRACE(SYN_STREAM,
              "recipeId {}, stage {} programCodeHandle {} pCsDataChunks {:#x}",
              recipeId,
              stage,
              programCodeHandle,
              (uint64_t)&rCsDataChunks);

    if (rCsDataChunks.getRecipeId() != recipeId)
    {
        return false;
    }

    if (rCsDataChunks.getExecutionStage() != stage)
    {
        return false;
    }

    if (rCsDataChunks.getCsDcExecutionType() == CS_DC_EXECUTION_TYPE_NOT_READY)
    {
        return false;
    }

    if (rCsDataChunks.getProgramCodeHandle() != programCodeHandle)
    {
        return false;
    }

    return true;
}

bool QueueComputeQman::_addStaticMapping(AddressRangeMapper& addressRangeMap) const
{
    if (GCFG_ENABLE_STAGED_SUBMISSION.value())
    {
        if (!addressRangeMap.addMapping(m_pStreamMasterHelper->getStreamMasterFenceBufferHandle(),
                                        m_pStreamMasterHelper->getStreamMasterFenceHostAddress(),
                                        m_pStreamMasterHelper->getStreamMasterFenceBufferSize(),
                                        AddressRangeMapper::ARM_MAPPING_TYPE_TAG))
        {
            return false;
        }

        if (!addressRangeMap.addMapping(m_pStreamMasterHelper->getStreamMasterFenceClearBufferHandle(),
                                        m_pStreamMasterHelper->getStreamMasterFenceClearHostAddress(),
                                        m_pStreamMasterHelper->getStreamMasterFenceClearBufferSize(),
                                        AddressRangeMapper::ARM_MAPPING_TYPE_TAG))
        {
            return false;
        }
    }
    else
    {
        if (!addressRangeMap.addMapping(m_pStreamMasterHelper->getStreamMasterBufferHandle(),
                                        m_pStreamMasterHelper->getStreamMasterBufferHostAddress(),
                                        m_pStreamMasterHelper->getStreamMasterBufferSize(),
                                        AddressRangeMapper::ARM_MAPPING_TYPE_TAG))
        {
            return false;
        }
    }

    return true;
}

void QueueComputeQman::_dfaLogCsDcInfo(CommandSubmissionDataChunks* csPtr, int logLevel, bool errorCsOnly)
{
    InternalRecipeHandle* pRecipeHandle = csPtr->getRecipeHandle();
    const uint64_t        recipeId      = csPtr->getRecipeId();

    SYN_LOG_TYPE(SYN_DEV_FAIL, logLevel, SEPARATOR_STR);
    SYN_LOG_TYPE(SYN_DEV_FAIL, logLevel, SEPARATOR_STR);
    SYN_LOG_TYPE(SYN_DEV_FAIL, logLevel, "| in recipe:     {:#x}", TO64(pRecipeHandle));
    SYN_LOG_TYPE(SYN_DEV_FAIL, logLevel, "| csdc has {} launches", csPtr->getAllWaitForEventHandles().size());
    SYN_LOG_TYPE(SYN_DEV_FAIL, logLevel, SEPARATOR_STR);

    auto recipeProcessorItr = m_recipeProcessorsDB.find(pRecipeHandle);  // get dynamicInfoProcessor
    if (recipeProcessorItr == m_recipeProcessorsDB.end())
    {
        LOG_ERR(SYN_DEV_FAIL, "### No recipe found for command submission");
        return;
    }

    const basicRecipeInfo&          rBasicRecipeInfo = recipeProcessorItr->second->getRecipeBasicInfo();
    const DeviceAgnosticRecipeInfo& rDevAgnosticInfo = recipeProcessorItr->second->getDevAgnosticInfo();

    RecipeManager::dfaLogRecipeInfo(*pRecipeHandle);

    if (errorCsOnly)
    {
        RecipeLogger::dfaDumpRecipe(rDevAgnosticInfo,
                                    rBasicRecipeInfo,
                                    false,
                                    recipeId,
                                    m_basicQueueInfo.getDescription());
    }
}

synStatus QueueComputeQman::getDynamicShapesTensorInfoArray(synRecipeHandle             recipeHandle,
                                                            std::vector<tensor_info_t>& tensorInfoArray) const
{
    VERIFY_IS_NULL_POINTER(SYN_API, recipeHandle, "recipeHandle");
    IDynamicInfoProcessor* pRecipeProcessor;

    STAT_GLBL_START(streamDbMutexDuration);
    std::unique_lock<std::mutex> lock(m_DBMutex);
    STAT_GLBL_COLLECT_TIME(streamDbMutexDuration, globalStatPointsEnum::streamDbMutexDuration);

    auto recipeProcessorItr    = m_recipeProcessorsDB.find(recipeHandle);  // get dynamicInfoProcessor
    auto recipeProcessorEndItr = m_recipeProcessorsDB.end();
    if (recipeProcessorItr == recipeProcessorEndItr)
    {
        LOG_ERR(SYN_STREAM,
                "{}: Stream 0x{:x} recipe 0x{:x} doesn't exist",
                HLLOG_FUNC,
                TO64(this),
                TO64(recipeHandle));
        return synFail;
    }

    pRecipeProcessor = recipeProcessorItr->second;

    tensorInfoArray = pRecipeProcessor->getDynamicShapesTensorInfoArray();
    return synSuccess;
}

synStatus QueueComputeQman::_runSifPreDownload(const basicRecipeInfo&          rBasicRecipeInfo,
                                               uint64_t                        recipeId,
                                               const DeviceAgnosticRecipeInfo& rDeviceAgnosticRecipeInfo,
                                               const RecipeStaticInfo&         rRecipeStaticInfo,
                                               InternalRecipeHandle*           pRecipeHandle,
                                               const synLaunchTensorInfoExt*   pLaunchTensorsInfo,
                                               uint32_t                        launchTensorsAmount,
                                               uint64_t                        programDataHostAddress)
{
    IDynamicInfoProcessor* pRecipeProcessor = nullptr;

    synStatus status = retrieveDynamicInfoProcessor(rBasicRecipeInfo,
                                                    rDeviceAgnosticRecipeInfo,
                                                    rRecipeStaticInfo,
                                                    pRecipeHandle,
                                                    recipeId,
                                                    pRecipeProcessor);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_STREAM,
                "{}: Cant perform launch on stream {} (retrieve processor IH2D)",
                HLLOG_FUNC,
                m_basicQueueInfo.getDescription());
        return synFail;
    }

    DynamicRecipe* pDsdPatcher = pRecipeProcessor->getDsdPatcher();
    HB_ASSERT_PTR(pDsdPatcher);

    if (!pDsdPatcher->takeOwnership())
    {
        // LOG defined inside the method itself
        return synFail;
    }

    std::vector<uint32_t>* pTensorIdx2userIdx;
    bool res = pRecipeProcessor->resolveTensorsIndices(pTensorIdx2userIdx, launchTensorsAmount, pLaunchTensorsInfo);
    if (!res)
    {
        LOG_DSD_ERR("Failed to resolve tensors' indices");
        pDsdPatcher->releaseOwnership();
        return synFail;
    }

    res = pDsdPatcher->runSifOnAllNodes(pLaunchTensorsInfo,
                                        launchTensorsAmount,
                                        pTensorIdx2userIdx,
                                        programDataHostAddress);
    if (!res)
    {
        LOG_DSD_ERR("DSD SIF failed");
        pDsdPatcher->releaseOwnership();
        return synFailedDynamicPatching;
    }

    return synSuccess;
}

void QueueComputeQman::releaseSifOwnership(const basicRecipeInfo&          rBasicRecipeInfo,
                                           const DeviceAgnosticRecipeInfo& rDeviceAgnosticRecipeInfo,
                                           const RecipeStaticInfo&         rRecipeStaticInfo,
                                           InternalRecipeHandle*           pRecipeHandle,
                                           uint64_t                        recipeId)
{
    // SW-74708 Soften the lock on the workspace address
    STAT_GLBL_START(streamComputeMutexDuration);
    std::unique_lock<std::mutex> mutex(m_mutex);
    STAT_GLBL_COLLECT_TIME(streamComputeMutexDuration, globalStatPointsEnum::streamComputeMutexDuration);

    IDynamicInfoProcessor* pRecipeProcessor = nullptr;
    synStatus              status           = retrieveDynamicInfoProcessor(rBasicRecipeInfo,
                                                    rDeviceAgnosticRecipeInfo,
                                                    rRecipeStaticInfo,
                                                    pRecipeHandle,
                                                    recipeId,
                                                    pRecipeProcessor);
    if (status == synSuccess)
    {
        DynamicRecipe* pDsdPatcher = pRecipeProcessor->getDsdPatcher();
        HB_ASSERT_PTR(pDsdPatcher);
        pDsdPatcher->releaseOwnership();
    }
    else
    {
        LOG_ERR(SYN_STREAM,
                "{}: We cannot release ownership w/o the DIP {} (retrieve processor)",
                HLLOG_FUNC,
                m_basicQueueInfo.getDescription());
    }
}

synStatus QueueComputeQman::_processRecipe(DeviceRecipeDownloaderInterface& rDeviceRecipeDownloader,
                                           std::vector<uint64_t>&           executionBlocksDeviceAddresses,
                                           uint64_t                         programCodeHandle,
                                           uint64_t                         programCodeDeviceAddress,
                                           uint64_t                         workspaceAddress,
                                           bool                             programCodeInCache)
{
    synStatus status = rDeviceRecipeDownloader.processRecipe(workspaceAddress,
                                                             m_csDcAllocator.getDcSizeCpDma(),
                                                             m_csDcAllocator.getDcSizeCommand(),
                                                             programCodeInCache,
                                                             programCodeHandle,
                                                             programCodeDeviceAddress,
                                                             executionBlocksDeviceAddresses);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_STREAM,
                "{}: stream {} downloadProgramDataBufferCache failed with status {}",
                HLLOG_FUNC,
                m_basicQueueInfo.getDescription(),
                status);
    }

    return status;
}

synStatus QueueComputeQman::_downloadBuffersToCache(DeviceRecipeDownloaderInterface& rDeviceRecipeDownloader,
                                                    std::vector<uint64_t>&           executionBlocksDeviceAddresses,
                                                    uint64_t                         programCodeHandle,
                                                    uint64_t                         programDataHandle,
                                                    uint64_t                         programDataDeviceAddress,
                                                    uint64_t                         workspaceAddress,
                                                    SpRecipeProgramBuffer            programDataRecipeBuffer,
                                                    bool                             programCodeInCache,
                                                    bool                             programDataInCache)
{
    // We will DWL the PRG-Data last, so in case it returns a failure, we could know that the PRG-Data host-buffer
    // could be deleted

    synStatus status = rDeviceRecipeDownloader.downloadExecutionBufferCache(programCodeInCache,
                                                                            programCodeHandle,
                                                                            executionBlocksDeviceAddresses);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_STREAM,
                "{}: stream {} downloadExecutionBufferCache failed with status {}",
                HLLOG_FUNC,
                m_basicQueueInfo.getDescription(),
                status);

        return status;
    }

    status = rDeviceRecipeDownloader.downloadProgramDataBufferCache(workspaceAddress,
                                                                    programDataInCache,
                                                                    programDataHandle,
                                                                    programDataDeviceAddress,
                                                                    programDataRecipeBuffer);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_STREAM,
                "{}: stream {} downloadProgramDataBufferCache failed with status {}",
                HLLOG_FUNC,
                m_basicQueueInfo.getDescription(),
                status);
    }

    return status;
}

synStatus QueueComputeQman::_downloadBuffersToWorkspace(DeviceRecipeDownloaderInterface& rDeviceRecipeDownloader,
                                                        bool&                            isSyncWithDmaSynapseRequired,
                                                        uint64_t                         recipeId,
                                                        SpRecipeProgramBuffer            programDataRecipeBuffer,
                                                        uint64_t                         programCodeHandle,
                                                        uint64_t                         programDataHandle,
                                                        uint64_t                         programCodeDeviceAddress,
                                                        uint64_t                         programDataDeviceAddress,
                                                        bool                             programCodeInCache,
                                                        bool                             programDataInCache)
{
    synStatus status = synSuccess;

    // We will DWL the PRG-Data last, so in case it returns a failure, we could know that the PRG-Data host-buffer
    // could be deleted

    // DWL PRG-Code
    {
        const bool isProgramCodeAlreadyDownloaded =
            ((m_lastWsDwldPrgCodeRecipeId == recipeId) && (m_lastWsDwldPrgCodeAddress == programCodeDeviceAddress));

        bool isDownloadWorkspaceProgramCode;
        status = rDeviceRecipeDownloader.downloadExecutionBufferWorkspace(this,
                                                                          programCodeInCache,
                                                                          programCodeHandle,
                                                                          programCodeDeviceAddress,
                                                                          isDownloadWorkspaceProgramCode);
        if (status != synSuccess)
        {
            LOG_ERR(SYN_STREAM,
                    "{}: stream {} downloadExecutionBufferWorkspace failed with status {}",
                    HLLOG_FUNC,
                    m_basicQueueInfo.getDescription(),
                    status);
            return status;
        }

        if (isDownloadWorkspaceProgramCode)
        {
            m_lastWsDwldPrgCodeAddress   = programCodeDeviceAddress;
            m_lastWsDwldPrgCodeRecipeId  = recipeId;
            isSyncWithDmaSynapseRequired = true;
        }
    }

    // DWL PRG-Data
    {
        const bool isProgramDataAlreadyDownloaded =
            ((m_lastWsDwldPrgDataRecipeId == recipeId) && (m_lastWsDwldPrgDataAddress == programDataDeviceAddress));

        bool isDownloadWorkspaceProgramData;

        status = rDeviceRecipeDownloader.downloadProgramDataBufferWorkspace(this,
                                                                            programDataInCache,
                                                                            programDataHandle,
                                                                            programDataDeviceAddress,
                                                                            isDownloadWorkspaceProgramData,
                                                                            programDataRecipeBuffer);
        if (status != synSuccess)
        {
            LOG_ERR(SYN_STREAM,
                    "{}: stream {} downloadProgramDataBufferWorkspace failed with status {}",
                    HLLOG_FUNC,
                    m_basicQueueInfo.getDescription(),
                    status);

            return status;
        }

        if (isDownloadWorkspaceProgramData)
        {
            m_lastWsDwldPrgDataAddress   = programDataDeviceAddress;
            m_lastWsDwldPrgDataRecipeId  = recipeId;
            isSyncWithDmaSynapseRequired = true;
        }
    }

    return status;
}

synStatus QueueComputeQman::_syncPreOperation(bool&                         rIsSyncWithDmaSynapseRequired,
                                              const IDynamicInfoProcessor*& pRecipeProcessor,
                                              uint64_t                      programCodeHandle,
                                              uint64_t                      programDataHandle)
{
    synStatus status = synSuccess;

    uint64_t lastProgramCodeHandle;
    pRecipeProcessor->getProgramCodeHandle(lastProgramCodeHandle);

    uint64_t lastProgramDataHandle;
    pRecipeProcessor->getProgramDataHandle(lastProgramDataHandle);

    if ((programDataHandle != lastProgramDataHandle) || (programCodeHandle != lastProgramCodeHandle))
    {
        rIsSyncWithDmaSynapseRequired = true;
    }

    // Load is using the Synapse DMA stream, while the Activate and Launch are using the local stream
    // Hence, a stream-sync is required in case load CS had been submitted
    if (rIsSyncWithDmaSynapseRequired)
    {
        m_isSyncWithDmaSynapseRequired = true;
    }

    // It might be that there is no new load, but we need to sync with the previous load operation
    if (m_isSyncWithDmaSynapseRequired)
    {
        status = _syncPreOperation(m_rStreamCopy);
        if (status != synSuccess)
        {
            LOG_ERR(SYN_STREAM,
                    "{}: Cant to perform post-load stream-synchronization on stream {}",
                    HLLOG_FUNC,
                    m_basicQueueInfo.getDescription());
            return synFail;
        }
        m_isSyncWithDmaSynapseRequired = false;
    }

    return synSuccess;
}

synStatus QueueComputeQman::_activateAndEnqueue(const synLaunchTensorInfoExt*&  launchTensorsInfo,
                                                uint32_t                        launchTensorsAmount,
                                                basicRecipeInfo&                basicRecipeHandle,
                                                const DeviceAgnosticRecipeInfo& rDeviceAgnosticRecipeInfo,
                                                const RecipeStaticInfo&         rRecipeStaticInfo,
                                                uint64_t                        recipeId,
                                                InternalRecipeHandle*           pRecipeHandle,
                                                IDynamicInfoProcessor*&         pRecipeProcessor,
                                                uint64_t                        workspaceAddress,
                                                uint64_t                        programCodeDeviceAddress,
                                                uint64_t                        programDataDeviceAddress,
                                                uint64_t                        programCodeHandle,
                                                uint64_t                        programDataHandle,
                                                uint64_t                        prgDataSubTypeAllocId,
                                                uint64_t                        assertAsyncMappedAddress,
                                                uint64_t&                       refCount,
                                                uint32_t                        flags,
                                                bool                            programCodeInCache,
                                                bool                            programDataInCache,
                                                EventWithMappedTensorDB&        events)
{
    synStatus status      = synSuccess;
    bool      isSubmitted = false;

    LOG_DEBUG(SYN_STREAM,
              "Stream {:#x} launch recipe {:#x} workspace {:#x} programDataHandle {:#x} programCodeHandle {:#x} "
              "programDataDeviceAddress {:#x} programCodeDeviceAddress {:#x} programCodeInCache {} programDataInCache "
              "{} m_isSyncWithDmaSynapseRequired {}",
              TO64(this),
              TO64(basicRecipeHandle.recipe),
              workspaceAddress,
              programDataHandle,
              programCodeHandle,
              programDataDeviceAddress,
              programCodeDeviceAddress,
              programCodeInCache,
              programDataInCache,
              m_isSyncWithDmaSynapseRequired);

    // Activate
    eCsDcProcessorStatus   csDcProcessingStatus  = CS_DC_PROCESSOR_STATUS_FAILED;
    eAnalyzeValidateStatus analyzeValidateStatus = ANALYZE_VALIDATE_STATUS_DO_ANALYZE;

    // Launch
    STAT_GLBL_START(streamComputeEnqueueAndSync);
    status = _enqueueAndSync(launchTensorsInfo,
                             launchTensorsAmount,
                             basicRecipeHandle,
                             rDeviceAgnosticRecipeInfo,
                             rRecipeStaticInfo,
                             recipeId,
                             pRecipeHandle,
                             pRecipeProcessor,
                             programDataHandle,
                             programCodeHandle,
                             workspaceAddress,
                             prgDataSubTypeAllocId,
                             assertAsyncMappedAddress,
                             flags,
                             programCodeInCache,
                             programDataInCache,
                             programDataDeviceAddress,
                             programCodeDeviceAddress,
                             analyzeValidateStatus,
                             events,
                             csDcProcessingStatus);
    STAT_GLBL_COLLECT_TIME(streamComputeEnqueueAndSync, globalStatPointsEnum::streamComputeEnqueueAndSync);

    if (status == synBusy)
    {
        LOG_DEBUG(SYN_STREAM,
                  "{}: Failed to perform enqueue on stream {}, device is busy",
                  HLLOG_FUNC,
                  m_basicQueueInfo.getDescription());
        // Todo [SW-103414] Call notifyDeviceRecipeAddressesAreNotUsed on each StreamComputeLaunch return failure
        return status;
    }
    if (status != synSuccess)
    {
        LOG_ERR(SYN_STREAM,
                "{}: cannot enqueue on stream {} {}",
                HLLOG_FUNC,
                m_basicQueueInfo.getDescription(),
                status);
        return status;
    }

    return status;
}

void QueueComputeQman::tryToReleaseCommandSubmissionDataChunk(CommandSubmissionDataChunks*& pCsDataChunks)
{
    if (pCsDataChunks->getCsDcExecutionType() == CS_DC_EXECUTION_TYPE_NOT_READY)
    {
        if (!m_csDcAllocator.releaseDataChunks(pCsDataChunks))
        {
            LOG_ERR(SYN_STREAM,
                    "{}: Failed to release pCsDataChunks {:#x} data chunks",
                    HLLOG_FUNC,
                    TO64(pCsDataChunks));
        }
        delete pCsDataChunks;
    }
}

void QueueComputeQman::tryToReleaseProcessorCsDc(IDynamicInfoProcessor& rRecipeProcessor)
{
    const uint32_t releaseAll = std::numeric_limits<uint32_t>::max();

    IDynamicInfoProcessor::CommandSubmissionDataChunksVec releasedElements;
    rRecipeProcessor.releaseCommandSubmissionDataChunks(releaseAll, releasedElements, false);

    for (auto pCsDataChunks : releasedElements)
    {
        if (pCsDataChunks->isWaitForEventHandleSet())
        {
            LOG_DEBUG(SYN_STREAM, "CS Data-Chunk is still in use. Can not clear its DCs");
            continue;
        }

        DataChunksAmounts dcAmountsAvailable {0, 0};
        if (!m_csDcAllocator.cleanupCsDataChunk(pCsDataChunks, dcAmountsAvailable))
        {
            LOG_ERR(SYN_STREAM,
                    "{}: Failed to release pCsDataChunks {:#x} data chunks",
                    HLLOG_FUNC,
                    TO64(pCsDataChunks));
        }
        delete pCsDataChunks;
    }
}

eCsDataChunkStatus QueueComputeQman::tryToRetrieveCsDc(synDeviceType                             deviceType,
                                                       const RecipeHandleToRecipeProcessorMap&   rRecipeProcessorsDB,
                                                       const RecipeTensorsInfo&                  rRecipeTensorInfo,
                                                       IDynamicInfoProcessor&                    rRecipeProcessor,
                                                       uint64_t                                  recipeId,
                                                       const recipe_t&                           rRecipe,
                                                       InternalRecipeHandle*                     pRecipeHandle,
                                                       uint64_t                                  cpDmaChunksAmount,
                                                       uint64_t                                  commandsDataChunksAmount,
                                                       bool                                      isOldCsDcReuseAllowed,
                                                       eExecutionStage                           stage,
                                                       bool                                      isFirst,
                                                       bool                                      isLast,
                                                       CommandSubmissionDataChunks*&             pCsDataChunks)
{
    HB_ASSERT(stage != EXECUTION_STAGE_LAST, "Illegal execution stage {}", stage);

    if ((cpDmaChunksAmount == 0) && (commandsDataChunksAmount == 0))
    {
        const bool createStatus =
            createCsDc(deviceType, pCsDataChunks, rRecipeTensorInfo, recipeId, rRecipe, pRecipeHandle, stage);

        return (createStatus) ? CS_DATA_CHUNKS_STATUS_COMPLETED : CS_DATA_CHUNKS_STATUS_FAILURE;
    }

    // First we want to retry re-using old CS-DC of any other execution-handle, for current recipe
    if (isOldCsDcReuseAllowed)
    {
        pCsDataChunks = rRecipeProcessor.getAvailableCommandSubmissionDataChunks(stage, false);
        if (pCsDataChunks != nullptr)
        {
            STAT_GLBL_COLLECT(1, csdcPartialReuse);
            LOG_TRACE(SYN_STREAM, "{}: Partial re-use CS-DC {:#x}", HLLOG_FUNC, (uint64_t)pCsDataChunks);
            return CS_DATA_CHUNKS_STATUS_COMPLETED;
        }
    }

    DataChunksAmounts dcAmountsAvailable = m_csDcAllocator.getDataChunksAmounts();

    const DataChunksAmounts dcAmountsRequired {cpDmaChunksAmount, commandsDataChunksAmount};

    DataChunksDBs dcDbs;

    // Try to acquire DataChunks
    // Request stream-recipe-processors to release CS-DCs, until having enough free DC items
    // If that fails, we will try to acquire new elements (only on the last time)
    if (!isFirst)
    {
        STAT_GLBL_START(streamComputeRetrieveCsDcRelease);
        const bool releaseStatus =
            releaseCsDataChunksFromProcessors(rRecipeProcessorsDB, dcAmountsAvailable, dcDbs, dcAmountsRequired);
        if (releaseStatus == false)
        {
            LOG_DEBUG(SYN_STREAM, "{}: Failure during acquire Data-Chunks", HLLOG_FUNC);
            return CS_DATA_CHUNKS_STATUS_FAILURE;
        }
        STAT_GLBL_COLLECT_TIME(streamComputeRetrieveCsDcRelease, globalStatPointsEnum::streamComputeRetrieveCsDcRelease);
    }

    STAT_GLBL_START(streamComputeRetrieveCsDcAcquire);

    // Acquire DCs for the CP-DMAs and the Commands
    const eCsDataChunkStatus acquireStatus =
        m_csDcAllocator.tryToAcquireDataChunkAmounts(dcAmountsAvailable, dcDbs, dcAmountsRequired, isLast);

    STAT_GLBL_COLLECT_TIME(streamComputeRetrieveCsDcAcquire, globalStatPointsEnum::streamComputeRetrieveCsDcAcquire);

    switch (acquireStatus)
    {
        case CS_DATA_CHUNKS_STATUS_FAILURE:
        {
            LOG_DEBUG(SYN_STREAM, "{}: Failure during acquire Data-Chunks", HLLOG_FUNC);
            break;
        }
        case CS_DATA_CHUNKS_STATUS_NOT_COMPLETED:
        {
            LOG_DEBUG(SYN_STREAM, "{}: All Stream's CS elements are busy (can not acquire Data-Chunks)", HLLOG_FUNC);
            break;
        }
        case CS_DATA_CHUNKS_STATUS_COMPLETED:
        {
            LOG_TRACE(SYN_STREAM, "{}: No re-use CS-DC {:#x}", HLLOG_FUNC, (uint64_t)pCsDataChunks);
            const bool createStatus =
                createCsDc(deviceType, pCsDataChunks, rRecipeTensorInfo, recipeId, rRecipe, pRecipeHandle, stage);

            if (!createStatus)
            {
                return CS_DATA_CHUNKS_STATUS_FAILURE;
            }

            // set command DC in CS
            pCsDataChunks->addCpDmaDataChunks(dcDbs[CS_DC_ALLOCATOR_CP_DMA]);
            pCsDataChunks->addProgramBlobsDataChunks(dcDbs[CS_DC_ALLOCATOR_COMMAND]);
            break;
        }
        default:
        {
            HB_ASSERT(false, "{}: acquireStatus {}", __FUNCTION__, acquireStatus);
        }
    }

    return acquireStatus;
}

bool QueueComputeQman::releaseCsDataChunksFromProcessors(const RecipeHandleToRecipeProcessorMap& rRecipeProcessorsDB,
                                                         DataChunksAmounts&                      dcAmountsAvailable,
                                                         DataChunksDBs&                          dcDbs,
                                                         const DataChunksAmounts&                dcAmountsRequired)
{
    uint64_t totalAvailableDataChunksPre = dcAmountAvailableCpDma + dcAmountAvailableCommand;
    LOG_DEBUG(SYN_STREAM,
              "{}: Available Data Chunks (pre) {} CP_DMA {} COMMAND {}",
              HLLOG_FUNC,
              totalAvailableDataChunksPre,
              dcAmountAvailableCpDma,
              dcAmountAvailableCommand);

    for (int i = 0; i < 2; i++)
    {
        bool keepOne = (i == 0);

        while ((dcAmountAvailableCpDma < dcAmountRequiredDma) || (dcAmountAvailableCommand < dcAmountRequiredCommand))
        {
            auto recipeProcessorItr    = rRecipeProcessorsDB.begin();
            auto recipeProcessorEndItr = rRecipeProcessorsDB.end();

            DataChunksAmounts dcAmountsReleased {0, 0};

            // Although it might be that we can release only some, we will perform a full single "loop",
            // and release items from each SRP
            while (recipeProcessorItr != recipeProcessorEndItr)
            {
                IDynamicInfoProcessor::CommandSubmissionDataChunksVec releasedElements;
                releasedElements.reserve(numOfElementsToRelease);

                IDynamicInfoProcessor* pStreamInfoProcessor = recipeProcessorItr->second;

                pStreamInfoProcessor->releaseCommandSubmissionDataChunks(numOfElementsToRelease,
                                                                         releasedElements,
                                                                         keepOne);

                for (auto pCsDataChunks : releasedElements)
                {
                    if (pCsDataChunks->isWaitForEventHandleSet())
                    {
                        LOG_DEBUG(SYN_STREAM, "CS Data-Chunk is still in use. Can not clear its DCs");
                        continue;
                    }

                    if (!m_csDcAllocator.cleanupCsDataChunk(pCsDataChunks, dcAmountsReleased))
                    {
                        LOG_ERR(SYN_STREAM,
                                "{}: Failed to release pCsDataChunks {:#x} data chunks",
                                HLLOG_FUNC,
                                TO64(pCsDataChunks));
                    }
                    delete pCsDataChunks;
                }
                recipeProcessorItr++;
            }

            if ((dcAmountsReleased[CS_DC_ALLOCATOR_CP_DMA] == 0) && (dcAmountsReleased[CS_DC_ALLOCATOR_COMMAND] == 0))
            {
                // No additional CS-DCs that could be released
                break;
            }

            dcAmountAvailableCpDma += dcAmountsReleased[CS_DC_ALLOCATOR_CP_DMA];
            dcAmountAvailableCommand += dcAmountsReleased[CS_DC_ALLOCATOR_COMMAND];
            STAT_GLBL_COLLECT(dcAmountsReleased[CS_DC_ALLOCATOR_CP_DMA] + dcAmountsReleased[CS_DC_ALLOCATOR_COMMAND],
                         csdcStreamComputeRelease);
        }  // while(not enough)
    }      // for

    uint64_t totalAvailableDataChunksPost = dcAmountAvailableCpDma + dcAmountAvailableCommand;
    LOG_DEBUG(SYN_STREAM,
              "{}: Available Data Chunks (post release) {} CP_DMA {} COMMAND {}",
              HLLOG_FUNC,
              totalAvailableDataChunksPost,
              dcAmountAvailableCpDma,
              dcAmountAvailableCommand);
    return true;
}

bool QueueComputeQman::createCsDc(synDeviceType                 deviceType,
                                  CommandSubmissionDataChunks*& pCsDataChunks,
                                  const RecipeTensorsInfo&      rRecipeTensorInfo,
                                  uint64_t                      recipeId,
                                  const recipe_t&               rRecipe,
                                  InternalRecipeHandle*         pRecipeHandle,
                                  eExecutionStage               stage)
{
    size_t csDcMappingDbSize = 0;
    switch (deviceType)
    {
        case synDeviceGaudi:
            csDcMappingDbSize = (size_t)gaudi_queue_id::GAUDI_QUEUE_ID_SIZE;
            break;
        default:
            LOG_ERR(SYN_STREAM, "{}: unsupported device type {}", HLLOG_FUNC, deviceType);
            return CS_DATA_CHUNKS_STATUS_FAILURE;
    }

    const uint64_t maxSectionId       = rRecipeTensorInfo.m_maxSectionId;
    const uint64_t numSectionsToPatch = rRecipeTensorInfo.m_numSectionsToPatch;

    try
    {
        pCsDataChunks = new CommandSubmissionDataChunks(CS_DC_TYPE_COMPUTE,
                                                        deviceType,
                                                        csDcMappingDbSize,
                                                        recipeId,
                                                        rRecipe,
                                                        pRecipeHandle,
                                                        stage,
                                                        maxSectionId,
                                                        numSectionsToPatch);
    }
    catch (const SynapseException& err)
    {
        LOG_DEBUG(SYN_STREAM, "{}: Failure during acquire Data-Chunks (failed to create CS-DC)", HLLOG_FUNC);
        return false;
    }

    return true;
}
