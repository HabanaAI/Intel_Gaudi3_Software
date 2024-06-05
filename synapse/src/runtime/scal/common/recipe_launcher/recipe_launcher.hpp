#pragma once

#include "runtime/scal/common/recipe_launcher/recipe_launcher_interface.hpp"
#include "arc_hbm_mem_mgr.hpp"
#include "define_synapse_common.hpp"
#include "hbm_global_mem_mgr.hpp"
#include "launch_tracker.hpp"
#include "mapped_mem_mgr.hpp"
#include "synapse_common_types.h"
#include "utils.h"

#include "runtime/common/recipe/recipe_patch_processor.hpp"
#include "runtime/common/queues/queue_compute_utils.hpp"

#include "runtime/scal/common/infra/scal_types.hpp"

#include "runtime/scal/common/recipe_static_info_scal.hpp"

struct recipe_t;
class ScalMemoryPool;
class QueueComputeScal;
class DevMemoryAllocInterface;
class ScalDev;
struct InternalRecipeHandle;
class LaunchTracker;
class MemMgrs;
class DynamicRecipe;
class ScalStreamCopyInterface;
class ScalStreamComputeInterface;

struct LaunchInfo
{
    LaunchInfo() = delete;
    LaunchInfo(const synLaunchTensorInfoExt* enqueueTensorsInfo,
               uint32_t                      enqueueTensorsInfoAmount,
               uint64_t                      workspaceAddress,
               uint64_t                      assertAsyncMappedAddress,
               InternalRecipeHandle*         pRecipeHandle,
               uint32_t                      launchFlags,
               const ScalDevSpecificInfo&    devSpecificInfo,
               EventWithMappedTensorDB&      events,
               uint8_t                       apiId)
    : pEnqueueTensorsInfo(enqueueTensorsInfo),
      pEnqueueTensorsInfoAmount(enqueueTensorsInfoAmount),
      workspaceAddress(workspaceAddress),
      pRecipeHandle(pRecipeHandle),
      launchFlags(launchFlags),
      devSpecificInfo(devSpecificInfo),
      events(events),
      m_assertAsyncMappedAddress(assertAsyncMappedAddress),
      m_apiId(apiId)
    {
    }

    LaunchInfo(LaunchInfo const&) = delete;

    LaunchInfo& operator=(LaunchInfo const&) = delete;

    const synLaunchTensorInfoExt* pEnqueueTensorsInfo;
    const uint32_t                pEnqueueTensorsInfoAmount;
    const uint64_t                workspaceAddress;
    const InternalRecipeHandle*   pRecipeHandle;
    const uint32_t                launchFlags;
    const ScalDevSpecificInfo&    devSpecificInfo;
    EventWithMappedTensorDB&      events;
    const uint64_t                m_assertAsyncMappedAddress;
    const uint8_t                 m_apiId;
};

/**********************************************************************************/
/* An instance of this class (RecipeLauncher) is created for every synLaunch      */
/* and it is responsible to handle the launch. Part of it (m_pre) is needed only  */
/* until the work is sent to the device, some of it (m_pre) is kept for after the */
/* work returns from the device. For debug, we sometimes keep the m_pre until     */
/* the work is done on the device                                                 */
/**********************************************************************************/
class RecipeLauncher : public RecipeLauncherInterface
{
public:
    RecipeLauncher(ScalStreamComputeInterface*     pComputeScalStream,
                   const ComputeCompoundResources* pComputeResources,
                   DevMemoryAllocInterface&        devMemoryAlloc,
                   MemMgrs&                        memMgrs,
                   LaunchTrackerInterface&         rLaunchTracker,
                   const InternalRecipeHandle*     pRecipeHandle,
                   DynamicRecipe*                  pDynamicRecipeProcessor,
                   uint64_t                        runningId,
                   uint8_t                         apiId);

    virtual ~RecipeLauncher() {}

    bool isCopyNotCompleted() const override { return m_longSoCopy != LongSoEmpty; };

    synStatus checkCompletionCopy(uint64_t timeout) override;

    synStatus checkCompletionCompute(uint64_t timeout) override;

    std::string getDescription() const override;

    bool dfaLogDescription(bool               oldestRecipeOnly,
                           uint64_t           currentLongSo,
                           bool               dumpRecipe,
                           const std::string& callerMsg,
                           bool               forTools) const override;

    virtual const InternalRecipeHandle& getInternalRecipeHandle() const override { return *m_pRecipeHandle; }

    synStatus launch(const LaunchInfo& launchInfo);

private:
    void updateSfgLongSos(uint64_t nbExtTensors, EventWithMappedTensorDB& events);

    synStatus scalEnqueue(uint64_t nbExtTensors);

    synStatus prepareForDownload(const LaunchInfo& launchInfo);

    synStatus analyzeTensors(const LaunchInfo& launchInfo);

    synStatus runSifPreDownload(const LaunchInfo& launchInfo);

    synStatus downloadToDev(const LaunchInfo& launchInfo);

    synStatus validateSectionsInfo(const LaunchInfo& launchInfo);

    synStatus debugCompareRecipeOnDev(const InternalRecipeHandle* recipeHandle, const std::string msg);

    synStatus getMappedMemoryInfo();

    unsigned getMemDownloadParamsCountForPdmaDownload(uint64_t sizeDc, uint64_t offsetMapped, uint64_t size);

    void      getMemDownloadParamsForPdmaDownload(uint64_t                   hbmAddr,
                                                  const MemoryMappedAddrVec& mapped,
                                                  uint64_t                   sizeDc,
                                                  uint64_t                   offsetMapped,
                                                  uint64_t                   size,
                                                  internalMemcopyParams&     memDownloadParams);
    synStatus scalMemDownload(internalMemcopyParams const& memcopyParams);

    synStatus pdmaDownload(const MemorySectionsScal& rSections);

    synStatus patch(const LaunchInfo& launchInfo);

    synStatus addBaseAddresses(const EngineGrpArr& engineGrpArr);

    synStatus addEcbListWithBarrierAndSend(uint64_t nbExtTensors);

    // Needed until launch (unless for debug)

    uint64_t getHbmAddr(SectionType sectionNum) const;

    void handleKernelsPrintf(const LaunchInfo& launchInfo);

    void clearMemMgrs();

    inline ScalStreamComputeInterface* getComputeScalStream() { return m_pComputeScalStream; };
    inline ScalStreamCopyInterface*    getRxCommandsScalStream() { return m_pComputeResources->m_pRxCommandsStream; };
    inline ScalStreamCopyInterface*    getTxCommandsScalStream() { return m_pComputeResources->m_pTxCommandsStream; };
    inline ScalStreamCopyInterface*    getDev2DevCommandsScalStream()
    {
        return m_pComputeResources->m_pDev2DevCommandsStream;
    };

    inline const ScalStreamComputeInterface* getComputeScalStream() const { return m_pComputeScalStream; };

    // needed before queueing on device
    ScalLongSyncObject          m_longSoHbmBuff;
    DevMemoryAllocInterface&    m_devMemoryAlloc;
    LaunchTrackerInterface&     m_rLaunchTracker;
    const InternalRecipeHandle* m_pRecipeHandle;
    MemorySectionsScal          m_sections;
    WorkSpacesInformation       m_wsInfo;
    std::vector<uint32_t>       m_tensorIdx2userIdx[tensor_info_t::ETensorType::INTERNAL_TENSOR];

    // need after queueing on device
    MemMgrs&                        m_memMgrs;
    ScalStreamComputeInterface*     m_pComputeScalStream;
    const ComputeCompoundResources* m_pComputeResources;
    ScalLongSyncObject              m_longSoCopy;
    ScalLongSyncObject              m_longSoCompute;
    const EntryIds                  m_entryIds;
    const bool                      m_isDsd;
    DynamicRecipe*                  m_pDynamicRecipeProcessor;
    const bool                      m_isIH2DRecipe;
    const uint8_t                   m_apiId;

    uint64_t m_computeWorkCompletionAddress = 0;
    uint32_t m_computeWorkCompletionValue   = 0;
};
