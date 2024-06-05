/*
.
1) This class is used for Memcopy operations
.
.
2) Usage of DCs (DC = Data Chunks) (execution operation) -
    a) Memcopy operation:
    We need to acquire CB DCs per execution (CB = Command Buffer)
.
.
3) Release of DCs due to a CS DC release -
    a) Memcopy operation:
    We need to clear the CB DCs
*/
#pragma once

#include "queue_base_qman_wcm.hpp"
#include "runtime/qman/common/data_chunk/data_chunks_allocator.hpp"

#include <memory>

class PhysicalQueuesManager;
class CommandSubmissionDataChunks;
class MemoryManager;
class PoolMemoryMapper;
class DevMemoryAllocInterface;

class RecipeProgramBuffer;
typedef std::shared_ptr<RecipeProgramBuffer> SpRecipeProgramBuffer;

namespace generic
{
class CommandBufferPktGenerator;
}

class QueueCopyQman : public QueueBaseQmanWcm
{
public:
    QueueCopyQman(const BasicQueueInfo&           rBasicQueueInfo,
                  uint32_t                        physicalQueueOffset,
                  synDeviceType                   deviceType,
                  PhysicalQueuesManagerInterface* pPhysicalStreamsManager,
                  WorkCompletionManagerInterface& rWorkCompletionManager,
                  DevMemoryAllocInterface&        rDevMemAlloc);

    virtual ~QueueCopyQman();

    virtual synStatus getMappedMemorySize(uint64_t& mappedMemorySize) const override;

    // TODO: Create internalMemcopyDescription which will include the memcopy unique parameters
    // Stream Common
    virtual synStatus memcopy(internalMemcopyParams& memcpyParams,
                              const internalDmaDir   direction,
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

    virtual bool isArbitrationDmaNeeded() { return m_isArbitrationDmaNeeded; }

    virtual void notifyCsCompleted(const WcmCsHandleQueue& rCsHandles, bool csFailed) override;

    virtual bool isRecipeHasInflightCsdc(InternalRecipeHandle* pRecipeHandle) override;

    virtual synStatus getDynamicShapesTensorInfoArray(synRecipeHandle             recipeHandle,
                                                      std::vector<tensor_info_t>& tensorInfoArray) const override;

    synStatus waitForRecipeCsdcs(uint64_t recipeId);

private:
    struct csCopyMetaData
    {
        internalMemcopyParams params;
        internalDmaDir        direction;
    };

    using csMetaDataMap = std::map<CommandSubmissionDataChunks*, csCopyMetaData>;

    void _notifyCsCompleted(uint64_t waitForEventHandle, bool csFailed);

    virtual void _dfaLogCsDcInfo(CommandSubmissionDataChunks* csPtr, int logLevel, bool errorCsOnly) override;

    synStatus retrieveCsDc(uint64_t totalCommandSize, DataChunksDB& rDataChunks);

    synStatus releaseCsDc(const DataChunksDB& rDataChunks);

    void clearCsDcBuffers(CommandSubmissionDataChunks& pCsDataChunks) const;

    static generic::CommandBufferPktGenerator* _getPacketGenerator(synDeviceType deviceType);

    static synStatus getLinDmaParams(generic::CommandBufferPktGenerator* pCmdBuffPktGenerator,
                                     const internalMemcopyParams&        rMemcpyParams,
                                     bool                                isArbitrationRequired,
                                     bool                                isUserRequest,
                                     uint64_t                            maxCommandSize,
                                     uint64_t&                           rMaxLinDmaBufferSize,
                                     uint64_t&                           rArbCommandSize,
                                     uint64_t&                           rSizeOfLinDmaCommand,
                                     uint64_t&                           rSizeOfWrappedLinDmaCommand,
                                     uint64_t&                           rSizeOfSingleCommandBuffer,
                                     uint64_t&                           rTotalCommandSize);

    void _clearSharedDBs();

    synStatus memCpyAsync(QueueInterface*              pPreviousStream,
                          const internalMemcopyParams& rMemcpyParams,
                          const internalDmaDir         direction,
                          DataChunksDB&                rDataChunks,
                          CommandSubmissionDataChunks* pCsDataChunks,
                          bool                         isUserRequest,
                          const uint64_t               overrideMemsetVal,
                          bool                         isInspectCopiedContent,
                          bool                         isMemset,
                          bool                         isArbitrationRequired,
                          uint64_t                     maxLinDmaBufferSize,
                          uint64_t                     arbCommandSize,
                          uint64_t                     sizeOfLinDmaCommand,
                          uint64_t                     sizeOfWrappedLinDmaCommand,
                          uint64_t                     sizeOfSingleCommandBuffer);

    synStatus isValidOperation(internalDmaDir direction, const internalMemcopyParams& rMemcpyParams);

    static synStatus _getTotalCommandSize(uint64_t&                    totalWrappedPacketsNum,
                                          uint64_t&                    rMaxLinDmaBufferSize,
                                          uint64_t&                    rTotalCommandSize,
                                          const internalMemcopyParams& rMemcpyParams,
                                          uint64_t                     maxCommandSize,
                                          uint64_t                     sizeOfWrappedLinDmaCommand,
                                          const bool                   isLimitLinDmaBufferSize);

    std::unique_ptr<MemoryManager>       m_pMemoryManager;
    std::unique_ptr<PoolMemoryMapper>    m_poolMemoryManager;
    std::unique_ptr<DataChunksAllocator> m_pAllocator;
    bool                                 m_isArbitrationDmaNeeded;
    size_t                               m_csDcMappingDbSize;
    uint64_t                             m_maxCommandSize;
    csMetaDataMap                        m_csDescriptionDB;

    bool                    m_cvFlag = false;
    std::mutex              m_condVarMutex;
    std::condition_variable m_cv;
};
