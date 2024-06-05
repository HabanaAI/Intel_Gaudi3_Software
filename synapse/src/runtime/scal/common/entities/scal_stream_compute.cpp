#include "scal_stream_compute.hpp"
#include "syn_logging.h"
#include "synapse_common_types.h"
#include "device_info_interface.hpp"
#include "scal_memory_pool.hpp"
#include "log_manager.h"
#include "runtime/scal/common/infra/scal_types.hpp"
#include "runtime/scal/common/infra/scal_utils.hpp"

#define LOG_SCALSTRM_TRACE(msg, ...)                                                                                   \
    LOG_TRACE(SYN_STREAM, "stream {}:" msg, m_name, ##__VA_ARGS__)
#define LOG_SCALSTRM_DEBUG(msg, ...)                                                                                   \
    LOG_DEBUG(SYN_STREAM, "stream {}:" msg, m_name, ##__VA_ARGS__)
#define LOG_SCALSTRM_INFO(msg, ...)                                                                                    \
    LOG_INFO(SYN_STREAM, "stream {}:" msg, m_name, ##__VA_ARGS__)
#define LOG_SCALSTRM_ERR(msg, ...)                                                                                     \
    LOG_ERR(SYN_STREAM, "stream {}:" msg, m_name, ##__VA_ARGS__)
#define LOG_SCALSTRM_CRITICAL(msg, ...)                                                                                \
    LOG_CRITICAL(SYN_STREAM, "stream {}:" msg, m_name, ##__VA_ARGS__)

ScalStreamCompute::ScalStreamCompute(const ScalStreamCtorInfo* pScalStreamCtorInfo)
: ScalStreamCopySchedulerMode(pScalStreamCtorInfo),
  m_globalFenceId(pScalStreamCtorInfo->globalFenceId)
{
    LOG_INFO(SYN_STREAM,
                  "scal stream {} globalFenceId {}",
                  m_name,
                  pScalStreamCtorInfo->globalFenceId);

    const common::DeviceInfoInterface& deviceInfoInterface = *pScalStreamCtorInfo->deviceInfoInterface;
    m_staticComputeEcbListBuffSize           = deviceInfoInterface.getStaticComputeEcbListBufferSize();
    m_dynamicComputeEcbListBuffSize          = deviceInfoInterface.getDynamicComputeEcbListBufferSize();
}

ScalStreamCompute::~ScalStreamCompute()
{
}

/*
 ***************************************************************************************************
 *   @brief addDispatchComputeEcbList() sends a pair of ecb list static/dynamic for a given engine
 *
 *   @param  logicalEngineId                    - engine-group's type  (TPC/MME/EDMA/ROT)
 *   @param  addrStatic/sizeStatic              - addr/size of static ecb list
 *   @param  singlePhysicalEngineStaticOffset   - the offset between each phsyical engine, in the static ecb list
 *   @param  addrDynamic/sizeDynamic            - addr/size of dynamic ecb list
 *   @return status
 *
 ***************************************************************************************************
 */
synStatus ScalStreamCompute::addDispatchComputeEcbList(uint8_t  logicalEngineId,
                                                       uint32_t addrStatic,
                                                       uint32_t sizeStatic,
                                                       uint32_t singlePhysicalEngineStaticOffset,
                                                       uint32_t addrDynamic,
                                                       uint32_t sizeDynamic,
                                                       bool     shouldUseGcNopKernel,
                                                       bool     send)
{
    const uint8_t engineGrpType = ScalUtils::convertLogicalEngineIdTypeToScalEngineGroupType(logicalEngineId);
    if (engineGrpType == SCAL_CME_GROUP)
    {
        return std::visit([&](auto pkts) {
                    using T = decltype(pkts);
                    return addPacket<DispatchCmeEcbListPkt<T>>(send,
                                                               engineGrpType,
                                                               sizeDynamic,
                                                               addrDynamic);
                       },
                       m_gxPackets);
    }
    else
    {
        return std::visit(
            [&](auto pkts) {
                using T = decltype(pkts);
                return addPacket<DispatchComputeEcbListPkt<T>>(send,
                                                               engineGrpType,
                                                               sizeStatic,
                                                               sizeDynamic,
                                                               m_staticComputeEcbListBuffSize,
                                                               m_dynamicComputeEcbListBuffSize,
                                                               addrStatic - addrDynamic,
                                                               singlePhysicalEngineStaticOffset,
                                                               addrDynamic,
                                                               shouldUseGcNopKernel,
                                                               getStreamIndex());
            },
            m_gxPackets);
    }
}

synStatus ScalStreamCompute::addUpdateRecipeBaseAddresses(const EngineGrpArr& engineGrpArr,
                                                          uint32_t            numOfRecipeBaseElements,
                                                          const uint64_t*     recipeBaseAddresses,
                                                          const uint16_t*     recipeBaseIndices,
                                                          bool                send)
{
    return std::visit(
        [&](auto pkts) {
            using T = decltype(pkts);
            return addPacketCommon<UpdateRecipeBaseV2Pkt<T>>(send,
                                                             UpdateRecipeBaseV2Pkt<T>::getSize(numOfRecipeBaseElements),
                                                             numOfRecipeBaseElements,
                                                             recipeBaseAddresses,
                                                             recipeBaseIndices,
                                                             engineGrpArr.numEngineGroups,
                                                             engineGrpArr.eng);
        },
        m_gxPackets);
}

/*
 ***************************************************************************************************
 *   @brief addAllocBarrierForLaunch(const EngineGrpArr& engineGrpArr,....) - sends an alloc_barrier. The function
 *   gets an array of up to 4 engines, calculates the target and calls addAllocBarrierV2()
 *
 *   @param  relSoSet, send - passed to the barrier packet
 *   , send
 *   @return status
 *
 ***************************************************************************************************
 */
synStatus ScalStreamCompute::addAllocBarrierForLaunch(bool allocSoSet, bool relSoSet, McidInfo mcidInfo, bool send)
{
    const uint32_t      target       = m_devStreamInfo.resourcesInfo[(uint8_t)ResourceStreamType::COMPUTE].targetVal;
    const EngineGrpArr& engineGrpArr = m_devStreamInfo.resourcesInfo[(uint8_t)ResourceStreamType::COMPUTE].engineGrpArr;

    return addAllocBarrierV2(target, allocSoSet, relSoSet, engineGrpArr, mcidInfo, send);
}

/*
 ***************************************************************************************************
 *   @brief addAllocBarrierV2() - sends an alloc_barrier and specify the engines of the recipe.
 *
 *   @param  targetVal - total number of engines in the stream
 *   @param  relSoSet, send - passed to the barrier packet
 *   @param engine_group array of all engine group to which the alloc barrier should be sent.
 *          and the numer of the engine group types to which the packet should be sent.
 *   , send
 *   @return status
 *
 ***************************************************************************************************
 */
synStatus ScalStreamCompute::addAllocBarrierV2(uint32_t targetVal, bool allocSoSet, bool relSoSet, const EngineGrpArr& engine_group, McidInfo mcidInfo, bool send)
{
    const uint32_t comp_group_index = m_pScalCompletionGroup->getIndexInScheduler();

    return std::visit(
        [&](auto pkts) {
            using T = decltype(pkts);
            return addPacket<AllocBarrierV2bPkt<T>>(send,
                                                    comp_group_index,
                                                    targetVal,
                                                    allocSoSet,
                                                    relSoSet,
                                                    engine_group.numEngineGroups,
                                                    engine_group.eng,
                                                    mcidInfo.mcidDegradeCount,
                                                    mcidInfo.mcidDiscardCount);
        },
        m_gxPackets);
}

/*
 ***************************************************************************************************
 *   @brief sendDispatchBarrier(internalStreamType) - sends a barrier (barrier + dispatch barrier)
 *
 *   @param  internalType
 *   @param  isUserReq, send, longSo (output)
 *   @return status
 *
 ***************************************************************************************************
 */
synStatus ScalStreamCompute::addDispatchBarrier(ResourceStreamType  resourceType,
                                                bool                isUserReq,
                                                bool                send,
                                                ScalLongSyncObject& rLongSo,
                                                uint16_t            additionalTdrIncrement)
{
    const EngineGrpArr& engineGrpArr = m_devStreamInfo.resourcesInfo[(uint8_t)resourceType].engineGrpArr;
    return addDispatchBarrier(engineGrpArr, isUserReq, send, rLongSo, additionalTdrIncrement);
}

/*
 ***************************************************************************************************
 *   @brief sendDispatchBarrier(EngineGrpArr) - sends a barrier (barrier + dispatch barrier)
 *
 *   @param  EngineGrpArr
 *   @param  isUserReq, send, longSo (output)
 *   @return status
 *
 ***************************************************************************************************
 */
synStatus ScalStreamCompute::addDispatchBarrier(const EngineGrpArr& engineGrpArr,
                                                bool                isUserReq,
                                                bool                send,
                                                ScalLongSyncObject& rLongSo,
                                                uint16_t            additionalTdrIncrement)
{
    const uint8_t num_engine_group_type = engineGrpArr.numEngineGroups;

    uint16_t watchdogVal = m_pScalCompletionGroup->getCgInfo().tdr_enabled ? (1 + additionalTdrIncrement) : 0;

    synStatus rc = std::visit(
        [&](auto pkts) {
            using T = decltype(pkts);
            return addPacket<DispatchBarrierPkt<T>>(send, num_engine_group_type, engineGrpArr.eng, watchdogVal);
        },
        m_gxPackets);

    if (rc != synSuccess)
    {
        return rc;
    }

    doneChunkOfCommands(isUserReq, rLongSo);

    LOG_TRACE(SYN_PROGRESS, "{:20} : {:>8x} : {:>8x} : {}/{}",
             m_name,
             rLongSo.m_index,
             rLongSo.m_targetValue,
             HLLOG_FUNC,
             __LINE__);

    return synSuccess;
}

/**
 * add a barrier, or zero-size pdma-cmd (depending on stream type).
 * used when the last added command was 'wait'
 * *** NOTICE: before calling, the 'm_userOpLock' mutex should be locked ***
 * @return status
 */
synStatus ScalStreamCompute::addBarrierOrEmptyPdma(ScalLongSyncObject& rLongSo)
{
    synStatus status = addBarrier(rLongSo);
    if (status != synSuccess)
    {
        LOG_SCALSTRM_ERR("failed to add barrier");
    }

    return status;
}

synStatus ScalStreamCompute::addBarrier(ScalLongSyncObject& rLongSo)
{
    const ResourceInformation& computeResourceInformation =
        m_devStreamInfo.resourcesInfo[(uint8_t)ResourceStreamType::COMPUTE];

    const uint32_t      target       = computeResourceInformation.targetVal;
    const EngineGrpArr& engineGrpArr = computeResourceInformation.engineGrpArr;

    // take global fence
    const uint32_t globalFenceTarget = 1;
    addGlobalFenceWait(globalFenceTarget, false);

    synStatus status = addAllocBarrierV2(target, false/*allocSoSet*/, false/*relSoSet*/, engineGrpArr, {0, 0}/*mcidInfo*/, false);
    if (status != synSuccess)
    {
        LOG_SCALSTRM_ERR("failed to add alloc barrier");
        return status;
    }

    status = addDispatchBarrier(ResourceStreamType::COMPUTE, true, true, rLongSo);
    if (status != synSuccess)
    {
        LOG_SCALSTRM_ERR("failed to add dispatch barrier");
        return status;
    }

    // release global fence
    addGlobalFenceInc(true);

    return synSuccess;
}