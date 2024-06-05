#pragma once

#include "scal_stream_compute_interface.hpp"
#include "scal_stream_copy_scheduler_mode.hpp"

struct ScalStreamCtorInfo : public ScalStreamCtorInfoBase
{
    FenceIdType globalFenceId = -1;
};

class ScalStreamCompute
: public ScalStreamCopySchedulerMode
, public ScalStreamComputeInterface
{
public:
    ScalStreamCompute(const ScalStreamCtorInfo* pScalStreamCtorInfo);

    virtual ~ScalStreamCompute();

    virtual synStatus addGlobalFenceWait(uint32_t target, bool send) override
    {
        return addFenceWait(target, m_globalFenceId, send, true /* isGlobal */);
    };

    virtual synStatus addGlobalFenceInc(bool send) override { return addFenceInc(m_globalFenceId, send); };

    virtual synStatus addAllocBarrierForLaunch(bool allocSoSet, bool relSoSet, McidInfo mcidInfo, bool send) override;

    virtual synStatus addDispatchBarrier(ResourceStreamType  resourceType,
                                         bool                isUserReq,
                                         bool                send,
                                         ScalLongSyncObject& rLongSo,
                                         uint16_t            additionalTdrIncrement = 0) override;

    virtual synStatus addDispatchBarrier(const EngineGrpArr& engineGrpArr,
                                         bool                isUserReq,
                                         bool                send,
                                         ScalLongSyncObject& rLongSo,
                                         uint16_t            additionalTdrIncrement = 0) override;

    virtual synStatus addUpdateRecipeBaseAddresses(const EngineGrpArr& engineGrpArr,
                                                   uint32_t            numOfRecipeBaseElements,
                                                   const uint64_t*     recipeBaseAddresses,
                                                   const uint16_t*     recipeBaseIndices,
                                                   bool                send) override;

    virtual synStatus addDispatchComputeEcbList(uint8_t  logicalEngineId,
                                                uint32_t addrStatic,
                                                uint32_t sizeStatic,
                                                uint32_t singlePhysicalEngineStaticOffset,
                                                uint32_t addrDynamic,
                                                uint32_t sizeDynamic,
                                                bool     shouldUseGcNopKernel,
                                                bool     send) override;

    virtual synStatus addBarrierOrEmptyPdma(ScalLongSyncObject& rLongSo) override;

protected:
    uint32_t m_staticComputeEcbListBuffSize;
    uint32_t m_dynamicComputeEcbListBuffSize;

private:
    synStatus addAllocBarrierV2(uint32_t            targetVal,
                                bool                allocSoSet,
                                bool                relSoSet,
                                const EngineGrpArr& engine_group,
                                McidInfo            mcidInfo,
                                bool                send);

    synStatus addBarrier(ScalLongSyncObject& rLongSo);

    FenceIdType m_globalFenceId;
};
