#pragma once

#include "scal_stream_copy_interface.hpp"

class ScalStreamComputeInterface : virtual public ScalStreamCopyInterface
{
public:
    virtual synStatus addGlobalFenceWait(uint32_t target, bool send) = 0;

    virtual synStatus addGlobalFenceInc(bool send) = 0;

    virtual synStatus addAllocBarrierForLaunch(bool allocSoSet, bool relSoSet, McidInfo mcidInfo, bool send) = 0;

    virtual synStatus addDispatchBarrier(ResourceStreamType  resourceType,
                                         bool                isUserReq,
                                         bool                send,
                                         ScalLongSyncObject& longSo,
                                         uint16_t            additionalTdrIncrement = 0) = 0;

    virtual synStatus addDispatchBarrier(const EngineGrpArr& engineGrpArr,
                                         bool                isUserReq,
                                         bool                send,
                                         ScalLongSyncObject& longSo,
                                         uint16_t            additionalTdrIncrement = 0) = 0;

    virtual synStatus addUpdateRecipeBaseAddresses(const EngineGrpArr& engineGrpArr,
                                                   uint32_t            numOfRecipeBaseElements,
                                                   const uint64_t*     recipeBaseAddresses,
                                                   const uint16_t*     recipeBaseIndices,
                                                   bool                send) = 0;

    virtual synStatus addDispatchComputeEcbList(uint8_t  logicalEngineId,
                                                uint32_t addrStatic,
                                                uint32_t sizeStatic,
                                                uint32_t singlePhysicalEngineStaticOffset,
                                                uint32_t addrDynamic,
                                                uint32_t sizeDynamic,
                                                bool     shouldUseGcNopKernel,
                                                bool     send) = 0;
};
