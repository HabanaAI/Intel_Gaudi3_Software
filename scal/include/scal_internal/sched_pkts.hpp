#pragma once

#include <memory>
#include "hl_logger/hllog_core.hpp"

#include "struct_fw_packets.hpp"

typedef uint8_t         EngineGroupType;
constexpr static uint32_t c_engineGroupArraySize = 4;
typedef EngineGroupType EngineGroupArrayType[c_engineGroupArraySize];

typedef uint8_t         FenceIdType;
constexpr static uint8_t c_fenceIdArraySize = 8;
typedef FenceIdType     FenceIdArrayType[c_fenceIdArraySize];

inline bool hasPayload(uint32_t payloadAddr)
{
    return (payloadAddr != 0);
}

/**************************************** DispatchComputeEcbList ****************************************************/
template<class Tfw>
struct DispatchComputeEcbListPkt
{
    using pktType = typename Tfw::sched_arc_cmd_dispatch_compute_ecb_list_t;

    static void build(void*    pktBuffer,
                      uint32_t engineGroupType,
                      uint32_t staticSize,
                      uint32_t dynamicSize,
                      uint32_t staticComputeEcbListBuffSize,
                      uint32_t dynamicComputeEcbListBuffSize,
                      uint32_t staticEcbListOffset,
                      uint32_t singlePhysicalEngineStaticOffset,
                      uint32_t dynamicEcbListAddr,
                      bool     shouldUseGcNopKernel,
                      uint32_t streamIdx);

    static void buildScal(void *pktBuffer,
                          uint32_t engineGoupType,
                          bool singleStaticChunk,
                          bool singleDynamicChunk,
                          uint32_t staticEcbListOffset,
                          uint32_t dynamicEcbListAddr);

    static constexpr uint64_t getSize() { return sizeof(pktType); };

    static uint64_t dump(const void* pktBuffer, hl_logger::LoggerSPtr logger, int level);
};

/**************************************** DispatchCmeEcbListPkt ****************************************************/
template<class Tfw>
struct DispatchCmeEcbListPkt
{
    using pktType = typename Tfw::sched_arc_cmd_dispatch_cme_ecb_list_t;

    static void build(void*    pktBuffer,
                      uint32_t engineGroupType,
                      uint32_t ecbListSize,
                      uint32_t ecbListAddr);

    static constexpr uint64_t getSize() { return sizeof(pktType); };

    static uint64_t dump(const void* pktBuffer, hl_logger::LoggerSPtr logger, int level);
};

/********************************* BatchedPdmaTransfer *************************************************************/
template<class Tfw>
struct BatchedPdmaTransferPkt
{
    using pktType       = typename Tfw::sched_arc_cmd_pdma_batch_transfer_t;
    using pdmaParamType = typename Tfw::sched_arc_pdma_commands_params_t;

    static void build(void*             pktBuffer,
                      const void* const paramsBuffer,
                      uint32_t          paramsCount,
                      uint8_t           engineGroupType,
                      uint32_t          workloadType,
                      uint8_t           ctxId,
                      uint8_t           apiId,
                      uint32_t          payload,
                      uint32_t          payloadAddr,
                      bool              watchDog,
                      bool              bMemset,
                      uint32_t          completionGroupIndex);

    static void buildScal(void*             pktBuffer,
                          uint64_t          src,
                          uint64_t          dst,
                          uint32_t          size,
                          uint8_t           engineGroupType,
                          int32_t           workloadType,
                          uint8_t           ctxId,
                          uint32_t          payload,
                          uint32_t          payloadAddr,
                          bool              bMemset,
                          uint32_t          signalToCg,
                          uint32_t          completionGroupIndex);

    static constexpr uint64_t getSize(unsigned paramsCount)
    {
        return sizeof(pktType) + paramsCount * sizeof(pdmaParamType);
    };

    static uint64_t dump(const void* pktBuffer, hl_logger::LoggerSPtr logger, int level);
};

/******************************** MonitorExpPkt ********************************************************************/
template<class Tfw>
struct MonitorExpirationPkt
{
    static uint64_t dump(const void* pktBuffer, hl_logger::LoggerSPtr logger, int level);
};

/************************************ FenceWait *********************************************************************/
template<class Tfw>
struct FenceWaitPkt
{
    using pktType = typename Tfw::sched_arc_cmd_fence_wait_t;

    static void               build(void* pktBuffer, FenceIdType fenceId, uint32_t target);
    static constexpr uint64_t getSize() { return sizeof(pktType); };
    static uint64_t           dump(const void* pktBuffer, hl_logger::LoggerSPtr logger, int level);
};

/********************************* MonitorExpirationFenceUpdate *****************************************************/
template<class Tfw>
struct MonitorExpirationFenceUpdatePkt
{
    using pktType = typename Tfw::sched_mon_exp_msg_t;

    static void               build(void* pktBuffer, FenceIdType fenceId);
    static constexpr uint64_t getSize() { return sizeof(pktType); };
    static uint64_t           dump(const void* pktBuffer, hl_logger::LoggerSPtr logger, int level);
};

/************************************ AllocBarrierV2b ****************************************************************/
template<class Tfw>
struct AllocBarrierV2bPkt
{
    using pktType = typename Tfw::sched_arc_cmd_alloc_barrier_v2_t;

    static void build(void*                      pktBuffer,
                      uint32_t                   compGroupIndex,
                      uint32_t                   targetValue,
                      bool                       allocSoSet,
                      bool                       relSoSet,
                      uint32_t                   numEngineGroupType,
                      EngineGroupArrayType const engineGroupType,
                      uint16_t                   degradeMcidCount,
                      uint16_t                   discardMcidCount);

    static constexpr uint64_t getSize() { return sizeof(pktType); };
    static uint64_t           dump(const void* pktBuffer, hl_logger::LoggerSPtr logger, int level);
};

/************************************ DispatchBarrier ****************************************************************/
template<class Tfw>
struct DispatchBarrierPkt
{
    using pktType = typename Tfw::sched_arc_cmd_dispatch_barrier_t;

    static void build(void*                      pktBuffer,
                      uint32_t                   numEngineGroupType,
                      const EngineGroupArrayType engineGroupType,
                      uint16_t                   watchdogVal);

    static constexpr uint64_t getSize() { return sizeof(pktType); };

    static uint64_t dump(const void* pktBuffer, hl_logger::LoggerSPtr logger, int level);
};

/************************************ FenceIncImmediate *******************************************************/
template<class Tfw>
struct FenceIncImmediatePkt
{
    using pktType = typename Tfw::sched_arc_cmd_fence_inc_immediate_t;

    static void               build(void* pktBuffer, uint8_t fenceId);
    static constexpr uint64_t getSize() { return sizeof(pktType); };
    static uint64_t           dump(const void* pktBuffer, hl_logger::LoggerSPtr logger, int level);
};

/************************************ UpdateRecipeBaseV2 ****************************************************/
template<class Tfw>
struct UpdateRecipeBaseV2Pkt
{
    using pktType = typename Tfw::sched_arc_cmd_update_recipe_base_v2_t;

    static void build(void*                      pktBuffer,
                      uint32_t                   numOfRecipeBaseElements,
                      const uint64_t*            recipeBaseAddresses,
                      const uint16_t*            recipeBaseIndices,
                      uint32_t                   numEngineGroupType,
                      const EngineGroupArrayType engineGroupType);

    static constexpr uint64_t getSize(uint32_t numOfRecipeBaseElements)
    {
        return sizeof(pktType) + (numOfRecipeBaseElements * sizeof(pktType::recipe_base_addr[0]));
    };

    static uint64_t dump(const void* pktBuffer, hl_logger::LoggerSPtr logger, int level);
};

/************************************ NopCmd ****************************************************************/
template<class Tfw>
struct NopCmdPkt
{
    using pktType = typename Tfw::sched_arc_cmd_nop_t;

    static void               build(void* pktBuffer, uint32_t padding);
    static void               buildForceOpcode(void* pktBuffer, uint32_t opcode, uint32_t padding);
    static constexpr uint64_t getSize() { return sizeof(pktType); };
    static uint64_t           dump(const void* pktBuffer, hl_logger::LoggerSPtr logger, int level);
};

/************************************ LbwWrite ****************************************************************/
template<class Tfw>
struct LbwWritePkt
{
    using pktType = typename Tfw::sched_arc_cmd_lbw_write_t;

    static void               build(void* pktBuffer, uint32_t dstAddr, uint32_t srcData, bool blockStream);
    static void               buildForceOpcode(void* pktBuffer, uint32_t opcode, uint32_t dstAddr, uint32_t srcData, bool blockStream);
    static constexpr uint64_t getSize() { return sizeof(pktType); };
    static uint64_t           dump(const void* pktBuffer, hl_logger::LoggerSPtr logger, int level);
};

/************************************ MemFence ****************************************************************/
template<class Tfw>
struct MemFencePkt
{
    using pktType = typename Tfw::sched_arc_cmd_mem_fence_t;

    static void               build(void* pktBuffer, bool isArc, bool isDupEngine, bool isArcDma);
    static constexpr uint64_t getSize() { return sizeof(pktType); };
    static uint64_t           dump(const void* pktBuffer, hl_logger::LoggerSPtr logger, int level);
};

/************************************ BatchedPdmaParams ****************************************************************/
template<class Tfw>
struct BatchedPdmaParamsPkt
{
    using pktType = typename Tfw::sched_arc_pdma_commands_params_t;

    static constexpr uint64_t getSize(unsigned paramsCount)
    {
        return paramsCount * sizeof(pktType);
    };
};

/************************************ LbwBurstWrite ****************************************************************/
template<class Tfw>
struct LbwBurstWritePkt
{
    using pktType = typename Tfw::sched_arc_cmd_lbw_burst_write_t;

    static void               build(void* pktBuffer, uint32_t dst_addr, const uint32_t* data, bool block_stream);
    static constexpr uint64_t getSize() { return sizeof(pktType); };
    static uint64_t           dump(const void* pktBuffer, hl_logger::LoggerSPtr logger, int level);
};
/************************************ LbwRead ****************************************************************/
template<class Tfw>
struct LbwReadPkt
{
    using pktType = typename Tfw::sched_arc_cmd_lbw_read_t;

    static void               build(void* pktBuffer, uint32_t dst_addr, uint32_t src_addr, uint32_t size);
    static constexpr uint64_t getSize() { return sizeof(pktType); };
    static uint64_t           dump(const void* pktBuffer, hl_logger::LoggerSPtr logger, int level);
};

/************************************ LbwRead ****************************************************************/
template<class Tfw>
struct AllocNicBarrier
{
    using pktType = typename Tfw::sched_arc_cmd_alloc_nic_barrier_t;

    static void build(void* pktBuffer, uint32_t opcode, uint32_t comp_group_index, uint32_t required_sobs);
    static constexpr uint64_t getSize() { return sizeof(pktType); };
};

/************************************ AcpFenceWait *******************************************************************/
template<class Tfw>
struct AcpFenceWaitPkt
{
    using pktType = typename Tfw::sched_arc_cmd_acp_fence_wait_t;

    static void               build(void* pktBuffer, FenceIdType fenceId, uint32_t target);
    static constexpr uint64_t getSize() { return sizeof(pktType); };
    static uint64_t           dump(const void* pktBuffer, hl_logger::LoggerSPtr logger, int level);
};

/************************************ AcpFenceIncImmediate ***********************************************************/
template<class Tfw>
struct AcpFenceIncImmediatePkt
{
    using pktType = typename Tfw::sched_arc_cmd_acp_fence_inc_immediate_t;

    static void               build(void* pktBuffer, FenceIdType fenceId, uint32_t value);
    static constexpr uint64_t getSize() { return sizeof(pktType); };
    static uint64_t           dump(const void* pktBuffer, hl_logger::LoggerSPtr logger, int level);
};

template<class Gx>
uint64_t dumpPacket(const uint8_t* pktBuffer, hl_logger::LoggerSPtr logger, int level);
