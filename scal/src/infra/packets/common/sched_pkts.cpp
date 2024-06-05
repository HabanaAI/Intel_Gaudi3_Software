#include <cstdint>
#include <infra/defs.h>

#include "scal_internal/pkt_macros.hpp"
#include "scal.h"

/***************************** DispatchComputeEcbList **************************************************************/
template<class Tfw>
    void DispatchComputeEcbListPkt<Tfw>::build(void*    pktBuffer,
                                        uint32_t engineGroupType,
                                        uint32_t staticSize,
                                        uint32_t dynamicSize,
                                        uint32_t staticComputeEcbListBuffSize,
                                        uint32_t dynamicComputeEcbListBuffSize,
                                        uint32_t staticEcbListOffset,
                                        uint32_t singlePhysicalEngineStaticOffset,
                                        uint32_t dynamicEcbListAddr,
                                        bool     shouldUseGcNopKernel,
                                        uint32_t streamIdx)
{
    // Fields values
    HB_ASSERT((staticEcbListOffset % dynamicComputeEcbListBuffSize) == 0,
              "Invalid static ECB-List offset {:#x}",
              staticEcbListOffset);
    const uint32_t staticEcbListOffsetInEcbListChunks = staticEcbListOffset / dynamicComputeEcbListBuffSize;
    //
    HB_ASSERT((dynamicEcbListAddr % dynamicComputeEcbListBuffSize) == 0,
              "Invalid dynamic ECB-List address {:#x}",
              dynamicEcbListAddr);
    const uint32_t dynamicEcbListAddressInEcbListChunks = dynamicEcbListAddr / dynamicComputeEcbListBuffSize;
    //
    HB_ASSERT((singlePhysicalEngineStaticOffset % dynamicComputeEcbListBuffSize) == 0,
              "Invalid static (single-engine) ECB-List offset {:#x}",
              singlePhysicalEngineStaticOffset);
    const uint32_t singlePhysicalEngineStaticOffsetInEcbListChunks =
        singlePhysicalEngineStaticOffset / staticComputeEcbListBuffSize;

    // Field limitation validation
    constexpr uint32_t staticEcbListOffsetMaxValue       = std::numeric_limits<uint16_t>::max();
    constexpr uint32_t singleEngineEcbListOffsetMaxValue = std::numeric_limits<uint16_t>::max();
    constexpr uint32_t dynamicEcbListAddressMaxValue     = 0xffffff;  // 24-bits
    //
    HB_ASSERT(staticEcbListOffsetInEcbListChunks <= staticEcbListOffsetMaxValue,
              "Static ECB-List offset {:#x} does not fit field's size",
              staticEcbListOffsetInEcbListChunks);
    HB_ASSERT(singlePhysicalEngineStaticOffsetInEcbListChunks <= singleEngineEcbListOffsetMaxValue,
              "Static (single-engine) ECB-List offset {:#x} does not fit field's size",
              singlePhysicalEngineStaticOffsetInEcbListChunks);
    HB_ASSERT(dynamicEcbListAddressInEcbListChunks <= dynamicEcbListAddressMaxValue,
              "Dynamic ECB-List address {:#x} does not fit field's size",
              dynamicEcbListAddressInEcbListChunks);

    const bool singleStaticChunk = (singlePhysicalEngineStaticOffsetInEcbListChunks != 0)
                                       ? (singlePhysicalEngineStaticOffset <= staticComputeEcbListBuffSize)
                                       : (staticSize <= staticComputeEcbListBuffSize);

    const bool singleDynamicChunk = (dynamicSize <= dynamicComputeEcbListBuffSize);

    HB_ASSERT_PTR(pktBuffer);

    pktType& pkt = *(reinterpret_cast<pktType*>(pktBuffer));
    memset(&pkt, 0, sizeof(pktType));

    pkt.opcode = Tfw::SCHED_COMPUTE_ARC_CMD_DISPATCH_COMPUTE_ECB_LIST_V3;

    // see qman_engine_group_type_t
    pkt.engine_group_type = engineGroupType;
    // physical stream index
    pkt.cmpt_stream_index = streamIdx;
    //  This bit indicates static ecb list contains only one chunk
    pkt.single_static_chunk = singleStaticChunk;
    // This bit indicates dynamic ecb list contains only one chunk
    pkt.single_dynamic_chunk = singleDynamicChunk;
    // Address offset of Static ECB List in memory with respect to dynamic ecb list
    pkt.static_ecb_list_offset = staticEcbListOffsetInEcbListChunks;
    // Static ECB List Engine Offset
    pkt.static_ecb_list_eng_offset = singlePhysicalEngineStaticOffsetInEcbListChunks;
    // Address of ECB List in memory  (in terms of core address, 32bit)
    pkt.dynamic_ecb_list_addr_256 = dynamicEcbListAddressInEcbListChunks;
    // Indicates that the Firmware should use the address set for the GC's NOP-Kernel,
    // located at the HBM (part of the PRG-Data), instead of the SCAL's SRAM supplied address
    pkt.use_gc_nop = shouldUseGcNopKernel;
};

template<class Tfw>
void DispatchComputeEcbListPkt<Tfw>::buildScal(void    *pktBuffer,
                                               uint32_t engineGoupType,
                                               bool     singleStaticChunk,
                                               bool     singleDynamicChunk,
                                               uint32_t staticEcbListOffset,
                                               uint32_t dynamicEcbListAddr)
{
        pktType& pkt = *(reinterpret_cast<pktType*>(pktBuffer));
        memset(&pkt, 0, sizeof(pktType));

        pkt.opcode = Tfw::SCHED_COMPUTE_ARC_CMD_DISPATCH_COMPUTE_ECB_LIST_V3;
        // see qman_engine_group_type_t
        pkt.engine_group_type = engineGoupType;
        //  This bit indicates static ecb list contains only one chunk
        pkt.single_static_chunk = singleStaticChunk;
        // This bit indicates dynamic ecb list contains only one chunk
        pkt.single_dynamic_chunk = singleDynamicChunk;
        // Address offset of Static ECB List in memory with respect to dynamic ecb list
        // This packet uses blocks of 256 => Dividing by 256
        pkt.static_ecb_list_offset = staticEcbListOffset >> 8;
        // Address of ECB List in memory  (in terms of core address, 32bit)
        pkt.dynamic_ecb_list_addr_256 = dynamicEcbListAddr >> 8;
}

template<class Tfw>
    uint64_t DispatchComputeEcbListPkt<Tfw>::dump(const void* pktBuffer, hl_logger::LoggerSPtr logger, int level)
{
    const pktType& pkt = *(reinterpret_cast<const pktType*>(pktBuffer));
    HLLOG_UNTYPED(logger,
                  level,
                  "opcode {:#x}, use_gc_nop {} engine_group_type {:#x}, single_static_chunk {:#x}, "
                  "single_dynamic_chunk {:#x}, Folowing in ecb-chunks units: static_ecb_list_offset {:#x}, "
                  "static_ecb_list_eng_offset {:#x}, dynamic_ecb_list_addr {:#x}",
                  pkt.opcode,
                  pkt.use_gc_nop,
                  pkt.engine_group_type,
                  pkt.single_static_chunk,
                  pkt.single_dynamic_chunk,
                  pkt.static_ecb_list_offset,
                  pkt.static_ecb_list_eng_offset,
                  pkt.dynamic_ecb_list_addr_256);

    return getSize();
}

template struct DispatchComputeEcbListPkt<G2Packets>;
template struct DispatchComputeEcbListPkt<G3Packets>;

/***************************** DispatchCmeEcbListPkt **************************************************************/
template<class Tfw>
void DispatchCmeEcbListPkt<Tfw>::build(void*    pktBuffer,
                                       uint32_t engineGroupType,
                                       uint32_t ecbListSize,
                                       uint32_t ecbListAddr)
{
    HB_ASSERT_PTR(pktBuffer);
    if constexpr(!std::is_same_v<Tfw, G2Packets>)
    {
        pktType& pkt = *(reinterpret_cast<pktType*>(pktBuffer));
        memset(&pkt, 0, sizeof(pktType));

        pkt.opcode = Tfw::SCHED_COMPUTE_ARC_CMD_DISPATCH_CME_ECB_LIST;

        HB_ASSERT((ecbListAddr % Tfw::DYNAMIC_COMPUTE_ECB_LIST_BUFF_SIZE) == 0,
                  "Invalid static ECB-List addr {:#x} (not aligned)",
                  ecbListAddr);

        // see qman_engine_group_type_t
        pkt.engine_group_type = engineGroupType;
        pkt.ecb_list_size     = ecbListSize;
        pkt.ecb_list_addr     = ecbListAddr / Tfw::DYNAMIC_COMPUTE_ECB_LIST_BUFF_SIZE;
    }
};

template<class Tfw>
    uint64_t DispatchCmeEcbListPkt<Tfw>::dump(const void* pktBuffer, hl_logger::LoggerSPtr logger, int level)
{
    if constexpr(!std::is_same_v<Tfw, G2Packets>)
    {
        const pktType& pkt = *(reinterpret_cast<const pktType*>(pktBuffer));

        HLLOG_UNTYPED(logger,
                      level,
                      "opcode {:#x}, engine_group_type {:#x}, ecb_list_size {:#x}, ecb_list_addr {:#x}",
                      pkt.opcode,
                      pkt.engine_group_type,
                      pkt.ecb_list_size,
                      pkt.ecb_list_addr);
    }
    return getSize();
}

template struct DispatchCmeEcbListPkt<G2Packets>;
template struct DispatchCmeEcbListPkt<G3Packets>;

/************************************** BatchedPdmaTransfer ********************************************************/
static const uint32_t MAX_COMP_SYNC_GROUP_COUNT = std::numeric_limits<uint32_t>::max();

template<class Tfw>
void BatchedPdmaTransferPkt<Tfw>::build(void*             pktBuffer,
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
                                        uint32_t          completionGroupIndex)
{
    uint32_t compGrpIdx =
        (completionGroupIndex == MAX_COMP_SYNC_GROUP_COUNT) ? Tfw::COMP_SYNC_GROUP_COUNT : completionGroupIndex;

    HB_ASSERT_PTR(pktBuffer);
    HB_ASSERT_PTR(paramsBuffer);

    pktType& pkt = *(reinterpret_cast<pktType*>(pktBuffer));
    memset(&pkt, 0, sizeof(pktType));

    const pdmaParamType* params = reinterpret_cast<const pdmaParamType*>(paramsBuffer);
    memcpy(&pkt.batch_params[0], params, paramsCount * sizeof(pdmaParamType));

    pkt.opcode            = Tfw::SCHED_COMPUTE_ARC_CMD_PDMA_BATCH_TRANSFER;
    pkt.engine_group_type = engineGroupType;
    pkt.workload_type     = workloadType == 0 ? Tfw::ENG_PDMA_ARC_CMD_BATCH_USER_DATA : Tfw::ENG_PDMA_ARC_CMD_BATCH_COMMANDS;
    pkt.stream_ctxt_id    = ctxId;
    pkt.api_id            = apiId;
    pkt.memset            = bMemset;
    pkt.batch_count       = paramsCount;
    if (hasPayload(payloadAddr))
    {
        HB_ASSERT(compGrpIdx == Tfw::COMP_SYNC_GROUP_COUNT, "Both Payload-address and CS-Index are set");
        pkt.has_payload = 1;
        pkt.pay_data    = payload;
        pkt.pay_addr    = payloadAddr;
    }
    else if (compGrpIdx != Tfw::COMP_SYNC_GROUP_COUNT)
    {
        pkt.signal_to_cg        = 1;
        pkt.watch_dog_sig_value = watchDog ? 1 : 0;
        pkt.pay_addr            = compGrpIdx;
    }
}

template<class Tfw>
void BatchedPdmaTransferPkt<Tfw>::buildScal(void*             pktBuffer,
                                            uint64_t src, uint64_t dst, uint32_t size,
                                            uint8_t           engineGroupType,
                                            int32_t           workloadType,
                                            uint8_t           ctxId,
                                            uint32_t          payload,
                                            uint32_t          payloadAddr,
                                            bool              bMemset,
                                            uint32_t          signalToCg,
                                            uint32_t          completionGroupIndex)
{
    pktType& pkt = *(reinterpret_cast<pktType*>(pktBuffer));
    memset(&pkt, 0, sizeof(pktType));

    if (workloadType == -1)
    {
        workloadType = (engineGroupType == SCAL_PDMA_TX_CMD_GROUP) ? Tfw::ENG_PDMA_ARC_CMD_BATCH_COMMANDS : Tfw::ENG_PDMA_ARC_CMD_BATCH_USER_DATA;
    }
    pkt.workload_type = workloadType;

    pkt.batch_params[0].src_addr = src;
    pkt.batch_params[0].dst_addr = dst;
    pkt.batch_params[0].transfer_size = size;

    pkt.opcode            = Tfw::SCHED_COMPUTE_ARC_CMD_PDMA_BATCH_TRANSFER;
    pkt.engine_group_type = engineGroupType;

    pkt.batch_count   = 1;
    pkt.signal_to_cg  = signalToCg;
    if (payloadAddr)// payload may be 0, pay_addr not
    {
        pkt.has_payload = 1;
        pkt.pay_data = payload;
        pkt.pay_addr = payloadAddr;
    }
    else if (signalToCg)
    {
        pkt.pay_addr = completionGroupIndex;
    }
}

template<class Tfw>
uint64_t BatchedPdmaTransferPkt<Tfw>::dump(const void* pktBuffer, hl_logger::LoggerSPtr logger, int level)
{
    const pktType& pkt = *(reinterpret_cast<const pktType*>(pktBuffer));

    if (HLLOG_UNLIKELY(hl_logger::getLoggingLevel(logger) <= level))
    {
        std::string params_str;
        const char* delimeter = "";
        for (unsigned i = 0; i < pkt.batch_count; ++i)
        {
            auto const& param = pkt.batch_params[i];
            params_str += fmt::format(FMT_COMPILE("{}{{src {:#x}, dst {:#x}, size {:#x}}}"),
                                      delimeter,
                                      param.src_addr,
                                      param.dst_addr,
                                      param.transfer_size);
            delimeter = ", ";
        }

        HLLOG_UNTYPED(logger,
                      level,
                      "opcode {:#x}, engine_group_type {:#x}, has_payload {:#x}, stream_ctxt_id {:#x}, api_id {:#x}, "
                      "pay_data {:#x}, pay_addr {:#x}, signal_to_cg {:#x}, watchDog {}, memset {:#x},"
                      "batch_count {} params [{}]",
                      pkt.opcode,
                      pkt.engine_group_type,
                      pkt.has_payload,
                      pkt.stream_ctxt_id,
                      pkt.api_id,
                      pkt.pay_data,
                      pkt.pay_addr,
                      pkt.signal_to_cg,
                      pkt.watch_dog_sig_value,
                      pkt.memset,
                      pkt.batch_count,
                      params_str);

        if (!pkt.signal_to_cg && hasPayload(pkt.pay_addr))
        {
            const void* schedMonExpMsgPacketBuffer = reinterpret_cast<const void*>(&pkt.pay_data);

            MonitorExpirationPkt<Tfw>::dump(schedMonExpMsgPacketBuffer, logger, level);
        }
    }
    return getSize(pkt.batch_count);
}

template struct BatchedPdmaTransferPkt<G2Packets>;
template struct BatchedPdmaTransferPkt<G3Packets>;

/************************************** MonitorExpiration ********************************************************/
template<class Tfw>
    uint64_t MonitorExpirationPkt<Tfw>::dump(const void* pktBuffer, hl_logger::LoggerSPtr logger, int level)
{
    const typename Tfw::sched_mon_exp_msg_t& pkt =
        *(reinterpret_cast<const typename Tfw::sched_mon_exp_msg_t*>(pktBuffer));

    uint32_t actualOpcode = pkt.fence.opcode;

    HLLOG_UNTYPED(logger, level, "sending opcode {} from addr {:#x}", actualOpcode, TO64(&pkt));
    switch (actualOpcode)
    {
        case Tfw::MON_EXP_FENCE_UPDATE:
        {
            break;
        }
        default:
        {
            HLLOG_UNTYPED(logger, level, "opcode {} not parsed", actualOpcode);
        }
    }

    return 0;
}

/************************************** MonitorExpiration ********************************************************/
template<class Tfw>
    void MonitorExpirationFenceUpdatePkt<Tfw>::build(void* pktBuffer, FenceIdType fenceId)
{
    HB_ASSERT_PTR(pktBuffer);

    pktType& pkt = *(reinterpret_cast<pktType*>(pktBuffer));
    memset(&pkt, 0, sizeof(pktType));

    pkt.fence.opcode   = Tfw::MON_EXP_FENCE_UPDATE;
    pkt.fence.fence_id = fenceId;
}

template<class Tfw>
    uint64_t MonitorExpirationFenceUpdatePkt<Tfw>::dump(const void* pktBuffer, hl_logger::LoggerSPtr logger, int level)
{
    HB_ASSERT_PTR(pktBuffer);

    const pktType& pkt = *(reinterpret_cast<const pktType*>(pktBuffer));

    HLLOG_UNTYPED(logger, level, "opcode {:#x}, fenceId {:#x}", pkt.fence.opcode, pkt.fence.fence_id);

    return getSize();
}

template struct MonitorExpirationFenceUpdatePkt<G2Packets>;
template struct MonitorExpirationFenceUpdatePkt<G3Packets>;

/************************************** FenceWait ********************************************************/
template<class Tfw>
    void FenceWaitPkt<Tfw>::build(void* pktBuffer, FenceIdType fenceId, uint32_t target)
{
    HB_ASSERT_PTR(pktBuffer);

    pktType& pkt = *(reinterpret_cast<pktType*>(pktBuffer));
    memset(&pkt, 0, sizeof(pktType));

    pkt.opcode   = Tfw::SCHED_COMPUTE_ARC_CMD_FENCE_WAIT;
    pkt.fence_id = fenceId;
    pkt.target   = target;
}

template<class Tfw>
    uint64_t FenceWaitPkt<Tfw>::dump(const void* pktBuffer, hl_logger::LoggerSPtr logger, int level)
{
    HB_ASSERT_PTR(pktBuffer);

    const pktType& pkt = *(reinterpret_cast<const pktType*>(pktBuffer));

    HLLOG_UNTYPED(logger, level, "opcode {:#x}, fenceId {:#x}, target {:#x}", pkt.opcode, pkt.fence_id, pkt.target);

    return getSize();
}

template struct FenceWaitPkt<G2Packets>;
template struct FenceWaitPkt<G3Packets>;

/************************************** AllocBarrierV2b ********************************************************/
template<class Tfw>
void AllocBarrierV2bPkt<Tfw>::build(void*                      pktBuffer,
                                    uint32_t                   compGroupIndex,
                                    uint32_t                   targetValue,
                                    bool                       allocSoSet,
                                    bool                       relSoSet,
                                    uint32_t                   numEngineGroupType,
                                    EngineGroupArrayType const engineGroupType,
                                    uint16_t                   degradeMcidCount,
                                    uint16_t                   discardMcidCount)
{
    HB_ASSERT_PTR(pktBuffer);

    pktType& pkt = *(reinterpret_cast<pktType*>(pktBuffer));
    memset(&pkt, 0, sizeof(pktType));

    pkt.opcode                = Tfw::SCHED_COMPUTE_ARC_CMD_ALLOC_BARRIER_V2;
    pkt.comp_group_index      = compGroupIndex;
    pkt.target_value          = targetValue;
    pkt.allocate_so_set       = allocSoSet;
    pkt.rel_so_set            = relSoSet;
    pkt.num_engine_group_type = numEngineGroupType;

    for (unsigned i = 0; i < std::min(numEngineGroupType, c_engineGroupArraySize); i++)
    {
        pkt.engine_group_type[i] = engineGroupType[i];
    }

    if constexpr(std::is_same_v<Tfw, G3Packets>)
    {
        pkt.degrade_mcid_count = degradeMcidCount;
        pkt.discard_mcid_count = discardMcidCount;
    }
}

template<class Tfw>
uint64_t AllocBarrierV2bPkt<Tfw>::dump(const void* pktBuffer, hl_logger::LoggerSPtr logger, int level)
{
    HB_ASSERT_PTR(pktBuffer);

    const pktType& pkt = *(reinterpret_cast<const pktType*>(pktBuffer));

    if constexpr(std::is_same_v<Tfw, G3Packets>)
    {
        HLLOG_UNTYPED(logger,
                      level,
                      "opcode {:#x}, comp_group_index {:#x}, target_value {:#x}, alloc_so_set {}, rel_so_set {}, "
                      "num_engine_group_type {:#x}, engine_group_type {:#x} {:#x} {:#x} {:#x},"
                      "degrade_mcid_count {:#X}, discard_mcid_count {:#X}",
                      pkt.opcode,
                      pkt.comp_group_index,
                      pkt.target_value,
                      pkt.allocate_so_set,
                      pkt.rel_so_set,
                      pkt.num_engine_group_type,
                      pkt.engine_group_type[0],
                      pkt.engine_group_type[1],
                      pkt.engine_group_type[2],
                      pkt.engine_group_type[3],
                      pkt.degrade_mcid_count,
                      pkt.discard_mcid_count);
    }
    else
    {
        HLLOG_UNTYPED(logger,
                      level,
                      "opcode {:#x}, comp_group_index {:#x}, target_value {:#x}, alloc_so_set {}, rel_so_set {}, "
                      "num_engine_group_type {:#x}, engine_group_type {:#x} {:#x} {:#x} {:#x}",
                      pkt.opcode,
                      pkt.comp_group_index,
                      pkt.target_value,
                      pkt.allocate_so_set,
                      pkt.rel_so_set,
                      pkt.num_engine_group_type,
                      pkt.engine_group_type[0],
                      pkt.engine_group_type[1],
                      pkt.engine_group_type[2],
                      pkt.engine_group_type[3]);
    }
    return getSize();
}

template struct AllocBarrierV2bPkt<G2Packets>;
template struct AllocBarrierV2bPkt<G3Packets>;

/************************************** DispatchBarrier ********************************************************/
template<class Tfw>
    void DispatchBarrierPkt<Tfw>::build(void*                      pktBuffer,
                                 uint32_t                   numEngineGroupType,
                                 const EngineGroupArrayType engineGroupType,
                                 uint16_t                   watchdogVal)
{
    HB_ASSERT_PTR(pktBuffer);

    pktType& pkt = *(reinterpret_cast<pktType*>(pktBuffer));
    memset(&pkt, 0, sizeof(pktType));

    pkt.opcode                = Tfw::SCHED_COMPUTE_ARC_CMD_DISPATCH_BARRIER;
    pkt.num_engine_group_type = numEngineGroupType;

    for (unsigned i = 0; i < std::min(numEngineGroupType, c_engineGroupArraySize); i++)
    {
        pkt.engine_group_type[i] = engineGroupType[i];
    }

    pkt.watch_dog_sig_value = watchdogVal;
}

template<class Tfw>
    uint64_t DispatchBarrierPkt<Tfw>::dump(const void* pktBuffer, hl_logger::LoggerSPtr logger, int level)
{
    HB_ASSERT_PTR(pktBuffer);

    const pktType& pkt = *(reinterpret_cast<const pktType*>(pktBuffer));

    HLLOG_UNTYPED(logger,
                  level,
                  "opcode {:#x}, num_engine_group_type {:#x},"
                  " engine_group_type {:#x} {:#x} {:#x} {:#x} watchdog {:#x}",
                  pkt.opcode,
                  pkt.num_engine_group_type,
                  pkt.engine_group_type[0],
                  pkt.engine_group_type[1],
                  pkt.engine_group_type[2],
                  pkt.engine_group_type[3],
                  pkt.watch_dog_sig_value);

    return getSize();
}

template struct DispatchBarrierPkt<G2Packets>;
template struct DispatchBarrierPkt<G3Packets>;

/************************************** FenceIncImmediate ********************************************************/
template<class Tfw>
    void FenceIncImmediatePkt<Tfw>::build(void* pktBuffer, uint8_t fenceId)
{
    HB_ASSERT_PTR(pktBuffer);

    pktType& pkt = *(reinterpret_cast<pktType*>(pktBuffer));
    memset(&pkt, 0, sizeof(pktType));

    pkt.opcode      = Tfw::SCHED_COMPUTE_ARC_CMD_FENCE_INC_IMMEDIATE;
    pkt.fence_index = fenceId;
}

template<class Tfw>
    uint64_t FenceIncImmediatePkt<Tfw>::dump(const void* pktBuffer, hl_logger::LoggerSPtr logger, int level)
{
    HB_ASSERT_PTR(pktBuffer);

    const pktType& pkt = *(reinterpret_cast<const pktType*>(pktBuffer));

    HLLOG_UNTYPED(logger,
                  level,
                  "opcode {:#x}, fenceIndex {:#x}",
                  pkt.opcode,
                  pkt.fence_index);

    return getSize();
}

template struct FenceIncImmediatePkt<G2Packets>;
template struct FenceIncImmediatePkt<G3Packets>;

/************************************** UpdateRecipeBaseV2 ********************************************************/
#define MAX_NUM_OF_RECIPE_BASE_ADDRESSES         (4)
#define RECIPE_BASE_ADDRESSES_INDICES_MASK       (0xF)
#define RECIPE_BASE_ADDRESSES_INDICES_FIELD_SIZE (4)

template<class Tfw>
void UpdateRecipeBaseV2Pkt<Tfw>::build(void*                      pktBuffer,
                                       uint32_t                   numOfRecipeBaseElements,
                                       const uint64_t*            recipeBaseAddresses,
                                       const uint16_t*            recipeBaseIndices,
                                       uint32_t                   numEngineGroupType,
                                       const EngineGroupArrayType engineGroupType)
{
    HB_ASSERT_PTR(pktBuffer);
    HB_ASSERT(numOfRecipeBaseElements <= MAX_NUM_OF_RECIPE_BASE_ADDRESSES, "Invalid amount of recipe-base elements");

    pktType& pkt = *(reinterpret_cast<pktType*>(pktBuffer));
    memset(&pkt, 0, sizeof(pktType));

    pkt.opcode = Tfw::SCHED_COMPUTE_ARC_CMD_UPDATE_RECIPE_BASE_V2;

    pkt.num_engine_group_type = numEngineGroupType;
    for (unsigned i = 0; i < std::min(numEngineGroupType, c_engineGroupArraySize); i++)
    {
        pkt.engine_group_type[i] = engineGroupType[i];
    }

    for (unsigned i = 0; i < numOfRecipeBaseElements; i++)
    {
#define SET_RECIPE_BASE_INDEX(X)                                                                \
    case X:                                                                                     \
        pkt.recipe_base_index##X = (recipeBaseIndices[X] & RECIPE_BASE_ADDRESSES_INDICES_MASK); \
        break;

        switch (i)
        {
            SET_RECIPE_BASE_INDEX(0);
            SET_RECIPE_BASE_INDEX(1);
            SET_RECIPE_BASE_INDEX(2);
            SET_RECIPE_BASE_INDEX(3);
        }
    }
    pkt.num_recipe_addrs = numOfRecipeBaseElements;

    std::memcpy((uint8_t*)pkt.recipe_base_addr, recipeBaseAddresses, numOfRecipeBaseElements * sizeof(uint64_t));
}

template<class Tfw>
    uint64_t UpdateRecipeBaseV2Pkt<Tfw>::dump(const void* pktBuffer, hl_logger::LoggerSPtr logger, int level)
{
    HB_ASSERT_PTR(pktBuffer);

    const pktType& pkt = *(reinterpret_cast<const pktType*>(pktBuffer));

    std::string recipe_base_info_str;
    const char* delimeter         = "";
    uint16_t    recipeBaseIndices = (pkt.recipe_base_indices);
    for (unsigned i = 0; i < pkt.num_recipe_addrs; ++i)
    {
        uint16_t currentBaseIndex =
            (recipeBaseIndices >> (RECIPE_BASE_ADDRESSES_INDICES_FIELD_SIZE * i)) & RECIPE_BASE_ADDRESSES_INDICES_MASK;

        recipe_base_info_str +=
            fmt::format(FMT_COMPILE("{}{{base-index {} base-address {:#x}}}"), delimeter, currentBaseIndex, pkt.recipe_base_addr[i]);
        delimeter = ", ";
    }

    HLLOG_UNTYPED(logger,
                  level,
                  "opcode {:#x}, num_engine_group_type {:#x}, engine_group_type {:#x} {:#x} {:#x} {:#x} [{}]",
                  pkt.opcode,
                  pkt.num_engine_group_type,
                  pkt.engine_group_type[0],
                  pkt.engine_group_type[1],
                  pkt.engine_group_type[2],
                  pkt.engine_group_type[3],
                  recipe_base_info_str);

    return getSize(pkt.num_recipe_addrs);
}

template struct UpdateRecipeBaseV2Pkt<G2Packets>;
template struct UpdateRecipeBaseV2Pkt<G3Packets>;

/************************************** NopCmd ********************************************************/
template<class Tfw>
void NopCmdPkt<Tfw>::build(void* pktBuffer, uint32_t padding)
{
    return buildForceOpcode(pktBuffer, Tfw::SCHED_COMPUTE_ARC_CMD_NOP, padding);
}

template<class Tfw>
void NopCmdPkt<Tfw>::buildForceOpcode(void* pktBuffer, uint32_t opcode, uint32_t padding)
{
    HB_ASSERT_PTR(pktBuffer);

    pktType& pkt = *(reinterpret_cast<pktType*>(pktBuffer));
    memset(&pkt, 0, sizeof(pktType));

    uint32_t paddingCount = (uint32_t)((padding - sizeof(typename Tfw::sched_arc_cmd_nop_t)) / sizeof(uint32_t));
    HB_ASSERT((padding - sizeof(typename Tfw::sched_arc_cmd_nop_t)) % sizeof(uint32_t) == 0, "invalid padding value");

    pkt.opcode        = opcode;
    pkt.padding_count = paddingCount;
}


template<class Tfw>
    uint64_t NopCmdPkt<Tfw>::dump(const void* pktBuffer, hl_logger::LoggerSPtr logger, int level)
{
    HB_ASSERT_PTR(pktBuffer);

    const pktType& pkt = *(reinterpret_cast<const pktType*>(pktBuffer));

    HLLOG_UNTYPED(logger, level, "opcode {:#x}, padding {:#x}", pkt.opcode, pkt.padding_count);

    return getSize() + pkt.padding_count * sizeof(uint32_t);
}

template struct NopCmdPkt<G2Packets>;
template struct NopCmdPkt<G3Packets>;

/************************************** LbwWrite ********************************************************/
template<class Tfw>
void LbwWritePkt<Tfw>::build(void* pktBuffer, uint32_t dstAddr, uint32_t srcData, bool blockStream)
{
    return buildForceOpcode(pktBuffer, Tfw::SCHED_COMPUTE_ARC_CMD_LBW_WRITE, dstAddr, srcData, blockStream);
}

template<class Tfw>
void LbwWritePkt<Tfw>::buildForceOpcode(void* pktBuffer, uint32_t opcode, uint32_t dstAddr, uint32_t srcData, bool blockStream)
{
        HB_ASSERT_PTR(pktBuffer);

        pktType& pkt = *(reinterpret_cast<pktType*>(pktBuffer));
        memset(&pkt, 0, sizeof(pktType));

        pkt.opcode     = opcode;
        pkt.dst_addr   = dstAddr;
        pkt.src_data   = srcData;
        pkt.block_next = blockStream;
}

template<class Tfw>
    uint64_t LbwWritePkt<Tfw>::dump(const void* pktBuffer, hl_logger::LoggerSPtr logger, int level)
{
    HB_ASSERT_PTR(pktBuffer);

    const pktType& pkt = *(reinterpret_cast<const pktType*>(pktBuffer));

    HLLOG_UNTYPED(logger,
                  level,
                  "opcode {:#x}, dst_addr {:#x} src_data {:#x}, block_next {}",
                  pkt.opcode,
                  pkt.dst_addr,
                  pkt.src_data,
                  pkt.block_next);

    return getSize();
}

template struct LbwWritePkt<G2Packets>;
template struct LbwWritePkt<G3Packets>;

/************************************** LbwWrite ********************************************************/
template<class Tfw>
    void MemFencePkt<Tfw>::build(void* pktBuffer, bool isArc, bool isDupEngine, bool isArcDma)
{
    HB_ASSERT_PTR(pktBuffer);

    pktType& pkt = *(reinterpret_cast<pktType*>(pktBuffer));
    memset(&pkt, 0, sizeof(pktType));

    pkt.opcode  = Tfw::SCHED_COMPUTE_ARC_CMD_MEM_FENCE;
    pkt.arc     = isArc;
    pkt.dup_eng = isDupEngine;
    pkt.arc_dma = isArcDma;
}

template<class Tfw>
    uint64_t MemFencePkt<Tfw>::dump(const void* pktBuffer, hl_logger::LoggerSPtr logger, int level)
{
    HB_ASSERT_PTR(pktBuffer);

    const pktType& pkt = *(reinterpret_cast<const pktType*>(pktBuffer));

    HLLOG_UNTYPED(logger,
                  level,
                  "opcode {:#x}, arc {} dup_eng {}, arc_dma {}",
                  pkt.opcode,
                  pkt.arc,
                  pkt.dup_eng,
                  pkt.arc_dma);

    return getSize();
}

template struct MemFencePkt<G2Packets>;
template struct MemFencePkt<G3Packets>;

/************************************** LbwBurstWrite ********************************************************/
template<class Tfw>
void LbwBurstWritePkt<Tfw>::build(void* pktBuffer, uint32_t dst_addr, const uint32_t* data, bool block_stream)
{
    pktType& pkt = *(reinterpret_cast<pktType*>(pktBuffer));
    memset(&pkt, 0, sizeof(pktType));

    pkt.opcode     = Tfw::SCHED_COMPUTE_ARC_CMD_LBW_BURST_WRITE;
    pkt.dst_addr   = dst_addr;
    pkt.src_data0  = data[0];
    pkt.src_data1  = data[1];
    pkt.src_data2  = data[2];
    pkt.src_data3  = data[3];
    pkt.block_next = block_stream;
}

template<class Tfw>
    uint64_t LbwBurstWritePkt<Tfw>::dump(const void* pktBuffer, hl_logger::LoggerSPtr logger, int level)
{
    const pktType& pkt = *(reinterpret_cast<const pktType*>(pktBuffer));

    HLLOG_UNTYPED(logger,
                  level,
                  "opcode {:#x}, dst_addr {:#x} data {:#x}/{:#x}/{:#x}/{:#x} block_stream {}",
                  pkt.opcode,
                  pkt.dst_addr,
                  pkt.src_data0,
                  pkt.src_data1,
                  pkt.src_data2,
                  pkt.src_data3,
                  pkt.block_next);

    return getSize();
}

template struct LbwBurstWritePkt<G2Packets>;
template struct LbwBurstWritePkt<G3Packets>;

/************************************ LbwRead ****************************************************************/
template<class Tfw>
    void LbwReadPkt<Tfw>::build(void* pktBuffer, uint32_t dst_addr, uint32_t src_addr, uint32_t size)
{
    pktType& pkt = *(reinterpret_cast<pktType*>(pktBuffer));
    memset(&pkt, 0, sizeof(pktType));

    pkt.opcode   = Tfw::SCHED_COMPUTE_ARC_CMD_LBW_READ;
    pkt.dst_addr = dst_addr;
    pkt.src_addr = src_addr;
    pkt.size     = size;
}

template<class Tfw>
    uint64_t LbwReadPkt<Tfw>::dump(const void* pktBuffer, hl_logger::LoggerSPtr logger, int level)
{
    const pktType& pkt = *(reinterpret_cast<const pktType*>(pktBuffer));

    HLLOG_UNTYPED(logger,
                  level,
                  "opcode {:#x}, dst_addr {:#x} src_addr {:#x} size {:#x}",
                  pkt.opcode,
                  pkt.dst_addr,
                  pkt.src_addr,
                  pkt.size);

    return getSize();
}

template struct LbwReadPkt<G2Packets>;
template struct LbwReadPkt<G3Packets>;

/************************************ AllocNicBarrier ****************************************************************/
template<class Tfw>
void AllocNicBarrier<Tfw>::build(void* pktBuffer, uint32_t opcode, uint32_t comp_group_index, uint32_t required_sobs)
{
    pktType& pkt = *(reinterpret_cast<pktType*>(pktBuffer));
    memset(&pkt, 0, sizeof(pktType));

    pkt.opcode           = opcode;
    pkt.comp_group_index = comp_group_index;
    pkt.required_sobs    = required_sobs;
}

template struct AllocNicBarrier<G2Packets>;
template struct AllocNicBarrier<G3Packets>;

/************************************** AcpFenceWait *****************************************************************/
template<class Tfw>
    void AcpFenceWaitPkt<Tfw>::build(void* pktBuffer, FenceIdType fenceId, uint32_t target)
{
    if constexpr(!std::is_same_v<Tfw, G2Packets>)
    {
        HB_ASSERT_PTR(pktBuffer);

        pktType& pkt = *(reinterpret_cast<pktType*>(pktBuffer));
        memset(&pkt, 0, sizeof(pktType));

        pkt.opcode   = Tfw::SCHED_COMPUTE_ARC_CMD_ACP_FENCE_WAIT;
        pkt.fence_id = fenceId;
        pkt.target   = target;
    }
    else
    {
        HB_ASSERT(false, "AcpFenceWaitPkt is not supported for Gaudi2");
    }
}

template<class Tfw>
    uint64_t AcpFenceWaitPkt<Tfw>::dump(const void* pktBuffer, hl_logger::LoggerSPtr logger, int level)
{
    if constexpr(!std::is_same_v<Tfw, G2Packets>)
    {
        HB_ASSERT_PTR(pktBuffer);

        const pktType& pkt = *(reinterpret_cast<const pktType*>(pktBuffer));

        HLLOG_UNTYPED(logger, level, "opcode {:x}, fenceId {:x}, target {:x}", pkt.opcode, pkt.fence_id, pkt.target);

        return getSize();
    }
    else
    {
        HB_ASSERT(false, "AcpFenceWaitPkt is not supported for Gaudi2");
        return 0;
    }
}

template struct AcpFenceWaitPkt<G2Packets>; // Not supported
template struct AcpFenceWaitPkt<G3Packets>;

/************************************** AcpFenceIncImmediate *********************************************************/
template<class Tfw>
    void AcpFenceIncImmediatePkt<Tfw>::build(void* pktBuffer, FenceIdType fenceId, uint32_t value)
{
    if constexpr (!std::is_same_v<Tfw, G2Packets>)
    {
        HB_ASSERT_PTR(pktBuffer);

        pktType& pkt = *(reinterpret_cast<pktType*>(pktBuffer));
        memset(&pkt, 0, sizeof(pktType));

        pkt.opcode   = Tfw::SCHED_COMPUTE_ARC_CMD_ACP_FENCE_INC_IMMEDIATE;
        pkt.fence_id = fenceId;
        pkt.value    = value;
    }
    else
    {
        HB_ASSERT(false, "AcpFenceIncImmediatePkt is not supported for Gaudi2");
    }
}

template<class Tfw>
    uint64_t AcpFenceIncImmediatePkt<Tfw>::dump(const void* pktBuffer, hl_logger::LoggerSPtr logger, int level)
{
    if constexpr (!std::is_same_v<Tfw, G2Packets>)
    {
        HB_ASSERT_PTR(pktBuffer);

        const pktType& pkt = *(reinterpret_cast<const pktType*>(pktBuffer));

        HLLOG_UNTYPED(logger,
                    level,
                    "opcode {:x}, fenceId {:x}, value {:x}",
                    pkt.opcode,
                    pkt.fence_id,
                    pkt.value);

        return getSize();
    }
    else
    {
        HB_ASSERT(false, "AcpFenceIncImmediatePkt is not supported for Gaudi2");
        return 0;
    }
}

template struct AcpFenceIncImmediatePkt<G2Packets>; // Not supported
template struct AcpFenceIncImmediatePkt<G3Packets>;

/*********************************************************************************************************************/

template<class Tfw>
static const char* packetToName(uint32_t opcode)
{
#define PACKET_CASE(X)                                                                                                 \
    case Tfw::X:                                                                                                       \
        return #X;

#define UNUSED_PACKET_CASE(X)                                                                                          \
    case Tfw::X:                                                                                                       \
        return "Unused opcode " #X

    switch (opcode)
    {
        PACKET_CASE(SCHED_COMPUTE_ARC_CMD_FENCE_WAIT);
        PACKET_CASE(SCHED_COMPUTE_ARC_CMD_LBW_WRITE);
        PACKET_CASE(SCHED_COMPUTE_ARC_CMD_LBW_BURST_WRITE);
        PACKET_CASE(SCHED_COMPUTE_ARC_CMD_ALLOC_BARRIER_V2);
        PACKET_CASE(SCHED_COMPUTE_ARC_CMD_DISPATCH_BARRIER);
        PACKET_CASE(SCHED_COMPUTE_ARC_CMD_FENCE_INC_IMMEDIATE);
        PACKET_CASE(SCHED_COMPUTE_ARC_CMD_LBW_READ);
        PACKET_CASE(SCHED_COMPUTE_ARC_CMD_MEM_FENCE);
        PACKET_CASE(SCHED_COMPUTE_ARC_CMD_UPDATE_RECIPE_BASE_V2);
        PACKET_CASE(SCHED_COMPUTE_ARC_CMD_NOP);
        PACKET_CASE(SCHED_COMPUTE_ARC_CMD_PDMA_BATCH_TRANSFER);
        PACKET_CASE(SCHED_COMPUTE_ARC_CMD_ACP_FENCE_WAIT);
        PACKET_CASE(SCHED_COMPUTE_ARC_CMD_ACP_FENCE_INC_IMMEDIATE);
        PACKET_CASE(SCHED_COMPUTE_ARC_CMD_DISPATCH_COMPUTE_ECB_LIST_V3);
        PACKET_CASE(SCHED_COMPUTE_ARC_CMD_COUNT);
        PACKET_CASE(SCHED_COMPUTE_ARC_CMD_SIZE);
    }
    if constexpr(std::is_same_v<Tfw, G3Packets>)
    {
        switch (opcode)
        {
           PACKET_CASE(SCHED_COMPUTE_ARC_CMD_DISPATCH_CME_ECB_LIST);
        }
    }

    return "Unknown opcode";
}

/************************************** LbwWrite ********************************************************/
template<class Tfw>
uint64_t dumpPacket(const uint8_t* pktBuffer, hl_logger::LoggerSPtr logger, int level)
{
    const typename Tfw::sched_arc_cmd_nop_t& pkt =
        reinterpret_cast<const typename Tfw::sched_arc_cmd_nop_t&>(*pktBuffer);

    uint32_t opCode  = pkt.opcode;
    uint64_t cmdSize = 0;

    HLLOG_UNTYPED(logger,
                  level,
                  "sending {} opcode {} from addr {:#x}",
                  packetToName<Tfw>(opCode),
                  opCode,
                  TO64(pktBuffer));

    switch (opCode)
    {
        case Tfw::SCHED_COMPUTE_ARC_CMD_DISPATCH_COMPUTE_ECB_LIST_V3:
        {
            cmdSize = DispatchComputeEcbListPkt<Tfw>::dump(pktBuffer, logger, level);
            break;
        }
        case Tfw::SCHED_COMPUTE_ARC_CMD_PDMA_BATCH_TRANSFER:
        {
            cmdSize = BatchedPdmaTransferPkt<Tfw>::dump(pktBuffer, logger, level);
            break;
        }
        case Tfw::SCHED_COMPUTE_ARC_CMD_FENCE_WAIT:
        {
            cmdSize = FenceWaitPkt<Tfw>::dump(pktBuffer, logger, level);
            break;
        }
        case Tfw::SCHED_COMPUTE_ARC_CMD_ALLOC_BARRIER_V2:
        {
            cmdSize = AllocBarrierV2bPkt<Tfw>::dump(pktBuffer, logger, level);
            break;
        }
        case Tfw::SCHED_COMPUTE_ARC_CMD_DISPATCH_BARRIER:
        {
            cmdSize = DispatchBarrierPkt<Tfw>::dump(pktBuffer, logger, level);
            break;
        }
        case Tfw::SCHED_COMPUTE_ARC_CMD_FENCE_INC_IMMEDIATE:
        {
            cmdSize = FenceIncImmediatePkt<Tfw>::dump(pktBuffer, logger, level);
            break;
        }
        case Tfw::SCHED_COMPUTE_ARC_CMD_UPDATE_RECIPE_BASE_V2:
        {
            cmdSize = UpdateRecipeBaseV2Pkt<Tfw>::dump(pktBuffer, logger, level);
            break;
        }
        case Tfw::SCHED_COMPUTE_ARC_CMD_NOP:
        {
            cmdSize = NopCmdPkt<Tfw>::dump(pktBuffer, logger, level);
            break;
        }
        case Tfw::SCHED_COMPUTE_ARC_CMD_LBW_WRITE:
        {
            cmdSize = LbwWritePkt<Tfw>::dump(pktBuffer, logger, level);
            break;
        }
        case Tfw::SCHED_COMPUTE_ARC_CMD_MEM_FENCE:
        {
            cmdSize = MemFencePkt<Tfw>::dump(pktBuffer, logger, level);
            break;
        }
    }
    // Workaround which fits G3 & G2, due to G3 specific supported packets
    // For G2, we will have a fallthrough to default => unexpected OPCODE
    if constexpr(std::is_same_v<Tfw, G3Packets>)
    {
        switch (opCode)
        {
            case G3Packets::SCHED_COMPUTE_ARC_CMD_ACP_FENCE_WAIT:
            {
                cmdSize = AcpFenceWaitPkt<G3Packets>::dump(pktBuffer, logger, level);
                break;
            }
            case G3Packets::SCHED_COMPUTE_ARC_CMD_ACP_FENCE_INC_IMMEDIATE:
            {
                cmdSize = AcpFenceIncImmediatePkt<G3Packets>::dump(pktBuffer, logger, level);
                break;
            }
            case G3Packets::SCHED_COMPUTE_ARC_CMD_DISPATCH_CME_ECB_LIST:
            {
                cmdSize = DispatchCmeEcbListPkt<G3Packets>::dump(pktBuffer, logger, level);
                break;
            }
        }
    }
    if (cmdSize == 0)
    {
        HLLOG_UNTYPED(logger, level, "{}: opcode {} not parsed", std::string_view(__func__), opCode);
    }
    return cmdSize;
}

template uint64_t dumpPacket<G2Packets>(const uint8_t* pktBuffer, hl_logger::LoggerSPtr logger, int level);
template uint64_t dumpPacket<G3Packets>(const uint8_t* pktBuffer, hl_logger::LoggerSPtr logger, int level);
