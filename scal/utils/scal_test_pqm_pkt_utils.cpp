#include "scal_test_pqm_pkt_utils.h"
#include "gaudi3/gaudi3_pqm_packets.h"
#include "scal.h"
#include "string.h"
#include "gaudi3/asic_reg_structs/sob_objs_regs.h"
#include "common/scal_macros.h"
#include "infra/sync_mgr.hpp"
#include <cassert>

inline uint64_t getSosAddr(unsigned smIndex, unsigned sos)
{
    uint64_t smBase = SyncMgrG3::getSmBase(smIndex);

    return smBase + varoffsetof(gaudi3::block_sob_objs, sob_obj_0[sos]);
}

uint32_t PqmPktUtils::getPayloadDataFenceInc(FenceIdType fenceId)
{
    union sync_object_update
    {
        struct
        {
            uint32_t value    :16;
            uint32_t reserved :13;
            uint32_t mode     :3;
        } so_update;
        uint32_t raw;
    };

    // Increments SOBJ (Fence) by 1
    sync_object_update sobjUpdate;
    sobjUpdate.raw = 0;
    //
    sobjUpdate.so_update.value = 1;
    sobjUpdate.so_update.mode  = 1; // add operation

    return sobjUpdate.raw;
}

uint32_t PqmPktUtils::getCqLongSoValue()
{
    union sync_object_update
    {
        struct
        {
            uint32_t sync_value :16;
            uint32_t reserved1  :8;
            uint32_t long_mode  :1;
            uint32_t reserved2  :5;
            uint32_t te         :1;
            uint32_t mode       :1;
        } so_update;
        uint32_t raw;
    };
    sync_object_update syncObjUpdate;
    syncObjUpdate.raw                  = 0;
    syncObjUpdate.so_update.long_mode  = 1; // Long-SO mode
    syncObjUpdate.so_update.mode       = 1; // Increment
    syncObjUpdate.so_update.sync_value = 1; // By 1

    return syncObjUpdate.raw;
}

#define MAX_VALID_FENCE_ID (3)

void PqmPktUtils::sendPdmaCommand(  bool       isDirectMode,
                                    std::variant<G2Packets , G3Packets> buildPkt,
                                    char*       pktBuffer,
                                    uint64_t    src,
                                    uint64_t    dst,
                                    uint32_t    size,
                                    uint8_t     engineGroupType,
                                    int32_t     workloadType,
                                    uint8_t     ctxId,
                                    uint32_t    payload,
                                    uint64_t    payloadAddr,
                                    bool        bMemset,
                                    uint32_t    signalToCg,
                                    bool        wr_comp,
                                    uint32_t    completionGroupIndex,
                                    uint64_t    longSoSmIdx,
                                    unsigned    longSoIndex)
{
    if (!isDirectMode)
    {
        fillScalPktNoSize<BatchedPdmaTransferPkt>(  buildPkt,
                                                    pktBuffer,
                                                    src,
                                                    dst,
                                                    size,
                                                    engineGroupType,
                                                    workloadType,
                                                    ctxId,
                                                    payload,
                                                    payloadAddr,
                                                    bMemset,
                                                    signalToCg,
                                                    completionGroupIndex);
    }
    else
    {
        uint8_t dir = pdmaDir::INVALID;
        switch (engineGroupType)
        {
            case SCAL_PDMA_TX_CMD_GROUP:
            case SCAL_PDMA_TX_DATA_GROUP:
                dir = pdmaDir::HOST_TO_DEVICE;
                break;
            case SCAL_PDMA_RX_GROUP:
            case SCAL_PDMA_RX_DEBUG_GROUP:
            case SCAL_PDMA_DEV2DEV_DEBUG_GROUP:
                dir = pdmaDir::DEVICE_TO_HOST;
                break;
        }

        buildPqmLpdmaPacket((uint8_t*)pktBuffer, src, dst, size, bMemset, dir,
                            wr_comp, signalToCg,
                            payloadAddr ? payloadAddr : getSosAddr(longSoSmIdx, longSoIndex),
                            payloadAddr ? payload : getCqLongSoValue(),
                            0, 0, MAX_VALID_FENCE_ID + 1);
    }
}

uint64_t PqmPktUtils::getPdmaCmdSize(bool isDirectMode,
                                    std::variant<G2Packets , G3Packets> buildPkt,
                                    bool     wr_comp,
                                    unsigned paramsCount)
{
    if (isDirectMode)
    {
        return getLpdmaCmdSize(wr_comp ? true : false);
    }
    return getPktSize<BatchedPdmaTransferPkt>(buildPkt, paramsCount);
}


union FenceData
{
    struct
    {
        uint32_t fenceDecVal : 4;
        uint32_t reserved    : 12;
        uint32_t fenceTarget : 14;
        uint32_t fenceId     : 2;
    };

    uint32_t raw;
};

struct g3_lin_pdma_fence_wo_wr_comp
{
    uint32_t reserved;
    uint32_t fence_data;
} __attribute__ ((aligned(8), __packed__));

struct g3_lin_pdma_wr_comp_data_lo
{
    union
    {
        uint32_t reserved_0;
        uint32_t wr_comp_addr_lo;
    };
    union
    {
        uint32_t reserved_1;
        uint32_t wr_comp_data;
    };
} __attribute__ ((aligned(8), __packed__));

struct g3_lin_pdma_wr_comp_data_hi
{
    union
    {
        uint32_t reserved_2;
        uint32_t wr_comp_addr_hi;
    };
    union
    {
        uint32_t reserved_3;
        uint32_t fence_data_with_wr_comp;
    };
} __attribute__ ((aligned(8), __packed__));

struct g3_lin_pdma_data
{
    union
    {
        uint32_t src_addr_hi;
        uint32_t memset_value_hi;
    };
    union
    {
        uint32_t dst_addr_hi;
        uint32_t reserved;
    };
} __attribute__ ((aligned(8), __packed__));

struct lin_pdma_single_wr_comp_t
{
    struct pqm_packet_lin_pdma      lin_pdma;
    struct g3_lin_pdma_data         lin_pdma_data;

    union
    {
        g3_lin_pdma_fence_wo_wr_comp fence_wo_wr_comp;
        g3_lin_pdma_wr_comp_data_lo  lin_pdma_wr_comp0_data_lo;
    };

    struct g3_lin_pdma_wr_comp_data_hi lin_pdma_wr_comp0_data_hi;
} __attribute__ ((aligned(8), __packed__));



void PqmPktUtils::buildPqmLpdmaPacket(  uint8_t* pktBuffer,
                                        uint64_t src,
                                        uint64_t dst,
                                        uint32_t size,
                                        bool     bMemset,
                                        uint8_t  direction,
                                        bool     useBarrier,
                                        bool     signalToCg,
                                        uint64_t barrierAddress,
                                        uint32_t barrierData,
                                        uint32_t fenceDecVal,
                                        uint32_t fenceTarget,
                                        uint32_t fenceId)
{
    lin_pdma_single_wr_comp_t& pkt = reinterpret_cast<lin_pdma_single_wr_comp_t&>(*pktBuffer);

    bool isFenceEnabled = (fenceId < MAX_VALID_FENCE_ID);

    memset(&pkt, 0x0, getLpdmaCmdSize(useBarrier));

    if (useBarrier)
    {
        pkt.lin_pdma.wrcomp                        = 1;
        pkt.lin_pdma_wr_comp0_data_lo.wr_comp_addr_lo = barrierAddress & 0xFFFFFFFF;
        pkt.lin_pdma_wr_comp0_data_hi.wr_comp_addr_hi = barrierAddress >> 32;
        pkt.lin_pdma_wr_comp0_data_lo.wr_comp_data    = barrierData;
    }

    if (isFenceEnabled)
    {
        pkt.lin_pdma.fence_en = 1;

        FenceData fenceData;
        fenceData.raw         = 0;
        fenceData.fenceId     = fenceId;
        fenceData.fenceDecVal = fenceDecVal;
        fenceData.fenceTarget = fenceTarget;

        if (useBarrier)
        {
            pkt.lin_pdma_wr_comp0_data_hi.fence_data_with_wr_comp = fenceData.raw;
        }
        else
        {
            pkt.fence_wo_wr_comp.fence_data = fenceData.raw;
        }
    }

    pkt.lin_pdma.opcode            = PQM_PACKET_LIN_PDMA;
    pkt.lin_pdma.direction         = direction;
    pkt.lin_pdma.memset            = bMemset;
    pkt.lin_pdma.en_desc_commit    = 1;
    pkt.lin_pdma.inc_context_id    = 1;
    pkt.lin_pdma.advcnt_hbw_complq = signalToCg ? 1 : 0; // increse the counter of CQ

    pkt.lin_pdma.src_addr_lo      = src & 0xFFFFFFFF;
    pkt.lin_pdma.dst_addr_lo      = dst & 0xFFFFFFFF;

    pkt.lin_pdma_data.src_addr_hi = src >> 32;
    pkt.lin_pdma_data.dst_addr_hi = dst >> 32;

    pkt.lin_pdma.tsize = size;
}

uint64_t PqmPktUtils::getLpdmaCmdSize(bool useBarrier)
{
    return useBarrier ? sizeof(lin_pdma_single_wr_comp_t) :
           sizeof(pqm_packet_lin_pdma) + sizeof(g3_lin_pdma_data);
}

uint64_t PqmPktUtils::getFenceCmdSize()
{
    return sizeof(pqm_packet_fence);
}

void PqmPktUtils::buildPqmFenceCmd( uint8_t* pktBuffer,
                                    uint8_t  fenceId,
                                    uint32_t fenceDecVal,
                                    uint32_t fenceTarget)
{
    pqm_packet_fence& pkt = *(reinterpret_cast<pqm_packet_fence*>(pktBuffer));

    assert(fenceId < MAX_VALID_FENCE_ID);
    memset(&pkt, 0x0, sizeof(pqm_packet_fence));
    pkt.dec_val     = fenceDecVal;
    pkt.target_val  = fenceTarget;
    pkt.id          = fenceId;
    pkt.opcode      = PQM_PACKET_FENCE;
}

void PqmPktUtils::buildPqmMsgLong(  void* pktBuffer,
                                    uint32_t val,
                                    uint64_t address)
{
    pqm_packet_msg_long& pkt = *(reinterpret_cast<pqm_packet_msg_long*>(pktBuffer));
    memset(&pkt, 0x0, sizeof(pqm_packet_msg_long));

    pkt.opcode = PQM_PACKET_MSG_LONG;
    pkt.value  = val;
    pkt.addr   = address;
}

uint64_t PqmPktUtils::getMsgLongCmdSize()
{
    return sizeof(pqm_packet_msg_long);
}

void PqmPktUtils::buildNopCmd(void* pktBuffer)
{
    pqm_packet_nop& pkt = *(reinterpret_cast<pqm_packet_nop*>(pktBuffer));

    pkt.opcode = PQM_PACKET_NOP;
}

uint64_t PqmPktUtils::getNopCmdSize()
{
    return sizeof(pqm_packet_nop);
}


