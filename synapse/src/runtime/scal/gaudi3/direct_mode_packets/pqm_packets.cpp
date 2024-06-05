#include "pqm_packets.hpp"
#include "infra/defs.h"
#include <cstring>
#include "log_manager.h"

using namespace pqm;
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
    pqm_packet_lin_pdma lin_pdma;
    g3_lin_pdma_data    lin_pdma_data;

    union
    {
        g3_lin_pdma_fence_wo_wr_comp fence_wo_wr_comp;
        g3_lin_pdma_wr_comp_data_lo  lin_pdma_wr_comp0_data_lo;
    };

    struct g3_lin_pdma_wr_comp_data_hi lin_pdma_wr_comp0_data_hi;
} __attribute__ ((aligned(8), __packed__));


void LinPdma::build(uint8_t*           pktBuffer,
                    uint64_t           src,
                    uint64_t           dst,
                    uint32_t           size,
                    bool               bMemset,
                    PdmaDir            direction,
                    LinPdmaBarrierMode barrierMode,
                    uint64_t           barrierAddress,
                    uint32_t           barrierData,
                    uint32_t           fenceDecVal,
                    uint32_t           fenceTarget,
                    uint32_t           fenceId)
{
    HB_ASSERT(direction != PdmaDir::INVALID, "lin pdma direction is invalid");

    lin_pdma_single_wr_comp_t& pkt = reinterpret_cast<lin_pdma_single_wr_comp_t&>(*pktBuffer);

    bool isFenceEnabled = (fenceId <= MAX_VALID_FENCE_ID);

    bool useBarrier = (barrierMode != LinPdmaBarrierMode::DISABLED);
    memset(&pkt, 0x0, getSize(useBarrier));
    if (useBarrier)
    {
        pkt.lin_pdma.wrcomp = 1;

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
    pkt.lin_pdma.direction         = (uint32_t) direction;
    pkt.lin_pdma.memset            = bMemset;
    pkt.lin_pdma.en_desc_commit    = 1;
    pkt.lin_pdma.inc_context_id    = 1;
    pkt.lin_pdma.advcnt_hbw_complq = (barrierMode == LinPdmaBarrierMode::INTERNAL);

    pkt.lin_pdma.src_addr_lo      = src & 0xFFFFFFFF;
    pkt.lin_pdma.dst_addr_lo      = dst & 0xFFFFFFFF;

    pkt.lin_pdma_data.src_addr_hi = src >> 32;
    pkt.lin_pdma_data.dst_addr_hi = dst >> 32;

    pkt.lin_pdma.tsize = size;
}

uint64_t LinPdma::getSize(bool useBarrier)
{
    return useBarrier ?
        sizeof(lin_pdma_single_wr_comp_t) :
        sizeof(pqm_packet_lin_pdma) + sizeof(g3_lin_pdma_data);
}

uint64_t LinPdma::dump(const uint8_t* pktBuffer)
{
    HB_ASSERT_PTR(pktBuffer);

    const lin_pdma_single_wr_comp_t& pkt = *(reinterpret_cast<const lin_pdma_single_wr_comp_t*>(pktBuffer));

    std::string paramsStr;

    HB_ASSERT(pkt.lin_pdma.wrcomp <= 1, "Only up-to a single WR-Comp is supported");
    bool useBarrier = (pkt.lin_pdma.wrcomp != 0);

    if (pkt.lin_pdma.fence_en)
    {
        FenceData fenceData;
        fenceData.raw =
            useBarrier ? pkt.lin_pdma_wr_comp0_data_hi.fence_data_with_wr_comp : pkt.fence_wo_wr_comp.fence_data;

        paramsStr += fmt::format("Fence: fenceId {} fenceTarget {} fenceDecVal {}{}",
                                 (unsigned)fenceData.fenceId,
                                 (unsigned)fenceData.fenceTarget,
                                 (unsigned)fenceData.fenceDecVal,
                                 useBarrier ? " " : "");
    }

    if (useBarrier)
    {
        paramsStr += fmt::format("WR-Completion: address [{:#x} {:#x}], data {:#x}",
                                 pkt.lin_pdma_wr_comp0_data_hi.wr_comp_addr_hi,
                                 pkt.lin_pdma_wr_comp0_data_lo.wr_comp_addr_lo,
                                 pkt.lin_pdma_wr_comp0_data_lo.wr_comp_data);
    }

    LOG_DEBUG(SYN_DM_STREAM,
                   "opcode {:#x}, direction {}, isMemset {}, size {}, SRC [{:#x} {:#x}], DST [{:#x} {:#x}], [{}]",
                   pkt.lin_pdma.opcode,
                   pkt.lin_pdma.direction ? "host to device" : "device to host",
                   pkt.lin_pdma.memset,
                   pkt.lin_pdma.tsize,
                   pkt.lin_pdma_data.src_addr_hi,
                   pkt.lin_pdma.src_addr_lo,
                   pkt.lin_pdma_data.dst_addr_hi,
                   pkt.lin_pdma.dst_addr_lo,
                   paramsStr);

    return getSize(useBarrier);
}

// Fence PQM packet

void Fence::build(void* pktBuffer,
                  uint8_t  fenceId,
                  uint32_t fenceDecVal,
                  uint32_t fenceTarget)
{
    pktType& pkt = *(reinterpret_cast<pktType*>(pktBuffer));

    HB_ASSERT(fenceId < MAX_VALID_FENCE_ID, "fence id is illegal!");
    memset(&pkt, 0x0, sizeof(pktType));
    pkt.dec_val     = fenceDecVal;
    pkt.target_val  = fenceTarget;
    pkt.id          = fenceId;
    pkt.opcode      = PQM_PACKET_FENCE;
}

uint64_t Fence::dump(const void* pktBuffer)
{
    HB_ASSERT_PTR(pktBuffer);

    const pktType& pkt = *(reinterpret_cast<const pktType*>(pktBuffer));

    LOG_DEBUG(SYN_DM_STREAM,
                   "PQM fence packet: opcode {:#x} fenceId {}, target {}, decVal {}",
                   pkt.opcode, pkt.id, pkt.target_val, pkt.dec_val);
    return getSize();
}

// Msg Long

void MsgLong::build(void* pktBuffer,
                    uint32_t val,
                    uint64_t address)
{
    pktType& pkt = *(reinterpret_cast<pktType*>(pktBuffer));
    memset(&pkt, 0x0, sizeof(pktType));

    pkt.opcode = PQM_PACKET_MSG_LONG;
    pkt.value  = val;
    pkt.addr   = address;
}

uint64_t MsgLong::dump(const void* pktBuffer)
{
    HB_ASSERT_PTR(pktBuffer);

    const pktType& pkt = *(reinterpret_cast<const pktType*>(pktBuffer));

    LOG_DEBUG(SYN_DM_STREAM,
                   "PQM Msg Long packet: opcode {:#x} value {:#x}, address {:#x}",
                   pkt.opcode, pkt.value, pkt.addr);
    return getSize();
}

// Msg Short

void MsgShort::build(void*    pktBuffer,
                     uint32_t val,
                     uint8_t  baseIndex,
                     uint16_t elementOffset)
{
    pktType& pkt = *(reinterpret_cast<pktType*>(pktBuffer));
    memset(&pkt, 0x0, sizeof(pktType));

    pkt.opcode          = PQM_PACKET_MSG_SHORT;
    pkt.value           = val;
    pkt.base            = baseIndex;
    pkt.msg_addr_offset = elementOffset;
}

uint64_t MsgShort::dump(const void* pktBuffer)
{
    HB_ASSERT_PTR(pktBuffer);

    const pktType& pkt = *(reinterpret_cast<const pktType*>(pktBuffer));

    LOG_DEBUG(SYN_DM_STREAM,
                   "PQM Msg Short packet: opcode {:#x} value {:#x}, baseIndex {:#x}, elementOffset {:#x}",
                   pkt.opcode, pkt.value, pkt.base, pkt.msg_addr_offset);
    return getSize();
}

// Nop

void Nop::build(void* pktBuffer)
{
    HB_ASSERT_PTR(pktBuffer);

    pktType& pkt = *(reinterpret_cast<pktType*>(pktBuffer));

    pkt.opcode = PQM_PACKET_NOP;
}

uint64_t Nop::dump(const void* pktBuffer)
{
    const pktType& pkt = *(reinterpret_cast<const pktType*>(pktBuffer));
    LOG_DEBUG(SYN_DM_STREAM, "opcode: {:#x}", pkt.opcode);
    return getSize();
}

void ChWreg32::build(void* pktBuffer, uint32_t regOffset, uint32_t value)
{
    pktType& pkt = *(reinterpret_cast<pktType*>(pktBuffer));

    memset(&pkt, 0x0, sizeof(pktType));
    pkt.value      = value;
    pkt.reg_offset = regOffset;
    pkt.opcode     = PQM_PACKET_CH_WREG_32;
}

uint64_t ChWreg32::dump(const void* pktBuffer)
{
    const pktType& pkt = *(reinterpret_cast<const pktType*>(pktBuffer));

    LOG_DEBUG(SYN_DM_STREAM,
                   "PQM CH_WREG32 packet: opcode {:#x} reg_offset {:#x} value {:#x}",
                   pkt.opcode,
                   pkt.reg_offset,
                   pkt.value);
    return getSize();
}

// Static functions

static const char* packetToName(uint32_t opcode)
{
#define PACKET_CASE(X)                  \
    case X:                             \
        return #X;

#define UNUSED_PACKET_CASE(X)           \
    case X:                             \
        return "Unused opcode " #X

    switch (opcode)
    {
        PACKET_CASE(PQM_PACKET_LIN_PDMA);
        PACKET_CASE(PQM_PACKET_FENCE);
        PACKET_CASE(PQM_PACKET_MSG_LONG);
        PACKET_CASE(PQM_PACKET_NOP);
        PACKET_CASE(PQM_PACKET_CH_WREG_32);

        default:
            return "Unknown PQM opcode";
    }

    // Cannot reach here
    return "NA";
}

uint64_t pqm::dumpPqmPacket(const uint8_t* pktBuffer)
{
    const pqm_packet_nop& pkt =
        reinterpret_cast<const pqm_packet_nop&>(*pktBuffer);

    uint32_t opCode  = pkt.opcode;
    uint64_t cmdSize = 0;

    LOG_DEBUG(SYN_DM_STREAM,
                   "sending {} opcode {} from addr {:x}",
                   packetToName(opCode),
                   opCode,
                   TO64(pktBuffer));
    switch (opCode)
    {
        case PQM_PACKET_LIN_PDMA:
        {
            cmdSize = LinPdma::dump(pktBuffer);
            break;
        }
        case PQM_PACKET_FENCE:
        {
            cmdSize = Fence::dump(pktBuffer);
            break;
        }
        case PQM_PACKET_MSG_LONG:
        {
            cmdSize = MsgLong::dump(pktBuffer);
            break;
        }
        case PQM_PACKET_MSG_SHORT:
        {
            cmdSize = MsgShort::dump(pktBuffer);
            break;
        }
        case PQM_PACKET_NOP:
        {
            cmdSize = Nop::dump(pktBuffer);
            break;
        }
        case PQM_PACKET_CH_WREG_32:
        {
            cmdSize = ChWreg32::dump(pktBuffer);
            break;
        }
    }

    return cmdSize;
}
