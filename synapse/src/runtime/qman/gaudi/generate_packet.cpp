#include "generate_packet.hpp"

#include <cstring>
#include "synapse_runtime_logging.h"

#include "gaudi/gaudi_packets.h"

using namespace gaudi;

GenWreg32::GenWreg32(uint32_t value, uint32_t regOffset, uint32_t engBarrier, uint32_t msgBarrier, uint32_t predicate)
: AddressFieldsPacket(1, offsetof(packet_wreg32, value))
{
    m_binary.opcode      = PACKET_WREG_32;
    m_binary.value       = value;
    m_binary.pred        = predicate;
    m_binary.reg_offset  = regOffset;
    m_binary.reg_barrier = 1;
    m_binary.eng_barrier = engBarrier;
    m_binary.msg_barrier = msgBarrier;
}

GenWregBulk::GenWregBulk(uint32_t  numOfBulkRegs,
                         uint32_t  regOffset,
                         uint32_t  engBarrier,
                         uint32_t  msgBarrier,
                         uint64_t* values,
                         uint32_t  predicate)
: AddressFieldsPacket(numOfBulkRegs * BULK_REGISTERS_GRANULARITY, offsetof(packet_wreg_bulk, values))
{
    m_binary.opcode      = PACKET_WREG_BULK;
    m_binary.size64      = numOfBulkRegs;
    m_binary.pred        = predicate;
    m_binary.reg_offset  = regOffset;
    m_binary.reg_barrier = 1;
    m_binary.msg_barrier = msgBarrier;
    m_binary.eng_barrier = engBarrier;

    m_pValues           = new uint64_t[numOfBulkRegs];
    uint32_t valuesSize = numOfBulkRegs * sizeof(uint64_t);
    memcpy(m_pValues, values, valuesSize);
}

GenWregBulk::~GenWregBulk()
{
    delete[] m_pValues;
}

void GenWregBulk::generatePacket(char*& ptr) const
{
    const uint64_t packetSize = sizeof(m_binary);
    memcpy(ptr, &m_binary, packetSize);
    ptr += packetSize;

    uint32_t valueSize = m_binary.size64 * sizeof(uint64_t);
    memcpy(ptr, m_pValues, valueSize);
    ptr += valueSize;
}

uint64_t GenWregBulk::getPacketSize()
{
    return sizeof(m_binary) + m_binary.size64 * sizeof(uint64_t);
}


GenMsgLong::GenMsgLong(uint32_t value,
                       uint32_t op,
                       uint32_t engBarrier,
                       uint32_t regBarrier,
                       uint32_t msgBarrier,
                       uint64_t addr,
                       uint32_t predicate)
{
    m_binary.opcode      = PACKET_MSG_LONG;
    m_binary.value       = value;
    m_binary.pred        = predicate;
    m_binary.op          = op;
    m_binary.msg_barrier = msgBarrier;
    m_binary.reg_barrier = regBarrier;
    m_binary.eng_barrier = engBarrier;
    m_binary.addr        = addr;
}

void GenMsgLong::generatePacket(char*&   pPacket,
                                uint32_t value,
                                uint32_t op,
                                uint32_t engBarrier,
                                uint32_t regBarrier,
                                uint32_t msgBarrier,
                                uint64_t addr,
                                bool     shouldIncrementPointer /* false */,
                                uint32_t predicate)
{
    auto pBinary = reinterpret_cast<packet_msg_long*>(pPacket);
    std::memset(pBinary, 0, sizeof(packet_msg_long));

    pBinary->opcode      = PACKET_MSG_LONG;
    pBinary->value       = value;
    pBinary->pred        = predicate;
    pBinary->op          = op;
    pBinary->msg_barrier = msgBarrier;
    pBinary->reg_barrier = regBarrier;
    pBinary->eng_barrier = engBarrier;
    pBinary->addr        = addr;

    if (shouldIncrementPointer)
    {
        pPacket += packetSize();
    }
}

GenMsgShort::GenMsgShort(uint32_t value,
                         uint32_t msgAddrOffset,
                         uint32_t op,
                         uint32_t base,
                         uint32_t engBarrier,
                         uint32_t regBarrier,
                         uint32_t msgBarrier)
{
    m_binary.opcode          = PACKET_MSG_SHORT;
    m_binary.value           = value;
    m_binary.msg_addr_offset = msgAddrOffset;
    m_binary.op              = op;
    m_binary.base            = base;
    m_binary.msg_barrier     = msgBarrier;
    m_binary.reg_barrier     = regBarrier;
    m_binary.eng_barrier     = engBarrier;
}

// Using non uin32_t to allow this overload
// Monitor Arm
GenMsgShort::GenMsgShort(uint8_t  syncGroupId,
                         uint8_t  syncMask,
                         bool     mode,
                         uint16_t syncValue,
                         uint32_t msgAddrOffset,
                         uint32_t op,
                         uint32_t base,
                         uint32_t engBarrier,
                         uint32_t regBarrier,
                         uint32_t msgBarrier)
{
    m_binary.opcode          = PACKET_MSG_SHORT;
    m_binary.msg_addr_offset = msgAddrOffset;
    m_binary.op              = op;
    m_binary.base            = base;
    m_binary.msg_barrier     = msgBarrier;
    m_binary.reg_barrier     = regBarrier;
    m_binary.eng_barrier     = engBarrier;

    m_binary.mon_arm_register.sync_group_id = syncGroupId;
    m_binary.mon_arm_register.mask          = syncMask;
    m_binary.mon_arm_register.mode          = mode;
    m_binary.mon_arm_register.sync_value    = syncValue;
}

// Sync-Object Update
GenMsgShort::GenMsgShort(uint16_t syncValue,
                         bool     te,
                         bool     mode,
                         uint32_t msgAddrOffset,
                         uint32_t op,
                         uint32_t base,
                         uint32_t engBarrier,
                         uint32_t regBarrier,
                         uint32_t msgBarrier)
{
    m_binary.opcode          = PACKET_MSG_SHORT;
    m_binary.msg_addr_offset = msgAddrOffset;
    m_binary.op              = op;
    m_binary.base            = base;
    m_binary.msg_barrier     = msgBarrier;
    m_binary.reg_barrier     = regBarrier;
    m_binary.eng_barrier     = engBarrier;

    m_binary.so_upd.sync_value = syncValue;
    m_binary.so_upd.te         = te;
    m_binary.so_upd.mode       = mode;
}

GenMsgProt::GenMsgProt(uint32_t value,
                       uint32_t op,
                       uint32_t engBarrier,
                       uint32_t regBarrier,
                       uint32_t msgBarrier,
                       uint64_t address,
                       uint32_t predicate)
{
    m_binary.opcode      = PACKET_MSG_PROT;
    m_binary.value       = value;
    m_binary.addr        = address;
    m_binary.pred        = predicate;
    m_binary.op          = op;
    m_binary.msg_barrier = msgBarrier;
    m_binary.reg_barrier = regBarrier;
    m_binary.eng_barrier = engBarrier;
}

GenFence::GenFence(uint32_t decVal,
                   uint32_t targetVal,
                   uint32_t id,
                   uint32_t engBarrier,
                   uint32_t regBarrier,
                   uint32_t msgBarrier,
                   uint32_t predicate)
{
    m_binary.opcode      = PACKET_FENCE;
    m_binary.pred        = predicate;
    m_binary.dec_val     = decVal;
    m_binary.target_val  = targetVal;
    m_binary.id          = id;
    m_binary.msg_barrier = msgBarrier;
    m_binary.reg_barrier = regBarrier;
    m_binary.eng_barrier = engBarrier;
}

void GenFence::generatePacket(char*&   pPacket,
                              uint32_t decVal,
                              uint32_t targetVal,
                              uint32_t id,
                              uint32_t engBarrier,
                              uint32_t regBarrier,
                              uint32_t msgBarrier,
                              bool     shouldIncrementPointer /* false */,
                              uint32_t predicate)
{
    auto pBinary = reinterpret_cast<packet_fence*>(pPacket);
    std::memset(pBinary, 0, sizeof(packet_fence));

    pBinary->opcode      = PACKET_FENCE;
    pBinary->pred        = predicate;
    pBinary->dec_val     = decVal;
    pBinary->target_val  = targetVal;
    pBinary->id          = id;
    pBinary->msg_barrier = msgBarrier;
    pBinary->reg_barrier = regBarrier;
    pBinary->eng_barrier = engBarrier;

    if (shouldIncrementPointer)
    {
        pPacket += packetSize();
    }
}

GenLinDma::GenLinDma(uint32_t tsize,
                     uint32_t engBarrier,
                     uint32_t msgBarrier,
                     uint32_t dmaDir,
                     uint64_t srcAddr,
                     uint64_t dstAddr,
                     uint8_t  dstContextIdHigh,
                     uint8_t  dstContextIdLow,
                     bool     wrComplete,
                     bool     transpose,
                     bool     dataType,
                     bool     memSet,
                     bool     compress,
                     bool     decompress)
{
    m_binary.opcode          = PACKET_LIN_DMA;
    m_binary.tsize           = tsize;
    m_binary.wr_comp_en      = wrComplete;
    m_binary.transpose       = transpose;
    m_binary.dtype           = dataType;
    m_binary.lin             = 1; /* must be 1 for linear DMA */
    m_binary.mem_set         = memSet;
    m_binary.compress        = compress;
    m_binary.decompress      = decompress;
    m_binary.context_id_low  = dstContextIdLow;
    m_binary.eng_barrier     = engBarrier;
    m_binary.reg_barrier     = 1; /* must be 1 */
    m_binary.msg_barrier     = msgBarrier;
    m_binary.src_addr        = srcAddr;
    m_binary.dst_addr        = dstAddr;
    m_binary.context_id_high = dstContextIdHigh;

    m_dmaDirection = dmaDir;
}

void GenLinDma::generateLinDma(char*&   pPacket,
                               uint32_t tsize,
                               uint32_t engBarrier,
                               uint32_t msgBarrier,
                               uint32_t dmaDir,
                               uint64_t srcAddr,
                               uint64_t dstAddr,
                               uint8_t  dstContextIdHigh,
                               uint8_t  dstContextIdLow,
                               bool     wrComplete,
                               bool     transpose,
                               bool     dataType,
                               bool     memSet,
                               bool     compress,
                               bool     decompress)
{
    auto pBinary = reinterpret_cast<packet_lin_dma*>(pPacket);
    std::memset(pBinary, 0, sizeof(packet_lin_dma));

    pBinary->opcode          = PACKET_LIN_DMA;
    pBinary->tsize           = tsize;
    pBinary->wr_comp_en      = wrComplete;
    pBinary->transpose       = transpose;
    pBinary->dtype           = dataType;
    pBinary->lin             = 1; /* must be 1 for linear DMA */
    pBinary->mem_set         = memSet;
    pBinary->compress        = compress;
    pBinary->decompress      = decompress;
    pBinary->context_id_low  = dstContextIdLow;
    pBinary->eng_barrier     = engBarrier;
    pBinary->reg_barrier     = 1; /* must be 1 */
    pBinary->msg_barrier     = msgBarrier;
    pBinary->src_addr        = srcAddr;
    pBinary->dst_addr        = dstAddr;
    pBinary->context_id_high = dstContextIdHigh;
}

uint32_t GenLinDma::getDirection() const
{
    return m_dmaDirection;
}

GenNop::GenNop(uint32_t engBarrier, uint32_t regBarrier, uint32_t msgBarrier)
{
    m_binary.opcode      = PACKET_NOP;
    m_binary.msg_barrier = msgBarrier;
    m_binary.reg_barrier = regBarrier;
    m_binary.eng_barrier = engBarrier;
}

void GenNop::generatePacket(char*&   pPacket,
                            uint32_t engBarrier,
                            uint32_t regBarrier,
                            uint32_t msgBarrier,
                            bool     shouldIncrementPointer /* false */)
{
    auto pBinary = reinterpret_cast<packet_nop*>(pPacket);
    std::memset(pBinary, 0, sizeof(packet_nop));

    pBinary->opcode      = PACKET_NOP;
    pBinary->msg_barrier = msgBarrier;
    pBinary->reg_barrier = regBarrier;
    pBinary->eng_barrier = engBarrier;

    if (shouldIncrementPointer)
    {
        pPacket += packetSize();
    }
}

GenStop::GenStop(uint32_t engBarrier)
{
    m_binary.opcode      = PACKET_STOP;
    m_binary.eng_barrier = engBarrier;
    m_binary.reg_barrier = 0; /* must be 0 */
    m_binary.msg_barrier = 0; /* must be 0 */
}

GenCpDma::GenCpDma(uint32_t tsize, uint32_t engBarrier, uint32_t msgBarrier, uint64_t addr, uint32_t predicate)
: AddressFieldsPacket(1, offsetof(packet_cp_dma, src_addr))
{
    m_binary.opcode      = PACKET_CP_DMA;
    m_binary.tsize       = tsize;
    m_binary.pred        = predicate;
    m_binary.eng_barrier = engBarrier;
    m_binary.reg_barrier = 1; /* must be 1 */
    m_binary.msg_barrier = msgBarrier;
    m_binary.src_addr    = addr;
}

void GenCpDma::generateCpDma(char*&   pPacket,
                             uint32_t tsize,
                             uint32_t engBarrier,
                             uint32_t msgBarrier,
                             uint64_t addr,
                             uint32_t predicate)
{
    auto pBinary = reinterpret_cast<packet_cp_dma*>(pPacket);
    std::memset(pBinary, 0, sizeof(packet_cp_dma));

    pBinary->opcode      = PACKET_CP_DMA;
    pBinary->tsize       = tsize;
    pBinary->pred        = predicate;
    pBinary->eng_barrier = engBarrier;
    pBinary->reg_barrier = 1; /* must be 1 */
    pBinary->msg_barrier = msgBarrier;
    pBinary->src_addr    = addr;
}

void GenCpDma::generateDefaultCpDma(char*& pPacket, uint32_t tsize, uint64_t addr)
{
    auto pBinary = reinterpret_cast<packet_cp_dma*>(pPacket);

    pBinary->opcode      = PACKET_CP_DMA;
    pBinary->tsize       = tsize;
    pBinary->pred        = 0;
    pBinary->eng_barrier = 0;
    pBinary->reg_barrier = 1; /* must be 1 */
    pBinary->msg_barrier = 0;
    pBinary->src_addr    = addr;
}

void* GenCpDma::getBinaryAddress()
{
    return &m_binary;
}

GenArbitrationPoint::GenArbitrationPoint(uint8_t priority, bool priorityRelease, uint32_t predicate)
{
    m_binary.opcode      = PACKET_ARB_POINT;
    m_binary.pred        = predicate;
    m_binary.eng_barrier = 0; /* must be 0 */
    m_binary.reg_barrier = 0; /* must be 0 */
    m_binary.msg_barrier = 0; /* must be 0 */
    m_binary.priority    = (unsigned)priority;
    m_binary.rls         = priorityRelease;
}

void GenArbitrationPoint::generateArbitrationPoint(char*&   pPacket,
                                                   uint8_t  priority,
                                                   bool     priorityRelease,
                                                   uint32_t predicate)
{
    packet_arb_point* pBinary = reinterpret_cast<packet_arb_point*>(pPacket);
    std::memset(pBinary, 0, sizeof(packet_arb_point));

    pBinary->opcode      = PACKET_ARB_POINT;
    pBinary->pred        = predicate;
    pBinary->eng_barrier = 0; /* must be 0 */
    pBinary->reg_barrier = 0; /* must be 0 */
    pBinary->msg_barrier = 0; /* must be 0 */
    pBinary->priority    = (unsigned)priority;
    pBinary->rls         = priorityRelease;
}

GenRepeat::GenRepeat(bool     isRepeatStart,
                     bool     isOuterLoop,
                     uint16_t jumpPtr,
                     uint32_t engBarrier,
                     uint32_t msgBarrier,
                     uint16_t predicate)
{
    m_binary.opcode      = PACKET_REPEAT;
    m_binary.pred        = predicate;
    m_binary.sore        = isRepeatStart;
    m_binary.o           = isOuterLoop;
    m_binary.jmp_ptr     = jumpPtr;
    m_binary.eng_barrier = engBarrier;
    m_binary.reg_barrier = 1; /* must be 1 */
    m_binary.msg_barrier = msgBarrier;
}

GenWait::GenWait(uint8_t incVal, uint8_t id, uint32_t numCyclesToWait, uint32_t engBarrier, uint32_t msgBarrier)
{
    m_binary.opcode             = PACKET_WAIT;
    m_binary.num_cycles_to_wait = numCyclesToWait;
    m_binary.inc_val            = incVal;
    m_binary.id                 = id;
    m_binary.eng_barrier        = engBarrier;
    m_binary.reg_barrier        = 0; /* must be 0 */
    m_binary.msg_barrier        = msgBarrier;
}

GenLoadAndExecute::GenLoadAndExecute(bool     isLoad,
                                     bool     isDst, /* else will load predicates */
                                     bool     isExecute,
                                     bool     isEType,
                                     uint32_t engBarrier,
                                     uint32_t msgBarrier,
                                     uint64_t srcAddr,
                                     uint16_t predicate)
{
    m_binary.opcode      = PACKET_LOAD_AND_EXE;
    m_binary.pred        = predicate;
    m_binary.dst         = isDst;
    m_binary.load        = isLoad;
    m_binary.exe         = isExecute;
    m_binary.etype       = isEType;
    m_binary.eng_barrier = engBarrier;
    m_binary.reg_barrier = 1; /* must be 1 */
    m_binary.msg_barrier = msgBarrier;
    m_binary.src_addr    = srcAddr;
}

void GenLoadAndExecute::generatePacket(char*&   pPacket,
                                       bool     isLoad,
                                       bool     isDst, /* else will load predicates */
                                       bool     isExecute,
                                       bool     isEType,
                                       uint32_t engBarrier,
                                       uint32_t msgBarrier,
                                       uint64_t srcAddr,
                                       bool     shouldIncrementPointer /* false */,
                                       uint16_t predicate)
{
    auto pBinary = reinterpret_cast<packet_load_and_exe*>(pPacket);
    std::memset(pBinary, 0, sizeof(packet_load_and_exe));

    pBinary->opcode      = PACKET_LOAD_AND_EXE;
    pBinary->pred        = predicate;
    pBinary->dst         = isDst;
    pBinary->load        = isLoad;
    pBinary->exe         = isExecute;
    pBinary->etype       = isEType;
    pBinary->eng_barrier = engBarrier;
    pBinary->reg_barrier = 1; /* must be 1 */
    pBinary->msg_barrier = msgBarrier;
    pBinary->src_addr    = srcAddr;

    if (shouldIncrementPointer)
    {
        pPacket += packetSize();
    }
}