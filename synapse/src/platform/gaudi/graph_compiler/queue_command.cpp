#include "platform/gaudi/graph_compiler/queue_command.h"

#include "address_fields_container_info.h"
#include "defs.h"
#include "gaudi/asic_reg/gaudi_blocks.h"
#include "gaudi/asic_reg_structs/dma_core_regs.h"
#include "gaudi/asic_reg_structs/mme_regs.h"
#include "gaudi/asic_reg_structs/qman_regs.h"
#include "gaudi/asic_reg_structs/sync_mngr_regs.h"
#include "gaudi/asic_reg_structs/tpc_regs.h"
#include "gaudi/gaudi_packets.h"
#include "graph_compiler/compilation_hal_reader.h"
#include "habana_global_conf.h"
#include "platform/gaudi/graph_compiler/sync/sync_conventions.h"
#include "platform/gaudi/utils.hpp"
#include "runtime/qman/gaudi/generate_packet.hpp"
#include "types.h"
#include "utils.h"
#include "utils.h"

#include <memory>
#include <string>

namespace gaudi
{

void setSendSyncEvents(uint32_t& raw)
{
    raw |= (1 << SEND_SYNC_BIT);
}

enum EMonPayloadSelect
{
    MON_PAYLOAD_ADDR_L,
    MON_PAYLOAD_ADDR_H,
    MON_PAYLOAD_DATA
};

unsigned getRegForLoadDesc(HabanaDeviceType type, unsigned deviceID)
{
    switch (type)
    {
    case DEVICE_MME:
        return offsetof(block_mme, arch);

    case DEVICE_TPC:
        return offsetof(block_tpc, qm_tensor_0);

    case DEVICE_DMA_DRAM_SRAM_BIDIRECTIONAL:
        return offsetof(block_dma_core, cfg_0);

    default:
        HB_ASSERT(0, "Unsupported device type");
        break;
    }
    return 0;
}

uint64_t getSyncObjectAddress(unsigned so)
{
    static const uint64_t baseAddr = mmSYNC_MNGR_GLBL_E_N_BASE +
                                     offsetof(block_sync_mngr, sync_mngr_objs);

    HB_ASSERT(so < 2048, "sync-obj out of bound");

    return baseAddr + varoffsetof(block_sob_objs, sob_obj[so]);
}

static uint64_t getMonPayloadAddress(unsigned mon, EMonPayloadSelect payloadSelect)
{
    static const uint64_t baseAddr = mmSYNC_MNGR_GLBL_E_N_BASE +
                                     offsetof(block_sync_mngr, sync_mngr_objs);

    HB_ASSERT(mon < 512, "monitor out of bound");

    switch (payloadSelect)
    {
    case MON_PAYLOAD_ADDR_L:
        return baseAddr + varoffsetof(block_sob_objs, mon_pay_addrl[mon]);

    case MON_PAYLOAD_ADDR_H:
        return baseAddr + varoffsetof(block_sob_objs, mon_pay_addrh[mon]);

    case MON_PAYLOAD_DATA:
        return baseAddr + varoffsetof(block_sob_objs, mon_pay_data[mon]);

    default:
        HB_ASSERT(0, "Unsupported monitor payload select");
    }
    return 0;
}

static unsigned getRegForExecute(HabanaDeviceType type, unsigned deviceID)
{
    switch (type)
    {
    case DEVICE_MME:
        return offsetof(block_mme, cmd);

    case DEVICE_TPC:
        return offsetof(block_tpc, tpc_execute);

    case DEVICE_DMA_DRAM_SRAM_BIDIRECTIONAL:
        return offsetof(block_dma_core, commit);

    default:
        HB_ASSERT(0, "Unsupported device type");
        break;
    }
    return 0;
}

unsigned getRegForEbPadding()
{
    // using DBGMEM_DATA_WR for EB bug Padding
    return (offsetof(block_tpc, dbgmem_data_wr));
}

static void prepareFieldInfosTwoDwordsHeader(BasicFieldInfoSet& basicFieldsInfoSet)
{
    BasicFieldInfoSet updatedBasicFieldSet;
    for(auto& singleBasicFieldsInfoPair : basicFieldsInfoSet)
    {
        auto copy = singleBasicFieldsInfoPair;
        // Adding two bytes for header size
        copy.first += 2;
        copy.second->setFieldIndexOffset(copy.second->getFieldIndexOffset() + 2);
        updatedBasicFieldSet.insert(copy);
    }

    basicFieldsInfoSet.clear();
    basicFieldsInfoSet.insert(updatedBasicFieldSet.begin(), updatedBasicFieldSet.end());
}

static void prepareFieldInfosTwoDwordsHeader(BasicFieldInfoSet& basicFieldsInfoSet, AddressFieldInfoSet& addressFieldsInfoSet)
{
    BasicFieldInfoSet tempBasicSet;
    transferAddressToBasic(addressFieldsInfoSet, tempBasicSet);
    prepareFieldInfosTwoDwordsHeader(tempBasicSet);
    transferBasicToAddress(tempBasicSet, addressFieldsInfoSet);

    prepareFieldInfosTwoDwordsHeader(basicFieldsInfoSet);
}

// --------------------------------------------------------
// ------------------ GaudiQueueCommand -------------------
// --------------------------------------------------------

GaudiQueueCommand::GaudiQueueCommand()
  : QueueCommand()
{
}

GaudiQueueCommand::GaudiQueueCommand(uint32_t packetType)
  : QueueCommand(packetType)
{
}

GaudiQueueCommand::GaudiQueueCommand(uint32_t packetType, uint64_t commandId)
  : QueueCommand(packetType, commandId)
{
}

GaudiQueueCommand::~GaudiQueueCommand()
{
}

void GaudiQueueCommand::WritePB(gc_recipe::generic_packets_container *pktCon)
{
    LOG_ERR(QMAN, "{}: should not be invoked for Gaudi", HLLOG_FUNC);
    HB_ASSERT(false, "Func should not be invoked for Gaudi");
}

void GaudiQueueCommand::WritePB(gc_recipe::generic_packets_container* pktCon, ParamsManager* params)
{
    LOG_ERR(QMAN, "{}: should not be invoked for Gaudi", HLLOG_FUNC);
    HB_ASSERT(false, "Func should not be invoked for Gaudi");
}

// --------------------------------------------------------
// ---------------------- DmaCommand ----------------------
// --------------------------------------------------------


// --------------------------------------------------------
// ------------------ DmaDeviceInternal -------------------
// --------------------------------------------------------

DmaDeviceInternal::DmaDeviceInternal(deviceAddrOffset  src,
                                     bool              srcInDram,
                                     deviceAddrOffset  dst,
                                     bool              dstInDram,
                                     uint64_t          size,
                                     bool              setEngBarrier,
                                     bool              isMemset,
                                     bool              wrComplete,
                                     uint16_t          contextId)
  : DmaCommand(),
    m_src(src),
    m_dst(dst)
{
    if (isMemset)
    {
        if (dstInDram)
        {
            m_operationStr = "DMA memset to DRAM:";
        }
        else
        {
            m_operationStr = "DMA memset to SRAM:";
        }
    }
    else
    {
        if (srcInDram && dstInDram)
        {
            m_operationStr = "DMA memcpy from DRAM to DRAM:";
        }
        else if (srcInDram && !dstInDram)
        {
            m_operationStr = "DMA memcpy from DRAM to SRAM:";
        }
        else if (!srcInDram && dstInDram)
        {
            m_operationStr = "DMA memcpy from SRAM to DRAM:";
        }
        else // !srcInDram && !dstInDram
        {
            m_operationStr = "DMA memcpy from SRAM to SRAM:";
        }
    }

    m_binary = {0};

    m_binary.tsize           = size;

    m_binary.wr_comp_en      = wrComplete ? 1 : 0;
    m_binary.transpose       = 0;
    m_binary.dtype           = 0;
    m_binary.lin             = 1;            /* must be 1 for linear DMA */
    m_binary.mem_set         = isMemset ? 1 : 0;
    m_binary.compress        = 0;
    m_binary.decompress      = 0;
    m_binary.reserved        = 0;
    m_binary.context_id_low  = (contextId & 0xFF);
    m_binary.context_id_high = ((contextId & 0xFF00) >> 8);
    m_binary.opcode          = PACKET_LIN_DMA;
    m_binary.eng_barrier     = setEngBarrier ? 1 : 0;
    m_binary.reg_barrier     = 1;            /* must be 1 */
    m_binary.msg_barrier     = 0x1;

    m_binary.src_addr = m_src;

    m_binary.dst_addr_ctx_id_raw = m_dst;
}

unsigned DmaDeviceInternal::GetBinarySize() const
{
    return sizeof(packet_lin_dma);
}

DmaDeviceInternal::~DmaDeviceInternal()
{
}

void DmaDeviceInternal::Print() const
{
    LOG_DEBUG(QMAN, "      {} src=0x{:x} dst=0x{:x} size=0x{:x}{}",
              m_operationStr,
              m_src,
              m_dst,
              m_binary.tsize,
              m_binary.wr_comp_en ? " Signal completion *****" : "");
}

uint64_t DmaDeviceInternal::writeInstruction(void *whereTo) const
{
    memcpy(whereTo, &m_binary, sizeof(m_binary));
    return sizeof(m_binary);
}

void DmaDeviceInternal::prepareFieldInfos()
{
    HB_ASSERT(4 >= m_addressContainerInfo.size(), "Unexpected number of patching points for LinDma command");
    prepareFieldInfosTwoDwordsHeader(m_addressContainerInfo.retrieveBasicFieldInfoSet(),
                                     m_addressContainerInfo.retrieveAddressFieldInfoSet());
}

// --------------------------------------------------------
// -------------------- DmaDramToSram ---------------------
// --------------------------------------------------------

DmaDramToSram::DmaDramToSram(deviceAddrOffset  dramPtr,
                             deviceAddrOffset  sramPtr,
                             uint64_t          size,
                             bool              wrComplete,
                             uint16_t          contextID)
  : DmaDeviceInternal(dramPtr, true, sramPtr, false, size, false, false, wrComplete, contextID)
{
}

// --------------------------------------------------------
// -------------------- DmaSramToDram ---------------------
// --------------------------------------------------------

DmaSramToDram::DmaSramToDram(deviceAddrOffset  dramPtr,
                             deviceAddrOffset  sramPtr,
                             uint64_t          size,
                             bool              wrComplete,
                             uint16_t          contextID)
  : DmaDeviceInternal(sramPtr, false, dramPtr, true, size, false, false, wrComplete, contextID)
{
}

// --------------------------------------------------------
// ------------------------ CpDma -------------------------
// --------------------------------------------------------

CpDma::CpDma(deviceAddrOffset addrOffset, uint64_t size, uint64_t dramBase, uint32_t predicate)
  : GaudiQueueCommand(),
    m_addrOffset(addrOffset),
    m_transferSize(size)
{
    UNUSED(dramBase);
    m_binary = { 0 };

    m_binary.tsize          = m_transferSize;

    m_binary.pred           = predicate;
    m_binary.opcode         = PACKET_CP_DMA;
    m_binary.eng_barrier    = 0x0;
    m_binary.reg_barrier    = 0x1;            /* must be 1 */
    m_binary.msg_barrier    = 0x1;

    m_binary.src_addr       = m_addrOffset;
}

void CpDma::Print() const
{
    LOG_DEBUG(QMAN, "      DMA from SRAM [0x{:x}] size {} to the tightly-coupled logic", m_addrOffset, m_transferSize);
}

unsigned CpDma::GetBinarySize() const
{
    return sizeof(packet_cp_dma);
}

uint64_t CpDma::writeInstruction(void *whereTo) const
{
    memcpy(whereTo, &m_binary, sizeof(m_binary));
    return sizeof(m_binary);
}

void CpDma::prepareFieldInfos()
{
    HB_ASSERT(2 >= m_addressContainerInfo.size(), "Unexpected number of patching points for CpDma command");
    prepareFieldInfosTwoDwordsHeader(m_addressContainerInfo.retrieveBasicFieldInfoSet(),
                                     m_addressContainerInfo.retrieveAddressFieldInfoSet());
}

// --------------------------------------------------------
// -------------------- WriteRegister ---------------------
// --------------------------------------------------------

WriteRegister::WriteRegister(unsigned regOffset, unsigned value, uint32_t predicate)
  : GaudiQueueCommand()
{
    HB_ASSERT(fitsInBits(regOffset, 16), "Cant write to register at offset {}", regOffset);
    fillBinary(regOffset, value, predicate);
}

WriteRegister::WriteRegister(unsigned regOffset, unsigned value, uint64_t commandId, uint32_t predicate)
  : GaudiQueueCommand(INVALID_PACKET_TYPE, commandId)
{
    HB_ASSERT(fitsInBits(regOffset, 16), "Cant write to register at offset {}", regOffset);
    fillBinary(regOffset, value, predicate);
}

void WriteRegister::fillBinary(unsigned regOffset, unsigned value, uint32_t predicate)
{
    m_binary = {0};

    m_binary.value       = value;

    m_binary.pred        = predicate;
    m_binary.reg_offset  = regOffset;
    m_binary.opcode      = PACKET_WREG_32;
    m_binary.eng_barrier = 0x0;
    m_binary.reg_barrier = 0x1;
    m_binary.msg_barrier = 0x1;
}

void WriteRegister::Print() const
{
    LOG_DEBUG(QMAN, "      Write 0x{:x} to register 0x{:x} pred={}", m_binary.value, m_binary.reg_offset, m_binary.pred);
}

unsigned WriteRegister::GetBinarySize() const
{
    return sizeof(packet_wreg32);
}

uint64_t WriteRegister::writeInstruction(void* whereTo) const
{
    memcpy(whereTo, &m_binary, sizeof(m_binary));
    return sizeof(m_binary);
}

void WriteRegister::prepareFieldInfos()
{

}

unsigned WriteRegister::getRegOffset() const
{
    return m_binary.reg_offset;
}

// --------------------------------------------------------
// -------------------- EbPadding ---------------------
// --------------------------------------------------------

EbPadding::EbPadding(unsigned numPadding) : GaudiQueueCommand()
{
    m_numPadding       = numPadding;
    unsigned regOffset = getRegForEbPadding();
    HB_ASSERT(fitsInBits(regOffset, 16), "fitsInBits failed");
    fillBinary(regOffset);
}

void EbPadding::fillBinary(unsigned regOffset)
{
    m_binary            = {0};
    m_binary.value      = {0};
    m_binary.opcode     = PACKET_WREG_32;
    m_binary.reg_offset = regOffset;
}

void EbPadding::Print() const
{
    LOG_DEBUG(QMAN,
              "      Padding before engine-barrier, filling {} WREG_32 writes to address 0x{:x}, pred={}",
              m_numPadding,
              getRegOffset(),
              m_binary.pred);
}

void EbPadding::prepareFieldInfos() {}

uint64_t EbPadding::writeInstruction(void* whereTo) const
{
    char*    whereToChar         = reinterpret_cast<char*>(whereTo);
    unsigned bytesWritten        = 0;
    uint32_t EbPaddingPacketSize = sizeof(packet_wreg32);
    for (unsigned i = 0; i < m_numPadding; ++i)
    {
        memcpy(whereToChar + bytesWritten, (uint32_t*)&m_binary, EbPaddingPacketSize);
        bytesWritten += EbPaddingPacketSize;
    }
    return bytesWritten;
}

unsigned EbPadding::getEbPaddingNumPadding() const
{
    return m_numPadding;
}

unsigned EbPadding::getRegOffset() const
{
    return m_binary.reg_offset;
}

unsigned EbPadding::GetBinarySize() const
{
    return sizeof(packet_wreg32) * m_numPadding;
}

uint32_t EbPadding::getValue() const
{
    return m_binary.value;
}

void EbPadding::setValue(uint32_t value)
{
    m_binary.value = value;
}

// --------------------------------------------------------
// ----------------- WriteManyRegisters -------------------
// --------------------------------------------------------

WriteManyRegisters::WriteManyRegisters(unsigned        firstRegOffset,
                                       unsigned        count32bit,
                                       const uint32_t* values,
                                       uint32_t        predicate)
  : GaudiQueueCommand(),
    m_alignmentReg(nullptr),
    m_remainderReg(nullptr),
    m_incZeroOffset(true),
    m_incOffsetValue(0)
{
    HB_ASSERT(count32bit > 0, "WriteManyRegisters should contain at least one register");

    unsigned totalRegs = count32bit;
    unsigned offset    = firstRegOffset;
    unsigned curReg    = 0;

    // If the first register offset isn't aligned to 8 bytes, we write it using wreg32 so next register will align
    if ((firstRegOffset & 0x7) != 0)
    {
        m_alignmentReg = new WriteRegister(offset, values[curReg], m_commandId, predicate);
        offset += sizeof(uint32_t);
        curReg++;
        m_incZeroOffset = false; // If we have a wreg32 here, then patch point in offset 0 should not be updated
        m_incOffsetValue++;     // Header of wreg32
    }

    // We use bulk-write for two or more 32bit registers
    unsigned bulkSize = (totalRegs - curReg) / 2;
    if (bulkSize)
    {
        m_writeBulkBinary = {0};
        m_writeBulkBinary.size64 = bulkSize;
        m_writeBulkBinary.pred = predicate;
        m_writeBulkBinary.reg_offset = offset;
        m_writeBulkBinary.opcode = PACKET_WREG_BULK;
        m_writeBulkBinary.eng_barrier = 0x0;
        m_writeBulkBinary.reg_barrier = 0x1; /* must be 1 */
        m_writeBulkBinary.msg_barrier = 0x1;

        unsigned bulkRegs = bulkSize * 2;
        m_valuesBinary.resize(bulkSize);
        memcpy(m_valuesBinary.data(), values + curReg, sizeof(values[0]) * bulkRegs);
        curReg += bulkRegs;
        offset += sizeof(uint32_t) * bulkRegs;
        m_incOffsetValue += 2;  // Headers of WriteBulk
    }

    // If we have odd number of registers (besides the alignment register), the last one is written using wreg32
    if (curReg < totalRegs)
    {
        m_remainderReg = new WriteRegister(offset, values[curReg], m_commandId, predicate);
        curReg++;
    }

    HB_ASSERT(curReg == totalRegs, "Something is wrong, shouldn't have any registers left unprocessed");
}

WriteManyRegisters::~WriteManyRegisters()
{
    delete m_alignmentReg;
    delete m_remainderReg;
}

void WriteManyRegisters::Print() const
{
    if (!LOG_LEVEL_AT_LEAST_DEBUG(QMAN)) return;

    if (m_alignmentReg != nullptr)
    {
        m_alignmentReg->Print();
    }
    if (!m_valuesBinary.empty())
    {
        uint64_t numOfDoubleRegsToPrint = std::min((uint64_t) GCFG_DEBUG_NUM_OF_DOUBLE_WBULK_REGS_TO_DUMP.value(),
                                                   (uint64_t) m_valuesBinary.size());
        LOG_DEBUG(QMAN, "      WREG_BULK {} 32bit registers to 0x{:x} pred={}. Printing {} registers:",
                  m_valuesBinary.size() * 2, m_writeBulkBinary.reg_offset,
                  m_writeBulkBinary.pred, numOfDoubleRegsToPrint * 2);

        uint32_t registerOffset = m_writeBulkBinary.reg_offset;
        for (const auto& singleValue : m_valuesBinary)
        {
            if (numOfDoubleRegsToPrint == 0)
            {
                break;
            }

            LOG_DEBUG(QMAN, "            value 0x{:x} to register 0x{:x}",
                      singleValue & 0xFFFFFFFF, registerOffset);
            LOG_DEBUG(QMAN, "            value 0x{:x} to register 0x{:x}",
                      (singleValue >> 32) & 0xFFFFFFFF, registerOffset + 4);

            registerOffset += 8;
            numOfDoubleRegsToPrint--;
        }
    }
    if (m_remainderReg != nullptr)
    {
        m_remainderReg->Print();
    }
}

unsigned WriteManyRegisters::GetFirstReg() const
{
    if (m_alignmentReg != nullptr)
    {
        return m_alignmentReg->getRegOffset();
    }
    else if (!m_valuesBinary.empty())
    {
        return m_writeBulkBinary.reg_offset;
    }
    else if (m_remainderReg != nullptr)
    {
        return m_remainderReg->getRegOffset();
    }
    else
    {
        HB_ASSERT(0, "How come I don't have even one register?");
        return 0;
    }
}

unsigned WriteManyRegisters::GetCount()
{
    unsigned numRegs = 0;
    if (m_alignmentReg != nullptr)
    {
        numRegs += 1;
    }
    if (!m_valuesBinary.empty())
    {
        numRegs += m_valuesBinary.size() * 2;
    }
    if (m_remainderReg != nullptr)
    {
        numRegs += 1;
    }
    return numRegs;
}

unsigned WriteManyRegisters::GetBinarySize() const
{
    unsigned size = 0;
    if (m_alignmentReg != nullptr)
    {
        size += m_alignmentReg->GetBinarySize();
    }
    if (!m_valuesBinary.empty())
    {
        size += sizeof(packet_wreg_bulk) + m_valuesBinary.size() * sizeof(uint64_t);
    }
    if (m_remainderReg != nullptr)
    {
        size += m_remainderReg->GetBinarySize();
    }
    return size;
}

uint64_t WriteManyRegisters::writeInstruction(void *whereTo) const
{
    // Impossible to perform ptr arithmetic with void*, so we use char*
    char* whereToChar = reinterpret_cast<char*>(whereTo);
    unsigned bytesWritten = 0;

    if (nullptr != m_alignmentReg)
    {
        bytesWritten += m_alignmentReg->writeInstruction(whereToChar + bytesWritten);
    }
    if (!m_valuesBinary.empty())
    {
        memcpy(whereToChar + bytesWritten, &m_writeBulkBinary, sizeof(m_writeBulkBinary));
        bytesWritten += sizeof(m_writeBulkBinary);

        unsigned valuesDataSize = sizeof(m_valuesBinary[0]) * m_valuesBinary.size();
        memcpy(whereToChar + bytesWritten, m_valuesBinary.data(), valuesDataSize);
        bytesWritten += valuesDataSize;
    }
    if (nullptr != m_remainderReg)
    {
        bytesWritten += m_remainderReg->writeInstruction(whereToChar + bytesWritten);
    }

    return bytesWritten;
}

void WriteManyRegisters::prepareFieldInfos()
{
    BasicFieldInfoSet basicSet;
    AddressFieldInfoSet& addressSet = m_addressContainerInfo.retrieveAddressFieldInfoSet();
    transferAddressToBasic(addressSet, basicSet);
    prepareFieldInfos(basicSet);
    transferBasicToAddress(basicSet, addressSet);

    prepareFieldInfos(m_addressContainerInfo.retrieveBasicFieldInfoSet());
}

void WriteManyRegisters::prepareFieldInfos(BasicFieldInfoSet& basicFieldsInfoSet)
{
    /*
        We update the AFCI according to this table (the first three columns
        determine the rest).
        +-----------+--------+-----------+-----------------+------------------+
        | wreg32 #1 | wrBulk | wreg32 #2 | Advance index 0 | Value to advance |
        +-----------+--------+-----------+-----------------+------------------+
        |         0 |      0 |         1 | irrelevant      |                0 |
        |         0 |      1 |         0 | true            |                2 |
        |         0 |      1 |         1 | true            |                2 |
        |         1 |      0 |         0 | false           |                0 |
        |         1 |      0 |         1 | false           |                1 |
        |         1 |      1 |         0 | false           |                3 |
        |         1 |      1 |         1 | false           |                3 |
        +-----------+--------+-----------+-----------------+------------------+
     */
    BasicFieldInfoSet updatedBasicFieldSet;

    for(auto &singleBasicFieldsInfoPair : basicFieldsInfoSet)
    {
        auto copy = singleBasicFieldsInfoPair;
        // Update according to the offset, and update offset 0 only if m_incZeroOffset
        if (0 != copy.first || m_incZeroOffset)
        {
            copy.first += m_incOffsetValue;
            copy.second->setFieldIndexOffset(copy.second->getFieldIndexOffset() + m_incOffsetValue);
        }

        updatedBasicFieldSet.insert(copy);
    }

    basicFieldsInfoSet.clear();
    basicFieldsInfoSet.insert(updatedBasicFieldSet.begin(), updatedBasicFieldSet.end());
}

// --------------------------------------------------------
// ------------------------ Execute -----------------------
// --------------------------------------------------------

Execute::Execute(HabanaDeviceType type, unsigned deviceID, uint32_t predicate, uint32_t value)
  : WriteRegister(getRegForExecute(type, deviceID), value, predicate),
    m_deviceType(type),
    m_deviceID(deviceID)
{
}

void Execute::Print() const
{
    LOG_DEBUG(QMAN, "      Execute {} {}", getDeviceName(m_deviceType), m_deviceID);
}

// --------------------------------------------------------
// -------------------- ExecuteDmaDesc --------------------
// --------------------------------------------------------

ExecuteDmaDesc::ExecuteDmaDesc(uint32_t         bits,
                               HabanaDeviceType type,
                               unsigned         deviceID,
                               bool             setEngBarrier,
                               uint32_t         predicate,
                               uint8_t          ctxIdHi)
: WriteRegister(getRegForExecute(type, deviceID), bits, predicate),
  m_deviceType(type),
  m_deviceID(deviceID),
  m_ctxIdHi(ctxIdHi)
{
    m_commit._raw = bits;
    if (setEngBarrier)
    {
        m_binary.eng_barrier = 1;
    }
}

ExecuteDmaDesc::~ExecuteDmaDesc()
{
}

void ExecuteDmaDesc::Print() const
{
    LOG_DEBUG(QMAN,
              "      Execute {} {}, signaling={}, memset={}, transpose={} ctxId={}",
              getDeviceName(m_deviceType),
              m_deviceID,
              m_commit.wr_comp_en ? "true" : "false",
              m_commit.mem_set ? "true" : "false",
              m_commit.transpose ? "true" : "false",
              m_commit.ctx_id | ((uint16_t)m_ctxIdHi << 8));
}

// --------------------------------------------------------
// -------------------------- Nop -------------------------
// --------------------------------------------------------

Nop::Nop() : GaudiQueueCommand()
{
    m_binary = {0};

    m_binary.reserved = 0;

    m_binary.opcode = PACKET_NOP;
    m_binary.msg_barrier = 0x1;
    m_binary.reg_barrier = 0x1;
    m_binary.eng_barrier = 0x0;
}

void Nop::Print() const
{
    LOG_DEBUG(QMAN, "      NOP");
}

unsigned Nop::GetBinarySize() const
{
    return sizeof(packet_nop);
}

uint64_t Nop::writeInstruction(void* whereTo) const
{
    memcpy(whereTo, &m_binary, sizeof(m_binary));
    return sizeof(m_binary);
}

void Nop::prepareFieldInfos() {}

// --------------------------------------------------------
// ----------------------- LoadDesc -----------------------
// --------------------------------------------------------

LoadDesc::LoadDesc(void*              desc,
                   unsigned           descSize,
                   unsigned           descOffset,
                   HabanaDeviceType   device,
                   unsigned           deviceID,
                   uint32_t           predicate)
  : WriteManyRegisters(getRegForLoadDesc(device, deviceID) + descOffset,
                       descSize / sizeof(uint32_t),
                       static_cast<uint32_t*>(desc),
                       predicate),
    m_deviceType(device),
    m_deviceID(deviceID)
{
}

LoadDesc::~LoadDesc()
{
}

void LoadDesc::Print() const
{
    if (!LOG_LEVEL_AT_LEAST_DEBUG(QMAN)) return;

    LOG_DEBUG(QMAN, "      {} id:{} loading descriptor:", getDeviceName(m_deviceType), m_deviceID);
    WriteManyRegisters::Print();
}

// --------------------------------------------------------
// ----------------------- Wait --------------------------
// --------------------------------------------------------

Wait::Wait(WaitID id, unsigned int waitCycles, unsigned int incrementValue) : GaudiQueueCommand()
{
    m_binary = {0};

    m_binary.num_cycles_to_wait = waitCycles;
    m_binary.inc_val            = incrementValue;
    m_binary.id                 = id;

    m_binary.opcode             = PACKET_WAIT;
    m_binary.eng_barrier        = 0x0;
    m_binary.reg_barrier        = 0x1;
    m_binary.msg_barrier        = 0x0;
}

Wait::~Wait() {}

void Wait::Print() const
{
    LOG_DEBUG(GC, "      Wait # {}, increment value {}, cycles {}", m_binary.id, m_binary.inc_val, m_binary.num_cycles_to_wait);
}

unsigned Wait::GetBinarySize() const
{
    return sizeof(packet_wait);
}

uint64_t Wait::writeInstruction(void* whereTo) const
{
    memcpy(whereTo, &m_binary, sizeof(m_binary));
    return sizeof(m_binary);
}

void Wait::prepareFieldInfos() {}

// --------------------------------------------------------
// ----------------------- Fence --------------------------
// --------------------------------------------------------

Fence::Fence(WaitID id, unsigned int targetValue) : GaudiQueueCommand(), m_targetValue(targetValue)
{
    //Fence packet
    // only 4 bits for dec val
    unsigned numPkts = div_round_up(targetValue, 0xF);
    unsigned currFenceAggValue = targetValue;
    m_binaries.resize(numPkts);

    for (unsigned i = 0 ; i < numPkts ; ++i)
    {
        m_binaries[i] = {0};

        if (i != numPkts - 1)
        {
            m_binaries[i].dec_val = 0xF;
            m_binaries[i].target_val = currFenceAggValue;
            currFenceAggValue -= 0xF;
        }
        else
        {
            m_binaries[i].dec_val = currFenceAggValue;
            m_binaries[i].target_val = currFenceAggValue;
            currFenceAggValue = 0;
        }

        m_binaries[i].id = id;

        m_binaries[i].pred = 0;
        m_binaries[i].opcode = PACKET_FENCE;
        m_binaries[i].eng_barrier = 0x0;
        m_binaries[i].reg_barrier = 0x1;
        m_binaries[i].msg_barrier = 0x0;
    }
}

Fence::~Fence()
{
}

void Fence::Print() const
{
    if (!LOG_LEVEL_AT_LEAST_DEBUG(QMAN)) return;

    for (const auto& binary : m_binaries)
    {
        LOG_DEBUG(QMAN, "      Fence # {}, target value {}, dec value {}", binary.id, binary.target_val, binary.dec_val);
    }
}

unsigned Fence::GetBinarySize() const
{
    return m_binaries.size() * sizeof(packet_fence);
}

uint64_t Fence::writeInstruction(void *whereTo) const
{
    if (m_binaries.empty())
    {
        return 0;
    }
    memcpy(whereTo, m_binaries.data(), m_binaries.size() * sizeof(m_binaries[0]));
    return m_binaries.size() * sizeof(m_binaries[0]);
}

void Fence::prepareFieldInfos() {}

// --------------------------------------------------------
// ------------------- MonitorSetup -----------------------
// --------------------------------------------------------

MonitorSetup::MonitorSetup(SyncObjectManager::SyncId mon,
                           WaitID                    waitID,
                           HabanaDeviceType          device,
                           unsigned                  deviceID,
                           uint32_t                  value,
                           unsigned                  streamID,
                           uint32_t                  predicate,
                           bool                      incSyncObject)
: GaudiQueueCommand(), m_mon(mon), m_predicate(predicate)
{
    memset(m_msBinaries, 0, sizeof(m_msBinaries));
    memset(m_mlBinaries, 0, sizeof(m_mlBinaries));

    uint32_t monitorValue = value;
    if (incSyncObject)
    {
        packet_msg_short msgShort;
        msgShort.so_upd.mode       = true;
        msgShort.so_upd.te         = false;
        msgShort.so_upd.sync_value = value;
        monitorValue               = msgShort.value;
    }

    if (m_predicate)
    {
        makeMonitorSetupBinaryMsgLong(getCPFenceOffset(device, deviceID, waitID, streamID), monitorValue);
    }
    else
    {
        makeMonitorSetupBinaryMsgShort(getCPFenceOffset(device, deviceID, waitID, streamID), monitorValue);
    }
}

MonitorSetup::MonitorSetup(SyncObjectManager::SyncId mon,
                           SyncObjectManager::SyncId syncId,
                           uint32_t                  value,
                           uint32_t                  predicate,
                           bool                      incSyncObject)
: GaudiQueueCommand(), m_mon(mon), m_predicate(predicate)
{
    memset(m_msBinaries, 0, sizeof(m_msBinaries));
    memset(m_mlBinaries, 0, sizeof(m_mlBinaries));

    uint32_t monitorValue = value;
    if (incSyncObject)
    {
        packet_msg_short msgShort{};
        msgShort.so_upd.mode       = true;
        msgShort.so_upd.te         = false;
        msgShort.so_upd.sync_value = value;
        monitorValue               = msgShort.value;
    }

    if (m_predicate)
    {
        makeMonitorSetupBinaryMsgLong(getSyncObjectAddress(syncId), monitorValue);
    }
    else
    {
        makeMonitorSetupBinaryMsgShort(getSyncObjectAddress(syncId), monitorValue);
    }
}

void MonitorSetup::makeMonitorSetupBinaryMsgShort(uint64_t address, uint32_t value)
{
    ptrToInt p;
    p.u64 = address;

    for(unsigned i = 0; i < m_numOfPackets; i++)
    {
        m_msBinaries[i] = {0};

        m_msBinaries[i].weakly_ordered = 0; // Default value
        m_msBinaries[i].no_snoop = 0; // Default value
        m_msBinaries[i].op = 0;
        m_msBinaries[i].base = 0;
        m_msBinaries[i].opcode = PACKET_MSG_SHORT;

        m_msBinaries[i].eng_barrier = 0;
        m_msBinaries[i].reg_barrier = 1;
        m_msBinaries[i].msg_barrier = 0;
    }

    unsigned monitorBlockBase = getMonPayloadAddress(0, MON_PAYLOAD_ADDR_L);

    // First config packet: low address of the sync payload
    m_msBinaries[0].msg_addr_offset = getMonPayloadAddress(m_mon, MON_PAYLOAD_ADDR_L) - monitorBlockBase;
    m_msBinaries[0].value = p.u32[0];

    // Second config packet: high address of the sync payload
    m_msBinaries[1].msg_addr_offset = getMonPayloadAddress(m_mon, MON_PAYLOAD_ADDR_H) - monitorBlockBase;
    m_msBinaries[1].value = p.u32[1];

    // Third config packet: the payload data, i.e. what to write when the sync triggers
    m_msBinaries[2].msg_addr_offset = getMonPayloadAddress(m_mon, MON_PAYLOAD_DATA) - monitorBlockBase;
    m_msBinaries[2].value = value;
}

void MonitorSetup::makeMonitorSetupBinaryMsgLong(uint64_t address, uint32_t value)
{
    ptrToInt p;
    p.u64 = address;

    for(unsigned i = 0; i < m_numOfPackets; i++)
    {
        m_mlBinaries[i] = {0};

        m_mlBinaries[i].opcode      = PACKET_MSG_LONG;
        m_mlBinaries[i].msg_barrier = 0;
        m_mlBinaries[i].reg_barrier = 1;
        m_mlBinaries[i].eng_barrier = 0;
        m_mlBinaries[i].pred        = m_predicate;
        m_mlBinaries[i].op          = 0;
    }

    // First config packet: low address of the sync payload
    m_mlBinaries[0].addr = getMonPayloadAddress(m_mon, MON_PAYLOAD_ADDR_L);
    m_mlBinaries[0].value = p.u32[0];

    // Second config packet: high address of the sync payload
    m_mlBinaries[1].addr = getMonPayloadAddress(m_mon, MON_PAYLOAD_ADDR_H);
    m_mlBinaries[1].value = p.u32[1];

    // Third config packet: the payload data, i.e. what to write when the sync triggers
    m_mlBinaries[2].addr = getMonPayloadAddress(m_mon, MON_PAYLOAD_DATA);
    m_mlBinaries[2].value = value;
}

void MonitorSetup::Print() const
{
    if (!LOG_LEVEL_AT_LEAST_DEBUG(QMAN)) return;

    ptrToInt p;
    p.u32[1] = m_predicate ? m_mlBinaries[1].value : m_msBinaries[1].value;
    p.u32[0] = m_predicate ? m_mlBinaries[0].value : m_msBinaries[0].value;

    LOG_DEBUG(QMAN,
              "      Setup monitor {} to write 0x{:x} to 0x{:x} with predicate {}",
              m_mon,
              m_predicate ? m_mlBinaries[2].value : m_msBinaries[2].value,
              p.u64,
              m_predicate);
}

unsigned MonitorSetup::GetBinarySize() const
{
    return m_numOfPackets * (m_predicate ? sizeof(packet_msg_long) : sizeof(packet_msg_short));
}

uint64_t MonitorSetup::writeInstruction(void *whereTo) const
{
    if (m_predicate)
    {
        memcpy(whereTo, m_mlBinaries,  sizeof(m_mlBinaries));
        return sizeof(m_mlBinaries);
    }
    else
    {
        memcpy(whereTo, m_msBinaries,  sizeof(m_msBinaries));
        return sizeof(m_msBinaries);
    }
}

void MonitorSetup::prepareFieldInfos()
{
    if (0 == m_addressContainerInfo.size())
    {
        return;
    }

    HB_ASSERT(2 == m_addressContainerInfo.size() || 1 == m_addressContainerInfo.size(),
              "Unexpected number of patching points for MonitorSetup command");
    BasicFieldInfoSet basicSet;
    AddressFieldInfoSet& addressSet = m_addressContainerInfo.retrieveAddressFieldInfoSet();
    transferAddressToBasic(addressSet, basicSet);
    prepareFieldInfos(basicSet);
    transferBasicToAddress(basicSet, addressSet);

    prepareFieldInfos(m_addressContainerInfo.retrieveBasicFieldInfoSet());
}

void MonitorSetup::prepareFieldInfos(BasicFieldInfoSet& basicFieldInfoSet)
{
    BasicFieldInfoSet updatedBasicFieldSet;
    for(auto &singleBasicFieldsInfoPair : basicFieldInfoSet)
    {
        auto copy = singleBasicFieldsInfoPair;
        if (copy.second->getFieldIndexOffset() == 1)
        {
            if (m_predicate)
            {
                // Accounting for the msg_long header and addr
                copy.first = 4;
                copy.second->setFieldIndexOffset(4);
            }
            else
            {
                // Accounting for the msg_short header
                copy.first = 2;
                copy.second->setFieldIndexOffset(2);
            }
        }
        else if (copy.second->getFieldIndexOffset() != 0)
        {
            HB_ASSERT(false, "Unexpected field index offset in MonitorSetup command");
        }

        updatedBasicFieldSet.insert(copy);
    }

    basicFieldInfoSet.clear();
    basicFieldInfoSet.insert(updatedBasicFieldSet.begin(), updatedBasicFieldSet.end());
}

// --------------------------------------------------------
// --------------------- MonitorArm -----------------------
// --------------------------------------------------------

MonitorArm::MonitorArm(SyncObjectManager::SyncId   syncObj,
                       SyncObjectManager::SyncId   mon,
                       MonitorOp                   operation,
                       unsigned                    syncValue,
                       Settable<uint8_t>           mask)
        : GaudiQueueCommand()
{
    unsigned monitorBlockBase = offsetof(block_sob_objs, mon_pay_addrl);

    m_mon       = mon;
    m_syncValue = syncValue;
    m_syncObj   = syncObj;
    m_operation = operation;
    m_mask      = mask;

    uint8_t syncMask;
    unsigned syncGroupId;
    if (mask.is_set())
    {
        syncMask     = ~(mask.value());
        syncGroupId  = syncObj;
    }
    else
    {
        syncMask     = ~(static_cast<uint8_t>(0x1U << (syncObj % 8)));
        syncGroupId  = syncObj / 8;
    }

    m_binary                                = {0};
    m_binary.mon_arm_register.sync_group_id = syncGroupId;
    m_binary.mon_arm_register.mask          = syncMask;
    m_binary.mon_arm_register.mode          = operation;
    m_binary.mon_arm_register.sync_value    = syncValue;

    m_binary.msg_addr_offset = varoffsetof(block_sob_objs, mon_arm[mon]) - monitorBlockBase;
    m_binary.weakly_ordered  = 0;  // Default value
    m_binary.no_snoop        = 0;  // Default value
    m_binary.op              = 0;
    m_binary.base            = 0;
    m_binary.opcode          = PACKET_MSG_SHORT;
    m_binary.eng_barrier     = 0;
    m_binary.reg_barrier     = 1;
    m_binary.msg_barrier     = 0;
}

void MonitorArm::Print() const
{
    LOG_DEBUG(QMAN, "      Arm monitor {} to wait {} for semaphore {} [is_mask: {}, mask: {:b}] to reach {}",
              m_mon,
              (m_operation == MONITOR_SO_OP_GREQ ? "(>=)" : "(==)"),
              (m_mask.is_set() ? 8 * m_syncObj : m_syncObj),
              m_mask.is_set(),
              (m_mask.is_set() ? m_mask.value() : 0),
              m_syncValue);

}

unsigned MonitorArm::GetBinarySize() const
{
    return sizeof(packet_msg_short);
}

uint64_t MonitorArm::writeInstruction(void *whereTo) const
{
    memcpy(whereTo, &m_binary, sizeof(m_binary));
    return sizeof(m_binary);
}

void MonitorArm::prepareFieldInfos() {}

// --------------------------------------------------------
// ------------------ SignalSemaphore ---------------------
// --------------------------------------------------------

SignalSemaphore::SignalSemaphore(SyncObjectManager::SyncId which, int16_t syncValue, int operation, int barriers)
: GaudiQueueCommand()
{
    m_syncValue = syncValue;
    m_operation = operation;

    bool mode = (operation == SYNC_OP_ADD);

    m_binary                   = {0};
    m_binary.so_upd.sync_value = syncValue;
    m_binary.so_upd.te         = 0;
    m_binary.so_upd.mode       = mode;

    m_binary.msg_addr_offset = which * sizeof(uint32_t);
    m_binary.weakly_ordered  = 0;  // Default value
    m_binary.no_snoop        = 0;  // Default value
    m_binary.op              = 0;
    m_binary.base            = 1;
    m_binary.opcode          = PACKET_MSG_SHORT;
    m_binary.eng_barrier     = (barriers & ENGINE_BARRIER) ? 1 : 0;
    m_binary.reg_barrier     = (barriers & REGISTER_BARRIER) ? 1 : 0;
    m_binary.msg_barrier     = (barriers & MESSAGE_BARRIER) ? 1 : 0;
}

void SignalSemaphore::Print() const
{
    unsigned so = (m_binary.msg_addr_offset >> 2);
    LOG_DEBUG(QMAN,
              "      Signal semaphore {} [0x{:x}] {} {}",
              so,
              getSyncObjectAddress(so),
              (m_operation ? " increment by " : " set to "),
              m_syncValue);
}

unsigned SignalSemaphore::GetBinarySize() const
{
    return sizeof(packet_msg_short);
}

uint64_t SignalSemaphore::writeInstruction(void *whereTo) const
{
    memcpy(whereTo, &m_binary, sizeof(m_binary));
    return sizeof(m_binary);
}

// --------------------------------------------------------
// ----------------- WaitForSemaphore ---------------------
// --------------------------------------------------------

WaitForSemaphore::WaitForSemaphore(SyncObjectManager::SyncId syncObj,
                                   SyncObjectManager::SyncId mon,
                                   MonitorOp                 operation,
                                   unsigned                  syncValue,
                                   Settable<uint8_t>         mask,
                                   WaitID                    waitID,
                                   unsigned int              fenceValue)
: m_monitorArm(syncObj, mon, operation, syncValue, mask),
  m_fence(waitID, fenceValue),
  m_mon(mon),
  m_syncObj(syncObj),
  m_operation(operation),
  m_syncValue(syncValue),
  m_mask(mask),
  m_waitID(waitID)
{
}

WaitForSemaphore::~WaitForSemaphore()
{
}

void WaitForSemaphore::Print() const
{
    LOG_DEBUG(QMAN,
              "      Arm monitor {} to wait {} for semaphore {} [is_mask: {}, mask: {:b}] to reach {} and wait for "
              "fence # {} to reach {}",
              m_mon,
              (m_operation == MONITOR_SO_OP_GREQ ? "(>=)" : "(==)"),
              (m_mask.is_set() ? 8 * m_syncObj : m_syncObj),
              m_mask.is_set(),
              (m_mask.is_set() ? m_mask.value() : 0),
              m_syncValue,
              m_waitID,
              m_fence.m_targetValue);
}
unsigned WaitForSemaphore::GetBinarySize() const
{
    return m_fence.GetBinarySize() + m_monitorArm.GetBinarySize();
}

uint64_t WaitForSemaphore::writeInstruction(void *whereTo) const
{
    unsigned bytesWritten = 0;
    char* whereToCharPtr = reinterpret_cast<char*>(whereTo);

    bytesWritten += m_monitorArm.writeInstruction(whereToCharPtr + bytesWritten);
    bytesWritten += m_fence.writeInstruction(whereToCharPtr + bytesWritten);

    return bytesWritten;
}

void WaitForSemaphore::prepareFieldInfos() {}

// --------------------------------------------------------
// ---------------------- Suspend -------------------------
// --------------------------------------------------------

Suspend::Suspend(WaitID waitID,
                 unsigned int waitCycles,
                 unsigned int incrementValue)
    : m_wait(waitID, waitCycles, incrementValue),
      m_fence(waitID, incrementValue),
      m_waitID(waitID),
      m_incrementValue(incrementValue),
      m_waitCycles(waitCycles)
{}

Suspend::~Suspend() {}

void Suspend::Print() const
{
    LOG_DEBUG(GC,
              "      Add suspension using wait and fence with ID {}, increment value {} and wait cycles {}",
              m_waitID, m_incrementValue, m_waitCycles);
}

unsigned Suspend::GetBinarySize() const
{
    return m_fence.GetBinarySize() + m_wait.GetBinarySize();
}

uint64_t Suspend::writeInstruction(void* whereTo) const
{
    unsigned bytesWritten = 0;
    char* whereToCharPtr = reinterpret_cast<char*>(whereTo);

    bytesWritten += m_wait.writeInstruction(whereToCharPtr + bytesWritten);
    bytesWritten += m_fence.writeInstruction(whereToCharPtr + bytesWritten);

    return bytesWritten;
}

void Suspend::prepareFieldInfos() {}

// --------------------------------------------------------
// ----------------- InvalidateTPCCaches ------------------
// --------------------------------------------------------

InvalidateTPCCaches::InvalidateTPCCaches(uint32_t predicate)
  : WriteRegister(offsetof(block_tpc, tpc_cmd), 0x0, predicate)
{
    tpc::reg_tpc_cmd cmd;
    cmd._raw              = 0;
    cmd.icache_invalidate = 1;
    cmd.dcache_invalidate = 1;
    cmd.lcache_invalidate = 1;
    cmd.tcache_invalidate = 1;

    m_binary.eng_barrier = 1;
    m_binary.value = cmd._raw;
}

InvalidateTPCCaches::~InvalidateTPCCaches()
{

}

void InvalidateTPCCaches::Print() const
{
    LOG_DEBUG(QMAN, "      Invalidate TPC caches");
}

// --------------------------------------------------------
// ------------------ UploadKernelsAddr -------------------
// --------------------------------------------------------

UploadKernelsAddr::UploadKernelsAddr(uint32_t uploadToLow, uint32_t uploadToHigh, uint32_t predicate)
  : GaudiQueueCommand(),
    m_highAddress(uploadToHigh),
    m_lowAddress(uploadToLow & CompilationHalReader::getHalReader()->getPrefetchAlignmentMask()), // Prefetch address must aligned to 13 bit
    m_predicate(predicate)
{
    // Will be patching as BasicFieldsContainerInfo in the following order: (low, high)
    for (auto& binary : m_binaries)
    {
        binary = {0};
        binary.pred = m_predicate;
        binary.opcode = PACKET_WREG_32;
        binary.reg_barrier = 1;
        binary.msg_barrier = 0;
    }

    m_binaries[0].reg_offset = offsetof(block_tpc, icache_base_adderess_low);
    m_binaries[0].value = m_lowAddress;
    m_binaries[0].eng_barrier = 0;

    m_binaries[1].reg_offset = offsetof(block_tpc, icache_base_adderess_high);
    m_binaries[1].value = m_highAddress;
    m_binaries[1].eng_barrier = 0;

    tpc::reg_tpc_cmd cmd;
    cmd._raw                 = 0;
    cmd.icache_prefetch_64kb = 1;
    m_binaries[2].reg_offset = offsetof(block_tpc, tpc_cmd);
    m_binaries[2].value = cmd._raw;
    m_binaries[2].eng_barrier = 1;

    ptrToInt kernelBaseAddr;
    kernelBaseAddr.u32[0]          = uploadToLow & CompilationHalReader::getHalReader()->getPrefetchAlignmentMask();
    kernelBaseAddr.u32[1]          = uploadToHigh;
    uint64_t                 memId = getMemoryIDFromVirtualAddress(kernelBaseAddr.u64);
    BasicFieldsContainerInfo addrContainer;

    addrContainer.addAddressEngineFieldInfo(nullptr,
                                            getMemorySectionNameForMemoryID(memId),
                                            memId,
                                            (uint64_t)&kernelBaseAddr.u32[0],
                                            (uint64_t)&kernelBaseAddr.u32[1],
                                            (uint64_t)&kernelBaseAddr.u32[0]);
    SetContainerInfo(addrContainer);
}

UploadKernelsAddr::~UploadKernelsAddr()
{
}

void UploadKernelsAddr::Print() const
{
    if (!LOG_LEVEL_AT_LEAST_DEBUG(QMAN)) return;

    ptrToInt p;
    p.u32[0] = m_lowAddress;
    p.u32[1] = m_highAddress;
    LOG_DEBUG(QMAN, "      Pre-fetch icache at base adderess 0x{:x} ", p.u64);
}
unsigned UploadKernelsAddr::GetBinarySize() const
{
    return 3 * sizeof(packet_wreg32); // This commands translates to 3 wreg32
}

uint64_t UploadKernelsAddr::writeInstruction(void *whereTo) const
{
    memcpy(whereTo, m_binaries, sizeof(m_binaries));
    return sizeof(m_binaries);
}

void UploadKernelsAddr::prepareFieldInfos()
{
    if (0 == m_addressContainerInfo.size())
    {
        return;
    }

    HB_ASSERT(2 == m_addressContainerInfo.size(), "Unexpected number of patching points for UploadKernelAddr command");
    BasicFieldInfoSet basicSet;
    AddressFieldInfoSet& addressSet = m_addressContainerInfo.retrieveAddressFieldInfoSet();
    transferAddressToBasic(addressSet, basicSet);
    prepareFieldInfos(basicSet);
    transferBasicToAddress(basicSet, addressSet);

    prepareFieldInfos(m_addressContainerInfo.retrieveBasicFieldInfoSet());
}

void UploadKernelsAddr::prepareFieldInfos(BasicFieldInfoSet& basicFieldsInfoSet)
{
    BasicFieldInfoSet updatedBasicFieldSet;
    for(auto &singleBasicFieldsInfoPair : basicFieldsInfoSet)
    {
        auto copy = singleBasicFieldsInfoPair;
        if (copy.second->getFieldIndexOffset() == 1)
        {
            // Accounting for msg_short header
            copy.first = 2;
            copy.second->setFieldIndexOffset(2);
        }
        else if (copy.second->getFieldIndexOffset() != 0)
        {
            HB_ASSERT(false, "UNexpected field index offset in UploadKernelAddr command");
        }

        updatedBasicFieldSet.insert(copy);
    }

    basicFieldsInfoSet.clear();
    basicFieldsInfoSet.insert(updatedBasicFieldSet.begin(), updatedBasicFieldSet.end());
}

// --------------------------------------------------------
// ----------------------- MsgLong ------------------------
// --------------------------------------------------------

MsgLong::MsgLong()
  : GaudiQueueCommand()
{
    m_binary = {0};
}

unsigned MsgLong::GetBinarySize() const
{
    return sizeof(packet_msg_long);
}

uint64_t MsgLong::writeInstruction(void *whereTo) const
{
    memcpy(whereTo, &m_binary, sizeof(m_binary));
    return sizeof(m_binary);
}

void MsgLong::prepareFieldInfos()
{
    HB_ASSERT(2 == m_addressContainerInfo.size(), "Unexpected number of patching points for CpDma command");
    prepareFieldInfosTwoDwordsHeader(m_addressContainerInfo.retrieveBasicFieldInfoSet(),
                                     m_addressContainerInfo.retrieveAddressFieldInfoSet());
}

// --------------------------------------------------------
// -------------------- ResetSyncObject -------------------
// --------------------------------------------------------

ResetSyncObject::ResetSyncObject(unsigned syncID, bool logLevelTrace, uint32_t predicate)
  : m_syncID(syncID),
    m_logLevelTrace(logLevelTrace)
{
    m_binary.opcode         = PACKET_MSG_LONG;
    m_binary.value          = 0;
    m_binary.pred           = predicate;
    m_binary.op             = 0;
    m_binary.msg_barrier    = 0;
    m_binary.reg_barrier    = 1;
    m_binary.eng_barrier    = 0;
    m_binary.addr           = getSyncObjectAddress(m_syncID);
}

ResetSyncObject::~ResetSyncObject()
{
}

void ResetSyncObject::Print() const
{
    if (!LOG_LEVEL_AT_LEAST_DEBUG(QMAN)) return;

    if (m_logLevelTrace)
    {
        LOG_TRACE(QMAN, "      Reset sync object #{} at address 0x{:x}", m_syncID, m_binary.addr);
    }
    else
    {
        LOG_DEBUG(QMAN, "      Reset sync object #{} at address 0x{:x}", m_syncID, m_binary.addr);
    }
}

// --------------------------------------------------------
// -------------------- IncrementFence --------------------
// --------------------------------------------------------

IncrementFence::IncrementFence(HabanaDeviceType deviceType, unsigned deviceID, WaitID waitID, unsigned streamID, uint32_t predicate)
{
    m_binary.opcode         = PACKET_MSG_LONG;
    m_binary.value          = 1;
    m_binary.pred           = predicate;
    m_binary.op             = 0;
    m_binary.msg_barrier    = 0;
    m_binary.reg_barrier    = 1;
    m_binary.eng_barrier    = 0;
    m_binary.addr           = getCPFenceOffset(deviceType, deviceID, waitID, streamID);
}

IncrementFence::~IncrementFence()
{
}

void IncrementFence::Print() const
{
    LOG_DEBUG(QMAN, "      Increment CP fence counter at address 0x{:x}", m_binary.addr);
}

// --------------------------------------------------------
// --------------------- LoadPredicates -------------------
// --------------------------------------------------------

LoadPredicates::LoadPredicates(deviceAddrOffset src, uint32_t predicate)
  : GaudiQueueCommand()
{
    m_binary = {0};

    m_binary.opcode      = PACKET_LOAD_AND_EXE;
    m_binary.pred        = predicate;
    m_binary.msg_barrier = 0x1;
    m_binary.reg_barrier = 0x1;
    m_binary.eng_barrier = 0x0;
    m_binary.load        = 0x1; // loading operation
    m_binary.dst         = 0x0; // loading predicates and not scalars
    m_binary.exe         = 0x0;
    m_binary.etype       = 0x0; // don't care (irrelevant for loading predicates)
    m_binary.src_addr    = src;
}

unsigned LoadPredicates::GetBinarySize() const
{
    return sizeof(packet_load_and_exe);
}

uint64_t LoadPredicates::writeInstruction(void* whereTo) const
{
    memcpy(whereTo, &m_binary, sizeof(m_binary));
    return sizeof(m_binary);
}

void LoadPredicates::prepareFieldInfos()
{
    HB_ASSERT(2 >= m_addressContainerInfo.size(), "Unexpected number of patching points for LoadPredicates command");
    prepareFieldInfosTwoDwordsHeader(m_addressContainerInfo.retrieveBasicFieldInfoSet(),
                                     m_addressContainerInfo.retrieveAddressFieldInfoSet());
}

void LoadPredicates::Print() const
{
    LOG_DEBUG(QMAN, "      LoadPredicates from address 0x{:x}", m_binary.src_addr);
}

uint64_t LoadPredicates::getSrcAddrForTesting() const
{
    return (uint64_t)m_binary.src_addr;
}


// --------------------------------------------------------
// ------------------ SignalSemaphoreWithPredicate --------
// --------------------------------------------------------

SignalSemaphoreWithPredicate::SignalSemaphoreWithPredicate(
    SyncObjectManager::SyncId which, int16_t syncValue, uint32_t pred, int operation, int barriers)
: MsgLong(), m_syncId(which)
{
    m_syncValue = syncValue;
    m_operation = operation;

    bool mode = (operation == SYNC_OP_ADD);

    // using short to set the bitfields of value
    packet_msg_short binary {};
    binary.value             = 0;
    binary.so_upd.sync_value = syncValue;
    binary.so_upd.te         = 0;
    binary.so_upd.mode       = mode;
    m_binary.opcode          = PACKET_MSG_LONG;
    m_binary.value           = binary.value;
    m_binary.addr            = getSyncObjectAddress(which);
    m_binary.op              = 0;
    m_binary.weakly_ordered  = 0;  // Default value
    m_binary.pred            = pred;
    m_binary.no_snoop        = 0;  // Default value
    m_binary.eng_barrier     = (barriers & ENGINE_BARRIER) ? 1 : 0;
    m_binary.reg_barrier     = (barriers & REGISTER_BARRIER) ? 1 : 0;
    m_binary.msg_barrier     = (barriers & MESSAGE_BARRIER) ? 1 : 0;
}

void SignalSemaphoreWithPredicate::Print() const
{
    LOG_DEBUG(QMAN,
              "      Signal semaphore {} [0x{:x}] {} {}",
              m_syncId,
              m_binary.addr,
              (m_operation ? " increment by " : " set to "),
              m_syncValue);
}


CompositeQueueCommand::CompositeQueueCommand(std::vector<std::shared_ptr<GaudiQueueCommand>> commands)
: m_commands(std::move(commands))
{
}

void CompositeQueueCommand::Print() const
{
    for (const auto& command : m_commands)
    {
        command->Print();
    }
}
unsigned CompositeQueueCommand::GetBinarySize() const
{
    return std::accumulate(
        m_commands.begin(),
        m_commands.end(),
        0,
        [](unsigned i, const std::shared_ptr<GaudiQueueCommand>& command) { return i + command->GetBinarySize(); });
}

uint64_t CompositeQueueCommand::writeInstruction(void* whereTo) const
{
    uint64_t totalWritten = 0;
    for (auto& command : m_commands)
    {
        totalWritten += command->writeInstruction((char*) whereTo + totalWritten);
    }
    return totalWritten;
}

DynamicExecute::DynamicExecute(std::vector<std::shared_ptr<GaudiQueueCommand>> commands, BypassType enableBypass)
: CompositeQueueCommand(commands), m_enableBypass(enableBypass) {}


void DynamicExecute::prepareFieldInfos()
{
    auto& data = m_addressContainerInfo.retrieveBasicFieldInfoSet();
    if (data.empty()) return;

    auto& fieldInfo = data.begin()->second;
    fieldInfo->setFieldIndexOffset(1); // pred starts with one

    const auto& dynamicFieldInfo = std::dynamic_pointer_cast<DynamicShapeFieldInfo>(fieldInfo);
    HB_ASSERT(dynamicFieldInfo != nullptr, "invalid field info");

    switch(fieldInfo->getType())
    {
        case(FieldType::FIELD_DYNAMIC_EXECUTE_NO_SIGNAL):
        {
            prepareFieldInfoNoSignal(dynamicFieldInfo);
            break;
        }
        case(FieldType::FIELD_DYNAMIC_EXECUTE_WITH_SIGNAL):
        {
            prepareFieldInfoSignalOnce(dynamicFieldInfo);
            break;
        }
        case(FieldType::FIELD_DYNAMIC_EXECUTE_MME):
        {
            prepareFieldInfoSignalMME(dynamicFieldInfo);
            break;
        }
        default:
        {
            HB_ASSERT(false, "invalid field info");
        }
    }

}

void DynamicExecute::prepareFieldInfoNoSignal(const DynamicShapeFieldInfoSharedPtr& fieldInfo)
{
    dynamic_execution_sm_params_t metadata;
    initMetaData(metadata, 1);

    struct DataLayout {
        uint32_t executePred : 5;
        uint32_t notPred : 27;
    };
    auto layoutedData = reinterpret_cast<DataLayout*>(metadata.commands);
    layoutedData->executePred = 31;

    updateFieldInfo(fieldInfo, metadata);
}

void DynamicExecute::prepareFieldInfoSignalOnce(const DynamicShapeFieldInfoSharedPtr& fieldInfo)
{
    dynamic_execution_sm_params_t metadata;
    initMetaData(metadata, 3);

    struct DataLayout {
        uint32_t executePred : 5;
        uint32_t notPred     : 27;   // do not use uint64_t because it would make DataLayout overaligned
        uint32_t notPred2    : 32;
        uint32_t signalPred  : 5;
        uint32_t notPred3    : 27;
    };
    auto layoutedData = reinterpret_cast<DataLayout*>(metadata.commands);
    auto currentExecutePred = layoutedData->executePred;
    layoutedData->executePred = layoutedData->signalPred;
    layoutedData->signalPred = currentExecutePred;

    updateFieldInfo(fieldInfo, metadata);
}

void DynamicExecute::prepareFieldInfoSignalMME(const DynamicShapeFieldInfoSharedPtr& fieldInfo)
{
    dynamic_execution_sm_params_t metadata;
    initMetaData(metadata, 7);

    struct DataLayout {
        uint32_t executePred      : 5;
        uint32_t exeNotPred       : 27; // 1
        uint32_t signal1NotPred0;       // 2
        uint32_t signal1Pred      : 5;
        uint32_t signal1NotPred1  : 27; // 3
        uint32_t signal1NotPred2;       // 4
        uint32_t signal1NotPred3;       // 5
        uint32_t signal2NotPred0;       // 6
        uint32_t signal2Pred      : 5;
        uint32_t signal2NotPred1  : 27; // 7
    };

    auto layoutedData = reinterpret_cast<DataLayout*>(metadata.commands);
    auto currentExecutePred = layoutedData->executePred;
    layoutedData->executePred = layoutedData->signal1Pred;
    layoutedData->signal1Pred = currentExecutePred;
    layoutedData->signal2Pred = currentExecutePred;

    updateFieldInfo(fieldInfo, metadata);
}

void DynamicExecute::initMetaData(dynamic_execution_sm_params_t &metadata, size_t cmdLen)
{
    memset(&metadata, 0, sizeof(metadata));
    metadata.cmd_len = cmdLen;
    metadata.should_bypass = m_enableBypass;

    std::vector<uint8_t> instr(GetBinarySize());
    writeInstruction(instr.data());
    // The first dword isn't part of the patching data.
    memcpy(metadata.commands,
           instr.data() + sizeof(uint32_t),
           metadata.cmd_len * sizeof(uint32_t));
}

void DynamicExecute::updateFieldInfo(const DynamicShapeFieldInfoSharedPtr& fieldInfo, dynamic_execution_sm_params_t& metadata)
{
    auto projections = fieldInfo->getOrigin()->getDynamicShapeProjectionsTensors();
    unsigned i = 0;
    for (auto &projection: projections)
    {
        metadata.projections[i].is_output = projection.isOutput;
        metadata.projections[i].tensor_dim = projection.tensorDim;
        metadata.projections[i].tensor_idx = projection.tensorIdx;
        i++;
    }

    metadata.num_projections = i;

    std::vector<uint8_t> serializedMetadata(sizeof(metadata));
    memcpy(serializedMetadata.data(), &metadata, sizeof(metadata));
    fieldInfo->setMetadata(std::move(serializedMetadata));
    fieldInfo->setSize(metadata.cmd_len);
}

// This composite class assumes there are exactly 2 commands: the first is the monitor setup and it should
// contain patchpoint, and the second is the monitor arm command and is should NOT contain patchpoint
SetupAndArm::SetupAndArm(std::vector<std::shared_ptr<GaudiQueueCommand>> commands) : CompositeQueueCommand(commands)
{
    HB_ASSERT(commands.size() == 2, "expecting exactly 2 commands (setup and arm)");
    HB_ASSERT(!commands[0]->getBasicFieldsContainerInfo().empty(), "Setup should have patchpoint");
    HB_ASSERT(commands[1]->getBasicFieldsContainerInfo().empty(), "Arm should not have patchpoint");
}

void SetupAndArm::prepareFieldInfos()
{
    // calling just the setup command prepare field info
    m_commands[0]->prepareFieldInfos();
}

const BasicFieldsContainerInfo& SetupAndArm::getBasicFieldsContainerInfo() const
{
    // returning just the setup command field container info
    return m_commands[0]->getBasicFieldsContainerInfo();
}

} // namespace gaudi
