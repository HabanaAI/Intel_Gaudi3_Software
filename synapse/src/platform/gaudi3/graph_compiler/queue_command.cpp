#include "queue_command.h"

#include "block_data.h"
#include "gaudi3/asic_reg_structs/tpc_regs.h"
#include "habana_global_conf.h"
#include "hal_reader/gaudi3/hal_reader.h"
#include "types.h"
#include "utils.h"

namespace gaudi3
{
void setSendSyncEvents(uint32_t& raw)
{
    raw |= (1 << SEND_SYNC_BIT);
}
// --------------------------------------------------------
// -------------------------- Nop -------------------------
// --------------------------------------------------------

Nop::Nop() : NopCommon()
{
    m_binary             = {0};
    m_binary.reserved    = 0;
    m_binary.opcode      = PACKET_NOP;
    m_binary.msg_barrier = 0x1;
    m_binary.eng_barrier = 0x0;
    m_binary.swtc        = 0x0;
}

uint8_t Nop::getSwtc() const
{
    return m_binary.swtc;
}

void Nop::setSwtc(uint8_t val)
{
    m_binary.swtc = val;
}

uint32_t Nop::getNopPacketSize() const
{
    return sizeof(packet_nop);
}

uint32_t* Nop::getNopPacketAddr() const
{
    return (uint32_t*)&m_binary;
}

// --------------------------------------------------------
// ----------------------- LoadDesc -----------------------
// --------------------------------------------------------

LoadDesc::LoadDesc(void*            desc,
                   unsigned         descSize,
                   unsigned         descOffset,
                   HabanaDeviceType device,
                   unsigned         deviceID,
                   uint32_t         predicate)
: WriteManyRegisters(getRegForLoadDesc(device, deviceID) + descOffset,
                     descSize / sizeof(uint32_t),
                     static_cast<uint32_t*>(desc),
                     predicate),
  m_deviceType(device),
  m_deviceID(deviceID)
{
}

LoadDesc::~LoadDesc() {}

void LoadDesc::Print() const
{
    if (!LOG_LEVEL_AT_LEAST_DEBUG(QMAN)) return;

    LOG_DEBUG(QMAN, "      {} id:{} loading descriptor:", getDeviceName(m_deviceType), m_deviceID);
    WriteManyRegisters::Print();
}
// --------------------------------------------------------
// ----------------------- Wait --------------------------
// --------------------------------------------------------

Wait::Wait(WaitID id, unsigned int waitCycles, unsigned int incrementValue) : WaitCommon()
{
    m_binary = {0};

    m_binary.num_cycles_to_wait = waitCycles;
    m_binary.inc_val            = incrementValue;
    m_binary.id                 = id;
    m_binary.opcode             = PACKET_WAIT;
    m_binary.eng_barrier        = 0x0;
    m_binary.msg_barrier        = 0x0;
}

uint8_t Wait::getId() const
{
    return m_binary.id;
}

uint8_t Wait::getIncVal() const
{
    return m_binary.inc_val;
}
uint32_t Wait::getNumCyclesToWait() const
{
    return m_binary.num_cycles_to_wait;
}
uint32_t Wait::getWaitPacketSize() const
{
    return sizeof(packet_wait);
}

uint32_t* Wait::getWaitPacketAddr() const
{
    return (uint32_t*)&m_binary;
}

// --------------------------------------------------------
// -------------------- WriteRegister ---------------------
// --------------------------------------------------------

WriteRegister::WriteRegister(unsigned regOffset, unsigned value, uint32_t predicate) : WriteRegisterCommon()
{
    HB_ASSERT(fitsInBits(regOffset, 16), "WriteRegister offsett invalid");
    m_binary        = {0};
    m_binary.opcode = PACKET_WREG_32;
    fillPackets(regOffset, value, predicate);
}

WriteRegister::WriteRegister(unsigned regOffset, unsigned value, uint64_t commandId, uint32_t predicate)
: WriteRegisterCommon(commandId)
{
    HB_ASSERT(fitsInBits(regOffset, 16), "WriteRegister offsett invalid");
    m_binary             = {0};
    m_binary.opcode      = PACKET_WREG_32;
    fillPackets(regOffset, value, predicate);
}

uint32_t WriteRegister::getValue() const
{
    return m_binary.value;
}

unsigned WriteRegister::getPred() const
{
    return m_binary.pred;
}

unsigned WriteRegister::getReg() const
{
    return m_binary.reg;
}

unsigned WriteRegister::getRegOffset() const
{
    return m_binary.reg_offset;
}

uint8_t WriteRegister::getSwtc() const
{
    return m_binary.swtc;
}

uint32_t WriteRegister::getWriteRegisterPacketSize() const
{
    return sizeof(packet_wreg32);
}

uint32_t* WriteRegister::getWriteRegisterPacketAddr() const
{
    return (uint32_t*)&m_binary;
}

void WriteRegister::setValue(uint32_t value)
{
    m_binary.value = value;
}

void WriteRegister::setPred(uint32_t predicate)
{
    m_binary.pred = predicate;
}

void WriteRegister::setRegOffset(unsigned regOffset)
{
    m_binary.reg_offset = regOffset;
}

void WriteRegister::setSwtc(uint8_t val)
{
    m_binary.swtc = val;
}

void WriteRegister::setReg(uint8_t reg)
{
    m_binary.reg = reg;
}

void WriteRegister::setEngBarrier(uint8_t value)
{
    m_binary.eng_barrier = value;
}

void WriteRegister::setMsgBarrier(uint8_t value)
{
    m_binary.msg_barrier = value;
}

// --------------------------------------------------------
// -------------------- EbPadding ---------------------
// --------------------------------------------------------

EbPadding::EbPadding(unsigned numPadding) : EbPaddingCommon(numPadding)
{
    unsigned regOffset = getRegForEbPadding();
    m_binary           = {0};
    m_binary.opcode    = PACKET_WREG_32;
    fillPackets(regOffset);
}

uint64_t EbPadding::writeInstruction(void* whereTo) const
{
    char*    whereToChar         = reinterpret_cast<char*>(whereTo);
    unsigned bytesWritten        = 0;
    uint32_t EbPaddingPacketSize = sizeof(packet_wreg32);
    for (unsigned i = 0; i < m_numPadding; ++i)
    {
        memcpy(whereToChar + bytesWritten, (void*)getEbPaddingPacketAddr(), EbPaddingPacketSize);
        bytesWritten += EbPaddingPacketSize;
    }
    return bytesWritten;
}

uint32_t EbPadding::getValue() const
{
    return m_binary.value;
}

unsigned EbPadding::getPred() const
{
    return m_binary.pred;
}

unsigned EbPadding::getReg() const
{
    return m_binary.reg;
}

unsigned EbPadding::getRegOffset() const
{
    return m_binary.reg_offset;
}

uint8_t EbPadding::getSwtc() const
{
    return m_binary.swtc;
}

unsigned EbPadding::getOpcode() const
{
    return m_binary.opcode;
}

unsigned EbPadding::GetBinarySize() const
{
    return sizeof(packet_wreg32) * m_numPadding;
}

uint32_t* EbPadding::getEbPaddingPacketAddr() const
{
    return (uint32_t*)&m_binary;
}

void EbPadding::setValue(uint32_t value)
{
    m_binary.value = value;
}

void EbPadding::setRegOffset(unsigned regOffset)
{
    m_binary.reg_offset = regOffset;
}

// --------------------------------------------------------
// ----------------- WriteManyRegisters -------------------
// --------------------------------------------------------

WriteManyRegisters::WriteManyRegisters(unsigned        firstRegOffset,
                                       unsigned        count32bit,
                                       const uint32_t* values,
                                       uint32_t        predicate)
: WriteManyRegistersCommon()
{
    m_writeBulkBinary             = {0};
    m_writeBulkBinary.opcode      = PACKET_WREG_BULK;
    m_writeBulkBinary.eng_barrier = 0x0;
    m_writeBulkBinary.msg_barrier = 0x1;
    m_writeBulkBinary.swtc        = 0x0;

    fillPackets(firstRegOffset, count32bit, predicate, values);
}

std::shared_ptr<WriteRegisterCommon>
WriteManyRegisters::createWriteRegister(unsigned offset, unsigned value, uint32_t predicate) const
{
    return std::make_shared<WriteRegister>(offset, value, m_commandId, predicate);
}

void WriteManyRegisters::setSize(unsigned bulkSize)
{
    m_writeBulkBinary.size64 = bulkSize;
}

void WriteManyRegisters::setPredicate(uint32_t predicate)
{
    m_writeBulkBinary.pred = predicate;
}

void WriteManyRegisters::setOffset(unsigned offset)
{
    m_writeBulkBinary.reg_offset = offset;
}

unsigned WriteManyRegisters::getRegOffset() const
{
    return m_writeBulkBinary.reg_offset;
}

unsigned WriteManyRegisters::getWregBulkSize() const
{
    return sizeof(m_writeBulkBinary);
}

unsigned WriteManyRegisters::getPacketWregBulkSize() const  // TODO is same as getWregBulkSize??
{
    return sizeof(packet_wreg_bulk);
}

uint32_t* WriteManyRegisters::getBulkBinaryPacketAddr() const
{
    return (uint32_t*)&m_writeBulkBinary;
}

void WriteManyRegisters::setWregBulkSwitchCQ(bool value)
{
    m_writeBulkBinary.swtc = value;
}

uint8_t WriteManyRegisters::getWregBulkSwitchCQ() const
{
    return m_writeBulkBinary.swtc;
}

uint32_t WriteManyRegisters::getWregBulkPredicate() const
{
    return m_writeBulkBinary.pred;
}

// --------------------------------------------------------
// --------------------- ArcExeWdTpc ----------------------
// --------------------------------------------------------

void ArcExeWdTpc::Print() const
{
    LOG_DEBUG(QMAN, "      ArcExeWdTpc, swtc={}, numCtxs={}", getSwtc(), getNumCtxs());
}

void ArcExeWdTpc::addCtx(const tpc_wd_ctxt_t& ctx)
{
    HB_ASSERT(m_ctxCount < sizeof(m_ctxs)/sizeof(tpc_wd_ctxt_t), "array overflow");
    m_ctxs[m_ctxCount++] = ctx;
}

uint8_t ArcExeWdTpc::getSwtc() const
{
    return m_ctxs[0].switch_bit; // it's enough to look at one of the contexts
}

void ArcExeWdTpc::setSwtc(uint8_t val)
{
    for (unsigned i = 0; i < m_ctxCount; i++)
    {
        m_ctxs[i].switch_bit = val;
    }
}

unsigned ArcExeWdTpc::GetBinarySize() const
{
    return m_ctxCount * sizeof(tpc_wd_ctxt_t);
}

const void* ArcExeWdTpc::getArcExeWdTpcCtxAddr() const
{
    return m_ctxs;
}
// --------------------------------------------------------
// --------------------- ArcExeWdDma ----------------------
// --------------------------------------------------------

uint8_t ArcExeWdDma::getSwtc() const
{
    return m_ctx.switch_bit;
}

void ArcExeWdDma::setSwtc(uint8_t val)
{
    m_ctx.switch_bit = val;
}

unsigned ArcExeWdDma::GetBinarySize() const
{
    return sizeof(edma_wd_ctxt_t);
}

const void* ArcExeWdDma::getArcExeWdDmaCtxAddr() const
{
    return &m_ctx;
}

// --------------------------------------------------------
// --------------------- ArcExeWdMme ----------------------
// --------------------------------------------------------

uint8_t ArcExeWdMme::getSwtc() const
{
    return m_ctx.switch_bit;
}

void ArcExeWdMme::setSwtc(uint8_t val)
{
    m_ctx.switch_bit = val;
}

unsigned ArcExeWdMme::GetBinarySize() const
{
    return sizeof(mme_wd_ctxt_t);
}

const void* ArcExeWdMme::getArcExeWdMmeCtxAddr() const
{
    return &m_ctx;
}

// --------------------------------------------------------
// --------------------- ArcExeWdRot ----------------------
// --------------------------------------------------------

uint8_t ArcExeWdRot::getSwtc() const
{
    return m_ctx.switch_bit;
}

void ArcExeWdRot::setSwtc(uint8_t val)
{
    m_ctx.switch_bit = val;
}

uint8_t ArcExeWdRot::getCplMsgEn() const
{
    return m_ctx.cpl_msg_en;
}

void ArcExeWdRot::setCplMsgEn(uint8_t val)
{
    m_ctx.cpl_msg_en = val;
}

unsigned ArcExeWdRot::GetBinarySize() const
{
    return sizeof(rot_wd_ctxt_t);
}

const void* ArcExeWdRot::getArcExeWdRotCtxAddr() const
{
    return &m_ctx;
}

// --------------------------------------------------------
// ----------------------- Fence --------------------------
// --------------------------------------------------------

Fence::Fence(WaitID waitID, unsigned int targetValue, uint32_t predicate) : FenceCommon(targetValue)
{
    m_binaries.resize(m_numPkts);

    for (unsigned i = 0; i < m_numPkts; ++i)
    {
        m_binaries[i]             = {0};
        m_binaries[i].opcode      = PACKET_FENCE;
        m_binaries[i].eng_barrier = 0x0;
        m_binaries[i].msg_barrier = 0x0;
        m_binaries[i].ch_id       = 0x0;
    }

    fillPackets(waitID, predicate);
}

uint8_t Fence::getIdByIndex(uint32_t idx) const
{
    return m_binaries[idx].id;
}

uint8_t Fence::getSwtcByIndex(uint32_t idx) const
{
    return m_binaries[idx].swtc;
}

uint16_t Fence::getTargetValByIndex(uint32_t idx) const
{
    return m_binaries[idx].target_val;
}

uint8_t Fence::getDecValByIndex(uint32_t idx) const
{
    return m_binaries[idx].dec_val;
}

uint32_t Fence::getFencePacketSize() const
{
    return sizeof(packet_fence);
}

uint32_t* Fence::getFencePacketAddr() const
{
    return (uint32_t*)(m_binaries.data());
}

void Fence::setID(uint32_t idx, uint8_t val)
{
    m_binaries[idx].id = val;
}

void Fence::setSwtc(uint32_t idx, uint8_t val)
{
    m_binaries[idx].swtc = val;
}

void Fence::setPredicate(uint32_t idx, uint8_t val)
{
    m_binaries[idx].pred = val;
}

void Fence::setTargetVal(uint32_t idx, uint16_t val)
{
    m_binaries[idx].target_val = val;
}

void Fence::setDecVal(uint32_t idx, uint8_t val)
{
    m_binaries[idx].dec_val = val;
}

// --------------------------------------------------------
// ---------------------- Suspend -------------------------
// --------------------------------------------------------

Suspend::Suspend(WaitID waitID, unsigned int waitCycles, unsigned int incrementValue)
: SuspendCommon(waitID, waitCycles, incrementValue)
{
    m_wait  = std::make_shared<Wait>(waitID, waitCycles, incrementValue);
    m_fence = std::make_shared<Fence>(waitID, incrementValue);
}

// --------------------------------------------------------
// ------------------------ Execute -----------------------
// --------------------------------------------------------

Execute::Execute(HabanaDeviceType type, unsigned deviceID, uint32_t predicate, uint32_t value)
: WriteRegister(getRegForExecute(type, deviceID), value, predicate), m_deviceType(type), m_deviceID(deviceID)
{
}

void Execute::Print() const
{
    LOG_DEBUG(QMAN,
              "      Execute {} {}, value=0x{:x}, swtc={}",
              getDeviceName(m_deviceType),
              m_deviceID,
              m_binary.value,
              m_binary.swtc);
}

// --------------------------------------------------------
// --------------------- WriteReg64----------------------
// --------------------------------------------------------

WriteReg64::WriteReg64(unsigned baseRegIndex,               // index of the base registers entry
                       uint64_t value,                      // value to add to the base register
                       unsigned targetRegisterInBytes,      // where to write the sum
                       bool     writeTargetLow /*=true*/,   // write the low part of the target
                       bool     writeTargetHigh /*=true*/,  // write the high part of the target
                       uint32_t predicate)
: WriteReg64Common()
{
    m_binaryLong  = {0};
    m_binaryShort = {0};

    HB_ASSERT(baseRegIndex < Gaudi3HalReader::instance()->getBaseRegistersCacheSize(), "invalid cache index");
    HB_ASSERT(writeTargetLow || writeTargetHigh, "at least one part must be set");

    // We have to use wreg64_long if the value is larger than 31 bits (due to sign extension done by the HW on bit
    // 32) or if we want to write only the low or the high part to the target address after summation, but not both
    // part. In any other case we can use wreg64_short.
    m_useLongBinary = (value & 0xFFFFFFFF80000000) || !writeTargetLow || !writeTargetHigh;

    if (m_useLongBinary)
    {
        fillLongBinary(baseRegIndex, value, targetRegisterInBytes, writeTargetLow, writeTargetHigh, predicate);
    }
    else
    {
        fillShortBinary(baseRegIndex, value, targetRegisterInBytes, predicate);
    }
}

uint32_t WriteReg64::getWriteReg64PacketSize() const
{
    return m_useLongBinary ? sizeof(packet_wreg64_long) : sizeof(packet_wreg64_short);
}

uint32_t* WriteReg64::getWriteReg64PacketAddr() const
{
    return m_useLongBinary ? (uint32_t*)&m_binaryLong : (uint32_t*)&m_binaryShort;
}

uint8_t WriteReg64::getDwEnable() const
{
    return m_binaryLong.dw_enable;
}

uint32_t WriteReg64::getCtl() const
{
    return m_useLongBinary ? m_binaryLong.ctl : m_binaryShort.ctl;
}

uint32_t WriteReg64::getDregOffset() const
{
    return m_useLongBinary ? m_binaryLong.dreg_offset : m_binaryShort.dreg_offset;
}

uint32_t WriteReg64::getBaseIndex() const
{
    return m_useLongBinary ? m_binaryLong.base : m_binaryShort.base;
}

uint64_t WriteReg64::getValue() const
{
    return m_useLongBinary ? m_binaryLong.offset : m_binaryShort.offset;
}

unsigned WriteReg64::getPred() const
{
    return m_useLongBinary ? m_binaryLong.pred : m_binaryShort.pred;
}

uint8_t WriteReg64::getSwtc() const
{
    return m_useLongBinary ? m_binaryLong.swtc : m_binaryShort.swtc;
}

void WriteReg64::setPred(uint32_t predicate)
{
    m_useLongBinary ? m_binaryLong.pred = predicate : m_binaryShort.pred = predicate;
}

void WriteReg64::setValue(uint64_t value)
{
    m_useLongBinary ? m_binaryLong.offset = value : m_binaryShort.offset = (uint32_t)value;
}

void WriteReg64::setDregOffset(uint32_t offset)
{
    m_useLongBinary ? m_binaryLong.dreg_offset = offset : m_binaryShort.dreg_offset = offset;
}

void WriteReg64::setBaseIndex(unsigned baseIndex)
{
    m_useLongBinary ? m_binaryLong.base = baseIndex : m_binaryShort.base = baseIndex;
}

void WriteReg64::setSwtc(uint8_t val)
{
    m_useLongBinary ? m_binaryLong.swtc = val : m_binaryShort.swtc = val;
}

void WriteReg64::setEngBarrier(uint8_t val)
{
    m_useLongBinary ? m_binaryLong.eng_barrier = val : m_binaryShort.eng_barrier = val;
}

void WriteReg64::setMsgBarrier(uint8_t val)
{
    m_useLongBinary ? m_binaryLong.msg_barrier = val : m_binaryShort.msg_barrier = val;
}

void WriteReg64::setDwEnable(uint8_t val)
{
    m_binaryLong.dw_enable = val;
}

void WriteReg64::setOpcode()
{
    m_useLongBinary ? m_binaryLong.opcode = PACKET_WREG_64_LONG : m_binaryShort.opcode = PACKET_WREG_64_SHORT;
}

void WriteReg64::setRel(uint8_t val)
{
    m_binaryLong.rel = val;
}

// --------------------------------------------------------
// ----------------- InvalidateTPCCaches ------------------
// --------------------------------------------------------

InvalidateTPCCaches::InvalidateTPCCaches(uint32_t predicate)
: WriteRegister(GET_ADDR_OF_TPC_BLOCK_FIELD(tpc_cmd), 0x0, predicate)
{
    m_binary.value       = calcTpcCmdVal();
    m_binary.eng_barrier = 1;
}

uint32_t InvalidateTPCCaches::calcTpcCmdVal()
{
    tpc::reg_tpc_cmd cmd;
    cmd._raw              = 0;
    cmd.icache_invalidate = 1;
    cmd.dcache_invalidate = 1;
    cmd.lcache_invalidate = 1;
    cmd.tcache_invalidate = 1;
    return cmd._raw;
}

void InvalidateTPCCaches::Print() const
{
    LOG_DEBUG(QMAN, "      Invalidate TPC caches, swtc={}", m_binary.swtc);
}

// --------------------------------------------------------
// ------------------ UploadKernelsAddr -------------------
// --------------------------------------------------------

UploadKernelsAddr::UploadKernelsAddr(uint32_t uploadToLow, uint32_t uploadToHigh, uint32_t predicate)
: UploadKernelsAddrCommon(uploadToLow, uploadToHigh, predicate, Gaudi3HalReader::instance()->getPrefetchAlignmentMask())
{
    tpc::reg_tpc_cmd cmd;
    cmd._raw                 = 0;
    cmd.icache_prefetch_64kb = 1;
    m_ebPadding = createEbPadding(Gaudi3HalReader::instance()->getNumUploadKernelEbPad());
    for (unsigned i = 0; i < m_numOfPackets; ++i)
    {
        m_binaries[i]             = {0};
        m_binaries[i].opcode      = PACKET_WREG_32;
        m_binaries[i].msg_barrier = 0x0;
        m_binaries[i].eng_barrier = (i == (m_numOfPackets - 1)) ? 1 : 0;
    }

    fillPackets(cmd._raw);
    ptrToInt kernelBaseAddr;
    kernelBaseAddr.u32[0]          = uploadToLow & Gaudi3HalReader::instance()->getPrefetchAlignmentMask();
    kernelBaseAddr.u32[1]          = uploadToHigh;
    uint64_t                 memId = getMemoryIDFromVirtualAddress(kernelBaseAddr.u64);
    BasicFieldsContainerInfo addrContainer;

    addrContainer.addAddressEngineFieldInfo(nullptr,
                                            getMemorySectionNameForMemoryID(memId),
                                            memId,
                                            kernelBaseAddr.u32[0],
                                            kernelBaseAddr.u32[1],
                                            (uint32_t)Gaudi3HalReader::instance()->getNumUploadKernelEbPad(),
                                            (uint32_t)Gaudi3HalReader::instance()->getNumUploadKernelEbPad() + 1,
                                            FIELD_MEMORY_TYPE_DRAM);
    SetContainerInfo(addrContainer);
}

void UploadKernelsAddr::Print() const
{
    if (!LOG_LEVEL_AT_LEAST_DEBUG(QMAN)) return;
    m_ebPadding->Print();
    ptrToInt p;
    p.u32[0] = m_lowAddress;
    p.u32[1] = m_highAddress;
    LOG_DEBUG(QMAN, "      Prefetch icache at base address 0x{:x}, swtc={}", p.u64, getSwtcByIndex(m_numOfPackets - 1));
}

std::shared_ptr<EbPaddingCommon> UploadKernelsAddr::createEbPadding(unsigned numPadding) const
{
    return std::make_shared<EbPadding>(numPadding);
}

uint8_t UploadKernelsAddr::getSwtcByIndex(uint32_t idx) const
{
    return m_binaries[idx].swtc;
}

unsigned UploadKernelsAddr::GetBinarySize() const
{
        return m_ebPadding->GetBinarySize() +
           m_numOfPackets * sizeof(packet_wreg32);  // This commands translates to 3 wreg32
}

uint64_t UploadKernelsAddr::writeInstruction(void* whereTo) const
{
    // Impossible to perform ptr arithmetic with void*, so we use char*
    char*    whereToChar  = reinterpret_cast<char*>(whereTo);
    unsigned bytesWritten = 0;
    bytesWritten += m_ebPadding->writeInstruction(whereToChar);
    memcpy(whereToChar + bytesWritten, (void*)getUploadKernelPacketAddr(), m_numOfPackets * sizeof(packet_wreg32));

    bytesWritten += m_numOfPackets * sizeof(packet_wreg32);
    return bytesWritten;
}

void UploadKernelsAddr::prepareFieldInfos()
{
    if (0 == m_addressContainerInfo.size())
    {
        return;
    }

    HB_ASSERT(2 == m_addressContainerInfo.size(), "Unexpected number of patching points for UploadKernelAddr command");
    BasicFieldInfoSet    basicSet;
    AddressFieldInfoSet& addressSet = m_addressContainerInfo.retrieveAddressFieldInfoSet();
    transferAddressToBasic(addressSet, basicSet);
    prepareFieldInfos(basicSet);
    transferBasicToAddress(basicSet, addressSet);

    prepareFieldInfos(m_addressContainerInfo.retrieveBasicFieldInfoSet());
}

void UploadKernelsAddr::prepareFieldInfos(BasicFieldInfoSet& basicFieldsInfoSet)
{
    BasicFieldInfoSet updatedBasicFieldSet;
    for (auto& singleBasicFieldsInfoPair : basicFieldsInfoSet)
    {
        // update offset value based on size of actual CMD, in this case each padding command is 64bit, therfore we need
        // to adjust patcpoint index to 2* index give
        auto copy = singleBasicFieldsInfoPair;
        if (copy.second->getFieldIndexOffset() == m_ebPadding->getEbPaddingNumPadding() + 1)
        {
            copy.first = (m_ebPadding->getEbPaddingNumPadding() + 1) * 2;
            copy.second->setFieldIndexOffset((m_ebPadding->getEbPaddingNumPadding() + 1) * 2);
        }
        else if (copy.second->getFieldIndexOffset() == m_ebPadding->getEbPaddingNumPadding())
        {
            copy.first = (m_ebPadding->getEbPaddingNumPadding()) * 2;
            copy.second->setFieldIndexOffset((m_ebPadding->getEbPaddingNumPadding()) * 2);
        }
        else
        {
            HB_ASSERT(false, "Unexpected field index offset in UploadKernelAddr command");
        }

        updatedBasicFieldSet.insert(copy);
    }

    basicFieldsInfoSet.clear();
    basicFieldsInfoSet.insert(updatedBasicFieldSet.begin(), updatedBasicFieldSet.end());
}

uint32_t* UploadKernelsAddr::getUploadKernelPacketAddr() const
{
    return (uint32_t*)m_binaries;
}

uint16_t UploadKernelsAddr::getAddrOfTpcBlockField(std::string_view name) const
{
    if (name == "icache_base_adderess_low")
    {
        return GET_ADDR_OF_TPC_BLOCK_FIELD(icache_base_adderess_low);
    }
    else if (name == "icache_base_adderess_high")
    {
        return GET_ADDR_OF_TPC_BLOCK_FIELD(icache_base_adderess_high);
    }
    else if (name == "tpc_cmd")
    {
        return GET_ADDR_OF_TPC_BLOCK_FIELD(tpc_cmd);
    }

    HB_ASSERT(0, "No field with name = {} in tpc_block", name);
    return 0;
}

void UploadKernelsAddr::setPacket(uint32_t idx)
{
    m_binaries[idx] = {0};
}

void UploadKernelsAddr::setPredicate(uint32_t idx, uint8_t val)
{
    m_binaries[idx].pred = val;
}

void UploadKernelsAddr::setSwtc(uint32_t idx, uint8_t val)
{
    m_binaries[idx].swtc = val;
}

void UploadKernelsAddr::setRegOffset(uint32_t idx, uint16_t val)
{
    m_binaries[idx].reg_offset = val;
}

void UploadKernelsAddr::setValue(uint32_t idx, uint32_t val)
{
    m_binaries[idx].value = val;
}

void UploadKernelsAddr::setEngBarrier(uint32_t idx, uint8_t val)
{
    m_binaries[idx].eng_barrier = val;
}

// --------------------------------------------------------
// ----------------------- MsgLong ------------------------
// --------------------------------------------------------

MsgLong::MsgLong() : MsgLongCommon()
{
    m_binary = {0};
}

void MsgLong::setSwtc(uint8_t val)
{
    m_binary.swtc = val;
}

uint8_t MsgLong::getSwtc() const
{
    return m_binary.swtc;
}

uint32_t MsgLong::getMsgLongPacketSize() const
{
    return sizeof(packet_msg_long);
}

uint32_t* MsgLong::getMsgLongPacketAddr() const
{
    return (uint32_t*)&m_binary;
}

// --------------------------------------------------------
// --------------------- QmanDelay ------------------------
// --------------------------------------------------------

// Add QmanDelay command as a WA for a HW issue - H6-3262 (https://jira.habana-labs.com/browse/H6-3262)
// To avoid race between updating regs in cache to read them using wreg64
QmanDelay::QmanDelay(uint32_t predicate) : QmanDelayCommon()
{
    m_wreg  = std::make_shared<WriteRegister>(QMAN_BLOCK_BASE + offsetof(block_qman, cp_fence2_rdata) +
                                                 sizeof(struct qman::reg_cp_fence2_rdata) * 4,
                                             1,
                                             predicate);
    m_fence = std::make_shared<Fence>(ID_2, 1, predicate);
}

// --------------------------------------------------------
// -------------------- McidRollover ----------------------
// --------------------------------------------------------

McidRollover::McidRollover(unsigned target, unsigned targetXps)
: m_switchBit(false), m_target(target), m_targetXps(targetXps)
{
    m_isDynamic = true;
}

void McidRollover::Print() const
{
    if (getTarget() == 0 && getTargetXps() == 0)
    {
        LOG_DEBUG(QMAN, "      McidRollover(virtual, canceled), swtc={}", getSwtc());
    }
    else
    {
        std::string targetXps("");
        if (getTargetXps() != 0) targetXps = std::string(", targetXps=") + std::to_string(getTargetXps());
        LOG_DEBUG(QMAN, "      McidRollover(virtual), swtc={}, target={}{}", getSwtc(), getTarget(), targetXps);
    }
}

unsigned McidRollover::GetBinarySize() const
{
    return 0;  // virtual command has no binary size from the program perspective
}

uint64_t McidRollover::writeInstruction(void* whereTo) const
{
    return 0;  // virtual command has no binary size from the program perspective
}

void McidRollover::setSwitchCQ()
{
    setSwtc(true);
}

void McidRollover::resetSwitchCQ()
{
    setSwtc(false);
}

void McidRollover::toggleSwitchCQ()
{
    setSwtc(!getSwtc());
}

bool McidRollover::isSwitchCQ() const
{
    return getSwtc();
}

}  // namespace gaudi3
