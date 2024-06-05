#include <string>
#include <memory>
#include <syn_logging.h>
#include "types.h"
#include "platform/gaudi2/graph_compiler/queue_command.h"
#include "utils.h"
#include "../utils.hpp"
#include "habana_global_conf.h"
#include "block_data.h"

namespace gaudi2
{

void setSendSyncEvents(uint32_t& raw)
{
    raw |= (1 << SEND_SYNC_BIT);
}

static void prepareFieldInfosTwoDwordsHeader(BasicFieldInfoSet& basicFieldsInfoSet)
{
    BasicFieldInfoSet updatedBasicFieldSet;
    for (auto& singleBasicFieldsInfoPair : basicFieldsInfoSet)
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

static void prepareFieldInfosTwoDwordsHeader(BasicFieldInfoSet&   basicFieldsInfoSet,
                                             AddressFieldInfoSet& addressFieldsInfoSet)
{
    BasicFieldInfoSet tempBasicSet;
    transferAddressToBasic(addressFieldsInfoSet, tempBasicSet);
    prepareFieldInfosTwoDwordsHeader(tempBasicSet);
    transferBasicToAddress(tempBasicSet, addressFieldsInfoSet);

    prepareFieldInfosTwoDwordsHeader(basicFieldsInfoSet);
}

// --------------------------------------------------------
// ------------------ Gaudi2QueueCommand -------------------
// --------------------------------------------------------

Gaudi2QueueCommand::Gaudi2QueueCommand() : QueueCommand() {}

Gaudi2QueueCommand::Gaudi2QueueCommand(uint32_t packetType) : QueueCommand(packetType) {}

Gaudi2QueueCommand::Gaudi2QueueCommand(uint32_t packetType, uint64_t commandId) : QueueCommand(packetType, commandId) {}

Gaudi2QueueCommand::~Gaudi2QueueCommand() {}

void Gaudi2QueueCommand::WritePB(gc_recipe::generic_packets_container* pktCon)
{
    LOG_ERR(QMAN, "{}: should not be invoked for Gaudi2", HLLOG_FUNC);
    HB_ASSERT(0, "Func should not be invoked for Gaudi2");
}

void Gaudi2QueueCommand::WritePB(gc_recipe::generic_packets_container* pktCon, ParamsManager* params)
{
    LOG_ERR(QMAN, "{}: should not be invoked for Gaudi2", HLLOG_FUNC);
    HB_ASSERT(0, "Func should not be invoked for Gaudi2");
}

// --------------------------------------------------------
// ---------------------- DmaCommand ----------------------
// --------------------------------------------------------

// --------------------------------------------------------
// ------------------ DmaDeviceInternal -------------------
// --------------------------------------------------------

DmaDeviceInternal::DmaDeviceInternal(deviceAddrOffset src,
                                     bool             srcInDram,
                                     deviceAddrOffset dst,
                                     bool             dstInDram,
                                     uint64_t         size,
                                     bool             setEngBarrier,
                                     bool             isMemset,
                                     bool             wrComplete,
                                     uint16_t         contextId)
: DmaCommand(), m_src(src), m_dst(dst)
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
        else  // !srcInDram && !dstInDram
        {
            m_operationStr = "DMA memcpy from SRAM to SRAM:";
        }
    }

    m_binary = {0};

    m_binary.tsize       = size;
    m_binary.wrcomp      = wrComplete;
    m_binary.endian      = 0;
    m_binary.memset      = isMemset ? 1 : 0;
    m_binary.opcode      = PACKET_LIN_DMA;
    m_binary.eng_barrier = setEngBarrier ? 1 : 0;
    m_binary.msg_barrier = 0x1;
    m_binary.swtc        = 0;
    m_binary.src_addr    = m_src;
    m_binary.dst_addr    = m_dst;
}

unsigned DmaDeviceInternal::GetBinarySize() const
{
    return sizeof(packet_lin_dma);
}

DmaDeviceInternal::~DmaDeviceInternal() {}

void DmaDeviceInternal::Print() const
{
    LOG_DEBUG(QMAN,
              "      {} src=0x{:x} dst=0x{:x} size=0x{:x} swtc={} {}",
              m_operationStr,
              m_src,
              m_dst,
              m_binary.tsize,
              m_binary.swtc,
              m_binary.wrcomp ? " Signal completion *****" : "");
}

uint64_t DmaDeviceInternal::writeInstruction(void* whereTo) const
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

void DmaDeviceInternal::setSwitchCQ()
{
    m_binary.swtc = 1;
}

void DmaDeviceInternal::resetSwitchCQ()
{
    m_binary.swtc = 0;
}

void DmaDeviceInternal::toggleSwitchCQ()
{
    m_binary.swtc = ~m_binary.swtc;
}

bool DmaDeviceInternal::isSwitchCQ() const
{
    return m_binary.swtc;
}

// --------------------------------------------------------
// -------------------- DmaDramToSram ---------------------
// --------------------------------------------------------

DmaDramToSram::DmaDramToSram(deviceAddrOffset dramPtr,
                             deviceAddrOffset sramPtr,
                             uint64_t         size,
                             bool             wrComplete,
                             uint16_t         contextID)
: DmaDeviceInternal(dramPtr, true, sramPtr, false, size, false, false, wrComplete, contextID)
{
}

// --------------------------------------------------------
// -------------------- DmaSramToDram ---------------------
// --------------------------------------------------------

DmaSramToDram::DmaSramToDram(deviceAddrOffset dramPtr,
                             deviceAddrOffset sramPtr,
                             uint64_t         size,
                             bool             wrComplete,
                             uint16_t         contextID)
: DmaDeviceInternal(sramPtr, false, dramPtr, true, size, false, false, wrComplete, contextID)
{
}

// --------------------------------------------------------
// ------------------------ CpDma -------------------------
// --------------------------------------------------------

CpDma::CpDma(deviceAddrOffset addrOffset, uint64_t size, uint64_t dramBase, uint32_t predicate)
: Gaudi2QueueCommand(), m_addrOffset(addrOffset), m_transferSize(size)
{
    UNUSED(dramBase);
    m_binary             = {0};
    m_binary.tsize       = m_transferSize;
    m_binary.pred        = predicate;
    m_binary.upper_cp    = 0x0;
    m_binary.opcode      = PACKET_CP_DMA;
    m_binary.eng_barrier = 0x0;
    m_binary.msg_barrier = 0x1;
    m_binary.src_addr    = m_addrOffset;
}

void CpDma::Print() const
{
    LOG_DEBUG(QMAN, "      DMA from SRAM [0x{:x}] size {} to the tightly-coupled logic", m_addrOffset, m_transferSize);
}

unsigned CpDma::GetBinarySize() const
{
    return sizeof(packet_cp_dma);
}

uint64_t CpDma::writeInstruction(void* whereTo) const
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
    m_binary        = {0};
    m_binary.opcode = PACKET_WREG_32;
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
// --------------------- ArcExeWdTpc ----------------------
// --------------------------------------------------------

uint8_t ArcExeWdTpc::getSwtc() const
{
    return m_ctx.switch_bit;
}

void ArcExeWdTpc::setSwtc(uint8_t val)
{
    m_ctx.switch_bit = val;
}

unsigned ArcExeWdTpc::GetBinarySize() const
{
    return sizeof(tpc_wd_ctxt_t);
}

const void* ArcExeWdTpc::getArcExeWdTpcCtxAddr() const
{
    return &m_ctx;
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

void ArcExeWdDma::prepareFieldInfos() {}

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
// -------------------- ExecuteDmaDesc --------------------
// --------------------------------------------------------

ExecuteDmaDesc::ExecuteDmaDesc(uint32_t         bits,
                               HabanaDeviceType type,
                               unsigned         deviceID,
                               bool             setEngBarrier,
                               uint32_t         predicate)
: WriteRegister(getRegForExecute(type, deviceID), bits, predicate), m_deviceType(type), m_deviceID(deviceID)
{
    m_commit._raw = bits;
    if (setEngBarrier)
    {
        m_binary.eng_barrier = 1;
    }
}

ExecuteDmaDesc::~ExecuteDmaDesc() {}

void ExecuteDmaDesc::Print() const
{
    LOG_DEBUG(QMAN,
              "      Execute {} {}, signaling={}, memset={}, swtc={}",
              getDeviceName(m_deviceType),
              m_deviceID,
              m_commit.wr_comp_en ? "true" : "false",
              m_commit.mem_set ? "true" : "false",
              m_binary.swtc);
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
// -------------------------- SFGCmd ----------------------
// --------------------------------------------------------

SFGCmd::SFGCmd(unsigned sigOutValue) : m_switchBit(false), m_sfgSyncObjValue(sigOutValue)
{
    m_isDynamic = true;
}

void SFGCmd::Print() const
{
    LOG_DEBUG(QMAN, "      SFGCmd(virtual), swtc={}, sob_val={}", getSwtc(), getSfgSyncObjValue());
}

unsigned SFGCmd::GetBinarySize() const
{
    return 0;
}

uint64_t SFGCmd::writeInstruction(void* whereTo) const
{
    return 0;
}

void SFGCmd::setSwitchCQ()
{
    setSwtc(true);
}

void SFGCmd::resetSwitchCQ()
{
    setSwtc(false);
}

void SFGCmd::toggleSwitchCQ()
{
    setSwtc(!getSwtc());
}

bool SFGCmd::isSwitchCQ() const
{
    return getSwtc();
}

bool SFGCmd::getSwtc() const
{
    return m_switchBit;
}

void SFGCmd::setSwtc(bool val)
{
    m_switchBit = val;
}

// --------------------------------------------------------
// ------------------------- SFGInitCmd -------------------
// --------------------------------------------------------

SFGInitCmd::SFGInitCmd(unsigned sigOutValue) : SFGCmd(sigOutValue) {}

void SFGInitCmd::Print() const
{
    LOG_DEBUG(QMAN, "      SFGInitCmd(virtual), swtc={}, sob_val={}", getSwtc(), getSfgSyncObjValue());
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

Wait::~Wait() {}

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
// ----------------------- Fence --------------------------
// --------------------------------------------------------

Fence::Fence(WaitID id, unsigned int targetValue, uint32_t predicate) : FenceCommon(targetValue)
{
    m_binaries.resize(m_numPkts);

    for (unsigned i = 0; i < m_numPkts; ++i)
    {
        m_binaries[i]             = {0};
        m_binaries[i].opcode      = PACKET_FENCE;
        m_binaries[i].eng_barrier = 0x0;
        m_binaries[i].msg_barrier = 0x0;
    }

    fillPackets(id, predicate);
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
// ------------------- MonitorSetup -----------------------
// --------------------------------------------------------

MonitorSetup::MonitorSetup(SyncObjectManager::SyncId mon,
                           WaitID                    waitID,
                           HabanaDeviceType          device,
                           unsigned                  deviceID,
                           uint32_t                  value,
                           unsigned                  streamID,
                           uint32_t                  predicate)
: Gaudi2QueueCommand(), m_mon(mon), m_predicate(predicate)
{
    memset(m_msBinaries, 0, sizeof(m_msBinaries));
    memset(m_mlBinaries, 0, sizeof(m_mlBinaries));

    if (m_predicate)
    {
        makeMonitorSetupBinaryMsgLong(getCPFenceOffset(device, deviceID, waitID, streamID), value);
    }
    else
    {
        makeMonitorSetupBinaryMsgShort(getCPFenceOffset(device, deviceID, waitID, streamID), value);
    }
}

MonitorSetup::MonitorSetup(SyncObjectManager::SyncId mon,
                           SyncObjectManager::SyncId syncId,
                           uint32_t                  value,
                           uint32_t                  predicate)
: Gaudi2QueueCommand(), m_mon(mon), m_predicate(predicate)
{
    memset(m_msBinaries, 0, sizeof(m_msBinaries));
    memset(m_mlBinaries, 0, sizeof(m_mlBinaries));

    if (m_predicate)
    {
        makeMonitorSetupBinaryMsgLong(getSyncObjectAddress(syncId), value);
    }
    else
    {
        makeMonitorSetupBinaryMsgShort(getSyncObjectAddress(syncId), value);
    }
}

void MonitorSetup::makeMonitorSetupBinaryMsgShort(uint64_t address, uint32_t value)
{
    ptrToInt p;
    p.u64 = address;

    for (unsigned i = 0; i < m_numOfPackets; i++)
    {
        m_msBinaries[i] = {0};

        m_msBinaries[i].weakly_ordered = 0;  // Default value
        m_msBinaries[i].no_snoop       = 0;  // Default value
        m_msBinaries[i].dw             = 0;
        m_msBinaries[i].op             = 0;
        m_msBinaries[i].base           = 0;
        m_msBinaries[i].opcode         = PACKET_MSG_SHORT;
        m_msBinaries[i].eng_barrier    = 0;
        m_msBinaries[i].msg_barrier    = 0;
        m_msBinaries[i].swtc           = 0;
    }

    unsigned monitorBlockBase = getMonPayloadAddress(0, MON_PAYLOAD_ADDR_L);

    // First config packet: low address of the sync payload
    m_msBinaries[0].msg_addr_offset = getMonPayloadAddress(m_mon, MON_PAYLOAD_ADDR_L) - monitorBlockBase;
    m_msBinaries[0].value           = p.u32[0];

    // Second config packet: high address of the sync payload
    m_msBinaries[1].msg_addr_offset = getMonPayloadAddress(m_mon, MON_PAYLOAD_ADDR_H) - monitorBlockBase;
    m_msBinaries[1].value           = p.u32[1];

    // Third config packet: the payload data, i.e. what to write when the sync triggers
    m_msBinaries[2].msg_addr_offset = getMonPayloadAddress(m_mon, MON_PAYLOAD_DATA) - monitorBlockBase;
    m_msBinaries[2].value           = value;
}

void MonitorSetup::makeMonitorSetupBinaryMsgLong(uint64_t address, uint32_t value)
{
    ptrToInt p;
    p.u64 = address;

    for (unsigned i = 0; i < m_numOfPackets; i++)
    {
        m_mlBinaries[i] = {0};

        m_mlBinaries[i].opcode      = PACKET_MSG_LONG;
        m_mlBinaries[i].msg_barrier = 0;
        m_mlBinaries[i].eng_barrier = 0;
        m_mlBinaries[i].swtc        = 0;
        m_mlBinaries[i].pred        = m_predicate;
        m_mlBinaries[i].op          = 0;
    }

    // First config packet: low address of the sync payload
    m_mlBinaries[0].addr  = getMonPayloadAddress(m_mon, MON_PAYLOAD_ADDR_L);
    m_mlBinaries[0].value = p.u32[0];

    // Second config packet: high address of the sync payload
    m_mlBinaries[1].addr  = getMonPayloadAddress(m_mon, MON_PAYLOAD_ADDR_H);
    m_mlBinaries[1].value = p.u32[1];

    // Third config packet: the payload data, i.e. what to write when the sync triggers
    m_mlBinaries[2].addr  = getMonPayloadAddress(m_mon, MON_PAYLOAD_DATA);
    m_mlBinaries[2].value = value;
}

void MonitorSetup::Print() const
{
    if (!LOG_LEVEL_AT_LEAST_DEBUG(QMAN)) return;

    ptrToInt p;
    p.u32[1] = m_predicate ? m_mlBinaries[1].value : m_msBinaries[1].value;
    p.u32[0] = m_predicate ? m_mlBinaries[0].value : m_msBinaries[0].value;

    LOG_DEBUG(QMAN,
              "      Setup monitor {} to write {} to 0x{:x} with predicate={}, swtc={}",
              m_mon,
              m_predicate ? m_mlBinaries[2].value : m_msBinaries[2].value,
              p.u64,
              m_predicate,
              m_predicate ? m_mlBinaries[m_numOfPackets - 1].swtc : m_msBinaries[m_numOfPackets - 1].swtc);
}

unsigned MonitorSetup::GetBinarySize() const
{
    return m_numOfPackets * (m_predicate ? sizeof(packet_msg_long) : sizeof(packet_msg_short));
}

uint64_t MonitorSetup::writeInstruction(void* whereTo) const
{
    if (m_predicate)
    {
        memcpy(whereTo, m_mlBinaries, sizeof(m_mlBinaries));
        return sizeof(m_mlBinaries);
    }
    else
    {
        memcpy(whereTo, m_msBinaries, sizeof(m_msBinaries));
        return sizeof(m_msBinaries);
    }
}

void MonitorSetup::prepareFieldInfos()
{
    if (0 == m_addressContainerInfo.size())
    {
        return;
    }

    HB_ASSERT(2 == m_addressContainerInfo.size(), "Unexpected number of patching points for MonitorSetup command");
    BasicFieldInfoSet    basicSet;
    AddressFieldInfoSet& addressSet = m_addressContainerInfo.retrieveAddressFieldInfoSet();
    transferAddressToBasic(addressSet, basicSet);
    prepareFieldInfos(basicSet);
    transferBasicToAddress(basicSet, addressSet);

    prepareFieldInfos(m_addressContainerInfo.retrieveBasicFieldInfoSet());
}

void MonitorSetup::prepareFieldInfos(BasicFieldInfoSet& basicFieldInfoSet)
{
    BasicFieldInfoSet updatedBasicFieldSet;
    for (auto& singleBasicFieldsInfoPair : basicFieldInfoSet)
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
            HB_ASSERT(false, "UNexpected field index offset in MonitorSetup command");
        }

        updatedBasicFieldSet.insert(copy);
    }

    basicFieldInfoSet.clear();
    basicFieldInfoSet.insert(updatedBasicFieldSet.begin(), updatedBasicFieldSet.end());
}

void MonitorSetup::setSwitchCQ()
{
    if (m_predicate)
    {
        m_mlBinaries[m_numOfPackets - 1].swtc = 1;  // set the switch on the last packet
    }
    else
    {
        m_msBinaries[m_numOfPackets - 1].swtc = 1;  // set the switch on the last packet
    }
}

void MonitorSetup::resetSwitchCQ()
{
    if (m_predicate)
    {
        m_mlBinaries[m_numOfPackets - 1].swtc = 0;  // reset the switch on the last packet
    }
    else
    {
        m_msBinaries[m_numOfPackets - 1].swtc = 0;  // reset the switch on the last packet
    }
}

void MonitorSetup::toggleSwitchCQ()
{
    if (m_predicate)
    {
        m_mlBinaries[m_numOfPackets - 1].swtc = ~m_mlBinaries[m_numOfPackets - 1].swtc;  // toggle the last packet
    }
    else
    {
        m_msBinaries[m_numOfPackets - 1].swtc = ~m_msBinaries[m_numOfPackets - 1].swtc;  // toggle the last packet
    }
}

bool MonitorSetup::isSwitchCQ() const
{
    if (m_predicate)
    {
        return m_mlBinaries[m_numOfPackets - 1].swtc;  // get from the last packet
    }
    else
    {
        return m_msBinaries[m_numOfPackets - 1].swtc;  // get from the last packet
    }
}

// --------------------------------------------------------
// --------------------- MonitorArm -----------------------
// --------------------------------------------------------

MonitorArm::MonitorArm(SyncObjectManager::SyncId syncObj,
                       SyncObjectManager::SyncId mon,
                       MonitorOp                 operation,
                       unsigned                  syncValue,
                       Settable<uint8_t>         mask)
: Gaudi2QueueCommand()
{
    unsigned monitorBlockBase = offsetof(block_sob_objs, mon_pay_addrl);

    m_mon       = mon;
    m_syncValue = syncValue;
    m_syncObj   = syncObj;
    m_operation = operation;
    m_mask      = mask;

    uint8_t  syncMask;
    unsigned syncGroupId;
    if (mask.is_set())
    {
        syncMask    = ~(mask.value());
        syncGroupId = syncObj;
    }
    else
    {
        syncMask    = ~(static_cast<uint8_t>(0x1U << (syncObj % 8)));
        syncGroupId = syncObj / 8;
    }

    m_binary                                = {0};
    m_binary.mon_arm_register.sync_group_id = syncGroupId;
    m_binary.mon_arm_register.mask          = syncMask;
    m_binary.mon_arm_register.mode          = operation;
    m_binary.mon_arm_register.sync_value    = syncValue;

    m_binary.msg_addr_offset = varoffsetof(block_sob_objs, mon_arm[mon]) - monitorBlockBase;
    m_binary.weakly_ordered  = 0;  // Default value
    m_binary.no_snoop        = 0;  // Default value
    m_binary.dw              = 0;
    m_binary.op              = 0;
    m_binary.base            = 0;
    m_binary.opcode          = PACKET_MSG_SHORT;
    m_binary.eng_barrier     = 0;
    m_binary.msg_barrier     = 0;
    m_binary.swtc            = 0;
}

void MonitorArm::Print() const
{
    LOG_DEBUG(QMAN,
              "      Arm monitor {} to wait {} for semaphore {} [is_mask: {}, mask: {:b}] to reach {}, swtc={}",
              m_mon,
              (m_operation == MONITOR_SO_OP_GREQ ? "(>=)" : "(==)"),
              (m_mask.is_set() ? 8 * m_syncObj : m_syncObj),
              m_mask.is_set(),
              (m_mask.is_set() ? m_mask.value() : 0),
              m_syncValue,
              m_binary.swtc);
}

unsigned MonitorArm::GetBinarySize() const
{
    return sizeof(packet_msg_short);
}

uint64_t MonitorArm::writeInstruction(void* whereTo) const
{
    memcpy(whereTo, &m_binary, sizeof(m_binary));
    return sizeof(m_binary);
}

void MonitorArm::prepareFieldInfos() {}

void MonitorArm::setSwitchCQ()
{
    m_binary.swtc = 1;
}

void MonitorArm::resetSwitchCQ()
{
    m_binary.swtc = 0;
}

void MonitorArm::toggleSwitchCQ()
{
    m_binary.swtc = ~m_binary.swtc;
}

bool MonitorArm::isSwitchCQ() const
{
    return m_binary.swtc;
}

// --------------------------------------------------------
// ------------------ SignalSemaphore ---------------------
// --------------------------------------------------------

SignalSemaphore::SignalSemaphore(SyncObjectManager::SyncId which, int16_t syncValue, int operation, int barriers)
: Gaudi2QueueCommand()
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
    m_binary.dw              = 0;
    m_binary.op              = 0;
    m_binary.base            = 1;
    m_binary.opcode          = PACKET_MSG_SHORT;
    m_binary.eng_barrier     = (barriers & ENGINE_BARRIER) ? 1 : 0;
    m_binary.msg_barrier     = (barriers & MESSAGE_BARRIER) ? 1 : 0;
    m_binary.swtc            = 0;
}

void SignalSemaphore::Print() const
{
    unsigned so = (m_binary.msg_addr_offset >> 2);
    LOG_DEBUG(QMAN,
              "      Signal semaphore {} [0x{:x}] {} {}, swtc={}",
              so,
              getSyncObjectAddress(so),
              (m_operation ? " increment by " : " set to "),
              m_syncValue,
              m_binary.swtc);
}

unsigned SignalSemaphore::GetBinarySize() const
{
    return sizeof(packet_msg_short);
}

uint64_t SignalSemaphore::writeInstruction(void* whereTo) const
{
    memcpy(whereTo, &m_binary, sizeof(m_binary));
    return sizeof(m_binary);
}

void SignalSemaphore::setSwitchCQ()
{
    m_binary.swtc = 1;
}

void SignalSemaphore::resetSwitchCQ()
{
    m_binary.swtc = 0;
}

void SignalSemaphore::toggleSwitchCQ()
{
    m_binary.swtc = ~m_binary.swtc;
}

bool SignalSemaphore::isSwitchCQ() const
{
    return m_binary.swtc;
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

WaitForSemaphore::~WaitForSemaphore() {}

void WaitForSemaphore::Print() const
{
    LOG_DEBUG(QMAN,
              "      Arm monitor {} to wait {} for semaphore {} [is_mask: {}, mask: {:b}] to reach {} and wait for "
              "fence # {} to reach {}, swtc={}",
              m_mon,
              (m_operation == MONITOR_SO_OP_GREQ ? "(>=)" : "(==)"),
              (m_mask.is_set() ? 8 * m_syncObj : m_syncObj),
              m_mask.is_set(),
              (m_mask.is_set() ? m_mask.value() : 0),
              m_syncValue,
              m_waitID,
              m_fence.getTargetVal(),
              m_fence.getSwtc());
}
unsigned WaitForSemaphore::GetBinarySize() const
{
    return m_fence.GetBinarySize() + m_monitorArm.GetBinarySize();
}

uint64_t WaitForSemaphore::writeInstruction(void* whereTo) const
{
    unsigned bytesWritten   = 0;
    char*    whereToCharPtr = reinterpret_cast<char*>(whereTo);

    bytesWritten += m_monitorArm.writeInstruction(whereToCharPtr + bytesWritten);
    bytesWritten += m_fence.writeInstruction(whereToCharPtr + bytesWritten);

    return bytesWritten;
}

void WaitForSemaphore::prepareFieldInfos() {}

void WaitForSemaphore::setSwitchCQ()
{
    m_fence.setSwitchCQ();  // set the switch on the last command
}

void WaitForSemaphore::resetSwitchCQ()
{
    m_fence.resetSwitchCQ();  // reset the switch on the last command
}

void WaitForSemaphore::toggleSwitchCQ()
{
    m_fence.toggleSwitchCQ();  // toggle the switch on the last command
}

bool WaitForSemaphore::isSwitchCQ() const
{
    return m_fence.isSwitchCQ();  // get the switch from the last command
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
// ----------------- InvalidateTPCCaches ------------------
// --------------------------------------------------------

InvalidateTPCCaches::InvalidateTPCCaches(uint32_t predicate)
: WriteRegister(GET_ADDR_OF_TPC_BLOCK_FIELD(tpc_cmd), 0x0, predicate)
{
    m_binary.eng_barrier = 1;
    m_binary.value       = calcTpcCmdVal();
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
: UploadKernelsAddrCommon(uploadToLow, uploadToHigh, predicate, Gaudi2HalReader::instance()->getPrefetchAlignmentMask())
{
    tpc::reg_tpc_cmd cmd;
    cmd._raw                 = 0;
    cmd.icache_prefetch_64kb = 1;

    m_ebPadding = createEbPadding(Gaudi2HalReader::instance()->getNumUploadKernelEbPad());
    for (unsigned i = 0; i < m_numOfPackets; ++i)
    {
        m_binaries[i]             = {0};
        m_binaries[i].opcode      = PACKET_WREG_32;
        m_binaries[i].msg_barrier = 0x0;
        // we set eng_barrier for reg_tpc_cmd, Prefetch HW Bug WA
        m_binaries[i].eng_barrier = (i == (m_numOfPackets - 1)) ? 1 : 0;
    }

    fillPackets(cmd._raw);
    ptrToInt kernelBaseAddr;
    kernelBaseAddr.u32[0]          = uploadToLow & Gaudi2HalReader::instance()->getPrefetchAlignmentMask();
    kernelBaseAddr.u32[1]          = uploadToHigh;
    uint64_t                 memId = getMemoryIDFromVirtualAddress(kernelBaseAddr.u64);
    BasicFieldsContainerInfo addrContainer;

    addrContainer.addAddressEngineFieldInfo(nullptr,
                                            getMemorySectionNameForMemoryID(memId),
                                            memId,
                                            kernelBaseAddr.u32[0],
                                            kernelBaseAddr.u32[1],
                                            (uint32_t)Gaudi2HalReader::instance()->getNumUploadKernelEbPad(),
                                            (uint32_t)Gaudi2HalReader::instance()->getNumUploadKernelEbPad() + 1,
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

    HB_ASSERT(baseRegIndex < Gaudi2HalReader::instance()->getBaseRegistersCacheSize(), "invalid cache index");
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
// -------------------- ResetSyncObject -------------------
// --------------------------------------------------------

ResetSyncObject::ResetSyncObject(unsigned syncID, bool logLevelTrace, uint32_t predicate)
: m_syncID(syncID), m_logLevelTrace(logLevelTrace)
{
    m_binary.opcode      = PACKET_MSG_LONG;
    m_binary.value       = 0;
    m_binary.pred        = predicate;
    m_binary.op          = 0;
    m_binary.msg_barrier = 0;
    m_binary.eng_barrier = 0;
    m_binary.swtc        = 0;
}

ResetSyncObject::~ResetSyncObject() {}

void ResetSyncObject::Print() const
{
    if (!LOG_LEVEL_AT_LEAST_DEBUG(QMAN)) return;

    if (m_logLevelTrace)
    {
        LOG_TRACE(QMAN, "      Reset sync obj #{} at address 0x{:x}, swtc={}", m_syncID, m_binary.addr, m_binary.swtc);
    }
    else
    {
        LOG_DEBUG(QMAN, "      Reset sync obj #{} at address 0x{:x}, swtc={}", m_syncID, m_binary.addr, m_binary.swtc);
    }
}

// --------------------------------------------------------
// -------------------- IncrementFence --------------------
// --------------------------------------------------------

IncrementFence::IncrementFence(HabanaDeviceType deviceType,
                               unsigned         deviceID,
                               WaitID           waitID,
                               unsigned         streamID,
                               uint32_t         predicate)
{
    m_binary.opcode      = PACKET_MSG_LONG;
    m_binary.value       = 1;
    m_binary.pred        = predicate;
    m_binary.op          = 0;
    m_binary.msg_barrier = 0;
    m_binary.eng_barrier = 0;
    m_binary.swtc        = 0;
    m_binary.addr        = getCPFenceOffset(deviceType, deviceID, waitID, streamID);
}

IncrementFence::~IncrementFence() {}

void IncrementFence::Print() const
{
    LOG_DEBUG(QMAN, "      Increment CP fence counter at address 0x{:x}, swtc={}", m_binary.addr, m_binary.swtc);
}

// --------------------------------------------------------
// --------------------- LoadPredicates -------------------
// --------------------------------------------------------

LoadPredicates::LoadPredicates(deviceAddrOffset src, uint32_t predicate) : Gaudi2QueueCommand()
{
    m_binary = {0};

    m_binary.opcode      = PACKET_LOAD_AND_EXE;
    m_binary.pred        = predicate;
    m_binary.msg_barrier = 0x1;
    m_binary.eng_barrier = 0x0;
    m_binary.swtc        = 0x0;
    m_binary.load        = 0x1;  // loading operation
    m_binary.dst         = 0x0;  // loading predicates and not scalars
    m_binary.exe         = 0x0;
    m_binary.etype       = 0x0;  // don't care (irrelevant for loading predicates)
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
    LOG_DEBUG(QMAN, "      LoadPredicates from address 0x{:x}, swtc={}", m_binary.src_addr, m_binary.swtc);
}

uint64_t LoadPredicates::getSrcAddrForTesting() const
{
    return (uint64_t)m_binary.src_addr;
}

void LoadPredicates::setSwitchCQ()
{
    m_binary.swtc = 1;
}

void LoadPredicates::resetSwitchCQ()
{
    m_binary.swtc = 0;
}

void LoadPredicates::toggleSwitchCQ()
{
    m_binary.swtc = ~m_binary.swtc;
}

bool LoadPredicates::isSwitchCQ() const
{
    return m_binary.swtc;
}

// --------------------------------------------------------
// ------------------ SignalSemaphoreWithPredicate --------
// --------------------------------------------------------

SignalSemaphoreWithPredicate::SignalSemaphoreWithPredicate(SyncObjectManager::SyncId which,
                                                           int16_t                   syncValue,
                                                           uint32_t                  pred,
                                                           int                       operation,
                                                           int                       barriers)
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
    m_binary.msg_barrier     = (barriers & MESSAGE_BARRIER) ? 1 : 0;
    m_binary.swtc            = 0;
}

void SignalSemaphoreWithPredicate::Print() const
{
    if (!LOG_LEVEL_AT_LEAST_DEBUG(QMAN)) return;

    LOG_DEBUG(QMAN,
              "      Signal semaphore {} [0x{:x}] {} {}, swtc={}",
              m_syncId,
              m_binary.addr,
              (m_operation ? " increment by " : " set to "),
              m_syncValue,
              m_binary.swtc);
}

// --------------------------------------------------------
// ------------------ CompositeQueueCommand ---------------
// --------------------------------------------------------

CompositeQueueCommand::CompositeQueueCommand(std::vector<std::shared_ptr<Gaudi2QueueCommand>> commands)
: m_commands(std::move(commands))
{
}

void CompositeQueueCommand::Print() const
{
    for (auto& command : m_commands)
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
        [](unsigned i, const std::shared_ptr<Gaudi2QueueCommand>& command) { return i + command->GetBinarySize(); });
}

uint64_t CompositeQueueCommand::writeInstruction(void* whereTo) const
{
    uint64_t totalWritten = 0;
    for (auto& command : m_commands)
    {
        totalWritten += command->writeInstruction((char*)whereTo + totalWritten);
    }
    return totalWritten;
}

void CompositeQueueCommand::setSwitchCQ()
{
    m_commands.back()->setSwitchCQ();  // set the switch on the last command
}

void CompositeQueueCommand::resetSwitchCQ()
{
    m_commands.back()->resetSwitchCQ();  // reset the switch on the last command
}

void CompositeQueueCommand::toggleSwitchCQ()
{
    m_commands.back()->toggleSwitchCQ();  // toggle the switch on the last command
}

bool CompositeQueueCommand::isSwitchCQ() const
{
    return m_commands.back()->isSwitchCQ();  // get the switch from the last command
}

DynamicExecute::DynamicExecute(std::vector<std::shared_ptr<Gaudi2QueueCommand>> commands)
: CompositeQueueCommand(commands)
{
}

void DynamicExecute::prepareFieldInfos()
{
    auto& data = m_addressContainerInfo.retrieveBasicFieldInfoSet();
    if (data.empty()) return;

    auto& fieldInfo = data.begin()->second;
    fieldInfo->setFieldIndexOffset(1);  // pred starts with one

    const auto& dynamicFieldInfo = std::dynamic_pointer_cast<DynamicShapeFieldInfo>(fieldInfo);
    HB_ASSERT(dynamicFieldInfo != nullptr, "invalid field info");

    switch (fieldInfo->getType())
    {
        case (FieldType::FIELD_DYNAMIC_EXECUTE_NO_SIGNAL):
        {
            prepareFieldInfoNoSignal(dynamicFieldInfo);
            break;
        }
        case (FieldType::FIELD_DYNAMIC_EXECUTE_WITH_SIGNAL):
        {
            prepareFieldInfoSignalOnce(dynamicFieldInfo);
            break;
        }
        case (FieldType::FIELD_DYNAMIC_EXECUTE_MME):
        {
            prepareFieldInfoSignalMME(dynamicFieldInfo);
            break;
        }
        default:
        {
            HB_ASSERT(0, "invalid field info");
        }
    }
}

void DynamicExecute::prepareFieldInfoNoSignal(const DynamicShapeFieldInfoSharedPtr& fieldInfo)
{
    dynamic_execution_sm_params_t metadata;
    initMetaData(metadata, 1);

    struct DataLayout
    {
        uint32_t executePred : 5;
        uint64_t notPred : 27;
    };
    auto layoutedData         = reinterpret_cast<DataLayout*>(metadata.commands);
    layoutedData->executePred = 31;

    updateFieldInfo(fieldInfo, metadata);
}

void DynamicExecute::prepareFieldInfoSignalOnce(const DynamicShapeFieldInfoSharedPtr& fieldInfo)
{
    dynamic_execution_sm_params_t metadata;
    initMetaData(metadata, 3);

    struct DataLayout
    {
        uint32_t executePred : 5;
        uint64_t notPred : 59;
        uint32_t signalPred : 5;
        uint32_t notPred2 : 27;
    };
    auto layoutedData         = reinterpret_cast<DataLayout*>(metadata.commands);
    auto currentExecutePred   = layoutedData->executePred;
    layoutedData->executePred = layoutedData->signalPred;
    layoutedData->signalPred  = currentExecutePred;

    updateFieldInfo(fieldInfo, metadata);
}

void DynamicExecute::prepareFieldInfoSignalMME(const DynamicShapeFieldInfoSharedPtr& fieldInfo)
{
    dynamic_execution_sm_params_t metadata;
    initMetaData(metadata, 7);

    // Disable bypass optimization for the MME, Currently there are 2 execute commands for a single mme roi.
    // If we enable this optimization then the second execute is bypassed (Which means it is executed).
    // This optimization can be enabled in the future again by setting the bypass metadata to true on the correct
    // PP.
    // TODO : SW-19925
    metadata.should_bypass = false;

    struct DataLayout
    {
        uint32_t executePred : 5;
        uint32_t exeNotPred : 27;  // 1
        uint32_t signal1NotPred0;  // 2
        uint32_t signal1Pred : 5;
        uint32_t signal1NotPred1 : 27;  // 3
        uint32_t signal1NotPred2;       // 4
        uint32_t signal1NotPred3;       // 5
        uint32_t signal2NotPred0;       // 6
        uint32_t signal2Pred : 5;
        uint32_t signal2NotPred1 : 27;  // 7
    };

    auto layoutedData         = reinterpret_cast<DataLayout*>(metadata.commands);
    auto currentExecutePred   = layoutedData->executePred;
    layoutedData->executePred = layoutedData->signal1Pred;
    layoutedData->signal1Pred = currentExecutePred;
    layoutedData->signal2Pred = currentExecutePred;

    updateFieldInfo(fieldInfo, metadata);
}

void DynamicExecute::initMetaData(dynamic_execution_sm_params_t& metadata, size_t cmdLen)
{
    memset(&metadata, 0, sizeof(metadata));
    metadata.cmd_len       = cmdLen;
    metadata.should_bypass = true;

    std::vector<uint8_t> instr(GetBinarySize());
    writeInstruction(instr.data());
    // The first dword isn't part of the patching data.
    memcpy(metadata.commands, instr.data() + sizeof(uint32_t), metadata.cmd_len * sizeof(uint32_t));
}

void DynamicExecute::updateFieldInfo(const DynamicShapeFieldInfoSharedPtr& fieldInfo,
                                     dynamic_execution_sm_params_t&        metadata)
{
    std::vector<uint8_t> serializedMetadata(sizeof(metadata));
    memcpy(serializedMetadata.data(), &metadata, sizeof(metadata));
    fieldInfo->setMetadata(std::move(serializedMetadata));
    fieldInfo->setSize(metadata.cmd_len);
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

}  // namespace gaudi2
