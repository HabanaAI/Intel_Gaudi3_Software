#pragma once

#include "types.h"
#include "utils.h"
#include "habana_device_types.h"
#include "../../../graph_compiler/queue_command.h"
#include "platform/common/queue_command.h"
#include "graph_compiler/sync/sync_object_manager.h"
#include "gaudi2_types.h"

#include "recipe_metadata.h"

#include "gaudi2/gaudi2_packets.h"
#include "gaudi2_arc_eng_packets.h"

namespace gaudi2
{
void setSendSyncEvents(uint32_t& raw);

// A base class for describing commands that can be pushed into FIFO queues on Gaudi2
class Gaudi2QueueCommand : public QueueCommand
{
public:
    virtual ~Gaudi2QueueCommand();

    virtual void WritePB(gc_recipe::generic_packets_container* pktCon) override;
    virtual void WritePB(gc_recipe::generic_packets_container* pktCon, ParamsManager* params) override;

protected:
    Gaudi2QueueCommand();
    Gaudi2QueueCommand(uint32_t packetType);
    Gaudi2QueueCommand(uint32_t packetType, uint64_t commandId);

private:
    Gaudi2QueueCommand(const Gaudi2QueueCommand&) = delete;
    void operator=(const Gaudi2QueueCommand&) = delete;
};

class CompositeQueueCommand : public Gaudi2QueueCommand
{
public:
    CompositeQueueCommand(std::vector<std::shared_ptr<Gaudi2QueueCommand>> commands);

    virtual void     Print() const override;
    virtual unsigned GetBinarySize() const override;
    virtual uint64_t writeInstruction(void* whereTo) const override;

    virtual void setSwitchCQ() override;
    virtual void resetSwitchCQ() override;
    virtual void toggleSwitchCQ() override;
    virtual bool isSwitchCQ() const override;

private:
    std::vector<std::shared_ptr<Gaudi2QueueCommand>> m_commands;
};

class DmaCommand : public Gaudi2QueueCommand
{
public:
    virtual ~DmaCommand() = default;

    // For testing
    virtual const packet_lin_dma& getPacket() const = 0;
};

class DmaDeviceInternal : public DmaCommand
{
public:
    DmaDeviceInternal(deviceAddrOffset src,
                      bool             srcInDram,
                      deviceAddrOffset dst,
                      bool             dstInDram,
                      uint64_t         size,
                      bool             setEngBarrier,
                      bool             isMemset,
                      bool             wrComplete,
                      uint16_t         contextId = 0);
    virtual ~DmaDeviceInternal();

    virtual void     Print() const override;
    virtual uint64_t writeInstruction(void* whereTo) const override;
    virtual void     prepareFieldInfos() override;
    virtual unsigned GetBinarySize() const override;

    virtual void setSwitchCQ() override;
    virtual void resetSwitchCQ() override;
    virtual void toggleSwitchCQ() override;
    virtual bool isSwitchCQ() const override;

    // For testing
    const packet_lin_dma& getPacket() const override { return m_binary; }

protected:
    deviceAddrOffset m_src;
    deviceAddrOffset m_dst;
    std::string      m_operationStr;
    packet_lin_dma   m_binary;
};

class DmaDramToSram : public DmaDeviceInternal
{
public:
    DmaDramToSram(deviceAddrOffset dramPtr,
                  deviceAddrOffset sramPtr,
                  uint64_t         size,
                  bool             wrComplete,
                  uint16_t         contextID);
};

class DmaSramToDram : public DmaDeviceInternal
{
public:
    DmaSramToDram(deviceAddrOffset dramPtr,
                  deviceAddrOffset sramPtr,
                  uint64_t         size,
                  bool             wrComplete,
                  uint16_t         contextID);
};

class CpDma : public Gaudi2QueueCommand
{
public:
    CpDma(deviceAddrOffset addrPtr, uint64_t size, uint64_t dramBase, uint32_t predicate = DEFAULT_PREDICATE);
    virtual ~CpDma() = default;

    virtual void     Print() const override;
    virtual unsigned GetBinarySize() const override;
    virtual uint64_t writeInstruction(void* whereTo) const override;
    virtual void     prepareFieldInfos() override;

    // CpDma has no switch bit
    virtual void setSwitchCQ() override {}
    virtual void resetSwitchCQ() override {}
    virtual void toggleSwitchCQ() override {}
    virtual bool isSwitchCQ() const override { return false; }

    inline deviceAddrOffset getTargetAddr() { return m_addrOffset; }

protected:
    deviceAddrOffset m_addrOffset;
    uint64_t         m_transferSize;
    packet_cp_dma    m_binary;
};

class WriteRegister : public WriteRegisterCommon
{
public:
    WriteRegister(unsigned regOffset, unsigned value, uint32_t predicate = DEFAULT_PREDICATE);
    WriteRegister(unsigned regOffset, unsigned value, uint64_t commandId, uint32_t predicate = DEFAULT_PREDICATE);
    virtual ~WriteRegister() = default;

    // For testing
    const packet_wreg32 getPacket() const { return m_binary; }
    void                setValue(uint32_t value) override;
    virtual uint32_t    getValue() const override;

    virtual unsigned getRegOffset() const override;

protected:
    packet_wreg32 m_binary;

    virtual unsigned  getPred() const override;
    virtual unsigned  getReg() const override;
    virtual uint8_t   getSwtc() const override;
    virtual uint32_t  getWriteRegisterPacketSize() const override;
    virtual uint32_t* getWriteRegisterPacketAddr() const override;
    virtual void      setPred(uint32_t predicate) override;
    virtual void      setRegOffset(unsigned regOffset) override;
    virtual void      setSwtc(uint8_t val) override;
    virtual void      setReg(uint8_t reg) override;
    virtual void      setEngBarrier(uint8_t value) override;
    virtual void      setMsgBarrier(uint8_t value) override;
};

class EbPadding : public EbPaddingCommon
{
public:
    EbPadding(unsigned numPadding);
    virtual ~EbPadding() = default;

    // For testing
    void             setValue(uint32_t value) override;
    virtual uint32_t getValue() const override;

    virtual unsigned  getRegOffset() const override;
    virtual unsigned  GetBinarySize() const override;
    virtual uint32_t* getEbPaddingPacketAddr() const override;

protected:
    packet_wreg32 m_binary;

    uint64_t writeInstruction(void* whereTo) const override;

    virtual unsigned getPred() const override;
    virtual unsigned getReg() const override;
    virtual uint8_t  getSwtc() const override;
    virtual unsigned getOpcode() const override;
    virtual void     setRegOffset(unsigned regOffset) override;
};

class WriteManyRegisters : public WriteManyRegistersCommon
{
public:
    WriteManyRegisters(unsigned        firstRegOffset,
                       unsigned        count,
                       const uint32_t* values,
                       uint32_t        predicate = DEFAULT_PREDICATE);
    virtual ~WriteManyRegisters() = default;

protected:
    packet_wreg_bulk m_writeBulkBinary;  // Not including the values
    virtual std::shared_ptr<WriteRegisterCommon>
                      createWriteRegister(unsigned offset, unsigned value, uint32_t predicate) const override;
    virtual void      setSize(unsigned bulkSize) override;
    virtual void      setPredicate(uint32_t predicate) override;
    virtual void      setOffset(unsigned offset) override;
    virtual unsigned  getRegOffset() const override;
    virtual unsigned  getWregBulkSize() const override;
    virtual unsigned  getPacketWregBulkSize() const override;
    virtual uint32_t* getBulkBinaryPacketAddr() const override;
    virtual void      setWregBulkSwitchCQ(bool value) override;
    virtual uint8_t   getWregBulkSwitchCQ() const override;
    virtual uint32_t  getWregBulkPredicate() const override;
};

class LoadDesc : public WriteManyRegisters
{
public:
    LoadDesc(void*            desc,
             unsigned         descSize,
             unsigned         descOffset,
             HabanaDeviceType device,
             unsigned         deviceID  = 0,
             uint32_t         predicate = DEFAULT_PREDICATE);

    virtual ~LoadDesc();

    virtual void Print() const override;

protected:
    // For the debug print
    HabanaDeviceType m_deviceType;
    unsigned         m_deviceID;
};

class Execute : public WriteRegister
{
public:
    Execute(HabanaDeviceType type, unsigned deviceID = 0, uint32_t predicate = DEFAULT_PREDICATE, uint32_t value = 0x1);

    virtual void Print() const override;

protected:
    // For the debug print
    HabanaDeviceType m_deviceType;
    unsigned         m_deviceID;
};

class ArcExeWdTpc : public ArcExeWdTpcCommon
{
public:
    ArcExeWdTpc(const tpc_wd_ctxt_t& ctx) : m_ctx(ctx) {}
    virtual ~ArcExeWdTpc() = default;

    const tpc_wd_ctxt_t& getFwCtxForTest() const { return m_ctx; }
    virtual void         prepareFieldInfos() override {}

protected:
    tpc_wd_ctxt_t m_ctx;

    virtual uint8_t     getSwtc() const override;
    virtual void        setSwtc(uint8_t val) override;
    virtual unsigned    GetBinarySize() const override;
    virtual const void* getArcExeWdTpcCtxAddr() const override;
};

class ArcExeWdDma : public ArcExeWdDmaCommon
{
public:
    ArcExeWdDma(const edma_wd_ctxt_t& ctx) : m_ctx(ctx) {}
    virtual ~ArcExeWdDma() = default;

    const edma_wd_ctxt_t& getFwCtxForTest() const { return m_ctx; }
    virtual void          prepareFieldInfos() override;

protected:
    edma_wd_ctxt_t m_ctx;

    virtual uint8_t     getSwtc() const override;
    virtual void        setSwtc(uint8_t val) override;
    virtual unsigned    GetBinarySize() const override;
    virtual const void* getArcExeWdDmaCtxAddr() const override;
};

class ArcExeWdMme : public ArcExeWdMmeCommon
{
public:
    ArcExeWdMme(const mme_wd_ctxt_t& ctx) : m_ctx(ctx) {}
    virtual ~ArcExeWdMme() = default;

    const mme_wd_ctxt_t& getFwCtxForTest() const { return m_ctx; }
    virtual void         prepareFieldInfos() override {}

protected:
    mme_wd_ctxt_t m_ctx;

    virtual uint8_t     getSwtc() const override;
    virtual void        setSwtc(uint8_t val) override;
    virtual unsigned    GetBinarySize() const override;
    virtual const void* getArcExeWdMmeCtxAddr() const override;
};

class ArcExeWdRot : public ArcExeWdRotCommon
{
public:
    ArcExeWdRot(const rot_wd_ctxt_t& ctx) : m_ctx(ctx) {}
    virtual ~ArcExeWdRot() = default;

    const rot_wd_ctxt_t& getFwCtxForTest() const { return m_ctx; }
    virtual void         prepareFieldInfos() override {}

protected:
    rot_wd_ctxt_t m_ctx;

    virtual uint8_t     getSwtc() const override;
    virtual void        setSwtc(uint8_t val) override;
    virtual uint8_t     getCplMsgEn() const override;
    virtual void        setCplMsgEn(uint8_t val) override;
    virtual unsigned    GetBinarySize() const override;
    virtual const void* getArcExeWdRotCtxAddr() const override;
};

class ExecuteDmaDesc : public WriteRegister
{
public:
    ExecuteDmaDesc(uint32_t         bits,
                   HabanaDeviceType type,
                   unsigned         deviceID      = 0,
                   bool             setEngBarrier = false,
                   uint32_t         predicate     = DEFAULT_PREDICATE);
    virtual ~ExecuteDmaDesc();

    virtual void Print() const override;

protected:
    // For the debug print
    HabanaDeviceType         m_deviceType;
    unsigned                 m_deviceID;
    dma_core_ctx::reg_commit m_commit;
};

class Nop : public NopCommon
{
public:
    Nop();
    virtual ~Nop() = default;

protected:
    packet_nop m_binary;

    virtual uint8_t   getSwtc() const override;
    virtual void      setSwtc(uint8_t val) override;
    virtual uint32_t  getNopPacketSize() const override;
    virtual uint32_t* getNopPacketAddr() const override;
};

// SFGCmd is a virtual command that doesn't encapsulate any QMAN command
class SFGCmd : public QueueCommandCommon
{
public:
    SFGCmd(unsigned sigOutValue);
    virtual ~SFGCmd() = default;

    virtual void     Print() const override;
    virtual unsigned GetBinarySize() const override;
    virtual uint64_t writeInstruction(void* whereTo) const override;

    virtual void setSwitchCQ() override;
    virtual void resetSwitchCQ() override;
    virtual void toggleSwitchCQ() override;
    virtual bool isSwitchCQ() const override;

    unsigned getSfgSyncObjValue() const { return m_sfgSyncObjValue; }

    virtual bool invalidateHistory() const override { return true; }

protected:
    virtual bool getSwtc() const;
    virtual void setSwtc(bool val);

private:
    bool     m_switchBit;
    unsigned m_sfgSyncObjValue;
};

// SFGInitCmd is a virtual command that doesn't encapsulate any QMAN command
class SFGInitCmd : public SFGCmd
{
public:
    SFGInitCmd(unsigned sigOutValue);
    virtual ~SFGInitCmd() = default;

    virtual void Print() const override;
};

class Wait : public WaitCommon
{
public:
    Wait(WaitID waitID, unsigned int waitCycles, unsigned int incrementValue);
    virtual ~Wait();

protected:
    packet_wait m_binary;

    virtual uint8_t   getId() const override;
    virtual uint8_t   getIncVal() const override;
    virtual uint32_t  getNumCyclesToWait() const override;
    virtual uint32_t  getWaitPacketSize() const override;
    virtual uint32_t* getWaitPacketAddr() const override;
};

class Fence : public FenceCommon
{
public:
    Fence(WaitID waitID, unsigned int targetValue, uint32_t predicate = DEFAULT_PREDICATE);
    virtual ~Fence() = default;

protected:
    std::vector<packet_fence> m_binaries;

    virtual uint8_t   getIdByIndex(uint32_t idx) const override;
    virtual uint8_t   getSwtcByIndex(uint32_t idx) const override;
    virtual uint16_t  getTargetValByIndex(uint32_t idx) const override;
    virtual uint8_t   getDecValByIndex(uint32_t idx) const override;
    virtual uint32_t  getFencePacketSize() const override;
    virtual uint32_t* getFencePacketAddr() const override;

    virtual void setID(uint32_t idx, uint8_t val) override;
    virtual void setSwtc(uint32_t idx, uint8_t val) override;
    virtual void setPredicate(uint32_t idx, uint8_t val) override;
    virtual void setTargetVal(uint32_t idx, uint16_t val) override;
    virtual void setDecVal(uint32_t idx, uint8_t val) override;
};

class Suspend : public SuspendCommon
{
public:
    Suspend(WaitID waitID, unsigned int waitCycles, unsigned int incrementValue = 1);
    virtual ~Suspend() = default;
};

class MonitorSetup : public Gaudi2QueueCommand
{
public:
    MonitorSetup(SyncObjectManager::SyncId mon,
                 WaitID                    waitID,
                 HabanaDeviceType          device,
                 unsigned                  deviceID,
                 uint32_t                  syncValue,
                 unsigned                  streamID,
                 uint32_t                  predicate = DEFAULT_PREDICATE);

    MonitorSetup(SyncObjectManager::SyncId mon,
                 SyncObjectManager::SyncId syncId,
                 uint32_t                  syncValue,
                 uint32_t                  predicate = DEFAULT_PREDICATE);

    virtual ~MonitorSetup() = default;

    virtual void     Print() const override;
    virtual unsigned GetBinarySize() const override;
    virtual uint64_t writeInstruction(void* whereTo) const override;
    virtual void     prepareFieldInfos() override;

    virtual void setSwitchCQ() override;
    virtual void resetSwitchCQ() override;
    virtual void toggleSwitchCQ() override;
    virtual bool isSwitchCQ() const override;

protected:
    void prepareFieldInfos(BasicFieldInfoSet& basicFieldInfoSet);

    static const unsigned m_numOfPackets = 3;

    SyncObjectManager::SyncId m_mon;
    packet_msg_short          m_msBinaries[m_numOfPackets];  // to be used without predicate
    packet_msg_long           m_mlBinaries[m_numOfPackets];  // to be used with predicate
    uint32_t                  m_predicate;

    void makeMonitorSetupBinaryMsgShort(uint64_t address, uint32_t value);
    void makeMonitorSetupBinaryMsgLong(uint64_t address, uint32_t value);
};

class MonitorArm : public Gaudi2QueueCommand
{
public:
    MonitorArm(SyncObjectManager::SyncId syncObj,
               SyncObjectManager::SyncId mon,
               MonitorOp                 operation,
               unsigned                  syncValue,
               Settable<uint8_t>         mask);
    virtual ~MonitorArm() = default;

    virtual void     Print() const override;
    virtual unsigned GetBinarySize() const override;
    virtual uint64_t writeInstruction(void* whereTo) const override;
    virtual void     prepareFieldInfos() override;
    virtual bool     isMonitorArm() const override { return true; }

    virtual void setSwitchCQ() override;
    virtual void resetSwitchCQ() override;
    virtual void toggleSwitchCQ() override;
    virtual bool isSwitchCQ() const override;

protected:
    SyncObjectManager::SyncId m_mon;
    unsigned                  m_syncValue;
    MonitorOp                 m_operation;
    SyncObjectManager::SyncId m_syncObj;
    Settable<uint8_t>         m_mask;

private:
    packet_msg_short m_binary;
};

class WaitForSemaphore : public Gaudi2QueueCommand
{
public:
    WaitForSemaphore(SyncObjectManager::SyncId syncObj,
                     SyncObjectManager::SyncId mon,
                     MonitorOp                 operation,
                     unsigned                  syncValue,
                     Settable<uint8_t>         mask,
                     WaitID                    waitID,
                     unsigned int              fenceValue);
    virtual ~WaitForSemaphore();

    virtual void     Print() const override;
    virtual unsigned GetBinarySize() const override;
    virtual uint64_t writeInstruction(void* whereTo) const override;
    virtual void     prepareFieldInfos() override;

    virtual void setSwitchCQ() override;
    virtual void resetSwitchCQ() override;
    virtual void toggleSwitchCQ() override;
    virtual bool isSwitchCQ() const override;

protected:
    MonitorArm                m_monitorArm;
    Fence                     m_fence;
    unsigned                  m_mon;
    SyncObjectManager::SyncId m_syncObj;
    MonitorOp                 m_operation;
    unsigned                  m_syncValue;
    Settable<uint8_t>         m_mask;
    WaitID                    m_waitID;
};

class SignalSemaphore : public Gaudi2QueueCommand
{
public:
    SignalSemaphore(SyncObjectManager::SyncId which, int16_t syncValue, int operation = 0, int barriers = ALL_BARRIERS);
    virtual ~SignalSemaphore() = default;

    virtual void     Print() const override;
    virtual unsigned GetBinarySize() const override;
    virtual uint64_t writeInstruction(void* whereTo) const override;

    virtual void setSwitchCQ() override;
    virtual void resetSwitchCQ() override;
    virtual void toggleSwitchCQ() override;
    virtual bool isSwitchCQ() const override;

protected:
    int16_t m_syncValue;
    int     m_operation;

private:
    packet_msg_short m_binary;
};

class InvalidateTPCCaches : public WriteRegister
{
public:
    InvalidateTPCCaches(uint32_t predicate = DEFAULT_PREDICATE);
    virtual ~InvalidateTPCCaches() = default;

    static uint32_t calcTpcCmdVal();
    virtual void Print() const override;
};

class UploadKernelsAddr : public UploadKernelsAddrCommon
{
public:
    UploadKernelsAddr(uint32_t low, uint32_t high, uint32_t predicate = DEFAULT_PREDICATE);
    virtual ~UploadKernelsAddr() = default;
    virtual unsigned GetBinarySize() const override;

protected:
    packet_wreg32 m_binaries[m_numOfPackets];
    std::shared_ptr<EbPaddingCommon> m_ebPadding;

    virtual std::shared_ptr<EbPaddingCommon> createEbPadding(unsigned numPadding) const;
    virtual void                             Print() const override;

    virtual uint8_t   getSwtcByIndex(uint32_t idx) const override;
    virtual uint32_t* getUploadKernelPacketAddr() const override;
    virtual uint16_t  getAddrOfTpcBlockField(std::string_view name) const override;
    uint64_t          writeInstruction(void* whereTo) const override;

    virtual void setPacket(uint32_t idx) override;
    virtual void setPredicate(uint32_t idx, uint8_t val) override;
    virtual void setSwtc(uint32_t idx, uint8_t val) override;
    virtual void setRegOffset(uint32_t idx, uint16_t val) override;
    virtual void setValue(uint32_t idx, uint32_t val) override;
    virtual void setEngBarrier(uint32_t idx, uint8_t val) override;

private:
    virtual void prepareFieldInfos(BasicFieldInfoSet& basicFieldsInfoSet) override;
};

class MsgLong : public MsgLongCommon
{
public:
    virtual ~MsgLong() = default;

protected:
    packet_msg_long m_binary;

    MsgLong();
    virtual void      setSwtc(uint8_t val) override;
    virtual uint8_t   getSwtc() const override;
    virtual uint32_t  getMsgLongPacketSize() const override;
    virtual uint32_t* getMsgLongPacketAddr() const override;
};

class ResetSyncObject : public MsgLong
{
public:
    ResetSyncObject(unsigned syncID, bool logLevelTrace = false, uint32_t predicate = DEFAULT_PREDICATE);
    virtual ~ResetSyncObject();

    virtual void Print() const override;

protected:
    // For the debug print
    unsigned m_syncID;
    bool     m_logLevelTrace;
};

class IncrementFence : public MsgLong
{
public:
    IncrementFence(HabanaDeviceType deviceType,
                   unsigned         deviceID,
                   WaitID           waitID,
                   unsigned         streamID,
                   uint32_t         predicate = DEFAULT_PREDICATE);
    virtual ~IncrementFence();

    virtual void Print() const override;
};

class LoadPredicates : public Gaudi2QueueCommand
{
public:
    LoadPredicates(deviceAddrOffset src, uint32_t predicate = DEFAULT_PREDICATE);
    virtual ~LoadPredicates() = default;

    virtual void     Print() const override;
    virtual unsigned GetBinarySize() const override;
    virtual uint64_t writeInstruction(void* whereTo) const override;
    virtual void     prepareFieldInfos() override;
    uint64_t         getSrcAddrForTesting() const;

    virtual void setSwitchCQ() override;
    virtual void resetSwitchCQ() override;
    virtual void toggleSwitchCQ() override;
    virtual bool isSwitchCQ() const override;

protected:
    packet_load_and_exe m_binary;
};

class SignalSemaphoreWithPredicate : public MsgLong
{
public:
    SignalSemaphoreWithPredicate(SyncObjectManager::SyncId which,
                                 int16_t                   syncValue,
                                 uint32_t                  pred,
                                 int                       operation = 0,
                                 int                       barriers  = ALL_BARRIERS);

    virtual void Print() const override;

protected:
    SyncObjectManager::SyncId m_syncId;
    int16_t                   m_syncValue;
    int                       m_operation;
};

class DynamicExecute : public CompositeQueueCommand
{
public:
    DynamicExecute(std::vector<std::shared_ptr<Gaudi2QueueCommand>> commands);
    virtual void prepareFieldInfos() override;

private:
    void prepareFieldInfoNoSignal(const DynamicShapeFieldInfoSharedPtr& fieldInfo);
    void prepareFieldInfoSignalOnce(const DynamicShapeFieldInfoSharedPtr& fieldInfo);
    void prepareFieldInfoSignalMME(const DynamicShapeFieldInfoSharedPtr& fieldInfo);

    void initMetaData(dynamic_execution_sm_params_t& metadata, size_t cmdLen);
    void updateFieldInfo(const DynamicShapeFieldInfoSharedPtr& fieldInfo, dynamic_execution_sm_params_t& metadata);
};

class WriteReg64 : public WriteReg64Common
{
public:
    WriteReg64(unsigned baseRegIndex,            // index of the base registers entry
               uint64_t value,                   // value to add to the base register
               unsigned targetRegisterInBytes,   // where to write the sum
               bool     writeTargetLow  = true,  // write the low part of the target
               bool     writeTargetHigh = true,  // write the high part of the target
               uint32_t predicate       = DEFAULT_PREDICATE);

    virtual ~WriteReg64() = default;

    virtual uint32_t  getWriteReg64PacketSize() const override;
    virtual uint32_t* getWriteReg64PacketAddr() const override;
    virtual uint8_t   getDwEnable() const override;
    virtual uint32_t  getCtl() const override;
    virtual uint32_t  getDregOffset() const override;
    virtual uint32_t  getBaseIndex() const override;
    virtual uint64_t  getValue() const override;
    virtual unsigned  getPred() const override;
    virtual uint8_t   getSwtc() const override;

protected:
    virtual void      setPred(uint32_t predicate) override;
    virtual void      setValue(uint64_t value) override;
    virtual void      setDregOffset(uint32_t offset) override;
    virtual void      setBaseIndex(unsigned baseIndex) override;
    virtual void      setSwtc(uint8_t val) override;
    virtual void      setEngBarrier(uint8_t val) override;
    virtual void      setMsgBarrier(uint8_t val) override;
    virtual void      setDwEnable(uint8_t val) override;
    virtual void      setOpcode() override;
    virtual void      setRel(uint8_t val) override;

    packet_wreg64_long  m_binaryLong;
    packet_wreg64_short m_binaryShort;
};

class QmanDelay : public QmanDelayCommon
{
public:
    QmanDelay(uint32_t predicate = DEFAULT_PREDICATE);
    virtual ~QmanDelay() = default;
};

}  // namespace gaudi2
