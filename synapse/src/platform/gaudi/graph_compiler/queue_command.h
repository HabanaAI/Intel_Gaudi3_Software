#pragma once

#include "types.h"
#include "utils.h"
#include "habana_device_types.h"
#include "../../../graph_compiler/queue_command.h"
#include "runtime/qman/gaudi/generate_packet.hpp"
#include "graph_compiler/sync/sync_object_manager.h"

#include "gaudi_types.h"
#include "gaudi/gaudi_packets.h"

#include "recipe_metadata.h"

namespace gaudi
{

unsigned getRegForLoadDesc(HabanaDeviceType type, unsigned deviceID);
uint64_t getSyncObjectAddress(unsigned so);
void setSendSyncEvents(uint32_t& raw);
unsigned getRegForEbPadding();

// A base class for describing commands that can be pushed into FIFO queues on Gaudi
class GaudiQueueCommand : public QueueCommand
{
public:
    virtual ~GaudiQueueCommand();

    virtual void WritePB(gc_recipe::generic_packets_container* pktCon) override;
    virtual void WritePB(gc_recipe::generic_packets_container* pktCon, ParamsManager* params) override;

    // gaudi1 doesn't support switch bit (only Arc platforms support it)
    virtual void setSwitchCQ() override {}
    virtual void resetSwitchCQ() override {}
    virtual void toggleSwitchCQ() override {}
    virtual bool isSwitchCQ() const override { return false; }

protected:
    GaudiQueueCommand();
    GaudiQueueCommand(uint32_t packetType);
    GaudiQueueCommand(uint32_t packetType, uint64_t commandId);

private:
    GaudiQueueCommand(const GaudiQueueCommand&)  = delete;
    void operator=(const GaudiQueueCommand&)     = delete;
};

class CompositeQueueCommand : public GaudiQueueCommand
{
public:
    CompositeQueueCommand(std::vector<std::shared_ptr<GaudiQueueCommand>> commands);

    virtual void     Print() const override;
    virtual unsigned GetBinarySize() const override;
    virtual uint64_t writeInstruction(void* whereTo) const override;

protected:
    std::vector<std::shared_ptr<GaudiQueueCommand>> m_commands;
};

class DmaCommand : public GaudiQueueCommand
{
public:
    virtual ~DmaCommand() = default;

    // For testing
    virtual const packet_lin_dma& getPacket() const = 0;
};

class DmaDeviceInternal : public DmaCommand
{
public:
    DmaDeviceInternal(deviceAddrOffset  src,
                      bool              srcInDram,
                      deviceAddrOffset  dst,
                      bool              dstInDram,
                      uint64_t          size,
                      bool              setEngBarrier,
                      bool              isMemset,
                      bool              wrComplete,
                      uint16_t          contextId = 0);
    virtual ~DmaDeviceInternal();

    virtual void     Print() const override;
    virtual uint64_t writeInstruction(void* whereTo) const               override;
    virtual void     prepareFieldInfos()                                 override;

    virtual unsigned GetBinarySize() const override;

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
    DmaDramToSram(deviceAddrOffset  dramPtr,
                  deviceAddrOffset  sramPtr,
                  uint64_t          size,
                  bool              wrComplete,
                  uint16_t          contextID);
};

class DmaSramToDram : public DmaDeviceInternal
{
public:
    DmaSramToDram(deviceAddrOffset  dramPtr,
                  deviceAddrOffset  sramPtr,
                  uint64_t          size,
                  bool              wrComplete,
                  uint16_t          contextID);
};

class CpDma : public GaudiQueueCommand
{
public:
    CpDma(deviceAddrOffset addrPtr, uint64_t size, uint64_t dramBase, uint32_t predicate = DEFAULT_PREDICATE);
    virtual ~CpDma() = default;

    virtual void            Print() const override;
    virtual unsigned GetBinarySize() const                               override;
    virtual uint64_t writeInstruction(void* whereTo) const               override;
    virtual void     prepareFieldInfos()                                 override;
    inline deviceAddrOffset getTargetAddr() {return  m_addrOffset;}

protected:
    deviceAddrOffset                 m_addrOffset;
    uint64_t                         m_transferSize;
    packet_cp_dma                    m_binary;
};

class WriteRegister : public GaudiQueueCommand
{
public:
    WriteRegister(unsigned regOffset, unsigned value, uint32_t predicate = DEFAULT_PREDICATE);
    WriteRegister(unsigned regOffset, unsigned value, uint64_t commandId, uint32_t predicate = DEFAULT_PREDICATE);
    virtual ~WriteRegister() = default;

    virtual void     Print() const override;
    virtual unsigned GetBinarySize() const                               override;
    virtual uint64_t writeInstruction(void* whereTo) const               override;
    virtual void     prepareFieldInfos()                                 override;
    virtual unsigned getRegOffset() const;
    // For testing
    const packet_wreg32 getPacket() const { return m_binary; }

protected:
    void fillBinary(unsigned regOffset, unsigned value, uint32_t predicate);
    packet_wreg32 m_binary;
};

class EbPadding : public GaudiQueueCommand
{
public:
    EbPadding(unsigned numPadding);
    virtual ~EbPadding() = default;
    virtual void     Print() const override;
    virtual void     prepareFieldInfos() override;
    virtual unsigned GetBinarySize() const override;
    virtual uint64_t writeInstruction(void* whereTo) const override;

    // For testing
    virtual uint32_t getValue() const;
    virtual void     setValue(uint32_t val);
    // called by other commands
    virtual unsigned getRegOffset() const;

protected:
    packet_wreg32 m_binary;
    unsigned      m_numPadding;

    virtual void fillBinary(unsigned regOffset);
    unsigned     getEbPaddingNumPadding() const;
};

class WriteManyRegisters : public GaudiQueueCommand
{
public:
    WriteManyRegisters(unsigned firstRegOffset, unsigned count, const uint32_t* values, uint32_t predicate = DEFAULT_PREDICATE);
    virtual ~WriteManyRegisters();

    virtual void     Print() const override;
    virtual unsigned GetBinarySize() const                               override;
    virtual uint64_t writeInstruction(void* whereTo) const               override;
    virtual void     prepareFieldInfos()                                 override;

    virtual unsigned GetFirstReg() const;
    virtual unsigned GetCount();
protected:
    void             prepareFieldInfos(BasicFieldInfoSet& basicFieldsInfoSet);

    WriteRegister*                      m_alignmentReg;  // alignment, in case the offset is not aligned to 8 bytes
    WriteRegister*                      m_remainderReg;   // remainder, in case the bulk has odd number of registers

    packet_wreg_bulk                    m_writeBulkBinary;  // Not including the values
    std::vector<uint64_t>               m_valuesBinary;

private:
    bool                                m_incZeroOffset;    // Tells if offset 0 should be updated
    unsigned                            m_incOffsetValue;   // The offset to update the patching point (num of headers)
};

class LoadDesc : public WriteManyRegisters
{
public:
    LoadDesc(void*             desc,
             unsigned          descSize,
             unsigned          descOffset,
             HabanaDeviceType  device,
             unsigned          deviceID = 0,
             uint32_t          predicate = DEFAULT_PREDICATE);

    virtual ~LoadDesc();

    virtual void Print() const override;

protected:
    //For the debug print
    HabanaDeviceType m_deviceType;
    unsigned         m_deviceID;
};

class Execute : public WriteRegister
{
public:
    Execute(HabanaDeviceType type, unsigned deviceID = 0, uint32_t predicate = DEFAULT_PREDICATE, uint32_t value = 0x1);

    virtual void Print() const override;

protected:
    //For the debug print
    HabanaDeviceType m_deviceType;
    unsigned         m_deviceID;
};

class ExecuteDmaDesc : public WriteRegister
{
public:
    ExecuteDmaDesc(uint32_t         bits,
                   HabanaDeviceType type,
                   unsigned         deviceID      = 0,
                   bool             setEngBarrier = false,
                   uint32_t         predicate     = DEFAULT_PREDICATE,
                   uint8_t          ctxIdHi       = 0);
    virtual ~ExecuteDmaDesc();

    virtual void Print() const override;

protected:
    //For the debug print
    HabanaDeviceType      m_deviceType;
    unsigned              m_deviceID;
    dma_core::reg_commit  m_commit;
    uint8_t               m_ctxIdHi;
};

class Nop : public GaudiQueueCommand
{
public:
    Nop();
    virtual ~Nop() = default;

    virtual void     Print() const override;
    virtual unsigned GetBinarySize() const                 override;
    virtual uint64_t writeInstruction(void* whereTo) const override;
    virtual void     prepareFieldInfos()                   override;

protected:
    packet_nop m_binary;

};

class Wait : public GaudiQueueCommand
{
public:
    Wait(WaitID waitID, unsigned int waitCycles, unsigned int incrementValue);
    virtual ~Wait();

    virtual void     Print() const override;
    virtual unsigned GetBinarySize() const                 override;
    virtual uint64_t writeInstruction(void* whereTo) const override;
    virtual void     prepareFieldInfos()                   override;

protected:
    packet_wait m_binary;
};

class Fence : public GaudiQueueCommand
{
public:
    Fence(WaitID waitID, unsigned int targetValue);
    virtual ~Fence();

    virtual void     Print() const override;
    virtual unsigned GetBinarySize() const                 override;
    virtual uint64_t writeInstruction(void* whereTo) const override;
    virtual void     prepareFieldInfos()                   override;
    unsigned int     m_targetValue;

protected:
    std::vector<packet_fence> m_binaries;
};

class Suspend : public GaudiQueueCommand
{
public:
    Suspend(WaitID waitID, unsigned int waitCycles, unsigned int incrementValue = 1);
    virtual ~Suspend();

    virtual void     Print() const override;
    virtual unsigned GetBinarySize() const override;
    virtual uint64_t writeInstruction(void* whereTo) const override;
    virtual void     prepareFieldInfos() override;

protected:
    Wait     m_wait;
    Fence    m_fence;

    WaitID       m_waitID;
    unsigned int m_incrementValue;
    unsigned int m_waitCycles;
};

class MonitorSetup : public GaudiQueueCommand
{
public:
    MonitorSetup(SyncObjectManager::SyncId mon,
                 WaitID                    waitID,
                 HabanaDeviceType          device,
                 unsigned                  deviceID,
                 uint32_t                  syncValue,
                 unsigned                  streamID,
                 uint32_t                  predicate     = DEFAULT_PREDICATE,
                 bool                      incSyncObject = false);

    MonitorSetup(SyncObjectManager::SyncId mon,
                 SyncObjectManager::SyncId syncId,
                 uint32_t                  syncValue,
                 uint32_t                  predicate     = DEFAULT_PREDICATE,
                 bool                      incSyncObject = false);

    virtual ~MonitorSetup() = default;

    virtual void     Print() const override;
    virtual unsigned GetBinarySize() const override;
    virtual uint64_t writeInstruction(void* whereTo) const override;
    virtual void     prepareFieldInfos() override;

protected:
    void             prepareFieldInfos(BasicFieldInfoSet& basicFieldInfoSet);

    static const unsigned m_numOfPackets = 3;

    SyncObjectManager::SyncId  m_mon;
    packet_msg_short           m_msBinaries[m_numOfPackets]; // to be used without predicate
    packet_msg_long            m_mlBinaries[m_numOfPackets]; // to be used with predicate
    uint32_t                   m_predicate;

    void makeMonitorSetupBinaryMsgShort(uint64_t address, uint32_t value);
    void makeMonitorSetupBinaryMsgLong(uint64_t address, uint32_t value);
};

class MonitorArm : public GaudiQueueCommand
{
public:
    MonitorArm(SyncObjectManager::SyncId       syncObj,
               SyncObjectManager::SyncId       mon,
               MonitorOp                       operation,
               unsigned                        syncValue,
               Settable<uint8_t>               mask);
    virtual ~MonitorArm() = default;

    virtual void     Print() const override;
    virtual unsigned GetBinarySize() const                               override;
    virtual uint64_t writeInstruction(void* whereTo) const               override;
    virtual void     prepareFieldInfos()                                 override;
    virtual bool     isMonitorArm() const override { return true; }

protected:
    SyncObjectManager::SyncId                           m_mon;
    unsigned                                            m_syncValue;
    MonitorOp                                           m_operation;
    SyncObjectManager::SyncId                           m_syncObj;
    Settable<uint8_t>                                   m_mask;

private:
    packet_msg_short m_binary;
};

class WaitForSemaphore : public GaudiQueueCommand
{
public:
    WaitForSemaphore(SyncObjectManager::SyncId       syncObj,
                     SyncObjectManager::SyncId       mon,
                     MonitorOp                       operation,
                     unsigned                        syncValue,
                     Settable<uint8_t>               mask,
                     WaitID                          waitID,
                     unsigned int                    fenceValue);
    virtual ~WaitForSemaphore();

    virtual void     Print() const override;
    virtual unsigned GetBinarySize() const                               override;
    virtual uint64_t writeInstruction(void* whereTo) const               override;
    virtual void     prepareFieldInfos()                                 override;

protected:
    MonitorArm                      m_monitorArm;
    Fence                           m_fence;
    unsigned                        m_mon;
    SyncObjectManager::SyncId       m_syncObj;
    MonitorOp                       m_operation;
    unsigned                        m_syncValue;
    Settable<uint8_t>               m_mask;
    WaitID                          m_waitID;
};


class SignalSemaphore : public GaudiQueueCommand
{
public:
    SignalSemaphore(SyncObjectManager::SyncId which,
                    int16_t                   syncValue,
                    int                       operation = 0,
                    int                       barriers  = ALL_BARRIERS);
    virtual ~SignalSemaphore() = default;

    virtual void     Print() const override;
    virtual unsigned GetBinarySize() const                               override;
    virtual uint64_t writeInstruction(void* whereTo) const               override;
protected:
    int16_t                    m_syncValue;
    int                        m_operation;
private:
    packet_msg_short m_binary;
};

class InvalidateTPCCaches : public WriteRegister
{
public:
    InvalidateTPCCaches(uint32_t predicate = DEFAULT_PREDICATE);
    virtual ~InvalidateTPCCaches();

    virtual void Print() const override;
};

class UploadKernelsAddr : public GaudiQueueCommand
{
public:
    UploadKernelsAddr(uint32_t low, uint32_t high, uint32_t predicate = DEFAULT_PREDICATE);
    virtual ~UploadKernelsAddr();

    virtual void     Print() const override;
    virtual unsigned GetBinarySize() const override;
    virtual uint64_t writeInstruction(void* whereTo) const override;
    virtual void     prepareFieldInfos() override;

private:
    void             prepareFieldInfos(BasicFieldInfoSet& basicFieldsInfoSet);

    uint32_t m_highAddress;
    uint32_t m_lowAddress;
    uint32_t m_predicate;

    packet_wreg32 m_binaries[3];
};

class MsgLong : public GaudiQueueCommand
{
public:
    MsgLong();
    virtual ~MsgLong() = default;

    virtual unsigned GetBinarySize() const                               override;
    virtual uint64_t writeInstruction(void* whereTo) const               override;
    virtual void     prepareFieldInfos()                                 override;

protected:
    packet_msg_long m_binary;
};

class ResetSyncObject : public MsgLong
{
public:
    ResetSyncObject(unsigned syncID, bool logLevelTrace = false, uint32_t predicate = DEFAULT_PREDICATE);
    virtual ~ResetSyncObject();

    virtual void Print() const override;

protected:
    //For the debug print
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

class LoadPredicates : public GaudiQueueCommand
{
public:
    LoadPredicates(deviceAddrOffset src, uint32_t predicate = DEFAULT_PREDICATE);
    virtual ~LoadPredicates() = default;

    virtual void     Print() const override;
    virtual unsigned GetBinarySize() const                               override;
    virtual uint64_t writeInstruction(void* whereTo) const               override;
    virtual void     prepareFieldInfos()                                 override;
    uint64_t         getSrcAddrForTesting() const;

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
    DynamicExecute(std::vector<std::shared_ptr<GaudiQueueCommand>> commands, BypassType enableBypass);
    virtual void prepareFieldInfos() override;

private:

    void prepareFieldInfoNoSignal(const DynamicShapeFieldInfoSharedPtr& fieldInfo);
    void prepareFieldInfoSignalOnce(const DynamicShapeFieldInfoSharedPtr& fieldInfo);
    void prepareFieldInfoSignalMME(const DynamicShapeFieldInfoSharedPtr& fieldInfo);

    void initMetaData(dynamic_execution_sm_params_t& metadata, size_t cmdLen);
    void updateFieldInfo(const DynamicShapeFieldInfoSharedPtr& fieldInfo, dynamic_execution_sm_params_t& metadata);

    BypassType m_enableBypass;
};

class SetupAndArm : public CompositeQueueCommand
{
public:
    SetupAndArm(std::vector<std::shared_ptr<GaudiQueueCommand>> commands);
    virtual void                            prepareFieldInfos() override;
    virtual bool                            isMonitorArm() const override { return true; }
    virtual const BasicFieldsContainerInfo& getBasicFieldsContainerInfo() const override;
};

}  // namespace gaudi
