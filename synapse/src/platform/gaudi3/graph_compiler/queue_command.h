#pragma once

#include "gaudi3/gaudi3_arc_eng_packets.h"
#include "gaudi3/gaudi3_packets.h"
#include "hal_reader/gaudi3/hal.h"
#include "platform/common/queue_command.h"
#include "types.h"
#include "utils.h"

#include <cstdint>

namespace gaudi3
{
void setSendSyncEvents(uint32_t& raw);

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

class Wait : public WaitCommon
{
public:
    Wait(WaitID id, unsigned int waitCycles, unsigned int incrementValue);
    virtual ~Wait() = default;

protected:
    packet_wait m_binary;

    virtual uint8_t   getId() const override;
    virtual uint8_t   getIncVal() const override;
    virtual uint32_t  getNumCyclesToWait() const override;
    virtual uint32_t  getWaitPacketSize() const override;
    virtual uint32_t* getWaitPacketAddr() const override;
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
    uint64_t         writeInstruction(void* whereTo) const override;

    virtual unsigned  getRegOffset() const override;
    virtual unsigned  GetBinarySize() const override;
    virtual uint32_t* getEbPaddingPacketAddr() const override;

protected:
    packet_wreg32 m_binary;

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

class ArcExeWdTpc : public ArcExeWdTpcCommon
{
public:
    ArcExeWdTpc() = default;
    virtual ~ArcExeWdTpc() = default;

    virtual void         Print() const override;
    void                 addCtx(const tpc_wd_ctxt_t& ctx);
    unsigned             getNumCtxs() const { return m_ctxCount; }
    const tpc_wd_ctxt_t& getFwCtxForTest() const { return m_ctxs[0]; }
    virtual void         prepareFieldInfos() override {}

protected:
    tpc_wd_ctxt_t m_ctxs[gaudi3::halFullChipSpecificInfo.numDcores] = {0};
    unsigned      m_ctxCount = 0;

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
    virtual void          prepareFieldInfos() override {}

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

class InvalidateTPCCaches : public WriteRegister
{
public:
    InvalidateTPCCaches(uint32_t predicate = DEFAULT_PREDICATE);
    static uint32_t calcTpcCmdVal();

    virtual ~InvalidateTPCCaches() = default;

    virtual void Print() const override;
};

class UploadKernelsAddr : public UploadKernelsAddrCommon
{
public:
    UploadKernelsAddr(uint32_t low, uint32_t high, uint32_t predicate = DEFAULT_PREDICATE);
    virtual ~UploadKernelsAddr() = default;
    virtual unsigned GetBinarySize() const override;
    uint64_t         writeInstruction(void* whereTo) const override;
    virtual void     prepareFieldInfos() override;

protected:
    packet_wreg32                    m_binaries[m_numOfPackets];
    std::shared_ptr<EbPaddingCommon> m_ebPadding;

    virtual std::shared_ptr<EbPaddingCommon> createEbPadding(unsigned numPadding) const;
    virtual void                             Print() const override;

    virtual uint8_t   getSwtcByIndex(uint32_t idx) const override;
    virtual uint32_t* getUploadKernelPacketAddr() const override;
    virtual uint16_t  getAddrOfTpcBlockField(std::string_view name) const override;

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

class QmanDelay : public QmanDelayCommon
{
public:
    QmanDelay(uint32_t predicate = DEFAULT_PREDICATE);
    virtual ~QmanDelay() = default;
};

// McidRollover is a virtual command; meaning, it encapsulates no QMAN command rather it produces ECB command
class McidRollover : public QueueCommandCommon
{
public:
    // Constructor's input params are the target SOB value just before the mcid rollover. The targetXps is needed
    // to support the rollover of transpose engine which is embedded inside the MME block.
    McidRollover(unsigned target, unsigned targetXps = 0);
    virtual ~McidRollover() = default;

    virtual void     Print() const override;
    virtual unsigned GetBinarySize() const override;
    virtual uint64_t writeInstruction(void* whereTo) const override;

    virtual void setSwitchCQ() override;
    virtual void resetSwitchCQ() override;
    virtual void toggleSwitchCQ() override;
    virtual bool isSwitchCQ() const override;

    unsigned getTarget() const { return m_target; }
    unsigned getTargetXps() const { return m_targetXps; }

    void setTarget(unsigned v) { m_target = v; }
    void setTargetXps(unsigned v) { m_targetXps = v; }

protected:
    virtual bool getSwtc() const { return m_switchBit; }
    virtual void setSwtc(bool val) { m_switchBit = val; }

private:
    bool     m_switchBit;
    unsigned m_target;
    unsigned m_targetXps;
};

}  // namespace gaudi3
