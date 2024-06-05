#pragma once

#include <cstdint>
#include "types.h"
#include "utils.h"
#include "habana_device_types.h"
#include "../../graph_compiler/queue_command.h"
#include "graph_compiler/params_file_manager.h"
#include "habana_global_conf.h"

static const unsigned DEFAULT_PREDICATE = 0;

// Common class for all queue commands of Gaudi2 and Gaudi3
class QueueCommandCommon : public QueueCommand
{
public:
    virtual ~QueueCommandCommon();
    virtual void WritePB(gc_recipe::generic_packets_container* pktCon) override;
    virtual void WritePB(gc_recipe::generic_packets_container* pktCon, ParamsManager* params) override;

protected:
    QueueCommandCommon();
    QueueCommandCommon(uint32_t packetType);
    QueueCommandCommon(uint32_t packetType, uint64_t commandId);

private:
    QueueCommandCommon(const QueueCommandCommon&) = delete;
    void operator=(const QueueCommandCommon&) = delete;
};

// Nop command for Gaudi2 and Gaudi3
class NopCommon : public QueueCommandCommon
{
public:
    virtual ~NopCommon() = default;

    virtual void     Print() const override;
    virtual unsigned GetBinarySize() const override;
    virtual uint64_t writeInstruction(void* whereTo) const override;
    virtual void     prepareFieldInfos() override;

    virtual void setSwitchCQ() override;
    virtual void resetSwitchCQ() override;
    virtual void toggleSwitchCQ() override;
    virtual bool isSwitchCQ() const override;

protected:
    NopCommon() {};

    virtual uint8_t   getSwtc() const          = 0;
    virtual void      setSwtc(uint8_t val)     = 0;
    virtual uint32_t  getNopPacketSize() const = 0;
    virtual uint32_t* getNopPacketAddr() const = 0;
};

// Wait command for Gaudi2 and Gaudi3
class WaitCommon : public QueueCommandCommon
{
public:
    virtual ~WaitCommon() = default;

    virtual void     Print() const override;
    virtual unsigned GetBinarySize() const override;
    virtual uint64_t writeInstruction(void* whereTo) const override;
    virtual void     prepareFieldInfos() override;

    // Wait has no switch bit
    virtual void setSwitchCQ() override {}
    virtual void resetSwitchCQ() override {}
    virtual void toggleSwitchCQ() override {}
    virtual bool isSwitchCQ() const override { return false; }

protected:
    WaitCommon() {};

    virtual uint8_t   getId() const              = 0;
    virtual uint8_t   getIncVal() const          = 0;
    virtual uint32_t  getNumCyclesToWait() const = 0;
    virtual uint32_t  getWaitPacketSize() const  = 0;
    virtual uint32_t* getWaitPacketAddr() const  = 0;
};

class WriteRegisterCommon : public QueueCommandCommon
{
public:
    WriteRegisterCommon();
    WriteRegisterCommon(uint64_t commandId);
    virtual ~WriteRegisterCommon() = default;
    virtual void     Print() const override;
    virtual void     prepareFieldInfos() override;
    virtual unsigned GetBinarySize() const override;
    virtual uint64_t writeInstruction(void* whereTo) const override;
    virtual void     setSwitchCQ() override;
    virtual void     resetSwitchCQ() override;
    virtual void     toggleSwitchCQ() override;
    virtual bool     isSwitchCQ() const override;

    // For testing
    virtual uint32_t getValue() const       = 0;
    virtual void     setValue(uint32_t val) = 0;
    // called by other commands
    virtual unsigned getRegOffset() const = 0;

protected:
    virtual void      fillPackets(unsigned regOffset, unsigned value, uint32_t predicate);
    virtual unsigned  getPred() const                    = 0;
    virtual unsigned  getReg() const                     = 0;
    virtual uint8_t   getSwtc() const                    = 0;
    virtual uint32_t  getWriteRegisterPacketSize() const = 0;
    virtual uint32_t* getWriteRegisterPacketAddr() const = 0;
    virtual void      setPred(uint32_t predicate)        = 0;
    virtual void      setRegOffset(unsigned regOffset)   = 0;
    virtual void      setSwtc(uint8_t val)               = 0;
    virtual void      setReg(uint8_t reg)                = 0;
    virtual void      setEngBarrier(uint8_t value)       = 0;
    virtual void      setMsgBarrier(uint8_t value)       = 0;
};

class EbPaddingCommon : public QueueCommandCommon
{
public:
    EbPaddingCommon();
    EbPaddingCommon(unsigned numPadding);
    virtual ~EbPaddingCommon() = default;
    virtual void     Print() const override;
    virtual void     prepareFieldInfos() override;
    virtual unsigned GetBinarySize() const override                 = 0;
    virtual uint64_t writeInstruction(void* whereTo) const override = 0;
    virtual void     setSwitchCQ() override;
    virtual void     resetSwitchCQ() override;
    virtual void     toggleSwitchCQ() override;
    virtual bool     isSwitchCQ() const override;
    virtual unsigned getEbPaddingNumPadding() const;

    // For testing
    virtual uint32_t getValue() const       = 0;
    virtual void     setValue(uint32_t val) = 0;
    // called by other commands
    virtual unsigned getRegOffset() const = 0;

protected:
    unsigned          m_numPadding;
    virtual void      fillPackets(unsigned regOffset);
    virtual unsigned  getPred() const                  = 0;
    virtual unsigned  getReg() const                   = 0;
    virtual uint8_t   getSwtc() const                  = 0;
    virtual uint32_t* getEbPaddingPacketAddr() const   = 0;
    virtual unsigned  getOpcode() const                = 0;
    virtual void      setRegOffset(unsigned regOffset) = 0;
};

class WriteManyRegistersCommon : public QueueCommandCommon
{
public:
    WriteManyRegistersCommon()          = default;
    virtual ~WriteManyRegistersCommon() = default;
    virtual void Print() const override;

    virtual unsigned GetBinarySize() const override;
    virtual uint64_t writeInstruction(void* whereTo) const override;
    virtual void     prepareFieldInfos() override;
    virtual unsigned GetFirstReg() const;
    virtual unsigned GetCount() const;

    virtual void setSwitchCQ() override;
    virtual void resetSwitchCQ() override;
    virtual void toggleSwitchCQ() override;
    virtual bool isSwitchCQ() const override;
    uint32_t     getValue(unsigned i) const; // public for testing

protected:
    void fillPackets(unsigned firstRegOffset, unsigned& count32bit, uint32_t predicate, const uint32_t* values);

    void prepareFieldInfos(BasicFieldInfoSet& basicFieldsInfoSet);

    std::shared_ptr<WriteRegisterCommon> m_alignmentReg;  // alignment, in case the offset is not aligned to 8 bytes
    std::shared_ptr<WriteRegisterCommon> m_remainderReg;  // remainder, in case the bulk has odd number of registers
    std::vector<uint64_t>                m_valuesBinary;
    bool                                 m_incZeroOffset = true;  // Tells if offset 0 should be updated
    unsigned m_incOffsetValue = 0;  // The offset to update the patching point (num of headers)

    void setValue(unsigned i, uint32_t value);

    void addAddressPatchPoint(BasicFieldsContainerInfo& container,
                              uint64_t                  memId,
                              ptrToInt                  fieldAddress,
                              uint64_t                  fieldIndexOffset,
                              pNode                     node);

    virtual std::shared_ptr<WriteRegisterCommon>
    createWriteRegister(unsigned offset, unsigned value, uint32_t predicate) const = 0;

    virtual void      setSize(unsigned bulkSize)       = 0;
    virtual void      setPredicate(uint32_t predicate) = 0;
    virtual void      setOffset(unsigned offset)       = 0;
    virtual void      setWregBulkSwitchCQ(bool value)  = 0;
    virtual uint8_t   getWregBulkSwitchCQ() const      = 0;
    virtual unsigned  getWregBulkSize() const          = 0;
    virtual uint32_t* getBulkBinaryPacketAddr() const  = 0;
    virtual unsigned  getPacketWregBulkSize() const    = 0;
    virtual unsigned  getRegOffset() const             = 0;
    virtual uint32_t  getWregBulkPredicate() const     = 0;
};

class ArcExeWdTpcCommon : public QueueCommandCommon
{
public:
    virtual ~ArcExeWdTpcCommon() = default;

    virtual void     Print() const override;
    virtual uint64_t writeInstruction(void* whereTo) const override;

    virtual void setSwitchCQ() override;
    virtual void resetSwitchCQ() override;
    virtual void toggleSwitchCQ() override;
    virtual bool isSwitchCQ() const override;

protected:
    ArcExeWdTpcCommon() { m_isDynamic = true; }
    virtual unsigned    GetBinarySize() const override = 0;
    virtual uint8_t     getSwtc() const                = 0;
    virtual void        setSwtc(uint8_t val)           = 0;
    virtual const void* getArcExeWdTpcCtxAddr() const  = 0;
};

class ArcExeWdDmaCommon : public QueueCommandCommon
{
public:
    virtual ~ArcExeWdDmaCommon() = default;

    virtual void     Print() const override;
    virtual uint64_t writeInstruction(void* whereTo) const override;

    virtual void setSwitchCQ() override;
    virtual void resetSwitchCQ() override;
    virtual void toggleSwitchCQ() override;
    virtual bool isSwitchCQ() const override;

protected:
    ArcExeWdDmaCommon() { m_isDynamic = true; }
    virtual unsigned    GetBinarySize() const override = 0;
    virtual uint8_t     getSwtc() const                = 0;
    virtual void        setSwtc(uint8_t val)           = 0;
    virtual const void* getArcExeWdDmaCtxAddr() const  = 0;
};

class ArcExeWdMmeCommon : public QueueCommandCommon
{
public:
    virtual ~ArcExeWdMmeCommon() = default;
    virtual void     Print() const override;
    virtual uint64_t writeInstruction(void* whereTo) const override;

    virtual void setSwitchCQ() override;
    virtual void resetSwitchCQ() override;
    virtual void toggleSwitchCQ() override;
    virtual bool isSwitchCQ() const override;

protected:
    ArcExeWdMmeCommon() { m_isDynamic = true; }
    virtual unsigned    GetBinarySize() const override = 0;
    virtual uint8_t     getSwtc() const                = 0;
    virtual void        setSwtc(uint8_t val)           = 0;
    virtual const void* getArcExeWdMmeCtxAddr() const  = 0;
};

class ArcExeWdRotCommon : public QueueCommandCommon
{
public:
    virtual ~ArcExeWdRotCommon() = default;

    virtual void     Print() const override;
    virtual uint64_t writeInstruction(void* whereTo) const override;

    virtual void setSwitchCQ() override;
    virtual void resetSwitchCQ() override;
    virtual void toggleSwitchCQ() override;
    virtual bool isSwitchCQ() const override;

protected:
    ArcExeWdRotCommon() { m_isDynamic = true; }
    virtual unsigned    GetBinarySize() const override = 0;
    virtual uint8_t     getSwtc() const                = 0;
    virtual void        setSwtc(uint8_t val)           = 0;
    virtual uint8_t     getCplMsgEn() const            = 0;
    virtual void        setCplMsgEn(uint8_t val)       = 0;
    virtual const void* getArcExeWdRotCtxAddr() const  = 0;
};

class FenceCommon : public QueueCommandCommon
{
public:
    virtual ~FenceCommon() = default;

    virtual void     Print() const override;
    virtual unsigned GetBinarySize() const override;
    virtual uint64_t writeInstruction(void* whereTo) const override;
    virtual void     prepareFieldInfos() override;
    virtual void     setSwitchCQ() override;
    virtual void     resetSwitchCQ() override;
    virtual void     toggleSwitchCQ() override;
    virtual bool     isSwitchCQ() const override;

    unsigned getSwtc() const { return getSwtcByIndex(m_numPkts - 1); }
    unsigned getTargetVal() const { return m_targetValue; }

protected:
    uint32_t m_numPkts;
    uint32_t m_targetValue;

    FenceCommon(unsigned int targetValue);
    void fillPackets(WaitID waitID, uint32_t predicate);

    virtual uint8_t   getIdByIndex(uint32_t idx) const        = 0;
    virtual uint8_t   getSwtcByIndex(uint32_t idx) const      = 0;
    virtual uint16_t  getTargetValByIndex(uint32_t idx) const = 0;
    virtual uint8_t   getDecValByIndex(uint32_t idx) const    = 0;
    virtual uint32_t  getFencePacketSize() const              = 0;
    virtual uint32_t* getFencePacketAddr() const              = 0;

    virtual void setID(uint32_t idx, uint8_t val)         = 0;
    virtual void setSwtc(uint32_t idx, uint8_t val)       = 0;
    virtual void setPredicate(uint32_t idx, uint8_t val)  = 0;
    virtual void setTargetVal(uint32_t idx, uint16_t val) = 0;
    virtual void setDecVal(uint32_t idx, uint8_t val)     = 0;
};

class SuspendCommon : public QueueCommandCommon
{
public:
    virtual ~SuspendCommon() = default;

    virtual void     Print() const override;
    virtual unsigned GetBinarySize() const override;
    virtual uint64_t writeInstruction(void* whereTo) const override;
    virtual void     prepareFieldInfos() override;

    virtual void setSwitchCQ() override;
    virtual void resetSwitchCQ() override;
    virtual void toggleSwitchCQ() override;
    virtual bool isSwitchCQ() const override;

protected:
    std::shared_ptr<WaitCommon>  m_wait;
    std::shared_ptr<FenceCommon> m_fence;

    WaitID       m_waitID;
    unsigned int m_waitCycles;
    unsigned int m_incrementValue;

    SuspendCommon(WaitID waitID, unsigned int waitCycles, unsigned int incrementValue);
};

class WriteReg64Common : public QueueCommandCommon
{
public:
    virtual ~WriteReg64Common() = default;

    virtual unsigned GetBinarySize() const override;
    virtual uint64_t writeInstruction(void* whereTo) const override;
    virtual void     prepareFieldInfos() override;
    virtual void     Print() const override;

    virtual void setSwitchCQ() override;
    virtual void resetSwitchCQ() override;
    virtual void toggleSwitchCQ() override;
    virtual bool isSwitchCQ() const override;

    virtual uint32_t  getWriteReg64PacketSize() const = 0;
    virtual uint32_t* getWriteReg64PacketAddr() const = 0;
    virtual uint8_t   getDwEnable() const             = 0;
    virtual uint32_t  getCtl() const                  = 0;
    virtual uint32_t  getDregOffset() const           = 0;
    virtual uint32_t  getBaseIndex() const            = 0;
    virtual uint64_t  getValue() const                = 0;
    virtual unsigned  getPred() const                 = 0;
    virtual uint8_t   getSwtc() const                 = 0;

    // For testing, returns the binary as 64bit integer. In case of the long packet, return only the first 64bit.
    uint64_t getBinForTesting() const;

protected:
    WriteReg64Common() = default;
    void fillLongBinary(unsigned baseRegIndex,
                        uint64_t value,
                        unsigned targetRegisterInBytes,
                        bool     writeTargetLow,
                        bool     writeTargetHigh,
                        uint32_t predicate);

    void
    fillShortBinary(unsigned baseRegIndex, uint32_t value, unsigned targetRegisterInBytes, uint32_t predicate);
    virtual void      setPred(uint32_t predicate)       = 0;
    virtual void      setValue(uint64_t value)          = 0;
    virtual void      setDregOffset(uint32_t offset)    = 0;
    virtual void      setBaseIndex(unsigned baseIndex)  = 0;
    virtual void      setSwtc(uint8_t val)              = 0;
    virtual void      setEngBarrier(uint8_t val)        = 0;
    virtual void      setMsgBarrier(uint8_t val)        = 0;
    virtual void      setDwEnable(uint8_t val)          = 0;
    virtual void      setOpcode()                       = 0;
    virtual void      setRel(uint8_t val)               = 0;

    bool m_useLongBinary = false;
};

class UploadKernelsAddrCommon : public QueueCommandCommon
{
public:
    virtual ~UploadKernelsAddrCommon() = default;

    virtual void     Print() const override;
    virtual unsigned GetBinarySize() const override = 0;
    virtual uint64_t writeInstruction(void* whereTo) const override;
    virtual void     prepareFieldInfos() override;

    virtual void setSwitchCQ() override;
    virtual void resetSwitchCQ() override;
    virtual void toggleSwitchCQ() override;
    virtual bool isSwitchCQ() const override;

protected:
    static const unsigned m_numOfPackets = 3;

    uint32_t m_highAddress;
    uint32_t m_lowAddress;
    uint32_t m_predicate;

    UploadKernelsAddrCommon(uint32_t low, uint32_t high, uint32_t predicate, uint32_t prefetchAlignmentMask);
    void fillPackets(uint32_t regTpcCmd);

    virtual uint8_t   getSwtcByIndex(uint32_t idx) const                  = 0;
    virtual uint32_t* getUploadKernelPacketAddr() const                   = 0;
    virtual uint16_t  getAddrOfTpcBlockField(std::string_view name) const = 0;

    virtual void setPacket(uint32_t idx)                  = 0;
    virtual void setPredicate(uint32_t idx, uint8_t val)  = 0;
    virtual void setSwtc(uint32_t idx, uint8_t val)       = 0;
    virtual void setRegOffset(uint32_t idx, uint16_t val) = 0;
    virtual void setValue(uint32_t idx, uint32_t val)     = 0;
    virtual void setEngBarrier(uint32_t idx, uint8_t val) = 0;

private:
    virtual void prepareFieldInfos(BasicFieldInfoSet& basicFieldsInfoSet);
};

class MsgLongCommon : public QueueCommandCommon
{
public:
    virtual ~MsgLongCommon() = default;

    virtual unsigned GetBinarySize() const override;
    virtual uint64_t writeInstruction(void* whereTo) const override;
    virtual void     prepareFieldInfos() override;

    virtual void setSwitchCQ() override;
    virtual void resetSwitchCQ() override;
    virtual void toggleSwitchCQ() override;
    virtual bool isSwitchCQ() const override;

protected:
    MsgLongCommon() = default;

    virtual uint8_t   getSwtc() const              = 0;
    virtual void      setSwtc(uint8_t val)         = 0;
    virtual uint32_t  getMsgLongPacketSize() const = 0;
    virtual uint32_t* getMsgLongPacketAddr() const = 0;
};

class QmanDelayCommon : public QueueCommandCommon
{
public:
    virtual ~QmanDelayCommon() = default;

    virtual void     Print() const override;
    virtual unsigned GetBinarySize() const override;
    virtual uint64_t writeInstruction(void* whereTo) const override;
    virtual void     prepareFieldInfos() override;

    virtual void setSwitchCQ() override;
    virtual void resetSwitchCQ() override;
    virtual void toggleSwitchCQ() override;
    virtual bool isSwitchCQ() const override;

protected:
    std::shared_ptr<WriteRegisterCommon> m_wreg;
    std::shared_ptr<FenceCommon>         m_fence;

    QmanDelayCommon() = default;
};

// ResetSobs is a virtual command; meaning, it encapsulates no QMAN command rather it produces ECB command
class ResetSobs : public QueueCommandCommon
{
public:
    // Constructor's input params are the target SOB value just before the reset and the total number of engines
    // involved in the reset (across all engine types). The targetXps is needed for Gaudi3 to support the reset
    // of transpose engine which is embedded inside the MME block.
    ResetSobs(unsigned target, unsigned totalNumEngs, unsigned targetXps = 0);
    virtual ~ResetSobs() = default;

    virtual void     Print() const override;
    virtual unsigned GetBinarySize() const override;
    virtual uint64_t writeInstruction(void* whereTo) const override;

    virtual void setSwitchCQ() override;
    virtual void resetSwitchCQ() override;
    virtual void toggleSwitchCQ() override;
    virtual bool isSwitchCQ() const override;

    unsigned getTarget() const { return m_target; }
    unsigned getTotalNumEngs() const { return m_totalNumEngs; }
    unsigned getTargetXps() const { return m_targetXps; }

    void setTarget(unsigned v) { m_target = v; }
    void setTotalNumEngs(unsigned v) { m_totalNumEngs = v; }
    void setTargetXps(unsigned v) { m_targetXps = v; }

protected:
    virtual bool getSwtc() const;
    virtual void setSwtc(bool val);

private:
    bool     m_switchBit;
    unsigned m_target;
    unsigned m_totalNumEngs;
    unsigned m_targetXps;
};
