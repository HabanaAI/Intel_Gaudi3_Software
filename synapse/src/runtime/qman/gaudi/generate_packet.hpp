#pragma once

#include "runtime/common/generate_packet.hpp"

#include <cstdint>
#include <cstring>
#include "gaudi/gaudi_packets.h"

// #define PACKET_GEN_ALIGNMENT_ENABLE

namespace gaudi
{
static const unsigned DEFAULT_PREDICATE = 0;

template<class TPacket>
using BasePacketImpl = CommonPktGen::BasePacketImpl<TPacket>;

using AddressFieldsPacket = CommonPktGen::AddressFieldsPacket;

class GenWreg32
: public BasePacketImpl<packet_wreg32>
, public AddressFieldsPacket
{
public:
    GenWreg32(uint32_t value,
              uint32_t regOffset,
              uint32_t engBarrier,
              uint32_t msgBarrier,
              uint32_t predicate = DEFAULT_PREDICATE);
};

class GenWregBulk
: public BasePacketImpl<packet_wreg_bulk>
, public AddressFieldsPacket
{
public:
    GenWregBulk(uint32_t  numOfBulkRegs,
                uint32_t  regOffset,
                uint32_t  engBarrier,
                uint32_t  msgBarrier,
                uint64_t* values,
                uint32_t  predicate = DEFAULT_PREDICATE);
    virtual ~GenWregBulk();

    virtual void     generatePacket(char*& ptr) const override;
    virtual uint64_t getPacketSize() override;

    static uint64_t packetSize();

private:
    // Each bulk's register is a uint64_t => two uint32_t fields
    static const unsigned BULK_REGISTERS_GRANULARITY = 2;
    uint64_t*             m_pValues;
};

class GenMsgLong : public BasePacketImpl<packet_msg_long>
{
public:
    GenMsgLong(uint32_t value,
               uint32_t op,
               uint32_t engBarrier,
               uint32_t regBarrier,
               uint32_t msgBarrier,
               uint64_t addr,
               uint32_t predicate = DEFAULT_PREDICATE);

    static void generatePacket(char*&   pPacket,
                               uint32_t value,
                               uint32_t op,
                               uint32_t engBarrier,
                               uint32_t regBarrier,
                               uint32_t msgBarrier,
                               uint64_t addr,
                               bool     shouldIncrementPointer = false,
                               uint32_t predicate              = DEFAULT_PREDICATE);
};

class GenMsgShort : public BasePacketImpl<packet_msg_short>
{
public:
    // For backward-compatibility support - will use value instead of the actual fields...
    // value is one of two structures - mon_arm_register and so_upd
    GenMsgShort(uint32_t value,
                uint32_t msgAddrOffset,
                uint32_t op,
                uint32_t base,
                uint32_t engBarrier,
                uint32_t regBarrier,
                uint32_t msgBarrier);

    // Using non uin32_t to allow this overload
    // Monitor Arm
    GenMsgShort(uint8_t  syncGroupId,
                uint8_t  syncMask,
                bool     mode,
                uint16_t syncValue,
                uint32_t msgAddrOffset,
                uint32_t op,
                uint32_t base,
                uint32_t engBarrier,
                uint32_t regBarrier,
                uint32_t msgBarrier);

    // Sync-Object Update
    GenMsgShort(uint16_t syncValue,
                bool     te,
                bool     mode,
                uint32_t msgAddrOffset,
                uint32_t op,
                uint32_t base,
                uint32_t engBarrier,
                uint32_t regBarrier,
                uint32_t msgBarrier);
};

class GenMsgProt : public BasePacketImpl<packet_msg_prot>
{
public:
    GenMsgProt(uint32_t value,
               uint32_t op,
               uint32_t engBarrier,
               uint32_t regBarrier,
               uint32_t msgBarrier,
               uint64_t address   = 0,
               uint32_t predicate = DEFAULT_PREDICATE);

};

class GenFence : public BasePacketImpl<packet_fence>
{
public:
    GenFence(uint32_t decVal,
             uint32_t targetVal,
             uint32_t id,
             uint32_t engBarrier,
             uint32_t regBarrier,
             uint32_t msgBarrier,
             uint32_t predicate = DEFAULT_PREDICATE);

    static void generatePacket(char*&   pPacket,
                               uint32_t decVal,
                               uint32_t targetVal,
                               uint32_t id,
                               uint32_t engBarrier,
                               uint32_t regBarrier,
                               uint32_t msgBarrier,
                               bool     shouldIncrementPointer = false,
                               uint32_t predicate              = DEFAULT_PREDICATE);
};

class GenLinDma : public BasePacketImpl<packet_lin_dma>
{
public:
    GenLinDma(uint32_t tsize,
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
              bool     decompress);

    uint32_t         getDirection() const;

    static void generateLinDma(char*&   pPacket,
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
                               bool     decompress);

private:
    uint32_t        m_dmaDirection;
};

class GenNop : public BasePacketImpl<packet_nop>
{
public:
    GenNop(uint32_t engBarrier, uint32_t regBarrier, uint32_t msgBarrier);

    static void generatePacket(char*&   pPacket,
                               uint32_t engBarrier,
                               uint32_t regBarrier,
                               uint32_t msgBarrier,
                               bool     shouldIncrementPointer = false);

};

class GenStop : public BasePacketImpl<packet_stop>
{
public:
    GenStop(uint32_t engBarrier);
};

class GenCpDma
: public CommonPktGen::BasePacket
, public AddressFieldsPacket
{
public:
    GenCpDma(uint32_t tsize,
             uint32_t engBarrier,
             uint32_t msgBarrier,
             uint64_t addr,
             uint32_t predicate = DEFAULT_PREDICATE);

    void generatePacket(char*& ptr) const override
    {
        memcpy(ptr, &m_binary, sizeof(m_binary));
        ptr += sizeof(m_binary);
    }
    uint64_t         getPacketSize() override { return sizeof(m_binary); }
    uint32_t         getPktType() override { return m_binary.opcode; }
    void*            getBinaryAddress();

    static void generateCpDma(char*&   ptr,
                              uint32_t tsize,
                              uint32_t engBarrier,
                              uint32_t msgBarrier,
                              uint64_t addr,
                              uint32_t predicate = DEFAULT_PREDICATE);

    static uint64_t packetSize() { return sizeof(packet_cp_dma); }

    static void generateDefaultCpDma(char*& pPacket, uint32_t tsize, uint64_t addr);

private:
#ifdef PACKET_GEN_ALIGNMENT_ENABLE
    alignas((1 << 5))
#endif
        packet_cp_dma m_binary = {};
};

class GenArbitrationPoint : public BasePacketImpl<packet_arb_point>
{
public:
    GenArbitrationPoint(uint8_t priority, bool priorityRelease, uint32_t predicate = DEFAULT_PREDICATE);

    static void generateArbitrationPoint(char*&   pPacket,
                                         uint8_t  priority,
                                         bool     priorityRelease,
                                         uint32_t predicate = DEFAULT_PREDICATE);
};

class GenRepeat : public BasePacketImpl<packet_repeat>
{
public:
    GenRepeat(bool     isRepeatStart,
              bool     isOuterLoop,
              uint16_t jumpPtr,
              uint32_t engBarrier,
              uint32_t msgBarrier,
              uint16_t predicate = DEFAULT_PREDICATE);
};

class GenWait : public BasePacketImpl<packet_wait>
{
public:
    GenWait(uint8_t incVal, uint8_t id, uint32_t numCyclesToWait, uint32_t engBarrier, uint32_t msgBarrier);

};

class GenLoadAndExecute : public BasePacketImpl<packet_load_and_exe>
{
public:
    GenLoadAndExecute(bool     isLoad,
                      bool     isDst, /* else will load predicates */
                      bool     isExecute,
                      bool     isEType,
                      uint32_t engBarrier,
                      uint32_t msgBarrier,
                      uint64_t srcAddr,
                      uint16_t predicate = DEFAULT_PREDICATE);

    static void generatePacket(char*&   pPacket,
                               bool     isLoad,
                               bool     isDst, /* else will load predicates */
                               bool     isExecute,
                               bool     isEType,
                               uint32_t engBarrier,
                               uint32_t msgBarrier,
                               uint64_t srcAddr,
                               bool     shouldIncrementPointer = false,
                               uint16_t predicate              = DEFAULT_PREDICATE);

};
}  // namespace gaudi
