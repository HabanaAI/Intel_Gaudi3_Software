#pragma once

#include <cstdint>

namespace CommonPktGen
{
class BasePacket
{
public:
    BasePacket() {}
    virtual ~BasePacket() {}

    virtual void     generatePacket(char*& ptr) const = 0;
    virtual uint64_t getPacketSize()                  = 0;
    virtual uint32_t getPktType()                     = 0;
};

class AddressFieldsPacket
{
public:
    AddressFieldsPacket(unsigned numOfRegisters, uint64_t registersOffset)
    : m_numOfRegisters(numOfRegisters), m_registersOffset(registersOffset) {};

    virtual ~AddressFieldsPacket() {};

    // Number of registers that resides on this packet
    inline unsigned getNumOfRegisters() { return m_numOfRegisters; }

    // The registers' location in offset from the start of the "generated-packet"
    inline uint64_t getRegistersPacketOffset() { return m_registersOffset; }

protected:
    unsigned m_numOfRegisters;
    uint64_t m_registersOffset;
};

template<class TPacket>
class BasePacketImpl : public BasePacket
{
public:
    uint64_t getPacketSize() override { return sizeof(TPacket); }
    uint32_t getPktType() override { return m_binary.opcode; }
    void     generatePacket(char*& ptr) const override
    {
        memcpy(ptr, &m_binary, sizeof(m_binary));
        ptr += sizeof(m_binary);
    }
    static uint64_t packetSize() { return sizeof(TPacket); }

protected:
    TPacket m_binary = {};
};

}  // namespace CommonPktGen
