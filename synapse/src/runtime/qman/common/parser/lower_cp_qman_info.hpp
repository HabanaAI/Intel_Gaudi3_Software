#pragma once

#include "qman_cp_info.hpp"

#include <limits>

namespace common
{
class LowerCpQmanInfo : virtual public QmanCpInfo
{
public:
    LowerCpQmanInfo() : QmanCpInfo() {};

    virtual ~LowerCpQmanInfo() override {};

    virtual void reset(uint64_t cpDmaPacketIndex = std::numeric_limits<uint64_t>::max());

    virtual void printStartParsing() override;

    virtual bool isValidPacket(uint64_t packetId) = 0;

    virtual std::string getIndentation() override;
    virtual std::string getCtrlBlockIndentation() override;

    virtual std::string getPacketIndexDesc() const override;

private:
    uint64_t m_cpDmaPacketIndex = std::numeric_limits<uint64_t>::max();

    static const unsigned INDENTATION_SIZE               = 8;
    static const unsigned CONTROL_BLOCK_INDENTATION_SIZE = INDENTATION_SIZE + 4;
};
}  // namespace common