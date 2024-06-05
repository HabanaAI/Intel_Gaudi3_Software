#pragma once

#include "qman_cp_info.hpp"

namespace common
{
class UpperCpQmanInfo : virtual public QmanCpInfo
{
public:
    UpperCpQmanInfo() : QmanCpInfo() {};

    virtual ~UpperCpQmanInfo() = default;

    virtual void printStartParsing() override { QmanCpInfo::printStartParsing("==> Parsing Upper-CP"); };

    virtual bool isValidPacket(uint64_t packetId) override { return true; };

    virtual std::string getIndentation() override { return std::string(INDENTATION_SIZE, ' '); };
    virtual std::string getCtrlBlockIndentation() override { return std::string(CONTROL_BLOCK_INDENTATION_SIZE, ' '); };

    virtual bool getLowerCpBufferHandleAndSize(uint64_t& handle, uint64_t& size) override;

    virtual bool checkFenceClearPacket(uint64_t expectedAddress, uint16_t expectedFenceValue) const override;

    virtual bool parseArbPoint() override = 0;
    virtual bool parseCpDma() override    = 0;

    virtual std::string getPacketIndexDesc() const override;

    bool isArbRelease() const;

    uint64_t m_lowerCpBufferHandle = 0;
    uint64_t m_lowerCpBufferSize   = 0;

    bool m_isArbRelease = true;

private:
    static const unsigned INDENTATION_SIZE               = 4;
    static const unsigned CONTROL_BLOCK_INDENTATION_SIZE = INDENTATION_SIZE + 2;
};
}  // namespace common