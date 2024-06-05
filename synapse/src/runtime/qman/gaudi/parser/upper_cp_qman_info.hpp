#pragma once

#include "qman_cp_info.hpp"

#include "runtime/qman/common/parser/upper_cp_qman_info.hpp"

namespace gaudi
{
class UpperCpQmanInfo
: public common::UpperCpQmanInfo
, public gaudi::QmanCpInfo
{
public:
    UpperCpQmanInfo() : common::UpperCpQmanInfo(), gaudi::QmanCpInfo() {};

    virtual ~UpperCpQmanInfo() = default;

    virtual bool parseArbPoint() override;
    virtual bool parseCpDma() override;

    // Resolving ambiguity
    virtual bool isValidPacket(uint64_t packetId) override { return common::UpperCpQmanInfo::isValidPacket(packetId); };
    virtual std::string getIndentation() override { return common::UpperCpQmanInfo::getIndentation(); };
    virtual std::string getCtrlBlockIndentation() override
    {
        return common::UpperCpQmanInfo::getCtrlBlockIndentation();
    };
};
}  // namespace gaudi