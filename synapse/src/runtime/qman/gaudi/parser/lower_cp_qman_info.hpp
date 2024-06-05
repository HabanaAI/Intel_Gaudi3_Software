#pragma once

#include "qman_cp_info.hpp"

#include "runtime/qman/common/parser/lower_cp_qman_info.hpp"

namespace gaudi
{
class LowerCpQmanInfo
: public common::LowerCpQmanInfo
, public gaudi::QmanCpInfo
{
public:
    LowerCpQmanInfo() : common::LowerCpQmanInfo(), gaudi::QmanCpInfo() {};

    virtual ~LowerCpQmanInfo() = default;

    virtual bool isValidPacket(uint64_t packetId) override;

    // Resolving ambiguity
    virtual std::string getIndentation() override { return common::LowerCpQmanInfo::getIndentation(); };
    virtual std::string getCtrlBlockIndentation() override
    {
        return common::LowerCpQmanInfo::getCtrlBlockIndentation();
    };
};
}  // namespace gaudi