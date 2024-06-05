#pragma once

#include "define.hpp"
#include "lower_cp_qman_info.hpp"
#include "upper_cp_qman_info.hpp"

#include "runtime/qman/common/inflight_cs_parser.hpp"

#include "runtime/qman/common/parser/hw_stream_parser_helper.hpp"

#include "sync/sync_types.h"

// Following helpers will define specific methods & information to parse each of the unique Upper-CP buffer
// Hence, we will define a base class and three successors - for Stream-Master, ARB-Master and ARB-Slaves
namespace gaudi
{
class HwStreamParserHelper : virtual public common::HwStreamParserHelper
{
public:
    HwStreamParserHelper(eParserHwStreamType queueType, InflightCsParserHelper& parserHelper)
    : common::HwStreamParserHelper(queueType, parserHelper) {};

    virtual ~HwStreamParserHelper() = default;

    virtual uint32_t getQmansIndex() override;

    // Retrieves CP-Index
    virtual uint32_t getUpperCpIndex() override;
    virtual uint32_t getLowerCpIndex() override;

protected:
    bool isCurrentUpperCpPacketCpDma() const;
    bool isCurrentUpperCpPacketArbPoint() const;

    virtual uint64_t getExpectedPacketForFenceClearState() override;
    virtual uint64_t getExpectedPacketForFenceSetState() override;
    virtual uint64_t getExpectedPacketForArbRequestState() override;

    virtual common::UpperCpQmanInfo* getUpperCpInfo() override { return &m_upperCpInfo; };
    virtual common::LowerCpQmanInfo* getLowerCpInfo() override { return &m_lowerCpInfo; };

    virtual std::string getPacketName(uint64_t packetId) override { return QmanCpInfo::getPacketName(packetId); };

    gaudi::UpperCpQmanInfo m_upperCpInfo;
    gaudi::LowerCpQmanInfo m_lowerCpInfo;
};
}  // namespace gaudi