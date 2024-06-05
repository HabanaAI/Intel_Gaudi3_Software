#pragma once

#include "hw_stream_parser_helper.hpp"

#include "runtime/qman/common/parser/arb_master_parser_helper.hpp"

namespace gaudi
{
class ArbMasterParserHelper
: public common::ArbMasterParserHelper
, public gaudi::HwStreamParserHelper
{
public:
    ArbMasterParserHelper(InflightCsParserHelper& parserHelper);

    virtual ~ArbMasterParserHelper() = default;

protected:
    uint64_t getStreamPhysicalOffset() { return gaudi::HwStreamParserHelper::getStreamPhysicalOffset(); };
};
}  // namespace gaudi