#pragma once

#include "hw_stream_parser_helper.hpp"

#include "runtime/qman/common/parser/arb_slave_parser_helper.hpp"

#include "define.hpp"

namespace gaudi
{
class ArbSlaveParserHelper
: public common::ArbSlaveParserHelper
, public gaudi::HwStreamParserHelper
{
public:
    ArbSlaveParserHelper(InflightCsParserHelper& parserHelper);

    virtual ~ArbSlaveParserHelper() override {};
};
}  // namespace gaudi