#pragma once

#include "hw_stream_parser_helper.hpp"

#include "runtime/qman/common/inflight_cs_parser.hpp"
#include "runtime/qman/common/parser/stream_maste_parser_helper.hpp"

namespace gaudi
{
class StreamMasterParserHelper
: public common::StreamMasterParserHelper
, public gaudi::HwStreamParserHelper
{
public:
    StreamMasterParserHelper(InflightCsParserHelper& parserHelper);

    virtual ~StreamMasterParserHelper() = default;

protected:
    uint64_t getStreamPhysicalOffset() { return gaudi::HwStreamParserHelper::getStreamPhysicalOffset(); };
};
}  // namespace gaudi