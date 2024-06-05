#pragma once

#include "hw_stream_parser_helper.hpp"

#include "define.hpp"

namespace common
{
class StreamMasterParserHelper : virtual public HwStreamParserHelper
{
public:
    StreamMasterParserHelper(InflightCsParserHelper& parserHelper);

    virtual ~StreamMasterParserHelper() = default;

    virtual eCpParsingState getFirstState() override;
    virtual eCpParsingState getNextState() override;

    virtual bool handleArbRequestState() override { return false; };
};
}  // namespace common