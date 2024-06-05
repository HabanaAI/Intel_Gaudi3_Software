#pragma once

#include "hw_stream_parser_helper.hpp"

#include "define.hpp"

namespace common
{
class ArbSlaveParserHelper : virtual public HwStreamParserHelper
{
public:
    ArbSlaveParserHelper(InflightCsParserHelper& parserHelper);

    virtual ~ArbSlaveParserHelper() override {};

    virtual eCpParsingState getFirstState() override;
    virtual eCpParsingState getNextState() override;

    virtual bool handleFenceClearState() override { return false; };
    virtual bool handleFenceState() override { return false; };

    virtual bool handleCpDmaState() override;
};
}  // namespace common