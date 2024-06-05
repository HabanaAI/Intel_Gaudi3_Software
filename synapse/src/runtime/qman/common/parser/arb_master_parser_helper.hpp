#pragma once

#include "hw_stream_parser_helper.hpp"

namespace common
{
class ArbMasterParserHelper : virtual public HwStreamParserHelper
{
public:
    ArbMasterParserHelper(InflightCsParserHelper& parserHelper);

    virtual ~ArbMasterParserHelper() = default;

    virtual eCpParsingState getFirstState() override;
    virtual eCpParsingState getNextState() override;

    virtual bool handleCpDmaState() override;
    virtual bool handleWorkCompletionState() override;
};
}  // namespace common