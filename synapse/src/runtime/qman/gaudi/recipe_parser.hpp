#pragma once

#include "hal_reader/hal_reader.h"
#include "runtime/qman/common/inflight_cs_parser.hpp"

#include <memory>

namespace gaudi
{
class InflightCsParser : public ::InflightCsParser
{
public:
    InflightCsParser();
    virtual ~InflightCsParser() = default;

    virtual bool parse(InflightCsParserHelper& parserHelper) override;

private:
    std::shared_ptr<HalReader> m_pHalReader;
};
}  // namespace gaudi