#include "arb_slave_parser_helper.hpp"

#include "synapse_runtime_logging.h"

using namespace gaudi;

ArbSlaveParserHelper::ArbSlaveParserHelper(InflightCsParserHelper& parserHelper)
: common::HwStreamParserHelper(PHWST_ARB_SLAVE, parserHelper),
  common::ArbSlaveParserHelper(parserHelper),
  gaudi::HwStreamParserHelper(PHWST_ARB_SLAVE, parserHelper)
{
}