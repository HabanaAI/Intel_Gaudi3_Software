#include "arb_master_parser_helper.hpp"

#include "runtime/qman/gaudi/master_qmans_definition.hpp"
#include "platform/gaudi/utils.hpp"

#include "gaudi/gaudi_packets.h"
#include "drm/habanalabs_accel.h"

using namespace gaudi;

ArbMasterParserHelper::ArbMasterParserHelper(InflightCsParserHelper& parserHelper)
: common::HwStreamParserHelper(PHWST_ARB_MASTER, parserHelper),
  common::ArbMasterParserHelper(parserHelper),
  gaudi::HwStreamParserHelper(PHWST_ARB_MASTER, parserHelper)
{
    uint64_t streamMasterStreamId =
        gaudi::QmansDefinition::getInstance()->getStreamMasterQueueIdForCompute() + getStreamPhysicalOffset();

    gaudi::HwStreamParserHelper::m_fenceClearExpexctedAddress =
        gaudi::getCPFenceOffset((gaudi_queue_id)streamMasterStreamId, GC_FENCE_WAIT_ID);
}