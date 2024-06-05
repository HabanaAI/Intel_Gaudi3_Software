#include "stream_maste_parser_helper.hpp"

#include "runtime/qman/gaudi/master_qmans_definition.hpp"
#include "platform/gaudi/utils.hpp"

#include "drm/habanalabs_accel.h"

using namespace gaudi;

StreamMasterParserHelper::StreamMasterParserHelper(InflightCsParserHelper& parserHelper)
: common::HwStreamParserHelper(PHWST_STREAM_MASTER, parserHelper),
  common::StreamMasterParserHelper(parserHelper),
  gaudi::HwStreamParserHelper(PHWST_STREAM_MASTER, parserHelper)
{
    uint64_t arbMasterStreamId =
        gaudi::QmansDefinition::getInstance()->getArbitratorMasterQueueIdForCompute() + getStreamPhysicalOffset();

    gaudi::HwStreamParserHelper::m_fenceClearExpexctedAddress =
        gaudi::getCPFenceOffset((gaudi_queue_id)arbMasterStreamId, GC_FENCE_WAIT_ID);
}