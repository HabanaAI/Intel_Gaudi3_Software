#include "recipe_parser.hpp"

#include "defenders.h"
#include "synapse_runtime_logging.h"

#include "platform/gaudi/graph_compiler/hal_conventions.h"

#include "runtime/qman/common/parser/sync_manager_info.hpp"

#include "runtime/qman/gaudi/parser/arb_master_parser_helper.hpp"
#include "runtime/qman/gaudi/parser/arb_slave_parser_helper.hpp"
#include "runtime/qman/gaudi/parser/hw_stream_parser_helper.hpp"
#include "runtime/qman/gaudi/parser/stream_maste_parser_helper.hpp"
#include "runtime/qman/gaudi/master_qmans_definition.hpp"

#include "drm/habanalabs_accel.h"

#include <array>

extern HalReaderPtr instantiateGaudiHalReader();

using namespace gaudi;

using ParserCsPqEntries       = InflightCsParserHelper::ParserCsPqEntries;
using ParserHwStreamCsEntries = InflightCsParserHelper::ParserHwStreamCsEntries;

static bool parseSingleEngineEntries(InflightCsParserHelper&       parserHelper,
                                     common::HwStreamParserHelper* pHwStreamHelper,
                                     ParserHwStreamCsEntries&      singleHwStreamCsEntries)
{
    CHECK_POINTER(SYN_CS_PARSER, pHwStreamHelper, "HW-Stream helper", false);

    for (auto singleCsEntry : singleHwStreamCsEntries)
    {
        uint64_t handle     = singleCsEntry.m_bufferAddress;
        uint64_t bufferSize = singleCsEntry.m_bufferSize;

        uint64_t mappedBufferSize = 0;

        uint64_t hostAddress = parserHelper.getHostAddress(mappedBufferSize, handle);
        if ((hostAddress == 0) || (bufferSize > mappedBufferSize))
        {
            LOG_GCP_FAILURE("Invalid handle 0x{:x} or invalid mapped size (buffer {} mapped {}) for HW-Queue {}",
                            handle,
                            bufferSize,
                            mappedBufferSize,
                            pHwStreamHelper->getQueueId());

            return false;
        }

        if (!pHwStreamHelper->setNewUpperCpInfo(handle, hostAddress, bufferSize))
        {
            return false;
        }

        if (!pHwStreamHelper->parseUpperCpBuffer(common::PARSING_DEFINITION_REGULAR))
        {
            return false;
        }
    }

    return pHwStreamHelper->finalize();
}

gaudi::InflightCsParser::InflightCsParser() : m_pHalReader(instantiateGaudiHalReader()) {}

bool gaudi::InflightCsParser::parse(InflightCsParserHelper& parserHelper)
{
    using SyncManagerInfoDatabase = common::SyncManagerInfoDatabase;

    const ParserCsPqEntries& hwStreamsCsEntries = parserHelper.getParserCsPqEntries();

    StreamMasterParserHelper streamMasterHelper(parserHelper);
    ArbMasterParserHelper    arbMasterHelper(parserHelper);
    ArbSlaveParserHelper     arbSlaveHelper(parserHelper);

    SyncManagerInfoDatabase syncManagersInfoDb(gaudi::SYNC_MNGR_NUM);

    uint32_t smInstanceId = gaudi::SYNC_MNGR_EAST_NORTH;
    for (auto singleSmMonitorsDb : syncManagersInfoDb)
    {
        singleSmMonitorsDb.init(smInstanceId);
        smInstanceId++;
    }

    HwStreamParserHelper* pCurrentHwStreamHelper = nullptr;

    LOG_GCP_VERBOSE("");
    LOG_GCP_VERBOSE("=== Start parsing {} HW-Stream entries ===", hwStreamsCsEntries.size());

    bool status = true;
    for (auto singleHwStreamEntry : hwStreamsCsEntries)
    {
        uint32_t hwStreamId = singleHwStreamEntry.first;
        if (hwStreamId >= GAUDI_QUEUE_ID_SIZE)
        {
            LOG_GCP_FAILURE("Invalid HW-Stream {}", hwStreamId);
            status = false;
            break;
        }

        ParserHwStreamCsEntries singleHwStreamCsEntries = singleHwStreamEntry.second;

        // Only supportiug Compute CS parsing

        if (gaudi::QmansDefinition::getInstance()->isStreamMasterQueueIdForCompute(hwStreamId))
        {
            pCurrentHwStreamHelper = &streamMasterHelper;
        }
        else if (gaudi::QmansDefinition::getInstance()->isArbMasterForComputeAndNewGaudiSyncScheme(hwStreamId))
        {
            pCurrentHwStreamHelper = &arbMasterHelper;
        }
        else if (gaudi::QmansDefinition::getInstance()->isComputeArbSlaveQueueId(hwStreamId))
        {
            pCurrentHwStreamHelper = &arbSlaveHelper;
        }
        else
        {
            LOG_GCP_FAILURE("HW-Stream {} is not a compute stream", hwStreamId);
            status = false;
            break;
        }

        // Adding phsyical offset
        uint64_t       streamPhysicalOffset = parserHelper.getStreamPhysicalOffset();
        gaudi_queue_id finalQueueId         = (gaudi_queue_id)(hwStreamId + streamPhysicalOffset);

        LOG_GCP_VERBOSE("");
        LOG_GCP_VERBOSE("Parsing {} HW-Stream ({} {})",
                        pCurrentHwStreamHelper->getStreamTypeName(),
                        finalQueueId,
                        getEngineName(finalQueueId, m_pHalReader.get()));

        if (!pCurrentHwStreamHelper->resetStreamInfo(hwStreamId, streamPhysicalOffset, &syncManagersInfoDb, true))
        {
            return false;
        }

        if (!parseSingleEngineEntries(parserHelper, pCurrentHwStreamHelper, singleHwStreamCsEntries))
        {
            return false;
        }
    }

    for (auto singleSmMonitorsDb : syncManagersInfoDb)
    {
        if (!singleSmMonitorsDb.checkMonitors())
        {
            return false;
        }
    }

    return status;
}