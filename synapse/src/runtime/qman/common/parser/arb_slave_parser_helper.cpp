#include "arb_slave_parser_helper.hpp"

#include "qman_cp_info.hpp"

#include "synapse_runtime_logging.h"

#include "drm/habanalabs_accel.h"

using namespace common;

ArbSlaveParserHelper::ArbSlaveParserHelper(InflightCsParserHelper& parserHelper)
: HwStreamParserHelper(PHWST_ARB_SLAVE, parserHelper)
{
}

eCpParsingState ArbSlaveParserHelper::getFirstState()
{
    return CP_PARSING_STATE_ARB_REQUEST;
}

eCpParsingState ArbSlaveParserHelper::getNextState()
{
    switch (m_state)
    {
        case CP_PARSING_STATE_ARB_REQUEST:
            return CP_PARSING_STATE_CP_DMAS;

        // ARB-Release is internaly handled during CP_PARSING_STATE_CP_DMAS state
        case CP_PARSING_STATE_CP_DMAS:
            return CP_PARSING_STATE_CP_DMAS;

        case CP_PARSING_STATE_ARB_RELEASE:
            return CP_PARSING_STATE_COMPLETED;

        default:
            return CP_PARSING_STATE_INVALID;
    }

    return CP_PARSING_STATE_INVALID;
}

bool ArbSlaveParserHelper::handleCpDmaState()
{
    // Has there are numerous CP-DMA prior of the ARB Release,
    // We will examine both command types

    if (isCurrentUpperCpPacketCpDma())
    {
        if (!parseLowerCpBuffer(false))
        {
            return false;
        }
    }
    else if (isCurrentUpperCpPacketArbPoint())
    {
        if (!getUpperCpInfo()->isArbRelease())
        {
            LOG_GCP_FAILURE("Invalid ARB-packet found Request instead of Release");

            return false;
        }

        // Update state, as we actually handled the ARB-Release activity
        m_state = CP_PARSING_STATE_ARB_RELEASE;
    }
    else
    {
        LOG_GCP_FAILURE("Invalid packet found {} (expected CP-DMA or ARB-Point)",
                        getPacketName(getUpperCpInfo()->getCurrentPacketId()));

        return false;
    }

    return true;
}