#include "arb_master_parser_helper.hpp"

#include "synapse_runtime_logging.h"

#include "drm/habanalabs_accel.h"

using namespace common;

ArbMasterParserHelper::ArbMasterParserHelper(InflightCsParserHelper& parserHelper)
: HwStreamParserHelper(PHWST_ARB_MASTER, parserHelper)
{
}

eCpParsingState ArbMasterParserHelper::getFirstState()
{
    return CP_PARSING_STATE_FENCE_SET;
}

eCpParsingState ArbMasterParserHelper::getNextState()
{
    switch (m_state)
    {
        case CP_PARSING_STATE_FENCE_SET:
            return CP_PARSING_STATE_ARB_REQUEST;

        case CP_PARSING_STATE_ARB_REQUEST:
            return CP_PARSING_STATE_CP_DMAS;

        // Next-state is handled internally
        case CP_PARSING_STATE_CP_DMAS:
            return CP_PARSING_STATE_CP_DMAS;

        // Next-state is handled internally
        case CP_PARSING_STATE_WORK_COMPLETION:
            return CP_PARSING_STATE_WORK_COMPLETION;

        case CP_PARSING_STATE_ARB_RELEASE:
            return CP_PARSING_STATE_FENCE_CLEAR;

        case CP_PARSING_STATE_FENCE_CLEAR:
            return CP_PARSING_STATE_COMPLETED;

        default:
            return CP_PARSING_STATE_INVALID;
    }

    return CP_PARSING_STATE_INVALID;
}

bool ArbMasterParserHelper::handleCpDmaState()
{
    // We expect numrous CP-DMA followed by Work-Completion, and then ARB-Release

    if (isCurrentUpperCpPacketCpDma())
    {
        parseLowerCpBuffer(false);
    }
    else  // Any other packet is allowed
    {
        m_state = CP_PARSING_STATE_WORK_COMPLETION;
    }

    return true;
}

bool ArbMasterParserHelper::handleWorkCompletionState()
{
    if (isCurrentUpperCpPacketCpDma())
    {
        LOG_GCP_FAILURE("Invalid packet (CP-DMA) found, during work-completion");

        return false;
    }
    else if (isCurrentUpperCpPacketArbPoint())
    {
        if (!getUpperCpInfo()->isArbRelease())
        {
            LOG_GCP_FAILURE("Invalid ARB-packet found Request instead of Release");

            return false;
        }

        m_state = CP_PARSING_STATE_ARB_RELEASE;
    }
    // any other packet is allowed, as part of the Work-Completion

    return true;
}