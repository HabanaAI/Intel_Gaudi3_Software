#include "stream_maste_parser_helper.hpp"

#include "drm/habanalabs_accel.h"

using namespace common;

StreamMasterParserHelper::StreamMasterParserHelper(InflightCsParserHelper& parserHelper)
: HwStreamParserHelper(PHWST_STREAM_MASTER, parserHelper)
{
}

eCpParsingState StreamMasterParserHelper::getFirstState()
{
    return CP_PARSING_STATE_FENCE_CLEAR;
}

eCpParsingState StreamMasterParserHelper::getNextState()
{
    switch (m_state)
    {
        case CP_PARSING_STATE_FENCE_CLEAR:
            return CP_PARSING_STATE_FENCE_SET;

        case CP_PARSING_STATE_FENCE_SET:
            return CP_PARSING_STATE_COMPLETED;

        default:
            return CP_PARSING_STATE_INVALID;
    }

    return CP_PARSING_STATE_INVALID;
}