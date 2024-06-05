#include "hw_stream_parser_helper.hpp"

#include "qman_cp_info.hpp"

#include "defenders.h"
#include "synapse_runtime_logging.h"

#include "drm/habanalabs_accel.h"

using namespace common;

const std::string HwStreamParserHelper::m_streamTypeName[PHWST_COUNT] = {"Stream-Master",
                                                                         "ARB-Master",
                                                                         "ARB-Slave",
                                                                         "Invalid"};

bool HwStreamParserHelper::finalize()
{
    LOG_GCP_VERBOSE("Finalizing CPs");
    return getUpperCpInfo()->finalize() && getLowerCpInfo()->finalize();
}

bool HwStreamParserHelper::parseUpperCpBuffer(eParsingDefinitions parsingDefs)
{
    if (!getUpperCpInfo()->isInitialized())
    {
        LOG_GCP_FAILURE("Parser upper-cp info is not initialized");
        return false;
    }

    if (parsingDefs == PARSING_DEFINITION_ON_HOST_LOWER_CP)
    {
        return parseLowerCpBuffer(true);
    }
    else if (parsingDefs == PARSING_DEFINITION_ON_HOST_UPPER_CP)
    {
        m_state = CP_PARSING_STATE_WORK_COMPLETION;
    }

    getUpperCpInfo()->printStartParsing();

    do
    {
        if (getUpperCpInfo()->getBufferSize() == 0)
        {
            break;
        }

        bool status = getUpperCpInfo()->parseSinglePacket(m_state);

        if (!status)
        {
            return false;
        }

        switch (m_state)
        {
            case CP_PARSING_STATE_INVALID:
            case CP_PARSING_STATE_BASIC_COMMANDS:
            {
                LOG_GCP_FAILURE("Invalid state");

                return false;
            }
            break;

            case CP_PARSING_STATE_FENCE_CLEAR:
            {
                if (!handleFenceClearState())
                {
                    return false;
                }
            }
            break;

            case CP_PARSING_STATE_FENCE_SET:
            {
                if (!handleFenceState())
                {
                    return false;
                }
            }
            break;

            case CP_PARSING_STATE_ARB_REQUEST:
            {
                if (!handleArbRequestState())
                {
                    return false;
                }
            }
            break;

            case CP_PARSING_STATE_CP_DMAS:
            {
                if (!handleCpDmaState())
                {
                    return false;
                }
            }
            break;

            case CP_PARSING_STATE_WORK_COMPLETION:
            {
                if (!handleWorkCompletionState())
                {
                    return false;
                }
            }
            break;

            // ARB-Release is internaly handled during CP_PARSING_STATE_CP_DMAS state
            case CP_PARSING_STATE_ARB_RELEASE:
            {
                LOG_GCP_FAILURE("Invalid internal state (ARB-Release)");

                return false;
            }
            break;

            case CP_PARSING_STATE_COMPLETED:
            {
                LOG_GCP_FAILURE("Redundant packet(s) found");

                return false;
            }
            break;
        }

        m_state = getNextState();
    } while (getUpperCpInfo()->getBufferSize() != 0);

    return true;
}

bool HwStreamParserHelper::parseLowerCpBuffer(bool isBufferOnHost)
{
    uint64_t lowerCpBufferHandle = 0;
    uint64_t lowerCpBufferSize   = 0;
    uint64_t hostAddress         = 0;
    uint64_t mappedBufferSize    = 0;
    uint64_t upperCpPacketIndex  = 0;

    if (!isBufferOnHost)
    {
        if (!getUpperCpInfo()->getLowerCpBufferHandleAndSize(lowerCpBufferHandle, lowerCpBufferSize))
        {
            return false;
        }

        hostAddress        = m_parserHelper.getHostAddress(mappedBufferSize, lowerCpBufferHandle);
        upperCpPacketIndex = getUpperCpInfo()->getPacketIndex();
    }
    else
    {
        hostAddress      = getUpperCpInfo()->getHostAddress();
        mappedBufferSize = getUpperCpInfo()->getBufferSize();

        lowerCpBufferHandle = hostAddress;
        lowerCpBufferSize   = mappedBufferSize;
    }
    if ((hostAddress == 0) || (lowerCpBufferSize > mappedBufferSize))
    {
        LOG_GCP_FAILURE("Invalid handle 0x{:x} or invalid mapped size (buffer {} mapped {} isBufferOnHost {})",
                        lowerCpBufferHandle,
                        lowerCpBufferSize,
                        mappedBufferSize,
                        isBufferOnHost);

        return false;
    }

    getLowerCpInfo()->reset(upperCpPacketIndex);
    bool status = getLowerCpInfo()->setNewInfo(lowerCpBufferHandle,
                                               hostAddress,
                                               lowerCpBufferSize,
                                               getLowerCpIndex(),
                                               1 << m_queuePhysicalOffset,
                                               m_pSyncManagersInfoDb);
    if (!status)
    {
        return false;
    }

    if (!getLowerCpInfo()->isInitialized())
    {
        LOG_GCP_FAILURE("Parser lower-cp info is not initialized");
        return false;
    }

    getLowerCpInfo()->printStartParsing();

    do
    {
        if (getLowerCpInfo()->getBufferSize() == 0)
        {
            break;
        }

        bool status = getLowerCpInfo()->parseSinglePacket(CP_PARSING_STATE_BASIC_COMMANDS);
        if (!status)
        {
            return false;
        }
    } while (getLowerCpInfo()->getBufferSize() != 0);

    return true;
}

bool HwStreamParserHelper::handleFenceClearState()
{
    uint64_t expectedPacketId = getExpectedPacketForFenceClearState();
    if (getUpperCpInfo()->getCurrentPacketId() != expectedPacketId)
    {
        LOG_GCP_FAILURE("Invalid packet found {} (expected {})",
                        getPacketName(getUpperCpInfo()->getCurrentPacketId()),
                        getPacketName(expectedPacketId));

        return false;
    }

    uint64_t expectedAddress         = m_fenceClearExpexctedAddress;
    uint16_t expectedFenceClearValue = 1;

    if (!getUpperCpInfo()->checkFenceClearPacket(expectedAddress, expectedFenceClearValue))
    {
        LOG_GCP_FAILURE("ARB-Master Fence-RDATA address is wrongly defined");
    }

    return true;
}

bool HwStreamParserHelper::handleFenceState()
{
    uint64_t expectedPacketId = getExpectedPacketForFenceSetState();
    if (getUpperCpInfo()->getCurrentPacketId() != expectedPacketId)
    {
        LOG_GCP_FAILURE("Invalid packet found {} (expected {})",
                        getPacketName(getUpperCpInfo()->getCurrentPacketId()),
                        getPacketName(expectedPacketId));

        return false;
    }

    return true;
}

bool HwStreamParserHelper::handleArbRequestState()
{
    uint64_t expectedPacketId = getExpectedPacketForArbRequestState();
    if (getUpperCpInfo()->getCurrentPacketId() != expectedPacketId)
    {
        LOG_GCP_FAILURE("Invalid packet found {} (expected {})",
                        getPacketName(getUpperCpInfo()->getCurrentPacketId()),
                        getPacketName(expectedPacketId));

        return false;
    }

    if (getUpperCpInfo()->isArbRelease())
    {
        LOG_GCP_FAILURE("Invalid ARB-packet found Release instead of Request");

        return false;
    }

    return true;
}

std::string HwStreamParserHelper::getStreamTypeName()
{
    return m_streamTypeName[m_streamType];
}

bool HwStreamParserHelper::setNewUpperCpInfo(uint64_t handle, uint64_t hostAddress, uint64_t bufferSize)
{
    return getUpperCpInfo()->setNewInfo(handle,
                                        hostAddress,
                                        bufferSize,
                                        getUpperCpIndex(),
                                        1 << m_queuePhysicalOffset,
                                        m_pSyncManagersInfoDb);
}

bool HwStreamParserHelper::resetStreamInfo(uint32_t                 hwStreamId,
                                           uint64_t                 streamPhysicalOffset,
                                           SyncManagerInfoDatabase* pSyncManagersInfoDb,
                                           bool                     shouldResetSyncManager)
{
    CHECK_POINTER(SYN_CS_PARSER, pSyncManagersInfoDb, "Sync-Manager Info DB", false);

    getUpperCpInfo()->reset(shouldResetSyncManager);
    m_state                = getFirstState();
    m_hwQueueId            = hwStreamId;
    m_queuePhysicalOffset  = streamPhysicalOffset;
    m_pSyncManagersInfoDb  = pSyncManagersInfoDb;
    m_isStreamInfoInit     = true;

    return true;
};