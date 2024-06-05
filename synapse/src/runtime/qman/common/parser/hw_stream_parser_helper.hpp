#pragma once

#include "define.hpp"
#include "lower_cp_qman_info.hpp"
#include "upper_cp_qman_info.hpp"

#include "runtime/qman/common/inflight_cs_parser.hpp"
#include "sync/sync_types.h"

enum eParserHwStreamType
{
    PHWST_STREAM_MASTER,
    PHWST_ARB_MASTER,
    PHWST_ARB_SLAVE,
    PHWST_INVALID,
    PHWST_LAST = PHWST_INVALID,
    PHWST_COUNT
};

namespace common
{
class SyncManagerInfo;
class SyncManagerInfoDatabase;
}  // namespace common

// Following helpers will define specific methods & information to parse each of the unique Upper-CP buffer
// Hence, we will define a base class and three successors - for Stream-Master, ARB-Master and ARB-Slaves
namespace common
{
class HwStreamParserHelper
{
public:
    HwStreamParserHelper(eParserHwStreamType queueType, InflightCsParserHelper& parserHelper)
    : m_streamType(queueType), m_parserHelper(parserHelper) {};

    virtual ~HwStreamParserHelper() = default;

    bool finalize();

    bool setNewUpperCpInfo(uint64_t handle, uint64_t hostAddress, uint64_t bufferSize);

    virtual uint32_t getQmansIndex() = 0;

    // Retrieves CP-Index
    virtual uint32_t getUpperCpIndex() = 0;
    virtual uint32_t getLowerCpIndex() = 0;

    bool resetStreamInfo(uint32_t                 hwStreamId,
                         uint64_t                 streamPhysicalOffset,
                         SyncManagerInfoDatabase* pSyncManagersInfoDb,
                         bool                     shouldResetSyncManager);

    virtual eCpParsingState getFirstState() = 0;
    virtual eCpParsingState getNextState()  = 0;

    bool parseUpperCpBuffer(eParsingDefinitions parsingDefs);
    bool parseLowerCpBuffer(bool isBufferOnHost);

    virtual bool handleFenceClearState();
    virtual bool handleFenceState();
    virtual bool handleArbRequestState();
    virtual bool handleCpDmaState() { return false; };
    virtual bool handleWorkCompletionState() { return false; };

    std::string getStreamTypeName();

    uint64_t getStreamPhysicalOffset() { return m_parserHelper.getStreamPhysicalOffset(); };

    uint32_t getQueueId() { return m_hwQueueId; };

protected:
    virtual bool isCurrentUpperCpPacketCpDma() const    = 0;
    virtual bool isCurrentUpperCpPacketArbPoint() const = 0;

    virtual uint64_t getExpectedPacketForFenceClearState() = 0;
    virtual uint64_t getExpectedPacketForFenceSetState()   = 0;
    virtual uint64_t getExpectedPacketForArbRequestState() = 0;

    virtual common::UpperCpQmanInfo* getUpperCpInfo() = 0;
    virtual common::LowerCpQmanInfo* getLowerCpInfo() = 0;

    virtual std::string getPacketName(uint64_t packetId) = 0;

    uint64_t m_fenceClearExpexctedAddress = 0;

    eCpParsingState m_state = CP_PARSING_STATE_INVALID;

    static const WaitID GC_FENCE_WAIT_ID = ID_0;

    uint32_t m_hwQueueId           = 0;
    uint64_t m_queuePhysicalOffset = 0;

private:
    SyncManagerInfoDatabase* m_pSyncManagersInfoDb = nullptr;

    eParserHwStreamType m_streamType = PHWST_INVALID;

    InflightCsParserHelper& m_parserHelper;

    bool m_isStreamInfoInit = false;

    static const std::string m_streamTypeName[PHWST_COUNT];
};
}  // namespace common