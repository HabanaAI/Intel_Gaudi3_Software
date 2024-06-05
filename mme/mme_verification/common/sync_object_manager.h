#pragma once

#include <cstdint>
#include <memory>
#include <unordered_set>
#include "drm/habanalabs_accel.h"
#include "json.hpp"

// coral includes
#include "coral_user_program_base.h"
#include "coral_user_utils.h"

#ifndef varoffsetof
#define varoffsetof(t, f) ((uint64_t)(&((t*) 0)->f))
#endif

namespace MmeCommon
{
class MmeHalReader;

static const unsigned c_so_group_size = 8;

// manager for test sync object index allocation.
class SyncObjectManager
{
public:
    SyncObjectManager(uint64_t smBase, unsigned mmeNr, const MmeCommon::MmeHalReader& mmeHal);
    virtual ~SyncObjectManager() = default;

    void resetTestGroup(unsigned stream);
    void alignSoIdx(nlohmann::json& test, unsigned alignment);

    void getSoIdx(std::pair<uint32_t, uint64_t>& so);

    void addPoleMonitor(CPProgram& prog,
                        unsigned stream,
                        unsigned masterIdx,
                        unsigned testIdInGroup,
                        unsigned int poleSoIdx,
                        unsigned int poleSoIdxSecondary,
                        unsigned int poleSlaveSoIdx,
                        unsigned int poleSlaveSoIdxSecondary,
                        unsigned int groupSize,
                        int soValue);
    SyncInfo createGroupSyncObj(unsigned groupSize);
    void addGroupMonitor(CPProgram& prog, unsigned stream);
    const SyncInfo& getCurrentGroupSyncObj() const { return m_currentGroupInfo; }
    void setUsedIdx(unsigned idx) { m_usedSoIdxsInGroup.insert(idx); }
    uint64_t getGroupSoAddress() const { return getSoAddress(m_groupSoIdx); }
    unsigned getCurrMonitorIdx() const { return m_monitorIdx; }

protected:
    // Create a monitor to watch SO and write to target address a specific value. A slight modification over the
    // function in funcsim
    virtual void addMonitor(CPProgram& prog,
                            const SyncInfo& syncInfo,
                            uint32_t monitorIdx,
                            uint64_t agentAddr,
                            unsigned agentPayload) = 0;
    virtual uint64_t getSoAddress(unsigned soIdx) const = 0;
    virtual unsigned uploadDmaQidFromStream(unsigned stream) = 0;

    const uint64_t m_smBase;
    const unsigned m_mmeNr;
    const unsigned c_so_group_size = 8;
private:
    unsigned getNewSoIdx();
    bool isIdxUsed(unsigned idx);

    const MmeCommon::MmeHalReader& m_mmeHal;

    unsigned m_minTestSoIdx;
    unsigned m_minSoIdx;
    unsigned m_maxSoIdx;

    unsigned m_groupSoIdx;
    unsigned m_nextSoIdx;
    unsigned m_monitorIdx;
    SyncInfo m_currentGroupInfo;
    std::unordered_set<unsigned> m_usedSoIdxsInGroup;
};
}  // namespace MmeCommon