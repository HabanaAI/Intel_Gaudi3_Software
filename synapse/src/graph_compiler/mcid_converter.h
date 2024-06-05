#pragma once
#include <stdint.h>
#include "cache_types.h"

class McidConverter
{
public:
    McidConverter();
    virtual ~McidConverter() { delete[] m_lastReleasingWindow; }

    void convertDegrade(LogicalMcid logicalMcid, PhysicalMcid& physicalMcid);
    void convertDiscard(LogicalMcid logicalMcid, PhysicalMcid& physicalMcid, unsigned& rolloverId);
    void convertReleaseDiscard(LogicalMcid logicalMcid, PhysicalMcid& physicalMcid, bool& changeToDegrade);
    void slideRolloverWindow() { m_currentRolloverWindowId++; }

    inline LogicalMcid  getMaxUsedLogicalDegrade() const { return m_maxUsedLogicalMcidDegrade; }
    inline LogicalMcid  getMaxUsedLogicalDiscard() const { return m_maxUsedLogicalMcidDiscard; }
    PhysicalMcid        getMaxUsedPhysicalDegrade() const;
    PhysicalMcid        getMaxUsedPhysicalDiscard() const;

private:
    LogicalMcid  m_maxUsedLogicalMcidDegrade = 0;
    LogicalMcid  m_maxUsedLogicalMcidDiscard = 0;
    PhysicalMcid m_degradeLimit              = 0;
    PhysicalMcid m_discardLimit              = 0;
    unsigned     m_currentRolloverWindowId   = 0;
    bool         m_releasingPhaseStarted     = false;
    int8_t*      m_lastReleasingWindow       = nullptr;
};
