#include "defs.h"
#include "mcid_converter.h"
#include "scal.h"

static constexpr uint16_t DEGRADE_MCID_COUNT = SCAL_MAX_DEGRADE_MCID_COUNT_GAUDI3;
static constexpr uint16_t DISCARD_MCID_COUNT = SCAL_MAX_DISCARD_MCID_COUNT_GAUDI3;

// Using the safety factor we account for the +1 that we add in the conversion functions (which intended to avoid the
// absolute 0) and in addition, another 1 MCID to keep safety distance from the limit given by SCAL. Thus, no need to
// worry about misunderstanding regarding the meaning of the limits and the reported used MCIDs in the recipe.
static constexpr uint16_t SAFETY_FACTOR = 2;

McidConverter::McidConverter()
 : m_degradeLimit(DEGRADE_MCID_COUNT - SAFETY_FACTOR),
   m_discardLimit(DISCARD_MCID_COUNT - SAFETY_FACTOR)
{
    // Modify the discard limit per GCFG for rollover testing purposes
    if (GCFG_CACHE_MAINT_MCID_DISCARD_LIMIT_FOR_TESTING.value() > 0)
    {
        m_discardLimit = GCFG_CACHE_MAINT_MCID_DISCARD_LIMIT_FOR_TESTING.value();
    }
}

void McidConverter::convertDegrade(LogicalMcid logicalMcid, PhysicalMcid& physicalMcid)
{
    m_maxUsedLogicalMcidDegrade = std::max(m_maxUsedLogicalMcidDegrade, logicalMcid);
    physicalMcid = 0; // init output first
    if (logicalMcid == 0) return; // this is not expected, but just in case
    physicalMcid = ((logicalMcid - 1) % m_degradeLimit) + 1; // never return 0 to differentiate from the absolute 0
}

void McidConverter::convertDiscard(LogicalMcid logicalMcid, PhysicalMcid& physicalMcid, unsigned& rolloverId)
{
    HB_ASSERT(m_releasingPhaseStarted == false, "don't call convertDiscard after convertReleaseDiscard");
    m_maxUsedLogicalMcidDiscard = std::max(m_maxUsedLogicalMcidDiscard, logicalMcid);
    physicalMcid = 0; // init output first
    rolloverId   = 0; // init output first
    if (logicalMcid == 0) return; // this is not expected, but just in case
    physicalMcid = ((logicalMcid - 1) % m_discardLimit) + 1; // never return 0 to differentiate from the absolute 0
    rolloverId   = (logicalMcid - 1) / m_discardLimit;
    // to relax this assert, change the data type of m_lastReleasingWindow from int8_t to int16_t
    HB_ASSERT(rolloverId < std::numeric_limits<int8_t>::max(), "exceeded max allowed rollover-windows");
}

void McidConverter::convertReleaseDiscard(LogicalMcid logicalMcid, PhysicalMcid& physicalMcid, bool& changeToDegrade)
{
    m_releasingPhaseStarted = true;
    HB_ASSERT(logicalMcid <= getMaxUsedLogicalDiscard(), "logical mcid exceeded the max seen by convertDiscard");
    physicalMcid    = 0;     // init output first
    changeToDegrade = false; // init output first
    if (logicalMcid == 0) return; // this is not expected, but just in case
    physicalMcid = ((logicalMcid - 1) % m_discardLimit) + 1; // never return 0 to differentiate from the absolute 0

    // If the graph has no rollover, no need to proceed
    if (getMaxUsedLogicalDiscard() <= m_discardLimit) return;

    // One-time initialization of the releasing windows history
    if (m_lastReleasingWindow == nullptr)
    {
        m_lastReleasingWindow = new int8_t[DISCARD_MCID_COUNT];
        for (unsigned i = 0; i < DISCARD_MCID_COUNT; i++) m_lastReleasingWindow[i] = -1;
    }

    // Handle the rollover corner cases in which we need to change a discard to degrade. Two corner cases to consider:
    //   1. We come to discard a logical mcid that was allocated by previous rollover-window
    //   2. We come to discard a logical mcid that is mapped to physical mcid which is still in use by previous window
    // In both cases we need to change the discard to degrade

    int8_t theReleasingWindow = (logicalMcid - 1) / m_discardLimit;
    HB_ASSERT(theReleasingWindow <= m_currentRolloverWindowId, "did you forget to call slideRolloverWindow?");

    // Handle case #1
    changeToDegrade = theReleasingWindow < m_currentRolloverWindowId;

    // Handle case #2
    if (m_lastReleasingWindow[physicalMcid] + 1 == theReleasingWindow)
    {
        // all previous window(s) released our physical mcid, so we are allowed to discard
        m_lastReleasingWindow[physicalMcid] = m_currentRolloverWindowId;
    }
    else if (m_lastReleasingWindow[physicalMcid] < theReleasingWindow)
    {
        // our physical mcid is still alive in previous window(s) so we are NOT allowed to discard
        changeToDegrade = true;
    }
}

PhysicalMcid McidConverter::getMaxUsedPhysicalDegrade() const
{
    return getMaxUsedLogicalDegrade() == 0 ?
               0 :
               std::min(getMaxUsedLogicalDegrade(), static_cast<LogicalMcid>(m_degradeLimit)) + SAFETY_FACTOR;
}

PhysicalMcid McidConverter::getMaxUsedPhysicalDiscard() const
{
    return getMaxUsedLogicalDiscard() == 0 ?
               0 :
               std::min(getMaxUsedLogicalDiscard(), static_cast<LogicalMcid>(m_discardLimit)) + SAFETY_FACTOR;
}
