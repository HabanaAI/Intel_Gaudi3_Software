#include "mcid_generator.h"

#include "define_synapse_common.hpp"
#include "utils.h"

MCIDGenerator::MCIDGenerator()
{
    initAllGenerators();
}

void MCIDGenerator::initAllGenerators()
{
    m_nextMCIDs[DEGRADE] = MIN_MCID_VAL;
    m_nextMCIDs[DISCARD] = MIN_MCID_VAL;
}

LogicalMcid MCIDGenerator::nextMCID(MCIDType mcidType)
{
    HB_ASSERT(mcidType < NOF_MCID_TYPES, "Unexpected MCID Type: {}", mcidType);
    HB_ASSERT(m_nextMCIDs[mcidType] >= MIN_MCID_VAL, "Unexpected wraparound in Logical MCID values pool");

    return m_nextMCIDs[mcidType]++;
}
