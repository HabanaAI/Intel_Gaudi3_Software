#pragma once

#include "cache_types.h"
#include <array>

class MCIDGenerator
{
public:
    static constexpr LogicalMcid MIN_MCID_VAL = 1;

    enum MCIDType
    {
        DISCARD,
        DEGRADE,

        // Must be last
        NOF_MCID_TYPES
    };

    MCIDGenerator();

    LogicalMcid nextMCID(MCIDType mcidType);

private:
    std::array<LogicalMcid, NOF_MCID_TYPES> m_nextMCIDs {};

    void initAllGenerators();
};
