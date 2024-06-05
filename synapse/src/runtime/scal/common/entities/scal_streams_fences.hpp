#pragma once

#include "synapse_common_types.h"

#include "runtime/scal/common/infra/scal_types.hpp"

class ScalStreamsFences
{
public:
    synStatus init(unsigned baseIdx, unsigned size);
    void      getFenceId(FenceIdType& fenceId);

private:
    uint64_t m_baseIdx;
    uint64_t m_size;
    uint64_t m_curIdx;
};
