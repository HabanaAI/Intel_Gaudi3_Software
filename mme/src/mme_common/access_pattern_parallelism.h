#pragma once

#include "mme_access_pattern.h"
#include "index_space_dimensions.h"

namespace MmeCommon::AccessPatternDetails
{
// This class adjusts an existing access pattern object to support deterministic concurrency or parallelism on some
// dimensions.
class AccessPatternParallelismAdjuster
{
public:
    AccessPatternParallelismAdjuster(AccessPattern& accessPattern) : m_accessPattern(accessPattern) {}

    // Set the given level of parallelism on the given dimension
    void setParallelized(Dim idxSpcDim, size_t parallelismLevel);

private:
    AccessPattern& m_accessPattern;
    std::unordered_map<Dim, size_t> m_parallelismLevels;

    void storeParallelism(Dim idxSpcDim, size_t parallelismLevel);
    uint64_t calcScale(Dim idxSpcDim, size_t parallelismLevel) const;
    void resizeIndexSpace(Dim idxSpcDim, size_t parallelismLevel);
    void rescaleOperandsAP(Dim idxSpcDim, size_t scale);
    void rescaleOperandAP(OperandRole role, size_t idxSpcDim, size_t scale);
    void addAuxAP(Dim idxSpcDim, size_t parallelismLevel);
};
}  // namespace MmeCommon::AccessPatternDetails