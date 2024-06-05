#pragma once

#include "bundle_view.h"
#include "strategy.h"
#include "brain_data.h"
#include "strategy_inflator_logic.h"

namespace gc::layered_brain
{
enum class InflationType
{
    INFLATE_FOR_UTILIZATION,
    INFLATE_FOR_BW,
    INFLATE_FOR_PERFORATION,
    INFLATE_FOR_NUM_SLICES,
};

// The strategy inflator is responsible to provide one step inflation according to inflation type:
// 1) INFLATE_FOR_UTILIZATION: Inflate BVDs marked as "inflate for utilization" in MME strategy.
// 2) INFLATE_FOR_BW: Inflate BVDs marked as "inflate for BW" in MME strategy.
// 3) INFLATE_FOR_NUM_SLICES: Inflate BVDs based on their distance from FCD:
// start with BVDs that are mapped to FCD dimensions on the bundle tensors.
class StrategyInflator
{
public:
    explicit StrategyInflator(const BundleViewContainerPtr& bundleViews);

    // One step inflation of BVD according to inflationType (see above).
    // Returns false if the given strategy can't be inflated (all dims are unsliced).
    bool inflateOneStep(InflationType inflationType, const StrategyPtr& strategy, const NodePtr& node = nullptr) const;

private:
    std::map<InflationType, std::unique_ptr<StrategyInflatorLogic>> m_inflatorPerType;
};

}  // namespace gc::layered_brain