#pragma once
#include <memory>
#include "bundle_view.h"
#include "strategy.h"

namespace gc::layered_brain
{
class SlicingDetails
{
public:
    SlicingDetails(const BundleViewContainerPtr& bundleViews, const StrategyPtr& strategy)
    : m_bundleViews(bundleViews), m_strategy(strategy)
    {
        HB_ASSERT_PTR(bundleViews);
    }
    uint64_t                getNofSlices() const;
    float                   getMmeUtil(const NodePtr& mme) const;
    float                   getMmeBw(const NodePtr& mme) const;
    std::optional<uint64_t> getNodePerforationBvdMultiplier(const NodePtr& n) const;
    std::optional<float>    getNodePerforationUtil(const NodePtr& n) const;

private:
    SolutionParamsPtr             getQOR(const NodePtr& mme) const;
    const BundleViewContainerPtr& m_bundleViews;
    const StrategyPtr&            m_strategy;
};

using ConstSlicingDetailsPtr = std::shared_ptr<const SlicingDetails>;
using SlicingDetailsPtr      = std::shared_ptr<SlicingDetails>;

}  // namespace gc::layered_brain