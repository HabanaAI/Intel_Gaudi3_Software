#pragma once

#include "bundle_view.h"
#include "strategy.h"

namespace gc::layered_brain
{
// The strategy inflator logic holds common inflation functionality.
class StrategyInflatorLogic
{
public:
    explicit StrategyInflatorLogic(const BundleViewContainerPtr& bundleViews) : m_bundleViews(bundleViews) {}
    virtual ~StrategyInflatorLogic() {};

    bool inflateOneStep(const StrategyPtr& strategy, const NodePtr& node) const;

protected:
    virtual std::vector<BundleViewId> getCandidatesForInflation(const StrategyPtr& strategy,
                                                                const NodePtr&     node) const = 0;
    virtual bool                      inflateCandidates(const StrategyPtr&               strategy,
                                                        const NodePtr&                   node,
                                                        const std::vector<BundleViewId>& inflationCandidates) const;
    bool                              canInflateBVD(const StrategyPtr& strategy, BundleViewId bvd) const;

    const BundleViewContainerPtr m_bundleViews;
};

// Specific implementation for inflation for utilization.
class StrategyInflatorForUtilization : public StrategyInflatorLogic
{
public:
    explicit StrategyInflatorForUtilization(const BundleViewContainerPtr& bundleViews)
    : StrategyInflatorLogic(bundleViews)
    {
    }

protected:
    bool                      inflateCandidates(const StrategyPtr&               strategy,
                                                const NodePtr&                   node,
                                                const std::vector<BundleViewId>& inflationCandidates) const override;
    std::vector<BundleViewId> getCandidatesForInflation(const StrategyPtr& strategy,
                                                        const NodePtr&     node) const override;
};

// Specific implementation for inflation for BW.
class StrategyInflatorForBW : public StrategyInflatorLogic
{
public:
    explicit StrategyInflatorForBW(const BundleViewContainerPtr& bundleViews) : StrategyInflatorLogic(bundleViews) {}

protected:
    std::vector<BundleViewId> getCandidatesForInflation(const StrategyPtr& strategy,
                                                        const NodePtr&     node) const override;
};

// Specific implementation for inflation for perforation.
class StrategyInflatorForPerforation : public StrategyInflatorLogic
{
public:
    explicit StrategyInflatorForPerforation(const BundleViewContainerPtr& bundleViews)
    : StrategyInflatorLogic(bundleViews)
    {
    }

protected:
    bool                      inflateCandidates(const StrategyPtr&               strategy,
                                                const NodePtr&                   node,
                                                const std::vector<BundleViewId>& inflationCandidates) const override;
    std::vector<BundleViewId> getCandidatesForInflation(const StrategyPtr& strategy,
                                                        const NodePtr&     node) const override;
};

// Specific implementation for inflation for number of slices.
class StrategyInflatorForNumSlices : public StrategyInflatorLogic
{
public:
    explicit StrategyInflatorForNumSlices(const BundleViewContainerPtr& bundleViews)
    : StrategyInflatorLogic(bundleViews)
    {
    }

protected:
    bool                      inflateCandidates(const StrategyPtr&               strategy,
                                                const NodePtr&                   node,
                                                const std::vector<BundleViewId>& inflationCandidates) const override;
    std::vector<BundleViewId> getCandidatesForInflation(const StrategyPtr& strategy,
                                                        const NodePtr&     node) const override;
    Dim                       getSmallestTensorDimInBVD(BundleViewId bvd) const;
    unsigned                  getNumOccurrencesOfTensorDimInBVD(BundleViewId bvd, Dim dim) const;
};

}  // namespace gc::layered_brain