#include "strategy_inflator.h"

using namespace gc::layered_brain;

StrategyInflator::StrategyInflator(const BundleViewContainerPtr& bundleViews)
{
    m_inflatorPerType[InflationType::INFLATE_FOR_UTILIZATION] =
        std::make_unique<StrategyInflatorForUtilization>(bundleViews);
    m_inflatorPerType[InflationType::INFLATE_FOR_BW] = std::make_unique<StrategyInflatorForBW>(bundleViews);
    m_inflatorPerType[InflationType::INFLATE_FOR_PERFORATION] =
        std::make_unique<StrategyInflatorForPerforation>(bundleViews);
    m_inflatorPerType[InflationType::INFLATE_FOR_NUM_SLICES] =
        std::make_unique<StrategyInflatorForNumSlices>(bundleViews);
}

bool StrategyInflator::inflateOneStep(InflationType      inflationType,
                                      const StrategyPtr& strategy,
                                      const NodePtr&     node) const
{
    SET_TEMP_LOG_CONTEXT("StrategyInflator");
    LOG_TRACE(LB_SLICER, "Inflate strategy {}, inflation type: {}", strategy->index(), inflationType);

    HB_ASSERT(m_inflatorPerType.find(inflationType) != m_inflatorPerType.end(),
              "Missing inflator for type {}",
              inflationType);

    return m_inflatorPerType.at(inflationType)->inflateOneStep(strategy, node);
}
