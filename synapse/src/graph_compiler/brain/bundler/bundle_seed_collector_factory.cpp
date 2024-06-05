#include "bundle_seed_collector_factory.h"
#include "bundle_seed_collectors.h"
#include "habana_graph.h"
#include <algorithm>
#include <memory>

using namespace gc::layered_brain::bundler;

template<typename CollectorType>
static std::pair<SeedCollector::Type, SeedCollectorPtr> makeCollectorMapItem(CollectorType&& collector)
{
    SeedCollectorPtr pc = std::make_shared<CollectorType>(collector);
    HB_ASSERT_PTR(pc);
    return std::make_pair(pc->getType(), pc);
}

SeedCollectorFactory::SeedCollectorFactory(HabanaGraph& graph)
: m_graph(graph),
  m_collectorMap({makeCollectorMapItem(SingleConvCollector(graph)),
                  makeCollectorMapItem(SingleBatchGemmCollector(graph)),
                  makeCollectorMapItem(SingleGemmCollector(graph)),
                  makeCollectorMapItem(MultiGemmCollector(graph)),
                  makeCollectorMapItem(AttentionCollector(graph))})
{
    HB_ASSERT(m_collectorMap.size() ==
                  static_cast<SeedCollector::CollectorEnumValueType>(SeedCollector::Type::NUM_TYPES),
              "Expecting factory to contain all collector types");
}

std::vector<SeedCollectorPtr> SeedCollectorFactory::getSeedCollectorsByPriority() const
{
    std::vector<SeedCollectorPtr> ret;
    ret.reserve(m_collectorMap.size());
    for (const auto& item : m_collectorMap)
    {
        ret.push_back(item.second);
    }
    return ret;
}