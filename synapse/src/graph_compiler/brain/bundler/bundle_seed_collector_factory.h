#pragma once

#include <vector>
#include "bundle_seed_collector.h"

class HabanaGraph;

namespace gc::layered_brain::bundler
{
class SeedCollectorFactory
{
public:
    explicit SeedCollectorFactory(HabanaGraph& graph);
    std::vector<SeedCollectorPtr> getSeedCollectorsByPriority() const;

private:
    HabanaGraph& m_graph;

    const std::map<SeedCollector::Type, SeedCollectorPtr, SeedCollector::TypeCompare> m_collectorMap;
};
}  // namespace gc::layered_brain::bundler
