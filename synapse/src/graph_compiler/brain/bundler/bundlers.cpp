#include "bundlers.h"
#include "bundle_seed_collector.h"
#include "bundle_seed_collector_factory.h"
#include <iterator>

using namespace gc::layered_brain;

std::vector<bundler::SeedCollectorPtr>
MmeBundler::getSupportedCollectors(const std::vector<bundler::SeedCollectorPtr>& derivedCollectors, HabanaGraph& graph)
{
    std::vector<bundler::SeedCollectorPtr>                                      collectors;
    std::set<bundler::SeedCollector::Type, bundler::SeedCollector::TypeCompare> requestedSeedCollectors {
        bundler::SeedCollector::Type::SINGLE_CONV,
        bundler::SeedCollector::Type::SINGLE_BATCH_GEMM,
        bundler::SeedCollector::Type::SINGLE_GEMM,
        bundler::SeedCollector::Type::MULTI_MME,
        bundler::SeedCollector::Type::ATTENTION};

    const bundler::SeedCollectorFactory seedCollectorFactory(graph);
    const auto                          seedCollectors(seedCollectorFactory.getSeedCollectorsByPriority());

    std::copy_if(seedCollectors.begin(),
                 seedCollectors.end(),
                 std::back_inserter(collectors),
                 [&requestedSeedCollectors](const auto& collector) {
                     return requestedSeedCollectors.find(collector->getType()) != requestedSeedCollectors.end();
                 });
    collectors.reserve(collectors.size() + derivedCollectors.size());
    for (const auto& collector : derivedCollectors)
    {
        collectors.push_back(collector);
    }
    return collectors;
}

MmeBundler::MmeBundler(HabanaGraph& graph, const std::vector<bundler::SeedCollectorPtr>& seedCollectors)
: Bundler(graph, getSupportedCollectors(seedCollectors, graph))
{
}