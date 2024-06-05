#include "bundler.h"
#include "bundle_seed_collector.h"
#include "bundler/bundle_expander.h"
#include "layered_brain.h"
#include "brain_conf.h"

using namespace gc::layered_brain;

Bundler::Bundler(HabanaGraph& graph, const std::vector<bundler::SeedCollectorPtr>& seedCollectors)
: m_graph(graph), m_seedCollectors(seedCollectors)
{
}
Bundler::~Bundler() = default;

bundler::BundleExpanderPtr Bundler::getNextExpander(bundler::BundleExpanders& expanders) const
{
    HB_ASSERT(!expanders.empty(), "Expecting a non-empty expanders container");
    // Make sure all preferred expanders finish expanding before letting the other expanders deploy
    const auto expanderIt = std::find_if(expanders.begin(), expanders.end(), [](const bundler::BundleExpanderPtr& ex) {
        return GCFG_ENABLE_LB_PREFER_CONSUMERS.value() ? !ex->isProducerExpander() : ex->isProducerExpander();
    });
    if (expanderIt != expanders.end())
    {
        // in case a preferred expander exists but isn't the first in the expanders list,
        // move preceeding non-preferred expanders to the back of the list
        expanders.splice(expanders.end(), expanders, expanders.begin(), expanderIt);
    }
    return expanders.front();
}

void Bundler::rotateExpanders(bundler::BundleExpanders& expanders)
{
    if (expanders.size() > 1)
    {
        expanders.splice(expanders.end(), expanders, expanders.begin());
    }
}

bool Bundler::expansionStep(const BundlePtr& bundle, bundler::BundleExpanders& expanders) const
{
    HB_ASSERT_PTR(bundle);
    while (!expanders.empty())
    {
        // proceed from the previously used expander
        auto expander = getNextExpander(expanders);
        // perform an expansion step attempt
        BundlePtr expandedBundle = expander->expand();
        if (expandedBundle == nullptr)
        {
            // expander exhausted
            expanders.pop_front();
        }
        else
        {
            // successful expansion step
            HB_ASSERT(expandedBundle->hasCandidates(), "Expecting bundle with candidates");
            rotateExpanders(expanders);
            return true;
        }
    }
    return false;
}

std::map<BundleIndex, NodeVector> Bundler::generateBundles()
{
    auto bundlesAndExpanders = gatherSeeds();
    expandBundles(bundlesAndExpanders);
    logGraphBundlingStatus();
    return toBundleMap(bundlesAndExpanders);
}

std::map<BundleIndex, NodeVector>
Bundler::toBundleMap(const std::vector<std::pair<BundlePtr, bundler::BundleExpanders>>& bundles)
{
    std::map<BundleIndex, NodeVector> bundleMap;
    for (const auto& bundleAndExpanders : bundles)
    {
        const auto& bundle = bundleAndExpanders.first;
        bundleMap.insert(std::make_pair(bundle->index(), bundle->getNodesCopy<NodeVector>()));
    }
    return bundleMap;
}

void Bundler::expandBundles(std::vector<std::pair<BundlePtr, bundler::BundleExpanders>>& bundlesAndExpanders)
{
    for (auto& bundleAndExpanders : bundlesAndExpanders)
    {
        auto& bundle = bundleAndExpanders.first;
        HB_ASSERT_PTR(bundle);
        auto expanders = bundleAndExpanders.second;
        while (!expanders.empty())
        {
            auto expander = getNextExpander(expanders);

            // bundle nodes added directly to bundle by the expander
            if (const auto& expandedBundle = expander->expand(); !expandedBundle)
            {
                // pop exhausted expanders
                expanders.pop_front();
            }
        }
    }
}

std::vector<std::pair<BundlePtr, bundler::BundleExpanders>> Bundler::gatherSeeds()
{
    std::vector<std::pair<BundlePtr, bundler::BundleExpanders>> bundles {};
    const auto&                                                 seedCollectors = getSeedCollectors();
    HB_ASSERT(!seedCollectors.empty(), "Expecting non-empty seed collectors container");

    for (auto& collector : seedCollectors)
    {
        auto bundleAndExpanderLists = collector->collect();
        LOG_DEBUG(LB_BUNDLER, "{} found {} seeds", toString(collector->getType()), bundleAndExpanderLists.size());
        bundles.insert(bundles.end(), bundleAndExpanderLists.begin(), bundleAndExpanderLists.end());
    }
    return bundles;
}

const std::vector<bundler::SeedCollectorPtr>& Bundler::getSeedCollectors() const
{
    return m_seedCollectors;
}

void Bundler::logGraphBundlingStatus() const
{
    if (!LOG_LEVEL_AT_LEAST_INFO(LB_BUNDLER)) return;

    // clear execution schedule cache to force re-calculation of it (for accurate logging)
    m_graph.invalidateExecutionSchedule();
    LOG_INFO(LB_BUNDLER, "Bundling Status:");
    const auto& nodes = m_graph.getExeSortedNodes();
    for (const auto& n : nodes)
    {
        const auto& bi = n->getNodeAnnotation().bundleInfo;
        if (!bi.is_set()) continue;
        LOG_INFO(LB_BUNDLER,
                 "Bundle-ID: {:>3} Node: {} [{}]",
                 std::to_string(bi->bundleIndex),
                 n->getNodeName(),
                 n->getNodeTypeStr());
    }
}