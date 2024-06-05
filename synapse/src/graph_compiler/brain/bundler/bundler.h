#pragma once

#include <optional>
#include <memory>

#include "bundle_seed_collector.h"
#include "node.h"
#include "bundle_plane_graph.h"
#include "bundle_seed_collector_factory.h"
#include "layered_brain_bundle.h"

namespace gc::layered_brain
{
class Bundler
{
public:
    Bundler(HabanaGraph& graph, const std::vector<bundler::SeedCollectorPtr>& seedCollectors);

    virtual ~Bundler() = 0;

    /**
     * @brief Generates a {bundleIndex-> bundle nodes vector} map corresponding to the graph
     *        provided to the bundler object upon construction
     *
     * @note  LAYERED BRAIN FORWARD PROGRESS MODE API METHOD
     */
    std::map<BundleIndex, NodeVector> generateBundles();

    /**
     * @brief Attempts performing and expansion step and returns true if
     *        expansions was successful, otherwise false
     * @note  LAYERED BRAIN ITERATIVE MODE API METHOD
     */
    bool expansionStep(const BundlePtr& bundle, bundler::BundleExpanders& expanders) const;

    /**
     * @brief Get container of seed collectors corresponding to the bundler
     *
     * @note  LAYERED BRAIN ITERATIVE MODE API METHOD
     */
    const std::vector<bundler::SeedCollectorPtr>& getSeedCollectors() const;

    /**
     * @brief Log <info> layered brain bundle status
     *
     * @note  LAYERED BRAIN ITERATIVE MODE API METHOD
     */
    void logGraphBundlingStatus() const;

protected:
    /**
     * @brief Collect all bundle seeds in the graph and returns them in a container
     *        of <bundle, bundle expander list> pairs.
     */
    std::vector<std::pair<BundlePtr, bundler::BundleExpanders>> gatherSeeds();

    std::map<BundleIndex, NodeVector>
    toBundleMap(const std::vector<std::pair<BundlePtr, bundler::BundleExpanders>>& bundlesAndExpanders);

    /**
     * @brief Recieves a container of <bundle, bundle expander list> pairs representing all bundles
     *        in the graph and expands each bundle.
     */
    void expandBundles(std::vector<std::pair<BundlePtr, bundler::BundleExpanders>>& seeds);

    HabanaGraph& m_graph;

private:
    /**
     * @brief Rotate input expanders list if list size > 1
     *
     */
    static void rotateExpanders(bundler::BundleExpanders& expanders);

    /**
     * @brief Returns the next expander object to perform an expansion step with.
     *        Enforces producer expanders all finish running before consumer
     *        expanders are deployed.
     */
    bundler::BundleExpanderPtr getNextExpander(bundler::BundleExpanders& expanders) const;

    std::vector<bundler::SeedCollectorPtr> m_seedCollectors;
};

using BundlerPtr = std::unique_ptr<Bundler>;

}  // namespace gc::layered_brain