#pragma once

#include "bundle.h"
#include "bundlizer.h"
#include "hal_reader/hal_reader.h"
#include "log_manager.h"
#include "mme_slicing_strategy.h"
#include "slicing_brain.h"

class HabanaGraph;

/* Class responsible for adding new nodes to a bundle strategies. In general the go through the following steps:
 * 1.) Searching for expansion candidates
 * 2.) For each subset of candidates a strategy will be created
 * */
class BundleExpander
{
public:
    BundleExpander(HabanaGraph&          graph,
                   const AllBrains&      brains,
                   const Bundlizer&      bundlizer,
                   BundleSolvingDataMap& solvingDataPerBundle)
    : m_graph(graph), m_brains(brains), m_bundlizer(bundlizer), m_solvingDataPerBundle(solvingDataPerBundle)
    {}

    /* Create expanded strategies for all roles */
    SlicingStrategyList generateExpandedStrategies(const pBundle& bundle) const;

    /* Find candidates for the listed roles
     * Precondition: For each role in the roles list, its dependency fulfilled
     * FE: Slave consumer depends on the MME slave*/
    std::list<pBundleExpansion> discoverExpansionCandidatesForBundle(const pBundle& bundle,
                                                                     const std::list<BundleExpansion::Role>& roles) const;

    static bool validateCandidatePaths(const HabanaGraph&      g,
                                       const pBundleExpansion& candidate,
                                       const NodeSet&          acceptedNodes,
                                       const NodeSet&          acceptedProducers);

private:
    HabanaGraph&          m_graph;
    const AllBrains&      m_brains;
    const Bundlizer&      m_bundlizer;
    BundleSolvingDataMap& m_solvingDataPerBundle;

    /* Create expanded strategies for the listed roles */
    SlicingStrategyList generateExpandedStrategies(const pBundle& bundle,
                                                   const std::list<BundleExpansion::Role>& roles) const;

    /* Adjust and add the given candidate to the strategy */
    void expandStrategiesWithCandidate(SlicingStrategyList& strategies,
                                       const pBundleExpansion& candidate,
                                       const pBundle& bundle) const;

    /* Given a strategy and a candidate returns true if candidates dependency fulfilled by strategy nodes set
     * false will be return otherwise.
     * For example: if the candidate is slave-tpc-consumer need to make sure that there is node n s.t.
     * 1.) n is a slave
     * 2.) n output is input of the candidate node */
    static bool validateCandidateDependency(const pMmeSlicingStrategy& strategies, const pBundleExpansion& candidate);

    bool validateCandidateOperands(const pBundleExpansion& candidate, const NodeSet& strategyNodes) const;

    bool isCandidateValidForStrategy(const pBundleExpansion& candidate,
                                     const pMmeSlicingStrategy& strategy,
                                     const pBundle& bundle) const;

    /* Check if the current candidate is producer of the stiched operand (false means it is a consumer) */
    static bool isCandidateProducer(const pBundleExpansion& candidate);

    /* A node list (related to a bundle) with comparison and hash, for a more efficient control dependency check */
    class HashableNodeList : public NodeList
    {
    public:
        explicit HashableNodeList(unsigned bundleId);
        bool operator==(const HashableNodeList& other) const;
        unsigned getBundleId() const;
    private:
        const unsigned m_bundleId;
    };

    class Hasher // hasher for HashableNodeList
    {
    public:
        std::size_t operator()(const HashableNodeList& hashableNodeList) const;
    };

    /* For efficient lookup - mapping from node list to the result of the dependency check */
    mutable std::unordered_map<HashableNodeList, bool, Hasher> m_mapNodeListToDepCheck;

    ExpansionCandidatesSet findBundleExpansionCandidatesForRole(const pBundle& bundle,
                                                                BundleExpansion::Role role) const;

    /* Detection of all enabled candidates that depends on given candidate */
    void findDependantCandidates(pBundleExpansion candidate) const;

    /* param useCache (performance optimization) - the caller is responsible to "know" that the graph is not changed
     * between different calls tot this function, since the goal of the cache is to prevent the temporary graph modification
     * and the check for wherever the temp graph is cyclic (if the graph can be changed - the caller should set to false)*/
    bool isCandidateCreatesInvalidDependencies(const pBundleExpansion& candidate,
                                               const pMmeSlicingStrategy& strategy,
                                               const pBundle& bundle,
                                               bool  useCache = false) const;

    bool isBundleDependentOnNodes(const NodeList& prevNodes,
                                  const NodeList& postNodes,
                                  const NodeList& midBundleNodes) const;

    bool isExpansionRoleEnabledForBundle(const pBundle& bundle, BundleExpansion::Role role) const;

    void findWinningStrategyForSlaveBundle(pBundleExpansion slaveCandidate, const pBundle& slaveBundle) const;
};
