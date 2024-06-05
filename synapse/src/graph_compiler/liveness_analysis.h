#pragma once

#include "allocators_utils.h"  // STATIC_TENSORS_COMPATIBILTY etc.
#include "tensor.h"            // TensorSetVector
#include "types.h"

#include <unordered_map>

/* The following class maintains information knowing when a tensor is alive or encountered for specific node.
 * The above is possible by using the following functions:
 *
 * 1. isRealTensorAliveAfterNode       : Returning true if the real tensor is still in use AFTER and NOT included in the
 *specified node.
 * 2. wasRealTensorEncounteredBeforeNode: Returning true if the real tesnor was encountered BEFORE and NOT included in
 *the specified node.
 * 3. getGraphMaxCapacity     : Returning max capacity for the all graph.
 * 4. getLiveTensorsEntries   : Returning a vector of live tensors for every node in the graph
 *
 **/

class HabanaGraph;

class LivenessAnalysis
{
public:
    LivenessAnalysis(const HabanaGraph* graph, const TensorsCompatibility& tensorsCompatibility);
    virtual ~LivenessAnalysis() = default;

    uint64_t getGraphMaxCapacity() const { return m_maxCapacity; };
    uint64_t getGraphMaxCapacityIdx() const { return m_maxCapacityIdx; };

    bool wasRealTensorEncounteredBeforeNode(const pNode& node, pTensor tensor)
        const;  // Returns true if the real tesnor was encountered BEFORE and NOT included the specified node.
    bool isRealTensorAliveAfterNode(const pNode& node, pTensor tensor)
        const;  // Returns true if the real tensor is still in use AFTER and NOT included in the specified node.
    TensorSetVector              getLiveTensorsEntries() const;
    const std::vector<uint64_t>& getUsedMem() const
    {
        return m_usedMem;
    };  // Returns a vector of live tensors for every node in the graph

    using TensorLifetimeMap = std::unordered_map<TensorPtr, Lifetime, TensorHasher>;

    Lifetime                 getTensorLifeTime(TensorPtr tensor) const;
    const TensorLifetimeMap& getTensorLifetimeMap() const { return m_tensorLifetimeMap; }

    struct LiveAndDie
    {
        TensorVector m_live;
        TensorVector m_die;
    };

    // Tensor that live and die at node scedule order
    // [nodeIdx][0 -> created tensors, 1 -> dead tensors]
    using LiveAndDieTensors = std::vector<LiveAndDie>;

    const LiveAndDieTensors& liveAndDieTensors() const;

private:
    std::pair<uint64_t, uint64_t> GetMaxCapacity(size_t nodeCount, const TensorLifetimeMap& tensorLifetimes);
    const unsigned                m_nodeCount;
    const TensorLifetimeMap       m_tensorLifetimeMap;
    uint64_t                      m_maxCapacity    = 0;
    uint64_t                      m_maxCapacityIdx = 0;
    std::vector<uint64_t>         m_usedMem;

    mutable LiveAndDieTensors m_liveAndDieTensors;
};

typedef std::shared_ptr<LivenessAnalysis> pLivenessAnalysis;
bool lifetimeIntersectsWith(const Lifetime& lifetime, const Lifetime& otherLifetime);

class ActivationsLivenessAnalysis : public LivenessAnalysis
{
public:
    ActivationsLivenessAnalysis(HabanaGraph* graph) : LivenessAnalysis(graph, ACTIVATIONS_TENSORS_COMPATIBILTY) {}
};

class StaticLivenessAnalysis : public LivenessAnalysis
{
public:
    StaticLivenessAnalysis(HabanaGraph* graph) : LivenessAnalysis(graph, STATIC_TENSORS_COMPATIBILTY) {}
};

class AllLivenessAnalysis : public LivenessAnalysis
{
public:
    AllLivenessAnalysis(HabanaGraph* graph) : LivenessAnalysis(graph, ALL_TENSORS_COMPATIBILTY) {}
};

class ForceDramAllocLivenessAnalysis : public LivenessAnalysis
{
public:
    ForceDramAllocLivenessAnalysis(HabanaGraph* graph) : LivenessAnalysis(graph, FORCE_DRAM_TENSORS_COMPATIBILTY) {}
};
