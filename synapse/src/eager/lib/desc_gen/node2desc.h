#pragma once

// eager includes (relative to src/eager/lib/)
#include "desc_gen/desc_base.h"
#include "utils/general_defs.h"

// synapse api (relative to include/)
#include "synapse_api_types.h"

namespace eager_mode
{
class EagerGraph;
class EagerNode;
class EagerNodes;

///////////////////////////////////////////////////////////////////////////////////////////////////
// SingleNode2Desc
///////////////////////////////////////////////////////////////////////////////////////////////////

// Representation of single-node-to-descriptor resolver
class SingleNode2Desc final
{
public:
    explicit SingleNode2Desc(EagerGraph& graph) : m_graph(graph) {}
    bool init(const EagerNode& node, const EagerNode* latestPhysicalProducer = nullptr);

    const EagerGraph& getGraph() const { return m_graph; }

    bool isSingleActivation() const { return getDescGen().getActivationNr() == 1; }
    bool generateDescriptors();

    DescGeneratorBase*       getDescGen(synNodeId id);
    const DescGeneratorBase& getDescGen() const;
    DescGeneratorBase&       getDescGen();

    const EagerNode*         getLatestPhysicalProducer() const { return m_latestPhysicalProducer; }

private:
    EagerGraph&          m_graph;
    DescGeneratorBasePtr m_descGenPtr;
    // Pointer to the latest node relative to execution sequence that produced
    // a tensor which is consumed by the node associated to this instance
    const EagerNode* m_latestPhysicalProducer = nullptr;
};

///////////////////////////////////////////////////////////////////////////////////////////////////
// Node2DescContainer
///////////////////////////////////////////////////////////////////////////////////////////////////

// Representation of multi-node-to-descriptor resolver. It's a container of SingleNode2Desc objects
// that are sorted according to their execution order.
class Node2DescContainer final
{
public:
    using sequence_type = VecNodes<SingleNode2Desc>;

    explicit Node2DescContainer(EagerGraph& graph) : m_graph(graph) {}
    bool init(const EagerNodes& nodes, const VecNodes<NodesNrType>& latestPhysicalProducers);
    bool isInitialized() const { return m_isInitialized; }

    const EagerGraph& getGraph() const { return m_graph; }

    bool isSingleDesc() const { return m_execSequence.size() == 1; }
    bool isSingleActivation() const { return isSingleDesc() && m_execSequence.front().isSingleActivation(); }
    const DescGeneratorBase& getFirstDescGen() const { return m_execSequence.front().getDescGen(); }

    bool generateDescriptors();

    sequence_type&       getExecSequence() { return m_execSequence; }
    const sequence_type& getExecSequence() const { return m_execSequence; }

    const AllStatisticsType&      getStatistics() const { return m_stats; }
    const StatisticsOfEngineType& getStatistics(EngineType engine) const
    {
        return m_stats[static_cast<unsigned>(engine)];
    }

private:
    EagerGraph&       m_graph;
    AllStatisticsType m_stats {};
    sequence_type     m_execSequence;  // List of all nodes sorted by execution order
    bool              m_isInitialized {};
};

}  // namespace eager_mode
