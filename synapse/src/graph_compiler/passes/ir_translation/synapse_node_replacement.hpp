#pragma once
#include "gc_protocol.hpp"
#include "llvm/small_vector.h"
#include "types.h"
#include "unordered_map"

class HabanaGraph;
constexpr unsigned NUM_NODES_IN_CLUSTER = 8;

// Struct to hold info required for replacing nodes.
// An instance of this struct holds info for a single and specific replacement
// of an old nodes cluster by a new nodes cluster.
struct NodeClusterReplacementInfo
{
    // ids of original nodes cluster that should be replaced
    llvm_vecsmall::SmallVector<unsigned,NUM_NODES_IN_CLUSTER> replacedNodeSynapseIds;
    // created new nodes that should replace the original nodes
    NodeList createdAdjacentNodes;
    // number of new nodes that should be created. Only when the created number reaches the expected number
    // a replacement will occur.
    unsigned adjacentNodesExpectedNum = 0;
};

// A class to manage node replacement in synapse side.
// It stores and updates instances of NodeClusterReplacementInfo, and performs node replacement when possible.
// Each instance of NodeClusterReplacementInfo is mapped to a "Leader" id, which provides fast access to it.
// The "Leader" id is the first id in NodeClusterReplacementInfo::replacedNodeSynapseIds,
// and it is determined in MLIR side.
class SynapseNodeReplacer
{
public:
    SynapseNodeReplacer(HabanaGraph& graph) : m_graph(graph) {};
    // Check if an instance of NodeClusterReplacementInfo with leaderId exists
    bool isNodeClusterReplacementInfoExist(unsigned leaderId);
    // Create new instance of NodeClusterReplacementInfo, init its inner fields with info from irNode
    void createNodeClusterReplacementInfo(const gc_protocol::ProtocolNode& irNode);
    void createNodeClusterReplacementInfo(unsigned leaderId);
    // Update NodeClusterReplacementInfo with new node
    void addSynapseNodeToNodeClusterReplacementInfo(NodePtr newNode, unsigned leaderId);
    // Check if an instance of NodeClusterReplacementInfo has all the info required for replacing nodes
    bool canReplaceNodes(unsigned leaderId);
    // Replace nodes according to a specific NodeClusterReplacementInfo
    bool replaceNodes(unsigned leaderId);

private:

    // Mapping between "Leader" id and the appropriate NodeClusterReplacementInfo.
    std::unordered_map<unsigned, NodeClusterReplacementInfo> m_nodeReplacementInfoMap;
    // The graph in which the replacement is done.
    HabanaGraph& m_graph;

};