#include "defs.h"
#include "graph_editor.h"
#include "habana_graph.h"
#include "habana_pass.h"
#include "synapse_common_types.h"
#include "synapse_node_replacement.hpp"

using namespace synapse;

bool SynapseNodeReplacer::isNodeClusterReplacementInfoExist(unsigned leaderId)
{
    return m_nodeReplacementInfoMap.find(leaderId) != m_nodeReplacementInfoMap.end();
}
void SynapseNodeReplacer::createNodeClusterReplacementInfo(const gc_protocol::ProtocolNode& irNode)
{
    unsigned leaderId = irNode.replacedNodeIds.data[0];
    HB_ASSERT(m_nodeReplacementInfoMap.find(leaderId) == m_nodeReplacementInfoMap.end(),
              "cluster of leader id {} already exists", leaderId);

    NodeClusterReplacementInfo newClusterReplacementInfo;
    newClusterReplacementInfo.adjacentNodesExpectedNum = irNode.newAdjacentNodeIds.size;
    newClusterReplacementInfo.replacedNodeSynapseIds = llvm_vecsmall::SmallVector<unsigned, NUM_NODES_IN_CLUSTER>(
                                                      irNode.replacedNodeIds.begin(),
                                                      irNode.replacedNodeIds.end());
    m_nodeReplacementInfoMap[leaderId] = newClusterReplacementInfo;
}

void SynapseNodeReplacer::createNodeClusterReplacementInfo(unsigned leaderId)
{
    HB_ASSERT(m_nodeReplacementInfoMap.find(leaderId) == m_nodeReplacementInfoMap.end(),
              "cluster of leader id {} already exists", leaderId);

    NodeClusterReplacementInfo newClusterReplacementInfo;
    newClusterReplacementInfo.replacedNodeSynapseIds = llvm_vecsmall::SmallVector<unsigned, NUM_NODES_IN_CLUSTER>();
    newClusterReplacementInfo.replacedNodeSynapseIds.push_back(leaderId);
    m_nodeReplacementInfoMap[leaderId] = newClusterReplacementInfo;
}

void SynapseNodeReplacer::addSynapseNodeToNodeClusterReplacementInfo(NodePtr newNode, unsigned leaderId)
{
    auto nodeReplacementItr = m_nodeReplacementInfoMap.find(leaderId);
    HB_ASSERT(nodeReplacementItr != m_nodeReplacementInfoMap.end(),
              "cluster of leader id {} expected to exist", leaderId);
    NodeClusterReplacementInfo& clusterReplacementInfo = nodeReplacementItr->second;
    clusterReplacementInfo.createdAdjacentNodes.push_back(newNode);
}
bool SynapseNodeReplacer::canReplaceNodes(unsigned leaderId)
{
    auto nodeReplacementItr = m_nodeReplacementInfoMap.find(leaderId);
    HB_ASSERT(nodeReplacementItr != m_nodeReplacementInfoMap.end(),
              fmt::format("cluster of leader id {} expected to exist", leaderId).c_str());
    // Only when the created number reaches the expected number a replacement will occur.
    return nodeReplacementItr->second.adjacentNodesExpectedNum ==
           nodeReplacementItr->second.createdAdjacentNodes.size();
}

bool SynapseNodeReplacer::replaceNodes(unsigned leaderId)
{
    LOG_DEBUG(GC_TRANSLATION, "Replacing nodes for leader Id {}", leaderId);
    auto nodeReplacementItr = m_nodeReplacementInfoMap.find(leaderId);
    HB_ASSERT(nodeReplacementItr != m_nodeReplacementInfoMap.end(),
              "cluster of leader id {} expected to exist", leaderId);
    // create oldNodes list
    NodeList oldNodes;
    for (auto oldId : nodeReplacementItr->second.replacedNodeSynapseIds)
    {
        NodePtr oldNode = m_graph.getNodeByID(oldId);
        HB_ASSERT(oldNode != nullptr, "old node with id {} not found in graph", oldId);
        oldNodes.push_back(oldNode);
    }
    // replace the nodes
    auto retVal = GraphEditor::replaceNodes(m_graph, oldNodes, nodeReplacementItr->second.createdAdjacentNodes);
    if (retVal == REPLACE_NODE_SUCCESS)
    {
        if(m_graph.getInferenceMode())
        {
            if (std::any_of(nodeReplacementItr->second.createdAdjacentNodes.begin(),
                        nodeReplacementItr->second.createdAdjacentNodes.end(),
                        [&](NodePtr n) { return m_graph.runsOnMME(n); } ))
            {
                // predicate will trigger update MME precision pass
                m_graph.turnOnPredicate(PREDICATE_ID_EXTERNAL_MME_NODE_CREATED);
            }
        }
        if (std::any_of(nodeReplacementItr->second.createdAdjacentNodes.begin(),
                        nodeReplacementItr->second.createdAdjacentNodes.end(),
                        [&](NodePtr n) {
                            return std::any_of(n->getOutputs().begin(), n->getOutputs().end(), [&](TensorPtr t) {
                                return t->getTensorType() == HOST_TO_DEVICE_TENSOR;
                            });
                        }))
        {
            // predicate will trigger update HodtMaxData
            m_graph.turnOnPredicate(PREDICATE_ID_NODE_CREATED_WITHOUT_OUTPUT_SHAPE);
        }
    }
    return retVal == REPLACE_NODE_SUCCESS;
}