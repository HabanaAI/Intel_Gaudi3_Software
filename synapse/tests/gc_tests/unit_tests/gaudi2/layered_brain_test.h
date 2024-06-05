#pragma once

#include "gaudi2_graph.h"
#include "graph_optimizer_test.h"
#include "node_factory.h"
#include "layered_brain.h"
#include "tpc_slicing_test_infra.h"

using namespace gc::layered_brain;

class LayeredBrainTest : public GraphOptimizerTest
{
protected:
    Gaudi2Graph m_graph;
    NodeVector  m_nodeChain;

    void createGraph(int numNodes)
    {
        for (int idx = 0; idx < numNodes; idx++)
        {
            addNodeToChain();
        }
    }

    void addNodeToChain()
    {
        m_nodeChain.push_back(newNode(m_nodeChain.size()));
        GraphEditor::addNode(m_graph, m_nodeChain.back());
    }

    NodePtr newNode(unsigned idx)
    {
        const auto& in  = lastTensor();
        const auto& out = newTensor();
        return TPCCustomIndexSpaceNode::createSliceableNode(in, out);
    }

    TensorPtr lastTensor() { return m_nodeChain.empty() ? newTensor() : m_nodeChain.back()->getOutput(0); }

    TensorPtr newTensor()
    {
        TSize sizes[] = {128, 128};
        return std::make_shared<Tensor>(2, sizes, syn_type_float);
    }

    // Bundles nodes nodeBegin..nodeEnd-1 in bundle with ID bundleId
    BundleNodes bundleNodes(unsigned bundleId, int nodeBegin, int nodeEnd)
    {
        BundleNodes ret;
        for (int n = nodeBegin; n < nodeEnd; n++)
        {
            bundleNode(bundleId, n);
            ret.push_back(m_nodeChain[n]);
        }
        return ret;
    }

    void bundleNode(unsigned bundleId, int nodeIndex) { bundleNode(bundleId, m_nodeChain[nodeIndex], nodeIndex); }

    void bundleNode(unsigned bundleId, NodePtr& node, unsigned opIdx)
    {
        HB_ASSERT(!node->getNodeAnnotation().bundleInfo.is_set(), "Node already bundled");
        node->getNodeAnnotation().bundleInfo.set(BundleInfo {bundleId, BundleType::UNDEFINED, opIdx});
        node->getNodeAnnotation().bundleInfo->threadIndex = 0;
    }
};