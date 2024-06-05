#pragma once

#include "graph_optimizer_test.h"
#include "habana_pass.h"
#include "tensor.h"
#include "node.h"
#include "node_factory.h"
#include "graph_editor.h"
#include "types.h"

template<class GraphType>
class MemsetSchedulingTest : public GraphOptimizerTest
{
public:
    pTensor createTensor(bool isInSram = false)
    {
        const std::vector<TSize> shape  = {64, 64, 2, 128};
        auto                     tenosr = std::make_shared<Tensor>(shape.size(), shape.data(), syn_type_bf16);
        if (isInSram)
        {
            tenosr->setTensorInSram();
        }
        return tenosr;
    }

    pNode createTpcNode(const std::string&  name,
                        const TensorVector& inputs,
                        const TensorVector& outputs,
                        unsigned            bundleIdx,
                        unsigned            opIdx)
    {
        pNode      node = NodeFactory::createGenericTPCNode(inputs,
                                                       outputs,
                                                       nullptr,
                                                       "relu_fwd_bf16",
                                                       name + "_" + std::to_string(bundleIdx));
        BundleInfo info(bundleIdx, UNDEFINED, opIdx);
        node->getNodeAnnotation().bundleInfo.set(info);
        return node;
    }

    pNode createReductionNode(const std::string&  name,
                              const TensorVector& inputs,
                              const TensorVector& outputs,
                              unsigned            bundleIdx,
                              unsigned            opIdx)
    {
        pNode      node = NodeFactory::createNode(inputs,
                                             outputs,
                                             nullptr,
                                             NodeFactory::reductionNodeTypeName,
                                             name + "_" + std::to_string(bundleIdx));
        BundleInfo info(bundleIdx, UNDEFINED, opIdx);
        node->getNodeAnnotation().bundleInfo.set(info);
        return node;
    }

    NodeList createBundleGraph(unsigned bundleIdx)
    {
        pTensor t1 = createTensor();
        pTensor t2 = createTensor();
        pNode   n1 = createTpcNode("N1", {t1}, {t2}, bundleIdx, 0);
        EXPECT_TRUE(GraphEditor::addNode(m_graph, n1));

        pTensor t3 = createTensor();
        pNode   n2 = createTpcNode("N2", {t2}, {t3}, bundleIdx, 1);
        EXPECT_TRUE(GraphEditor::addNode(m_graph, n2));

        pTensor t4 = createTensor(true);
        pNode   n3 = createTpcNode("N3", {t3}, {t4}, bundleIdx, 2);
        EXPECT_TRUE(GraphEditor::addNode(m_graph, n3));

        pTensor t5 = createTensor();
        pTensor t6 = createTensor(true);
        pNode   n4 = createTpcNode("N4", {t5}, {t6}, bundleIdx, 3);
        EXPECT_TRUE(GraphEditor::addNode(m_graph, n4));

        pTensor t7 = createTensor();
        pTensor t8 = createTensor(true);
        pNode   n5 = createTpcNode("N5", {t7}, {t8}, bundleIdx, 4);
        EXPECT_TRUE(GraphEditor::addNode(m_graph, n5));

        pTensor t9     = createTensor(true);
        pNode   memset = NodeFactory::createNode({},
                                               {t9},
                                               nullptr,
                                               NodeFactory::memsetNodeTypeName,
                                               "memset_" + std::to_string(bundleIdx));
        EXPECT_TRUE(GraphEditor::addNode(m_graph, memset));

        pTensor t10       = createTensor(true);
        NodePtr reduction = createReductionNode("reduction", {t4, t6, t8, t9}, {t10}, bundleIdx, 5);
        EXPECT_TRUE(GraphEditor::addNode(m_graph, reduction));

        pTensor t11 = createTensor();
        pNode   n6  = createTpcNode("N6", {t10}, {t11}, bundleIdx, 6);
        EXPECT_TRUE(GraphEditor::addNode(m_graph, n6));

        // The memset is expected to be scheduled before the first node in the bundle.
        // Other bundle nodes should be scheduled according to operation idx.
        NodeList expectedExecOrder = {memset, n1, n2, n3, n4, n5, reduction, n6};
        return expectedExecOrder;
    }

    GraphType m_graph;

    void reduction_memset_scheduling()
    {
        const unsigned numBundles = 5;
        NodeVector     expectedExecOrder;

        for (auto i = 0; i < numBundles; i++)
        {
            const NodeList& expectedBundleExecOrder = createBundleGraph(i);
            expectedExecOrder.insert(expectedExecOrder.end(),
                                     expectedBundleExecOrder.begin(),
                                     expectedBundleExecOrder.end());
        }

        ASSERT_TRUE(m_graph.getExeSortedNodes() == expectedExecOrder);
    }
};
