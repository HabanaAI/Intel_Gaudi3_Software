#include "defs.h"
#include "graph_editor.h"
#include "log_manager.h"
#include "node.h"
#include "node_factory.h"
#include "habana_graph.h"
#include "habana_nodes.h"
#include "habana_pass.h"
#include "types.h"
#include "llvm/small_vector.h"
#include <cstdint>

// Create a sequence of expand dim nodes, such that the rank of the MME inputs will be the same (full broadcast)
// m_name - MME node name
// tensorToExpnad - The MME input to expand
// numNewDims - number of new dims to add
NodeVector createExpandDimsNodes(const std::string& m_name, TensorPtr tensorToExpand, unsigned int numNewDims)
{
    NodeVector nodes;

    for (auto i = 0; i < numNewDims; i++)
    {
        auto expandedTensor = tensorToExpand->clone(false, false);
        expandedTensor->resizeDims(tensorToExpand->getDim() + 1);
        synExpandDimsParams expandDimsParams = {0};
        expandDimsParams.axis                = tensorToExpand->getDim();

        NodePtr expandNode = NodeFactory::createNode({tensorToExpand},
                                                     {expandedTensor},
                                                     &expandDimsParams,
                                                     NodeFactory::expandDimsNodeTypeName,
                                                     fmt::format("{}_expand{}", m_name, expandDimsParams.axis));

        HB_ASSERT(expandNode != nullptr, "Failed to allocate expand node");
        // push the newest node last, such that dims expansion grows with the vector index
        nodes.push_back(expandNode);
        tensorToExpand = expandedTensor;
    }

    return nodes;
}

bool alignAsymmetricBgemm(HabanaGraph& g)
{
    if (!GCFG_ALIGN_BATCH_GEMM_DIMS.value())
    {
        return true;
    }

    const NodeVector nodes = g.getExeSortedNodes();

    for (const NodePtr& n : nodes)
    {
        // Check if node is asymmetric batch-gemm with implict broadcast operand
        if (n->isBatchGemm())
        {
            auto bgemm = std::dynamic_pointer_cast<BatchGemmNode>(n);
            HB_ASSERT_PTR(bgemm);

            if (bgemm->isFullBroadcastLayout() && (bgemm->getInput(0)->getDim() != bgemm->getInput(1)->getDim()))
            {
                int          input0Rank                = bgemm->getInput(0)->getDim();
                int          input1Rank                = bgemm->getInput(1)->getDim();
                unsigned int numOfImplictBroadcastDims = std::abs(input0Rank - input1Rank);
                unsigned int broadcastOperandIdx       = (input0Rank > input1Rank) ? 1 : 0;

                LOG_DEBUG(GC,
                          "Handling implicit batch-gemm node {}. Num of implict broadcast dims is {}",
                          bgemm->getNodeName(),
                          numOfImplictBroadcastDims);

                // Create a vector of expand dims nodes, ordered by their data dependency, from bgemm producer to bgemm
                // input
                auto expandNodes = createExpandDimsNodes(bgemm->getNodeName(),
                                                         n->getInput(broadcastOperandIdx),
                                                         numOfImplictBroadcastDims);
                // attach the last expand dims output to the bgemm input
                GraphEditor::replaceInput(g, bgemm, broadcastOperandIdx, expandNodes.back()->getOutput(0));
                auto rc = GraphEditor::addNodes(g, expandNodes);
                HB_ASSERT(rc == true, "failed to add expand dim node to the graph");
                for (NodePtr& node : expandNodes)
                {
                    // Maintain tracking of origin nodes for debug purposes
                    node->setOriginNodes(n->getOriginNodes());
                }
            }
        }
    }
    return true;
}