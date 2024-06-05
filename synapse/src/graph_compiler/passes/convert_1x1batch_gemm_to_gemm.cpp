#include "passes.h"  // convert1x1BatchGemmToGemm

#include "graph_editor.h"  // GraphEditor
#include "habana_graph.h"  // Habana Graph
#include "log_manager.h"   // GC, LOG_ERR
#include "node_factory.h"  // NodeFactory

bool convert1x1BatchGemmToGemm(HabanaGraph& g)
{
    // Planned BatchGemm->Gemm replacements
    std::vector<std::pair<NodePtr, NodePtr>> changes;

    for (const NodePtr& node : g.getNodes())
    {
        const char* gemmType = nullptr;
        switch(node->getNodeType())
        {
            case Node::TYPE_BATCH_GEMM:         gemmType = "gemm";      break;
            case Node::TYPE_BATCH_GEMM_DEDX:    gemmType = "gemm_dedx"; break;
            case Node::TYPE_BATCH_GEMM_DEDW:    gemmType = "gemm_dedw"; break;
            default:
                continue;
        }

        const TensorPtr& opA = node->getInput(TENSOR_IFM);
        const TensorPtr& opB = node->getInput(TENSOR_WEIGHT);
        const TensorPtr& opC = node->getOutput(TENSOR_OFM);

        {
            bool is1x1batchGemm = true;
            for (auto i = (unsigned) DIM_GEMM_BATCH; i < opC->getDim(); ++i)
            {
                if (opC->getSizeInElements(i) != 1) is1x1batchGemm = false;
            }
            if (!is1x1batchGemm) continue;
        }
        BatchGemmNode* batchGemmNode = static_cast<BatchGemmNode*>(node.get());
        const auto&    params        = batchGemmNode->getGEMMParams();
        const auto     newNode =
            NodeFactory::createNode({opA, opB}, {opC}, &params, gemmType, node->getNodeName() + "_gemm");
        newNode->getNodeAnnotation() = node->getNodeAnnotation();

        const MmeExpBias& mmeExpBias = batchGemmNode->getMmeExpBias();
        MMENodePtr mmeNode    = std::dynamic_pointer_cast<MmeNode>(newNode);
        HB_ASSERT(mmeNode, "could not downcast Node to MME Node");
        mmeNode->setMmeExpBias(mmeExpBias);

        changes.emplace_back(node, newNode);
    }

    for (const auto& change: changes)
    {
        if (GraphEditor::replaceNodes(g, {change.first}, {change.second}) != REPLACE_NODE_SUCCESS)
        {
            LOG_ERR(GC, "extract batch gemm node {} could not be completed", change.first->getNodeName());
            return false;
        }
    }

    return true;
}