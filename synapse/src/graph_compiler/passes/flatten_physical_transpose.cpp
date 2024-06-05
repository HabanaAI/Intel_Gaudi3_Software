#include "habana_graph.h"
#include "log_manager.h"
#include "node_factory.h"
#include "habana_global_conf.h"
#include "transpose_node.h"
#include "utils.h"
#include "transpose_utils.h"

using gc::Permutation;

typedef std::shared_ptr<MmeTransposeNode> pPhysicalTransposeNode;

bool flattenPhysicalTranspose(HabanaGraph &graph)
{
    NodeVector nodes = graph.getExeSortedNodes();
    for (const NodePtr &node : nodes)
    {
        pPhysicalTransposeNode physicalTransposeNode = std::dynamic_pointer_cast<MmeTransposeNode>(node);
        if (physicalTransposeNode == nullptr)
        {
            continue;
        }
        //add lowering and de-lowering before and after transpose node
        TensorPtr IFM = physicalTransposeNode->getInput(TENSOR_IFM);
        TensorPtr OFM = physicalTransposeNode->getOutput(TENSOR_OFM);

        const std::string& name = IFM->getName();

        // 1. perform lowering before the mme physical transpose operation

        // 1a. calculate the lowering dimensions
        unsigned axis = -1;
        //Todo SW-30161 Need to handle all reshapes within transpose for all nodes created by transpose creator
        NSizeArray nodeIfmReshapedSizes = lowerPhysicalTransposeTo2d(*IFM, physicalTransposeNode->permutation(), axis);

        // 1b. generate the reshape node with with the lowered shape
        insertReshapeNodeAfter(graph, IFM, physicalTransposeNode, name + "_r0", nodeIfmReshapedSizes,
                insertNodeLocation::AFTER_INPUT);

        // 2. perform de-lowering after the mme physical transpose operation

        // 2a. switch the FCD and spatial dimensions
        std::swap(nodeIfmReshapedSizes[DIM_C], nodeIfmReshapedSizes[DIM_W]);

        // 2b. generate the reshape node with the lowered shape
        insertReshapeNodeAfter(graph, OFM, physicalTransposeNode, name + "_r1", nodeIfmReshapedSizes,
                insertNodeLocation::AFTER_OUTPUT);
    }

    return true;
}