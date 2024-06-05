#include "passes.h"
#include "habana_graph.h"
#include "reduction_node.h"

bool markReductionInputs(HabanaGraph& g)
{
    for (NodePtr node : g.getExeSortedNodes())
    {
        auto reductionNode = std::dynamic_pointer_cast<ReductionNode>(node);
        if (reductionNode == nullptr) continue;

        for (TensorPtr tensor : node->getInputs())
        {
            // At this pass, all input tensors are annotated, so that tpc glue code can work.
            // In a later pass (after tpc kernels were loaded and the graph is stable), the
            // tensor from the first producer should be "unannotated"
            // REDUCTION_SET does not require HW support, therefore we disable the isReductionEnabled flag
            TensorAnnotation &ann = tensor->getTensorAnnotation();
            auto              reductionOp              = reductionNode->getReductionOperation();
            ann.tensorReductionInfo.reductionOperation = reductionOp;
            ann.tensorReductionInfo.isReductionEnabled = reductionOp != REDUCTION_SET;
        }
    }

    return true;
}
