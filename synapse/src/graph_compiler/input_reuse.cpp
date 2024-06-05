#include "habana_graph.h"
#include "input_reuse.h"
#include "graph_editor.h"
#include "handle_memory_reuse.h"

bool InputInplaceReuse::isAlreadyReused(const HabanaGraph& g, const TensorPtr& input, const Node& node)
{
    for (const TensorPtr& output : node.getOutputs())
    {
        // Check if output already reuses input
        if (Tensor::getRealTensor(output) == Tensor::getRealTensor(input)) return true;
        // Check if output already alias the input
        if (isAlreadyReusedPersistentTensors(g, input, output)) return true;
    }
    return false;
}

bool InputInplaceReuse::isAlreadyReusedPersistentTensors(const HabanaGraph& g,
                                                         const TensorPtr&   input,
                                                         const TensorPtr&   output)
{
    return MemoryReuseHandler::sameMemorySection(input, output) &&
           MemoryReuseHandler::getRealTensorOffset(input) == MemoryReuseHandler::getRealTensorOffset(output);
}

bool InputInplaceReuse::runInputInplaceReuse(HabanaGraph& g)
{
    const NodeVector nodes = g.getExeSortedNodes();

    for (const NodePtr& n : nodes)
    {
        const std::map<TensorPtr, TensorVector, TensorComparator>& reusePairs = getReusePairs(n);
        for (const auto& candidates : reusePairs)
        {
            const TensorPtr&    output         = candidates.first;
            const TensorVector& reusableInputs = candidates.second;
            if (!reusableInputs.empty() && outputViableForInplace(g, n, output))
            {
                if (!applyReuse(g, n, output, reusableInputs))
                {
                    HB_ASSERT(0, "Failed to apply reuse for node {}", n->getNodeName());
                    return false;
                }
            }
        }
    }

    return true;
}