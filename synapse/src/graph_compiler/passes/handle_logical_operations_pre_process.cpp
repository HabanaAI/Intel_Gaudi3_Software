
#include "habana_graph.h"
#include "node.h"
#include "operand_reuse_logical_node.h"
#include "input_reuse.h"

// right now we support only the base case, where there is only one
// reuseable input (so there is also one reuseable output).
void replaceInputReuseNodesNodes(HabanaGraph& g)
{
    NodeVector operandReuseNodes;
    // Since we replace nodes in the graph we can't iterate over the cached nodes that returned from "g.getNodes()",
    // to avoid copy all the nodes, at first we copy only the nodes that needed to be handle, but not handle them yet.
    for (const auto& node : g.getNodes())
    {
        const auto& reuseMap = node->getReusableInputBinding();
        // Check if there is only one reusable output.
        if (reuseMap.size() != 1)
        {
            if (reuseMap.size() > 1)
            {
                LOG_WARN(OPT_LOGICAL_OPS,
                         "{}: there are {} reused outputs, skip replace",
                         node->getNodeName(),
                         reuseMap.size());
            }
            continue;
        }

        const auto& [output, inputs] = *reuseMap.begin();
        // Check if there is only one reusable input.
        if (inputs.size() != 1)
        {
            LOG_WARN(OPT_LOGICAL_OPS,
                     "{}: there are {} input candidates for reuse, skip replace",
                     node->getNodeName(),
                     inputs.size());
            continue;
        }

        // If tensors are already reuse, nothing to do
        if (InputInplaceReuse::isAlreadyReused(g, inputs.front(), *node)) continue;
        // If the input and output allocated in different locations (DRAM/SRAM) we can't handle them as logical node
        if (inputs.front()->location() != output->location())
        {
            LOG_DEBUG(OPT_LOGICAL_OPS,
                      "{}: the input and the output allocated in different memory types, skip replace",
                      node->getNodeName());
            continue;
        }
        operandReuseNodes.push_back(node);
    };

    // Here we actually handle the nodes.
    if (!operandReuseNodes.empty())
    {
        LOG_DEBUG(OPT_LOGICAL_OPS, "start to replace input reuse nodes with internal logical nodes");
        for (const auto& node : operandReuseNodes)
        {
            const auto reuseMap          = node->getReusableInputBinding();
            const auto& [output, inputs] = *reuseMap.begin();

            const auto inputIndex  = node->getInputIndexOfTensor(inputs.front());
            const auto outputIndex = node->getOutputIndexOfTensor(output);

            LOG_TRACE(OPT_LOGICAL_OPS,
                      "Handle reuse input binding for node {}: input index: {}, output index: {}",
                      node->getNodeName(),
                      inputIndex,
                      outputIndex);

            auto newNode = std::make_shared<OperandReuseInternalLogicalNode>(node, inputIndex, outputIndex);
            auto res     = GraphEditor::replaceNodes(g, {node}, {newNode});
            HB_ASSERT(res == REPLACE_NODE_SUCCESS, "Failed to replace node");
        }
    }
}

bool handleLogicalOpsPreProcess(HabanaGraph& g)
{
    if (GCFG_ENABLE_INPUT_REUSE_AS_LOGICAL_NODE.value())
    {
        replaceInputReuseNodesNodes(g);
    }
    return true;
}