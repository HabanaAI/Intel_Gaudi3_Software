#include "flash_attention_slicer.h"
#include "node_factory.h"

void FlashAttentionSlicer::sliceFlashAttentionNodes()
{
    if (!GCFG_ENABLE_FLASH_ATTENTION_SLICING.value())
    {
        return;
    }
    const auto nodes = m_graph.getNodes();  // Copy graph nodes since they may be changed during the iteration
    for (const auto& node : nodes)
    {
        if (!matchPattern(node)) continue;

        auto slicedNodes = sliceNode(node);
        auto splitNodes  = splitInputs(node, slicedNodes);
        auto concatNodes = concatOutputs(node, slicedNodes);

        std::move(splitNodes.begin(), splitNodes.end(), std::back_inserter(slicedNodes));
        std::move(concatNodes.begin(), concatNodes.end(), std::back_inserter(slicedNodes));

        auto res = GraphEditor::replaceNodes(m_graph, {node}, slicedNodes);
        HB_ASSERT(REPLACE_NODE_SUCCESS == res, "Failed to slice flash attention node {}", node->getNodeName());
    }
}

bool FlashAttentionSlicer::matchPattern(const NodePtr& node) const
{
    if (std::find(m_guids.begin(), m_guids.end(), node->getGUID()) == m_guids.end())
    {
        return false;
    }
    for (const auto& t : node->getOperands())
    {
        if (!t) continue;
        if ((t->getDim() - 1) != m_slicedDim)  // Support slicing on outer batch dim only
        {
            return false;
        }
        if (t->getSizeInElements(m_slicedDim) <= 1)  // Nothing to slice
        {
            return false;
        }
    }
    return true;
}

TSize FlashAttentionSlicer::numSlices(const NodePtr& node) const
{
    HB_ASSERT_PTR(node->getInput(0));
    return node->getInput(0)->getSizeInElements(m_slicedDim);
}

NodeVector FlashAttentionSlicer::sliceNode(const NodePtr& node) const
{
    NodeVector slicedNodes;
    for (auto i = 0; i < numSlices(node); i++)
    {
        NodePtr slicedNode = node->cloneWithTensors();
        slicedNode->setName(fmt::format("{}_fa_slice_{}", node->getNodeName(), std::to_string(i)));
        for (auto& operand : slicedNode->getOperands())
        {
            if (!operand) continue;
            HB_ASSERT(operand->getSizeInElements(m_slicedDim) == numSlices(node),
                      "Expected the same sliced dim size for all operands ({})",
                      numSlices(node));
            auto sliceSize            = operand->getAllSizesInElements();
            sliceSize.at(m_slicedDim) = 1;
            operand->reshape(operand->getDim(), sliceSize.data());
        }
        slicedNodes.push_back(slicedNode);
    }
    return slicedNodes;
}

NodeVector FlashAttentionSlicer::splitInputs(const NodePtr& node, const NodeVector& slicedNodes) const
{
    NodeVector splitNodes;
    for (auto inIdx = 0; inIdx < node->getNumInputs(); inIdx++)
    {
        auto input = node->getInput(inIdx);
        if (!input) continue;
        TensorVector outputs;
        for (const auto& slice : slicedNodes)
        {
            outputs.push_back(slice->getInput(inIdx));
        }
        synSplitParams params    = {.axis = m_slicedDim};
        auto           splitNode = NodeFactory::createNode({input},
                                                 outputs,
                                                 &params,
                                                 NodeFactory::splitNodeInternalTypeName,
                                                 "split_" + input->getName());
        splitNodes.push_back(splitNode);
    }
    return splitNodes;
}

NodeVector FlashAttentionSlicer::concatOutputs(const NodePtr& node, const NodeVector& slicedNodes) const
{
    NodeVector concatNodes;
    for (auto outIdx = 0; outIdx < node->getNumOutputs(); outIdx++)
    {
        auto output = node->getOutput(outIdx);
        if (!output) continue;
        TensorVector inputs;
        for (const auto& slice : slicedNodes)
        {
            inputs.push_back(slice->getOutput(outIdx));
        }
        synConcatenateParams params     = {.axis = m_slicedDim};
        auto                 concatNode = NodeFactory::createNode(inputs,
                                                  {output},
                                                  &params,
                                                  NodeFactory::concatenateNodeInternalTypeName,
                                                  "concat_" + output->getName());
        concatNodes.push_back(concatNode);
    }
    return concatNodes;
}