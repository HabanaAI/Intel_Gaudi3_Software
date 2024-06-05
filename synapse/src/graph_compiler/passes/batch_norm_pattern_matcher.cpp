#include "batch_norm_pattern_matcher.h"

InstanceNormToBatchNormPatternMatcher::InstanceNormToBatchNormPatternMatcher(const HabanaGraph& graph) : m_graph(graph)
{
    if (!GCFG_SKIP_BN_SPLIT_FOR_IN_REDUCED_TO_BN.value())
    {
        return;
    }

    for (const auto& node : graph.getNodes())
    {
        if (!matchBatchNormSubPattern(node)) continue;

        const auto& splitIfm    = matchSplitSubPattern(node, IFM_INPUT_IDX);
        const auto& splitGradIn = matchSplitSubPattern(node, GRAD_IN_INPUT_IDX);
        const auto& splitMean   = matchSplitSubPattern(node, MEAN_INPUT_IDX);
        const auto& splitIstd   = matchSplitSubPattern(node, ISTD_INPUT_IDX);
        const auto& gamma       = node->getInput(GAMMA_INPUT_IDX);  // Gamma tensor is not splitted

        const auto& concatGradOut   = matchConcatSubPattern(node, GRAD_OUT_OUTPUT_IDX);
        const auto& concatGradBeta  = matchConcatSubPattern(node, GRAD_BETA_OUTPUT_IDX);
        const auto& concatGradGamma = matchConcatSubPattern(node, GRAD_GAMMA_OUTPUT_IDX);

        if (splitIfm && splitGradIn && splitMean && splitIstd && gamma && concatGradOut && concatGradBeta &&
            concatGradGamma)
        {
            const auto& ifmSplitOutputTensors = (*splitIfm)->getOutputs();
            bool        batchSizeOne =
                std::all_of(ifmSplitOutputTensors.begin(), ifmSplitOutputTensors.end(), [](const TensorPtr& t) {
                    return (t && (t->getDim() >= 1) &&
                            (t->getSizeInElements(t->getDim() - 1) == 1));  // Checks if outer dim = 1
                });
            if (batchSizeOne)
            {
                m_operandsToNodes[{*splitIfm,
                                   *splitGradIn,
                                   *splitMean,
                                   *splitIstd,
                                   gamma,
                                   *concatGradOut,
                                   *concatGradBeta,
                                   *concatGradGamma}]
                    .emplace_back(node);
            }
        }
    }

    for (const auto& [operands, nodes] : m_operandsToNodes)
    {
        const auto concurrencyLevel = nodes.size();
        LOG_DEBUG(GC,
                  "InstanceNormToBatchNormPatternMatcher: found matched pattern for IFM split {}, concurrency level = "
                  "{}, Nodes: ",
                  std::get<0>(operands)->getNodeName(),
                  concurrencyLevel);
        for (const auto& n : nodes)
        {
            LOG_DEBUG(GC, "      {}", n->getNodeName());
            m_matchedNodes.insert({n, concurrencyLevel});
        }
    }
}

std::pair<bool, TSize> InstanceNormToBatchNormPatternMatcher::matchPattern(const NodePtr& node) const
{
    auto it = m_matchedNodes.find(node);
    if (it == m_matchedNodes.end()) return {false, 0};
    return {true, it->second};
}

bool InstanceNormToBatchNormPatternMatcher::matchBatchNormSubPattern(const NodePtr& node) const
{
    if (!node) return false;
    if (std::find(m_supportedBNGuids.begin(), m_supportedBNGuids.end(), node->getGUID()) == m_supportedBNGuids.end())
    {
        return false;
    }
    if (node->getNumInputs() != NUM_INPUTS) return false;
    if (node->getNumOutputs() != NUM_OUTPUTS) return false;

    return true;
}

std::optional<NodePtr> InstanceNormToBatchNormPatternMatcher::matchSplitSubPattern(const NodePtr& node,
                                                                                   unsigned       inputIdx) const
{
    const auto& inputProducer = m_graph.getTensorProducer(node->getInput(inputIdx));
    if (!inputProducer || (inputProducer->getNodeType() != Node::TYPE_INTERNAL_SPLIT)) return {};
    const auto& splitNode = std::dynamic_pointer_cast<SplitNode>(inputProducer);
    // splitNode might be null for SplitFcdNode
    if (splitNode && (splitNode->getAggregationDim() == (splitNode->getInput(0)->getDim() - 1)))
    {
        return splitNode;  // Split on outer dim
    }
    return {};
}

std::optional<NodePtr> InstanceNormToBatchNormPatternMatcher::matchConcatSubPattern(const NodePtr& node,
                                                                                    unsigned       outputIdx) const
{
    const auto& outputConsumers = m_graph.getTensorConsumers(node->getOutput(outputIdx));
    if (outputConsumers.size() != 1) return {};
    if (outputConsumers.front()->getNodeType() != Node::TYPE_INTERNAL_CONCAT) return {};
    const auto& concatNode = std::dynamic_pointer_cast<ConcatenateNode>(outputConsumers.front());
    // concatNode might be null for ConcatFcdNode
    if (concatNode && (concatNode->getAggregationDim() == (concatNode->getOutput(0)->getDim() - 1)))
    {
        return concatNode;  // Concat on outer dim
    }
    return {};
}