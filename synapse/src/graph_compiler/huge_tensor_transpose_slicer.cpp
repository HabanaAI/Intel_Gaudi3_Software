#include "huge_tensor_transpose_slicer.h"
#include "tensor.h"
#include "node_factory.h"
#include "habana_graph.h"
#include "node.h"
#include "code_generation/tensor_size_validator.h"
#include "habana_nodes/transpose_node.h"
#include "habana_nodes/transpose_utils.h"
#include "compilation_hal_reader.h"

bool HugeTensorTransposeSlicer::isHugeTensorForTranspose(TransposeNode*    node,
                                                         const TensorPtr&  t,
                                                         const NSizeArray& sizes)
{
    const auto          transposeEngineType = CompilationHalReader::getHalReader()->getTransposeEngine();
    TensorSizeValidator validator(static_cast<unsigned>(SPDLOG_LEVEL_TRACE));
    return !validator.validateTensor(node->shared_from_this(),
                                     t,
                                     sizes,
                                     calculateDefaultStrides(t->getElementSizeInBytes(), sizes),
                                     transposeEngineType);
}

bool HugeTensorTransposeSlicer::isHugeTensorForTranspose(const NSizeArray& sizes) const
{
    if (isHugeTensorForTranspose(m_node, m_node->getInput(0), sizes)) return true;
    auto outSizes = applyPermutationOnSizes(sizes, m_node->permutation());
    return isHugeTensorForTranspose(m_node, m_node->getOutput(0), outSizes);
}

bool HugeTensorTransposeSlicer::doesRequireSlicing(TransposeNode* node)
{
    const auto& in      = node->getInput(0);
    const auto& inSizes = in->getAllNSizesInElements();
    if (isSameDataMemOrder(*in, node->permutation()))
    {
        return false;  // this is actually just a logical reshape
    }
    const auto& out      = node->getOutput(0);
    const auto& outSizes = out->getAllNSizesInElements();
    return isHugeTensorForTranspose(node, in, inSizes) || isHugeTensorForTranspose(node, out, outSizes);
}

NodeVector HugeTensorTransposeSlicer::slice()
{
    LOG_DEBUG(HUGE_TENSOR_SLICE, "start to slice {} transpose node", m_node->getNodeName());

    if (m_splitPattern.has_value() && !isHugeTensorForTranspose(m_splitPattern->chunkSize))
    {
        LOG_TRACE(HUGE_TENSOR_SLICE, "valid split suggestion provided");
    }
    else
    {
        LOG_TRACE(HUGE_TENSOR_SLICE, "valid split suggestion not provided, generate split pattern");
        const TensorPtr& t         = m_node->getInput(0);
        auto             chunkSize = t->getAllNSizesInElements();

        DimVector splitOrder = getTransposeSplitOrder(m_node->permutation());
        LOG_TRACE(HUGE_TENSOR_SLICE, "transpose dimensions split order: [{}]", toString(splitOrder, ','));

        bool found = false;
        for (const auto& dim : splitOrder)
        {
            chunkSize.at(dim) = 1;
            if (isHugeTensorForTranspose(chunkSize))
            {
                LOG_TRACE(HUGE_TENSOR_SLICE, "slice on dim {}, the remaining size is still huge tensor", dim);
                continue;
            }
            LOG_TRACE(HUGE_TENSOR_SLICE, "slice on dim {}, the remaining size is not huge tensor", dim);
            // Restore the original size of the sliced dimension
            chunkSize.at(dim) = t->getSizeInElements(dim);
            // Find the optimal slice size
            chunkSize.at(dim) = findOptimalChunkSize(chunkSize, dim);
            found             = true;
            break;
        }
        HB_ASSERT(found, "no split found for tensor");
        m_splitPattern = {t, chunkSize};
    }

    return sliceTranspose();
}

// Since transpose input and output dimensions order is different, each dimension value is min(input, output)
// so the split order is from "most" outer to inner
// Example: when permutation is [0, 3, 2, 1] (FCD on left), the split order is [2, (1 or 3), (3 or 1), 0]
//      dim = 3, permutation[dim] = 1 -> res = [],        visited = {1}
//      dim = 2, permutation[dim] = 2 -> res = [2],       visited = {1,2}
//      dim = 1, permutation[dim] = 3 -> res = [2,1,3],   visited = {1,2,3}
//      dim = 0, permutation[dim] = 0 -> res = [2,1,3,0], visited = {0,1,2,3}
DimVector HugeTensorTransposeSlicer::getTransposeSplitOrder(const TransposePermutationArray& permutation)
{
    std::set<unsigned> visited;
    DimVector          res;
    for (int dim = permutation.size() - 1; dim >= 0; --dim)
    {
        if (visited.count(dim) == 1)
        {
            res.push_back(dim);
        }
        if (permutation.at(dim) >= dim)
        {
            res.push_back(permutation.at(dim));
        }
        visited.insert(permutation.at(dim));
    }
    return res;
}

TSize HugeTensorTransposeSlicer::findOptimalChunkSize(const NSizeArray& chunkSize, const unsigned dim) const
{
    std::function<bool(const TSize&)> isValidChunkSize = [&](const TSize& val) {
        NSizeArray tmp = chunkSize;
        tmp.at(dim)    = val;
        return !isHugeTensorForTranspose(tmp);
    };
    return HugeTensorNodeSlicerBase::findOptimalChunkSize(chunkSize, dim, isValidChunkSize);
}

NodeVector HugeTensorTransposeSlicer::sliceTranspose()
{
    HB_ASSERT(m_splitPattern.has_value(), "Split pattern not set yet");

    LOG_TRACE(HUGE_TENSOR_SLICE,
              "slice transpose to new transposes with chunk size: [{}]",
              fmt::join(m_splitPattern->chunkSize.begin(),
                        m_splitPattern->chunkSize.begin() + m_splitPattern->tensor->getDim(),
                        ","));

    TensorSplitSuggestion inputSplitPattern;
    TensorSplitSuggestion outputSplitPattern;
    if (m_splitPattern->tensor == m_node->getInput(0))
    {
        inputSplitPattern            = m_splitPattern.value();
        outputSplitPattern.tensor    = m_node->getOutput(0);
        outputSplitPattern.chunkSize = applyPermutationOnSizes(m_splitPattern->chunkSize, m_node->permutation(), false);
    }
    else
    {
        HB_ASSERT(m_splitPattern->tensor == m_node->getOutput(0),
                  "{} is not connected to {}",
                  m_splitPattern->tensor->getName(),
                  m_node->getNodeName());
        outputSplitPattern       = m_splitPattern.value();
        inputSplitPattern.tensor = m_node->getInput(0);
        inputSplitPattern.chunkSize =
            applyPermutationOnSizes(m_splitPattern->chunkSize, inversePermutation(m_node->permutation()), false);
    }

    auto [splitNodes, newInputs]   = sliceTensor(inputSplitPattern, true /* isInput */);
    auto [concatNodes, newOutputs] = sliceTensor(outputSplitPattern, false /* isInput */);

    NodeVector res = splitNodes;
    res.insert(res.end(), concatNodes.begin(), concatNodes.end());

    // Permute the location on the input to find the corresponding output
    for (auto& newInput : newInputs)
    {
        NCoordArray tmp;
        tmp.fill(0);
        applyPermutation(newInput.coordinates.data(), m_node->permutation(), tmp.data());
        newInput.coordinates = std::move(tmp);
    }

    std::sort(newInputs.begin(), newInputs.end());
    std::sort(newOutputs.begin(), newOutputs.end());

    HB_ASSERT(newInputs.size() == newOutputs.size(), "Size mismatch");

    for (unsigned i = 0; i < newInputs.size(); ++i)
    {
        HB_ASSERT(newInputs[i].coordinates == newOutputs[i].coordinates, "Location mismatch");

        synTransposeParamsNDims params = permutationToParams(m_node->permutation());
        res.push_back(NodeFactory::createNode({newInputs[i].tensor},
                                              {newOutputs[i].tensor},
                                              &params,
                                              NodeFactory::transposeNodeTypeName,
                                              fmt::format("{}/{}", m_node->getNodeName(), i)));
    }
    return res;
}
