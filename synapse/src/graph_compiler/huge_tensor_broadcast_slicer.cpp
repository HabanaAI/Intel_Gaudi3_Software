#include "huge_tensor_broadcast_slicer.h"

#include "broadcast_node.h"
#include "broadcast_node_creator.h"
#include "code_generation/tensor_size_validator.h"
#include "habana_graph.h"
#include "node.h"
#include "node_factory.h"
#include "tensor.h"
#include "compilation_hal_reader.h"

HugeTensorBroadcastSlicer::HugeTensorBroadcastSlicer(BroadcastNode*                       node,
                                                     const OptionalTensorSplitSuggestion& splitPattern)
: m_node(node), m_splitPattern(splitPattern)
{
    const auto& in  = node->getInput(0);
    const auto& out = node->getOutput(0);
    for (unsigned dim = 0; dim < out->getDim(); ++dim)
    {
        m_isBroadcastedDim.push_back(in->getSizeInElements(dim) != out->getSizeInElements(dim));
    }
}

bool HugeTensorBroadcastSlicer::isHugeTensorForBroadcast(BroadcastNode*    node,
                                                         const TensorPtr&  t,
                                                         const NSizeArray& sizes)
{
    const auto          broadcastEngineType = CompilationHalReader::getHalReader()->getBroadcastEngine();
    TensorSizeValidator validator(static_cast<unsigned>(SPDLOG_LEVEL_TRACE));
    return !validator.validateTensor(node->shared_from_this(),
                                     t,
                                     sizes,
                                     calculateDefaultStrides(t->getElementSizeInBytes(), sizes),
                                     broadcastEngineType);
}

bool HugeTensorBroadcastSlicer::isHugeTensorForBroadcast(const NSizeArray& sizes) const
{
    return isHugeTensorForBroadcast(m_node, m_node->getOutput(0), sizes);
}

bool HugeTensorBroadcastSlicer::doesRequireSlicing(BroadcastNode* node)
{
    const auto& t     = node->getOutput(0);
    const auto& sizes = t->getAllNSizesInElements();
    return isHugeTensorForBroadcast(node, t, sizes);
}

bool HugeTensorBroadcastSlicer::existsValidSplitPattern() const
{
    return m_splitPattern.has_value() && m_splitPattern->tensor == m_node->getOutput(0) &&
           !isHugeTensorForBroadcast(m_splitPattern->chunkSize);
}

NodeVector HugeTensorBroadcastSlicer::slice()
{
    LOG_DEBUG(HUGE_TENSOR_SLICE, "start to slice {} broadcast node", m_node->getNodeName());
    if (existsValidSplitPattern())
    {
        LOG_TRACE(HUGE_TENSOR_SLICE, "valid split suggestion provided");
    }
    else
    {
        LOG_TRACE(HUGE_TENSOR_SLICE, "valid split suggestion not provided, generate split pattern");
        const TensorPtr& output    = m_node->getOutput(0);
        auto             chunkSize = output->getAllNSizesInElements();

        bool found = false;
        for (int dim = output->getDim() - 1; dim >= 0; --dim)
        {
            chunkSize.at(dim) = 1;
            if (isHugeTensorForBroadcast(chunkSize))
            {
                LOG_TRACE(HUGE_TENSOR_SLICE, "slice on dim {}, the remaining size is still huge tensor", dim);
                continue;
            }
            LOG_TRACE(HUGE_TENSOR_SLICE, "slice on dim {}, the remaining size is not huge tensor", dim);

            // Restore the original size of the sliced dimension
            chunkSize.at(dim) = output->getSizeInElements(dim);
            // Find the optimal slice size
            chunkSize.at(dim) = findOptimalChunkSize(chunkSize, dim);
            found             = true;
            break;
        }
        HB_ASSERT(found, "no split found for tensor");
        m_splitPattern = {output, chunkSize};
    }

    return sliceBroadcast();
}

TSize HugeTensorBroadcastSlicer::findOptimalChunkSize(const NSizeArray& chunkSize, const unsigned dim) const
{
    std::function<bool(const TSize&)> isValidChunkSize = [&](const TSize& val) {
        NSizeArray tmp = chunkSize;
        tmp.at(dim)    = val;
        return !isHugeTensorForBroadcast(tmp);
    };
    return HugeTensorNodeSlicerBase::findOptimalChunkSize(chunkSize, dim, isValidChunkSize);
}

NodeVector HugeTensorBroadcastSlicer::sliceBroadcast()
{
    HB_ASSERT(existsValidSplitPattern(), "Invalid split pattern");

    LOG_TRACE(HUGE_TENSOR_SLICE,
              "slice broadcast to new broadcasts with chunk size: [{}]",
              fmt::join(m_splitPattern->chunkSize.begin(),
                        m_splitPattern->chunkSize.begin() + m_splitPattern->tensor->getDim(),
                        ","));

    const auto& input  = m_node->getInput(0);
    const auto& output = m_node->getOutput(0);

    TensorSplitSuggestion& outputSplitPattern = m_splitPattern.value();
    TensorSplitSuggestion  inputSplitPattern {.tensor = input, .chunkSize = outputSplitPattern.chunkSize};

    // For the input we don't need to slice on broadcasted dimensions
    for (unsigned dim = 0; dim < output->getDim(); ++dim)
    {
        if (BroadcastNodeCreator::isTrivialDim(input, dim))
        {
            inputSplitPattern.chunkSize.at(dim) = 1;
        }
    }

    auto [splitNodes, newInputs]   = sliceTensor(inputSplitPattern, true /* isInput */);
    auto [concatNodes, newOutputs] = sliceTensor(outputSplitPattern, false /* isInput */);

    NodeVector res = splitNodes;
    res.insert(res.end(), concatNodes.begin(), concatNodes.end());

    std::optional<SlicedTensorVector> newShapes;
    if (m_node->isDynamicShape())
    {
        newShapes         = SlicedTensorVector();
        const auto& shape = m_node->getInput(1);
        HB_ASSERT_PTR(shape);

        TensorSplitSuggestion shapeSplitPattern {.tensor = shape, .chunkSize = outputSplitPattern.chunkSize};
        NodeVector            splitShapeNodes;
        std::tie(splitShapeNodes, newShapes.value()) = sliceTensor(shapeSplitPattern, true /* isInput */);
        res.insert(res.end(),
                   std::make_move_iterator(splitShapeNodes.begin()),
                   std::make_move_iterator(splitShapeNodes.end()));
    }

    const auto& linkedTensorsVector = linkOutputsToInputs(newInputs, newOutputs, newShapes);
    HB_ASSERT(linkedTensorsVector.size() == newOutputs.size(), "Size mismatch");

    unsigned idx = 0;
    for (const auto& [newOutput, inputs] : linkedTensorsVector)
    {
        res.push_back(NodeFactory::createNode(inputs,
                                              {newOutput},
                                              nullptr,
                                              NodeFactory::broadcastNodeTypeName,
                                              fmt::format("{}/{}", m_node->getNodeName(), idx++)));
    }
    return res;
}

// When we split on broadcasted dimension, the input is not sliced on that dimension, so the the number of inputs
// is less than the number of outputs, and we need to match each output to the corresponding input
HugeTensorBroadcastSlicer::LinkedTensorsVector
HugeTensorBroadcastSlicer::linkOutputsToInputs(const SlicedTensorVector&                inputs,
                                               const SlicedTensorVector&                outputs,
                                               const std::optional<SlicedTensorVector>& shapes) const
{
    HB_ASSERT(!shapes.has_value() || shapes->size() == outputs.size(),
              "if shape tensors exists, then must be one to each output");

    HB_ASSERT(
        std::is_sorted(inputs.begin(),
                       inputs.end(),
                       [](const SlicedTensor& a, const SlicedTensor& b) { return a.coordinates < b.coordinates; }),
        "The inputs must be sorted by coordinates");

    LinkedTensorsVector res;

    const auto& maxCoord = inputs.back().coordinates;
    for (unsigned i = 0; i < outputs.size(); ++i)
    {
        // Calculate the "strides" that needed to "flatten" N-dim array into 1-dim array
        // i.e. arr[i][j][k] -> arr[i * StrideA + j * StrideB + k]
        uint64_t index = 0;
        for (uint64_t dim = 0, stride = 1; dim < outputs[i].coordinates.size(); ++dim)
        {
            if (dim != 0)
            {
                stride *= maxCoord[dim - 1] + 1;
            }
            // If the output is sliced on a broadcasted dim, it mean that the input is not sliced on that dim,
            // sw, we need to stay in the same coordinates.
            if (maxCoord[dim] != 0)
            {
                index += stride * outputs[i].coordinates[dim];
            }
        }

        auto input = inputs.at(index).tensor;
        res.emplace_back(outputs[i].tensor, TensorVector({input}));
        if (shapes.has_value())
        {
            HB_ASSERT(shapes.value()[i].coordinates == outputs[i].coordinates, "Location mismatch");
            res.back().second.push_back(shapes.value()[i].tensor);
        }
    }
    return res;
}