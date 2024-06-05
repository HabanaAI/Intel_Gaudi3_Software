#include "huge_tensor_node_slicer.h"
#include "tensor.h"
#include "node_factory.h"

std::pair<NodeVector, HugeTensorNodeSlicerBase::SlicedTensorVector>
HugeTensorNodeSlicerBase::sliceTensor(const TensorSplitSuggestion& suggestion, bool isInput)
{
    const unsigned tensorDim        = suggestion.tensor->getDim();
    const auto&    chunkSize        = suggestion.chunkSize;
    const auto&    originalMaxSizes = suggestion.tensor->getNSizesInElements();
    const auto&    originalMinSizes = suggestion.tensor->getNMinimalSizesInElements();
    const bool     isShapeTensor    = suggestion.tensor->isShapeTensor();

    NCoordArray coordinates;
    coordinates.fill(0);

    SlicedTensorVector returnedTensors = {{suggestion.tensor, coordinates}};
    NodeVector         returnedNodes;

    unsigned slicedDims = 0;
    for (int dim = tensorDim - 1; dim >= 0; --dim)
    {
        if (chunkSize.at(dim) == originalMaxSizes.at(dim)) continue;
        ++slicedDims;

        HB_ASSERT(chunkSize.at(dim) < originalMaxSizes.at(dim), "Chunk size is bigger than original size");
        unsigned numOfTensors = div_round_up(originalMaxSizes.at(dim), chunkSize.at(dim));

        SlicedTensorVector toSplit;
        std::swap(toSplit, returnedTensors);

        for (const auto& [t, location] : toSplit)
        {
            TSize        availableMaxTensorSize = originalMaxSizes.at(dim);
            TSize        availableMinTensorSize = originalMinSizes.at(dim);
            auto         maxSizes               = t->getNSizesInElements();
            auto         minSizes               = t->getNMinimalSizesInElements();
            TensorVector newTensors;
            for (unsigned i = 0; i < numOfTensors; ++i)
            {
                TensorPtr newTensor = t->clone(false, false, false);
                newTensor->setName(fmt::format("{}/{}", t->getName(), i));
                maxSizes.at(dim) = std::min(chunkSize.at(dim), availableMaxTensorSize);
                minSizes.at(dim) = std::min(chunkSize.at(dim), availableMinTensorSize);

                availableMaxTensorSize -= maxSizes.at(dim);
                availableMinTensorSize -= minSizes.at(dim);
                newTensor->reshape(tensorDim, maxSizes.data(), nullptr, minSizes.data());

                newTensors.push_back(newTensor);

                auto newLocation    = location;
                newLocation.at(dim) = i;
                returnedTensors.push_back({newTensor, newLocation});
            }
            synAxisParams params = {.axis = dim};
            if (isInput)  // Split node
            {
                returnedNodes.push_back(NodeFactory::createNode({t},
                                                                newTensors,
                                                                &params,
                                                                isShapeTensor ? NodeFactory::splitShapeNodeTypeName
                                                                              : NodeFactory::splitNodeInternalTypeName,
                                                                fmt::format("{}/split", t->getName())));
            }
            else  // Concat node
            {
                HB_ASSERT(!isShapeTensor, "Concat Shape node not implemented");
                returnedNodes.push_back(NodeFactory::createNode(newTensors,
                                                                {t},
                                                                &params,
                                                                NodeFactory::concatenateNodeLogicalInternalTypeName,
                                                                fmt::format("{}/concat", t->getName())));
            }
        }
    }
    LOG_TRACE(HUGE_TENSOR_SLICE,
              "slicing {} tensor {} into {} slices, using {} depth {} nodes",
              isShapeTensor ? "shape" : (isInput ? "input" : "output"),
              suggestion.tensor->getName(),
              returnedTensors.size(),
              slicedDims,
              (isInput ? "split" : "concat"));
    return {returnedNodes, returnedTensors};
}

NStrideArray HugeTensorNodeSlicerBase::calculateDefaultStrides(const unsigned elementSize, const NSizeArray& sizes)
{
    static_assert(NSizeArray().size() + 1 == NStrideArray().size(), "array sizes mismatch");

    NStrideArray res = {0};
    res[0] = elementSize;
    for (unsigned dim = 1; dim <= sizes.size(); ++dim)
    {
        res[dim] = res[dim - 1] * sizes[dim - 1];
    }
    return res;
}

TSize HugeTensorNodeSlicerBase::findOptimalChunkSize(const NSizeArray&                        chunkSize,
                                                     const unsigned                           dim,
                                                     const std::function<bool(const TSize&)>& isValidChuckSize)
{
    HB_ASSERT(chunkSize.at(dim) != 0, "Invalid chunk size");
    TSize maxValidChunkSize = binarySearch<TSize>(1, chunkSize.at(dim), isValidChuckSize);

    TSize       optimalChunkSize;
    const TSize numOfSlices = div_round_up(chunkSize.at(dim), maxValidChunkSize);

    // Find the chunk that has the biggest power of two as divisor.
    // The loop complexity is at most log(maxSize - div_round_up(originalSize, numOfSlices))
    TSize optimalChunkSizeCandidate = div_round_up(chunkSize.at(dim), numOfSlices);  // The most balance split;
    do
    {
        optimalChunkSize = optimalChunkSizeCandidate;
        optimalChunkSizeCandidate += biggestDivisorWhichIsPowerOf2(optimalChunkSizeCandidate);
    } while (optimalChunkSizeCandidate <= maxValidChunkSize);
    LOG_TRACE(HUGE_TENSOR_SLICE,
              "dim: {}, number of slices: {}, slice size:[min: {}, max: {}, optimal: {}]",
              dim,
              numOfSlices,
              div_round_up(chunkSize.at(dim), numOfSlices),
              maxValidChunkSize,
              optimalChunkSize);
    return optimalChunkSize;
}