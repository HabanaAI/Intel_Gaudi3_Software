#include <algorithm>
#include <tpc_node.h>
#include "tpc_slicing_blacklist.h"

struct TensorAndDims
{
    unsigned tensorIndex;
    DimSet tensorDims;
    unsigned firstAccessPatternIndex;
    bool isDynamic;
};

static unsigned countDataTensors(const TensorVector& tensors);
static unsigned countShapeTensors(const TensorVector& tensors);
static std::vector<TensorAndDims> findRelevantTensorsAndDims(const TensorVector&                     tensors,
                                                             unsigned                                fromTensorIndex,
                                                             unsigned                                toTensorIndex,
                                                             const tpc_lib_api::TensorAccessPattern* accessPatterns,
                                                             unsigned                                indexSpaceDim);
static bool                       isTensorDimAllRequired(const tpc_lib_api::TensorAccessPattern& accessPattern,
                                                         unsigned                                accessPatternDimIndex,
                                                         uint64_t                                tensorDimSize);
static bool                       isAnyTensorDimAllRequired(const tpc_lib_api::TensorAccessPattern accessPatterns[],
                                                            unsigned                               accessPatternIndex,
                                                            const TensorPtr&                       tensor,
                                                            DimSet                                 dimList);
static bool  indexSpaceDimIsUnsliceable (const TPCNode& node, unsigned indexSpaceDim);

// ================================================================

DimSet getUnsliceableIndexSpaceDims (const TPCNode& node)
{
    DimSet unsliceableDims;
    const auto& instance = node.getInstance();
    for (unsigned i = 0; i < instance.indexSpaceRank; ++i)
    {
        if (indexSpaceDimIsUnsliceable(node, i))
        {
            unsliceableDims.insert(i);
        }
    }
    return unsliceableDims;
}

static bool  indexSpaceDimIsUnsliceable (const TPCNode& node, unsigned indexSpaceDim)
{
    const auto& inputs = node.getInputs();
    const auto& outputs = node.getOutputs();

    // Cannot slice if:
    //
    // 1.    index space dim is mapping dynamic output dim but does not map dynamic input, and
    //       no shape tensor for that tensor.
    // 2.    no shape tensor and inputs dims are allRequired (according to allRequired check –
    //       can be done by using the granularity), and output is not alRequired.
    //       This is actually a generalisation of case 1 so 1 is not checked on its own.
    // 3.    If no shape tensor and all inputs granularity is allRequired (tensor) while the tensor
    //       is > vector size, and output tensors are not all required (currently not implemented)
    //

    auto nShapeTensors = countShapeTensors(inputs);
    auto nOutputs = countDataTensors(outputs);

    if (nShapeTensors >= nOutputs)
    {
        // all outputs are covered by shape tensors, it is OK to slice
        return false;
    }

    auto nDataInputs = countDataTensors(inputs);

    bool someInputsAllRequired = false;
    bool allInputsAllRequired = true;
    bool haveInputs = nDataInputs > 0;

    const auto& instance = node.getInstance();

    // For inputs: only use data tensors (not shape, not auxiliary)
    auto relevantInputs = findRelevantTensorsAndDims(inputs, 0, nDataInputs, instance.inputTensorAccessPattern, indexSpaceDim);
    // For outputs: only use output tensors that do not have corresponding shape tensors (first shape tensor corresponds to first
    // data tensor etc, so start from nShapeTensors)
    auto relevantOutputs = findRelevantTensorsAndDims(outputs, nShapeTensors, nOutputs, instance.outputTensorAccessPattern, indexSpaceDim);

    if (relevantInputs.empty() && !relevantOutputs.empty())
    {
        LOG_WARN(GC, "Node {} index space dimension {} is unsliceable", node.getNodeName(), indexSpaceDim);
        return true;
    }

    return false;

}

// Static functions

static unsigned countDataTensors(const TensorVector& tensors)
{
    unsigned count = std::count_if(tensors.begin(), tensors.end(),
            [](const TensorPtr& tensor){ return tensor->isDataTensor(); });
    return count;
}
static unsigned countShapeTensors(const TensorVector& tensors)
{
    unsigned count = std::count_if(tensors.begin(), tensors.end(),
            [](const TensorPtr& tensor){ return tensor->isShapeTensor(); });
    return count;
}

static std::vector<TensorAndDims> findRelevantTensorsAndDims(const TensorVector&                     tensors,
                                                             unsigned                                fromTensorIndex,
                                                             unsigned                                toTensorIndex,
                                                             const tpc_lib_api::TensorAccessPattern* accessPatterns,
                                                             unsigned                                indexSpaceDim)
{
    std::vector<TensorAndDims> ret;

    unsigned accessPatternIndex = 0;

    // Find access pattern index to start with
    for (unsigned tensorIndex = 0; tensorIndex < fromTensorIndex; tensorIndex++)
    {
        accessPatternIndex += TPCNode::numTensorGlueCodeAccessPatternEntries(tensors[tensorIndex]);
    }

    for (unsigned tensorIndex = fromTensorIndex; tensorIndex < toTensorIndex; ++tensorIndex)
    {
        auto tensor = tensors[tensorIndex];
        for (unsigned dimBase = 0; dimBase < tensor->getDim(); dimBase += tpc_lib_api::MAX_INDEX_SPACE_DIM_SIZE)
        {
            const auto& accessPattern = accessPatterns[accessPatternIndex];

            // data under accessPatterns[accessPatternIndex].dim is only meaningful if
            // it is not an allRequired pattern
            if (!accessPattern.allRequired)
            {
                for (unsigned dimOffset = 0; dimOffset < tpc_lib_api::MAX_INDEX_SPACE_DIM_SIZE; ++dimOffset)
                {
                    if (dimBase + dimOffset >= tensor->getDim()) break;
                    auto tensorDim = dimBase + dimOffset;

                    if (accessPattern.mapping[dimOffset].indexSpaceDim == indexSpaceDim)
                    {
                        auto existing = std::find_if(ret.begin(), ret.end(),
                                [tensorIndex](const TensorAndDims& a) { return a.tensorIndex == tensorIndex; });
                        if (existing == ret.end())
                        {
                            ret.push_back(TensorAndDims{tensorIndex, DimSet(tensorDim), accessPatternIndex, tensor->isDynamicDim(tensorDim)});
                        }
                        else
                        {
                            existing->tensorDims.insert(tensorDim);
                            existing->isDynamic |= tensor->isDynamicDim(tensorDim);
                        }
                    }
                }
            }

            ++accessPatternIndex;
        }
    }

    // remove static tensors
    ret.erase(std::remove_if(ret.begin(), ret.end(),
                [](const TensorAndDims& td) { return !td.isDynamic; }), ret.end());

    // remove all-required tensors
    ret.erase(std::remove_if(ret.begin(), ret.end(),
                [&](const TensorAndDims& td) {
                return isAnyTensorDimAllRequired(accessPatterns, td.firstAccessPatternIndex, tensors[td.tensorIndex], td.tensorDims);
                }), ret.end());

    return ret;
}

static bool isTensorDimAllRequired(const tpc_lib_api::TensorAccessPattern& accessPattern,
        unsigned accessPatternDimIndex, uint64_t tensorDimSize)
{
    // A dimension is "all required" if:
    // Same size node tiles map to different size tensor tiles (except maybe edges)
    // => In TPC access pattern: end_a != start_a
    // Different node tiles map to one tensor tile
    // => In TPC access pattern: if 'a' is less then 1 (fractional steps may be rounded to the same tensor
    //    tile for several different node tiles).
    // A single node resolution element is mapped to the entire tensor
    // => In TPC access pattern: the range from start_b to end_b is larger than the dimension size

    if (accessPattern.allRequired)
    {
        return true;
    }

    const auto& transform = accessPattern.mapping[accessPatternDimIndex];
    if (transform.a < 1 || (transform.end_b - transform.start_b >= tensorDimSize))
    {
        return true;
    }
    return false;
}

static bool isAnyTensorDimAllRequired(const tpc_lib_api::TensorAccessPattern accessPatterns[],
                                      unsigned                               firstAccessPatternIndex,
                                      const TensorPtr&                       tensor,
                                      DimSet                                 dimList)
{
    return dimList.any([&](unsigned dim) {
            unsigned accessPatternOffset = dim / tpc_lib_api::MAX_INDEX_SPACE_DIM_SIZE;
            unsigned accessPatternDimIndex = dim % tpc_lib_api::MAX_INDEX_SPACE_DIM_SIZE;
            auto size = tensor->getSizeInElements(dim);
            return isTensorDimAllRequired(accessPatterns[firstAccessPatternIndex+accessPatternOffset], accessPatternDimIndex, size);
            });
}
