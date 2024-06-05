#include "optimize_tpc_kernels.h"

#include "code_generation/tensor_size_validator.h"
#include "data_type_utils.h"
#include "defs.h"
#include "graph_editor.h"
#include "graph_traits.h"
#include "habana_graph.h"
#include "habana_nodes.h"
#include "physical_memory_ops_nodes.h"
#include "layout.h"
#include "log_manager.h"
#include "node_factory.h"
#include "node.h"
#include "passes.h"
#include "perf_lib_layer_params.h"
#include "tpc_kernel_loader.h"
#include "transpose_node.h"
#include "transpose_utils.h"
#include "types_exception.h"
#include "types.h"

#include <algorithm>
#include <iterator>
#include <optional>
#include <string_view>

// Tile Node currently supports up to 4D tiling.
// We calculate the number of dimensions using the according kernel params member.
#define REPEATS_DIM       ( (int)(sizeof(ns_TileKernel::Params::repeat) / sizeof(int)) )

NodePtr
SuggestedManipulationHandlerBase::getReshapeNode(const TensorPtr& in, const TensorPtr& out, std::string_view name)
{
    // create a special node that knows that dynamic dimensions do not change,
    // therefore a shape tensor is not needed
    std::string_view guid =
        in->isShapeTensor() ? NodeFactory::staticReshapeShapeNodeTypeName : NodeFactory::staticReshapeNodeTypeName;
    return NodeFactory::createInternalNode({in}, {out}, nullptr, guid, name);
}

NodePtr SuggestedManipulationHandlerBase::getTransposeNode(const TensorPtr& in,
                                                           const TensorPtr& out,
                                                           std::string_view name,
                                                           const uint32_t*  permutation)
{
    synTransposeParams transposeParams;
    transposeParams.tensorDim = in->getDim();
    for (int i = 0; i < MAX_DIMENSIONS_NUM; ++i)
    {
        transposeParams.permutation[i] = static_cast<TransposePermutationDim>(permutation[i]);
    }
    std::string_view guid =
        in->isShapeTensor() ? NodeFactory::transposedShapeNodeTypeName : NodeFactory::transposeNodeTypeName;
    return NodeFactory::createInternalNode({in}, {out}, &transposeParams, guid, name);
}

std::string_view SuggestedManipulationHandlerBase::getTileGuid(synDataType type)
{
    switch (type)
    {
        case syn_type_fixed:
            return "tile_fwd_i8";
        case syn_type_uint8:
            return "tile_fwd_u8";
        case syn_type_bf16:
            return "tile_fwd_bf16";
        case syn_type_fp16:
            return "tile_fwd_f16";
        case syn_type_single:
            return "tile_fwd_f32";
        case syn_type_int16:
            return "tile_fwd_i16";
        case syn_type_uint16:
            return "tile_fwd_u16";
        case syn_type_int32:
            return "tile_fwd_i32";
        case syn_type_uint32:
            return "tile_fwd_u32";
        case syn_type_int64:
            return "tile_fwd_i64";
        case syn_type_uint64:
            return "tile_fwd_u64";
        case syn_type_fp8_143:
            return "tile_fwd_hf8";
        case syn_type_fp8_152:
            return "tile_fwd_f8";
        case syn_type_hb_float:
            return "tile_fwd_f32";
        case syn_type_na:
        default:
            HB_ASSERT(false, "unsupported tile data type");
            return "";
    }
}

TPCNodePtr SuggestedManipulationHandlerBase::getTileNode(const TensorPtr& in,
                                                         const TensorPtr& out,
                                                         const uint64_t*  newShape,
                                                         std::string_view name)
{
    ns_TileKernel::Params params { .repeat={1, 1, 1, 1 } }; // do not repeat Undefined dims for tensors less than 4D
    const NSizeArray&     inputSizes    = in->getAllNSizesInElements();
    const NSizeArray&     outputSizes   = out->getAllNSizesInElements();
    int numTensorDims = in->getDim();
    int loopDimSize = std::min(numTensorDims, REPEATS_DIM);

    // Validation of shapes was done on getTiledTensorShape.

    for (int idx = 0; idx < loopDimSize; ++idx)
    {
        HB_ASSERT(newShape[idx] > 0, "Negative dim size on already Validated suggestion: dim {} = {}",
                  idx, newShape[idx]);
        HB_ASSERT(newShape[idx] % inputSizes[idx] == 0, "Non-whole math on already validated suggestion: {} % {} != 0",
                  newShape[idx], inputSizes[idx]);

        // Validate that output was manipulated correctly
        if (newShape[idx] != outputSizes[idx])
        {
            LOG_WARN(GC,
                     "mismatch on dim {} of tile node output. tensor shape({}) != suggestion shape({}). "
                     "Cannot Create tile node.",
                     idx,
                     outputSizes[idx],
                     newShape[idx]);
            return nullptr;
        }

        // Fill up the kernel's params
        params.repeat[idx] = newShape[idx] / inputSizes[idx];
    }

    // Create the new TPC node with the above params
    std::string_view guid    = getTileGuid(in->getElementType());
    NodePtr          node    = NodeFactory::createInternalNode({in}, {out}, &params, guid, name, "tile_fwd");
    TPCNodePtr       tpcNode = std::static_pointer_cast<TPCNode>(node);
    LOG_DEBUG(GC, "Created Tile Node with input shapes: {}, Output shapes: {}", in->getDimSizesStr(),out->getDimSizesStr());
    return tpcNode;
}

uint32_t SuggestedManipulationHandlerBase::getNewShapeDim(const TensorPtr& tensor, const TSize* newShape)
{
    // Get the new shape number of non one dimensions
    uint32_t dimNum = tensor->getDim();
    for (uint32_t idx = SYN_MAX_TENSOR_DIM; idx > 0; --idx)
    {
        if (newShape[idx - 1] > 1)
        {
            return std::max(idx, dimNum);
        }
    }
    return dimNum;
}

/*
    usually dynamic dimensions cannot be reshaped.
    the only scenario this is ok is if the only dynamic dimension is the tensor outer dimension.
    this function detects that case, and return the minSizes for the suggestion.
*/
bool SuggestedManipulationHandlerBase::isOuterDynamicDimReshaped(const TensorPtr& t,
                                                                 unsigned         firstDynamicDim,
                                                                 NSizeArray&      suggestion /* INOUT */,
                                                                 uint32_t         opDims)
{
    HB_ASSERT(t->isDynamicDim(firstDynamicDim), "expected firstDynamicDim to be dynamic");
    unsigned lastDim = t->getDim() - 1;

    // outer dim is no the first dynamic dim, so it can't be reshaped according to DSD spec.
    if (firstDynamicDim != lastDim) return false;

    TSize maxSize = t->getSizeInElements(lastDim);
    // if last dynamic dimension is 1 there is an ambiguity concerning the reshaping of this dimension.
    // in this case, don't assume that it was reshaped.
    if (maxSize == 1) return false;

    // find the new last dimension in the suggestion
    unsigned newLastDim = opDims - 1;
    for (; newLastDim >= 0; newLastDim--)
    {
        if (suggestion[newLastDim] > 1) break;
    }

    if (suggestion[newLastDim] == maxSize)
    {
        return false;
    }
    else
    {
        LOG_DEBUG(GC, "outer dynamic dimension of {} is being reshaped due to perf lib suggestion", t->getName());
        HB_ASSERT(
            suggestion[newLastDim] % maxSize == 0,
            "suggested manipulation does not contain last dynamic dim! minSizes: {}. maxSizes: {}. suggestion: {}",
            toString(t->getAllMinimalSizesInElements(), ','),
            toString(t->getAllSizesInElements(), ','),
            toString(suggestion, ','));

        TSize minSize = t->getMinimalSizeInElements(lastDim);
        suggestion[newLastDim] /= maxSize;
        suggestion[newLastDim] *= minSize;
        return true;
    }
}

// Check if only the first dynamic dimension got flattened and others did not change or move
NSizeArray SuggestedManipulationHandlerBase::getReshapeMinTensorShapeWithFDD(const TensorPtr& tensor,
                                                                             const TSize*     newMaxShape,
                                                                             unsigned         newFirstDynamicDim,
                                                                             unsigned         origFirstDynamicDim)
{
    NSizeArray minSizes;
    memcpy(minSizes.data(), newMaxShape, tpc_lib_api::MAX_TENSOR_DIM * sizeof(minSizes.data()[0]));
    origFirstDynamicDim++;  // exclude FDD from sequence lookup
    // verify that other dynamic dims did not move
    SizeArray maxSizes    = tensor->getAllSizesInElements();
    auto      seqLocation = std::search(minSizes.begin() + origFirstDynamicDim,
                                   minSizes.end(),
                                   maxSizes.begin() + origFirstDynamicDim,
                                   maxSizes.begin() + tensor->getDim());
    HB_ASSERT(seqLocation != minSizes.data() + tpc_lib_api::MAX_TENSOR_DIM,
              "Did not find dynamic sizes in reshape suggestion excluding first dynamic dim! minSizes: {}. maxSizes: "
              "{}. suggestion: {}",
              toString(tensor->getAllMinimalSizesInElements(), ','),
              toString(maxSizes, ','),
              toString(newMaxShape, newMaxShape + tpc_lib_api::MAX_TENSOR_DIM, ','));

    TSize origMinFDDSize = tensor->getMinimalSizeInElements(origFirstDynamicDim - 1);

    // we need to calculate new min size by multiplying all static sizes from
    // new FDD to original FDD with min original FDD size
    minSizes[newFirstDynamicDim] = std::accumulate(maxSizes.data() + newFirstDynamicDim,
                                                   maxSizes.data() + origFirstDynamicDim - 1,
                                                   origMinFDDSize,
                                                   std::multiplies<TSize>());

    for (auto it = seqLocation; it < seqLocation + tensor->getDim() - origFirstDynamicDim; it++)
    {
        *it = tensor->getMinimalSizeInElements(origFirstDynamicDim + (it - seqLocation));
    }

    return minSizes;
}

/*
    Inferring the min size from a suggested manipulation is not trivial.
    TPC can only change dimensions to the left of the first dynamic dimension,
    without moving dynamic dimensions around. Meaning, the original dynamic dimensions are kept together.
    However, 1s can be inserted on the left and on the right of the dynamic dimensions. For example:
    MAX: [1,256,1,1,2]    =>  [64,4,1,2,1]
    MIN: [1,256,1,0,2]    =>     ?
    The solution is obviously [64,4,0,2,1], but it is not trivially inferred. To infer this will do:
    1. find first dynamic dimension in the original tensor (first dim where min != max)
    1.5. check scenario where dynamic dimension CAN be reshaped (see isOuterDynamicDimReshaped)
    2. find the first dimension after all static elements (in the example, dim 2).
    3. using (1), find the dynamic dimensions (in our case, the pattern [1,2]) in the inferred new max sizes.
    4. starting at the location found at (3), replace the dynamic dimensions of the new max sizes, with the values of
   MIN
*/
NSizeArray SuggestedManipulationHandlerBase::getReshapeMinTensorShape(const TensorPtr& tensor,
                                                                      const TSize*     newMaxShape,
                                                                      uint32_t         opDims)
{
    // [CID: 46226] Declaring minSizes without initializer - intentional, because it will be initialized in the next row.
    NSizeArray minSizes;
    // copy static elements
    memcpy(minSizes.data(), newMaxShape, tpc_lib_api::MAX_TENSOR_DIM * sizeof(minSizes.data()[0]));

    // (1)
    std::optional<unsigned> origFirstDynamicDim = tensor->getFirstDynamicDimIndex();
    if (!origFirstDynamicDim)
    {
        return minSizes;  // If returned false it means there is no dynamic dim - handle the static case.
    }

    // (1.5)
    if (isOuterDynamicDimReshaped(tensor, *origFirstDynamicDim, minSizes /* INOUT */, opDims))
    {
        return minSizes;
    }

    // (2)
    uint64_t numStaticElements = 1;  // calculate number of static reshaped elements.
    for (unsigned dim = 0; dim < *origFirstDynamicDim; dim++)
    {
        numStaticElements *= tensor->getSizeInElements(dim);
    }
    unsigned newFirstDynamicDim = 0;  // the first output dimension after all static elements
    for (uint64_t elems = newMaxShape[0]; elems <= numStaticElements; elems *= newMaxShape[newFirstDynamicDim])
    {
        if (newFirstDynamicDim >= tpc_lib_api::MAX_TENSOR_DIM) break;
        newFirstDynamicDim++;
        if (elems == numStaticElements) break;
    }

    // (3)
    const NSizeArray& maxSizes    = tensor->getAllNSizesInElements();
    auto      seqLocation = std::search(minSizes.begin() + newFirstDynamicDim,
                                   minSizes.end(),
                                   maxSizes.begin() + *origFirstDynamicDim,
                                   maxSizes.begin() + tensor->getDim());
    if (seqLocation == minSizes.data() + tpc_lib_api::MAX_TENSOR_DIM)
    {
        LOG_WARN(GC,
                 "Did not find dynamic sizes in reshape suggestion! minSizes: {}. maxSizes: {}. suggestion: {}",
                 toString(tensor->getAllMinimalSizesInElements(), ','),
                 toString(maxSizes, ','),
                 toString(newMaxShape, newMaxShape + tpc_lib_api::MAX_TENSOR_DIM, ','));
        // Try to find suggestion including first dynamic dim
        minSizes = getReshapeMinTensorShapeWithFDD(tensor, newMaxShape, newFirstDynamicDim, *origFirstDynamicDim);
    }
    else
    {
    // (4)
    for (auto it = seqLocation; it < seqLocation + tensor->getDim() - *origFirstDynamicDim; it++)
    {
        *it = tensor->getMinimalSizeInElements(*origFirstDynamicDim + (it - seqLocation));
    }
    }

    // validation
    HB_ASSERT(multiplyElements(minSizes.begin(), minSizes.begin() + opDims) == tensor->getMinimalElements(),
              "got illegal reshape suggestion! tensor: {}, suggestion {}",
              tensor->getName(),
              getDimStr(newMaxShape, tensor->getDim()));

    return minSizes;
}

bool SuggestedManipulationHandlerBase::getTiledTensorShape(const TensorPtr& tensor,
                                                           const uint64_t*  newShape,
                                                           NSizeArray&      newTiledShape)
{
    // for now, do not support dynamic shapes for tile suggestion.
    if (tensor->getFirstDynamicDimIndex())
    {
        LOG_WARN(GC, "failed to supplement Tile Suggestion for tensor {} since it has a dynamic shape",
                     tensor->getName());
        return false;
    }

    // Validate Suggestion can be fulfilled using integer math.
    const NSizeArray& inputSizes              = tensor->getAllNSizesInElements();
    int       numTensorDims           = tensor->getDim();
    int       loopDimSize             = std::min(numTensorDims, REPEATS_DIM);
    bool      allRepeatsAreEqualToOne = true;

    for (int idx = 0; idx < loopDimSize; ++idx)
    {
        if (newShape[idx] <= 0)
        {
            LOG_WARN(GC, "newShape must be > 0 on all dims. got {} on dim {}.",
                         newShape[idx], idx);
            return false; // can't proceed with optimization.
        }
        int curRepeat = newShape[idx] / inputSizes[idx];
        if ( newShape[idx] % inputSizes[idx] != 0 )
        {
            LOG_WARN(GC, "Failed to create Tile Node: dimension {} mismatch on tile suggestion: {} X {} != {}.",
                         idx, curRepeat, inputSizes[idx], newShape[idx]);
            return false; // can't proceed with optimization.
        }
        allRepeatsAreEqualToOne &= (curRepeat == 1);
    }

    // if all repeats are 1, tile does nothing - abort.
    if (allRepeatsAreEqualToOne)
    {
        LOG_WARN(GC, "all repeats are 1: no point to create Tile Node. aborting manipulation.");
        return false;  // can't proceed with optimization.
    }

    castNcopy(newTiledShape.data(), newShape, tpc_lib_api::MAX_TENSOR_DIM);
    return true;
}

TensorPtr SuggestedManipulationHandlerBase::createClonedTensor(const TensorPtr& tensor,
                                                               const TSize*     newMaxShape,
                                                               const TSize*     newMinShape,
                                                               std::string_view nameSuffix)
{
    TensorPtr clonedTensor = tensor->clone(false, false, false, TensorNameClonePolicy::EMPTY_NAME);
    setManipulatedTensorNameName(*clonedTensor, nameSuffix);
    clonedTensor->resetAliasing();
    clonedTensor->maskOutput();
    // tensors are carefully placed in SRAM prior to this pass.
    // we cannot create additional tensors in SRAM
    if (tensor->inSram() && !tensor->isPartOfRMWSection())
    {
        clonedTensor->setTensorInWorkspace();
    }

    uint32_t dimNum = getNewShapeDim(tensor, newMaxShape);

    clonedTensor->reshape(dimNum, newMaxShape, nullptr, newMinShape);
    return clonedTensor;
}

bool SuggestedManipulationHandlerBase::isTensorSparseAfterRunLogicalOps(const TensorPtr& tensor,
                                                                        const NodeList&  nodes,
                                                                        bool             isInput)
{
    for (const auto& node : nodes)
    {
        if (node && node->isLogicalOperation())
        {
            // The tensors are cloned as well (to avoid duplicated aliased tensors which require mem-copies) -> no need
            // to reset the logical op.
            auto clonedNode = std::static_pointer_cast<LogicalOpNode>(node->cloneWithTensors());
            if (!clonedNode->getRunLogicalOperationDone())
            {
                clonedNode->runAndSetLogicalOp();
                const auto tensorIdx =
                    isInput ? node->getInputIndexOfTensor(tensor) : node->getOutputIndexOfTensor(tensor);
                const auto& clonedTensor = isInput ? clonedNode->getInput(tensorIdx) : clonedNode->getOutput(tensorIdx);
                if (!clonedTensor->isDenseLayout())
                {
                    return true;
                }
            }
        }
    }
    return false;
}

bool SuggestedManipulationHandlerBase::isTensorSparse(const TensorPtr& tensor) const
{
    if (!tensor->isDenseLayout())
    {
        return true;
    }

    // If the pass order is such that optimizeTpcKernels happens before handleLogicalOps,
    // we don't know if the tensor is sparse, since handleLogicalOps might change the strides.
    // Instead of running the full handleLogicalOps pass, we handle the producer and the
    // consumers of the tensor only.
    return isTensorSparseAfterRunLogicalOps(tensor, m_graph.getTensorConsumers(tensor), true) ||
           isTensorSparseAfterRunLogicalOps(tensor, {m_graph.getTensorProducer(tensor)}, false);
}

// Reject memcopy to already aliased tensor S.A reused memory to prevent endless memcopy->reshape chains
//[SW-109709]
bool SuggestedManipulationHandlerBase::shouldRejectAliasedMemcopy() const
{
    if (!isMemcpy(m_node))
    {
        return false;
    }
    auto tensors = m_node.getOperands();
    bool isAliased =
        std::any_of(tensors.begin(), tensors.end(), [&](const TensorPtr& t) { return t->isAliasedTensor(); });
    return isAliased;
}

bool SuggestedManipulationHandlerBase::shouldRejectMemcopyOptimization(const TPCNode& n, const HabanaGraph& g)
{
    // Prevent endless loop between handle logical ops and optimize tpc kernels by preventing
    // kernel optimization of a tpc memcpy planted by handle logical ops to solve a
    // logical operation with a strided real in aliasing operand.
    // The following patterns lead to the endless loop:
    //
    // []->(tpc memcpy)->[real in aliasing + strided]
    //
    // [real in aliasing + strided]->(tpc memcpy)->[]

    // In gaudi3 memcpy is performed by tpc which causes the endless loop
    if (g.getDeviceType() != synDeviceGaudi3 || !isMemcpy(n)) return false;
    for (const auto& operands : {n.getInputs(), n.getOutputs()})
    {
        for (const auto& t : operands)
        {
            if (!t->isTrivialStrided() && t->isRealInAliasing())
            {
                return true;
            }
        }
    }
    return false;
}

bool SuggestedManipulationHandlerBase::isHugeTensor(const TensorPtr& tensor) const
{
    TensorSizeValidator validator(m_graph, /* print only on trace */ SPDLOG_LEVEL_TRACE);
    NStrideArray        strides = {0};
    tensor->getNStridesInBytes(strides.data());
    auto engineType = m_graph.getHALReader()->getTransposeEngine();
    return !validator.validateTensor(tensor, tensor->getAllNSizesInElements(), strides, engineType);
}

bool SuggestedManipulationHandlerBase::shouldRejectSparseTensorShapeManipulation(
    const TensorPtr tensor,
    bool            stridedTensorManipulationFeatureEnabled,
    float           stridedTensorManipulationUtilizationThreshold,
    unsigned        tpcVectorSize) const
{
    // reject shape manipulations if the tensor is sparse and at least one of these conditions apply:
    // 1. tensor is in SRAN - we avoid SRAM->SRAM copy
    // 2. GCFG_ENABLE_TPC_STRIDED_TENSOR_SHAPE_MANIPULATION is disabled
    // 3. tpc vector utilization is above the MAX_TPC_VEC_UTIL_FOR_STRIDED_RESHAPE_MANIPULATION threshold
    // 4. If kernel is memcpy and sparse, it will potentially create a new memcpy in Logical ops which will end in a
    // loop.
    bool  featureDisabled         = !stridedTensorManipulationFeatureEnabled;
    auto  fcdInBytes              = tensor->getAllNSizesInElements()[0] * tensor->getElementSizeInBytes();
    float utilThreshold           = stridedTensorManipulationUtilizationThreshold;
    bool  isVecUtilAboveThreshold = (fcdInBytes / tpcVectorSize) > utilThreshold;
    bool  isSparse                = isTensorSparse(tensor);
    bool  isMemCpy                = isMemcpy(m_node);

    return ((featureDisabled || tensor->inSram() || isVecUtilAboveThreshold || isMemCpy) && isSparse);
}

bool SuggestedManipulationHandlerBase::applyManipulationCheckAndProcess(const tpc_lib_api::TensorOperation* suggestion,
                                                                        const TensorVector&                 tensors,
                                                                        TensorVector&                       newTensors,
                                                                        bool& abortManipulation,
                                                                        bool  isInput)
{
    bool manipulationFound = false;
    bool  stridedTensorManipulationFeatureEnabled = GCFG_ENABLE_TPC_STRIDED_TENSOR_SHAPE_MANIPULATION.value();
    float stridedTensorManipulationUtilizationThreshold =
        GCFG_MAX_TPC_VEC_UTIL_FOR_STRIDED_RESHAPE_MANIPULATION.value();
    unsigned tpcVectorSize = m_graph.getHALReader()->getTpcVectorSize();
    for (int inIdx = 0; inIdx < tensors.size(); inIdx++)
    {
        const TensorPtr& curTensor = tensors[inIdx];
        const auto&      curTensorOperation = suggestion[inIdx];
        auto             opType             = curTensorOperation.opType;
        TensorPtr manipulatedTensor = nullptr;
        if ((opType != tpc_lib_api::TENSOR_OP_NONE) && curTensor->isPartOfRMWSection())
        {
            LOG_WARN(GC,
                     "Ignoring the suggested manipulation for node {} because tensor {} is part of RMW section",
                     m_node.getNodeName(),
                     curTensor->getName());
            abortManipulation = true;  // Ignore suggested manipulation for RMW tensors
            return false;
        }
        switch (opType)
        {
            case tpc_lib_api::TENSOR_OP_TRANSPOSE:
            {
                if (shouldRejectSparseTensorShapeManipulation(tensors[inIdx],
                                                              stridedTensorManipulationFeatureEnabled,
                                                              stridedTensorManipulationUtilizationThreshold,
                                                              tpcVectorSize))
                {
                    LOG_DEBUG_AND_PERF(GC,
                                       "{} : Ignoring the suggested op manipulation in op {} because tensor:"
                                       " {} is sparse and cannot be transposed.",
                                       HLLOG_FUNC,
                                       m_node.getNodeName(),
                                       tensors[inIdx]->getName());
                    abortManipulation = true;
                    return false;
                }
                if (isHugeTensor(tensors[inIdx]))
                {
                    LOG_DEBUG_AND_PERF(GC,
                                       "{} : Ignoring the suggested op manipulation in op {} because tensor:"
                                       " {} is huge and cannot be transposed.",
                                       HLLOG_FUNC,
                                       m_node.getNodeName(),
                                       tensors[inIdx]->getName());
                    abortManipulation = true;
                    return false;
                }
                NSizeArray        newTransposedMaxShape = {};
                NSizeArray        newTransposedMinShape = {};
                const NSizeArray& originalMaxSizes      = curTensor->getAllNSizesInElements();
                const NSizeArray& originalMinSizes      = curTensor->getNMinimalSizesInElements();
                const auto&      dims             = curTensor->getDim();

                HB_ASSERT(dims <= ARRAY_SIZE(curTensorOperation.permutation),
                          "Permutation not covering all dims in node {} and tensor {}",
                          m_node.getNodeName(),
                          curTensor->getName());

                for (int i = 0; i < dims; ++i)
                {
                    newTransposedMaxShape[i] = originalMaxSizes[curTensorOperation.permutation[i]];
                    newTransposedMinShape[i] = originalMinSizes[curTensorOperation.permutation[i]];
                }

                manipulatedTensor = createClonedTensor(curTensor,
                                                       newTransposedMaxShape.data(),
                                                       newTransposedMinShape.data(),
                                                       "_transposed");

                if (isInput)
                {
                    // we update inputPermutations, which is part of the node annotation, and is used during
                    // instantiation of the tpc node, in order to allow the tpc kernel to be aware of the change in
                    // permutation
                    DimVector          suggestedPermutation;
                    suggestedPermutation.reserve(dims);
                    for (int i = 0; i < dims; ++i)
                    {
                        suggestedPermutation.push_back(curTensorOperation.permutation[i]);
                    }

                    if (m_newInputPermutations[inIdx].isIdentity())
                    {
                        m_newInputPermutations[inIdx] = suggestedPermutation;
                    }
                    else
                    {
                        m_newInputPermutations[inIdx].permute({suggestedPermutation});
                    }
                }
                break;
            }
            case tpc_lib_api::TENSOR_OP_NONE:
                newTensors.push_back(tensors[inIdx]);
                break;
            case tpc_lib_api::TENSOR_OP_RESHAPE:
            {
                if (shouldRejectSparseTensorShapeManipulation(tensors[inIdx],
                                                              stridedTensorManipulationFeatureEnabled,
                                                              stridedTensorManipulationUtilizationThreshold,
                                                              tpcVectorSize))
                {
                    LOG_WARN(GC,
                             "{} : Ignoring the suggested op manipulation in op {} because tensor:"
                             " {} is sparse and cannot be reshaped.",
                             HLLOG_FUNC,
                             m_node.getNodeName(),
                             tensors[inIdx]->getName());
                    abortManipulation = true;
                    return false;
                }
                if (shouldRejectAliasedMemcopy())
                {
                    LOG_WARN(GC,
                             "{} : Ignoring the suggested op manipulation in op {}"
                             "because tensor is already aliased in memcopy optimization",
                             HLLOG_FUNC,
                             m_node.getNodeName());
                    abortManipulation = true;
                    return false;
                }
                if (m_skipDynamicNodeHandling)
                {
                    manipulatedTensor =
                        createClonedTensor(curTensor, curTensorOperation.maxNewShape, nullptr, "_reshaped");
                }
                else
                {
                    NSizeArray newTransposedMinShape =
                        getReshapeMinTensorShape(curTensor, curTensorOperation.maxNewShape, curTensorOperation.dims);
                    manipulatedTensor = createClonedTensor(curTensor,
                                                           curTensorOperation.maxNewShape,
                                                           newTransposedMinShape.data(),
                                                           "_reshaped");
                }
                break;
            }
            case tpc_lib_api::TENSOR_OP_TILE:
            {
                NSizeArray newTiledShape;
                // Size of Tile result is new input size of original node.
                bool valid = getTiledTensorShape(curTensor, curTensorOperation.maxNewShape, newTiledShape);
                if (!valid)
                {  // invalid tile shape OR dynamic shapes (not supported) - ignore suggestion.
                    abortManipulation = true;
                    break;
                }
                manipulatedTensor =
                    createClonedTensor(curTensor, curTensorOperation.maxNewShape, newTiledShape.data(), "_tiled");
                break;
            }
            default:
            {
                LOG_WARN(GC, "{}: Unknown tensor manipulation: {} for {}", HLLOG_FUNC, opType, m_node.getGUID());
                abortManipulation = true;  // mark suggestion as erroneous.
                return false;
            }
        }
        if (manipulatedTensor != nullptr)
        {
            manipulationFound = true;
            // isReductionEnabled flag is set to true only for the tensors that are the direct producers of the
            // reduction node (all but the first tensor). When creating the final descriptors we'll use the
            // tensor's isReductionEnabled() method that go down the alias chain to find the correct value
            manipulatedTensor->getTensorAnnotation().tensorReductionInfo.isReductionEnabled = false;
            LOG_DEBUG(GC,
                      "Tensor Manipulation found for suggestion {}, manipulated tensor sizes: {}",
                      inIdx,
                      manipulatedTensor->getDimSizesStr());
            newTensors.push_back(std::move(manipulatedTensor));
        }
    }
    return manipulationFound;
}

// The following two functions need dims parameter because the input
// permutation array is not necessarily filled to the end (i.e. if
// corresponding tensor is of rank 2, the array will be [1,0,0,0,0].
// This is not a valid permutation per se. This causes failures in
// debug mode if TransposePermutationArray is created with size
// of MAX_DIMENSIONS_NUM.
void SuggestedManipulationHandlerBase::calcInversedTransposePermutation(const uint32_t* permutation,
                                                                        unsigned*       inversedPermutation,
                                                                        unsigned        dims)
{
    for (size_t i = 0; i < dims; ++i)
    {
        inversedPermutation[permutation[i]] = i;
    }
}

bool SuggestedManipulationHandlerBase::addNodesNeededForSelectedManipulation(const TensorVector* tensors,
                                                                             TensorVector*       newTensors,
                                                                             bool                isInput)
{
    const TensorVector*    inputTensors  = isInput ? tensors : newTensors;
    const TensorVector*    outputTensors = isInput ? newTensors : tensors;
    for (int inIdx = 0; inIdx < tensors->size(); inIdx++)
    {
        const auto& op = isInput ? m_suggestion.inputTensors[inIdx] : m_suggestion.outputTensors[inIdx];
        switch (op.opType)
        {
            case tpc_lib_api::TENSOR_OP_TRANSPOSE:
            {
                // output transpose node permutation has to be inversed
                unsigned outputTransposePermutation[tpc_lib_api::MAX_TENSOR_DIM];
                if (!isInput)
                {
                    calcInversedTransposePermutation(op.permutation,
                                                     outputTransposePermutation,
                                                     (*tensors)[inIdx]->getDim());
                }

                // create transpose node
                NodePtr transposeNode =
                    getTransposeNode((*inputTensors)[inIdx],
                                     (*outputTensors)[inIdx],
                                     getSuggestedTensorManipulationNodeName(op.opType, isInput, inIdx),
                                     isInput ? op.permutation : outputTransposePermutation);
                m_nodesToAdd.push_back(std::move(transposeNode));
            }
            break;
            case tpc_lib_api::TENSOR_OP_NONE:
                break;
            case tpc_lib_api::TENSOR_OP_RESHAPE:
            {
                NodePtr reshapeNode = getReshapeNode((*inputTensors)[inIdx],
                                                     (*outputTensors)[inIdx],
                                                     getSuggestedTensorManipulationNodeName(op.opType, isInput, inIdx));

                reshapeNode->getNodeAnnotation().sliceIndex = m_node.getNodeAnnotation().sliceIndex;
                reshapeNode->getNodeAnnotation().rangeIndex = m_node.getNodeAnnotation().rangeIndex;
                reshapeNode->getNodeAnnotation().bundleInfo = m_node.getNodeAnnotation().bundleInfo;

                LOG_DEBUG(GC, "Created Reshape Node with input shapes: {}, Output shapes: {}",
                              reshapeNode->getInput(0)->getDimSizesStr(),reshapeNode->getOutput(0)->getDimSizesStr());
                m_nodesToAdd.push_back(std::move(reshapeNode));
                break;
            }
            case tpc_lib_api::TENSOR_OP_TILE:
            {
                if ((*inputTensors)[inIdx]->isShapeTensor())
                {
                    LOG_DEBUG(
                        GC,
                        "node: {}, cannot request tile shape manipulation on shape tensors. aborting manipulation.",
                        m_node.getNodeName());
                    return false;
                }
                TPCNodePtr tileNode = getTileNode((*inputTensors)[inIdx],
                                                  (*outputTensors)[inIdx],
                                                  op.maxNewShape,
                                                  getSuggestedTensorManipulationNodeName(op.opType, isInput, inIdx));
                if (tileNode == nullptr)
                {
                    LOG_DEBUG(GC,
                              "Unable to create tile node - revert tensor manipulation to original for node {}",
                              m_node.getNodeName());
                    return false;
                }

                tileNode->getNodeAnnotation().sliceIndex = m_node.getNodeAnnotation().sliceIndex;
                tileNode->getNodeAnnotation().rangeIndex = m_node.getNodeAnnotation().rangeIndex;
                tileNode->getNodeAnnotation().bundleInfo = m_node.getNodeAnnotation().bundleInfo;
                // Tile node created successfully, ready to add
                m_nodesToAdd.push_back(std::move(tileNode));
                break;
            }
            default:
            {
                HB_ASSERT(false,
                          "{}: cannot create nodes for tensor maniuplation {} for {}",
                          __FUNCTION__,
                          op.opType,
                          m_node.getGUID());
            }
        }
    }
    return true;
}

TensorVector SuggestedManipulationHandlerBase::filterAuxAndNullTensors(const TensorVector& src)
{
    TensorVector dst;
    dst.reserve(src.size());
    std::copy_if(src.begin(), src.end(), std::back_inserter(dst), [](const TensorPtr& t) {
        return t && !t->isAuxTensor();
    });
    return dst;
}

bool SuggestedManipulationHandlerBase::isDifferentShape(const TensorPtr&                    tensor,
                                                        const tpc_lib_api::TensorOperation& newOp)
{
    const NSizeArray& tensorShape = tensor->getAllNSizesInElements();
    unsigned          tensorDims  = tensor->getDim();
    return tensorDims != newOp.dims ||
           !std::equal(tensorShape.data(), tensorShape.data() + tensorDims, newOp.maxNewShape);
}

bool SuggestedManipulationHandlerBase::isIdentityPermutation(const TensorPtr& tensor, const uint32_t* newPermutation)
{
    for (int i = 0; i < tensor->getDim(); ++i)
    {
        if (newPermutation[i] != i) return false;
    }
    return true;
}

bool SuggestedManipulationHandlerBase::isEmptyTensorManipulationSuggestion() const
{
    using OperandsAndSuggestions = std::pair<const TensorVector&, const tpc_lib_api::TensorOperation*>;
    for (const auto& [operands, suggestions] :
         {OperandsAndSuggestions(m_node.getInputs(), m_suggestion.inputTensors),
          OperandsAndSuggestions(m_node.getOutputs(), m_suggestion.outputTensors)})
    {
        for (size_t i = 0; i < operands.size(); i++)
        {
            auto opType = suggestions[i].opType;
            if (operands[i] != nullptr)
            {
                switch (opType)
                {
                    case tpc_lib_api::TENSOR_OP_TRANSPOSE:
                        if (!isIdentityPermutation(operands[i], suggestions[i].permutation)) return false;
                    case tpc_lib_api::TENSOR_OP_NONE:
                        break;
                    default:
                        if (isDifferentShape(operands[i], suggestions[i])) return false;
                }
            }
        }
    }
    return true;
}

bool SuggestedManipulationHandlerBase::isSuggestedTensorManipulationAvailable()
{
    m_operandsSuggestions.resize(m_node.getNumInputs(), m_node.getNumOutputs());
    m_suggestion.inputTensors  = m_operandsSuggestions.getInputs();
    m_suggestion.outputTensors = m_operandsSuggestions.getOutputs();

    if (m_node.getSuggestedTensorManipulation(&m_suggestion) != tpc_lib_api::GLUE_SUCCESS)
    {
        LOG_DEBUG(GC, "{}: getSuggestedTensorManipulation for {} may not be supported", HLLOG_FUNC, m_node.getGUID());
        return false;
    }
    // only apply suggestion if it is not empty
    if (isEmptyTensorManipulationSuggestion())
    {
        // Prevent following glue code queries.
        m_node.setSuggestedOptimizationDone(true);
        return false;
    }
    return true;
}

bool SuggestedManipulationHandlerBase::shouldSkipSuggestedTensorManipulation(TPCNode& node, const HabanaGraph& graph)
{
    if (node.isSuggestedOptimizationDone()) return true;  // do not try to optimize if already optimized.
    // Todo [SW-101233] The condition below can be relaxed in some circumstances
    if (node.isMemset() && !node.getOutput(0)->isDenseLayout()) return true;
    if (node.isMemset() && node.getOutput(0)->isDynamicShape() && node.getNumInputs() == 0) return true;
    if (node.isRestrictedShapeRandomNode()) return true;
    // For dynamic memcpy in slice and stride ops
    if (isMemcpy(node) && node.getNumInputs() > 1) return true;
    if (shouldRejectMemcopyOptimization(node, graph))
    {
        LOG_DEBUG(GC,
                  "Reject kernel optimization for {}[{}], in_size: {}, in_stride: {}, out_size: {}, out_stride: {}",
                  node.getNodeName(),
                  node.getNodeTypeStr(),
                  toString(node.getInput(0)->getAllSizesInElements(), ','),
                  toString(node.getInput(0)->getAllStridesInBytes(), ','),
                  toString(node.getOutput(0)->getAllSizesInElements(), ','),
                  toString(node.getOutput(0)->getAllStridesInBytes(), ','));
        return true;
    }
    return false;
}

// This function keeps the isMemset() property of the return type
TPCNodePtr GraphModeSuggestedManipulationHandler::createConcreteTpcNode(const TensorVector& newInputs,
                                                                        const TensorVector& newOutputs) const
{
    std::string new_node_name = fmt::format("{}_optimized", m_node.getNodeName());
    TPCNodePtr  ret;
    if (m_node.isMemset())
    {
        ret = std::static_pointer_cast<TPCNode>(NodeFactory::createInternalNode(newInputs,
                                                                                newOutputs,
                                                                                nullptr,
                                                                                NodeFactory::tpcMemsetNodeTypeName,
                                                                                new_node_name,
                                                                                NodeFactory::tpcMemsetNodeTypeName));
    }
    else if (dynamic_cast<const SerializeNode<TPCMemcpyNode>*>(&m_node) != nullptr)
    {
        // cannot use createGenericTPCNode here because we need to create a subclass
        // of TPCNode and createGenericTPCNode cannot do that
        // not using node.getGUID() because the guid is not the same as the
        // node type name used for node creation
        ret = std::dynamic_pointer_cast<TPCNode>(NodeFactory::createNode(newInputs,
                                                                         newOutputs,
                                                                         m_node.getParams(),
                                                                         m_node.getParamsSize(),
                                                                         NodeFactory::serializeTPCNodeTypeName,
                                                                         new_node_name));
    }
    else if (dynamic_cast<const DeserializeNode<TPCMemcpyNode>*>(&m_node) != nullptr)
    {
        // cannot use createGenericTPCNode here because we need to create a subclass
        // of TPCNode and createGenericTPCNode cannot do that
        // not using node.getGUID() because the guid is not the same as the
        // node type name used for node creation
        ret = std::dynamic_pointer_cast<TPCNode>(NodeFactory::createNode(newInputs,
                                                                         newOutputs,
                                                                         m_node.getParams(),
                                                                         m_node.getParamsSize(),
                                                                         NodeFactory::deserializeTPCNodeTypeName,
                                                                         new_node_name));
    }
    else
    {
        ret = std::static_pointer_cast<TPCNode>(NodeFactory::createGenericTPCNode(newInputs,
                                                                                  newOutputs,
                                                                                  m_node.getParams(),
                                                                                  m_node.getParamsSize(),
                                                                                  m_node.getGUID(),
                                                                                  new_node_name));
    }
    HB_ASSERT_PTR(ret);
    return ret;
}

void GraphModeSuggestedManipulationHandler::addShapeTensorSuggestions()
{
    unsigned numShapeTensor = 0;
    for (unsigned i = 0; i < m_node.getNumInputs(); i++)
    {
        const TensorPtr& in = m_node.getInput(i);
        if (in->getTensorType() == synTensorType::OUTPUT_DESCRIBING_SHAPE_TENSOR)
        {
            numShapeTensor++;
            LOG_DEBUG(GC, "adding shape manipulation suggestion for input shape tensor {}", in->getName());
            m_suggestion.inputTensors[i] = m_suggestion.outputTensors[0];
        }
    }
}

bool GraphModeSuggestedManipulationHandler::applySuggestedTensorManipulation()
{
    LOG_DEBUG(GC, "Applying non-empty tensor manipulation suggestion for node {}", m_node.getNodeName());
    // Preparing for 2nd init call, removing leftovers from previous init run
    TensorVector inputs  = filterAuxAndNullTensors(m_node.getInputs());
    TensorVector outputs = filterAuxAndNullTensors(m_node.getOutputs());
    // These will hold modified Tensors
    TensorVector newInputs;
    newInputs.reserve(inputs.size());
    TensorVector newOutputs;
    newOutputs.reserve(outputs.size());

    m_newInputPermutations = m_node.getNodeAnnotation().inputPermutations;

    bool abortManipulation = false;

    // process inputs
    bool inputsModified = applyManipulationCheckAndProcess(m_suggestion.inputTensors,
                                                           inputs,
                                                           newInputs,
                                                           abortManipulation,
                                                           true);
    // process outputs
    bool outputsModified = applyManipulationCheckAndProcess(m_suggestion.outputTensors,
                                                            outputs,
                                                            newOutputs,
                                                            abortManipulation,
                                                            false);

    // if there is no manipulation to do, or there was an unexpected error while applying the manipulation
    // leave the graph unmodified. there is an assumption here that if outputs are modified, inputs should too
    // for all current suggestion cases.
    if ((!inputsModified && !outputsModified) || abortManipulation) return false;

    LOG_DEBUG(GC, "Trying to reinstantiate {} with suggested tensor manipulation.", m_node.getNodeName());
    TPCNodePtr newTpcNode = createConcreteTpcNode(newInputs, newOutputs);

    // New node is created without the original's annotation, which should not be lost.
    newTpcNode->getNodeAnnotation()                   = m_node.getNodeAnnotation();
    newTpcNode->getNodeAnnotation().inputPermutations = m_newInputPermutations;

    auto ret = newTpcNode->init(deviceTypeToDeviceID(m_graph.getDeviceType()),
                                &m_graph.getGraphAnnotation().cachedAuxiliaryTensors,
                                m_graph.getNextTPCKernelUniqueId());
    if (ret != tpc_lib_api::GLUE_SUCCESS)
    {
        // if Unable to perform optimization - add a warning. no need to fail compilation.
        LOG_WARN(GC,
                 "Failed to create a new TPCNode with suggested tensor manipulation. Old node: {}",
                 m_node.getNodeName());
        return false;  // can't proceed with optimization.
    }

    newTpcNode->setSuggestedOptimizationDone(true);

    // add nodes for inputs and outputs
    if (!addNodesNeededForSelectedManipulation(&inputs, &newInputs, true) ||
        !addNodesNeededForSelectedManipulation(&outputs, &newOutputs, false))

    {
        // if something went wrong when trying to add needed nodes, abort the suggestion altogether.
        return false;
    }

    LOG_DEBUG(GC, "Replacing TPC Node {} with new TPC node {}", m_node.getNodeName(), newTpcNode->getNodeName());

    // if old tpc node took part in an atomic pair, replaces the new tpc node in its place.
    m_graph.getGraphAnnotation().replaceAtomicNode(&m_node, newTpcNode);

    m_nodesToAdd.push_back(std::move(newTpcNode));

    return true;
}

std::string
GraphModeSuggestedManipulationHandler::getSuggestedTensorManipulationNodeName(tpc_lib_api::TensorOperationType opType,
                                                                              unsigned tensorIdx,
                                                                              bool     isInput)
{
    const std::string_view nameSuffix = isInput ? "_in" : "_out";
    switch (opType)
    {
        case tpc_lib_api::TENSOR_OP_TRANSPOSE:
            return fmt::format("{}_transpose_TPC{}{}", m_node.getNodeName(), nameSuffix, tensorIdx);
        case tpc_lib_api::TENSOR_OP_RESHAPE:
            return fmt::format("{}_reshape_TPC{}{}", m_node.getNodeName(), nameSuffix, tensorIdx);
        case tpc_lib_api::TENSOR_OP_TILE:
            return fmt::format("{}_tile_TPC{}{}", m_node.getNodeName(), nameSuffix, tensorIdx);
        default:
            return "";
    }
}

void GraphModeSuggestedManipulationHandler::setManipulatedTensorNameName(Tensor& t, std::string_view nameSuffix)
{
    return t.setName(fmt::format("{}{}", t.getName(), nameSuffix));
}

bool optimizeTpcKernels(HabanaGraph& g)
{
    if (!GCFG_ENABLE_TPC_TENSOR_SHAPE_MANIPULATION.value()) return true;

    //This optimization is not currently compatible with 'Memory Oriented' feature, since it may disable SRAM reuse in tpc kernels
    if (g.getMemoryOrientedCompilationEnabled())  // This optimization is not currently compatible with 'BigImages'
    {
        return true;
    }

    const NodeVector sortedNodes = g.getExeSortedNodes();

    for (const NodePtr& node : sortedNodes)
    {
        if (!g.runsOnTPC(node)) continue;
        // we expect this pass to happen before ROIs are generated
        if (g.nodeHasROIs(node))
        {
            LOG_DEBUG(GC,
                      "{}: ROIs for node {} were already generated. not trying to optimize.",
                      HLLOG_FUNC,
                      node->getNodeName());
            continue;
        }
        auto tpcNode = static_cast<TPCNode*>(node.get());
        if (GraphModeSuggestedManipulationHandler::shouldSkipSuggestedTensorManipulation(*tpcNode, g)) continue;
        GraphModeSuggestedManipulationHandler handler(g, *tpcNode);
        if (handler.isSuggestedTensorManipulationAvailable())
        {
            handler.addShapeTensorSuggestions();  // [SW-95667] TODO - remove when issue is resolved
            if (handler.applySuggestedTensorManipulation())
            {
                auto status = GraphEditor::replaceNodes(g, {node}, handler.extract());
                HB_ASSERT(status == REPLACE_NODE_SUCCESS,
                          "{}: failed to apply suggestion for tpc node {}",
                          __FUNCTION__,
                          node->getNodeName());
                LOG_INFO(
                    GC,
                    "Optimized: replaced {} TPC node and added optimized nodes based on TPC node suggestion, GUID {}",
                    node->getNodeName(),
                    node->getGUID());
            }
        }
    }
    return true;
}
