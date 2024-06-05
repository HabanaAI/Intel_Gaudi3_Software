#include "mme_desc_gen_utils.h"
#include "habana_graph.h"
#include "mme_node.h"
#include "mme_services.h"
#include "tensor.h"
#include "synapse_common_types.h"
#include "compilation_hal_reader.h"

void getTensorRolesCommon(const MmeNode&        node,
                          MmeCommon::EMmeOpType opType,
                          TensorPtr&            xTensor,
                          TensorPtr&            wTensor,
                          TensorPtr&            yTensor,
                          TensorPtr&            oTensor)
{
    const Node::eNodeType nodeType = node.getNodeType();
    oTensor                        = node.getOutput(TENSOR_SECONDARY_OFM);

    switch (opType)
    {
        case MmeCommon::e_mme_fwd:
        case MmeCommon::e_mme_ab:
        case MmeCommon::e_mme_atb:
        case MmeCommon::e_mme_abt:
        case MmeCommon::e_mme_atbt:
        case MmeCommon::e_mme_reductionAdd:
            switch (nodeType)
            {
                case Node::TYPE_BATCH_GEMM_DEDW:
                case Node::TYPE_GEMM_DEDW:
                    wTensor = node.getInput(TENSOR_DEDY);
                    xTensor = node.getInput(TENSOR_X_BWD);
                    yTensor = node.getOutput(TENSOR_DEDW);
                    break;
                case Node::TYPE_BATCH_GEMM_DEDX:
                case Node::TYPE_GEMM_DEDX:
                    xTensor = node.getInput(TENSOR_DEDY);
                    yTensor = node.getOutput(TENSOR_DEDX);
                    wTensor = node.getInput(TENSOR_WEIGHT);
                    break;
                default:
                    xTensor = node.getInput(TENSOR_IFM);
                    wTensor = node.getInput(TENSOR_WEIGHT);
                    yTensor = node.getOutput(TENSOR_OFM);
            }
            break;
        case MmeCommon::e_mme_dedw:
            yTensor = node.getInput(TENSOR_DEDY);
            xTensor = node.getInput(TENSOR_X_BWD);
            wTensor = node.getOutput(TENSOR_DEDW);
            if ((node.getNodeType() != Node::TYPE_DEDW) && (node.getNodeType() != Node::TYPE_GEMM_DEDW))
            {
                // when operation=dedw we can have the following nodeTypes-
                // in Gaudi2\3 - dedw only
                // in Gaudi - dedw, gemm_dedw, gemm+transpose A
                // in the gemm+transpose case we want X to be input[0] and Y to be input[1] - so we need to swap.
                // should be removed once SW-144790 is resolved.
                std::swap(xTensor, yTensor);
            }
            break;
        case MmeCommon::e_mme_dedx:
        case MmeCommon::e_mme_transposed_dedx:
            yTensor = node.getInput(TENSOR_DEDY);
            xTensor = node.getOutput(TENSOR_DEDX);
            wTensor = node.getInput(TENSOR_WEIGHT);
            break;
        case MmeCommon::e_mme_memcpy:
        case MmeCommon::e_mme_trans:
            // can reuse the case for bgemm/fwd x & y values are the same because w value is n/a
            xTensor = node.getInput(TENSOR_IFM);
            wTensor = nullptr;
            yTensor = node.getOutput(TENSOR_OFM);
            break;
        case MmeCommon::e_mme_gemm_transpose:
            xTensor = node.getInput(TENSOR_IFM);
            wTensor = node.getInput(TENSOR_UNIT_MATRIX);
            yTensor = node.getOutput(TENSOR_OFM);
            break;
        default:
            HB_ASSERT(0, "opType is not supported");
            break;
    }
    return;
}

void setTensorViewByOp(const MmeNode&                  node,
                       MmeCommon::MmeLayerParams&      params,
                       const MmeCommon::MmeTensorView& aView,
                       const MmeCommon::MmeTensorView& bView,
                       const MmeCommon::MmeTensorView& outView)
{
    const Node::eNodeType nodeType = node.getNodeType();

    switch (params.opType)
    {
        case MmeCommon::e_mme_fwd:
        case MmeCommon::e_mme_ab:
        case MmeCommon::e_mme_atb:
        case MmeCommon::e_mme_abt:
        case MmeCommon::e_mme_atbt:
        case MmeCommon::e_mme_reductionAdd:
            switch (nodeType)
            {
                case Node::TYPE_BATCH_GEMM_DEDW:
                case Node::TYPE_GEMM_DEDW:
                    params.w = aView;
                    params.x = bView;
                    params.y = outView;
                    break;
                default:
                    params.x = aView;
                    params.w = bView;
                    params.y = outView;
                    break;
            }
            break;
        case MmeCommon::e_mme_dedw:
            params.y = aView;
            params.x = bView;
            params.w = outView;
            if ((node.getNodeType() != Node::TYPE_DEDW) && (node.getNodeType() != Node::TYPE_GEMM_DEDW))
            {
                // when operation=dedw we can have the following nodeTypes-
                // in Gaudi2\3 - dedw only
                // in Gaudi - dedw, gemm_dedw, gemm+transpose A
                // in the gemm+transpose case we want X to be input[0] and Y to be input[1] - so we need to swap.
                // should be removed once SW-144790 is resolved.
                std::swap(params.x, params.y);
            }
            break;
        case MmeCommon::e_mme_dedx:
        case MmeCommon::e_mme_transposed_dedx:
            params.y = aView;
            params.x = outView;
            params.w = bView;
            break;
        case MmeCommon::e_mme_memcpy:
        case MmeCommon::e_mme_trans:
            // can reuse the case for bgemm/fwd x & y values are the same because w value is n/a
            params.x = aView;
            params.w = MmeCommon::MmeTensorView();  // should be default object
            params.y = outView;
            break;
        case MmeCommon::e_mme_gemm_transpose:
            params.x = aView;
            params.w = bView;
            params.y = outView;
            break;
        default:
            HB_ASSERT(0, "opType is not supported");
            break;
    }
    return;
}

MmeCommon::EMmeOpType getOperationTypeCommon(MmeCommon::ChipType chipType, const MmeNode& node)

{
    switch (node.getNodeType())
    {
        case Node::TYPE_DEDW:
            return MmeCommon::e_mme_dedw;
        case Node::TYPE_DEDX:
            return MmeCommon::e_mme_dedx;
        case Node::TYPE_TRANSPOSED_DEDX:
            return MmeCommon::e_mme_transposed_dedx;
        case Node::TYPE_CONVOLUTION:
            return MmeCommon::e_mme_fwd;
        case Node::TYPE_BATCH_GEMM:
        case Node::TYPE_BATCH_GEMM_DEDX:
        case Node::TYPE_BATCH_GEMM_DEDW:
        case Node::TYPE_MASKED_BATCH_GEMM:
        {
            const synGEMMParams& gemmParams = static_cast<const GEMMNode*>(&node)->getGEMMParams();
            if (!gemmParams.transpose_a && !gemmParams.transpose_b)
            {
                return MmeCommon::e_mme_ab;
            }
            else if (gemmParams.transpose_a && !gemmParams.transpose_b)
            {
                return MmeCommon::e_mme_atb;
            }
            else if (!gemmParams.transpose_a && gemmParams.transpose_b)
            {
                return MmeCommon::e_mme_abt;
            }
            else
            {
                return MmeCommon::e_mme_atbt;
            }
            break;
        }
        case Node::TYPE_FC:
        case Node::TYPE_GEMM:
        case Node::TYPE_GEMM_DEDX:
        case Node::TYPE_GEMM_DEDW:
        {
            //TODO: [SW-144790] remove Gaudi specification
            const synGEMMParams& gemmParams = static_cast<const GEMMNode*>(&node)->getGEMMParams();
            if (!gemmParams.transpose_a && !gemmParams.transpose_b)
            {
                return (chipType==MmeCommon::e_mme_Gaudi) ? MmeCommon::e_mme_fwd : MmeCommon::e_mme_ab;
            }
            else if (gemmParams.transpose_a && !gemmParams.transpose_b)
            {
                return (chipType==MmeCommon::e_mme_Gaudi) ? MmeCommon::e_mme_dedw :MmeCommon::e_mme_atb;
            }
            else if (!gemmParams.transpose_a && gemmParams.transpose_b)
            {
                return (chipType==MmeCommon::e_mme_Gaudi) ? MmeCommon::e_mme_dedx :MmeCommon::e_mme_abt;
            }
            else
            {
                return MmeCommon::e_mme_atbt;
            }
            break;
        }
        case Node::TYPE_INTERNAL_TRANSPOSE:
            HB_ASSERT(chipType == MmeCommon::e_mme_Gaudi3, "Unexpected chip type in Internal Transpose operation");
            return node.isTransposeViaGemm() ? MmeCommon::e_mme_gemm_transpose : MmeCommon::e_mme_trans;
        default:
            HB_ASSERT(false, "Unsupported MME type {}", node.getNodeType());
    }
    return MmeCommon::e_mme_fwd;
}

MmeCommon::MmeTensorView
getSemanticTensorView(MmeCommon::ChipType chipType, const Tensor& tensor, const HalReader& halReader, bool isDmaOp)
{
    MmeCommon::MmeTensorView view {};
    synDataType              elementType = tensor.getElementType();

    if (isDmaOp)
    {
        HB_ASSERT(chipType == MmeCommon::e_mme_Gaudi3, "Unsupported MME data type {}", elementType);
        elementType = MmeCommon::MmeServices::getDtypeForTranspose(tensor);
    }
    else
    {
        HB_ASSERT(halReader.isSupportedMmeDataType(elementType), "Unsupported MME data type {}", elementType);
    }

    view.elementType =
        getMmeElementTypeCommon(elementType,
                                chipType == MmeCommon::e_mme_Gaudi2 && GCFG_GAUDI2_FORCE_MME_FP32_IEEE.value());
    TSize arrayTSize[MME_MAX_TENSOR_DIMS];
    tensor.getAllSizesInElementsCondensed(arrayTSize, MME_MAX_TENSOR_DIMS);
    castNcopy(view.sizes.data(), arrayTSize, MME_MAX_TENSOR_DIMS);

    return view;
}

MmeCommon::MmeTensorView
getTensorViewCommon(MmeCommon::ChipType chipType, const Tensor& tensor, const HalReader& halReader, bool isDmaOp)
{
    MmeCommon::MmeTensorView view = getSemanticTensorView(chipType, tensor, halReader, isDmaOp);
    view.strides[0]       = 1;  // FCD strides
    unsigned tensorMaxDim = tensor.getDim();
    TStride  maxStride    = static_cast<TStride>(view.strides[0]) * view.sizes[0];
    int      firstDegeneratedDim  = tensorMaxDim;
    while (firstDegeneratedDim > 1 && view.sizes[firstDegeneratedDim - 1] == 1)
    {
        firstDegeneratedDim--;
    }
    for (unsigned int dim = 1; dim < MME_MAX_TENSOR_DIMS; ++dim)
    {
        if (dim < firstDegeneratedDim)
        {
            view.strides[dim] = tensor.getStrideInElements(dim);
            maxStride         = std::max<TStride>(maxStride, static_cast<TStride>(view.strides[dim]) * view.sizes[dim]);
        }
        else
        {
            view.strides[dim] = maxStride;
        }
    }
    HB_ASSERT(maxStride <= std::numeric_limits<std::remove_reference<decltype(view.strides[0])>::type>::max(),
              "max stride of {} is too large for mme params",
              tensor.getName());

    return view;
}

MmeCommon::EMmeDataType getMmeElementTypeCommon(synDataType elementType, bool fp32ForcedToIEEE)
{
    switch (elementType)
    {
        case syn_type_fp16:
            return MmeCommon::e_type_fp16;
        case syn_type_bf16:
            return MmeCommon::e_type_bf16;
        case syn_type_single:
            return fp32ForcedToIEEE ? MmeCommon::e_type_fp32_ieee : MmeCommon::e_type_fp32;
        case syn_type_tf32:
            return MmeCommon::e_type_tf32;
        case syn_type_hb_float:
            return MmeCommon::e_type_fp32;
        case syn_type_fp8_143:
            return MmeCommon::e_type_fp8_143;
        case syn_type_fp8_152:
            return MmeCommon::e_type_fp8_152;
        default:
            HB_ASSERT(false, "Unsupported MME data type {}", elementType);
    }
    return MmeCommon::e_type_fp16;
}

bool getAlignedAddresses(const MmeNode*        mmeNode,
                         MmeCommon::EMmeOpType opType,
                         bool                  ignoreTensorAliasing)
{
    TensorPtr xTensor, wTensor, yTensor, oTensor;
    getTensorRolesCommon(*mmeNode, opType, xTensor, wTensor, yTensor, oTensor);

    bool alignedAddresses = false;
    switch (opType)
    {
        case MmeCommon::e_mme_ab:
        case MmeCommon::e_mme_atb:
        case MmeCommon::e_mme_abt:
        case MmeCommon::e_mme_atbt:
        case MmeCommon::e_mme_fwd:
        case MmeCommon::e_mme_gemm_transpose:
            alignedAddresses = isTensorAddressCacheLineAligned(xTensor, ignoreTensorAliasing) &&
                               isTensorAddressCacheLineAligned(wTensor, ignoreTensorAliasing);
            break;
        case MmeCommon::e_mme_dedx:
        case MmeCommon::e_mme_transposed_dedx:
            alignedAddresses = isTensorAddressCacheLineAligned(yTensor, ignoreTensorAliasing) &&
                               isTensorAddressCacheLineAligned(wTensor, ignoreTensorAliasing);
            break;
        case MmeCommon::e_mme_dedw:
        case MmeCommon::e_mme_deterministic_dedw:
            alignedAddresses = isTensorAddressCacheLineAligned(xTensor, ignoreTensorAliasing) &&
                               isTensorAddressCacheLineAligned(yTensor, ignoreTensorAliasing);
            break;
        case MmeCommon::e_mme_memcpy:
        case MmeCommon::e_mme_trans:
            alignedAddresses = isTensorAddressCacheLineAligned(xTensor, ignoreTensorAliasing);
            break;
        default:
            HB_ASSERT(false, "Unsupported MME operation type {}", opType);
            break;
    }
    return alignedAddresses;
}

bool isTensorAddressCacheLineAligned(TensorPtr tensor, bool ignoreTensorAliasing)
{
    if (tensor == nullptr) return true;
    if (!GCFG_MME_STRATEGY_ALIGNED_ADDRESSES_ENABLED.value()) return false;

    const unsigned cacheLineAlignment = CompilationHalReader::getHalReader()->getCacheLineSizeInBytes();
    if (tensor->isPersistent() && ((tensor->getMemorySectionOffset() % cacheLineAlignment) != 0))
    {
        return false;
    }

    TensorPtr realTensor = tensor;
    if (!ignoreTensorAliasing)
    {
        // the loop is copied from getRealTensor()
        while (realTensor->isAliasedTensor())
        {
            // In the case of: aliased tensor N -> aliased tensor N-1 -> ... -> real tensor
            // We return true only if each of the tensors offset is cache-line aligned
            // This can be tuned further by accumulating the offset and returning true when the total offset is
            // cache-line aligned.
            if ((realTensor->getAliasedByteOffset() % cacheLineAlignment) != 0) return false;
            realTensor = realTensor->getAliasTensor();
            if (realTensor == nullptr) return true;
        }
    }

    const uint64_t alignment = realTensor->getTensorAnnotation().memory.alignment;
    return (alignment % cacheLineAlignment) == 0;
}

MmeCommon::MmeTensorView getTensorViewFromTile(MmeCommon::ChipType chipType,
                                               const TensorPtr& tensor,
                                               const DcoreTile& tensorTile,
                                               const HalReader& halReader,
                                               bool isDmaOp)
{
    MmeCommon::MmeTensorView view;
    synDataType elementType = tensor->getElementType();

    if (isDmaOp)
    {
        HB_ASSERT(chipType == MmeCommon::e_mme_Gaudi3, "Unsupported MME data type {}", elementType);
        elementType = MmeCommon::MmeServices::getDtypeForTranspose(*tensor);
    }
    else
    {
        HB_ASSERT(halReader.isSupportedMmeDataType(elementType), "Unsupported MME data type {}", elementType);
    }

    view.elementType = getMmeElementTypeCommon(elementType,
                                               chipType == MmeCommon::e_mme_Gaudi2 && GCFG_GAUDI2_FORCE_MME_FP32_IEEE.value());

    castNcopy(view.sizes, tensorTile.geometry, std::min((unsigned)MME_MAX_TENSOR_DIMS, (unsigned)tensorTile.geometry.size()));
    castNcopy(view.dcoreBases, tensorTile.offset, std::min((unsigned)MME_MAX_TENSOR_DIMS, (unsigned)tensorTile.offset.size()));

    view.strides[0] = 1;  // FCD strides
    unsigned tensorMaxDim = tensor->getDim() + 1;
    for (unsigned int dim = 1; dim < MME_MAX_TENSOR_DIMS; ++dim)
    {
        if (dim < tensorMaxDim)
        {
            view.strides[dim] = tensor->getStrideInElements(dim);
        }
        else
        {
            view.strides[dim] = tensor->getStrideInElements(tensorMaxDim - 1);
        }
    }
    return view;
}

bool isAllocPolicySuitableForTensor(CacheDirective allocPolicy, const TensorPtr& tensor)
{
    // 1. If the tensor is unaligned in address and FCD is > 256, allocD is NOT suitable
    // 2. If the tensor is unaligned in strides and FCD is > 256, allocD is NOT suitable
    if (allocPolicy != DcoreAllocate) return true;
    auto cacheLineSize     = CompilationHalReader::getHalReader()->getCacheLineSizeInBytes();
    bool areStridesAligned = tensor->getStrideInBytes(1) % cacheLineSize == 0;
    // if the tensor is dense, it's enough to check stride 1 since the rest will be its multiple
    // if the tensor is not dense, we need to verify each and every stride
    if (!tensor->isDenseLayout())
    {
        for (unsigned d = 2; areStridesAligned && d < tensor->getDim(); ++d)
        {
            areStridesAligned = areStridesAligned && (tensor->getNStridesInBytes()[d] % cacheLineSize == 0);
        }
    }
    bool isAddressAligned  = isTensorAddressCacheLineAligned(tensor, false);
    bool isFcdLargerThanCL = tensor->getSizeInBytes(0) > cacheLineSize;
    return ((!isAddressAligned || !areStridesAligned) && isFcdLargerThanCL) ? false : true;
}

void setTracing(MmeCommon::MmeLayerParams& params)
{
    // We set the context id to 0 as a placeholder - will be patched later.
    params.tracing.ctxId = 0;
    if (!GCFG_ENABLE_PROFILER.value())
    {
        params.tracing.traceMode = MmeCommon::e_mme_trace_mode_none;
    }
    else if (GCFG_MME_ADVANCED_PROFILE.value())
    {
        params.tracing.traceMode = MmeCommon::e_mme_trace_mode_advanced;
    }
    else if (GCFG_MME_PROFILE_PER_DESC.value())
    {
        params.tracing.traceMode = MmeCommon::e_mme_trace_mode_desc;
    }
    else
    {
        params.tracing.traceMode = MmeCommon::e_mme_trace_mode_layer_act;
    }
}

void unifyBroadCastRepresentation(MmeCommon::MmeLayerParams& params)
{
    auto& opA = params.getOperand(MmeCommon::e_mme_op_a);
    auto& opB = params.getOperand(MmeCommon::e_mme_op_b);
    for (unsigned batchDim = MmeCommon::GEMM_DIM_B1; batchDim <= MmeCommon::GEMM_DIM_B3; ++batchDim)
    {
        if (opA.strides[batchDim] == 0)
        {
            opA.sizes[batchDim] = 1;
            opA.strides[batchDim] = opA.strides[batchDim - 1] * opA.sizes[batchDim - 1];
        }

        if (opB.strides[batchDim] == 0)
        {
            opB.sizes[batchDim] = 1;
            opB.strides[batchDim] = opB.strides[batchDim - 1] * opB.sizes[batchDim - 1];
        }
    }
}

void flattenContiguousBatchDims(const MmeNode& mmeNode, MmeCommon::MmeLayerParams& params)
{
    // early break in case of strided tensors
    if (mmeNode.isDynamicShape()) return;
    for (auto op : {MmeCommon::e_mme_op_a, MmeCommon::e_mme_op_b, MmeCommon::e_mme_op_c})
    {
        if (params.getOperand(op).isStrided()) return;
    }

    for (unsigned batchDim = MmeCommon::GEMM_DIM_B3; batchDim > MmeCommon::GEMM_DIM_B1; --batchDim)
    {
        bool canFlatten = true;
        std::optional<unsigned> curBatchSize, prevBatchSize;
        for (auto op : {MmeCommon::e_mme_op_a, MmeCommon::e_mme_op_b, MmeCommon::e_mme_op_c})
        {
            // the check here can be softened to only check the relevant dim and not the whole tensor
            auto& tensor = params.getOperand(op);
            if (!curBatchSize.has_value()) curBatchSize.emplace(tensor.sizes[batchDim]);
            if (!prevBatchSize.has_value()) prevBatchSize.emplace(tensor.sizes[batchDim - 1]);
            bool isBroadcasted = curBatchSize.value() != tensor.sizes[batchDim] || prevBatchSize.value() != tensor.sizes[batchDim - 1];
            uint64_t newBatchSize = (uint64_t)tensor.sizes[batchDim] * tensor.sizes[batchDim - 1];
            // if the tensors is strided or broadcasted abort flattening, also if it would trigger an MMe recipe descriptor split
            if (isBroadcasted || newBatchSize > 256)
            {
                canFlatten = false;
                break;
            }
        }

        if (canFlatten)
        {
            for (auto op : {MmeCommon::e_mme_op_a, MmeCommon::e_mme_op_b, MmeCommon::e_mme_op_c})
            {
                auto& tensor = params.getOperand(op);
                tensor.sizes[batchDim - 1] *= tensor.sizes[batchDim];
                tensor.strides[batchDim] *= tensor.sizes[batchDim]; // we can do this because we checked above that the tensor is dense
                tensor.sizes[batchDim] = 1;
            }
        }
    }
}

// there are many cases where MME issues could have been solved by a node reshape werent fixed because
// the graph could not reshape the node.
// in this function we will locally perform such reshapes
void normalizeTensorDims(const MmeNode& mmeNode, MmeCommon::ChipType chipType, MmeCommon::MmeLayerParams& params)
{
    if (params.isGemmOperation() && !params.strategy.maskedBgemm)
    {
        unifyBroadCastRepresentation(params);
        if (chipType == MmeCommon::e_mme_Gaudi2)
        {
            // this logic can only be done inside the MME brain so that the returned multipliers would be correct and not exceed actual dim size
            flattenContiguousBatchDims(mmeNode, params);
        }
    }
}
