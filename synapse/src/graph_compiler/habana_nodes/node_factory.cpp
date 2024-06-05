#include "node_factory.h"

#include "cache_warmup_node.h"
#include "data_type_utils.h"
#include "habana_nodes.h"
#include "node_visitor.h"
#include "perf_lib_layer_params.h"

#include "types_exception.h"
#include "utils.h"

#include "broadcast_node.h"
#include "dedw_node.h"
#include "dedx_node.h"
#include "dma_memcopy_node.h"
#include "dma_memset_node.h"
#include "dma_transpose_node.h"
#include "dynamic_range_node.hpp"
#include "dynamic_reshape_shape_node.h"
#include "dynamic_split_node.h"
#include "einsum_node.h"
#include "expand_dims_shape_node.h"
#include "extract_shape_node.h"
#include "flatten_shape_node.h"
#include "frobenius_norm_node.h"
#include "h2d_tensor_op_nodes.h"
#include "hal_reader/hal_reader.h"
#include "identity_node.h"
#include "infer_max_node.h"
#include "infer_shape_node.h"
#include "memcopy_node.h"
#include "memset_node.h"
#include "merge_shape_node.h"
#include "moments_node.h"
#include "multi_insert_node.h"
#include "nms_node.h"
#include "physical_concat_node.h"
#include "physical_concat_split_subnode.h"
#include "physical_memory_ops_nodes.h"
#include "physical_reshape_node.h"
#include "physical_split_node.h"
#include "quantizer_factory.h"
#include "reduction_node.h"
#include "reinterpret_cast_node.h"
#include "shape_to_h2d_conversion_nodes.h"
#include "slice_bwd_logical_node.h"
#include "slice_fwd_logical_node.h"
#include "slice_fwd_node.h"
#include "slice_grad_node.h"
#include "slice_insert_logical_node.h"
#include "slice_insert_node.h"
#include "split_shape_node.h"
#include "squeeze_node.h"
#include "static_reshape_shape_node.h"
#include "strided_insert_logical_node.h"
#include "strided_insert_node.h"
#include "strided_view_logical_node.h"
#include "strided_view_node.h"
#include "tensor_view_node.h"
#include "tensor_view_node.h"
#include "tf_batch_norm_node.h"
#include "tile_shape_node.h"
#include "tpc_memset_node.h"
#include "tpc_node.h"
#include "transpose_node.h"
#include "transpose_nodes_creator.h"
#include "transposed_shape_node.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <string_view>

const char* NodeFactory::convolutionNodeTypeName         = "spatial_convolution";
const char* NodeFactory::convolution3DNodeTypeName       = "spatial_convolution3d";
const char* NodeFactory::gemmNodeTypeName                = "gemm";
const char* NodeFactory::gemmDeDxNodeTypeName            = "gemm_dedx";
const char* NodeFactory::gemmDeDwNodeTypeName            = "gemm_dedw";
const char* NodeFactory::batchGemmNodeTypeName           = "batch_gemm";
const char* NodeFactory::batchGemmDeDxNodeTypeName       = "batch_gemm_dedx";
const char* NodeFactory::batchGemmDeDwNodeTypeName       = "batch_gemm_dedw";
const char* NodeFactory::maskedBatchGemmNodeTypeName     = "masked_batch_gemm";
const char* NodeFactory::transposeNodeTypeName           = "transpose";
const char* NodeFactory::transposeDmaNodeTypeName        = "transpose_dma";
const char* NodeFactory::transposeMmeNodeTypeName        = "transpose_mme";
const char* NodeFactory::transposeLogicNodeTypeName      = "transpose_logic";
const char* NodeFactory::broadcastNodeTypeName           = "broadcast";
const char* NodeFactory::concatenateNodeTypeName         = "concat";
const char* NodeFactory::concatenateNodeInternalTypeName = "concat_internal";

const char* NodeFactory::concatenateNodeLogicalInternalTypeName = "concat_logical_internal";

const char* NodeFactory::reductionNodeTypeName           = "reduction";
const char* NodeFactory::stridedViewNodeTypeName         = "strided_view";
const char* NodeFactory::stridedInsertNodeTypeName       = "strided_insert";
const char* NodeFactory::multiInsertNodeTypeName         = "multi_insert";
const char* NodeFactory::logicalStridedViewTypeName      = "logical_strided_view";
const char* NodeFactory::logicalStridedInsertTypeName    = "logical_strided_insert";
const char* NodeFactory::flattenNodeTypeName             = "flatten";
const char* NodeFactory::expandDimsNodeTypeName          = "expand_dims";
const char* NodeFactory::splitNodeTypeName               = "split";
const char* NodeFactory::splitNodeInternalTypeName       = "split_internal";
const char* NodeFactory::sliceAxisNodeTypeName           = "slice_axis";
const char* NodeFactory::sliceNodeTypeName               = "slice";
const char* NodeFactory::logicalSliceFwdNodeTypeName     = "logical_slice_fwd";
const char* NodeFactory::reshapeNodeTypeName             = "reshape";
const char* NodeFactory::staticReshapeNodeTypeName       = "static_reshape";
const char* NodeFactory::dynamicReshapeNodeTypeName      = "dynamic_reshape";
const char* NodeFactory::tpcMemcpyNodeTypeName           = "memcpy_tpc";
const char* NodeFactory::dmaMemcpyNodeTypeName           = "memcpy_dma";
const char* NodeFactory::addNodeTypeName                 = "add";
const char* NodeFactory::reluNodeTypeName                = "relu";
const char* NodeFactory::reverseNodeTypeName             = "reverse";
const char* NodeFactory::deDxNodeTypeName                = "dedx";
const char* NodeFactory::deDx3DNodeTypeName              = "dedx3d";
const char* NodeFactory::deDwNodeTypeName                = "dedw";
const char* NodeFactory::deDw3DNodeTypeName              = "dedw3d";
const char* NodeFactory::tensorViewNodeTypeName          = "tensor_view";
const char* NodeFactory::embeddingNodeTypeName           = "embedding";
const char* NodeFactory::memcpyNodeTypeName              = "memcpy";
const char* NodeFactory::memsetNodeTypeName              = "memset";
const char* NodeFactory::dmaMemsetNodeTypeName           = "memset_dma";
const char* NodeFactory::tpcMemsetNodeTypeName           = "memset_tpc";
const char* NodeFactory::clAwareMemsetNodeTypeName       = "cl_aware_memset";
const char* NodeFactory::clAwareMemgetNodeTypeName       = "cl_aware_memget";
const char* NodeFactory::clAwareHybridNodeTypeName       = "cl_aware_memset_memget";
const char* NodeFactory::nmsNodeTypeName                 = "non_max_suppression";
const char* NodeFactory::waitNodeTypeName                = "wait";
const char* NodeFactory::DebugNodeTypeName               = "debug";
const char* NodeFactory::identityNodeTypeName            = "identity";
const char* NodeFactory::momentsFwdNodeTypeName          = "moments_fwd";
const char* NodeFactory::tfBatchNormNodeTypeName         = "tf_batch_normalization_fwd";
const char* NodeFactory::tfFusedBatchNormGradName        = "tf_fused_batch_norm_grad";
const char* NodeFactory::stridedSliceGradNodeTypeName    = "strided_slice_grad";
const char* NodeFactory::sliceInsertNodeTypeName         = "slice_insert";
const char* NodeFactory::logicalSliceBwdNodeTypeName     = "logical_slice_bwd";
const char* NodeFactory::logicalSliceInsertNodeTypeName  = "logical_slice_insert";
const char* NodeFactory::logicalRequantNodeTypeName      = "logical_requant";
const char* NodeFactory::rotateNodeTypeName              = "image_rotate";
const char* NodeFactory::serializeDMANodeTypeName        = "serialize_dma";
const char* NodeFactory::deserializeDMANodeTypeName      = "deserialize_dma";
const char* NodeFactory::serializeTPCNodeTypeName        = "serialize_tpc";
const char* NodeFactory::deserializeTPCNodeTypeName      = "deserialize_tpc";
const char* NodeFactory::dynamicStridedDMANodeTypeName   = "dynamic_strided_dma";
const char* NodeFactory::dynamicSliceDMANodeTypeName     = "dynamic_slice_dma";
const char* NodeFactory::dynamicStridedTPCNodeTypeName   = "dynamic_strided_tpc";
const char* NodeFactory::dynamicSliceTPCNodeTypeName     = "dynamic_slice_tpc";
const char* NodeFactory::physicalReshapeNodeTypeName     = "physical_reshape";
const char* NodeFactory::physicalConcatNodeTypeName      = "physical_concat";
const char* NodeFactory::extractShapeNodeTypeName        = "extract_shape";
const char* NodeFactory::mergeShapesNodeTypeName         = "merge_shapes";
const char* NodeFactory::splitShapeNodeTypeName          = "split_shape";
const char* NodeFactory::flattenShapeNodeTypeName        = "flatten_shape";
const char* NodeFactory::expandDimsShapeNodeTypeName     = "expand_dims_shape";
const char* NodeFactory::squeezeShapeNodeTypeName        = "squeeze_shape";
const char* NodeFactory::transposedShapeNodeTypeName     = "transpose_shape";
const char* NodeFactory::transposeSliceH2DNodeTypeName   = "transpose_slice_h2d";
const char* NodeFactory::squeezeNodeTypeName             = "squeeze";
const char* NodeFactory::FrobeniusNormTypeName           = "frobenius_norm_fwd";
const char* NodeFactory::memcpyNdNodeTypeName            = "memcpy_nd";
const char* NodeFactory::memcpyNdInt64NodeTypeName       = "memcpy_nd_i64";
const char* NodeFactory::memcpyNdUint64NodeTypeName      = "memcpy_nd_u64";
const char* NodeFactory::physicalSplitNodeTypeName       = "physical_split";
const char* NodeFactory::dynamicSplitNodeTypeName        = "dynamic_split";
const char* NodeFactory::physicalFlattenNodeTypeName     = "physical_flatten";
const char* NodeFactory::dynamicRangeNodeTypeName        = "dynamic_range";

const char* NodeFactory::physicalConcatSplitSubNodeTypeNameDMA = "physical_concat_split_sub_dma";
const char* NodeFactory::physicalConcatSplitSubNodeTypeNameTPC = "physical_concat_split_sub_tpc";
const char* NodeFactory::einsumTypeName                     = "einsum";
const char* NodeFactory::einsumExpandShapeNodeTypeName      = "einsum_expand";
const char* NodeFactory::inferShapeNodeTypeName             = "infer_shape";
const char* NodeFactory::reinterpretCastNodeTypeName        = "reinterpret_cast";
const char* NodeFactory::inferMaxShapeNodeTypeName          = "infer_max_shape";
const char* NodeFactory::tileShapeNodeTypeName              = "tile_shape";
const char* NodeFactory::transposedDeDxNodeTypeName         = "transposed_dedx";
const char* NodeFactory::transposedDeDx3DNodeTypeName       = "transposed_dedx3d";

// H2D manipulation nodes
const char* NodeFactory::dynamicStridedDmaExpandH2DNodeTypeName      = "dynamic_strided_dma_h2d_expand";
const char* NodeFactory::dynamicStridedDmaReinterpretH2DNodeTypeName = "dynamic_strided_dma_h2d_reinterpret";
const char* NodeFactory::dynamicSliceDmaExpandH2DNodeTypeName   = "dynamic_slice_dma_h2d";
const char* NodeFactory::stridedOpsConversionNodeTypeName       = "shape_to_h2d_strided_ops";
const char* NodeFactory::sliceConversionNodeTypeName            = "shape_to_h2d_slice";

// TPC kernels guids or Complex guids
// (guids defined for data type selection, quantization, etc.)
const char* NodeFactory::bitshiftNodeTypeName            = "bitshift";
const char* NodeFactory::tanhNodeTypeName                = "tanh";
const char* NodeFactory::sigmoidNodeTypeName             = "sigmoid";
const char* NodeFactory::sequenceLengthNodeTypeName      = "convert_b_to_t";
const char* NodeFactory::sequenceMaskNodeTypeName        = "sequence_mask";
const char* NodeFactory::rnnNodeTypeName                 = "rnncell";
const char* NodeFactory::softmaxNodeTypeName             = "softmax";
const char* NodeFactory::raggedSoftmaxNodeTypeName       = "ragged_softmax";
const char* NodeFactory::maxPoolRoiNodeTypeName          = "maxpool_roi";
const char* NodeFactory::andNodeTypeName                 = "and";
const char* NodeFactory::maxPool2dNodeTypeName           = "maxpool_2d";
const char* NodeFactory::avgPool2dNodeTypeName           = "avg_pool_2d";
const char* NodeFactory::orNodeTypeName                  = "or";
const char* NodeFactory::xorNodeTypeName                 = "xor";
const char* NodeFactory::dropOutNodeTypeName             = "dropout";
const char* NodeFactory::notNodeTypeName                 = "not";
const char* NodeFactory::leakyReluNodeTypeName           = "leakyrelu";
const char* NodeFactory::batchNormNodeTypeName           = "batch_norm";
const char* NodeFactory::maxPool3dNodeTypeName           = "maxpool_3d";
const char* NodeFactory::constantNodeTypeName            = "constant";
const char* NodeFactory::sequenceReverseNodeTypeName     = "sequence_reverse";
const char* NodeFactory::upsampleNodeTypeName            = "upsample";
const char* NodeFactory::negNodeTypeName                 = "neg";
const char* NodeFactory::clipNodeTypeName                = "clamp";
const char* NodeFactory::filter2dNodeTypeName            = "filter_2d";
const char* NodeFactory::staticReshapeShapeNodeTypeName  = "static_reshape_shape";
const char* NodeFactory::cropMirorNormNodeTypeName       = "crop_mirror_norm";
const char* NodeFactory::beamSearchNodeTypeName          = "topk";


// End of TPC kernel guids list

std::string_view NodeFactory::getSerializeNodeGUID()
{
    return CompilationHalReader::getHalReader()->getDeviceType() == synDeviceGaudi3
               ? serializeTPCNodeTypeName
               : serializeDMANodeTypeName;
}
std::string_view NodeFactory::getDeserializeNodeGUID()
{
    return CompilationHalReader::getHalReader()->getDeviceType() == synDeviceGaudi3
               ? deserializeTPCNodeTypeName
               : deserializeDMANodeTypeName;
}
std::string_view NodeFactory::getDynamicStridedMemcpyNodeGUID()
{
    return CompilationHalReader::getHalReader()->getDeviceType() == synDeviceGaudi3
               ? dynamicStridedTPCNodeTypeName
               : dynamicStridedDMANodeTypeName;
}
std::string_view NodeFactory::getDynamicSliceMemcpyNodeGUID()
{
    return CompilationHalReader::getHalReader()->getDeviceType() == synDeviceGaudi3
                ? dynamicSliceTPCNodeTypeName
                : dynamicSliceDMANodeTypeName;
}
std::string_view NodeFactory::getPhysicalSplitConcatSubNodeGUID()
{
    return CompilationHalReader::getHalReader()->getDeviceType() == synDeviceGaudi3
                ? physicalConcatSplitSubNodeTypeNameTPC
                : physicalConcatSplitSubNodeTypeNameDMA;
}

NodeFactory::NodeFactory()
{
    m_factoryMap.emplace(gemmNodeTypeName, &GEMMNode::createNode);
    m_factoryMap.emplace(gemmDeDxNodeTypeName, &GEMMDeToDxNode::createNode);
    m_factoryMap.emplace(gemmDeDwNodeTypeName, &GEMMDeToDwNode::createNode);
    m_factoryMap.emplace(batchGemmNodeTypeName, &BatchGemmNode::createNode);
    m_factoryMap.emplace(batchGemmDeDxNodeTypeName, &BatchGemmDeToDxNode::createNode);
    m_factoryMap.emplace(batchGemmDeDwNodeTypeName, &BatchGemmDeToDwNode::createNode);
    m_factoryMap.emplace(maskedBatchGemmNodeTypeName, &MaskedBatchGemmNode::createNode);
    m_factoryMap.emplace(transposeLogicNodeTypeName, &LogicalTransposeNode::createNode);
    m_factoryMap.emplace(broadcastNodeTypeName, &BroadcastNode::createNode);
    m_factoryMap.emplace(concatenateNodeTypeName, &ConcatenateNode::createNode);
    m_factoryMap.emplace(concatenateNodeInternalTypeName, &ConcatenateNode::createNodeInternal);
    m_factoryMap.emplace(concatenateNodeLogicalInternalTypeName, &ConcatenateNode::createNodeLogicalInternal);
    m_factoryMap.emplace(reductionNodeTypeName, &ReductionNode::createNode);
    m_factoryMap.emplace(stridedViewNodeTypeName, &StridedViewNode::createNode);
    m_factoryMap.emplace(stridedInsertNodeTypeName, &StridedInsertNode::createNode);
    m_factoryMap.emplace(multiInsertNodeTypeName, &MultiInsertNode::createNode);
    m_factoryMap.emplace(logicalStridedViewTypeName, &LogicalStridedViewNode::createNode);
    m_factoryMap.emplace(logicalStridedInsertTypeName, &LogicalStridedInsertNode::createNode);
    m_factoryMap.emplace(flattenNodeTypeName, &FlattenNode::createNode);
    m_factoryMap.emplace(expandDimsNodeTypeName, &ExpandDimsNode::createNode);
    m_factoryMap.emplace(splitNodeTypeName, &SplitNode::createNode);
    m_factoryMap.emplace(splitNodeInternalTypeName, &SplitNode::createNodeInternal);
    m_factoryMap.emplace(reshapeNodeTypeName, &ReshapeNode::createNode);
    m_factoryMap.emplace(staticReshapeNodeTypeName, &StaticReshapeNode::createNode);
    m_factoryMap.emplace(dynamicReshapeNodeTypeName, &DynamicReshapeShapeNode::createNode);
    m_factoryMap.emplace(dmaMemcpyNodeTypeName, &DMAMemcpyNode::createNode);
    m_factoryMap.emplace(tensorViewNodeTypeName, &TensorViewNode::createNode);
    m_factoryMap.emplace(memcpyNodeTypeName, &MemcpyNode::createNode);
    m_factoryMap.emplace(dmaMemsetNodeTypeName, &DMAMemsetNode::createNode);
    m_factoryMap.emplace(tpcMemsetNodeTypeName, &TPCMemsetNode::createNode);
    m_factoryMap.emplace(clAwareMemsetNodeTypeName, &CacheWarmupNode::createNode);
    m_factoryMap.emplace(clAwareMemgetNodeTypeName, &CacheWarmupNode::createNode);
    m_factoryMap.emplace(clAwareHybridNodeTypeName, &CacheWarmupNode::createNode);
    m_factoryMap.emplace(memsetNodeTypeName, &MemsetNode::createNode);
    m_factoryMap.emplace(nmsNodeTypeName, &NMSNode::createNode);
    m_factoryMap.emplace(waitNodeTypeName, &WaitNode::createNode);
    m_factoryMap.emplace(DebugNodeTypeName, &DebugNode::createNode);
    m_factoryMap.emplace(identityNodeTypeName, &IdentityNode::createNode);
    m_factoryMap.emplace(momentsFwdNodeTypeName, &MomentsNode::createNode);
    m_factoryMap.emplace(tfBatchNormNodeTypeName, &TfBatchNormNode::createNode);
    m_factoryMap.emplace(tfFusedBatchNormGradName, &TfFusedBatchNormGradNode::createNode);
    m_factoryMap.emplace(transposeDmaNodeTypeName, &DMATransposeNode::createNode);
    m_factoryMap.emplace(transposeMmeNodeTypeName, &MmeTransposeNode::createNode);
    m_factoryMap.emplace(logicalRequantNodeTypeName, &LogicalRequantNode::createNode);
    m_factoryMap.emplace(rotateNodeTypeName, &RotateNode::createNode);
    m_factoryMap.emplace(physicalReshapeNodeTypeName, &PhysicalReshapeNode::createNode);
    m_factoryMap.emplace(serializeDMANodeTypeName, &SerializeNode<DMAMemcpyNode>::createNode);
    m_factoryMap.emplace(deserializeDMANodeTypeName, &DeserializeNode<DMAMemcpyNode>::createNode);
    m_factoryMap.emplace(serializeTPCNodeTypeName, &SerializeNode<TPCMemcpyNode>::createNode);
    m_factoryMap.emplace(deserializeTPCNodeTypeName, &DeserializeNode<TPCMemcpyNode>::createNode);
    m_factoryMap.emplace(dynamicStridedDMANodeTypeName, &DynamicStridedDMAMemcpyNode::createNode);  //<DMAMemcpyNode>
    m_factoryMap.emplace(dynamicSliceDMANodeTypeName, &DynamicSliceDMAMemcpyNode::createNode);      //<DMAMemcpyNode>
    m_factoryMap.emplace(dynamicStridedTPCNodeTypeName, &DynamicStridedTPCMemcpyNode::createNode);  //<TPCMemcpyNode>
    m_factoryMap.emplace(dynamicSliceTPCNodeTypeName, &DynamicSliceTPCMemcpyNode::createNode);      //<TPCMemcpyNode>
    m_factoryMap.emplace(physicalConcatNodeTypeName, &PhysicalConcatNode::createNode);
    m_factoryMap.emplace(extractShapeNodeTypeName, &ExtractShapeNode::createNode);
    m_factoryMap.emplace(mergeShapesNodeTypeName, &MergeShapesNode::createNode);
    m_factoryMap.emplace(splitShapeNodeTypeName, &SplitShapeNode::createNode);
    m_factoryMap.emplace(flattenShapeNodeTypeName, &FlattenShapeNode::createNode);
    m_factoryMap.emplace(expandDimsShapeNodeTypeName, &ExpandDimsShapeNode::createNode);
    m_factoryMap.emplace(squeezeShapeNodeTypeName, &SqueezeShapeNode::createNode);
    m_factoryMap.emplace(squeezeNodeTypeName, &SqueezeNode::createNode);
    m_factoryMap.emplace(FrobeniusNormTypeName, &FrobeniusNormNode::createNode);
    m_factoryMap.emplace(physicalSplitNodeTypeName, &PhysicalSplitNode::createNode);
    m_factoryMap.emplace(dynamicSplitNodeTypeName, &DynamicSplitNode::createNode);
    m_factoryMap.emplace(physicalFlattenNodeTypeName, &PhysicalFlattenNode::createNode);
    m_factoryMap.emplace(dynamicRangeNodeTypeName, &DynamicRangeNode::createNode);
    m_factoryMap.emplace(inferShapeNodeTypeName, &InferShapeNode::createNode);
    m_factoryMap.emplace(reinterpretCastNodeTypeName, &ReinterpretCastNode::createNode);
    m_factoryMap.emplace(inferMaxShapeNodeTypeName, &InferMaxShapeNode::createNode);

    m_factoryMap.emplace(physicalConcatSplitSubNodeTypeNameDMA, &PhysicalConcatSplitSubnodeDMA::createNode);
    m_factoryMap.emplace(physicalConcatSplitSubNodeTypeNameTPC, &PhysicalConcatSplitSubnodeTPC::createNode);

    m_factoryMap.emplace(einsumTypeName, &EinsumNode::createNode);
    m_factoryMap.emplace(einsumExpandShapeNodeTypeName, &EinsumExpandShapeNode::createNode);
    m_factoryMap.emplace(logicalSliceBwdNodeTypeName, &LogicalSliceBwdNode::createNode);
    m_factoryMap.emplace(logicalSliceInsertNodeTypeName, &LogicalSliceInsertNode::createNode);
    m_factoryMap.emplace(logicalSliceFwdNodeTypeName, &LogicalSliceFwdNode::createNode);
    m_factoryMap.emplace(staticReshapeShapeNodeTypeName, &StaticReshapeShapeNode::createNode);

    m_factoryMap.emplace(dynamicStridedDmaExpandH2DNodeTypeName, &DynamicStridedDmaExpandH2DNode::createNode);
    m_factoryMap.emplace(dynamicStridedDmaReinterpretH2DNodeTypeName, &DynamicStridedDmaReinterpretH2DNode::createNode);
    m_factoryMap.emplace(dynamicSliceDmaExpandH2DNodeTypeName, &DynamicSliceDmaExpandH2DNode::createNode);
    m_factoryMap.emplace(stridedOpsConversionNodeTypeName, &StridedOpsConversionNode::createNode);
    m_factoryMap.emplace(sliceConversionNodeTypeName, &SliceConversionNode::createNode);

    m_factoryMapWithSize.emplace(convolutionNodeTypeName, &ConvolutionNode::createNode);
    m_factoryMapWithSize.emplace(convolution3DNodeTypeName, &ConvolutionNode::createNode);

    m_factoryMapWithSize.emplace(deDxNodeTypeName, &DeToDxNode::createNode);
    m_factoryMapWithSize.emplace(deDx3DNodeTypeName, &DeToDxNode::createNode);
    m_factoryMapWithSize.emplace(deDwNodeTypeName, &DeToDwNode::createNode);
    m_factoryMapWithSize.emplace(deDw3DNodeTypeName, &DeToDwNode::createNode);
    m_factoryMapWithSize.emplace(transposeNodeTypeName, &TransposeNode::createNode);
    m_factoryMapWithSize.emplace(transposedShapeNodeTypeName, &TransposedShapeNode::createNode);
    m_factoryMapWithSize.emplace(transposeSliceH2DNodeTypeName, &TransposeSliceH2DNode::createNode);
    m_factoryMapWithSize.emplace(sliceNodeTypeName, &SliceFwdNode::createNode);
    m_factoryMapWithSize.emplace(stridedSliceGradNodeTypeName, &SliceGradNode::createNode);
    m_factoryMapWithSize.emplace(sliceInsertNodeTypeName, &SliceInsertNode::createNode);
    m_factoryMapWithSize.emplace(sliceAxisNodeTypeName, &SliceAxisNode::createNode);
    m_factoryMapWithSize.emplace(transposedDeDxNodeTypeName, &TransposedDedxNode::createNode);
    m_factoryMapWithSize.emplace(transposedDeDx3DNodeTypeName, &TransposedDedxNode::createNode);
    m_factoryMapWithSize.emplace(tileShapeNodeTypeName, &TileShapeNode::createNode);
    m_factoryMapWithSize.emplace(tpcMemcpyNodeTypeName, &TPCMemcpyNode::createNode);

    m_internalNodes.emplace(logicalSliceFwdNodeTypeName);
    m_internalNodes.emplace(logicalSliceBwdNodeTypeName);
    m_internalNodes.emplace(logicalSliceInsertNodeTypeName);
    m_internalNodes.emplace(dmaMemsetNodeTypeName);
    m_internalNodes.emplace(tpcMemsetNodeTypeName);
    m_internalNodes.emplace(clAwareMemsetNodeTypeName);
    m_internalNodes.emplace(clAwareMemgetNodeTypeName);
    m_internalNodes.emplace(transposeDmaNodeTypeName);
    m_internalNodes.emplace(transposeMmeNodeTypeName);
    m_internalNodes.emplace(tensorViewNodeTypeName);
    m_internalNodes.emplace(tpcMemcpyNodeTypeName);
    m_internalNodes.emplace(dmaMemcpyNodeTypeName);
    m_internalNodes.emplace(dynamicStridedDMANodeTypeName);
    m_internalNodes.emplace(dynamicSliceDMANodeTypeName);
    m_internalNodes.emplace(reductionNodeTypeName);
    m_internalNodes.emplace(waitNodeTypeName);
    m_internalNodes.emplace(logicalRequantNodeTypeName);
    m_internalNodes.emplace(logicalStridedViewTypeName);
    m_internalNodes.emplace(logicalStridedInsertTypeName);
    m_internalNodes.emplace(multiInsertNodeTypeName);
    m_internalNodes.emplace(transposeLogicNodeTypeName);
    m_internalNodes.emplace(splitNodeInternalTypeName);
    m_internalNodes.emplace(concatenateNodeInternalTypeName);
    m_internalNodes.emplace(inferMaxShapeNodeTypeName);
    m_internalNodes.emplace(tileShapeNodeTypeName);
    m_internalNodes.emplace(inferShapeNodeTypeName);
    m_internalNodes.emplace(reverseNodeTypeName);
    m_internalNodes.emplace(transposedDeDxNodeTypeName);
    m_internalNodes.emplace(transposedDeDx3DNodeTypeName);
    m_internalNodes.emplace(transposeSliceH2DNodeTypeName);

    m_apiNodes.emplace(NodeFactory::momentsFwdNodeTypeName);
    m_apiNodes.emplace(NodeFactory::FrobeniusNormTypeName);
    m_apiNodes.emplace(NodeFactory::einsumTypeName);
    m_apiNodes.emplace(NodeFactory::transposeNodeTypeName);
    m_apiNodes.emplace(NodeFactory::splitShapeNodeTypeName);
    m_apiNodes.emplace(NodeFactory::stridedViewNodeTypeName);
    m_apiNodes.emplace(NodeFactory::stridedInsertNodeTypeName);
    m_apiNodes.emplace(NodeFactory::stridedSliceGradNodeTypeName);
    m_apiNodes.emplace(NodeFactory::sliceInsertNodeTypeName);
    m_apiNodes.emplace(NodeFactory::squeezeNodeTypeName);
    m_apiNodes.emplace(NodeFactory::splitNodeTypeName);
    m_apiNodes.emplace(NodeFactory::sliceAxisNodeTypeName);
    m_apiNodes.emplace(NodeFactory::sliceNodeTypeName);
    m_apiNodes.emplace(NodeFactory::flattenNodeTypeName);
    m_apiNodes.emplace(NodeFactory::expandDimsNodeTypeName);
    m_apiNodes.emplace(NodeFactory::identityNodeTypeName);
    m_apiNodes.emplace(NodeFactory::memcpyNodeTypeName);
    m_apiNodes.emplace(NodeFactory::memsetNodeTypeName);
    m_apiNodes.emplace(NodeFactory::reinterpretCastNodeTypeName);
    m_apiNodes.emplace(NodeFactory::reshapeNodeTypeName);
    m_apiNodes.emplace(NodeFactory::broadcastNodeTypeName);
    m_apiNodes.emplace(NodeFactory::concatenateNodeTypeName);
    m_apiNodes.emplace(NodeFactory::batchGemmNodeTypeName);
    m_apiNodes.emplace(NodeFactory::batchGemmDeDxNodeTypeName);
    m_apiNodes.emplace(NodeFactory::batchGemmDeDwNodeTypeName);
    m_apiNodes.emplace(NodeFactory::convolutionNodeTypeName);
    m_apiNodes.emplace(NodeFactory::convolution3DNodeTypeName);
    m_apiNodes.emplace(NodeFactory::deDxNodeTypeName);
    m_apiNodes.emplace(NodeFactory::deDx3DNodeTypeName);
    m_apiNodes.emplace(NodeFactory::deDwNodeTypeName);
    m_apiNodes.emplace(NodeFactory::deDw3DNodeTypeName);
    m_apiNodes.emplace(NodeFactory::gemmNodeTypeName);
    m_apiNodes.emplace(NodeFactory::gemmDeDxNodeTypeName);
    m_apiNodes.emplace(NodeFactory::gemmDeDwNodeTypeName);
    m_apiNodes.emplace(NodeFactory::maskedBatchGemmNodeTypeName);
}

NodeFactory::~NodeFactory()
{
    clear();
}

void NodeFactory::clear() {}

NodePtr NodeFactory::createNode(const TensorVector&         inputs,
                                const TensorVector&         outputs,
                                UserParams                  userParams,
                                unsigned                    paramsSize,
                                std::string_view            guid,
                                std::string_view            name,
                                const Node::NodeProperties& properties,
                                const synDeviceType*        deviceType)
{
    bool   isMarkedAsTPCNode = false;
    size_t guidLength        = guid.length();
    // Force as TPCNode if the guid name ends with "_runOnTpc"
    if (guidLength >= TPCNode::RUN_ON_TPC.length() &&
        !guid.compare(guidLength - TPCNode::RUN_ON_TPC.length(),
                         TPCNode::RUN_ON_TPC.length(),
                         TPCNode::RUN_ON_TPC))
    {
        guidLength = guidLength - TPCNode::RUN_ON_TPC.length();
        isMarkedAsTPCNode = true;
    }
    std::string_view updatedGuid(guid.data(), guidLength);
    std::string guidWithoutDType = std::string(extractGUIDFromFullGUID(updatedGuid));
    std::transform(guidWithoutDType.begin(), guidWithoutDType.end(), guidWithoutDType.begin(), ::tolower);

    return createNode(inputs,
                      outputs,
                      userParams,
                      paramsSize,
                      updatedGuid,
                      guidWithoutDType,
                      name,
                      isMarkedAsTPCNode,
                      properties,
                      deviceType);
}

NodePtr NodeFactory::createNode(const TensorVector&         inputs,
                                const TensorVector&         outputs,
                                UserParams                  userParams,
                                unsigned                    paramsSize,
                                std::string_view            guid,
                                std::string_view            guidWithoutDType,
                                std::string_view            name,
                                bool                        isMarkedAsTPCNode,
                                const Node::NodeProperties& properties,
                                const synDeviceType*        deviceType)
{
    NodePtr node;
    StringViewWithHash guidWithHash(guidWithoutDType);
    if (!isMarkedAsTPCNode)
    {
        auto itr = NodeFactory::getInstance().m_factoryMap.find(guidWithHash);
        if (itr != NodeFactory::getInstance().m_factoryMap.end())
        {
            node = itr->second(inputs, outputs, userParams, guidWithoutDType, name);
        }
        else
        {
            auto itrWithSize = NodeFactory::getInstance().m_factoryMapWithSize.find(guidWithHash);
            if (itrWithSize != NodeFactory::getInstance().m_factoryMapWithSize.end())
            {
                node = itrWithSize->second(inputs, outputs, userParams, paramsSize, guidWithoutDType, name);
            }
        }
    }

    if (node == nullptr)
    {
        node = TPCNode::createNode(inputs, outputs, userParams, paramsSize, guid, name);
    }
    else if (node->getGUID().empty())
    {
        node->setGUID(guidWithHash);
    }

    if (guid.size() > guidWithoutDType.size())
    {
        node->setNodePrecisionFromGUID(guid);
    }
    node->setQuantizer(QuantizerFactory::getNodeQuantizer(guidWithHash));

    if (!properties.inputLayouts.empty())
    {
        node->setInputLayouts(properties.inputLayouts);
    }
    if (!properties.outputLayouts.empty())
    {
        node->setOutputLayouts(properties.outputLayouts);
    }

    bool postponeValidation = false;
    if (node->requiresOutputMaxDimInfer())
    {
        if (deviceType && std::all_of(inputs.begin(), inputs.end(), [](const TensorPtr& t) {
                return !t || t->isPropSet(synTensorPropGeometryMax);
            }))
        {
            // Opportunistic SIF
            LOG_TRACE(GC, "max-dims infer required for node \"{}\"", node->getNodeName());
            if (!node->inferOutputsSizes(*deviceType, /*inferMax*/ true))
            {
                LOG_ERR(GC, "Failure to update output shape for node: \"{}\"", node->getNodeName());
                throw InvalidNodeParamsException(node->getNodeName());
            }
        }
        else
        {
            LOG_TRACE(GC,
                      "max-dims infer required for node \"{}\" but postponed until graph compilation since at node "
                      "addition time input shape is unknown",
                      node->getNodeName());
            postponeValidation = true;
        }
    }
    if (postponeValidation)
    {
        LOG_INFO(HABANA_NODE,
                 "Node Validation for node {} postponed until after max dim inference during compilation.",
                 node->getNodeName());
    }
    else if (!node->validateNode())
    {
        LOG_ERR(HABANA_NODE, "Node Validation Failed. Cannot create node {}.", node->getNodeName());
        throw InvalidNodeParamsException(node->getNodeName());
    }

    if (userParams != nullptr)
    {
        node->setParamsRawData(userParams, paramsSize);
    }

    return node;
}

NodePtr NodeFactory::createGenericTPCNode(const TensorVector& inputs,
                                          const TensorVector& outputs,
                                          UserParams          userParams,
                                          unsigned            paramsSize,
                                          std::string_view    guid,
                                          std::string_view    name)
{
    // createGenericTPCNode enable to create a node directly without any validation,
    // and the node will be fused in pass in the future
    auto node = TPCNode::createNode(inputs, outputs, userParams, paramsSize, guid, name);
    if (userParams != nullptr)
    {
        node->setParamsRawData(userParams, paramsSize);
    }
    node->setNodePrecisionFromGUID(guid);
    return node;
}

//Debug nodes
NodePtr NodeFactory::createDebugNode(const TensorPtr& opA, const TensorPtr& opB, const std::string& name)
{
    NodePtr ret = NodePtr(new DebugNode(opA, opB, name));
    return ret;
}
NodePtr NodeFactory::createDebug2Node(const TensorPtr& opA, const TensorPtr& opB, const std::string& name)
{
    NodePtr ret = NodePtr(new Debug2Node(opA, opB, name));
    return ret;
}
NodePtr NodeFactory::createDebugForkNode(const TensorPtr& opA, const TensorPtr& opB, const TensorPtr& opC, const std::string& name)
{
    NodePtr ret = NodePtr(new DebugForkNode(opA, opB, opC, name));
    return ret;
}
NodePtr NodeFactory::createDebugJoinNode(const TensorPtr& opA, const TensorPtr& opB, const TensorPtr& opC, const std::string& name)
{
    NodePtr ret = NodePtr(new DebugJoinNode(opA, opB, opC, name));
    return ret;
}

std::size_t NodeFactory::getNumApiNodes()
{
    return NodeFactory::getInstance().m_apiNodes.size();
}

// TODO [SW-117765]: Use isApiNode() instead of isInternalNode()
bool NodeFactory::isApiNode(const std::string& guid)
{
    const auto& apiNodes = NodeFactory::getInstance().m_apiNodes;
    return apiNodes.find(guid) != apiNodes.end();
}

bool NodeFactory::isInternalNode(const StringViewWithHash& guidWithHash)
{
    const auto& internalNodes = NodeFactory::getInstance().m_internalNodes;
    return internalNodes.find(guidWithHash) != internalNodes.end();
}
