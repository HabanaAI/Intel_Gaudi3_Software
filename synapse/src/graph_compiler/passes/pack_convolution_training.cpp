#include "data_type_utils.h"
#include "defs.h"
#include "habana_graph.h"
#include "graph_editor.h"
#include "log_manager.h"
#include "perf_lib_layer_params.h"
#include "node_factory.h"
#include "compilation_hal_reader.h"
#include "pack_convolution_training.h"
#include "types.h"

#include "eager/eager_interface.h"
#include "eager/lib/eager_brain_base.h"
#include "eager/lib/utils/general_defs.h"

bool ConvolutionPackingManagerTraining::packingEnabled() const
{
    return GCFG_ENABLE_CONV_PACKING_TRAINING.value();
}

bool ConvolutionPackingManagerTraining::shouldBlockPacking(const synConvolution3DParamsV2& convParams,
                                                           const TensorPtr&                wTensor,
                                                           const TensorPtr&                outputTensor,
                                                           bool                            isBwd) const
{
    // In Fwd- block dynamic W/C dimension, as packing factor might not be suitable to the actual size
    // In Bwd- the sif is not aware of which dims are dynamic, so the shape tensor overrides the node output entirely.
    //         The user is unaware of the packing, so shape tensor is wrong for packed dedx.
    //         Also, in case any of the kernels' strides != 1, should block packing.
    bool isFwdDynamicWorC = !isBwd && (outputTensor->isDynamicDim(DIM_W) || outputTensor->isDynamicDim(DIM_C));
    bool isBwdDynamic     = isBwd && (outputTensor->isDynamicShape());
    bool isBwdStrided =
        isBwd && (convParams.stride[CONV_STRIDE_WIDTH] != 1 || convParams.stride[CONV_STRIDE_HEIGHT] != 1 ||
                  convParams.stride[CONV_STRIDE_DEPTH] != 1);
    if (isFwdDynamicWorC || isBwdDynamic || isBwdStrided)
    {
        return true;
    }

    auto weightElementType    = wTensor->getElementType();
    bool bIsSupportedDataType = (weightElementType == syn_type_bf16) || (weightElementType == syn_type_fp16) ||
                                (weightElementType == syn_type_float) || weightElementType == syn_type_hb_float ||
                                (weightElementType == syn_type_fp8_152);

    if (!bIsSupportedDataType)
    {
        return true;
    }

    return ConvolutionPackingManager::shouldBlockPacking(convParams,
                                                         wTensor,
                                                         outputTensor,
                                                         isBwd);
}

void ConvolutionPackingManagerTraining::packOutput(const NodePtr&   node,
                                                   const TensorPtr& outTensorOrig,
                                                   unsigned         packingFactor)
{
    ConvolutionPackingManager::packOutput(node, outTensorOrig, packingFactor);

    // Handle shape tensor if exists. This is required because static dedx might have shape tensor.
    const TensorPtr& shapeTensor = node->getInput(2);
    if (shapeTensor && shapeTensor->getTensorType() == OUTPUT_DESCRIBING_SHAPE_TENSOR)
    {
        HB_ASSERT(!shapeTensor->isDynamicShape(),
                  "cannot pack node {} with dynamic shape tensor!",
                  node->getNodeName());
        m_shapeTensorInputToRemove = shapeTensor;
    }
}

// Returns the number of common dim elements, which is a lower limit on the MME computation. Any smaller number of
// common dim elements will have the same time cost. If the utilization is minimal anyway - returns 0.
unsigned ConvolutionPackingManagerTraining::minCDElementsForFullMmeUtil(synDataType inType,
                                                                        synDataType outType,
                                                                        unsigned    outputHeight) const
{
    // The MME minimal common dim size (bytes / elements) is set for best MME utilization. It makes sure no computation
    // bubbles happen. In Gaudi such compuitation bubbles may happen due to slower rollup, which is derived from how
    // many lines should be sent out (lines are evicted one by one), compared to multipliers calculation, which is
    // derived from how many elements are summed by the muptiplier (num of elements in the common dim)
    const auto& halReader     = CompilationHalReader::getHalReader();
    unsigned    minCDElements = halReader->getMmeMinCDInElements(inType, outType);

    // 128 is the estimation of cycles for next input reading to EU, which may also create a compute bubble
    minCDElements = std::max(minCDElements, 128U);

    // In 4xh geometry, which is probable for the packing scenario because K is small, every EU receives InputRows/4 to
    // work on, so the output size for each EU is 1/4 the total output rows.

    float outHeightPerEU = (outputHeight / 4);
    float maxEUHeight    = halReader->getMmeMaximalEUHeightInElems(inType);
    if (outHeightPerEU < maxEUHeight)
    {
        // For output height smaller than EU height, utilization is not worth the packing overhead.
        minCDElements = 0;
    }
    return minCDElements;
}

// add a tpc kernel to pack the weights
void ConvolutionPackingManagerTraining::packWeights(const MMENodePtr& convNode,
                                                    const TensorPtr&  origWeights,
                                                    unsigned          stride,
                                                    unsigned          packingFactor)
{
    bool isBwd = convNode->getNodeType() == Node::TYPE_DEDX;
    // the packed dim is the output channels, so it's set by the op direction
    unsigned packedDim = isBwd ? WEIGHT_DIM_C : WEIGHT_DIM_K;

    SizeArray newSizes;
    origWeights->getAllSizesInElements(newSizes);
    // weights are being duplicated for each output packingFactor
    newSizes[packedDim] = newSizes[packedDim] * packingFactor;
    // each duplication adds stride to S dimension, so it is filled with zero when shouldn't be taken into account
    newSizes[WEIGHT_DIM_S] = newSizes[WEIGHT_DIM_S] + static_cast<TSize>(stride) * (packingFactor - 1);

    TensorPtr tpcOut = std::make_shared<Tensor>(origWeights->getDim(), newSizes.data(), origWeights->getElementType());
    tpcOut->setName(fmt::format("{}_packed", origWeights->getName()));
    tpcOut->getTensorAnnotation().dataInfo.packing[PACKING_X] = packingFactor;

    ns_WtPack::Params params  = {packingFactor, stride};
    const std::string guidStr = fmt::format("{}{}",
                                            isBwd ? "conv_weight_packing_bwd_" : "conv_weight_packing_fwd_",
                                            getDtypeSuffixFromSynDataType(origWeights->getElementType()));
    m_packingNode             = NodeFactory::createNode({origWeights},
                                            {tpcOut},
                                            &params,
                                            guidStr,
                                            fmt::format("{}_tpc_packing", convNode->getNodeName()));
}

void ConvolutionPackingManagerTraining::applyChangeToGraph(const NodePtr& node)
{
    // changes due to weights packing
    bool status = GraphEditor::addNode(m_graph, m_packingNode);
    HB_ASSERT(status, "Failed adding tpc packing node for {}", node->getNodeName());
    // replace the old weights with the new packed weights
    GraphEditor::replaceInput(m_graph, node, TENSOR_WEIGHT, m_packingNode->getOutput(0));

    // changes due to output packing
    GraphEditor::replaceTensor(m_graph, node, m_reshapeNodeOfm->getOutput(0), m_reshapeNodeOfm->getInput(0));
    GraphEditor::addNode(m_graph, m_reshapeNodeOfm);
    if (m_shapeTensorInputToRemove != nullptr)
    {
        GraphEditor::editNode(m_graph, node, [&](const NodePtr& n) { n->removeInput(m_shapeTensorInputToRemove); });
    }
}

void ConvolutionPackingManagerTraining::resetTemporaryState()
{
    m_packingNode.reset();
    m_reshapeNodeOfm.reset();
    m_shapeTensorInputToRemove.reset();
}

bool packingMmeNodes(HabanaGraph& g)
{
    ConvolutionPackingManagerTraining convolutionPackingMgr(g);
    return convolutionPackingMgr.packConvolutionNodes();
}

void EagerConvolutionPackingManagerTraining::applyChangeToGraph(const NodePtr& node)
{
    // changes due to weights packing
    node->replaceInput(TENSOR_WEIGHT, m_packingNode->getOutput(0));

    // changes due to output packing
    node->replaceTensor(m_reshapeNodeOfm->getOutput(0), m_reshapeNodeOfm->getInput(0));
    if (m_shapeTensorInputToRemove != nullptr)
    {
        node->removeInput(m_shapeTensorInputToRemove);
    }
}

bool EagerConvolutionPackingManagerTraining::canExtract() const
{
    if (!isCandidateForPacking(m_node))
    {
        return false;
    }
    m_packingFactor = choosePackingFactor(m_node);
    if (m_packingFactor == 1) return false;
    // the fact there is a potential for packing does not mean we would use packing.
    // packing has an impact to compile time so we wish to make another cheap check
    // to understand if the gain in runtime jusify the degradation to compile time.
    const eager_mode::EagerMmeBrainBase& eagerMmeBrain = eager_mode::getEagerMmeBrain(m_graph);
    const auto&                       convNode   = static_cast<const ConvBaseNode&>(*m_node);
    return eagerMmeBrain.shouldPackConvWeights(convNode, m_packingFactor);
}

std::pair<NodePtr, NodePtr> EagerConvolutionPackingManagerTraining::extract()
{
    m_node->getNodeAnnotation().mmeMetaData.packing[PACKING_X] = m_packingFactor;
    packConvNode(m_node);
    return {m_packingNode, m_reshapeNodeOfm};
}