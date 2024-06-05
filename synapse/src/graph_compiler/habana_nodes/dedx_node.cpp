#include "dedx_node.h"

#include "graph_traits.h"
#include "habana_graph.h"
#include "node_factory.h"
#include "synapse_types_operators.h"
#include "tensor_shape.h"
#include "utils.h"

DeToDxNode::DeToDxNode(const TensorVector& inputs,
                       const TensorVector& outputs,
                       std::string_view    name,
                       Node::eNodeType     type)
: BaseClass(inputs, outputs, name, type, SIF_CONV_DEDX)
{
}

NodePtr DeToDxNode::createNode(const TensorVector& inputs,
                               const TensorVector& outputs,
                               UserParams          userParams,
                               unsigned            userParamsSize,
                               std::string_view    guid,
                               std::string_view    name)
{
    DeToDxNode* DeToDxNode_node = new DeToDxNode(inputs, outputs, name);
    DeToDxNode_node->BaseClass::setGUID(guid);
    DeToDxNode_node->setParams(userParams, userParamsSize);
    return NodePtr(DeToDxNode_node);
}

bool DeToDxNode::is3DConvolutionGuid() const
{
    return BaseClass::getGUID() == NodeFactory::deDx3DNodeTypeName;
}

NodePtr DeToDxNode::clone() const
{
    return NodePtr(new DeToDxNode(*this));
}

TensorSemanticType DeToDxNode::getParamSemanticType(const TensorPtr& param) const
{
   return TYPE_ACTIVATION;
}

TensorShape DeToDxNode::getInputShape(const TensorShape& output, uint32_t outputIdx, uint32_t inputIdx) const
{
    TensorShape inputShape;
    HB_ASSERT(outputIdx == TENSOR_DEDX, "output index mismatch, real:{}, expected:{}", outputIdx, TENSOR_DEDX);
    if (inputIdx == TENSOR_DEDY)
    {
        TensorShape shiftedOutput = output;
        // Check if the queried shape base is the beginning of the output tensor, meaning it's the first tensor slice.
        if (output.getBases() == CoordArray {0})
        {
            CoordArray shiftedBase {0};
            // The first output slice in dedx "real" start is the padding offset. Adjust the base of the given slice to
            // reflect that. Shift all dims so the input shape func won't fail on the non sliced dims.
            const synConvolution3DParamsV2& convParams = getConvolutionParams();
            shiftedBase[DIM_W] -=
                convParams.padding[ConvBaseNode::dimIndexToConvParamsIndices(DIM_W).paddingBeforeIndex];
            shiftedBase[DIM_H] -=
                convParams.padding[ConvBaseNode::dimIndexToConvParamsIndices(DIM_H).paddingBeforeIndex];
            shiftedBase[DIM_D_FOR_5D_TENSOR] -=
                convParams.padding[ConvBaseNode::dimIndexToConvParamsIndices(DIM_D_FOR_5D_TENSOR).paddingBeforeIndex];
            shiftedOutput.setBase(shiftedBase);
        }
        inputShape = getYOperandShape(shiftedOutput);
    }
    else
    {
        inputShape = Node::getInputShape(output, outputIdx, inputIdx);
    }

    return inputShape;
}

bool DeToDxNode::validateNode() const
{
    if ((m_inputs.size() != 2 && m_inputs.size() != 3) || m_outputs.size() != 1)
    {
        LOG_ERR(HABANA_NODE, "Invalid number of operands (expecting 2 or 3 inputs and 1 output)");
        return false;
    }
    if (m_inputs.size() == 3 && !m_inputs.back()->isShapeTensor())
    {
        LOG_ERR(HABANA_NODE, "Invalid inputs, expecting shape tensor at index 2");
        return false;
    }
    return BaseClass::validateNode();
}

bool DeToDxNode::RunOnCpu()
{
    HB_ASSERT(false, "currently not implemented");
    return false;
}

bool DeToDxNode::validateNodeForGraph(const HabanaGraph& g) const
{
    return g.getTraits().trainingGraph() && BaseClass::validateNodeForGraph(g);
}

bool DeToDxNode::isOperandTransposed(const TensorPtr& t) const
{
    return t == getWOperand();
}

TensorPtr DeToDxNode::getXOperand() const
{
    return getOutput(TENSOR_DEDX);
}

TensorPtr DeToDxNode::getYOperand() const
{
    return getInput(TENSOR_DEDY);
}

TensorPtr DeToDxNode::getWOperand() const
{
    return getInput(TENSOR_WEIGHT);
}

bool DeToDxNode::isSpatialSlicingSupported(unsigned dim) const
{
    // slicing packed node on W is problematic, as this dimension structure is assumed for packing to be correct
    bool dedxSupported = GCFG_SRAM_SLICER_4D_DEDX_SPATIAL_SLICE_ENABLED.value() &&
                         ((getNodeAnnotation().mmeMetaData.packing[PACKING_X] == 1) || dim != DIM_W);

    return (ConvBaseNode::isSpatialSlicingSupported(dim) && dedxSupported);
}

// Returns the minimal ROI size according to the convolution parameters.
// This function doesn't consider the tensor size, and its output may have to be clipped.
// TODO handle paddingType here <===== PADDING_TYPE
TSize DeToDxNode::getMinSpatialDimOutputROI(unsigned dim) const
{
    const synConvolution3DParamsV2& convParams  = getConvolutionParams();
    auto                            spatialDims = getSpatialDims();
    HB_ASSERT(std::find(spatialDims.begin(), spatialDims.end(), dim) != spatialDims.end(), "Invalid spatial dim");
    ConvParamsIndices convIdx       = dimIndexToConvParamsIndices(dim);
    unsigned          kernel        = convParams.kernel[convIdx.spatialIndex];
    unsigned          stride        = convParams.stride[convIdx.spatialIndex];
    unsigned          dilation      = convParams.dilation[convIdx.spatialIndex];
    int               paddingBefore = convParams.padding[convIdx.paddingBeforeIndex];
    int               overlap       = getInputROIOverlapForDim(TENSOR_IFM, dim);

    TSize minSize = 0;
    // Handle negative limits by setting them to 0
    // The overlap limit validates each input slice includes more than the overlap elements
    TSize overlapLimit = std::max(overlap, 0);
    // The padding limit validates the first X operand slice includes more than the padding dummy elements
    TSize paddingLimit = std::max(paddingBefore, 0);

    // For dedx - the overlap is actually the offset of the first Y slice, which is the input.
    // So need to make sure the Y slice size is larger than the twice the overlap
    // (first overlap for the first slice to have at least 1 real line, and second overlap to make sure
    // the second slice doesn't start in negative offset, which is difficult to handle).
    TSize yMinSize = overlapLimit * 2 + 1;
    // Calculate the required output (dX) slice size based on the minimal Y slice.
    int slicePadBefore = 0, slicePadAfter = 0;
    getXStrideAlignedROIPaddingForDim(dim, yMinSize, slicePadBefore, slicePadAfter);
    overlapLimit = convInputDimSize(yMinSize, kernel, stride, slicePadBefore + slicePadAfter, dilation);
    // Add 1 to the limit so the min size doesn't turn out to be 0 after the padding / overlap are reduced
    minSize = std::max(overlapLimit, paddingLimit) + 1;

    // For dedx - align the minimal size with the stride size. The dedx slices are assumed to start in stride aligned
    // position, so the minimal slice size must allow all slices to keep the stride aligned assumption.
    if ((stride != 1) && (minSize % stride) != 0)
    {
        // Align to stride and increase by 1 stride to be larger than the unaligned min
        minSize = minSize - (minSize % stride) + stride;
    }
    HB_ASSERT(minSize > 0, "Output ROI min size must be > 0");
    return minSize;
}

TensorPtr DeToDxNode::getShapeOperand() const
{
    return getInput(TENSOR_SHAPE_DEDX);
}

// TransposedDedx Node
TransposedDedxNode::TransposedDedxNode(const TensorVector& inputs, const TensorVector& outputs, std::string_view name)
: DeToDxNode(inputs, outputs, name, Node::TYPE_TRANSPOSED_DEDX)
{
}

NodePtr TransposedDedxNode::createNode(const TensorVector& inputs,
                                       const TensorVector& outputs,
                                       UserParams          userParams,
                                       unsigned            userParamsSize,
                                       std::string_view    guid,
                                       std::string_view    name)
{
    TransposedDedxNode* TransposedDedxNode_node = new TransposedDedxNode(inputs, outputs, name);
    TransposedDedxNode_node->BaseClass::setGUID(guid);
    TransposedDedxNode_node->setParams(userParams, userParamsSize);
    return NodePtr(TransposedDedxNode_node);
}

NodePtr TransposedDedxNode::clone() const
{
    return NodePtr(new TransposedDedxNode(*this));
}

bool TransposedDedxNode::isOperandTransposed(const TensorPtr& t) const
{
    return false;
}

bool TransposedDedxNode::is3DConvolutionGuid() const
{
    const char* guid = BaseClass::getGUID().c_str();
    return std::strcmp(guid, NodeFactory::transposedDeDx3DNodeTypeName) == 0;
}

bool TransposedDedxNode::validateNodeLayout() const
{
    SET_TEMP_LOG_CONTEXT(getNodeName());

    bool ret = MmeNode::validateNodeLayout();

    // Validate convolution size only for non extracted nodes
    // Extracted nodes are created from graph compiler multi nodes
    // If the conv is lowered the validation is incorrect due to
    // reshape params not fully updated in the conv node
    if (ret && !getNodeAnnotation().isExtracted && !getWOperand()->isLowered())
    {
        ret = validateTransposedDedxSize(getXOperand()->getAllSizesInElements(),
                                         getWOperand()->getAllSizesInElements(),
                                         getYOperand()->getAllSizesInElements(),
                                         getYOperand()->getDim(),
                                         m_params);
        ret &= validateConvPadding(getXOperand()->getAllSizesInElements(),
                                   getWOperand()->getAllSizesInElements(),
                                   getYOperand()->getAllSizesInElements(),
                                   getYOperand()->getDim(),
                                   m_params);
    }

    return ret;
}