#include "conv_base_node.h"

#include "convolution_node.h"
#include "habana_graph.h"
#include "hal_reader/hal_reader.h"
#include "infra/cpu_calculator.h"
#include "node_factory.h"
#include "synapse_types_operators.h"
#include "tensor_shape.h"
#include "utils.h"

static void extractQunatizationParamsFromTensors(ConvQuantizationParams* result, TensorPtr IFM, TensorPtr W, TensorPtr OFM, TensorPtr CIN)
{
    if (IFM)
    {
        result->x = IFM->getQuantizationParams();
    }
    if (W)
    {
        result->w = W->getQuantizationParams();
    }
    if (OFM)
    {
        result->out = OFM->getQuantizationParams();
    }
    if (CIN)
    {
        result->residual = CIN->getQuantizationParams();
    }
}

ConvolutionNode::ConvolutionNode(const TensorVector& inputs, const TensorVector& outputs, std::string_view name)
: BaseClass(inputs, outputs, name, Node::TYPE_CONVOLUTION, SIF_CONVOLUTION), m_tpcLowering(TPC_LOWERING_NONE)
{
    if(inputs.size() > TENSOR_CIN)
    {
        m_cin = inputs[TENSOR_CIN] != nullptr;
    }
    else
    {
        m_cin = false;
    }
}

NodePtr ConvolutionNode::createNode(const TensorVector& inputs,
                                    const TensorVector& outputs,
                                    UserParams          userParams,
                                    unsigned            userParamsSize,
                                    std::string_view    guid,
                                    std::string_view    name)
{
    ConvolutionNode* convolution_node = new ConvolutionNode(inputs, outputs, name);
    convolution_node->setGUID(guid);
    convolution_node->setParams(userParams, userParamsSize);
    return NodePtr(convolution_node);
}

bool ConvolutionNode::is3DConvolutionGuid() const
{
    return BaseClass::getGUID() == NodeFactory::convolution3DNodeTypeName;
}

NodePtr ConvolutionNode::clone() const
{
    return NodePtr(new ConvolutionNode(*this));
}

TensorSemanticType ConvolutionNode::getParamSemanticType(const TensorPtr& param) const
{
    if (param == getInput(TENSOR_WEIGHT)) return TYPE_WEIGHTS;
    return MmeNode::getParamSemanticType(param);
}

void ConvolutionNode::setTPCLowering(TpcLoweringType type)
{
    m_tpcLowering = type;
}

TpcLoweringType ConvolutionNode::getTPCLowering() const
{
    return m_tpcLowering;
}

bool ConvolutionNode::loweredByTPC() const
{
    return !(m_tpcLowering == TPC_LOWERING_NONE);
}

TensorShape ConvolutionNode::getInputShape(const TensorShape& output, uint32_t outputIdx, uint32_t inputIdx) const
{
    const unsigned int dim = output.getDim();
    TensorShape inputShape;
    inputShape.setDim(dim);
    HB_ASSERT(outputIdx == TENSOR_OFM, "dimension mismatch real:{}, expected:{}", outputIdx, TENSOR_OFM);
    if (inputIdx == TENSOR_IFM)
    {
        inputShape = getXOperandShape(output);
    }
    else if (inputIdx == TENSOR_CIN)
    {
        inputShape = output;

    }
    else
    {
        inputShape = Node::getInputShape(output, outputIdx, inputIdx);
    }

    return inputShape;
}

bool ConvolutionNode::validateNode() const
{
    TensorPtr cin = getInput(TENSOR_CIN);
    if (cin != nullptr)
    {
        if (!(cin->getDenseSizeInElements() >= getOutput(TENSOR_OFM)->getDenseSizeInElements()))
        {
            LOG_ERR(HABANA_NODE, "Invalid Cin in node {}. Cin must have at least the same number of elements as OFM, but tensor"
                    " Cin {} has {} elements and tensor OFM {} has {} elements.",
                    getNodeName(), cin->getName(), cin->getDenseSizeInElements(),
                    getOutput(TENSOR_OFM)->getName(), getOutput(TENSOR_OFM)->getDenseSizeInElements());
            return false;
        }
        if (!cin->isDenseLayout())
        {
            LOG_ERR(HABANA_NODE, "Invalid Cin in node {}. Cin must be dense but tensor {} is strided", getNodeName(), cin->getName());
            return false;
        }
    }
    return BaseClass::validateNode();
}

bool ConvolutionNode::validateNodeLayout() const
{
    if (loweredByTPC()) return true;

    SET_TEMP_LOG_CONTEXT(getNodeName());

    return BaseClass::validateNodeLayout();
}

unsigned ConvolutionNode::getKDimIndex()
{
    return inputDimNameToIndex(TENSOR_WEIGHT, 'K');
}


template<typename InputType,
         typename WeightType,
         typename OutputType,
         typename StorageFormat,
         typename IntermediateClamp>
bool ConvolutionNode::calculateConvolution()
{
    TensorPtr IFM = getInput(TENSOR_IFM);
    TensorPtr w   = getInput(TENSOR_WEIGHT);
    TensorPtr b   = getInput(TENSOR_BIAS);
    TensorPtr cin = getInput(TENSOR_CIN);
    TensorPtr OFM = getOutput(TENSOR_OFM);

    bool bias = hasBias();
    //assert(IFM->getElementType() == syn_type_single);
    //assert(w->getElementType()   == syn_type_single);
    if (bias)
    {
        //assert(b->getElementType() == syn_type_single);
    }
    //assert(OFM->getElementType() == syn_type_single);

    HB_ASSERT((IFM->getDim() > 2) && (IFM->getDim() <= SYN_MAX_TENSOR_DIM), "invalid input dimensions");
    HB_ASSERT(OFM->getDim() == IFM->getDim(), "dimensions mismatch output:{}, input:{}", OFM->getDim(), IFM->getDim());
    HB_ASSERT((w->getDim()   == 4) || (w->getDim()   == SYN_MAX_TENSOR_DIM), "invalid weight dimensions");
    if (bias)
    {
        HB_ASSERT(b->getDim() == 1, "bias must have 1 dimension");
    }

    if (m_cin)
    {
        HB_ASSERT(cin->getDim() == OFM->getDim(),
                  "dimensions mismatch cin:{}, output:{}", cin->getDim(), OFM->getDim());
        HB_ASSERT(cin->getElementType() == IFM->getElementType(), "cin element type not equal to input element type");
    }

    unsigned nIFM      = IFM->getSizeInElements(0);
    unsigned nOFM      = OFM->getSizeInElements(0);
    unsigned batch     = (IFM->getDim() > 3) ? IFM->getSizeInElements(3) : 1;

    HB_ASSERT(w->getSizeInElements(2) == m_params.kernel[CONV_KERNEL_WIDTH], "dimension size mismatch");
    HB_ASSERT(w->getSizeInElements(3) == m_params.kernel[CONV_KERNEL_HEIGHT], "dimension size mismatch");
    HB_ASSERT(w->getSizeInElements(1) == nIFM, "dimension size mismatch");
    HB_ASSERT(w->getSizeInElements(0) == nOFM, "dimension size mismatch");

    if (bias)
    {
        HB_ASSERT(b->getSizeInElements(0) == nOFM, "dimension size mismatch");
    }

    if (m_cin)
    {
        HB_ASSERT(cin->getSizeInElements(0) == nOFM, "dimension size mismatch");
    }

    unsigned wIFM = IFM->getSizeInElements(1);
    unsigned hIFM = IFM->getSizeInElements(2);
    unsigned wOFM = OFM->getSizeInElements(1);
    unsigned hOFM = OFM->getSizeInElements(2);
    unsigned padded_width  = wIFM + (m_params.padding[CONV_PAD_LEFT] + m_params.padding[CONV_PAD_RIGHT]);
    unsigned padded_height = hIFM + (m_params.padding[CONV_PAD_TOP] + m_params.padding[CONV_PAD_BOTTOM]);
    synActivationParams* activationParams = &m_params.activation;
    ConvQuantizationParams qParams;
    extractQunatizationParamsFromTensors(&qParams, IFM, w, OFM, cin);

    InputType* pIFM  = static_cast<InputType*>(IFM->map());
    WeightType* pW   = static_cast<WeightType*>(w->map());
    InputType* pCIN  = m_cin ? static_cast<InputType*>(cin->map()) :  nullptr;

    OutputType* pB   = bias ? static_cast<OutputType*>(b->map())   : nullptr;
    OutputType* pOFM = static_cast<OutputType*>(OFM->map());

    DoConvolution2D<InputType, WeightType, OutputType, StorageFormat, IntermediateClamp>(
                    pIFM, pW, pB, pCIN, pOFM, wIFM, hIFM, nIFM, wOFM, hOFM, nOFM, padded_width,
                    padded_height, m_params.padding[CONV_PAD_LEFT], m_params.padding[CONV_PAD_TOP],
                    m_params.kernel[CONV_KERNEL_WIDTH], m_params.kernel[CONV_KERNEL_HEIGHT],
                    m_params.stride[CONV_STRIDE_WIDTH], m_params.stride[CONV_STRIDE_HEIGHT],
                    m_params.dilation[CONV_DIL_HEIGHT], m_params.dilation[CONV_DIL_WIDTH], batch, activationParams, &qParams);
    return true;
}


bool ConvolutionNode::RunOnCpu()
{
    TensorPtr IFM = getInput(TENSOR_IFM);
    TensorPtr w   = getInput(TENSOR_WEIGHT);
    TensorPtr OFM = getOutput(TENSOR_OFM);

    synDataType in_type  = IFM->getElementType();
    synDataType w_type   = w->getElementType();
    synDataType out_type = OFM->getElementType();

    if ((out_type == syn_type_single) && (w_type == syn_type_single) && (in_type == syn_type_single))
    {
        return calculateConvolution<float, float, float, float, float>();
    }

    if ((out_type == syn_type_uint8) && (w_type == syn_type_uint8) && (in_type == syn_type_uint8))
    {
        return calculateConvolution<uint8_t, uint8_t, uint8_t>();
    }

    if ((out_type == syn_type_int32) && (w_type == syn_type_uint8) && (in_type == syn_type_uint8))
    {
        return calculateConvolution<uint8_t, uint8_t, int32_t>();
    }

    if ((out_type == syn_type_fixed) && (w_type == syn_type_fixed) && (in_type == syn_type_fixed))
    {
        return calculateConvolution<int8_t, int8_t, int8_t>();
    }

    if ((out_type == syn_type_int32) && (w_type == syn_type_fixed) && (in_type == syn_type_fixed))
    {
        return calculateConvolution<int8_t, int8_t, int32_t>();
    }

    if ((out_type == syn_type_int32) && (w_type == syn_type_int16) && (in_type == syn_type_int16))
    {
        return calculateConvolution<int16_t, int16_t, int32_t>();
    }

    if ((out_type == syn_type_uint16) && (w_type == syn_type_uint8) && (in_type == syn_type_uint8))
    {
        return calculateConvolution<uint8_t, uint8_t, uint16_t>();
    }

    if ((out_type == syn_type_uint16) && (w_type == syn_type_uint16) && (in_type == syn_type_uint16))
    {
        return calculateConvolution<uint16_t, uint16_t, uint16_t>();
    }

    if ((out_type == syn_type_int32) && (w_type == syn_type_uint16) && (in_type == syn_type_uint16))
    {
        return calculateConvolution<uint16_t, uint16_t, int32_t>();
    }
    return false;
}

std::map<TensorPtr, TensorVector, TensorComparator> ConvolutionNode::getReusableInputs() const
{
    std::map<TensorPtr, TensorVector, TensorComparator> ret;
    if (m_cin)
    {
        TensorPtr cin = getInput(TENSOR_CIN);
        TensorPtr ofm = getOutput(TENSOR_OFM);
        if (cin->getElementSizeInBytes() >= ofm->getElementSizeInBytes())
        {
            ret[ofm] = TensorVector();
            ret[ofm].push_back(cin);
        }
    }
    return ret;
}

bool ConvolutionNode::validateNodeForGraph(const HabanaGraph& g) const
{
    bool ret = false;
    // SimGraph may not have hal reader
    if (g.getHALReader() != nullptr)
    {
        ret = g.getHALReader()->isMmeCinSupported() || getInput(TENSOR_CIN) == nullptr;
    }
    return ret && BaseClass::validateNodeForGraph(g);
}

TensorPtr ConvolutionNode::getXOperand() const
{
    return getInput(TENSOR_IFM);
}

TensorPtr ConvolutionNode::getYOperand() const
{
    return getOutput(TENSOR_OFM);
}

TensorPtr ConvolutionNode::getWOperand() const
{
    return getInput(TENSOR_WEIGHT);
}

bool ConvolutionNode::isSpatialSlicingSupported(unsigned dim) const
{
    const TensorPtr& ifm = getInput(TENSOR_IFM);
    bool isDynamicIfm = ifm->isDynamicDim(DIM_W) || ifm->isDynamicDim(DIM_H) || ifm->isDynamicDim(DIM_D_FOR_5D_TENSOR);
    bool isDynamicSupported = isDynamicIfm ? GCFG_SRAM_SLICER_DYNAMIC_4D_CONV_SPATIAL_SLICE_ENABLED.value() : true;
    bool convSupported      = GCFG_SRAM_SLICER_4D_CONV_SPATIAL_SLICE_ENABLED.value() && isDynamicSupported;

    return (convSupported && ConvBaseNode::isSpatialSlicingSupported(dim));
}

// Returns the minimal ROI size according to the convolution parameters.
// This function doesn't consider the tensor size, and its output may have to be clipped.
// TODO handle paddingType here <===== PADDING_TYPE
TSize ConvolutionNode::getMinSpatialDimOutputROI(unsigned dim) const
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

    // For node types that X is the input (not dedx) - the padding and the overlap limits refer to the min size
    // of the input operand. Project them from conv input size to conv output size, as this function calculates
    // output min size. To make sure the second slice doesn't start in negative offset - the first slice size
    // must be at least padding + overlap, thus the overlap projection includes the padding param.
    // The projection shrinks the limit - it might return negative. In this case this limit is smaller than
    // the kernel, and can be set to 0.
    overlapLimit = std::max(convOutputDimSize(overlapLimit, kernel, stride, paddingBefore, dilation), 0);
    // Set padding to 0 for the padding projection, to get the number of output lines calculated from it alone.
    // This limit is redundant given the overlap limit took it into account.
    paddingLimit = std::max(convOutputDimSize(paddingLimit, kernel, stride, 0, dilation), 0);
    // Add 1 to the limit so the min size doesn't turn out to be 0 after the padding / overlap are reduced
    minSize = std::max(overlapLimit, paddingLimit) + 1;

    HB_ASSERT(minSize > 0, "Output ROI min size must be > 0");
    return minSize;
}
