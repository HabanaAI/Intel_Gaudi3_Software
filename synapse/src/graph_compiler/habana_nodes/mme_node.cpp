#include "mme_node.h"

#include "data_type_utils.h"
#include "graph_traits.h"
#include "habana_graph.h"
#include "mme/mme_brain_ifc.h"
#include "compilation_hal_reader.h"
#include "node.h"

#include "types_exception.h"
#include "utils.h"

MmeNode::MmeNode(const TensorVector& inputs,
                 const TensorVector& outputs,
                 std::string_view    name,
                 eNodeType           type,
                 ShapeFuncID         sifId)
: Node(inputs, outputs, name, type, sifId)
{
    // TODO - SW-23318 Check int16ltd for cin
    for (const TensorPtr& input : inputs)
    {
        if (input == nullptr) continue;
        input->setInt16Limited(true);
        m_mmeExpBias.fp8BiasIn.push_back(input->getExpBias());
    }
    m_mmeExpBias.fp8BiasOut = outputs[0]->getExpBias();

    // If the Mme node is created when the chip type is already known, init the MmeBrainIfc
    if (CompilationHalReader::isHalReaderSet())
    {
        initMmeBrainIfc(CompilationHalReader::getHalReader()->getDeviceType());
    }
}

MmeNode::MmeNode(const MmeNode& other) : Node(other), m_mmeExpBias(other.m_mmeExpBias)
{
    if (CompilationHalReader::isHalReaderSet())
    {
        initMmeBrainIfc(CompilationHalReader::getHalReader()->getDeviceType());
    }
}

void MmeNode::initMmeBrainIfc(synDeviceType deviceType)
{
    m_mmeBrainIfc = std::make_shared<MmeBrainIfc>(*this, deviceType);
}

bool MmeNode::hasBias() const
{
    auto biasTensor = getInput(TENSOR_BIAS);
    return (biasTensor != nullptr && !biasTensor->isShapeTensor() && m_type != Node::TYPE_MASKED_BATCH_GEMM);
}

bool MmeNode::hasCin() const
{
    return (getInput(TENSOR_CIN) != nullptr && m_type != Node::TYPE_MASKED_BATCH_GEMM);
}

void MmeNode::addMMETensor(const TensorPtr& tensor, unsigned tensorIndex)
{
    HB_ASSERT(tensor != nullptr, "Cannot set null tensor in MME node");
    tensor->setInt16Limited(true);

    if (m_inputs.size() > tensorIndex)
    {
        replaceInput(tensorIndex, tensor);
        return;
    }
    else
    {
        LayoutVector nodesLayout = getInputLayouts();
        for (int i = m_inputs.size(); i < tensorIndex; i++)
        {
            TensorPtr empty = nullptr;
            nodesLayout.emplace_back(gc::Layout(""));
            m_inputs.push_back(empty);
        }
        m_inputs.push_back(tensor);
        nodesLayout.emplace_back(gc::Layout(""));
        setInputLayouts(nodesLayout);
        return;
    }
}

TensorSemanticType MmeNode::getParamSemanticType(const TensorPtr& param) const
{
    if (hasBias() && param == getInput(TENSOR_BIAS)) return TYPE_BIAS;
    return Node::getParamSemanticType(param);
}

bool MmeNode::validateNode() const
{
    if (m_inputs.size() < 2 || m_inputs.size() > TENSOR_INPUT_MAX || m_outputs.size() != 1)
    {
        LOG_ERR(HABANA_NODE, "Invalid number of operands (expecting 2-4 inputs and 1 output)");
        return false;
    }

    if (getInput(TENSOR_IFM) == nullptr || getInput(TENSOR_WEIGHT) == nullptr || getOutput(TENSOR_OFM) == nullptr)
    {
        LOG_ERR(HABANA_NODE,
                "{}: node is missing key tensors: must have 1 output and at least 2 inputs",
                getNodeName());
        return false;
    }

    return Node::validateNode();
}

bool MmeNode::validateNodeLayout() const
{
    if (hasCin())
    {
        TensorPtr OFM = getOutput(TENSOR_OFM);
        TensorPtr CIN = getInput(TENSOR_CIN);

        // CIN and OFM should have the same geometry
        // Note: they may be different after the large images algorithm or packing, but this function is called
        // before
        if (!OFM->compareGeometry(*CIN))
        {
            LOG_ERR(HABANA_NODE,
                    "MME node {} CIN and OFM size mismatch. CIN: {}, OFM: {}",
                    getNodeName(),
                    CIN->getDimSizesStr(),
                    OFM->getDimSizesStr());
            return false;
        }
    }

    return true;
}

bool MmeNode::validateNodeForGraph(const HabanaGraph& g) const
{
    if (!g.getInferenceMode())
    {
        if (getInput(TENSOR_IFM)->getElementType() != getInput(TENSOR_WEIGHT)->getElementType())
        {
            LOG_ERR(HABANA_NODE,
                    "Node {} must have both inputs with same element type for training graph",
                    getNodeName());
            return false;
        }
    }

    if (g.isFP32LimitedDevice())
    {
        auto failCompilation = std::any_of(getInputs().begin(), getInputs().end(), [](const TensorPtr& input) {
            return (input != nullptr &&
                    (input->getElementType() == syn_type_float || input->getElementType() == syn_type_tf32 ||
                     input->getElementType() == syn_type_hb_float));
        });

        if (failCompilation)
        {
            LOG_ERR(HABANA_NODE, "FP32 operations are not supported on this device. Node Name {}", m_name);
            throw DeviceLimitationFP32Exception(m_name);
        }
    }

    return true;
}

NodeROI MmeNode::generateRoi() const
{
    NodeROI fullRoi;
    HB_ASSERT(m_graphTraits != nullptr, "node hal reader is null");

    if (GCFG_ENABLE_MME_INDEX_SPACE.value())
    {
        const auto& ap = getNodeAccessPattern();
        if (ap == nullptr) return fullRoi;
        const auto& nodeResolution = ap->getNodeResolution();
        HB_ASSERT(nodeResolution.size() <= ARRAY_SIZE(fullRoi.size), "access pattern and ROI sizes dont match");

        for (int dim = 0; dim < nodeResolution.size(); dim++)
        {
            fullRoi.size[dim] = nodeResolution[dim];
        }
        std::fill(fullRoi.size + nodeResolution.size(), fullRoi.size + ARRAY_SIZE(fullRoi.size), 1);
        LOG_TRACE(
            GC,
            "ROI generated for node '{}': Size in elements : [{}]x[{}] = [{}], Size in index space [{}], Offset [{}]",
            getNodeName(),
            toString(getInput(0)->getAllSizesInElements(), ','),
            toString(getInput(1)->getAllSizesInElements(), ','),
            toString(getOutput(0)->getAllSizesInElements(), ','),
            toString(std::begin(fullRoi.size), std::end(fullRoi.size), ','),
            toString(std::begin(fullRoi.baseOffset), std::end(fullRoi.baseOffset), ','));
    }
    else
    {
        getOutput(TENSOR_OFM)->getAllSizesInElements(fullRoi.size, ARRAY_SIZE(fullRoi.size));
        unsigned int fullSpatialSize = multiplyElements(fullRoi.size + 1, fullRoi.size + ARRAY_SIZE(fullRoi.size));
        unsigned int vectorSize      = safeSizeInBitsToElements(m_graphTraits->getHalReader()->getMmeVectorSize(),
                                                           getInput(TENSOR_WEIGHT)->getElementSizeInBits());

        fullRoi.numIterations     = div_round_up(fullSpatialSize, vectorSize);
        fullRoi.spatialSizeMinus1 = (fullSpatialSize - 1) % vectorSize;
        fullRoi.vectorSize        = vectorSize;
    }

    return fullRoi;
}

/*
 * Get the required weights (Tensor B) data type.
 * In case current weights data type isn't supported by HW, the default weights data type will be returned.
 */
synDataType MmeNode::getRequiredWeightsDataType() const
{
    synDataType weightsDataType = getInput(TENSOR_WEIGHT)->getElementType();
    if (!m_graphTraits->getHalReader()->isSupportedMmeInputDataType(weightsDataType))
    {
        LOG_DEBUG(HABANA_NODE,
                  "{}, Tensor B data type {} isn't supported in MME. getting default data type.",
                  HLLOG_FUNC,
                  getStringFromSynDataType(weightsDataType));
        weightsDataType = getDefaultRequiredDataType();
    }
    HB_ASSERT(weightsDataType != syn_type_na, "Unexpected weights DataType");
    return weightsDataType;
}

void MmeNode::printMmeParams(const synConvolution3DParamsV2& mmeParams)
{
    LOG_DEBUG(GRAPH_DATA, "  Node params: {}", MmeNode::synConvolution3DParamsToString(mmeParams));
}

std::string_view MmeNode::getEngineTypeStr() const
{
    return "MME";
}

synDataType MmeNode::getRequiredInputType(uint32_t tensorIdx) const
{
    return getRequiredTensorType(tensorIdx, true);
}

bool MmeNode::isOutputTensorPartialWrites(unsigned tensorOutputIndex) const
{
    // TODO SW-146022 - expose MME partial write per strategy
    return false;
}

bool MmeNode::isTensorDataTypeSupported(uint32_t tensorIdx, synDataType tensorDataType, bool isInput) const
{
    auto hal = m_graphTraits->getHalReader();
    if (!isInput && tensorIdx == TENSOR_OFM)
    {
        return hal->isSupportedMmeDataType(tensorDataType);
    }
    else if (tensorIdx == TENSOR_WEIGHT || (isInput && tensorIdx == TENSOR_IFM))
    {
        return hal->isSupportedMmeInputDataType(tensorDataType);
    }
    else if (tensorIdx == TENSOR_BIAS)
    {
        return (tensorDataType == syn_type_int32 || tensorDataType == syn_type_float) &&
               hal->isSupportedMmeInputDataType(tensorDataType);
    }
    else
    {
        HB_ASSERT(false, "unexpected tensor index");
    }
    return false;
}

bool MmeNode::matchTensorDataTypeToWeights(uint32_t     tensorIdx,
                                           synDataType& tensorDataType,
                                           synDataType  weightsDataTypeAfterCast,
                                           bool         isInput) const
{
    auto hal = m_graphTraits->getHalReader();
    if (!isInput && tensorIdx == TENSOR_OFM)
    {
        bool isTensorFloat  = isTypeFloat(tensorDataType);
        bool isWeightsFloat = isTypeFloat(weightsDataTypeAfterCast);
        if (isTensorFloat != isWeightsFloat)
        {
            tensorDataType = getDefaultRequiredDataType(isWeightsFloat);
            HB_ASSERT(isTypeFloat(tensorDataType) == isWeightsFloat,
                      "The tensor and the weights tensor must have the same basic type (int/float)");
            return true;
        }
        return false;
    }
    else if (tensorIdx == TENSOR_WEIGHT || (isInput && tensorIdx == TENSOR_IFM))
    {
        if (tensorDataType != weightsDataTypeAfterCast)
        {
            tensorDataType = weightsDataTypeAfterCast;
            return true;
        }
        return false;
    }
    else
    {
        HB_ASSERT(false, "unexpected tensor index");
    }
    return false;
}

/*
 * Get the required tensor data type according to constraints of HW and weights (B) tensor.
 */
synDataType MmeNode::getRequiredTensorType(uint32_t tensorIdx, bool isInput) const
{
    auto        hal = m_graphTraits->getHalReader();
    synDataType tensorDataType =
        isInput ? getInput(tensorIdx)->getElementType() : m_outputs[tensorIdx]->getElementType();
    const std::string tensorName = isInput ? getInput(tensorIdx)->getName() : m_outputs[tensorIdx]->getName();
    LOG_DEBUG(HABANA_NODE,
              "{}: Get required data type for {} tensor {} of node {}",
              HLLOG_FUNC,
              isInput ? "input" : "output",
              tensorName,
              getNodeName());

    synDataType weightsDataTypeAfterCast = getRequiredWeightsDataType();

    /*
     * First cover 3 cases that require special handling
     *  1. tensor in training graph
     *  2. bias tensor
     *  3. Live B scenario (weights \ B isn't static)
     */
    if (m_graphTraits->trainingGraph())
    {  // Training - no cast constraints
        if (hal->isSupportedMmeInputDataType(tensorDataType))
        {
            return tensorDataType;
        }
        else
        {
            LOG_WARN(HABANA_NODE,
                     "{}: Tensor {} data type {} is not supported in MME. returning syn_type_single",
                     HLLOG_FUNC,
                     tensorName,
                     getStringFromSynDataType(tensorDataType));
            return syn_type_single;
        }
    }
    // else - Inference
    if (tensorIdx == TENSOR_BIAS)
    {
        return tensorDataType;
    }
    // handle live B
    if (!getInput(TENSOR_WEIGHT)->isStaticParam() &&
        (tensorIdx == TENSOR_WEIGHT || (isInput && tensorIdx == TENSOR_IFM)))
    {
        // Inference "live B" scenario - B isn't static
        LOG_DEBUG(HABANA_NODE, "{}, Tensor B {} of MME node {} isn't static", HLLOG_FUNC, tensorName, getNodeName());
        return weightsDataTypeAfterCast;
    }

    synDataType requiredDataType = tensorDataType;
    // if tensor data type doesn't match HW constraints - should return default type
    if (!isTensorDataTypeSupported(tensorIdx, tensorDataType, isInput))
    {
        requiredDataType = getDefaultRequiredDataType();
        LOG_INFO(HABANA_NODE,
                 "{}: Required data type for tensor {} doesn't match HW. "
                 "Will return the default tensor data type {}",
                 HLLOG_FUNC,
                 tensorName,
                 getStringFromSynDataType(requiredDataType));
    }

    // make sure the tensor data type matches the weights tensor constraints
    if (matchTensorDataTypeToWeights(tensorIdx, requiredDataType, weightsDataTypeAfterCast, isInput))
    {
        LOG_INFO(HABANA_NODE,
                 "{}: Required data type for tensor {} doesn't match weights tensor data type constraints. "
                 "Will return {}",
                 HLLOG_FUNC,
                 tensorName,
                 getStringFromSynDataType(requiredDataType));
    }

    LOG_DEBUG(HABANA_NODE,
              "{}: Required data type for tensor {} is {}",
              HLLOG_FUNC,
              tensorName,
              getStringFromSynDataType(requiredDataType));
    return requiredDataType;
}

synDataType MmeNode::getRequiredOutputType(uint32_t tensorIdx) const
{
    return getRequiredTensorType(tensorIdx, false);
}

synDataType MmeNode::getDefaultRequiredDataType(bool isFloat) const
{
    synDataType defaultDataType = m_graphTraits->inferenceGraph() ? syn_type_bf16 : syn_type_single;
    LOG_DEBUG(HABANA_NODE,
              "{}, default data type for node {} - {}",
              HLLOG_FUNC,
              getNodeName(),
              getStringFromSynDataType(defaultDataType));
    return defaultDataType;
}

synConvolution3DParamsV2 MmeNode::convert2DconvTo3DconvStruct(const synConvolutionParamsV2& userConvParam)
{
    synConvolution3DParamsV2 cov3DParams = userConvParam;  // up-convert
    return cov3DParams;
}

synConvolutionParamsV2 MmeNode::convert3DconvTo2DconvStruct(const synConvolution3DParamsV2& cov3DParams)
{
    HB_ASSERT(cov3DParams.kernel[CONV_KERNEL_DEPTH] * cov3DParams.stride[CONV_STRIDE_DEPTH] *
                      cov3DParams.dilation[CONV_DIL_DEPTH] ==
                  1,
              "Cannot convert 3D to 2D params");
    HB_ASSERT(cov3DParams.padding[CONV_PAD_FRONT] == 0 && cov3DParams.padding[CONV_PAD_BACK] == 0,
              "Cannot convert 3D to 2D params");

    synConvolutionParamsV2 userConvParam;

    userConvParam.kW = cov3DParams.kernel[CONV_KERNEL_WIDTH];
    userConvParam.kH = cov3DParams.kernel[CONV_KERNEL_HEIGHT];

    userConvParam.dW = cov3DParams.stride[CONV_STRIDE_WIDTH];
    userConvParam.dH = cov3DParams.stride[CONV_STRIDE_HEIGHT];

    userConvParam.padL = cov3DParams.padding[CONV_PAD_LEFT];
    userConvParam.padR = cov3DParams.padding[CONV_PAD_RIGHT];
    userConvParam.padT = cov3DParams.padding[CONV_PAD_TOP];
    userConvParam.padB = cov3DParams.padding[CONV_PAD_BOTTOM];

    userConvParam.dilW = cov3DParams.dilation[CONV_DIL_WIDTH];
    userConvParam.dilH = cov3DParams.dilation[CONV_DIL_HEIGHT];

    userConvParam.activation = cov3DParams.activation;
    userConvParam.nGroups    = cov3DParams.nGroups;

    userConvParam.paddingType = cov3DParams.paddingType;

    return userConvParam;
}

unsigned MmeNode::getKDimIndex()
{
    TensorPtr weightTensor = getInput(TENSOR_WEIGHT);
    return weightTensor->getDim() - 1;
}

bool MmeNode::areConvParamsGEMMConvertible(const synConvolution3DParamsV2& params)
{
    return params.kernel[CONV_KERNEL_WIDTH] == 1 && params.kernel[CONV_KERNEL_HEIGHT] == 1 &&
           params.kernel[CONV_KERNEL_DEPTH] == 1 && params.stride[CONV_STRIDE_WIDTH] == 1 &&
           params.stride[CONV_STRIDE_HEIGHT] == 1 && params.stride[CONV_STRIDE_DEPTH] == 1 &&
           params.padding[CONV_PAD_RIGHT] == 0 && params.padding[CONV_PAD_TOP] == 0 &&
           params.padding[CONV_PAD_BOTTOM] == 0 && params.padding[CONV_PAD_FRONT] == 0 &&
           params.padding[CONV_PAD_LEFT] == 0 && params.padding[CONV_PAD_BACK] == 0 &&
           params.paddingType == PADDING_EXPLICIT;
}

std::string MmeNode::synConvolutionParamsToString(const synConvolutionParamsV2& params)
{
    return fmt::format("kW={}, kH={}, dW={}, dH={}, padLeft={}, padRight={}, padTop={}, padBottom={}, "
                       "dilW={}, dilH={}, reluEnable={}, resAfterPWL={}, numChannels={}, nGroups={}, paddingType={}",
                       params.kW,
                       params.kH,
                       params.dW,
                       params.dH,
                       params.padL,
                       params.padR,
                       params.padT,
                       params.padB,
                       params.dilW,
                       params.dilH,
                       params.activation.reluEnable,
                       params.activation.resAfterPWL,
                       params.activation.numChannels,
                       params.nGroups,
                       params.paddingType == PADDING_SAME ? "SAME" : "EXPLICIT");
}

std::string MmeNode::synConvolution3DParamsToString(const synConvolution3DParamsV2& params)
{
    return fmt::format("kW={}, kH={}, kD={}, "
                       "dW={}, dH={}, dD={}, "
                       "padLeft={}, padRight={}, padTop={}, "
                       "padBottom={}, padFront={}, padBack={}, "
                       "dilW={}, dilH={}, dilD={}, "
                       "reluEnable={}, resAfterPWL={}, numChannels={}, "
                       "nGroups={} paddingType={}, ",
                       params.kernel[CONV_KERNEL_WIDTH],
                       params.kernel[CONV_KERNEL_HEIGHT],
                       params.kernel[CONV_KERNEL_DEPTH],
                       params.stride[CONV_STRIDE_WIDTH],
                       params.stride[CONV_STRIDE_HEIGHT],
                       params.stride[CONV_STRIDE_DEPTH],
                       params.padding[CONV_PAD_LEFT],
                       params.padding[CONV_PAD_RIGHT],
                       params.padding[CONV_PAD_TOP],
                       params.padding[CONV_PAD_BOTTOM],
                       params.padding[CONV_PAD_FRONT],
                       params.padding[CONV_PAD_BACK],
                       params.dilation[CONV_DIL_WIDTH],
                       params.dilation[CONV_DIL_HEIGHT],
                       params.dilation[CONV_DIL_DEPTH],
                       params.activation.reluEnable,
                       params.activation.resAfterPWL,
                       params.activation.numChannels,
                       params.nGroups,
                       params.paddingType == PADDING_SAME ? "SAME" : "EXPLICIT");
}

bool MmeNode::isDmaOperation(const NodePtr& node)
{
    MmeNode* mmeNode = dynamic_cast<MmeNode*>(node.get());
    if (mmeNode != nullptr)
    {
        return mmeNode->isDmaOperation();
    }
    return false;
}

bool MmeNode::isDmaOperation() const
{
    Node::eNodeType opType = getNodeType();
    return (opType == Node::TYPE_MEMCOPY || (opType == Node::TYPE_INTERNAL_TRANSPOSE && !isTransposeViaGemm()));
}

bool MmeNode::isCdIndexSpaceDim(unsigned indexSpaceDim) const
{
    return getMmeBrainIfc()->isCdDim(indexSpaceDim);
}