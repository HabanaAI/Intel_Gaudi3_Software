#include "habana_graph.h"
#include "data_type_utils.h"
#include "node_factory.h"
#include "graph_editor.h"
#include "scoped_configuration_change.h"

typedef std::vector<double> ScalesVec;
struct ScaleInfo
{
    TensorPtr tensorPtr = nullptr;
    NodePtr   producer  = nullptr;
    ScalesVec scales;
};
typedef std::vector<ScaleInfo> ScaleInfoVec;

template<class T>
void setBuffer(TensorPtr tensor, ScalesVec scalesVec)
{
    std::vector<T> castedVector;
    const TSize    elementsNumber = scalesVec.size();
    std::size_t    buffSize       = sizeof(T) * elementsNumber;

    for (const auto& scale : scalesVec)
    {
        T castedVal((float)scale);
        castedVector.push_back(castedVal);
    }
    tensor->setTensorBuffer(castedVector.data(), buffSize, tensor->getElementType());
}

bool initScalesData(TensorPtr scalingTensor, ScalesVec scalesVec)
{
    switch (scalingTensor->getElementType())
    {
        case syn_type_bf16:
        {
            setBuffer<bf16_t>(scalingTensor, scalesVec);
            return true;;
        }
        case syn_type_fp16:
        {
            setBuffer<fp16_t>(scalingTensor, scalesVec);
            return true;;
        }
        case syn_type_float:
        {
            setBuffer<float>(scalingTensor, scalesVec);
            return true;;
        }
        default:
        {
            LOG_WARN(QUANT, "Unsupported buffer data type {} for tensor {}", scalingTensor->getElementType(), scalingTensor->getName());
            return false;
        }
    }
}

NodePtr createScaleNode(TensorPtr             inputTensor,
                        TensorPtr             scaleTensor,
                        TensorPtr             outputTensor,
                        bool                  isInput,
                        const std::string&    name,
                        tpc_lib_api::DeviceId deviceId)
{
    // The 'div' node serves scaling and 'mult' node serves descaling.
    std::string scaleKernelName = isInput ? "div_fwd_" : "mult_fwd_" ;
    scaleKernelName += getDtypeSuffixFromSynDataType(inputTensor->getElementType());

    // If device id is passed, validate the kernel exists
    if (deviceId != tpc_lib_api::DEVICE_ID_MAX && !KernelDB::instance().isKernelExist(scaleKernelName, deviceId))
    {
        LOG_ERR(QUANT, "{}: kernel {} does not exist", __FUNCTION__, scaleKernelName);
        return nullptr;
    }

    LOG_DEBUG(QUANT, "Creating scale node with kernel: {}", scaleKernelName);
    NodePtr node = NodeFactory::createNode({inputTensor, scaleTensor}, {outputTensor}, nullptr, scaleKernelName, name);
    if (node == nullptr)
    {
        LOG_ERR(QUANT, "{}: failed to create scale node", __FUNCTION__);
        return nullptr;
    }

    NodeAnnotation& nodeAnnotation = node->getNodeAnnotation();
    nodeAnnotation.insertedNode    = true;
    return node;
}

TensorPtr createScaleTensor(TensorPtr scaledTensor, const std::string& name, const ScalesVec& scales)
{
    std::stringstream ss;
    for (auto i = 0; i < scales.size(); ++i)
    {
        ss << fmt::format("{}{}", scales[i], i != scales.size() - 1 ? ", " : "");
    }
    LOG_DEBUG(QUANT, "{}: Creating a scale tensor with scales: {}", __FUNCTION__, ss.str());

    // Broadcast is supported by the scaling kernels (div / mult)
    const TSize size[] = {scales.size()};
    TensorPtr tensor = std::make_shared<Tensor>(1U, size, scaledTensor->getElementType());
    tensor->setName(name);
    tensor->setAsDataTypeMatchData();
	tensor->setAsStaticParam();
    tensor->setMemorySectionID(MEMORY_ID_RESERVED_FOR_PROGRAM_DATA);
    HB_ASSERT(initScalesData(tensor, scales), "Failed creating a scale tensor {}", tensor->getName());

    return tensor;
}

TensorPtr createTensor(TensorPtr sourceTensor, const std::string& name)
{
    LOG_DEBUG(QUANT, "{}: Creating tensor with the same shape and dtype as: {}", __FUNCTION__, sourceTensor->getName());
    TensorPtr tensor = sourceTensor->clone(false, false);
    tensor->setName(name);
    return tensor;
}

bool injectInputsScale(HabanaGraph& g, const ScaleInfoVec& infoVec, ScalesVec& scalesProducts)
{
    // Remove after SW-151898 or SW-156757 solved
    ScopedConfigurationChange disableGCOpValidation("ENABLE_GC_NODES_VALIDATION_BY_OPS_DB", "false");
    for (auto& info : infoVec)
    {
        std::string nodeName          = info.producer->getNodeName();
        ScalesVec   scales            = info.scales;
        TensorPtr   originInputTensor = info.tensorPtr;
        TensorPtr   scaleTensor       = createScaleTensor(originInputTensor, nodeName + "_scale", scales);
        TensorPtr   scaledTensor      = createTensor(originInputTensor, nodeName + "_scaled");
        NodePtr     scaleNode         = createScaleNode(originInputTensor,
                                            scaleTensor,
                                            scaledTensor,
                                            true,
                                            nodeName + "_scale",
                                            g.getDeviceId());  // Creating a div node for scaling

        if (scaleNode == nullptr || !GraphEditor::addNode(g, scaleNode))
        {
            LOG_ERR(QUANT, "{}: failed to inject scale for {} where needed", __FUNCTION__, nodeName);
            return false;
        }
        // replacing the cast node input
        GraphEditor::replaceInput(g, info.producer, TENSOR_IFM, scaledTensor);

        if (scales.size() == 1)
        {
            double scaleFactor = scales[0];
            std::transform(scalesProducts.begin(),
                           scalesProducts.end(),
                           scalesProducts.begin(),
                           [&scaleFactor](const double& val) { return val * scaleFactor; });
        }
        else if (scales.size() == scalesProducts.size())
        {
            std::transform(scalesProducts.begin(),
                           scalesProducts.end(),
                           scales.begin(),
                           scalesProducts.begin(),
                           std::multiplies<double>());
        }
        else
        {
            HB_ASSERT(false, "Invalid scales vector sizes");
        }
    }
    return true;
}

bool injectOutputScale(HabanaGraph& g, NodePtr node, ScalesVec& descaleValues)
{
    // Remove after SW-151898 or SW-156757 solved
    ScopedConfigurationChange disableGCOpValidation("ENABLE_GC_NODES_VALIDATION_BY_OPS_DB", "false");
    const std::string&  mmeNodeName = node->getNodeName();
    const TensorVector& outputs     = node->getOutputs();
    for (unsigned int index = 0; index < outputs.size(); ++index)
    {
        TensorPtr origOutputTensor = outputs[index];
        TensorPtr scaledOutput     = createTensor(origOutputTensor, mmeNodeName + "_output_scaled");
        TensorPtr descaleTensor    = createScaleTensor(origOutputTensor, mmeNodeName + "_output_descale", descaleValues);
        NodePtr   deScaleNode      = createScaleNode(scaledOutput,
                                              descaleTensor,
                                              origOutputTensor,
                                              false,
                                              mmeNodeName + "_descale_" + std::to_string(index),
                                              g.getDeviceId());  // Creating a mult node for descaling
        HB_ASSERT(deScaleNode != nullptr, "failed to inject descale node for {}", mmeNodeName);
        GraphEditor::replaceOutput(g, node, index, scaledOutput);
        if (!GraphEditor::addNode(g, deScaleNode))
        {
            LOG_ERR(QUANT, "{}: failed to inject de-scale for {} where needed", __FUNCTION__, node->getNodeName());
            return false;
        }
    }
    return true;
}

bool injectScaleNodes(HabanaGraph& g, NodePtr node, ScaleInfoVec& infoVec)
{
    LOG_DEBUG(QUANT, "{}: Inject scaling for node {} collected tensors", __FUNCTION__, node->getNodeName());
    // Store the product of all scaling values. The outcome represents 'descaleValue'.
    size_t maxScalesSize = 0;
    for (const auto& info : infoVec)
    {
        maxScalesSize = std::max(maxScalesSize, info.scales.size());
    }
    ScalesVec scalesProducts = ScalesVec(maxScalesSize, 1);

    LOG_DEBUG(QUANT, "{}: Inject scaling for node {} inputs", __FUNCTION__, node->getNodeName());
    if (!injectInputsScale(g, infoVec, scalesProducts))
    {
        return false;
    }

    LOG_DEBUG(QUANT, "{}: Inject descaling for node {} outputs", __FUNCTION__, node->getNodeName());
    if (!injectOutputScale(g, node, scalesProducts))
    {
        return false;
    }
    return true;
}

bool collectAndInjectScaleNodes(HabanaGraph& g, NodePtr node)
{
    LOG_DEBUG(QUANT, "{}: Collect and inject scales for node {} where scaling is needed", __FUNCTION__, node->getNodeName());
    const TensorVector& tensors = node->getInputs();
    ScaleInfoVec        scaleInfoVec;
    for (const TensorPtr& tensor : tensors)
    {
        // Skip tensor if it's not of 8-bit float type
        if (!is8BitFloat(tensor->getElementType())) continue;

        NodePtr producer = g.getTensorProducer(tensor);

        if (producer == nullptr || !producer->isCast()) continue;

        TensorPtr tensorToScale = producer->getInput(TENSOR_IFM);
        HB_ASSERT(tensorToScale != nullptr, "tensorToScale is nullptr");

        // The pass meant to inject scale for float tensors
        if (!isTypeFloat(tensorToScale->getElementType())) continue;

        ScalesVec scales = tensor->getQuantizationParams().getScaleVector();

        if (scales.size() > 1)
        {
            HB_ASSERT(tensor->isPerChannelQuant(),
                      "tensor with multiple scales in its quant info must be marked as per channel");
        }

        if (std::all_of(scales.begin(), scales.end(), [](const double& value) { return value == 1; })) continue;

        ScaleInfo info;
        info.scales    = scales;
        info.tensorPtr = tensorToScale;
        info.producer  = producer;
        scaleInfoVec.emplace_back(info);

        LOG_DEBUG(QUANT,
                  "{}: input tensor {} of cast node {} needs scaling",
                  __FUNCTION__,
                  tensorToScale->getName(),
                  producer->getNodeName());

        LOG_DEBUG(QUANT,
                  "{}: set default scale (1.0) for {} tensor",
                  __FUNCTION__,
                  tensor->getName());
        tensor->setScale(1.0);
    }

    if (scaleInfoVec.size() > 0 && !injectScaleNodes(g, node, scaleInfoVec))
    {
        return false;
    }
    return true;
}

bool injectScaleForMMENodes(HabanaGraph& g)
{
    if (!isInferenceQuantization(g))
    {
        LOG_DEBUG(QUANT,
                  "scale for mme nodes pass is enabled in synapse only for Inference Mode. "
                  "Skip {} Pass",
                  __FUNCTION__);
        return true;
    }

    const NodeVector& nodes = g.getExeSortedNodes();
    for (const NodePtr& node : nodes)
    {
        if (g.runsOnMME(node))
        {
            LOG_DEBUG(QUANT,
                      "{}: node {} I/O tensors are candidates for scale injection",
                      __FUNCTION__,
                      node->getNodeName());
            if (!collectAndInjectScaleNodes(g, node))
            {
                LOG_ERR(QUANT, "{} pass failed", __FUNCTION__);
                return false;
            }
        }
    }

    return true;
}