#include "node.h"

#include "access_pattern.h"
#include "data_type_utils.h"
#include "defs.h"
#include "habana_global_conf.h"
#include "layout.h"
#include "log_manager.h"
#include "node_annotation.h"
#include "passes/quantization_utils.h"
#include "quantizer_factory.h"
#include "quantizer.h"

#include "tensor_shape.h"
#include "tensor_view_node.h"
#include "types_exception.h"
#include "utils.h"

#include <algorithm>
#include <atomic>
#include <iterator>
#include <list>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <unordered_set>

using namespace gc;

std::atomic<synNodeId> Node::NODE_ID {0};
const TensorPtr Node::NULL_TENSOR;

Node::Node(const TensorVector& inputs,
           const TensorVector& outputs,
           std::string_view    name,
           Node::eNodeType     type,
           ShapeFuncID         sifId,
           bool                createNodeIoManager)
: m_inputs(inputs),
  m_outputs(outputs),
  m_inputLayouts(inputs.size()),
  m_outputLayouts(outputs.size()),
  m_io(createNodeIoManager ? std::make_unique<NodeIOManager>(this) : nullptr),
  m_type(type),
  m_id(++NODE_ID),
  m_annotation(inputs.size()),
  m_name(!name.empty() ? name : fmt::format("{}{}", Node::getNodeTypeStr(), m_id)),
  m_shapeNode(*this),
  m_physicalRois(nullptr),
  m_logicalRois(nullptr),
  // TODO fix the type of m_shapeInferenceFunctionID?
  m_shapeInferenceFunctionID(sifId == SHAPE_FUNC_MAX_ID ? INVALID_SHAPE_FUNC_ID : sifId),
  m_quantizer(QuantizerFactory::getDefaultNodeQuantizer()),
  m_precision(syn_type_na),
  m_deterministic(false),
  m_roundingMode(synRoundingMode::synRoundToNearest)
{
    m_originNodes.insert(m_id);
    m_parentId = m_id;  // the initial parent ID is the ID of the node itself
}

Node::Node(const Node& other)
: m_inputs(other.m_inputs),
  m_outputs(other.m_outputs),
  m_inputLayouts(other.m_inputLayouts),
  m_outputLayouts(other.m_outputLayouts),
  m_io(other.hasNodeIOManagerSpecialization() ? nullptr : std::make_unique<NodeIOManager>(this)),
  m_type(other.m_type),
  m_id(++NODE_ID),
  m_parentId(other.m_parentId),
  m_annotation(other.m_annotation),
  m_paddingValues(other.m_paddingValues),
  m_name(other.m_name),
  m_shapeNode(other.m_shapeNode, *this),
  m_physicalRois(other.m_physicalRois),
  m_logicalRois(other.m_logicalRois),
  m_GUID(other.m_GUID),
  m_shapeInferenceFunctionID(other.m_shapeInferenceFunctionID),
  m_quantizer(other.m_quantizer),
  m_precision(other.m_precision),
  m_paramsRawData(other.m_paramsRawData),
  m_deterministic(other.m_deterministic),
  m_roundingMode(other.m_roundingMode),
  m_originNodes(other.m_originNodes)

{
}

Node& Node::operator=(const Node& other)
{
    if (this != &other)
    {
        m_inputs                   = other.m_inputs;
        m_outputs                  = other.m_outputs;
        m_inputLayouts             = other.m_inputLayouts;
        m_outputLayouts            = other.m_outputLayouts;
        m_type                     = other.m_type;
        m_annotation               = other.m_annotation;
        m_paddingValues            = other.m_paddingValues;
        m_name                     = other.m_name;
        m_GUID                     = other.m_GUID;
        m_shapeNode                = other.m_shapeNode;
        m_physicalRois             = other.m_physicalRois;
        m_logicalRois              = other.m_logicalRois;
        m_shapeInferenceFunctionID = other.m_shapeInferenceFunctionID;
        m_quantizer                = other.m_quantizer;
        m_precision                = other.m_precision;
        m_paramsRawData            = other.m_paramsRawData;
        m_deterministic            = other.m_deterministic;
        m_originNodes              = other.m_originNodes;
        m_roundingMode             = other.m_roundingMode;
        if (!other.hasNodeIOManagerSpecialization())
        {
            m_io = std::make_unique<NodeIOManager>(this);
        }
        m_parentId = other.m_parentId;
    }
    return *this;
}

void Node::replaceFirstTensor(const TensorPtr& oldTensor, const TensorPtr& newTensor)
{
    replaceTensorPadding(oldTensor, newTensor);

    if (! replaceFirst(m_inputs, oldTensor, newTensor) &&
        ! replaceFirst(m_outputs, oldTensor, newTensor) &&
        ! replaceFirst(m_controlInputs, oldTensor, newTensor) &&
        ! replaceFirst(m_controlOutputs, oldTensor, newTensor))
    {
        HB_ASSERT(false, "Node tensor to replace does not exist");
    }
}

void Node::replaceTensorPadding(const TensorPtr& oldTensor, const TensorPtr& newTensor)
{
    if (oldTensor == nullptr || newTensor == nullptr) return;
    auto oldPaddingIter = m_paddingValues.find(oldTensor);
    if (oldPaddingIter == m_paddingValues.end()) return;
    uint32_t padding = oldPaddingIter->second;
    m_paddingValues.erase(oldPaddingIter);
    m_paddingValues.emplace(newTensor, padding);
}

void Node::replaceTensor(const TensorPtr& oldTensor, const TensorPtr& newTensor)
{
    replaceTensorPadding(oldTensor, newTensor);

    if (std::find(m_inputs.begin(), m_inputs.end(), oldTensor) != m_inputs.end())
    {
        std::replace(m_inputs.begin(), m_inputs.end(), oldTensor, newTensor);
    }
    else if (std::find(m_outputs.begin(), m_outputs.end(), oldTensor) != m_outputs.end())
    {
        std::replace(m_outputs.begin(), m_outputs.end(), oldTensor, newTensor);
    }
    else if (std::find(m_controlInputs.begin(), m_controlInputs.end(), oldTensor) != m_controlInputs.end())
    {
        std::replace(m_controlInputs.begin(), m_controlInputs.end(), oldTensor, newTensor);
    }
    else if (std::find(m_controlOutputs.begin(), m_controlOutputs.end(), oldTensor) != m_controlOutputs.end())
    {
        std::replace(m_controlOutputs.begin(), m_controlOutputs.end(), oldTensor, newTensor);
    }
    else
    {
        HB_ASSERT(false, "Node tensor to replace does not exist");
    }
}

void Node::replaceTensor(unsigned index, const TensorPtr& newTensor, TensorVector& dest)
{
    replaceTensorPadding(dest[index], newTensor);
    dest[index] = newTensor;
}

void Node::emplaceTensor(unsigned index, const TensorPtr& newTensor, bool isInput)
{
    TensorVector& destTensorVector = isInput ? m_inputs : m_outputs;
    LayoutVector& destLayout       = isInput ? m_inputLayouts : m_outputLayouts;
    if (index <= destTensorVector.size())
    {
        destTensorVector.emplace(destTensorVector.begin() + index, newTensor);
        gc::Layout layout;
        destLayout.emplace(destLayout.begin() + index, layout);
        if (isInput)
        {
            auto& permutations = m_annotation.inputPermutations;
            permutations.emplace(permutations.begin() + index, Permutation(newTensor->getDim()));
        }
    }
    else
    {
        LOG_ERR(HABANA_NODE, "Attempt to emplace tensor at illegal index {}", index);
        HB_ASSERT(false, "Attempt to emplace tensor at illegal index");
    }

}

bool Node::hasHighRankOperand() const
{
    for (const auto& t : getOperands())
    {
        if (t && t->getDim() > MAX_DIMENSIONS_NUM) return true;
    }
    return false;
}

void Node::replaceInput(unsigned index, const TensorPtr& newTensor, eTensorType tensorType)
{
    TensorVector& inputsVector = (tensorType == TENSOR_TYPE_DATA) ? m_inputs : m_controlInputs;
    if (index < inputsVector.size())
    {
        replaceTensor(index, newTensor, inputsVector);
    }
    else
    {
        const char* inputTypeString = (tensorType == TENSOR_TYPE_DATA) ? "input" : "control input";
        LOG_ERR(HABANA_NODE, "Attempt to replace {} at index {} greater than number of {}s", inputTypeString, index, inputsVector.size());
        HB_ASSERT(false, "index to replace is out of range");
    }
}

void Node::addInput(unsigned         index,
                    const TensorPtr& newTensor,
                    eTensorType      tensorType /*= TENSOR_TYPE_DATA*/,
                    bool             padWithNull /*= false*/,
                    gc::Layout       layout /*= gc::Layout()*/)
{
    TensorVector& inputsVector = (tensorType == TENSOR_TYPE_DATA) ? m_inputs : m_controlInputs;
    uint64_t maxIndex = (tensorType == TENSOR_TYPE_DATA && padWithNull) ? TENSOR_INPUT_MAX - 1 : inputsVector.size();

    if (index > inputsVector.size())
    {
        if (padWithNull && index <= maxIndex)
        {
            // fill empty cells of inputsVector with nullptr
            for (int i = inputsVector.size(); i < index; i++)
            {
                inputsVector.push_back(nullptr);
            }
        }
        else
        {
            const char* inputTypeString = (tensorType == TENSOR_TYPE_DATA) ? "input" : "control input";
            LOG_ERR(HABANA_NODE,
                    "Cannot add {} to node {} at index {}, index should be less or equal to number of {}s {}",
                    inputTypeString,
                    m_name,
                    index,
                    inputTypeString,
                    maxIndex);
            return;
        }
    }

    if (inputsVector.size() == index)
    {
        inputsVector.push_back(newTensor);
        if (tensorType == TENSOR_TYPE_DATA)
        {
            LOG_TRACE(DATA_LAYOUT, "adding new tensor in index {} for node {} with layout {}",
                      index, m_name, layout.toString());
            m_inputLayouts.emplace_back(layout);
        }
        return;
    }
    else if (inputsVector[index] == nullptr)
    {
        replaceInput(index, newTensor, tensorType);
        return;
    }
    else
    {
        LOG_ERR(HABANA_NODE,
                "Cannot add new input to node {} at index {}, already has input tensor in the requested index",
                m_name,
                index);
        return;
    }
}

void Node::removeInput(const TensorPtr& toRemove, eTensorType tensorType)
{
    TensorVector& tensorVector = (tensorType == TENSOR_TYPE_DATA) ? m_inputs : m_controlInputs;
    auto          it           = std::find(tensorVector.begin(), tensorVector.end(), toRemove);
    HB_ASSERT(it != tensorVector.end(), "can't remove non existing tensor");
    if (tensorType == TENSOR_TYPE_DATA)
    {
        unsigned index = std::distance(tensorVector.begin(), it);
        m_inputLayouts.erase(std::next(m_inputLayouts.begin(), index));
    }
    tensorVector.erase(it);
}

// Trim all data inputs from a given index
void Node::removeDataInputsFromIndex(size_t newEndIdx)
{
    if (newEndIdx < m_inputs.size())
    {
        m_inputs.resize(newEndIdx);
        m_inputLayouts.resize(newEndIdx);
    }
}

void Node::addOutput(const TensorPtr& newTensor, eTensorType tensorType)
{
    if (tensorType == TENSOR_TYPE_DATA)
    {
        m_outputs.push_back(newTensor);
        m_outputLayouts.emplace_back();
    }
    else if (tensorType == TENSOR_TYPE_CONTROL)
    {
        m_controlOutputs.push_back(newTensor);
    }
    else
    {
        LOG_ERR(HABANA_NODE,"Unexpected tensor type {}, can't add output", tensorType);
        HB_ASSERT(false, "Unexpected tensor type");
    }
}

void Node::removeOutput(const TensorPtr& toRemove, eTensorType tensorType)
{
    TensorVector& tensorVector = (tensorType == TENSOR_TYPE_DATA) ? m_outputs : m_controlOutputs;
    auto          it           = std::find(tensorVector.begin(), tensorVector.end(), toRemove);
    HB_ASSERT(it != tensorVector.end(), "can't remove non existing tensor");
    if (tensorType == TENSOR_TYPE_DATA)
    {
        unsigned index = std::distance(tensorVector.begin(), it);
        m_outputLayouts.erase(std::next(m_outputLayouts.begin(), index));
    }
    tensorVector.erase(it);
}

void Node::replaceOutput(unsigned index, const TensorPtr& newTensor)
{
    if (index < m_outputs.size())
    {
        replaceTensor(index, newTensor, m_outputs);
    }
    else
    {
        LOG_ERR(HABANA_NODE, "Attempt to replace output at index {} greater than number of outputs", index);
        HB_ASSERT(false, "Output index to replace is out of range");
    }
}

void Node::emplaceInput(unsigned index, const TensorPtr& newTensor)
{
    emplaceTensor(index, newTensor, true);
}
void Node::emplaceOutput(unsigned index, const TensorPtr& newTensor)
{
    emplaceTensor(index, newTensor, false);
}

TensorVector Node::getOperands() const
{
    TensorVector ret = getInputs();
    ret.insert(ret.end(), m_outputs.begin(), m_outputs.end());
    return ret;
}

TensorVector Node::getInputODSTs() const
{
    // return vector of inputs OUTPUT_DESCRIBING_SHAPE_TENSOR ordered by #index
    TensorVector ret;
    for (const TensorPtr& inputTensor : m_inputs)
    {
        if (inputTensor->getTensorType() == synTensorType::OUTPUT_DESCRIBING_SHAPE_TENSOR)
        {
            ret.push_back(inputTensor);
        }
    }
    return ret;
}

void Node::setInputLayouts(const LayoutVector& layouts)
{
    // we have to override the previous layouts in case "setInputLayouts" was performed more than once

    if (layouts.empty())
    {
        // if no layouts given, set "don't care" layout for each input
        m_inputLayouts = LayoutVector(m_inputs.size());
    }
    else
    {
        HB_ASSERT(layouts.size() == m_inputs.size(), "Each input must be given a corresponding layout");
        m_inputLayouts = layouts;
    }
}

void Node::setOutputLayouts(const LayoutVector& layouts)
{
    // we have to override the previous layouts in case "setOutputLayouts" was performed more than once

    if (layouts.empty())
    {
        // if no layouts given, set "don't care" layout for each output
        m_outputLayouts = LayoutVector(m_outputs.size());
    }
    else
    {
        HB_ASSERT(layouts.size() == m_outputs.size(), "Each output must be given a corresponding layout");
        m_outputLayouts = layouts;
    }
}

unsigned Node::getNumInputs(eTensorType tensorType) const
{
    if (tensorType == TENSOR_TYPE_DATA)
    {
        return (unsigned)(m_inputs.size());
    }
    else if (tensorType == TENSOR_TYPE_CONTROL)
    {
        return (unsigned)(m_controlInputs.size());
    }

    return (unsigned)(m_inputs.size() + m_controlInputs.size());
}

unsigned Node::getNumOutputs(eTensorType tensorType) const
{
    if (tensorType == TENSOR_TYPE_DATA)
    {
        return (unsigned)(m_outputs.size());
    }
    else if (tensorType == TENSOR_TYPE_CONTROL)
    {
        return (unsigned)(m_controlOutputs.size());
    }

    return (unsigned)(m_outputs.size() + m_controlOutputs.size());
}

unsigned Node::getNumInputsShapeTensors() const
{
    return std::count_if(m_inputs.begin(), m_inputs.end(), [](const TensorPtr& t) { return t && t->isShapeTensor(); });
}

unsigned Node::getNumOutputsShapeTensors() const
{
    return std::count_if(m_outputs.begin(), m_outputs.end(), [](const TensorPtr& t) { return t && t->isShapeTensor(); });
}

unsigned Node::getNumInputsH2DTensors() const
{
    return std::count_if(m_inputs.begin(), m_inputs.end(), [](const TensorPtr& t) { return t && t->isHost2DeviceTensor(); });
}

unsigned Node::getNumOutputsH2DTensors() const
{
    return std::count_if(m_outputs.begin(), m_outputs.end(), [](const TensorPtr& t) { return t && t->isHost2DeviceTensor(); });
}

unsigned Node::getNumInputsDataTensors() const
{
    return getNumInputs() - getNumInputsShapeTensors();
}

unsigned Node::getNumOutputsDataTensors() const
{
    return getNumOutputs() - getNumOutputsShapeTensors();
}

unsigned Node::getInputIndexOfTensor(const TensorPtr& tensor) const
{
    unsigned i = 0;
    while (i < m_inputs.size())
    {
        if (getInput(i) == tensor) break;
        ++i;
    }
    HB_ASSERT(i != m_inputs.size(), "tensor {} is not an input of this node!", tensor->getName());
    return i;
}

unsigned Node::getOutputIndexOfTensor(const TensorPtr& tensor) const
{
    unsigned i = 0;
    while (i < m_outputs.size())
    {
        if (getOutput(i) == tensor) break;
        ++i;
    }
    HB_ASSERT(i != m_outputs.size(), "tensor is not an output of this node!");
    return i;
}

Node::eParamUsage Node::getParamUsage(const TensorPtr& param) const
{
    //Look for it everywhere. If this becomes a problem we'll cache a map of param to usage
    for (const TensorPtr& it : m_inputs)
    {
        if (it == nullptr) continue;
        if (param == it)
        {
            return Node::USAGE_INPUT;
        }
    }
    for (const TensorPtr& it : m_controlInputs)
    {
        if (it == nullptr) continue;
        if (param == it)
        {
            return Node::USAGE_INPUT;
        }
    }
    for (const TensorPtr& it : m_outputs)
    {
        if (it == nullptr) continue;
        if (param == it)
        {
            return Node::USAGE_OUTPUT;
        }
    }
    for (const TensorPtr& it : m_controlOutputs)
    {
        if (it == nullptr) continue;
        if (param == it)
        {
            return Node::USAGE_OUTPUT;
        }
    }
    return Node::UNUSED;
}

TensorSemanticType Node::getParamSemanticType(const TensorPtr& param) const
{
    return TYPE_ACTIVATION;
}

unsigned Node::getKDimIndex()
{
    const TensorPtr& weightTensor = getInput(TENSOR_WEIGHT);
    if (weightTensor == nullptr)
    {
        LOG_ERR(HABANA_NODE, "Weight tensor does not exist");
        return Tensor::c_tensorMaxDim;
    }

    return inputDimNameToIndex(TENSOR_WEIGHT, 'K');
}

std::string Node::getNodeTypeStr() const
{
    // clang-format off
    // Note: keep in sync with Node::eNodeType
    static const char* nodeTypeStrings[] = {
        "TPC", "Add", "Convolution", "GEMM", "Pooling", "ReLU", "FullyConnected", "DEDX", "DEDW", "BatchNormalization",
        "DMA", "AddBias", "Concatenate", "Flatten", "Split", "ExpandDims", "Reshape", "Transpose", "LogicalTranspose",
        "LogicalBroadcast", "Custom", "Debug", "Debug2", "Packing", "Lowering", "BatchGemm", "MaskedBatchGemm",
        "SliceAxis", "TensorView", "Reduction", "StridedView", "StridedInsert", "MultiInsert", "Slice", "Memcpy",
        "Memset", "NMS", "BatchGemmDeDx", "BatchGemmDeDw", "GemmDeDx", "GemmDeDw", "Identity", "Moments",
        "TfBatchNormalization", "TfFusedBatchNormGrad", "FcdBroadcast", "Broadcast", "StridedSliceGrad", "SliceInsert",
        "StridedSliceBwd", "Wait", "LogicalRequant", "ImageRotate", "PhysicalConcat", "ExtractShape", "MergeShapes",
        "SplitShape", "FlattenShape", "ExpandDimsShape", "SqueezeShape", "PhysicalReshape", "TransposedShape",
        "StaticReshape", "TensorViewShape", "Squeeze", "FrobeniusNorm", "DynamicSplit", "PhysicalSplit", "Einsum",
        "DynamicReShape", "EinsumExpand", "PhysicalFlatten", "DynamicRange", "InferShape", "ReinterpretCast",
        "InferMaxShape", "H2DOp", "TransposedDedx", "Reverse", "TileShape", "OperandReuseInternal"
    };
    // clang-format on
    static_assert(ARRAY_SIZE(nodeTypeStrings) == Node::TYPE_MAX, "nodeTypeStrings out of sync with eNodeType");

    HB_ASSERT(m_type < ARRAY_SIZE(nodeTypeStrings), "Unknown m_type {}", m_type);
    return nodeTypeStrings[m_type];
}

std::string_view Node::getEngineTypeStr() const
{
    return "None";
}

std::string Node::getNodeParametersStr() const
{
    return "";
}

uint32_t Node::getPaddingValue(const TensorPtr& t) const
{
    auto tPadingVal = m_paddingValues.find(t);
    if (tPadingVal != m_paddingValues.end())
    {
        return tPadingVal->second;
    }
    else
    {
        // zero point of zero is the frequent case so we wish to optimize for it.
        double zeroPoint = t->getZeroPoint();
        if (zeroPoint == 0) return 0;
        if (t->getElementType() == syn_type_na)
        {
            return zeroPoint;
        }
        unsigned typeSize = t->getElementSizeInBits();
        uint32_t pad;
        padBuffWithValue(&pad, BITS_PER_BYTE * sizeof(pad) / typeSize, zeroPoint, t->getElementType());
        return pad;
    }
}

void Node::setPaddingValue(const TensorPtr& t, float padVal)
{
    unsigned typeSize = t->getElementSizeInBytes();
    uint32_t pad;

    float realValArray[] = {padVal};
    QuantizationData quantInfo(t->getQuantizationParams());

    switch (t->getElementType())
    {
    case syn_type_fixed:
    {
        int8_t* quantized = static_cast<int8_t*>(QuantizationUtils::quantRealData(realValArray, quantInfo, 1));
        padBuffWithValue(&pad, sizeof(pad) / typeSize, *quantized, t->getElementType());
        delete[] quantized;
        break;
    }
    case syn_type_uint8:
    {
        uint8_t* quantized = static_cast<uint8_t*>(QuantizationUtils::quantRealData(realValArray, quantInfo, 1));
        padBuffWithValue(&pad, sizeof(pad) / typeSize, *quantized, t->getElementType());
        delete[] quantized;
        break;
    }
    case syn_type_int16:
    {
        int16_t* quantized = static_cast<int16_t*>(QuantizationUtils::quantRealData(realValArray, quantInfo, 1));
        padBuffWithValue(&pad, sizeof(pad) / typeSize, *quantized, t->getElementType());
        delete[] quantized;
        break;
    }
    case syn_type_uint16:
    {
        uint16_t* quantized = static_cast<uint16_t*>(QuantizationUtils::quantRealData(realValArray, quantInfo, 1));
        padBuffWithValue(&pad, sizeof(pad) / typeSize, *quantized, t->getElementType());
        delete[] quantized;
        break;
    }
    default:
        HB_ASSERT(false, "Unknown data type");
    }

    m_paddingValues[t] = pad;
}

gc::access_pattern::NodeAccessPatternPtr Node::getNodeAccessPattern() const
{
    if (!m_nodeAccessPatternCache)
    {
        m_nodeAccessPatternCache = generateNodeAccessPattern();
    }
    return m_nodeAccessPatternCache;
}

NodeROI Node::generateRoi() const
{
    NodeROI fullRoi;
    const TensorPtr& t = !m_inputs.empty() ? getInput(TENSOR_IFM) : getOutput(TENSOR_OFM);
    fullRoi.size[0] = t->getTotalSizeInBytes()/t->getElementSizeInBytes();
    std::fill(fullRoi.size + 1, fullRoi.size + ARRAY_SIZE(fullRoi.size), 1);
    return fullRoi;
}

Settable<NodeROI> Node::getInputROI(const NodeROI& roi, uint32_t tensorIdx) const
{
    // Return unset object
    return Settable<NodeROI>();
}

Settable<NodeROI> Node::getOutputROI(const NodeROI& roi, uint32_t tensorIdx) const
{
    return roi;
}

TensorShape Node::getInputShape(const TensorShape& outputShape, uint32_t outputIdx, uint32_t inputIdx) const
{
    const TensorPtr& tensor = getInput(inputIdx);
    if (tensor == nullptr)
    {
        LOG_ERR(HABANA_NODE, "Node has no input!");
        throw(NodeHasNoInput(getNodeName()));
    }
    NSizeArray size;
    tensor->getAllNSizesInElements(size.data());
    TensorShape inputShape(tensor->getDim(), size);

    return inputShape;
}

synDataType Node::getRequiredInputType(uint32_t tensorIdx) const
{
    return getInput(tensorIdx)->getElementType();
}

synDataType Node::getRequiredOutputType(uint32_t tensorIdx) const
{
    return getOutput(tensorIdx)->getElementType();
}

bool Node::validateNode() const
{
    auto valid = validateNode64BitOperands();
    return valid;
}

bool Node::validateNodeLayout() const
{
    return true; // default implementation
}

bool Node::is64BitOperands() const
{
    const TensorVector& inputs  = getInputs();
    const TensorVector& outputs = getOutputs();
    return is64BitOperands(inputs, outputs);
}

bool Node::is64BitOperands(const TensorVector& inputs, const TensorVector& outputs)
{
    auto        is64BitInput = std::any_of(inputs.begin(), inputs.end(), [](const TensorPtr& input) {
        return input && input->isDataTensor() && input->is64BitElementSize();
    });

    auto is64BitOutput = std::any_of(outputs.begin(), outputs.end(), [](const TensorPtr& output) {
        return output && output->isDataTensor() && output->is64BitElementSize();
    });
    return is64BitInput || is64BitOutput;
}

bool DebugNodeBase::isNode64BitCompatible() const
{
    return true;
}

bool Node::isNode64BitCompatible() const
{
    return false;
}

bool Node::validateNode64BitOperands() const
{
    if (!is64BitOperands())
    {
        return true;
    }

    if (!isNode64BitCompatible())
    {
        LOG_ERR(HABANA_NODE,
                "Failed 64Bit operand validation. Node {} does not support 64Bit operands.",
                getNodeName());
        return false;
    }
    return true;
}

void Node::print() const
{
    if (!LOG_LEVEL_AT_LEAST_DEBUG(GRAPH_DATA)) return;

    LOG_DEBUG(GRAPH_DATA,
              "Node {} (type: {}, GUID: {}) execOrderedIdx = {}, parentId = {}, {}",
              getNodeName(),
              Node::getNodeTypeStr(),
              getGUID(),
              getExecutionOrderedIndex(),
              getParentId(),
              isLogicalOperation() ? "is a logical node" : "");
    printParamsRawData();
    if (getNodeAnnotation().bundleInfo.is_set())
    {
        LOG_DEBUG(GRAPH_DATA, "    In bundle. bundleIndex: {}, operationIndex: {}", getNodeAnnotation().bundleInfo.value().bundleIndex, getNodeAnnotation().bundleInfo.value().operationIndex);
    }

    LOG_DEBUG(GRAPH_DATA, "  Inputs:");
    for (auto t : getInputs())
    {
        if (t == nullptr) continue;
        PrintOperand(t);
    }
    LOG_DEBUG(GRAPH_DATA, "  Outputs:");
    for (auto t : getOutputs())
    {
        if (t == nullptr) continue;
        PrintOperand(t);
    }
    LOG_DEBUG(GRAPH_DATA, "  Control Inputs:");
    for (auto t : getControlInputs())
    {
        if (t == nullptr) continue;
        PrintOperand(t);
    }
    LOG_DEBUG(GRAPH_DATA, "  Control Outputs:");
    for (auto t : getControlOutputs())
    {
        if (t == nullptr) continue;
        PrintOperand(t);
    }
}

void Node::PrintOperand(const TensorPtr& t) const
{
    if (!LOG_LEVEL_AT_LEAST_DEBUG(GRAPH_DATA)) return;

    LOG_DEBUG(GRAPH_DATA,
              "    {} {} offset (0x{:x}) ({}) sizes={} minSizes={} strides={} isDense={} allowPermutation={} "
              "permutation={} isReduction={}"
              " isPersistent={} memSectionId={} isShape={} isExternal={} is {}{}",
              t->getName(),
              t->tensorAllocatedInSram() ? "SRAM" : "DRAM",
              t->getTensorOffset(),
              getStringFromSynDataType(t->getElementType()),
              t->getDimSizesStr(),
              t->getDimSizesStr(false, true),
              t->getStridesStr(),
              t->isDenseLayout(),
              t->getTensorAnnotation().memory.allowPermutation,
              t->getPermutation().has_value() ? t->getPermutation().value().toString() : "None",
              t->isReductionEnabled() ? "1" : "0",
              t->isPersistent() ? "1" : "0",
              t->getMemorySectionID(),
              t->isShapeTensor() ? "1" : "0",
              t->getTensorIsExternal() ? "1" : "0",
              t->getTensorLocationString(),
              t->isAliasedTensor() ? fmt::format(" and is an alias of {}", Tensor::getRealTensor(t)->getName()) : "");

    if (t->isStaticParam())
    {
        LOG_DEBUG(GRAPH_DATA, "       Tensor is a static tensor that {}. DRAM offset (0x{:x}). {}",
                  !t->tensorAllocatedInSram() ? "resides in DRAM" :
                  (t->getTensorAnnotation().memory.pinned ? "is pinned to SRAM" : "will be prefetched to SRAM"),
                  t->getTensorOffset(), t->getTensorAnnotation().sparseAccess ? "Tensor is sparsely accessed" : "");
        LOG_DEBUG(GRAPH_DATA, "       Static data buffer details: address 0x{:x}, type {}, size in bytes {}",
                  (uint64_t)(t->getData()), getStringFromSynDataType(t->getBufferDataType()), t->getBufferSizeInBytes());
    }

    // User memory sections
    if(t->isPersistent())
    {
        LOG_DEBUG(GRAPH_DATA,
                  "       User memory section: type=Persistent id={} offset=0x{:x}",
                  t->getMemorySectionID(),
                  t->getMemorySectionOffset());
    }
    else if (t->isPartOfRMWSection())
    {
        const auto& sectionInfo = t->getTensorAnnotation().nonPersistentSectionInfo;
        LOG_DEBUG(GRAPH_DATA,
                  "       User memory section: type=RMW id={} offset=0x{:x}",
                  sectionInfo.sectionId.value(),
                  sectionInfo.offsetFromBase.value());
    }

    if (!t->getTensorAnnotation().memorySpaceInfo.barriers.empty())
    {
        LOG_DEBUG(GRAPH_DATA, "       Tensor is waiting for the following nodes:");
        for (auto barrier : t->getTensorAnnotation().memorySpaceInfo.barriers)
        {
            LOG_DEBUG(GRAPH_DATA, "           * Node {}", barrier->getNodeName());
        }
    }
}

void Node::printParamsRawData() const
{
}

bool Node::equalTo(const Node& o) const
{
    //Default: node A equals node B if they have the same enum and the same Tensor geometry in the same order
    if (o.getNodeType()   != getNodeType())   return false;
    if (o.getNumInputs()  != getNumInputs())  return false;
    if (o.getNumOutputs() != getNumOutputs()) return false;

    TensorVector my_operands = getOperands();
    TensorVector o_operands  = o.getOperands();

    HB_ASSERT(my_operands.size() == o_operands.size(), "operands size mismatch");

    for (unsigned i = 0; i < my_operands.size(); ++i)
    {
        if (getParamUsage(my_operands[i]) != o.getParamUsage(o_operands[i])) return false;
        if (my_operands[i] != nullptr && o_operands[i] != nullptr)
        {
            if (*my_operands[i] != *o_operands[i]) return false;
        }
        else if (!(my_operands[i] == nullptr && o_operands[i] == nullptr))
        {
            return false;
        }
    }
    return true;
}

bool Node::RunOnCpu()
{
    //Nop by default
    return false;
}

bool Node::RunOnCpu(const HabanaGraph& g)
{
    return RunOnCpu();
}

bool Node::isLogicalOperation() const
{
    return (m_type == TYPE_DEBUG) || (m_type == TYPE_DEBUG2);
}

bool Node::isShapeOperation() const
{
    return false;
}

void Node::runLogicalOperation() const
{
    // Do nothing
}

bool Node::isTranspose() const
{
    return m_type == Node::TYPE_INTERNAL_TRANSPOSE || m_type == Node::TYPE_LOGICAL_TRANSPOSE;
}

bool Node::isMemset() const
{
    return getNodeType() == Node::TYPE_MEMSET;
}

bool Node::isCast() const
{
    return false;
}

bool Node::isBatchGemm() const
{
    return false;
}

bool Node::isSplit() const
{
    return getNodeType() == Node::TYPE_INTERNAL_SPLIT;
}

bool Node::isPartOfRMWSection() const
{
    const std::unordered_set<uint64_t>& rmwSectionIds = getRMWSectionIds();
    return !rmwSectionIds.empty();
}

uint64_t Node::getRMWSectionId() const
{
    const std::unordered_set<uint64_t>& rmwSectionIds = getRMWSectionIds();
    // The assert should not happen, verified in addNode.
    HB_ASSERT(rmwSectionIds.size() == 1, "Invalid complex GUID node {} - uses more than one section", getNodeName());
    return *rmwSectionIds.begin();
}

bool Node::validateRMWSection(uint64_t maxRMWSectionSize) const
{
    bool         ret        = true;
    TensorVector RMWTensors = getRMWTensors();
    for (const auto& t : RMWTensors)
    {
        if (t->getTensorAnnotation().nonPersistentSectionInfo.offsetFromBase.value() + t->getTotalSizeInBytes() >
            maxRMWSectionSize)
        {
            LOG_ERR(GC, "tensor {} of node {} exceeds max size of RMW section", t->getName(), getNodeName());
            ret = false;
        }
    }

    const std::unordered_set<uint64_t>& rmwSectionIds = getRMWSectionIds(RMWTensors);
    if (rmwSectionIds.size() > 1)
    {
        LOG_ERR(GC, "node {} uses more than one rmw section", getNodeName());
        ret = false;
    }
    return ret;
}

TensorVector Node::getRMWTensors() const
{
    TensorVector RMWTensors;
    for (const auto& t : getOperands())
    {
        if (t && t->isPartOfRMWSection())
        {
            RMWTensors.push_back(t);
        }
    }
    return RMWTensors;
}

std::unordered_set<uint64_t> Node::getRMWSectionIds() const
{
    TensorVector rmwTensors = getRMWTensors();
    return getRMWSectionIds(rmwTensors);
}

std::unordered_set<uint64_t> Node::getRMWSectionIds(TensorVector& RMWTensors) const
{
    std::unordered_set<uint64_t> ids;
    for (const auto& t : RMWTensors)
    {
        HB_ASSERT(t && t->isPartOfRMWSection(), "expected only RMW tensors");
        const uint64_t id = t->getTensorAnnotation().nonPersistentSectionInfo.sectionId.value();
        ids.insert(id);
    }
    return ids;
}

bool Node::isDramFill() const
{ // node is a dram fill is it's input is in DRAM and output in SRAM.
    if ((getNumOutputs() != 1) || (getNumInputs() != 1)) return false;
    return getInput(0)->inDram() && !getOutput(0)->inDram();
}

bool Node::isDramSpill() const
{ // node is a dram spill is it's input is in SRAM and output in DRAM.
    if ((getNumOutputs() != 1) || (getNumInputs() != 1)) return false;
    return !getInput(0)->inDram() && getOutput(0)->inDram();
}

bool Node::isDynamicShape() const
{
    for (const TensorPtr& tensor : m_inputs)
    {
        if (tensor != nullptr && tensor->isDynamicShape())  return true;
    }
    for (const TensorPtr& tensor : m_outputs)
    {
        if (tensor != nullptr && (tensor->isDynamicShape() || tensor->isHost2DeviceTensor()))  return true;
    }

    return false;
}

bool Node::validateDynamicShapes() const
{
    return true;
}

bool Node::isTensorRoiDynamic(const pTensor& tensor, const TensorROI& tensorRoi) const
{
    SizeArray minSizes = tensor->getAllMinimalSizesInElements();
    const TensorROILayout& roiLayout = tensorRoi.getLayout();

    auto dims = tensor->getDim();
    HB_ASSERT(minSizes.size() >= dims, "Min sizes must be greater or equalt to number of dims {}", dims);
    HB_ASSERT(Tensor::c_tensorMaxDim >= dims, "Dims {} must not be greater the max number of dims", dims);

    for (int d = 0; d < dims; d++)
    {
        // If the roi offset + size is bigger than the min size, this roi is in the dynamic region of the tensor
        if (minSizes[d] < roiLayout.m_baseOffset[d] + roiLayout.m_size[d])
        {
            return true;
        }
    }

    return false;
}

bool Node::isROIDynamic(const NodeROI* roi) const
{
    unsigned roiIndex = 0;
    for (int i = 0; i < m_inputs.size(); i++)
    {
        // In ROISplitter::projectDmaRoisToFullSizes we only generate ROI for
        // the first tensor regardless of the number of input/output tensors.
        if (roiIndex == roi->inputRois.size()) break;

        const pTensor& tensor = m_inputs[i];
        // In ROISplitter::projectTPCRois we skip allocating shape tensors ROI.
        if (tensor != nullptr && tensor->isShapeTensor()) continue;

        if (tensor != nullptr && tensor->isDynamicShape() && isTensorRoiDynamic(tensor, roi->inputRois[roiIndex]))
        {
            return true;
        }
        roiIndex++;
    }

    roiIndex = 0;
    for (int i = 0; i < m_outputs.size(); i++)
    {
        // In ROISplitter::projectDmaRoisToFullSizes we only generate ROI for
        // the first tensor regardless of the number of input/output tensors.
        if (roiIndex == roi->outputRois.size()) break;

        const pTensor& tensor = m_outputs[i];
        // In ROISplitter::projectTPCRois we skip allocating shape tensors ROI.
        if (tensor != nullptr && tensor->isShapeTensor()) continue;

        if (tensor != nullptr && tensor->isDynamicShape() && isTensorRoiDynamic(tensor, roi->outputRois[roiIndex]))
        {
            return true;
        }
        roiIndex++;
    }

    return false;
}

sm_function_id_t Node::getShapeInferenceFunctionId(bool skipStatic) const
{
    sm_function_id_t ret;
    ret.sm_func_index = m_shapeInferenceFunctionID;

    if (skipStatic && !GCFG_ENABLE_SIF_FOR_STATIC_NODES.value() && !isDynamicShape() && !requiresOutputMaxDimInfer())
    {
        LOG_DEBUG(GC, "DSD Node {} Is static and requires no max-dim infer and has no sif", this->getNodeName());
        ret.sm_func_index = INVALID_SHAPE_FUNC_ID;
    }

    return ret;
}

uint64_t Node::getShapeInferenceFunctionVersion() const
{
    return GC_SIF_VERSION;
}

SifNodeParams Node::getShapeInferenceFunctionUserParams()
{
    return nullptr;
}

size_t Node::getShapeInferenceFunctionUserParamsSize() const
{
    return 0;
}

bool Node::isBroadcastableOperation() const
{
    return false;
}

void Node::replaceAllTensors(TensorVector&& inputs, TensorVector&& outputs)
{
    if (m_paddingValues.empty())
    {
        m_inputs  = std::move(inputs);
        m_outputs = std::move(outputs);
    }
    else
    {
        for (unsigned i = 0; i < m_inputs.size(); ++i)
        {
            replaceTensor(i, inputs[i], m_inputs);
        }
        for (unsigned i = 0; i < m_outputs.size(); ++i)
        {
            replaceTensor(i, outputs[i], m_outputs);
        }
    }
}

NodePtr Node::cloneWithTensors() const
{
    auto fillTensorVector = [](const TensorVector& inputs, TensorVector& outputs) {
        outputs.reserve(inputs.size());
        for (const auto& t : inputs)
        {
            outputs.push_back(t ? t->clone() : t);
        }
    };

    TensorVector inputs;
    fillTensorVector(m_inputs, inputs);
    TensorVector outputs;
    fillTensorVector(m_outputs, outputs);
    NodePtr ret = clone();
    ret->replaceAllTensors(std::move(inputs), std::move(outputs));
    return ret;
}

void Node::cloneConnectivityFromNode(const Node& o)
{
    m_inputs         = o.m_inputs;
    m_outputs        = o.m_outputs;
    m_controlInputs  = o.m_controlInputs;
    m_controlOutputs = o.m_controlOutputs;
    m_inputLayouts   = o.m_inputLayouts;
    m_outputLayouts  = o.m_inputLayouts;
}

NodePtr Node::getSlice() const
{
    // Default implementation
    return clone();
}

unsigned Node::inputDimNameToSize(unsigned inputId, char dimensionName) const
{
    return getDimensionNameToSize(inputId, dimensionName, true);
}

unsigned Node::outputDimNameToSize(unsigned outputId, char dimensionName) const
{
    return getDimensionNameToSize(outputId, dimensionName, false);
}

unsigned Node::inputDimNameToIndex(unsigned inputId, char dimensionName) const
{
    return getDimensionNameToIndex(inputId, dimensionName, true);
}

unsigned Node::outputDimNameToIndex(unsigned outputId, char dimensionName) const
{
    return getDimensionNameToIndex(outputId, dimensionName, false);
}

unsigned Node::getDimensionNameToSize(unsigned tensorId, char dimensionName, bool isInput) const
{
    auto& tensors = isInput ? m_inputs : m_outputs;
    HB_ASSERT(tensorId < tensors.size(), "Invalid tensor index");
    unsigned dim = getDimensionNameToIndex(tensorId, dimensionName, isInput);
    if (dim >= Tensor::c_tensorMaxDim)
    {
        return 0;
    }

    return tensors[tensorId]->getSizeInElements(dim);
}

unsigned Node::getDimensionNameToIndex(unsigned layoutId, char dimensionName, bool isInput) const
{
    auto& layouts = isInput ? m_inputLayouts : m_outputLayouts;
    HB_ASSERT(layoutId < layouts.size(), "Invalid layout index");
    gc::Layout layout = layouts[layoutId];
    if (isInput)
    {
        auto& permutations = m_annotation.inputPermutations;
        if (permutations.size() > 0)
        {
            HB_ASSERT(layoutId < permutations.size(), "Invalid permutation index");
            layout = layout.permute(permutations[layoutId]);
        }
    }
    unsigned dim = 0;
    try
    {
        dim = layout.getIndexByName(dimensionName);
    }
    catch (const LayoutException& p)
    {
        LOG_ERR(HABANA_NODE, "Trying to find dimension name {} which doesn't exist", dimensionName);
        return Tensor::c_tensorMaxDim;
    }
    HB_ASSERT(dim < Tensor::c_tensorMaxDim, "Invalid dim index");
    return dim;
}

bool Node::setGraphTraits(const std::shared_ptr<GraphTraits>& traits)
{
    if (m_graphTraits != nullptr && traits != nullptr && m_graphTraits != traits)
    {
        LOG_ERR(HABANA_NODE, "Node {} is already assigned to a graph", getNodeName());
        HB_ASSERT(false, "Assign node to multiple graphs is not allowed");
        return false;
    }
    m_graphTraits = traits;

    return m_graphTraits != nullptr;
}

std::map<TensorPtr, TensorVector, TensorComparator> Node::getReusableInputs() const
{
    std::map<TensorPtr, TensorVector, TensorComparator> ret;
    return ret;
}

std::map<TensorPtr, TensorVector, TensorComparator> Node::getReusableInputBinding() const
{
    std::map<TensorPtr, TensorVector, TensorComparator> ret;
    return ret;
}

bool Node::hasBindingInputReuse() const
{
    // getReusableInputBinding returns a map containing all outputs. There's a binding reuse if one of them is mapped to
    // a non-empty container.
    for (const auto& tensorAndReusableInputs : getReusableInputBinding())
    {
        if (!tensorAndReusableInputs.second.empty()) return true;
    }
    return false;
}

void Node::setParamsRawData(void* params, size_t size)
{
    const uint8_t* p = static_cast<uint8_t*>(params);
    m_paramsRawData.clear();
    m_paramsRawData.insert(m_paramsRawData.begin(), p, p + size);
}

void Node::printParamsRawData(void* params, uint32_t size) const
{
    if (!LOG_LEVEL_AT_LEAST_TRACE(GRAPH_DATA)) return;

    const uint8_t*   p = static_cast<uint8_t*>(params);
    std::vector<int> params_vec(p, p + size);
    LOG_TRACE(GRAPH_DATA, "    Params raw data: {}", toString(params_vec, ','));
}

uint64_t Node::getUsedBytesByROI(TensorLocation location, unsigned clSize, const TensorROIVector& tensorRois) const
{
    unsigned bytes = 0;
    for (const TensorROI& tRoi : tensorRois)
    {
        if (tRoi.m_overlapRoi.isSram ^ (location == TensorLocation::TENSOR_IN_SRAM)) continue;
        for (const auto& subRoi : *tRoi.m_overlapRoi.subRois)
        {
            for (const auto& range : subRoi.ranges)
            {
                unsigned rangeSize = range.size();
                unsigned pad = 0;
                if (rangeSize % clSize != 0)
                {
                    pad = clSize - (rangeSize % clSize);
                }
                bytes += (rangeSize + pad);
            }
        }
    }
    return bytes;
}

void Node::setNodePrecision(synDataType precision)
{
    if (precision == getNodePrecision()) return;

    LOG_DEBUG(DATA_TYPES,
              "Updating node {} precision {} to {}",
              getNodeName(),
              getStringFromSynDataType(getNodePrecision()),
              getStringFromSynDataType(precision));
    m_precision = precision;
}

void Node::setNodePrecisionFromGUID(std::string_view guid)
{
    std::string_view guidDTypeStr  = isCast() ? extractDtypeFromCastGUID(guid) : extractDtypeFromGUID(guid);
    synDataType userPrecision = getSynDataTypeFromDtypeSuffix(guidDTypeStr);
    setNodePrecision(userPrecision);
}

uint64_t Node::getReadBytes(TensorLocation location, const std::list<NodeROI>& rois, unsigned clSize) const
{
    if (isLogicalOperation()) return 0;
    uint64_t bytes = 0;
    for (auto& roi : rois)
    {
        bytes += getUsedBytesByROI(location, clSize, roi.inputRois);
    }
    return bytes;
}

uint64_t Node::getWriteBytes(TensorLocation location, const std::list<NodeROI>& rois, unsigned clSize) const
{
    if (isLogicalOperation()) return 0;
    uint64_t bytes = 0;
    for (auto& roi : rois)
    {
        bytes += getUsedBytesByROI(location, clSize, roi.outputRois);
    }
    return bytes;
}

// Used for testing since normal flows do not call inferOutputsSizes unless inference is required
bool Node::inferOutputsShape(synDeviceType deviceType, bool inferMax)
{
    if ((!inferMax && isDynamicShape()) || (inferMax && requiresOutputMaxDimInfer()))
    {
        return inferOutputsSizes(deviceType, inferMax);
    }
    return true;
}

void applyWideBucketPolicy(const TensorPtr&   inputTensor,
                           bool               inferMax,
                           const std::string& nodesGuid,
                           TensorShapeInfo*   inTensorShape)
{
    static const std::string exceptions[] =
    {
        "reduce_",
        "argmax_",
        "argmin_"
    };

    if (!inferMax)
    {
        // reduce and similar hernels do something unreasonable with zero sized tensors.
        // Give it 1 instead.

        bool isException = std::find_if(std::begin(exceptions), std::end(exceptions),
                                        [&nodesGuid](const std::string& except) {
                                           // in c++20 use starts_with
                                           // Here we compare up to length of the second string
                                           return (nodesGuid.compare(0, except.size(), except) == 0);
                                        }) != std::end(exceptions);

        for (unsigned d = 0; d < inTensorShape->geometry.dims; ++d)
        {
            // cannot check isDynamicSize because sometimes min > max and we cannot
            // reverse that
            if (inputTensor->getSizeInElements(d) > inputTensor->getMinimalSizeInElements(d))
            {
                inTensorShape->geometry.maxSizes[d] = isException ? 1 : 0;
            }
        }
    }
}

void Node::prepareInputTensorForSif(const TensorVector&            inputTensors,
                                        bool                           inferMax,
                                        std::vector<TensorShapeInfo>&  inTensorShapes,
                                        std::vector<TensorShapeInfo*>& inTensorsPointers)
{
    for (int i = 0; i < inputTensors.size(); i++)
    {
        inTensorShapes[i].geometry.dims = inputTensors[i]->getDim();

        // NDims is not supported yet
        HB_ASSERT(inTensorShapes[i].geometry.dims <= tpc_lib_api::MAX_TENSOR_DIM,
                  "NDims tensors are not supported! Found a tensor {} with {} dimensions for node {}",
                  inputTensors[i]->getName(),
                  inTensorShapes[i].geometry.dims,
                  getNodeName());
        // Cap dims in case we do not throw in the above assert
        inTensorShapes[i].geometry.dims = std::min(inTensorShapes[i].geometry.dims, tpc_lib_api::MAX_TENSOR_DIM);

        const SizeArray& inTensorSizes =
            inferMax ? inputTensors[i]->getAllSizesInElements() : inputTensors[i]->getAllMinimalSizesInElements();

        memcpy(inTensorShapes[i].geometry.maxSizes, inTensorSizes.data(), inTensorShapes[i].geometry.dims * sizeof(TSize));

        if (GCFG_ENABLE_WIDE_BUCKET.value())
        {
            applyWideBucketPolicy(inputTensors[i], inferMax, getGUID(), &(inTensorShapes[i]));
        }

        if (inputTensors[i]->hasHostData())
        {
            char* hostData = inferMax ? inputTensors[i]->getHostMaxData() : inputTensors[i]->getHostMinData();
            inTensorShapes[i].hostAddress = reinterpret_cast<unsigned*>(hostData);
        }

        inTensorsPointers[i] = &(inTensorShapes[i]);
    }
}

bool Node::inferOutputsSizes(synDeviceType deviceType, bool inferMax, bool forbidInvalid, bool skipStatic)
{
    auto filterOutAuxAndNull = [](const TensorVector& tensors) {
        TensorVector result;
        std::copy_if(tensors.begin(), tensors.end(), std::back_inserter(result), [](const TensorPtr& value) {
            return value && !value->isAuxTensor();
        });
        return result;
    };

    TensorVector                  inputTensors = filterOutAuxAndNull(getInputs());
    SifParams                     sifParams {0};
    SifOutputs                    sifOutputs {nullptr};
    std::vector<TensorShapeInfo>  inTensorShapes(inputTensors.size());
    std::vector<TensorShapeInfo*> inTensorsPointers(inputTensors.size());

    prepareInputTensorForSif(inputTensors, inferMax, inTensorShapes, inTensorsPointers);

    TensorVector                  outputTensors = filterOutAuxAndNull(getOutputs());
    std::vector<TensorShapeInfo>  outTensorShapes(outputTensors.size());
    std::vector<TensorShapeInfo*> outTensorPointers(outputTensors.size());

    if(outTensorShapes.size() > 0)
    {
        auto arrayDataSize = sizeof(outTensorShapes[0]) * outTensorShapes.size();
        memset(outTensorShapes.data(), 0, arrayDataSize);
    }
    for (int i = 0; i < outputTensors.size(); i++)
    {
        // currently dims are not being infered by the tpc and other SIF functions.
        // So we take the dims from the first input if present or from the output itself.
        // This is not always correct, but if we are able to infer the shape correctly
        // then we'll also be able to infer it for later duplicated graphs since the
        // shape dimensions are going to be part of the shape agnostic jit graph key
        // in Pytorch bridge.
        unsigned dims = outputTensors[i]->getDim();
        if (inferMax && !outputTensors[i]->isPropSet(synTensorPropGeometryMax) && !inputTensors.empty())
        {
            dims = inputTensors[0]->getDim();
        }
        outTensorShapes[i].geometry.dims = dims;
        outTensorPointers[i] = &(outTensorShapes[i]);

        // NDims is not supported yet
        if (outTensorShapes[i].geometry.dims > SYN_MAX_TENSOR_DIM)
        {
            LOG_ERR(GC,
                    "NDims tensors are not supported! Found a tensor {} with {} dimensions for node {}",
                    outputTensors[i]->getName(),
                    outTensorShapes[i].geometry.dims,
                    getNodeName());
            return false;
        }

        if (outputTensors[i]->hasHostData())
        {
            char* hostData = inferMax ? outputTensors[i]->getHostMaxData() : outputTensors[i]->getHostMinData();
            outTensorShapes[i].hostAddress = reinterpret_cast<unsigned*>(hostData);
        }
    }

    // Create an array of bit fields in length corresponding to the output tensor count.
    size_t invalidMaskSize = 1;
    if (!outputTensors.empty())
    {
        invalidMaskSize = div_round_up(outputTensors.size(), BITS_IN_UINT32);
    }

    std::vector<uint32_t> invalidMask(invalidMaskSize, 0);

    auto inputPermutations  = getInputPermutations();

    HB_ASSERT(inputPermutations.size() == 0 || inputPermutations.size() >= inputTensors.size(),
              "number of input permutations {} is less than the number of inputs {}",
              inputPermutations.size(),
              inputTensors.size());

    sifParams.inputTensorsNr            = inputTensors.size();
    sifParams.inputTensors              = inTensorsPointers.data();
    sifParams.nodeParams.nodeParams     = getShapeInferenceFunctionUserParams();
    sifParams.nodeParams.nodeParamsSize = getShapeInferenceFunctionUserParamsSize();
    sifParams.outputTensorsNr           = outputTensors.size();
    sifParams.inputPermutations         = inputPermutations.empty() ? nullptr : inputPermutations.data();
    sifParams.outputPermutations        = nullptr;

    sifOutputs.outputTensors = outTensorPointers.data();
    sifOutputs.invalidMask   = invalidMask.data();

    bool ret = runShapeInferenceFunction(deviceType, &sifParams, &sifOutputs, inferMax, skipStatic);
    // Note - for clean compilation reset the pointers so it will not be possible to use them to address deleted ptrs.
    sifParams.inputTensors = nullptr;
    sifOutputs.outputTensors = nullptr;

    if (!ret)
    {
        return false;
    }

    for (int i = 0; i < outputTensors.size(); i++)
    {
        if (invalidMask[i / BITS_IN_UINT32] & (1 << (i % BITS_IN_UINT32)))
        {
            if (forbidInvalid) return false;
            continue;
        }

        if (outputTensors[i]->isHost2DeviceTensor())
        {
            // we should not set the size, SIF sets the data
            continue;
        }

        // check getTensorType() rather than isDynamicShape() because
        // isDynamicShape() is not well-defined until after SIF pass
        // has executed

        auto expectedSizes =
            inferMax ? outputTensors[i]->getNSizesInElements() : outputTensors[i]->getNMinimalSizesInElements();
        auto expectedSizesBegin = expectedSizes.begin();
        auto expectedSizesEnd   = expectedSizesBegin + outputTensors[i]->getDim();
        auto inferredSizesBegin = outTensorShapes[i].geometry.maxSizes;
        auto inferredSizesEnd   = outTensorShapes[i].geometry.maxSizes + outputTensors[i]->getDim();
        auto sizesEqual = std::equal(expectedSizesBegin, expectedSizesEnd, outTensorShapes[i].geometry.maxSizes);

        LOG_DEBUG(GC,
                  "Node {} GUID {} tensor {}: inferred sizes [{}], expected sizes [{}] {}",
                  m_name,
                  getGUID(),
                  outputTensors[i]->getName(),
                  toString(inferredSizesBegin, inferredSizesEnd, ','),
                  toString(expectedSizesBegin, expectedSizesEnd, ','),
                  (GCFG_ENABLE_WIDE_BUCKET.value() && !sizesEqual) ? "but it's valid because wide bucket enabled" : "");

        if (outputTensors[i]->isPersistent() || (inferMax && outputTensors[i]->isPropSet(synTensorPropGeometryMax)) ||
            (!inferMax && GCFG_ENABLE_SIF_FOR_STATIC_NODES.value() && !isDynamicShape()))
        {
            HB_ASSERT(outputTensors[i]->isPropSet(synTensorPropGeometryMax),
                      "Persistent tensor geomerty inference is not supported as they are used for validation");

            HB_ASSERT(outputTensors[i]->getDim() <= tpc_lib_api::MAX_TENSOR_DIM,
                      "Unsupported dim {} tensor {}",
                      outputTensors[i]->getDim(),
                      outputTensors[i]->getName());

            // this is a static tensor but we are running a sif...
            if ((!GCFG_ENABLE_WIDE_BUCKET.value() || inferMax) && !sizesEqual)
            {
                LOG_ERR(GC,
                        "Node {} GUID {}: shape inference for a persistent static tensor {} returns an inconsistent "
                        "result, "
                        "inferred sizes [{}], expected sizes [{}]",
                        m_name,
                        getGUID(),
                        outputTensors[i]->getName(),
                        toString(inferredSizesBegin, inferredSizesEnd, ','),
                        toString(expectedSizesBegin, expectedSizesEnd, ','));
                return false;
            }
            continue;
        }
        else
        {
            // convert glue-code size array type to tensors's size array type
            NSizeArray tSizes;
            memcpy(tSizes.data(), outTensorShapes[i].geometry.maxSizes, tpc_lib_api::MAX_TENSOR_DIM * sizeof(TSize));

            if (inferMax)
            {
                if (!sizesEqual)
                {
                    HB_ASSERT(outputTensors[i]->isTrivialStrided(),
                              "Node {} tensor {} cannot reshape a non trivially strided tensor!",
                              m_name,
                              outputTensors[i]->getName());
                    outputTensors[i]->reshape(outTensorShapes[i].geometry.dims, tSizes.data());
                }
                outputTensors[i]->setProp(synTensorPropGeometryMax);
            }
            else
            {
                outputTensors[i]->setMinSize(tSizes.data());

                if (GCFG_ENABLE_WIDE_BUCKET.value())
                {
                    for (unsigned d = 0; d < outTensorShapes[i].geometry.dims; ++d)
                    {
                        if (outputTensors[i]->getSizeInElements(d) > outputTensors[i]->getMinimalSizeInElements(d))
                        {
                            tSizes[d] = 0;
                        }
                    }
                    outputTensors[i]->setMinSize(tSizes.data());
                }

                outputTensors[i]->setProp(synTensorPropGeometryMin);
            }
        }
    }
    if (!inferMax)
    {
        for (auto& srcDstTensors : getShapeNode()->getPostSifUpdates())
        {
            const TensorPtr& src = srcDstTensors.first;
            const TensorPtr& dst = srcDstTensors.second;
            dst->setMinSize(src->getAllMinimalSizesInElements().data());
            dst->setProp(synTensorPropGeometryMin);
        }
    }

    return true;
}

bool Node::runShapeInferenceFunction(synDeviceType deviceType,
                                     SifParams*    params,
                                     SifOutputs*   outputs,
                                     bool          inferMax,
                                     bool          skipStatic)
{
    const sm_function_id_t sifId = getShapeInferenceFunctionId(skipStatic);
    sif_t sif;
    std::tie(sif, params->pGuid) = ShapeFuncRegistry::instance().getSIFandGuidInfo(sifId);

    CHECK_RET_FALSE(sif != nullptr, "Missing SIF for node {} (guid: {})", m_name, getGUID());

    // TODO: remove this when [SW-159125] is done
    // pGuid can be nullptr if not a dynamic node, so guid is not needed in this case
    if (params->pGuid)
    {
        params->guid = *params->pGuid;
    }
    params->maxAvailableTpc = TPCNode::getMaxAvailableTpc(deviceTypeToDeviceID(deviceType));

    SifReturn ret = sif(deviceTypeToDeviceID(deviceType), params, outputs);

    if (ret != tpc_lib_api::GLUE_SUCCESS)
    {
        LOG_ERR(GC, "Running shape inference for node {} guid {} error {}", m_name, getGUID(), enumToString(ret));
        return false;
    }

    return true;
}

std::vector<Node::NodeDynamicShapeProjection> Node::getDynamicShapeProjectionsTensors() const
{
    LOG_TRACE(GC, "Node::getDynamicShapeProjectionsTensors node {} guid {}", m_name, getGUID());
    return std::vector<Node::NodeDynamicShapeProjection>();
}

bool Node::requiresOutputMaxDimInfer() const
{
    const auto& outputs = getOutputs();
    return std::any_of(outputs.begin(), outputs.end(), [](const TensorPtr& t) {
        return t && (!t->isPropSet(synTensorPropGeometryMax) || t->isHost2DeviceTensor());
    });
}

void Node::permuteParams(const PermutationVector& inputPermutations)
{
    HB_ASSERT(m_paramsRawData.empty(),
              "Node {} with type {} has params and should override permuteParams function",
              getNodeName(),
              getNodeTypeStr());
}

void Node::updateCache()
{
    if (m_nodeAccessPatternCache)
    {
        // Update the content of the cache without changing the pointer,
        // since this pointer might be shared in other objects.
        auto newAP = generateNodeAccessPattern();
        HB_ASSERT_PTR(newAP);
        *m_nodeAccessPatternCache = *newAP;
    }
}

static void gcPermutationsToSif(const PermutationVector& in, std::vector<SifPermutation>& out)
{
    bool allIdentity = true;
    for (unsigned i = 0; i < in.size() && allIdentity; ++i)
    {
        if (!in[i].isIdentity())
        {
            allIdentity = false;
        }
    }

    if (allIdentity)
    {
        out.clear();
        return;
    }

    out.resize(in.size());
    for (unsigned i = 0; i < out.size(); ++i)
    {
        auto& outPerm = out[i];
        std::iota(std::begin(outPerm.permutation), std::end(outPerm.permutation), 0);
    }

    // copy from annotations to sif permutations
    for (unsigned i = 0; i < in.size(); ++i)
    {
        auto&       outPerm = out[i];
        const auto& inPerm  = in[i];
        HB_ASSERT(std::size(inPerm.getValues()) <= std::size(outPerm.permutation),
                  "Input permutation is too big: {} elements",
                  std::size(inPerm.getValues()));
        std::copy(std::begin(inPerm.getValues()), std::end(inPerm.getValues()), std::begin(outPerm.permutation));
    }
}

std::vector<SifPermutation> Node::getInputPermutations() const
{
    std::vector<SifPermutation> res;
    gcPermutationsToSif(getNodeAnnotation().inputPermutations, res);
    return res;
}

HabanaDeviceType Node::getNodeDeviceType() const
{
    return LAST_HABANA_DEVICE;
}

// remove after [SW-93826] is done (?)
void Node::setParams(UserParams userParams, unsigned userParamsSize)
{
    setParamsRawData(userParams, userParamsSize);
}

bool Node::isGemmNode(NodePtr n)
{
    Node::eNodeType type = n->getNodeType();
    return (type == Node::TYPE_GEMM || type == Node::TYPE_GEMM_DEDX || type == Node::TYPE_GEMM_DEDW);
}

// TODO SW-61420 - replace calling to isBatchGemm member with this func
bool Node::isBatchGemmNode(NodePtr n)
{
    Node::eNodeType type = n->getNodeType();
    return (type == Node::TYPE_BATCH_GEMM || type == Node::TYPE_BATCH_GEMM_DEDX || type == Node::TYPE_BATCH_GEMM_DEDW ||
            type == Node::TYPE_MASKED_BATCH_GEMM);
}

bool Node::isDedxNode(NodePtr n)
{
    Node::eNodeType type = n->getNodeType();
    return (type == Node::TYPE_DEDX || type == Node::TYPE_TRANSPOSED_DEDX);
}

bool Node::isForkNode(const NodePtr& n)
{
    return (n->getNodeType() == Node::TYPE_TENSOR_VIEW || n->getNodeType() == Node::TYPE_TENSOR_VIEW_SHAPE_NODE) &&
           static_cast<TensorViewNode*>(n.get())->realTensorIsInput();
}

bool Node::isJoinNode(const NodePtr& n)
{
    return (n->getNodeType() == Node::TYPE_TENSOR_VIEW || n->getNodeType() == Node::TYPE_TENSOR_VIEW_SHAPE_NODE) &&
           !static_cast<TensorViewNode*>(n.get())->realTensorIsInput();
}
