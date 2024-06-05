#include <perf_lib_layer_params.h>
#include "quantizer.h"
#include "types.h"
#include "quant_info_calculator.h"
#include "habana_global_conf.h"
#include "infra/defs.h"
#include "data_type_utils.h"

// Dummy dynamic range used for skipping enforceNodePrecision pass
void setDummyDynamicRange(const TensorPtr& tensor)
{
    if (!tensor->getDynamicRange().isSet && is8BitFloat(tensor->getElementType()))
    {
        DynamicRange dynamicRange;
        dynamicRange.min   = 0;
        dynamicRange.max   = 1;
        dynamicRange.isSet = true;
        tensor->setDynamicRange(dynamicRange);
    }
}

const int Quantizer::index_na = -1;

bool Quantizer::isConflictedWith(TensorPtr tensor, QuantizationMap& newQuant)
{
    QuantizationMap tensorQuant = tensor->getAllQuantizationParams();
    if (tensorQuant.size() != newQuant.size())
    {
        return true;
    }

    // compare the two quantization maps
    for (int i = 0; i < quant_type_max; i++)
    {
        eQuantDataType type = (eQuantDataType)i;
        int tensorQuantExist = 0;
        int newQuantExist = 0;

        if (tensorQuant.find(type) != tensorQuant.end())
        {
            tensorQuantExist = 1;
        }

        if (newQuant.find(type) != newQuant.end())
        {
            newQuantExist = 1;
        }

        if (!tensorQuantExist & !newQuantExist)
        {
            // quantization doesn't exist in this dtype
            continue;
        }

        if (tensorQuantExist ^ newQuantExist)
        {
            // maps are not identical in keys
            return true;
        }

        if (!(newQuant[type] == tensorQuant[type]))
        {
            // quant values are not identical
            return true;
        }
    }

    return false;
}

QuantizationMap Quantizer::getSingleScaleFromTensors(TensorVector tensors, int index)
{
    QuantizationMap result;

    HB_ASSERT(index < (int)tensors.size(), "Quantizer invalid index");
    if (index != Quantizer::index_na && index < (int)tensors.size())
    {
        tensors = {tensors[index]};
    }

    for (const TensorPtr& tensor : tensors)
    {
        if (tensor == nullptr)
        {
            continue;
        }

        QuantizationMap qinfo = tensor->getAllQuantizationParams();
        for (const std::pair<const uint32_t, QuantizationData>& quantization : qinfo)
        {
            if (!(tensor->getDynamicRange().isSet || quantization.second.m_isUserQuantInfo ||
                  quantization.second.m_isUserPCQuantInfo))
            {
                continue;
            }
            // all tensors should have the same scale in the same dtype, if not, take the last scale found.
            if (result.find(quantization.first) != result.end() &&
                !(result[quantization.first] == quantization.second))
            {
                LOG_WARN(QUANT,
                         "{}: tensor {} has different scale in dtype {}, choosing this scale.",
                         HLLOG_FUNC,
                         tensor->getName(),
                         quantization.first);
            }

            result[quantization.first] = quantization.second;
        }
    }
    return result;
}

bool Quantizer::isAllLocked(TensorVector& tensors)
{
    HB_ASSERT(tensors.size() > 0, "Tensors vector is empty");
    for (const TensorPtr& tensor : tensors)
    {
        if (tensor == nullptr) continue;
        if (!tensor->isLocked()) return false;
    }
    return true;
}

bool Quantizer::isQuantMapEmpty(QuantizationMap& quantMap)
{
    if (!quantMap.empty())
    {
        for (const std::pair<const uint32_t, QuantizationData>& quantization : quantMap)
        {
            if (quantization.first != quant_type_na) return false;
        }
    }

    return true;
}

void Quantizer::revertAndRequantLock(TensorPtr inputTensor, NodePtr inputNode, NodePtr node, QuantizationMap& quantMap)
{
    // save current quantization map with this node's name
    inputTensor->saveConflicts(node, quantMap);
    // set quantization to the measured quantization
    inputTensor->revertQuantization();
    // lock for requant using the input node's name
    inputTensor->requantLock(inputNode);
}

void Quantizer::setInputScale(HabanaGraph& g, NodePtr node, QuantizationMap& quantInfo,
                              std::vector<uint32_t> numSuccessorsPerInput, int index)
{
    TensorVector inputTensors = node->getInputs();
    HB_ASSERT(numSuccessorsPerInput.size() == inputTensors.size(), "Input vectors size mismatch");

    HB_ASSERT(index < (int)inputTensors.size(), "Quantizer invalid index");
    if (index != Quantizer::index_na && index < (int)inputTensors.size())
    {
        inputTensors = {inputTensors[index]};
        numSuccessorsPerInput = {numSuccessorsPerInput[index]};
    }

    bool emptyQuantMap = isQuantMapEmpty(quantInfo);
    for (int i = 0; i < inputTensors.size(); i++)
    {
        TensorPtr inputTensor = inputTensors[i];
        if (inputTensor == nullptr) continue;
        std::string_view              guid         = extractGUIDFromFullGUID(node->getGUID());
        if (guid != "convert_b_to_t" && numSuccessorsPerInput[i] > 1 )
        {
            revertAndRequantLock(inputTensor, g.getTensorProducer(inputTensor), node, quantInfo);
            continue;
        }

        if (!emptyQuantMap)
        {
            if (inputTensor->isLocked() && isConflictedWith(inputTensor, quantInfo))
            {
                revertAndRequantLock(inputTensor, g.getTensorProducer(inputTensor), node, quantInfo);
                continue;
            }
            else
            {
                inputTensor->setAllQuantizationParams(quantInfo);
            }
            // TODO - [SW-41255] adjust enforceNodePrecision pass
            setDummyDynamicRange(inputTensor);
        }

        inputTensor->lockQuantization(node);
    }
}

void Quantizer::lockTensors(NodePtr node, TensorVector& tensors)
{
    std::string nodeName = node->getNodeName();

    for (const TensorPtr& tensor : tensors)
    {
        if (tensor == nullptr) continue;
        tensor->lockQuantization(node);
    }
}

void Quantizer::setOutputScale(NodePtr node, QuantizationMap& quantInfo, int index)
{
    TensorVector outputTensors = node->getOutputs();

    HB_ASSERT(index < (int)outputTensors.size(), "Quantizer invalid index");
    if (index != Quantizer::index_na && index < (int)outputTensors.size())
    {
        outputTensors = {outputTensors[index]};
    }

    bool emptyQuantMap = isQuantMapEmpty(quantInfo);
    for (const TensorPtr& outputTensor : outputTensors)
    {
        if (outputTensor == nullptr) continue;
        if (!emptyQuantMap)
        {
            if (outputTensor->isLocked() && isConflictedWith(outputTensor, quantInfo) && !outputTensor->isRequantLocked())
            {
                // if not requant locked, save conflicts and requant lock.
                outputTensor->saveConflicts();
                outputTensor->setAllQuantizationParams(quantInfo);
                outputTensor->requantLock(node);
            }
            else
            {
                // if already requant locked, just change the quantization to the correct one
                outputTensor->setAllQuantizationParams(quantInfo);
            }
            // TODO - [SW-41255] adjust enforceNodePrecision pass
            setDummyDynamicRange(outputTensor);
        }
        outputTensor->lockQuantization(node);
    }
}

bool Quantizer::shouldEnforceFixedPoint(TensorVector tensors, int index)
{
    HB_ASSERT(index < (int)tensors.size(), "Quantizer invalid index");
    if (index != Quantizer::index_na && index < (int)tensors.size())
    {
        tensors = {tensors[index]};
    }

    for (const TensorPtr& t : tensors)
    {
        if (t == nullptr) continue;
        if (t->isInt8FixedPoint())
            return true;
    }

    return false;
}

bool Quantizer::shouldEnforceInt16Ltd(TensorVector tensors, int index)
{
    HB_ASSERT(index < (int)tensors.size(), "Quantizer invalid index");
    if (index != Quantizer::index_na && index < (int)tensors.size())
    {
        tensors = {tensors[index]};
    }

    for (const TensorPtr& t : tensors)
    {
        if (t == nullptr) continue;
        if (t->isInt16Limited()) return true;
    }

    return false;
}

void Quantizer::enforceFixedPoint(TensorVector tensors, int index)
{
    HB_ASSERT(index < (int)tensors.size(), "Quantizer invalid index");
    if (index != Quantizer::index_na && index < tensors.size())
    {
        tensors = {tensors[index]};
    }

    for (const TensorPtr& t : tensors)
    {
        t->enforceInt8FixedPoint();
    }
}

void Quantizer::enforceInt16Ltd(TensorVector tensors, int index)
{
    HB_ASSERT(index < (int)tensors.size(), "Quantizer invalid index");
    if (index != Quantizer::index_na && index < tensors.size())
    {
        tensors = {tensors[index]};
    }

    for (const TensorPtr& t : tensors)
    {
        if (t == nullptr) continue;
        t->enforceInt16Limited();
    }
}

void Quantizer::correctSpecialIntQuantization(TensorVector& tensors)
{
    for (const TensorPtr& t : tensors)
    {
        if (t == nullptr) continue;

        if (t->isInt16Limited())
        {
            t->enforceInt16Limited();
        }

        if (t->isInt8FixedPoint())
        {
            t->enforceInt8FixedPoint();
        }
    }
}

void Quantizer::adjustRestrictions(NodePtr node, bool isForwardPass)
{

}

void Quantizer::adjustScales(HabanaGraph& g, NodePtr node, bool isForwardPass, std::vector<uint32_t>& numSuccessorsPerInput)
{
    if (!isForwardPass) return;
    LOG_TRACE(QUANT, "Locking {} scales", node->getNodeName());
    TensorVector inputs = node->getInputs();
    TensorVector outputs = node->getOutputs();
    lockTensors(node, inputs);
    lockTensors(node, outputs);
}

/** Backward Quantizer **/
void BackwardQuantizer::adjustRestrictions(NodePtr node, bool isForwardPass)
{

}

void BackwardQuantizer::adjustScales(HabanaGraph& g, NodePtr node, bool isForwardPass, std::vector<uint32_t>& numSuccessorsPerInput)
{
    if (isForwardPass) return;
    TensorVector outputs = node->getOutputs();
    LOG_TRACE(QUANT, "Locking {} output scales", node->getNodeName());
    lockTensors(node, outputs);
    QuantizationMap outputQuantization = getSingleScaleFromTensors(outputs);
    LOG_TRACE(QUANT, "Assign {} output scales to input scales", node->getNodeName());
    setInputScale(g, node, outputQuantization, numSuccessorsPerInput);
}

/** Backward Don't Care Quantizer **/
void BackwardDontCareQuantizer::adjustScales(HabanaGraph& g, NodePtr node, bool isForwardPass, std::vector<uint32_t>& numSuccessorsPerInput)
{
    TensorVector inputs = node->getInputs();
    TensorVector outputs = node->getOutputs();
    if (isForwardPass) return;
    LOG_TRACE(QUANT, "Locking {} output scales", node->getNodeName());
    lockTensors(node, outputs);
    LOG_TRACE(QUANT, "Locking {} input scales", node->getNodeName());
    lockTensors(node, inputs);
}

/** Selective Backward Quantizer **/
SelectiveBackwardQuantizer::SelectiveBackwardQuantizer(std::map<uint32_t, uint32_t>& outputToInputIndexMap)
{
    m_outputToInputIndexMap = outputToInputIndexMap;
}

void SelectiveBackwardQuantizer::adjustRestrictions(NodePtr node, bool isForwardPass)
{

}

void SelectiveBackwardQuantizer::adjustScales(HabanaGraph& g, NodePtr node, bool isForwardPass, std::vector<uint32_t>& numSuccessorsPerInput)
{
    TensorVector inputs = node->getInputs();
    TensorVector outputs = node->getOutputs();
    if (!isForwardPass)
    {
        // during backward pass, lock output scale and enforce the specified output scales on the specified inputs
        LOG_TRACE(QUANT, "Locking {} output scales", node->getNodeName());
        lockTensors(node, outputs);
        for (const std::pair<const uint32_t, uint32_t>& p : m_outputToInputIndexMap)
        {
            LOG_TRACE(QUANT, "Assign {} output {} scale to input {} scales", node->getNodeName(), p.first, p.second);
            QuantizationMap outputQuantization = getSingleScaleFromTensors(outputs, p.first);
            setInputScale(g, node, outputQuantization, numSuccessorsPerInput, p.second);
        }
    }
    else
    {
        // during forward pass, lock the rest of the inputs
        LOG_TRACE(QUANT, "Locking {} input scales", node->getNodeName());
        lockTensors(node, inputs);
    }
}

/** Forward Quantizer **/
ForwardQuantizer::ForwardQuantizer() : m_specificInput(Quantizer::index_na)
{
}

ForwardQuantizer::ForwardQuantizer(int specificInput)
{
    m_specificInput = specificInput;
}

void ForwardQuantizer::adjustRestrictions(NodePtr node, bool isForwardPass)
{

}

void ForwardQuantizer::adjustScales(HabanaGraph& g, NodePtr node, bool isForwardPass, std::vector<uint32_t>& numSuccessorsPerInput)
{
    if (!isForwardPass) return;
    TensorVector inputs = node->getInputs();
    LOG_TRACE(QUANT, "Locking {} input scales", node->getNodeName());
    lockTensors(node, inputs);
    QuantizationMap inputQuantization = getSingleScaleFromTensors(inputs, m_specificInput);
    LOG_TRACE(QUANT, "Assign {} input scales to output scales", node->getNodeName());
    setOutputScale(node, inputQuantization);
}

/** Align Inputs Quantizer **/
void AlignInputsQuantizer::adjustRestrictions(NodePtr node, bool isForwardPass)
{

}

void AlignInputsQuantizer::adjustScales(HabanaGraph& g, NodePtr node, bool isForwardPass, std::vector<uint32_t>& numSuccessorsPerInput)
{
    if (isForwardPass) return;
    TensorVector inputs = node->getInputs();
    TensorVector outputs = node->getOutputs();

    // lock output scales, find the max scale and enforce it on all inputs
    LOG_TRACE(QUANT, "Locking {} output scales", node->getNodeName());
    lockTensors(node, outputs);
    QuantizationMap maxQuantMap;
    double maxScale = 0;
    for (const TensorPtr& input : inputs)
    {
        if (input == nullptr) continue;
        QuantizationData inputQuant = input->getQuantizationParams();
        if (maxScale < inputQuant.scale())
        {
            maxScale = inputQuant.scale();
            maxQuantMap = input->getAllQuantizationParams();
        }
    }
    HB_ASSERT(maxScale != 0, "Max quantization map was not set properly");
    LOG_TRACE(QUANT, "Assign {} input scales to the maximum scale of all inputs", node->getNodeName());
    setInputScale(g, node, maxQuantMap, numSuccessorsPerInput);
}

/** Don't Care Quantizer **/
void DontCareQuantizer::adjustRestrictions(NodePtr node, bool isForwardPass)
{

}

void DontCareQuantizer::adjustScales(HabanaGraph& g, NodePtr node, bool isForwardPass, std::vector<uint32_t>& numSuccessorsPerInput)
{
    TensorVector inputs = node->getInputs();
    TensorVector outputs = node->getOutputs();

    if (isForwardPass && !isAllLocked(outputs))
    {
        // if already adjusted in backward pass, don't adjust in forward pass,
        // because a requant node may be implanted after adjustment
        // pass scale from inputs to outputs
        LOG_TRACE(QUANT, "Locking {} input scales", node->getNodeName());
        lockTensors(node, inputs);
        QuantizationMap inputQuantization = getSingleScaleFromTensors(inputs);
        LOG_TRACE(QUANT, "Assign {} input scales to output scales", node->getNodeName());
        setOutputScale(node, inputQuantization);
    }
    else if (!isForwardPass && isAllLocked(outputs))
    {
        // pass scale from outputs to inputs, if outputs are locked.
        LOG_TRACE(QUANT, "all {} outputs are locked, set output scale to inputs", node->getNodeName());
        QuantizationMap outputQuantization = getSingleScaleFromTensors(outputs);
        setInputScale(g, node, outputQuantization, numSuccessorsPerInput);
    }
}

/** Boolean Output Quantizer **/
void BooleanOutputQuantizer::adjustRestrictions(NodePtr node, bool isForwardPass)
{
    if (isForwardPass)
    {
        for (const TensorPtr& output : node->getOutputs())
        {
            if (output == nullptr) continue;
            output->setAllQuantizationParams(QuantInfoCalculator::basicQuantizationMap(output));
            output->lockQuantization(node);

            // TODO - [SW-41255] adjust enforceNodePrecision pass
            setDummyDynamicRange(output);
        }
    }

    AlignInputsQuantizer::adjustRestrictions(node, isForwardPass);
}

/** Boolean Input Quantizer **/
void BooleanInputQuantizer::adjustRestrictions(NodePtr node, bool isForwardPass)
{
    if (isForwardPass)
    {
        for (const TensorPtr& input : node->getInputs())
        {
            if (input == nullptr) continue;
            input->setAllQuantizationParams(QuantInfoCalculator::basicQuantizationMap(input));
            input->lockQuantization(node);

            // TODO - [SW-41255] adjust enforceNodePrecision pass
            setDummyDynamicRange(input);
        }
    }

    BooleanOutputQuantizer::adjustRestrictions(node, isForwardPass);
}

/** Cast Quantizer **/
void CastQuantizer::adjustRestrictions(NodePtr node, bool isForwardPass)
{
    // if this is a user cast, set default quant params to output tensor so it will behave like cpp cast.
    if (isForwardPass && !node->getNodeAnnotation().insertedNode)
    {
        for (const TensorPtr& output : node->getOutputs())
        {
            if (output == nullptr) continue;
            if (isQuantDtype(getDtypeSuffixFromSynDataType(output->getElementType())) &&
                (output->getDynamicRange().isSet || output->getQuantizationParams().m_isUserQuantInfo))
            {
                // skip reset QuantizationParams in CastQuantizer::adjustRestrictions
                // if it is a cast to fp8 and the quantization params were already set
                continue;
            }
            output->setAllQuantizationParams(QuantInfoCalculator::basicQuantizationMap(output));
            output->lockQuantization(node);

            // TODO - [SW-41255] adjust enforceNodePrecision pass
            setDummyDynamicRange(output);
        }
    }

    Quantizer::adjustRestrictions(node, isForwardPass);
}

/** Top K Quantizer **/
const unsigned TopKQuantizer::indicesInput = 1;

void TopKQuantizer::adjustRestrictions(NodePtr node, bool isForwardPass)
{
    if (!isForwardPass)
    {
        // indices output must have zp 0 scale 1
        const TensorPtr& output = node->getOutput(TopKQuantizer::indicesInput);
        if (output == nullptr) return;
        output->setAllQuantizationParams(QuantInfoCalculator::basicQuantizationMap(output));
        output->lockQuantization(node);

        // TODO - [SW-41255] adjust enforceNodePrecision pass
        setDummyDynamicRange(output);
    }

    Quantizer::adjustRestrictions(node, isForwardPass);
}

/** Sequence Reverse Quantizer **/
const unsigned SequenceReverseQuantizer::sequenceLensInput = 1;

void SequenceReverseQuantizer::adjustRestrictions(NodePtr node, bool isForwardPass)
{
    if (isForwardPass)
    {
        HB_ASSERT(node && HabanaGraph::runsOnTPC(node), "expected TPC node");
        auto& tpcNode = static_cast<TPCNode&>(*node);

        HB_ASSERT(tpcNode.getParams() != nullptr, "sequence reverse params are null");
        if (static_cast<ns_SequenceLength::Params*>(tpcNode.getParams())->use_sequence_length)
        {
            // sequence length input must have zp 0 scale 1
            const TensorPtr& input = node->getInput(SequenceReverseQuantizer::sequenceLensInput);
            if (input == nullptr) return;
            input->setAllQuantizationParams(QuantInfoCalculator::basicQuantizationMap(input));
            input->lockQuantization(node);

            // TODO - [SW-41255] adjust enforceNodePrecision pass
            setDummyDynamicRange(input);
        }
    }

    Quantizer::adjustRestrictions(node, isForwardPass);
}

/** Sequence Length Quantizer **/
void SequenceLengthQuantizer::adjustRestrictions(NodePtr node, bool isForwardPass)
{
    if (!isForwardPass)
    {
        // all inputs and outputs must have zp 0 scale 1
        for (const TensorPtr& input : node->getInputs())
        {
            if (input == nullptr) continue;
            input->setAllQuantizationParams(QuantInfoCalculator::basicQuantizationMap(input));
            input->lockQuantization(node);

            // TODO - [SW-41255] adjust enforceNodePrecision pass
            setDummyDynamicRange(input);
        }

        for (const TensorPtr& output : node->getOutputs())
        {
            if (output == nullptr) continue;
            output->setAllQuantizationParams(QuantInfoCalculator::basicQuantizationMap(output));
            output->lockQuantization(node);

            // TODO - [SW-41255] adjust enforceNodePrecision pass
            setDummyDynamicRange(output);
        }
    }

    BackwardQuantizer::adjustRestrictions(node, isForwardPass);
}

/** Embedding Quantizer **/
const unsigned EmbeddingQuantizer::dataInput = 0;
const unsigned EmbeddingQuantizer::indicesInput = 1;
const unsigned EmbeddingQuantizer::dataOutput = 0;

EmbeddingQuantizer::EmbeddingQuantizer()
{
    m_byWeightsQuantizer = std::make_shared<ForwardQuantizer>(EmbeddingQuantizer::dataInput);
    std::map<uint32_t, uint32_t> embeddingSelectiveMap;
    embeddingSelectiveMap[EmbeddingQuantizer::dataOutput] = EmbeddingQuantizer::dataInput;
    m_byOutputQuantizer = std::make_shared<SelectiveBackwardQuantizer>(embeddingSelectiveMap);
}

QuantizerPtr EmbeddingQuantizer::getEmbeddingQuantizer()
{
    if (GCFG_EMBED_BY_WEIGHTS.value())
    {
        return m_byWeightsQuantizer;
    }
    else
    {
        return m_byOutputQuantizer;
    }
}

void EmbeddingQuantizer::adjustRestrictions(NodePtr node, bool isForwardPass)
{
    if (!isForwardPass)
    {
        // indices input must have zp 0 scale 1
        const TensorPtr& input = node->getInput(EmbeddingQuantizer::indicesInput);
        if (input == nullptr) return;
        if (input->getElementType() != syn_type_int32 && input->getElementType() != syn_type_int16)
        {
            changeTensorElementTypeSafe(input, syn_type_int32);
        }

        input->setAllQuantizationParams(QuantInfoCalculator::basicQuantizationMap(input));
        input->lockQuantization(node);

        // TODO - [SW-41255] adjust enforceNodePrecision pass
        setDummyDynamicRange(input);
    }

    getEmbeddingQuantizer()->adjustRestrictions(node, isForwardPass);
}

void EmbeddingQuantizer::adjustScales(HabanaGraph& g, NodePtr node, bool isForwardPass, std::vector<uint32_t>& numSuccessorsPerInput)
{
    getEmbeddingQuantizer()->adjustScales(g, node, isForwardPass, numSuccessorsPerInput);
}
