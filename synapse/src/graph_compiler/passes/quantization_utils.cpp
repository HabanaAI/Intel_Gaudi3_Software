#include "quantization_utils.h"

#include "habana_nodes.h"
#include "utils.h"
#include "graph_editor.h"
#include "quantization_data.h"
#include "json_utils.h"
#include "node_utils.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

namespace QuantizationUtils
{
// store exp bias with its corresponding maxAbs value
using ExpBiasMaxAbsPair = std::pair<unsigned, double>;

constexpr double maxFp8152      = 1879048192;
constexpr double maxFp8143      = 3840;
constexpr double maxFp8143Exp7  = 240;
constexpr double maxFp8143Exp15 = 0.9375;
constexpr double minRangeScale  = 0.46875;

constexpr const char* INPUTS = "inputs";
constexpr const char* OUTPUTS = "outputs";
constexpr const char* WEIGHTS = "weights";
constexpr const char* PARAMS = "params";

// tables to store exponent bias and corresponding max abs values, used for exp bias calculation.
constexpr std::array<ExpBiasMaxAbsPair, 32> expBiasTableFp8152 {
    {{31, 0.875},    {30, 1.75},     {29, 3.5},      {28, 7},       {27, 14},      {26, 28},      {25, 56},
     {24, 112},      {23, 224},      {22, 448},      {21, 896},     {20, 1792},    {19, 3584},    {18, 7168},
     {17, 14336},    {16, 28672},    {15, 57344},    {14, 114688},  {13, 229376},  {12, 458752},  {11, 917504},
     {10, 1835008},  {9, 3670016},   {8, 7340032},   {7, 14680064}, {6, 29360128}, {5, 58720256}, {4, 117440512},
     {3, 234881024}, {2, 469762048}, {1, 939524096}, {0, maxFp8152}}};
constexpr std::array<ExpBiasMaxAbsPair, 4> expBiasTableFp8143 {{{15, 0.9375}, {11, 15}, {7, 240}, {3, maxFp8143}}};
std::unordered_map<double, unsigned> scaleToExpBiasTableFp8143 = {{{16, 3}, {1, 7}, {0.0625, 11}, {0.00390625, 15}}};
std::unordered_map<double, unsigned> invScaleToExpBiasTableFp8143 = {{{0.0625, 3}, {1, 7}, {16, 11}, {256, 15}}};

constexpr double MIN_SCALE = 1e-7;

constexpr unsigned FP8_GEMM_SCALE_IFM_IDX = 2;
constexpr unsigned FP8_GEMM_SCALE_WEIGHT_IDX = 3;
constexpr unsigned FP8_CONV_2D_SCALE_IFM_IDX = 3;
constexpr unsigned FP8_CONV_2D_SCALE_WEIGHT_IDX = 4;

/*
 * Return true if the given dimension is the FCD in synapse convention -
 * A dimension is the SCD if all following dimensions are size 1.
 */
bool isSCD(const SizeArray& sizes, const unsigned dimI)
{
    for (unsigned i = 0; i < sizes.size(); i++)
    {
        if (i > dimI && sizes[i] > 1)
        {
            return false;
        }
    }
    return true;
}

/*
 * Gets the channel indices according to the kIndex and the channel number.
 */
std::set<unsigned> getChannelIndices(const SizeArray& sizes,
                                     const unsigned   kIndex,
                                     const unsigned   channelNum,
                                     bool             isSCD,
                                     unsigned long    spatialElementsNum)
{
    std::set<unsigned> indexSet;

    if (isSCD)
    {
        const unsigned baseOffset = channelNum * spatialElementsNum;
        const unsigned maxOffset  = (channelNum + 1) * spatialElementsNum;

        for (unsigned i = baseOffset; i < maxOffset; i++)
        {
            indexSet.insert(i);
        }
    }
    else if (kIndex == 0)  // K is FCD
    {
        unsigned kSize = sizes[kIndex];

        for (unsigned i = 0; i < spatialElementsNum; ++i)
        {
            indexSet.insert(channelNum + kSize * i);
        }
    }
    else  // fallback to general kIndex case
    {
        indexSet = dimIndexArray(sizes, kIndex, channelNum);
    }

    return indexSet;
}

/**
    Get the minimum value of an array in given indices

    @param pRealArray pointer
    @param indices set of indices of the real array to calculate
    @return minimum value in array
*/
template<typename T>
T getMinValue(T* pRealArray, std::set<unsigned>& indices)
{
    T min = pRealArray[*(indices.begin())];
    for (unsigned i : indices)
    {
        if (min > pRealArray[i])
        {
            min = pRealArray[i];
        }
    }

    return min;
}

/**
    Get the maximum value of an array in given indices

    @param pRealArray pointer
    @param indices set of indices of the real array to calculate
    @return maximum value in array
*/
template<typename T>
T getMaxValue(T* pRealArray, std::set<unsigned>& indices)
{
    T max = pRealArray[*(indices.begin())];
    for (unsigned i : indices)
    {
        if (max < pRealArray[i])
        {
            max = pRealArray[i];
        }
    }

    return max;
}

// TODO[c++20]: use std::span instead of templating this function
template<size_t N>
static unsigned getExpBiasForValue(double value, const std::array<ExpBiasMaxAbsPair, N>& table, double backoffFactor)
{
    // take expBias for max range: 3 for fp8143, 0 for fp8152
    unsigned expBias = table[table.size()-1].first;

    // iterate table to find the first element which its maxAbs >= (value * backoff factor) and get its expBias
    for (unsigned i = 0; i < N; i++)
    {
        if (value <= backoffFactor * table[i].second)
        {
            expBias = table[i].first;
            LOG_TRACE(QUANT,
                      "Found for value {} exponent bias {} for max abs value {} with backoff factor {}",
                      value,
                      expBias,
                      table[i].second,
                      backoffFactor);
            break;
        }
    }
    return expBias;
}

bool calcExpBias(double tensorMaxAbs, eQuantDataType dtype, unsigned& expBias, double backoffFactor)
{
    switch (dtype)
    {
        case quant_type_fp8_143:
            if (GCFG_ENABLE_ARBITRARY_SCALE.value())
            {
                expBias =  QuantizationData::S_EXP_BIAS_143_DEFAULT;
                break;
            }
            expBias = getExpBiasForValue(tensorMaxAbs, expBiasTableFp8143, backoffFactor);
            break;
        case quant_type_fp8_152:
            expBias = QuantizationData::S_EXP_BIAS_152_DEFAULT;
            break;
        default:
            HB_ASSERT(false,
                      "Unexpected quant data type {} for expBias calculation",
                      QuantizationData::getDataTypeString(dtype));
            return false;
    }
    return true;
}

// In the per channel case the expBias is the default for all channels
void calcPCScaleForFp8_143(float absMax, eQuantDataType dtype, double& scale, double backoffFactor)
{
    if (dtype == quant_type_fp8_143)
    {
        scale = absMax / (maxFp8143Exp7 * backoffFactor);
    }
}

void calcScaleForFp8_143(float absMax, QuantizationData& quantInfo, double backoffFactor)
{
    double scale = 1;
    if (quantInfo.m_qDataType == quant_type_fp8_143)
    {
        float val = absMax / backoffFactor;
        if (val > maxFp8143)
        {
            scale = val / maxFp8143;
        }
        else if (val < minRangeScale)
        {
            scale = val / maxFp8143Exp15;
        }
        if (GCFG_ENABLE_ARBITRARY_SCALE.value())
        {
            scale = val / maxFp8143Exp7;
        }
    }
    quantInfo.setScale(scale);
}

bool calcDataScaleZp(double minVal, double maxVal, eQuantDataType dtype, double& scale, double& zp,
                     bool isFixedPoint, bool isSparsityWeights)
{
    int  numBits   = 0;
    bool updatedZp = false;
    //TODO - SW-6009 Move number of bits to config file
    switch (dtype)
    {
        case quant_type_int4:
        case quant_type_uint4:
            numBits = 4;
            break;
        case quant_type_int8:
        case quant_type_uint8:
        case quant_type_fp8_143:
        case quant_type_fp8_152:
            numBits = 8;
            break;
        case quant_type_int16:
        case quant_type_uint16:
            numBits = 16;
            break;
        case quant_type_int32:
        case quant_type_uint32:
            numBits = 32;
            break;
        case quant_type_int16ltd:
            numBits = GCFG_INT16_LIMITED_BITS.value();
            if (numBits < 1 || numBits > 16)
            {
                LOG_ERR(QUANT, "invalid configuration value: GCFG_INT16_LIMITED_BITS={}", numBits);
                return false;
            }
            break;
        case quant_type_int8fp:
            isFixedPoint = true;
            numBits      = 8;
            break;
        default:
            LOG_ERR(QUANT, "Unknown data type {}", QuantizationData::getDataTypeString(dtype));
            return false;
    }

    double qMax = std::pow(2.f, numBits - 1.f);
    double qMin = -qMax;

    if (dtype == quant_type_uint16 || dtype == quant_type_uint8 || dtype == quant_type_uint4)
    {
        qMax *= 2;
        qMin = 0;
    }
    qMax -= 1;

    minVal = minVal > 0 ? 0 : minVal;
    maxVal = maxVal < 0 ? 0 : maxVal;

    minVal = minVal == -std::numeric_limits<float>::infinity() ? qMin : minVal;
    maxVal = maxVal == std::numeric_limits<float>::infinity() ? qMax : maxVal;

    double dynamicRange = 0;
    if (isFixedPoint || numBits > 8 ||
        (isSparsityWeights && dtype != quant_type_uint4 && dtype != quant_type_uint8))
    {
        zp        = 0;
        updatedZp = true;

        dynamicRange = std::max(std::abs(maxVal), std::abs(minVal));
        scale        = dynamicRange / qMax;
    }
    else
    {
        dynamicRange = maxVal - minVal;
        scale        = dynamicRange / (qMax - qMin);
    }

    if (allClose(scale, 0.0))
    {
        double minScale = scale >= 0.0 ? MIN_SCALE : -MIN_SCALE;
        LOG_INFO(QUANT, "Scale {}, is infinitesimal, scale will be set as {}, zp will be set as minimum range value {}",
                 scale, minScale, minVal);
        zp        = minVal;
        scale     = minScale;
        updatedZp = true;
    }

    if (!updatedZp)
    {
        zp = qMin - (minVal / scale);
        if (zp < qMin)
        {
            zp = qMin;
        }
        else if (zp > qMax)
        {
            zp = qMax;
        }
        else
        {
            double sign = zp < 0 ? -1 : 1;
            zp          = sign * std::floor(std::abs(zp) + 0.5);
        }

        double numIntervalsNegative = std::abs(qMin - zp);
        double numIntervalsPositive = std::abs(qMax - zp);
        double scaleNegative        = 0;
        if (numIntervalsNegative > 0)
        {
            scaleNegative = std::abs(minVal / numIntervalsNegative);
        }

        double scalePositive = 0;
        if (numIntervalsPositive > 0)
        {
            scalePositive = std::abs(maxVal / numIntervalsPositive);
        }

        scale = std::max(scalePositive, scaleNegative);
    }

    if (isFixedPoint || (numBits > 8 && !allClose(scale, 0.0)))
    {
        scale = std::pow(2.0, std::ceil(std::log2(scale)));
    }

    if (allClose(minVal, 0.0) && allClose(maxVal, 0.0))
    {
        LOG_INFO(QUANT, "Dynamic range ({}, {}) is infinitesimal, zp will be set as 0, scale will be set 1",
                 minVal, maxVal);
        scale = 1;
        zp    = 0;
    }

    return true;
}

bool hasBias(NodePtr& node)
{
    return (node->getInput(TENSOR_BIAS) != nullptr);
}

bool isFilter2d(const NodePtr& node)
{
    std::shared_ptr<TPCNode> tpcNode = std::dynamic_pointer_cast<TPCNode>(node);
    return (tpcNode != nullptr && tpcNode->isGuidPrefix("filter_2d"));
}

bool shouldQuantizeAsMMENode(const HabanaGraph& g, const NodePtr& node)
{
    return isFilter2d(node) || (g.runsOnMME(node) && node->getNodeType() != Node::TYPE_INTERNAL_TRANSPOSE);
}

void removeTensorFromTensorList(TensorVector& tensorsList, TensorPtr& tensor)
{
    auto tensorsIterator = std::find(tensorsList.begin(), tensorsList.end(), tensor);
    if (tensorsIterator != tensorsList.end())
    {
        tensorsList.erase(tensorsIterator);
    }
}

void add1DTensor(HabanaGraph& g,
                 NodePtr&     node,
                 int          index,
                 unsigned     length,
                 double       scale,
                 synDataType  dtype,
                 void*        pData,
                 std::string  suffix)
{
    if (pData == nullptr)
    {
        LOG_ERR(QUANT, "Unexpected nullptr for add1DTensor, node {}", node->getNodeName());
        return;
    }

    TSize sizes[Tensor::c_tensorMaxDim] = {1, 1, 1, 1, 1};
    sizes[0]                               = length;
    TensorPtr tensor                       = std::make_shared<Tensor>(1U, sizes, dtype);
    tensor->setAsStaticParam(true);
    tensor->bind(reinterpret_cast<char*>(pData), true);
    tensor->setName(node->getNodeName() + suffix);
    tensor->setScale(scale);
    tensor->setZeroPoint(0);
    tensor->getTensorAnnotation().dataInfo.isBiasFixedUp = true;
    GraphEditor::editNode(g, node, [&]() { node->addInput(index, tensor); });
}

/**
    Perform Gemmlowp quanntization to array of float32

    real = (quantized - zero_point) * scale
    => quantized = clamp(round(zerpo_point + real/scale))

    @param realValueArray real values array (float32)
    @param length length of the realValueArray array
    @param scale gemmlowp scale
    @param zp gemmlowp zero point
    @return quantized array in valType
*/
template<typename valType>
valType* RealToIntGemmlowp(float* realValueArray, QuantizationData& quantInfo, unsigned numElements, SizeArray* sizes, unsigned kIndex)
{
    if (realValueArray == nullptr)
    {
        LOG_ERR(QUANT, "Unexpected nullptr for a realValueArray when performing quantization");
        return nullptr;
    }

    for (unsigned idx = 0; idx < quantInfo.m_numChannels; idx++)
    {
        // lower the threshold for undefined scale, as it can be very close to zero
        if (allClose(quantInfo.scale(idx), 0.0, 1e-05f, 1e-14f))
        {
            LOG_ERR(QUANT, "Scale {} is undefined (too close to 0), index {}", quantInfo.scale(idx), idx);
            return nullptr;
        }
    }

    valType* newBuffer  = new valType[numElements];
    if (quantInfo.isPerChannel())
    {
        LOG_DEBUG(QUANT, "{}, Quantizing buffer per-channel", HLLOG_FUNC);
        HB_ASSERT_PTR(sizes);
        bool     kIsSCD             = isSCD(*sizes, kIndex);
        unsigned spatialElementsNum = 1;
        for (unsigned j = 0; j < (*sizes).size(); j++)
        {
            if (j == kIndex) continue;
            spatialElementsNum *= (*sizes)[j];
        }

        for (unsigned idx = 0; idx < quantInfo.m_numChannels; idx++)
        {
            double             scale            = quantInfo.scale(idx);
            double             zp               = quantInfo.zp(idx);
            std::set<unsigned> channelIdxVector = getChannelIndices(*sizes, kIndex, idx, kIsSCD, spatialElementsNum);
            for (unsigned i : channelIdxVector)
            {
                newBuffer[i] = RealToIntValue<valType>(realValueArray[i], scale, zp, quantInfo.m_qDataType);
            }
        }
    }
    else
    {
        LOG_DEBUG(QUANT, "{}, Quantizing buffer globally", HLLOG_FUNC);
        double scale = quantInfo.scale();
        double zp    = quantInfo.zp();
        for (unsigned idx = 0; idx < numElements; idx++)
        {
            newBuffer[idx] = RealToIntValue<valType>(realValueArray[idx], scale, zp, quantInfo.m_qDataType);
        }
    }

    return newBuffer;
}

/**
    Decide which gemmlowp quantization to perform in RT (u/int8, u/int16, u/int32)

    @param realValueArray real values array (float32)
    @param length length of the realValueArray array
    @param quantInfo quantization info
    @return quantized array in valType
*/
void* quantRealData(float* realValArray, QuantizationData& quantInfo, unsigned numElements, SizeArray* sizes, unsigned kIndex)
{
    if (realValArray == nullptr)
    {
        LOG_ERR(QUANT, "Unexpected nullptr for a realValue when performing quantization");
    }

    void* qBuffer = nullptr;
    switch (quantInfo.m_qDataType)
    {
        // 4 bits data types elements are arranged as a pair in a byte
        case quant_type_int4:
        case quant_type_int8:
            qBuffer = RealToIntGemmlowp<int8_t>(realValArray, quantInfo, numElements, sizes, kIndex);
            break;
            // 4 bits data types elements are arranged as a pair in a byte
        case quant_type_uint4:
        case quant_type_uint8:
            qBuffer = RealToIntGemmlowp<uint8_t>(realValArray, quantInfo, numElements, sizes, kIndex);
            break;
        case quant_type_int16:
            qBuffer = RealToIntGemmlowp<int16_t>(realValArray, quantInfo, numElements, sizes, kIndex);
            break;
        case quant_type_uint16:
            qBuffer = RealToIntGemmlowp<uint16_t>(realValArray, quantInfo, numElements, sizes, kIndex);
            break;
        case quant_type_int32:
            qBuffer = RealToIntGemmlowp<int32_t>(realValArray, quantInfo, numElements, sizes, kIndex);
            break;
        case quant_type_uint32:
            qBuffer = RealToIntGemmlowp<uint32_t>(realValArray, quantInfo, numElements, sizes, kIndex);
            break;
        default:
            LOG_ERR(QUANT, "Unknown data type for quantization {}", quantInfo.m_qDataType);
            return nullptr;
    }
    return qBuffer;
}

bool nodeRequestPCQ(HabanaGraph& g, const NodePtr& node)
{
    bool isMmeNode      = g.runsOnMME(node) && node->getNodeType() != Node::TYPE_INTERNAL_TRANSPOSE;
    bool isFilter2dNode = isFilter2d(node);

    if (!isMmeNode && !isFilter2dNode)
    {
        return false;
    }
    TensorPtr weightTensor = node->getInput(TENSOR_WEIGHT);

    for (const TensorPtr& tensor : node->getInputs())
    {
        if (tensor != nullptr && tensor != weightTensor && tensor->isPerChannelQuant())
        {
            // warn for every tensor that isn't a weight and marked as per channel
            LOG_WARN(QUANT,
                     "Cannot apply per channel quantization for tensor {} in node {}",
                     tensor->getName(),
                     node->getNodeName());
            tensor->setPerChannelQuant(false);
        }
    }

    return weightTensor != nullptr && weightTensor->isPerChannelQuant();
}

/*
 * Calculate the indices of each channel in the weights tensor, according to the K dim index.
 * Return a vector of sets , at the i-th place - a set of the i-th channel indices.
 */
channelsIndices calcChannelsIndices(TensorPtr tensor, const unsigned kIndex)
{
    unsigned channelsNum = tensor->getSizeInElements(kIndex);
    LOG_DEBUG(QUANT, "Tensor {} channelsNum - {}", tensor->getName(), channelsNum);

    channelsIndices tensorChannelIndices;

    const SizeArray weightSizes        = tensor->getAllSizesInElements();
    bool            kIsSCD             = isSCD(weightSizes, kIndex);
    unsigned        spatialElementsNum = 1;
    for (unsigned j = 0; j < weightSizes.size(); j++)
    {
        if (j == kIndex) continue;
        spatialElementsNum *= weightSizes[j];
    }
    for (unsigned channel = 0; channel < channelsNum; channel++)
    {
        std::set<unsigned> newIndices = getChannelIndices(weightSizes, kIndex, channel, kIsSCD, spatialElementsNum);
        tensorChannelIndices.push_back(newIndices);
    }
    return tensorChannelIndices;
}

bool calcWeightScales(const NodePtr&    node,
                      QuantizationData& quantInfo,
                      double            backoffFactor,
                      bool              isSparsityWeights)
{
    TensorPtr weightTensor = node->getInput(TENSOR_WEIGHT);
    HB_ASSERT_PTR(weightTensor);

    if (weightTensor->isAuxTensor())
    {
        LOG_DEBUG(QUANT, "Cannot quantize aux tensor {}", weightTensor->getName());
        return false;
    }

    if (quantInfo.getSynDataType() == syn_type_na)
    {
        LOG_WARN(QUANT, "Cannot quantize tensor {} for unknown data type", weightTensor->getName());
        return false;
    }

    resetWeightQuantInfo(node, weightTensor, quantInfo);
    QuantizationData weightQuantInfo = weightTensor->getQuantizationParams(quantInfo.m_qDataType);

    if (weightQuantInfo.m_isUserPCQuantInfo && weightQuantInfo.m_qDataType == quantInfo.m_qDataType &&
        weightQuantInfo.m_numChannels == quantInfo.m_numChannels)
    {
        LOG_DEBUG(QUANT,
                  "Using user pc ExpBias and pc Scales for {} tensor dtype {}",
                  weightTensor->getName(),
                  QuantizationData::getDataTypeString(quantInfo.m_qDataType));
        quantInfo.setQuantParamsVector(weightQuantInfo.getQuantParamsVector());
        return true;
    }

    LOG_DEBUG(QUANT,
              "Calculate per channel scales for {} tensor dtype {}",
              weightTensor->getName(),
              QuantizationData::getDataTypeString(quantInfo.m_qDataType));

    auto perChannelDynamicRange = weightTensor->getPerChannelDynamicRange();
    HB_ASSERT(perChannelDynamicRange.isSet, "Per channel dynamic range must be set");
    for (unsigned idx = 0; idx < perChannelDynamicRange.numChannels; idx++)
    {
        float minVal = perChannelDynamicRange.ranges[idx].min;
        float maxVal = perChannelDynamicRange.ranges[idx].max;
        float absMax = std::max(std::abs(minVal), std::abs(maxVal));

        QuantizationUtils::calcPCScaleForFp8_143(absMax, quantInfo.m_qDataType, quantInfo.getChannelParams(idx).scale, backoffFactor);

        LOG_DEBUG(QUANT,
                  "{}, Calculated quant info for tensor {}, dtype={}, channel={}, expBias={}, scale={}",
                  HLLOG_FUNC,
                  weightTensor->getName(),
                  QuantizationData::getDataTypeString(quantInfo.m_qDataType),
                  idx,
                  quantInfo.expBias(idx),
                  quantInfo.scale(idx));
    }
    return true;
}

void resetWeightQuantInfo(const NodePtr& node, const TensorPtr& weightTensor, QuantizationData& quantInfo)
{
    unsigned kIndex      = node->getKDimIndex();
    unsigned numChannels = weightTensor->getSizeInElements(kIndex);
    unsigned channelSize = 1;
    for (unsigned i = 0; i < weightTensor->getDim(); i++)
    {
        if (i != kIndex)
        {
            channelSize *= weightTensor->getSizeInElements(i);
        }
    }

    quantInfo.reset(numChannels, quantInfo.m_qDataType);
}

template<typename valType>
valType* createZeroPointsBufferInternal(const unsigned bufferSizeInElements, const double* originBuffer)
{
    valType* newBuffer = new valType[bufferSizeInElements];
    for (unsigned idx = 0; idx < bufferSizeInElements; idx++)
    {
        newBuffer[idx] = static_cast<valType>(originBuffer[idx]);
    }
    return newBuffer;
}

void* createZeroPointsBuffer(const synDataType dataType,
                             const unsigned    bufferSizeInElements,
                             const double*     originBuffer)
{
    switch (dataType)
    {
        case syn_type_int4:
        case syn_type_int8:
        {
            return createZeroPointsBufferInternal<int8_t>(bufferSizeInElements, originBuffer);
        }
        case syn_type_uint4:
        case syn_type_uint8:
        {
            return createZeroPointsBufferInternal<uint8_t>(bufferSizeInElements, originBuffer);
        }
        default:
        {
            HB_ASSERT(false, "Invalid data type for creating zero points buffer");
            return nullptr;
        }
    }
}

template<typename T>
void getZeroPointsAsDoubleInternal(std::vector<double>& zpVector, TensorPtr zpTensor)
{
    T* zpData = reinterpret_cast<T*>(zpTensor->getData());
    zpVector.resize(zpTensor->getDenseSizeInElements());
    for (size_t i = 0; i < zpVector.size(); i++)
    {
        zpVector[i] = static_cast<double>(zpData[i]);
    }
}

void getZeroPointsAsDouble(std::vector<double>& zpVector, TensorPtr zpTensor)
{
    switch (zpTensor->getElementType())
    {
        case syn_type_int4:
        case syn_type_int8:
        {
            getZeroPointsAsDoubleInternal<int8_t>(zpVector, zpTensor);
            return;
        }
        case syn_type_uint4:
        case syn_type_uint8:
        {
            getZeroPointsAsDoubleInternal<uint8_t>(zpVector, zpTensor);
            return;
        }
        default:
        {
            HB_ASSERT(false, "Invalid data type for getting zero points as double");
        }
    }
}

// TODO[c++20]: use std::span instead of templating this function
template<size_t N>
static bool isExpBiasInTable(unsigned value, const std::array<ExpBiasMaxAbsPair, N>& table)
{
    // iterate table to find if the given val is a suitable expBias for the data type
    for (unsigned i = 0; i < table.size(); i++)
    {
        if (value == table[i].first)
        {
            return true;
        }
    }
    return false;
}

bool expBiasIsInRangeOfType(unsigned expBias, synDataType type)
{
    switch (type)
    {
        case syn_type_fp8_152:
            return isExpBiasInTable(expBias, expBiasTableFp8152);
        case syn_type_fp8_143:
            return isExpBiasInTable(expBias, expBiasTableFp8143);
        default:
            return true;
    }
}

template<typename T>
void setRange(const TensorPtr& tensor)
{
    T* data = reinterpret_cast<T*>(tensor->getData());
    if (data == nullptr)
    {
        LOG_WARN(QUANT, "Cannot calc quant info for tensor {} with no data", tensor->getName());
        return;
    }

    unsigned numElements = tensor->getTotalElements();
    if (numElements < 1)
    {
        LOG_WARN(QUANT, "Cannot calculate dynamic range for empty data buffer, tensor {}", tensor->getName());
        return;
    }

    T min = data[0];
    T max = data[0];
    for (int i = 1; i < numElements; i++)
    {
        T val = data[i];
        if (min > val)
        {
            min = val;
        }
        if (max < val)
        {
            max = val;
        }
    }

    DynamicRange dynamicRange;
    dynamicRange.min   = (double)min;
    dynamicRange.max   = (double)max;
    dynamicRange.isSet = true;
    LOG_DEBUG(QUANT, "Setting dynamic range {} {}", (double)min, (double)max);
    tensor->setDynamicRange(dynamicRange);
}

void calcPerTensorDynamicRange(const TensorPtr& tensor)
{
    synDataType bufferType = tensor->getBufferDataType();
    switch (bufferType)
    {
        case syn_type_uint8:
        {
            setRange<uint8_t>(tensor);
            break;
        }
        case syn_type_int8:
        {
            setRange<int8_t>(tensor);
            break;
        }
        case syn_type_uint16:
        {
            setRange<uint16_t>(tensor);
            break;
        }
        case syn_type_int16:
        {
            setRange<int16_t>(tensor);
            break;
        }
        case syn_type_uint32:
        {
            setRange<uint32_t>(tensor);
            break;
        }
        case syn_type_int32:
        {
            setRange<int32_t>(tensor);
            break;
        }
        case syn_type_bf16:
        {
            setRange<bf16_t>(tensor);
            break;
        }
        case syn_type_fp16:
        {
            setRange<fp16_t>(tensor);
            break;
        }
        case syn_type_float:
        {
            setRange<float>(tensor);
            break;
        }
        default:
        {
            LOG_WARN(QUANT, "Unsupported buffer data type for tensor {}", tensor->getName());
            return;
        }
    }
}

template<typename T>
void setPerChannelRange(const TensorPtr& tensor, unsigned numChannels, channelsIndices& channelsIndices)
{
    T* tensorData = reinterpret_cast<T*>(tensor->getData());
    HB_ASSERT(tensorData != nullptr, "Static param tensor {} have no data", tensor->getName());
    HB_ASSERT(numChannels == channelsIndices.size(),
              "unexpected size of channels indices vector in tensor {}",
              tensor->getName());

    RangesVector ranges;
    for (unsigned idx = 0; idx < numChannels; idx++)
    {
        synQuantDynamicRange range;
        range.min = (double)getMinValue<T>(tensorData, channelsIndices[idx]);
        range.max = (double)getMaxValue<T>(tensorData, channelsIndices[idx]);
        ranges.push_back(range);
    }

    PerChannelDynamicRange perChannelDynamicRange;
    perChannelDynamicRange.numChannels   = numChannels;
    perChannelDynamicRange.ranges        = ranges;
    perChannelDynamicRange.isSet         = true;

    tensor->setPerChannelDynamicRange(perChannelDynamicRange);
}

void calcWeightMMEPerChannelDynamicRange(const NodePtr& node)
{
    TensorPtr weightTensor = node->getInput(TENSOR_WEIGHT);
    HB_ASSERT_PTR(weightTensor);
    if (weightTensor->getPerChannelDynamicRange().isSet || !weightTensor->isStaticParam()) return;

    const unsigned  kIndex          = node->getKDimIndex();
    channelsIndices channelsIndices = calcChannelsIndices(weightTensor, kIndex);
    unsigned        numChannels     = weightTensor->getSizeInElements(kIndex);

    synDataType bufferType = weightTensor->getBufferDataType();
    switch (bufferType)
    {
        case syn_type_uint8:
        {
            setPerChannelRange<uint8_t>(weightTensor, numChannels, channelsIndices);
            break;
        }
        case syn_type_int8:
        {
            setPerChannelRange<int8_t>(weightTensor, numChannels, channelsIndices);
            break;
        }
        case syn_type_uint16:
        {
            setPerChannelRange<uint16_t>(weightTensor, numChannels, channelsIndices);
            break;
        }
        case syn_type_int16:
        {
            setPerChannelRange<int16_t>(weightTensor, numChannels, channelsIndices);
            break;
        }
        case syn_type_uint32:
        {
            setPerChannelRange<uint32_t>(weightTensor, numChannels, channelsIndices);
            break;
        }
        case syn_type_int32:
        {
            setPerChannelRange<int32_t>(weightTensor, numChannels, channelsIndices);
            break;
        }
        case syn_type_bf16:
        {
            setPerChannelRange<bf16_t>(weightTensor, numChannels, channelsIndices);
            break;
        }
        case syn_type_fp16:
        {
            setPerChannelRange<fp16_t>(weightTensor, numChannels, channelsIndices);
            break;
        }
        case syn_type_float:
        {
            setPerChannelRange<float>(weightTensor, numChannels, channelsIndices);
            break;
        }
        default:
        {
            LOG_WARN(QUANT, "Unsupported buffer data type for tensor {}", weightTensor->getName());
            return;
        }
    }
}

void updateExpBias(const NodePtr& node, TensorPtr tensorPtr, double scaleToUpdate)
{
    unsigned expBias = QuantizationData::S_EXP_BIAS_143_DEFAULT - log2(scaleToUpdate);
    HB_ASSERT(tensorPtr->setExpBiasForDtype(expBias), "Node {} should have quant data for type fp8", node->getNodeName());
    LOG_DEBUG(QUANT, "Set tensor {} with exp bias {}", tensorPtr->getName(), expBias);
}

void updateScale(const NodePtr& node, TensorPtr tensorPtr, double scaleToUpdate)
{
    HB_ASSERT(scaleToUpdate != 0, "Scale to update in tensor {} is 0", tensorPtr->getName());
    double scale = 1.0 / scaleToUpdate;
    HB_ASSERT(tensorPtr->setScaleForDtype(scale), "Node {} should have quant data for type fp8", node->getNodeName());
    LOG_DEBUG(QUANT, "Set inputs tensor {} with scale {}", tensorPtr->getName(), scale);
}

bool updateMMENodeScales(const NodePtr& node, const json_utils::Json& json, bool useExpBias = true)
{
    LOG_DEBUG(QUANT, "Updating mme node {} with values from JSON", node->getNodeName());
    json_utils::Json inputs;
    json_utils::Json outputs;
    if (json.count(INPUTS) != 0)
    {
        inputs = json_utils::get(json, INPUTS);
    }
    if (json.count(OUTPUTS) != 0)
    {
        outputs = json_utils::get(json, OUTPUTS);
    }
    json_utils::Json params;
    if (json.count(PARAMS) != 0)
    {
        params = json_utils::get(json, PARAMS);
    }

    std::vector<double> scalesToUpdate;
    for (const auto& jsonInput : inputs)
    {
        scalesToUpdate.push_back(json_utils::get(jsonInput, "scale_after"));
    }
    // json file contains 2 scales for each node - either both under inputs or one under inputs and the other under params
    if (scalesToUpdate.size() == 1)
    {
        scalesToUpdate.push_back(json_utils::get(json_utils::get(params, WEIGHTS), "scale_after")[0]);
    }
    for (const auto& jsonOutput : outputs)
    {
        scalesToUpdate.push_back(json_utils::get(jsonOutput, "scale_after"));
    }
    HB_ASSERT(scalesToUpdate.size() == 2 || scalesToUpdate.size() == 3, "Should have 2/3 scales to update in node {}", node->getNodeName());

    TensorPtr inputTensor  = node->getInput(TENSOR_IFM);
    TensorPtr weightTensor = node->getInput(TENSOR_WEIGHT);
    TensorPtr outputTensor = node->getOutput(TENSOR_OFM);

    if (node->getNodeName().find("addmm") != std::string::npos)
    {
        LOG_DEBUG(QUANT, "Handling addmm case for Node {}", node->getNodeName());
        inputTensor  = node->getInput(1);
        weightTensor = node->getInput(2);
    }

    if (useExpBias)
    {
        updateExpBias(node, inputTensor, scalesToUpdate[0]);
        updateExpBias(node, weightTensor, scalesToUpdate[1]);
        if (scalesToUpdate.size() == 3)
        {
            updateExpBias(node, outputTensor, scalesToUpdate[2]);
        }

    }
    else
    {
        updateScale(node, inputTensor, scalesToUpdate[0]);
        updateScale(node, weightTensor, scalesToUpdate[1]);
        if (scalesToUpdate.size() == 3)
        {
            updateScale(node, outputTensor, scalesToUpdate[2]);
        }
    }

    return true;
}

bool updateCastNodeScales(const NodePtr& node, const json_utils::Json& json, bool useExpBias = true)
{
    LOG_DEBUG(QUANT, "Updating cast node {} with values from JSON", node->getNodeName());
    json_utils::Json inputs;
    json_utils::Json outputs;
    bool input_scale = false;
    if (json.count(INPUTS) != 0)
    {
        input_scale = true;
        inputs = json_utils::get(json, INPUTS);
    }
    if (json.count(OUTPUTS) != 0)
    {
        outputs = json_utils::get(json, OUTPUTS);
    }

    std::vector<double> scalesToUpdate;
    for (const auto& jsonInput : inputs)
    {
        scalesToUpdate.push_back(json_utils::get(jsonInput, "scale_after"));
    }
    for (const auto& jsonOutput : outputs)
    {
        scalesToUpdate.push_back(json_utils::get(jsonOutput, "scale_after"));
    }
    HB_ASSERT(scalesToUpdate.size() == 1, "Should have 1 scales to update in node {}", node->getNodeName());

    TensorPtr updaedTensor = input_scale ? node->getInput(TENSOR_IFM) : node->getOutput(TENSOR_OFM);

    if (useExpBias)
    {
        updateExpBias(node, updaedTensor, scalesToUpdate[0]);
    }
    else
    {
        updateScale(node, updaedTensor, scalesToUpdate[0]);
    }

    return true;
}

bool updateDynamicRange(NodePtr node, json_utils::Json& tensors, const std::string& tensor_type, unsigned *index)
{
    TensorPtr tensor;
    for (auto& jsonInputDynRange : tensors)
    {
        if(tensor_type == WEIGHTS) tensor = node->getInput(1);
        if(tensor_type == OUTPUTS) tensor = node->getOutput(*index);
        if(tensor_type == INPUTS) tensor  = node->getInput(*index);
        if (jsonInputDynRange.size() != 2)
        {
            LOG_WARN(QUANT,
                     "Got json dyn range list with size {} for tensor {}",
                     jsonInputDynRange.size(),
                     tensor->getName());
            return false;
        }
        DynamicRange dynRange;
        dynRange.min   = jsonInputDynRange.at(0);
        dynRange.max   = jsonInputDynRange.at(1);
        dynRange.isSet = true;
        LOG_WARN(QUANT,
                 "Setting dyn range from json to tensor {}, min={},max={}",
                 tensor->getName(),
                 dynRange.min,
                 dynRange.max);
        tensor->setDynamicRange(dynRange);
        (*index)++;
    }
    return true;
}

bool updateNodeTensorsDynamicRange(NodePtr node, json_utils::Json& dynRangeJson)
{
    LOG_DEBUG(QUANT, "Updating node {} with info from JSON", node->getNodeName());
    json_utils::Json inputs  = json_utils::get(dynRangeJson, INPUTS);
    json_utils::Json outputs = json_utils::get(dynRangeJson, OUTPUTS);
    json_utils::Json weights = json_utils::get(dynRangeJson, WEIGHTS);
    // set dyn range for inputs
    unsigned index = 0;
    if(!updateDynamicRange(node, inputs, INPUTS, &index)) return false;

    if (index < 2)
    {
        if(!updateDynamicRange(node, weights, WEIGHTS, &index)) return false;
    }

    index = 0;
    if(!updateDynamicRange(node, outputs, OUTPUTS, &index)) return false;

    return false;
}

void editString(std::string& original)
{
    std::replace( original.begin(), original.end(), '.', '/');
}

bool nodeNotApplicableForParamsLoading(const std::string& nodeName)
{
    return nodeName.find("gemm") == std::string::npos
            && nodeName.find("addmm") == std::string::npos
            && nodeName.find("convert_from_fp8_bf16") == std::string::npos
            && nodeName.find("convert_to_fp8_bf16") == std::string::npos
            && nodeName.find("lm_head") == std::string::npos;
}

bool loadQuantizationParams(const HabanaGraph& graph, const std::string& filePath, const std::string& mode)
{
    // validate json file
    json_utils::Json json = json_utils::jsonFromFile(filePath);
    if (json.empty()) return false;
    json_utils::Json jsonNodes;
    //nodes split
    for (auto& jsonIter : json_utils::Json::iterator_wrapper(json))
    {
        if (jsonIter.key() == "Nodes")
            jsonNodes = jsonIter.value();
    }
    std::unordered_map<std::string, NodePtr> jsonNameToPtr;

    for (auto& jsonIter : json_utils::Json::iterator_wrapper(jsonNodes))
    {
        std::string jsonNodeName = jsonIter.key();
        editString(jsonNodeName);

        //Node names in json should be unique
        HB_ASSERT(jsonNameToPtr.find(jsonNodeName) == jsonNameToPtr.end() , "Node {} already added", jsonNodeName);

        jsonNameToPtr[jsonNodeName] = nullptr;
    }

    for (const auto& node : graph.getExeSortedNodes())
    {
        if (node == nullptr) continue;
        const auto nodeName = node->getNodeName();

        if (nodeNotApplicableForParamsLoading(nodeName))
            continue;

        for(auto& [jsonNodeName, value]: jsonNameToPtr)
        {
            if (nodeName.find(jsonNodeName) != std::string::npos)
            {
                //if jsonNodeName is bmm and nodeName from graph is bmm2 we get that jsonNodeName
                //is in nodeName but they are not the same so here we make sure.
                if ((nodeName.find("bmm2") == std::string::npos) != (jsonNodeName.find("bmm2") == std::string::npos))
                {
                    continue;
                }
                HB_ASSERT(value == nullptr, "Node {} already added", jsonNodeName);
                jsonNameToPtr[jsonNodeName] = node;
            }
        }
    }

    for (auto& jsonIter : json_utils::Json::iterator_wrapper(jsonNodes))
    {
        std::string jsonNodeName = jsonIter.key();
        editString(jsonNodeName);
        NodePtr currentNode = jsonNameToPtr[jsonNodeName];
        if (currentNode == nullptr)
        {
            LOG_WARN(QUANT, "Didn't find a gc node for json node {}", jsonNodeName);
            continue;
        }
        if (mode == "DynamicRange")
        {
            updateNodeTensorsDynamicRange(currentNode, jsonIter.value());
        }
        if (mode == "Scale")
        {
            //TODO: [SW-155608] Finish implementing inject scale nodes flow
            if (jsonNodeName.find("convert") != std::string::npos)
            {
                updateCastNodeScales(currentNode, jsonIter.value());
            }
            else
            {
                updateMMENodeScales(currentNode, jsonIter.value());
            }
        }
    }
    return true;
}

unsigned getFP8MmeInputScaleIndex(NodePtr mmeNode)
{
    if (isFp8GemmGuid(mmeNode))
    {
        return FP8_GEMM_SCALE_IFM_IDX;
    }
    else if (isFp8ConvGuid(mmeNode))
    {
        return FP8_CONV_2D_SCALE_IFM_IDX;
    }
    else
    {
        HB_ASSERT(false, "Can't get input scale index for MME fp8 cguid {}", mmeNode->getGUID());
        return 0;
    }
}

unsigned getFP8MmeWeightScaleIndex(NodePtr mmeNode)
{
    if (isFp8GemmGuid(mmeNode))
    {
        return FP8_GEMM_SCALE_WEIGHT_IDX;
    }
    else if (isFp8ConvGuid(mmeNode))
    {
        return FP8_CONV_2D_SCALE_WEIGHT_IDX;
    }
    else
    {
        HB_ASSERT(false, "Can't get weight scale index for MME fp8 cguid {}", mmeNode->getGUID());
        return 0;
    }
}

bool isConvertExpBiasHwAligned(NodePtr convertNode)
{
    TensorPtr invScaleTensor = convertNode->getInput(CONVERT_INV_SCALE_IDX);
    HalReaderPtr halReader = convertNode->getGraphTraits()->getHalReader();
    if (invScaleTensor && !isInvScaleExpBiasHWAligned(halReader, invScaleTensor)) return false;
    return true;
}

bool isFp8MmeExpBiasHwAligned(NodePtr mmeNode)
{
    unsigned inputScaleIndex  = getFP8MmeInputScaleIndex(mmeNode);
    unsigned weightScaleIndex = getFP8MmeWeightScaleIndex(mmeNode);

    TensorPtr scaleIfm = mmeNode->getInput(inputScaleIndex);
    TensorPtr scaleWeight = mmeNode->getInput(weightScaleIndex);
    HalReaderPtr halReader = mmeNode->getGraphTraits()->getHalReader();
    if (scaleIfm && !isScaleExpBiasHWAligned(halReader, scaleIfm)) return false;
    if (scaleWeight && !isScaleExpBiasHWAligned(halReader, scaleWeight)) return false;

    return true;
}

bool isInvScaleExpBiasHWAligned(HalReaderPtr halReader, TensorPtr invScaleTensor)
{
    if (invScaleTensor->getTotalElements() != 1 || !invScaleTensor->isStaticParam()) return false;
    float invScale = getScaleFromTensor(invScaleTensor);
    return halReader->isInvScaleExpBiasHWAligned(invScale);
}

bool isScaleExpBiasHWAligned(HalReaderPtr halReader, TensorPtr scaleTensor)
{
    if (scaleTensor->getTotalElements() != 1 || !scaleTensor->isStaticParam()) return false;
    float scale = getScaleFromTensor(scaleTensor);
    return halReader->isScaleExpBiasHWAligned(scale);
}

}  // namespace QuantizationUtils
