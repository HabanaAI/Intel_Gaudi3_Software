#pragma once

#include "habana_graph.h"
#include "hal_reader/hal_reader.h"
#include "limits_4bit.h"
#include "quantization_data.h"
#include "synapse_common_types.h"
#include "tensor.h"
#include "types.h"
#include "utils.h"

namespace QuantizationUtils
{
bool expBiasIsInRangeOfType(unsigned expBias, synDataType type);

float getMinValue(float* pRealArray, std::set<unsigned>& indices);

float getMaxValue(float* pRealArray, std::set<unsigned>& indices);

bool calcDataScaleZp(double minVal, double maxVal, eQuantDataType dtype, double& scale, double& zp,
                     bool isFixedPoint=false, bool isSparsityWeights=false);

bool calcExpBias(double tensorMaxAbs, eQuantDataType dtype, unsigned& expBias, double backoffFactor = 1.0);

void calcScaleForFp8_143(float absMax, QuantizationData& quantInfo, double backoffFactor);

void calcPCScaleForFp8_143(float absMax, eQuantDataType dtype, double& scale, double backoffFactor);

bool hasBias(NodePtr& node);

bool isFilter2d(const NodePtr& node);

bool shouldQuantizeAsMMENode(const HabanaGraph& g, const NodePtr& node);

void removeTensorFromTensorList(TensorVector& tensorsList, TensorPtr& tensor);

void add1DTensor(HabanaGraph& g, NodePtr& node, int index, unsigned length, double scale, synDataType dtype,
                 void* pData, std::string suffix);

/**
    Return the sign of value

    @param val value
    @return -1 if v < 0
             1 if v >= 0
*/
template<typename valType>
static int sign(valType val)
{
    return (valType(0) < val) - (val < valType(0));
}



/**
    Round an input in decimal

    **This function has no meaning if using non decimal number

    @param n value
    @return rounded n
*/
template<typename valType>
static valType roundAwayFromZero(valType n)
{
    valType absN = fabs(n);
    valType rnd = sign(n) * floor(absN + 0.5);
    return rnd;
}

template<typename valType>
valType RealToIntValue(float realValue, double scale, double zp, eQuantDataType dt = quant_type_na)
{

    float new_number = 0;
    float min_clip = 0, max_clip = 0;
    new_number = (realValue / (float)scale) + (float)zp;
    new_number = roundAwayFromZero<float>(new_number);
    switch (dt)
    {
        case quant_type_int4:
            min_clip = INT4_MIN_VAL;
            max_clip = INT4_MAX_VAL;
            break;
        case quant_type_uint4:
            min_clip = UINT4_MIN_VAL;
            max_clip = UINT4_MAX_VAL;
            break;
        default:
            min_clip = std::numeric_limits<valType>::min();
            max_clip = std::numeric_limits<valType>::max();
    }

    new_number = clip<float>(new_number, min_clip, max_clip);
    return (valType)new_number;
}

template<typename valType>
float IntToRealValue(valType intValue, double scale, double zp)
{
    return (intValue - (float)zp) * (float)scale;
}

template<typename valType>
valType* RealToIntGemmlowp(float* realValueArray, QuantizationData& quantInfo, unsigned numElements, SizeArray* sizes=nullptr, unsigned kIndex=0);

void* quantRealData(float* realValArray, QuantizationData& quantInfo, unsigned numElements, SizeArray* sizes=nullptr, unsigned kIndex=0);

bool nodeRequestPCQ(HabanaGraph& g, const NodePtr& node);

typedef std::vector<std::set<unsigned>> channelsIndices;
channelsIndices calcChannelsIndices(TensorPtr tensor, unsigned kIndex);

bool calcWeightScales(const NodePtr& node, QuantizationData& quantInfo, double backoffFactor, bool isSparsityWeights);

void resetWeightQuantInfo(const NodePtr& node, const TensorPtr& weightTensor, QuantizationData& quantInfo);

void* createZeroPointsBuffer(const synDataType dataType,
                             const unsigned    bufferSizeInElements,
                             const double*     originBuffer);

void getZeroPointsAsDouble(std::vector<double>& zpVector, TensorPtr zpTensor);

template<typename T>
void setRange(const TensorPtr& tensor);

void calcPerTensorDynamicRange(const TensorPtr& tensor);

template<typename T>
void setPerChannelRange(const TensorPtr& tensor, unsigned numChannels, channelsIndices& channelsIndices);

void calcWeightMMEPerChannelDynamicRange(const NodePtr& node);

bool loadQuantizationParams(const HabanaGraph& graph, const std::string& filePath, const std::string& mode);

bool isScaleExpBiasHWAligned(HalReaderPtr halReader, TensorPtr scaleTensor);

bool isInvScaleExpBiasHWAligned(HalReaderPtr halReader, TensorPtr invScaleTensor);

bool isConvertExpBiasHwAligned(NodePtr convertNode);

bool isFp8MmeExpBiasHwAligned(NodePtr mmeNode);
}; //namespace
