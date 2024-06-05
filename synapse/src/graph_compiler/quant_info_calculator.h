#pragma once

#include "habana_graph.h"
#include "node_factory.h"
#include "quantization_data.h"

#define NUM_SUPPORTED_QUANTIZATION_TYPES 2

class QuantInfoCalculator
{
public:
    bool runCalcQuantInfo(HabanaGraph& g);
    static QuantizationMap basicQuantizationMap(const pTensor& tensor);
    const static eQuantDataType supportedDTypes[NUM_SUPPORTED_QUANTIZATION_TYPES];

protected:
    virtual bool canApplyPerChannelQuant(HabanaGraph& g, const pNode& node) = 0;

private:
    std::string m_mode = "";
    std::string m_quantizationParamsPath = "";
    bool initQuantParamsLoadingMode();
    bool loadQuantParamsByMode (const HabanaGraph& graph);
    bool tryLoadQuantParamsFromFile(const HabanaGraph& graph);
    void calcWeightMMEQuantInfo(const pNode node, bool perChannel, bool isGlobalSparsityWeights,
                                TensorVector& tensorsToHandle, double backoffFactor);
    void calcGlobalQuantInfoPerDType(pTensor tensor, QuantizationData& quantInfo, double backoffFactor,
                                     bool isSparsityWeights=false);
    void calcTensorGlobalQuantInfo(pTensor tensor, double backoffFactor, bool isSparsityWeights=false);
};
