#include "habana_pass.h"
#include "quant_info_calculator.h"
#include "data_type_utils.h"

bool validateQuantization(HabanaGraph& g)
{
    if (!isInferenceQuantization(g))
    {
        LOG_DEBUG(QUANT, "Graph is not in inference mode, validate quantization won't run.");
        return true;
    }

    if (!GCFG_ENABLE_SYNAPSE_QUANTIZATION.value())
    {
        LOG_DEBUG(QUANT, "Quantization is disabled in synapse. Skip {} Pass", HLLOG_FUNC);
        return true;
    }

    const TensorSet& graphTensors = g.getTensors();
    for (const TensorPtr& tensor : graphTensors)
    {
        QuantizationMap quantizationMap = tensor->getAllQuantizationParams();
        // user-quantized tensors often given quantization values only for the tensor's data type.
        if (tensor->isDataTypeMatchData())
        {
            if (isQuantDtype(tensor->getElementType()) && (quantizationMap.find(QuantizationData::synTypeToQuantType(
                                                               tensor->getElementType())) == quantizationMap.end()))
            {
                LOG_ERR(QUANT,
                        "{}: tensor {} has no quantization info in its type {}",
                        HLLOG_FUNC,
                        tensor->getName(),
                        getStringFromSynDataType(tensor->getElementType()));
                return false;
            }
        }
        else
        {
            for (eQuantDataType dtype : QuantInfoCalculator::supportedDTypes)
            {
                if (quantizationMap.find(dtype) == quantizationMap.end())
                {
                    LOG_ERR(QUANT,
                            "{}: tensor {} has no quantization info in type {}",
                            HLLOG_FUNC,
                            tensor->getName(),
                            getStringFromSynDataType(QuantizationData::quantTypeToSynType(dtype)));
                    return false;
                }
            }
        }
    }
    return true;
}
