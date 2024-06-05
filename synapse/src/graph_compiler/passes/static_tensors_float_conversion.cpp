
#include "data_type_utils.h"
#include "tensor.h"
#include "habana_graph.h"

bool is16BitFloat(synDataType dataType)
{
    return dataType == syn_type_bf16 || dataType == syn_type_fp16;
}

bool convertTensorTo16BitFloat(pTensor tensor)
{
    HB_ASSERT(!tensor->isAuxTensor(), "expected a non aux tensor");
    if (tensor->isDataTypeMatchData())
    {
        LOG_TRACE(GC, "{} Tensor {} is already converted", HLLOG_FUNC, tensor->getName());
        return true;
    }
    unsigned elementsNum = tensor->getTotalElements();
    if (elementsNum == 0)
    {
        LOG_WARN(GC, "{} Tensor {} is empty, won't convert", HLLOG_FUNC, tensor->getName());
        return true;
    }

    synDataType dataType  = tensor->getElementType();
    void* convertedBuffer = nullptr;

    if (dataType == syn_type_bf16)
    {
        LOG_DEBUG(GC, "{} Converting tensor {} to bf16", HLLOG_FUNC, tensor->getName());
        convertedBuffer = convertBuffer<bf16_t>(reinterpret_cast<float*>(tensor->getData()), elementsNum);
    }
    else if (dataType == syn_type_fp16)
    {
        LOG_DEBUG(GC, "{} Converting tensor {} to fp16", HLLOG_FUNC, tensor->getName());
        convertedBuffer = convertBuffer<fp16_t>(reinterpret_cast<float*>(tensor->getData()), elementsNum);
    }
    else
    {
        LOG_ERR(GC, "{} Unexpected data type {} for tensor {}", HLLOG_FUNC, dataType, tensor->getName());
        return false;
    }

    tensor->unbind();
    tensor->bind(convertedBuffer, true);
    tensor->setAsDataTypeMatchData();
    LOG_DEBUG(GC, "{} Converted successfully tensor {}", HLLOG_FUNC, tensor->getName());
    return true;
}

bool staticTensorsFloatConversion(HabanaGraph& g)
{
    for (pTensor t : g.getTensors())
    {
        if (t->isStaticParam() && is16BitFloat(t->getElementType()))
        {
            if (!convertTensorTo16BitFloat(t))
            {
                LOG_ERR(GC, "{} Error while converting tensor {}, pass will finish unsuccessfully.", HLLOG_FUNC, t->getName());
                return false;
            }
        }
    }
    return true;
}

