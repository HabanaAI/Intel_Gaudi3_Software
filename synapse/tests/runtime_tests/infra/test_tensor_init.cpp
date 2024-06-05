#include "test_tensor_init.hpp"

#include "log_manager.h"

#include "test_utils.h"

static void setTensorConst(synDataType dataType, void* output, uint64_t numElements, uint64_t initializer)
{
    UNUSED(initializer);
    switch (dataType)
    {
        case syn_type_uint8:
        case syn_type_int8:
            setBuffer<uint8_t>(output, numElements, [initializer]() { return initializer; });
            break;
        case syn_type_bf16:
            setBuffer<uint16_t>(output, numElements, [initializer]() { return bfloat16((float)initializer); });
            break;
        case syn_type_fp8_152:
            setBuffer<fp8_152_t>(output, numElements, [initializer]() { return fp8_152_t((float)initializer); });
            break;
        case syn_type_int64:
        case syn_type_uint64:
            setBuffer<int64_t>(output, numElements, [initializer]() { return initializer; });
            break;
        case syn_type_uint32:
        case syn_type_int32:
            setBuffer<int>(output, numElements, [initializer]() { return initializer; });
            break;
        case syn_type_single:
            setBuffer<float>(output, numElements, [initializer]() { return initializer; });
            break;
        default:
            HB_ASSERT(false, "unsupported tensor data type: {}", dataType);
    }
}

bool initBufferValues(OneTensorInitInfo oneTensorInitInfo, synDataType dataType, uint64_t numElements, void* output)
{
    TensorInitOp tensorInitOp       = oneTensorInitInfo.m_tensorInitOp;
    uint64_t     initializer        = oneTensorInitInfo.m_initializer;
    bool         isDefaultGenerator = oneTensorInitInfo.m_isDefaultGenerator;

    switch (tensorInitOp)
    {
        case TensorInitOp::NONE:
            return true;

        case TensorInitOp::RANDOM_WITH_NEGATIVE:
        case TensorInitOp::RANDOM_POSITIVE:
        {
            MemInitType memInitType = (tensorInitOp == TensorInitOp::RANDOM_WITH_NEGATIVE)
                                          ? MEM_INIT_RANDOM_WITH_NEGATIVE
                                          : MEM_INIT_RANDOM_POSITIVE;

            randomBufferValues(memInitType, dataType, numElements, output, isDefaultGenerator);
            return true;
        }

        case TensorInitOp::ALL_ONES:
            setTensorConst(dataType, output, numElements, 1);
            break;

        case TensorInitOp::ALL_ZERO:
            setTensorConst(dataType, output, numElements, 0);
            break;

        case TensorInitOp::CONST:
            setTensorConst(dataType, output, numElements, initializer);
            break;
    }
    return true;
}
