#pragma once

#include "types.h"

// CpuCaster performs cast operation on cpu
class CpuCaster
{
public:
    CpuCaster(TensorPtr castFromTensor, TensorPtr castToTensor);
    // Cast input tensor data into output tensor data according to their element types and quantization info if relevant
    bool doCast();

    void setCastDataTypes(synDataType castFromType, synDataType castToType);

private:
    // Each castFrom* method below handles different case of cast input type.
    // Casting from fp32 - directly cast to cast output type.
    bool castFromFp32();
    // Casting from Custom float - first cast to fp32 then cast from fp32 to cast output type.
    // Custom float if a float type that is not fp32 (bf16 / fp16 for example)
    template <typename CastFromType>
    bool castFromCustomFloat();
    // Casting from int - first cast to fp32 then cast from fp32 to cast output type.
    template <typename CastFromType>
    bool castFromInt();
    template <typename CastFromType>
    bool castFromFp8();

    TensorPtr   m_castFromTensor = nullptr;
    TensorPtr   m_castToTensor   = nullptr;

    synDataType m_castFromType   = syn_type_na; // data type of input tensor
    synDataType m_castToType     = syn_type_na; // data type of output tensor

    unsigned    m_elementsNum    = 0;
};

