#include "cast_utils.hpp"

#include "data_type_utils.h"
#include "include/mme_common/mme_common_enum.h"
#include "mme_reference/data_types/fp8.h"
#include "quantization_utils.h"

/*=== Casting utils ===*/
template <typename CastFromType>
void castIntToFp32(CastFromType*           fromBuffer,
                   float*                  toBuffer,
                   unsigned                elementsNum,
                   const QuantizationData& fromQuantInfo)
{
    // don't allow compilation if types don't match function
    static_assert(std::is_integral<CastFromType>());
#pragma omp parallel for
    for (unsigned i = 0; i < elementsNum ; i++)
    {
        toBuffer[i] = QuantizationUtils::IntToRealValue<CastFromType>(fromBuffer[i],
                                                                      fromQuantInfo.scale(),
                                                                      fromQuantInfo.zp());
    }
}

template <typename CastToType>
void castFp32ToInt(float*                  fromBuffer,
                   CastToType*             toBuffer,
                   unsigned                elementsNum,
                   const QuantizationData& toQuantInfo)
{
    // don't allow compilation if types don't match function
    static_assert(std::is_integral<CastToType>());
#pragma omp parallel for
    for (unsigned i = 0; i < elementsNum; i++)
    {
        toBuffer[i] = QuantizationUtils::RealToIntValue<CastToType>(fromBuffer[i],
                                                                    toQuantInfo.scale(),
                                                                    toQuantInfo.zp(),
                                                                    toQuantInfo.m_qDataType);
    }
}

template <typename CastFromType, typename CastToType>
void castAnyFloatToAnyFloat(CastFromType* fromBuffer,
                            CastToType*   toBuffer,
                            unsigned      elementsNum)
{
    // don't allow compilation if types don't match function
    static_assert(!std::is_integral<CastToType>());
#pragma omp parallel for
    for (unsigned i = 0; i < elementsNum; i++)
    {
        // cast first to fp32 then to target type
        toBuffer[i] = CastToType(float(fromBuffer[i]));
    }
}

template <typename CastFromType, typename CastToType>
void castIntToCustomFloat(CastFromType*           fromBuffer,
                          CastToType*             toBuffer,
                          unsigned                elementsNum,
                          const QuantizationData& fromQuantInfo)
{
    // don't allow compilation if types don't match function
    static_assert(std::is_integral<CastFromType>() && !std::is_integral<CastToType>());
#pragma omp parallel for
    for (unsigned i = 0; i < elementsNum ; i++)
    {
        // cast first to fp32 then to target type
        float tempFloat = QuantizationUtils::IntToRealValue<CastFromType>(fromBuffer[i],
                                                                          fromQuantInfo.scale(),
                                                                          fromQuantInfo.zp());
        toBuffer[i] = CastToType(tempFloat);
    }
}

template <typename CastFromType, typename CastToType>
void castCustomFloatToInt(CastFromType*           fromBuffer,
                          CastToType*             toBuffer,
                          unsigned                elementsNum,
                          const QuantizationData& toQuantInfo)
{
    // don't allow compilation if types don't match function
    static_assert(!std::is_integral<CastFromType>() && std::is_integral<CastToType>());
#pragma omp parallel for
    for (unsigned i = 0; i < elementsNum ; i++)
    {
        // cast first to fp32 than to target type
        float tempFloat = float(fromBuffer[i]);
        toBuffer[i]  = QuantizationUtils::RealToIntValue<CastToType>(tempFloat, toQuantInfo.scale(), toQuantInfo.zp());

    }
}

template <typename CastFromType, typename CastToType>
void castIntToInt(CastFromType*           fromBuffer,
                  CastToType*             toBuffer,
                  unsigned                elementsNum,
                  const QuantizationData& fromQuantInfo,
                  const QuantizationData& toQuantInfo)
{
    // don't allow compilation if types don't match function
    static_assert(std::is_integral<CastFromType>() && std::is_integral<CastToType>());
#pragma omp parallel for
    for (unsigned i = 0; i < elementsNum ; i++)
    {
        // cast first to fp32 than to target type
        float tempFloat = QuantizationUtils::IntToRealValue<CastFromType>(fromBuffer[i],
                                                                          fromQuantInfo.scale(),
                                                                          fromQuantInfo.zp());

        toBuffer[i]  = QuantizationUtils::RealToIntValue<CastToType>(tempFloat, toQuantInfo.scale(), toQuantInfo.zp());
    }
}

template<typename CastFromType, typename CastToType>
void castAnyFloatToFp8(CastFromType*                 fromBuffer,
                       CastToType*                   toBuffer,
                       unsigned                      elementsNum,
                       const QuantizationData&       toQuantInfo,
                       const MmeCommon::RoundingMode rm = MmeCommon::RoundingMode::RoundToNearest)
{
    // don't allow compilation if types don't match function
    static_assert(!std::is_integral<CastFromType>());
#pragma omp parallel for
    for (unsigned i = 0; i < elementsNum; i++)
    {
        // cast first to fp32 then to target type
        // using mme reference fp8 for the conversion
        CastToType tempFloat8(float(fromBuffer[i]), rm, toQuantInfo.expBias());
        toBuffer[i] = tempFloat8;
    }
}

template<typename CastFromType, typename CastToType>
void castIntToFp8(CastFromType*                 fromBuffer,
                  CastToType*                   toBuffer,
                  unsigned                      elementsNum,
                  const QuantizationData&       fromQuantInfo,
                  const QuantizationData&       toQuantInfo,
                  const MmeCommon::RoundingMode rm = MmeCommon::RoundingMode::RoundToNearest)
{
    // don't allow compilation if types don't match function
    static_assert(std::is_integral<CastFromType>());
#pragma omp parallel for
    for (unsigned i = 0; i < elementsNum; i++)
    {
        // cast first to fp32 then to target type
        float tempFloat =
            QuantizationUtils::IntToRealValue<CastFromType>(fromBuffer[i], fromQuantInfo.scale(), fromQuantInfo.zp());
        // using mme reference fp8 for the conversion
        CastToType tempFloat8(tempFloat, rm, toQuantInfo.expBias());
        toBuffer[i] = tempFloat8;
    }
}

/*=== CpuCaster class ===*/
CpuCaster::CpuCaster(TensorPtr castFromTensor, TensorPtr castToTensor) :
  m_castFromTensor(castFromTensor),
  m_castToTensor(castToTensor),
  m_castFromType(castFromTensor->getElementType()),
  m_castToType(castToTensor->getElementType()),
  m_elementsNum(castToTensor->getTotalElements())
{}

void CpuCaster::setCastDataTypes(synDataType castFromType, synDataType castToType)
{
    m_castFromType = castFromType;
    m_castToType   = castToType;
}

bool CpuCaster::castFromFp32()
{
    LOG_TRACE(GC, "Casting from fp32 to {}", getStringFromSynDataType(m_castToType));
    float* castFromBuffer    = reinterpret_cast<float*>(m_castFromTensor->getData());
    auto   castToQuantInfo   = m_castToTensor->getQuantizationParams(m_castToType);
    switch (m_castToType)
    {
        case syn_type_uint4:
        case syn_type_int4:
            LOG_WARN(GC, "Cast to type {} is currently not supported in casting on CPU from int", m_castToType);
            return false;
        case syn_type_fp8_152:
            castAnyFloatToFp8(castFromBuffer,
                              reinterpret_cast<fp8_152_t*>(m_castToTensor->getData()),
                              m_elementsNum,
                              castToQuantInfo);
            break;
        case syn_type_fp8_143:
            castAnyFloatToFp8(castFromBuffer,
                              reinterpret_cast<fp8_143_t*>(m_castToTensor->getData()),
                              m_elementsNum,
                              castToQuantInfo);
            break;
        case syn_type_float:
            castAnyFloatToAnyFloat(castFromBuffer,
                                   reinterpret_cast<float*>(m_castToTensor->getData()),
                                   m_elementsNum);
            break;
        case syn_type_bf16:
            castAnyFloatToAnyFloat(castFromBuffer,
                                   reinterpret_cast<bfloat16*>(m_castToTensor->getData()),
                                   m_elementsNum);
            break;
        case syn_type_fp16:
            castAnyFloatToAnyFloat(castFromBuffer,
                                   reinterpret_cast<fp16_t*>(m_castToTensor->getData()),
                                   m_elementsNum);
            break;
        case syn_type_int8:
            castFp32ToInt(castFromBuffer,
                          reinterpret_cast<int8_t*>(m_castToTensor->getData()),
                          m_elementsNum,
                          castToQuantInfo);
            break;
        case syn_type_uint8:
            castFp32ToInt(castFromBuffer,
                          reinterpret_cast<uint8_t*>(m_castToTensor->getData()),
                          m_elementsNum,
                          castToQuantInfo);
            break;
        case syn_type_int16:
            castFp32ToInt(castFromBuffer,
                          reinterpret_cast<int16_t*>(m_castToTensor->getData()),
                          m_elementsNum,
                          castToQuantInfo);
            break;
        case syn_type_uint16:
            castFp32ToInt(castFromBuffer,
                          reinterpret_cast<uint16_t*>(m_castToTensor->getData()),
                          m_elementsNum,
                          castToQuantInfo);
            break;
        case syn_type_int32:
            castFp32ToInt(castFromBuffer,
                          reinterpret_cast<int32_t*>(m_castToTensor->getData()),
                          m_elementsNum,
                          castToQuantInfo);
            break;
        case syn_type_uint32:
            castFp32ToInt(castFromBuffer,
                          reinterpret_cast<uint32_t*>(m_castToTensor->getData()),
                          m_elementsNum,
                          castToQuantInfo);
            break;
        default:
            HB_ASSERT(false, "Unexpected syn data type to cast to {}", getStringFromSynDataType(m_castToType));
    }
    return true;
}

template <typename CastFromType, typename CastToType>
void castFp8ToAnyFloat(CastFromType*           fromBuffer,
                       CastToType*             toBuffer,
                       unsigned                elementsNum,
                       const QuantizationData& fromQuantInfo)
{
    // don't allow compilation if types don't match function
    static_assert(!std::is_integral<CastToType>());
#pragma omp parallel for
    for (unsigned i = 0; i < elementsNum; i++)
    {
        // cast first to fp32 then to target type
        toBuffer[i] = CastToType(fromBuffer[i].toFloat(fromQuantInfo.expBias()));
    }
}

template <typename CastFromType>
bool CpuCaster::castFromCustomFloat()
{
    LOG_TRACE(GC,
              "Casting from custom float type {} to {}",
              getStringFromSynDataType(m_castFromType),
              getStringFromSynDataType(m_castToType));
    CastFromType* castFromBuffer   = reinterpret_cast<CastFromType*>(m_castFromTensor->getData());
    auto          castToQuantInfo  = m_castToTensor->getQuantizationParams(m_castToType);
    switch (m_castToType)
    {
        case syn_type_uint4:
        case syn_type_int4:
            LOG_WARN(GC, "Cast to type {} is currently not supported in casting on CPU from custom float", m_castToType);
            return false;
        case syn_type_fp8_152:
            castAnyFloatToFp8(castFromBuffer,
                              reinterpret_cast<fp8_152_t*>(m_castToTensor->getData()),
                              m_elementsNum,
                              castToQuantInfo);
            break;
        case syn_type_fp8_143:
            castAnyFloatToFp8(castFromBuffer,
                              reinterpret_cast<fp8_143_t*>(m_castToTensor->getData()),
                              m_elementsNum,
                              castToQuantInfo);
            break;
        case syn_type_bf16:
            castAnyFloatToAnyFloat(castFromBuffer,
                                   reinterpret_cast<bfloat16*>(m_castToTensor->getData()),
                                   m_elementsNum);
            break;
        case syn_type_fp16:
            castAnyFloatToAnyFloat(castFromBuffer,
                                   reinterpret_cast<fp16_t*>(m_castToTensor->getData()),
                                   m_elementsNum);
            break;
        case syn_type_float:
            castAnyFloatToAnyFloat(castFromBuffer,reinterpret_cast<float*>(m_castToTensor->getData()), m_elementsNum);
            break;
        case syn_type_int8:
            castCustomFloatToInt(castFromBuffer,
                                 reinterpret_cast<int8_t*>(m_castToTensor->getData()),
                                 m_elementsNum,
                                 castToQuantInfo);
            break;
        case syn_type_uint8:
            castCustomFloatToInt(castFromBuffer,
                                 reinterpret_cast<uint8_t*>(m_castToTensor->getData()),
                                 m_elementsNum,
                                 castToQuantInfo);
            break;
        case syn_type_int16:
            castCustomFloatToInt(castFromBuffer,
                                 reinterpret_cast<int16_t*>(m_castToTensor->getData()),
                                 m_elementsNum,
                                 castToQuantInfo);
            break;
        case syn_type_uint16:
            castCustomFloatToInt(castFromBuffer,
                                 reinterpret_cast<uint16_t*>(m_castToTensor->getData()),
                                 m_elementsNum,
                                 castToQuantInfo);
            break;
        case syn_type_int32:
            castCustomFloatToInt(castFromBuffer,
                                 reinterpret_cast<int32_t*>(m_castToTensor->getData()),
                                 m_elementsNum,
                                 castToQuantInfo);
            break;
        case syn_type_uint32:
            castCustomFloatToInt(castFromBuffer,
                                 reinterpret_cast<uint32_t*>(m_castToTensor->getData()),
                                 m_elementsNum,
                                 castToQuantInfo);
            break;
        default:
            HB_ASSERT(false, "Unexpected syn data type to cast to {}", getStringFromSynDataType(m_castToType));
    }
    return true;
}

template<typename CastFromType>
bool CpuCaster::castFromFp8()
{
    LOG_TRACE(GC,
              "Casting from type {} to {}",
              getStringFromSynDataType(m_castFromType),
              getStringFromSynDataType(m_castToType));
    CastFromType* castFromBuffer    = reinterpret_cast<CastFromType*>(m_castFromTensor->getData());
    auto          castFromQuantInfo = m_castFromTensor->getQuantizationParams(m_castFromType);
    switch (m_castToType)
    {
        case syn_type_uint4:
        case syn_type_int4:
        case syn_type_int8:
        case syn_type_uint8:
        case syn_type_int16:
        case syn_type_uint16:
        case syn_type_int32:
        case syn_type_uint32:
        case syn_type_fp8_152:
        case syn_type_fp8_143:
            LOG_WARN(GC,
                     "Cast to type {} is currently not supported in casting on CPU from {}",
                     m_castToType,
                     m_castFromType);
            return false;
        case syn_type_float:
            castFp8ToAnyFloat(castFromBuffer,
                              reinterpret_cast<float*>(m_castToTensor->getData()),
                              m_elementsNum,
                              castFromQuantInfo);
            break;
        case syn_type_bf16:
            castFp8ToAnyFloat(castFromBuffer,
                              reinterpret_cast<bfloat16*>(m_castToTensor->getData()),
                              m_elementsNum,
                              castFromQuantInfo);
            break;
        case syn_type_fp16:
            castFp8ToAnyFloat(castFromBuffer,
                              reinterpret_cast<fp16_t*>(m_castToTensor->getData()),
                              m_elementsNum,
                              castFromQuantInfo);
            break;
        default:
            HB_ASSERT(false, "Unexpected syn data type to cast to {}", getStringFromSynDataType(m_castToType));
    }
    return true;
}

template <typename CastFromType>
bool CpuCaster::castFromInt()
{
    LOG_TRACE(GC,
              "Casting from int type {} to {}",
              getStringFromSynDataType(m_castFromType),
              getStringFromSynDataType(m_castToType));
    CastFromType* castFromType = reinterpret_cast<CastFromType*>(m_castFromTensor->getData());
    auto castFromQuantInfo = m_castFromTensor->getQuantizationParams(m_castFromType);
    auto castToQuantInfo   = m_castToTensor->getQuantizationParams(m_castToType);
    switch (m_castToType)
    {
        case syn_type_uint4:
        case syn_type_int4:
            LOG_WARN(GC, "Cast to type {} is currently not supported in casting on CPU from int", m_castToType);
            return false;
        case syn_type_fp8_143:
            castIntToFp8(castFromType,
                         reinterpret_cast<fp8_143_t*>(m_castToTensor->getData()),
                         m_elementsNum,
                         castFromQuantInfo,
                         castToQuantInfo);
            break;
        case syn_type_fp8_152:
            castIntToFp8(castFromType,
                         reinterpret_cast<fp8_152_t*>(m_castToTensor->getData()),
                         m_elementsNum,
                         castFromQuantInfo,
                         castToQuantInfo);
            break;
        case syn_type_bf16:
            castIntToCustomFloat(castFromType,
                                 reinterpret_cast<bfloat16*>(m_castToTensor->getData()),
                                 m_elementsNum,
                                 castFromQuantInfo);
            break;
        case syn_type_fp16:
            castIntToCustomFloat(castFromType,
                                 reinterpret_cast<fp16_t*>(m_castToTensor->getData()),
                                 m_elementsNum,
                                 castFromQuantInfo);
            break;
        case syn_type_float:
            castIntToFp32(castFromType,
                          reinterpret_cast<float*>(m_castToTensor->getData()),
                          m_elementsNum,
                          castFromQuantInfo);
            break;
        case syn_type_int8:
            castIntToInt(castFromType,
                         reinterpret_cast<int8_t*>(m_castToTensor->getData()),
                         m_elementsNum,
                         castFromQuantInfo,
                         castToQuantInfo);
            break;
        case syn_type_uint8:
            castIntToInt(castFromType,
                         reinterpret_cast<uint8_t*>(m_castToTensor->getData()),
                         m_elementsNum,
                         castFromQuantInfo,
                         castToQuantInfo);
            break;
        case syn_type_int16:
            castIntToInt(castFromType,
                         reinterpret_cast<int16_t*>(m_castToTensor->getData()),
                         m_elementsNum,
                         castFromQuantInfo,
                         castToQuantInfo);
            break;
        case syn_type_uint16:
            castIntToInt(castFromType,
                         reinterpret_cast<uint16_t*>(m_castToTensor->getData()),
                         m_elementsNum,
                         castFromQuantInfo,
                         castToQuantInfo);
            break;
        case syn_type_int32:
            castIntToInt(castFromType,
                         reinterpret_cast<int32_t*>(m_castToTensor->getData()),
                         m_elementsNum,
                         castFromQuantInfo,
                         castToQuantInfo);
            break;
        case syn_type_uint32:
            castIntToInt(castFromType,
                         reinterpret_cast<uint32_t*>(m_castToTensor->getData()),
                         m_elementsNum,
                         castFromQuantInfo,
                         castToQuantInfo);
            break;
        default:
            HB_ASSERT(false, "Unexpected syn data type to cast to {}", getStringFromSynDataType(m_castToType));
    }
    return true;
}

/*
 * Perform cast on cpu and save the cast output result in the output tensor buffer.
 *
 * Casting from input type to output type has different flows, depend on the types.
 * We only know how to cast directly to/from fp32 (dequatization/quantization for ints, c-style cast for custom floats).
 * So if the input type is not fp32, first we cast it to fp32, then to the target type.
 * Input type flows:
 * 1. If input type is int - first cast to fp32 using input quantization params (dequant)
 * 2. If input type is custom float (non fp32 float) - first cast to fp32 with c-style cast.
 * 3. If input type is fp32 - cast directly to output type.
 * After we have verified that we are casting from fp32, there are output type flows :
 * 1. If output type is int - cast from fp32 using output quantization params (quant)
 * 2. If output type is custom float (non fp32 float) - cast from fp32 with c-style cast.
 * 3. if output type is fp32 - the cast is completed.
 */
bool CpuCaster::doCast()
{
    LOG_TRACE(GC,
              "Running cast on cpu. Cast from type - {}, cast from buffer type - {}, cast to type  - {},"
              "number of casted elements - {}",
              getStringFromSynDataType(m_castFromType),
              getStringFromSynDataType(m_castFromTensor->getBufferDataType()),
              getStringFromSynDataType(m_castToType),
              m_elementsNum);
    HB_ASSERT(m_castToTensor->getData() != nullptr, "Cast output tensor has null buffer");

    if (m_castFromTensor->getBufferDataType() == syn_type_float)
    {
        // This is a simple case similar to tensor quantization / float conversion
        return castFromFp32();
    }
    else
    {
        // Casting from non-fp32 buffer requires additional handling, according to cast source type
        switch (m_castFromType)
        {
            case syn_type_uint4:
            case syn_type_int4:
                LOG_WARN(GC, "Cast from type {} is currently not supported in casting on CPU", m_castFromType);
                return false;
            case syn_type_fp8_143:
                castFromFp8<fp8_143_t>();
                break;
            case syn_type_fp8_152:
                castFromFp8<fp8_152_t>();
                break;
            case syn_type_bf16:
                castFromCustomFloat<bfloat16>();
                break;
            case syn_type_fp16:
                castFromCustomFloat<fp16_t>();
                break;
            case syn_type_int8:
                castFromInt<int8_t>();
                break;
            case syn_type_uint8:
                castFromInt<uint8_t>();
                break;
            case syn_type_int16:
                castFromInt<int16_t>();
                break;
            case syn_type_uint16:
                castFromInt<uint16_t>();
                break;
            case syn_type_int32:
                castFromInt<int32_t>();
                break;
            case syn_type_uint32:
                castFromInt<uint32_t>();
                break;
            default:
                HB_ASSERT(false, "Unexpected syn data type to cast from {}", getStringFromSynDataType(m_castFromType));
        }
    }
    return true;
}
