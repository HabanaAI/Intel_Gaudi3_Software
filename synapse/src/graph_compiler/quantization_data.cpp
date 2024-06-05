#include "quantization_data.h"
#include "infra/defs.h"
#include "types.h"

static_assert(quant_type_na + 1 == quant_type_max_including_na,
              "static arrays bellow rely on quant_type_max_including_na being last");

static std::array<QuantizationData, quant_type_max_including_na> getAllDefaultQuantizationData()
{
    std::array<QuantizationData, quant_type_max_including_na> defaultQuantization;
    for (int typeIdx = 0; typeIdx < quant_type_max_including_na; ++typeIdx)
    {
        auto type                 = static_cast<eQuantDataType>(typeIdx);
        defaultQuantization[type] = QuantizationData(type);
    }
    return defaultQuantization;
}

static std::array<QuantizationData::QuantizationParams, quant_type_max_including_na> getAllDefaultQuantizationParams()
{
    std::array<QuantizationData::QuantizationParams, quant_type_max_including_na> defaultQuantization;
    for (int typeIdx = 0; typeIdx < quant_type_max_including_na; ++typeIdx)
    {
        auto type = static_cast<eQuantDataType>(typeIdx);
        defaultQuantization[type].expBias =
            QuantizationData::getDefaultExpBias(QuantizationData::quantTypeToSynType(type));
    }
    return defaultQuantization;
}

// order of initialization matters as QuantizationData constructor
// depends on defaultQuantizationChannelParams.
const std::array<QuantizationData::QuantizationParams, quant_type_max_including_na>
    QuantizationData::defaultQuantizationChannelParams(getAllDefaultQuantizationParams());
const std::array<QuantizationData, quant_type_max_including_na>
    QuantizationData::defaultQuantizationData(getAllDefaultQuantizationData());

std::string QuantizationData::getDataTypeString(eQuantDataType t)
{
    constexpr const char* invalid = "invalid";
    switch (t)
    {
        case quant_type_int4:
            return "int4";
        case quant_type_uint4:
            return "uint4";
        case quant_type_int8:
            return "int8";
        case quant_type_uint8:
            return "uint8";
        case quant_type_int16:
            return "int16";
        case quant_type_uint16:
            return "uint16";
        case quant_type_int32:
            return "int32";
        case quant_type_uint32:
            return "uint32";
        case quant_type_int8fp:
            return "int8fp";
        case quant_type_int16ltd:
            return "int16ltd";
        case quant_type_float32:
            return "float32";
        case quant_type_bf16:
            return "bf16";
        case quant_type_fp16:
            return "fp16";
        case quant_type_fp8_143:
            return "hf8";
        case quant_type_fp8_152:
            return "f8";
        case quant_type_int64:
            return "int64";
        case quant_type_uint64:
            return "uint64";
        case quant_type_ufp16:
            return "ufp16";

        case quant_type_max:
        case quant_type_na:
            return invalid;
    }
    return invalid;
}

eQuantDataType QuantizationData::synTypeToQuantType(synDataType t)
{
    constexpr eQuantDataType invalid = quant_type_na;
    switch (t)
    {
        case syn_type_int4:
            return quant_type_int4;
        case syn_type_uint4:
            return quant_type_uint4;
        case syn_type_int8:
            return quant_type_int8;
        case syn_type_uint8:
            return quant_type_uint8;
        case syn_type_int16:
            return quant_type_int16;
        case syn_type_uint16:
            return quant_type_uint16;
        case syn_type_int32:
            return quant_type_int32;
        case syn_type_uint32:
            return quant_type_uint32;
        case syn_type_float:
            return quant_type_float32;
        case syn_type_bf16:
            return quant_type_bf16;
        case syn_type_fp16:
            return quant_type_fp16;
        case syn_type_fp8_143:
            return quant_type_fp8_143;
        case syn_type_fp8_152:
            return quant_type_fp8_152;
        case syn_type_int64:
            return quant_type_int64;
        case syn_type_uint64:
            return quant_type_uint64;
        case syn_type_ufp16:
            return quant_type_ufp16;
        case syn_type_na:
        case syn_type_tf32:
        case syn_type_hb_float:
        case syn_type_max:
            return invalid;
    }
    return invalid;
}

synDataType QuantizationData::quantTypeToSynType(eQuantDataType t)
{
    constexpr synDataType invalid = syn_type_na;
    switch (t)
    {
        case quant_type_int4:
            return syn_type_int4;
        case quant_type_uint4:
            return syn_type_uint4;
        case quant_type_int8:
            return syn_type_int8;
        case quant_type_uint8:
            return syn_type_uint8;
        case quant_type_int16:
            return syn_type_int16;
        case quant_type_uint16:
            return syn_type_uint16;
        case quant_type_int8fp:
            return syn_type_int8;
        case quant_type_int16ltd:
            return syn_type_int16;
        case quant_type_int32:
            return syn_type_int32;
        case quant_type_uint32:
            return syn_type_uint32;
        case quant_type_float32:
            return syn_type_float;
        case quant_type_bf16:
            return syn_type_bf16;
        case quant_type_fp16:
            return syn_type_fp16;
        case quant_type_fp8_143:
            return syn_type_fp8_143;
        case quant_type_fp8_152:
            return syn_type_fp8_152;
        case quant_type_int64:
            return syn_type_int64;
        case quant_type_uint64:
            return syn_type_uint64;
        case quant_type_ufp16:
            return syn_type_ufp16;

        case quant_type_max:
        case quant_type_na:
            return invalid;
    }
    return invalid;
}

unsigned QuantizationData::getDefaultExpBias(synDataType dtype)
{
    switch (dtype)
    {
        case syn_type_fp8_143:
            return QuantizationData::S_EXP_BIAS_143_DEFAULT;
        case syn_type_fp8_152:
            return QuantizationData::S_EXP_BIAS_152_DEFAULT;
        case syn_type_fp16:
            return QuantizationData::S_EXP_BIAS_FP16_DEFAULT;
        default:
            return 0;
    }
}

unsigned QuantizationData::getDefaultExpBias(eQuantDataType dtype)
{
    return defaultQuantizationChannelParams[dtype].expBias;
}

const QuantizationData::QuantizationParams& QuantizationData::getDefaultQuantizationChannelParams(synDataType dtype)
{
    return defaultQuantizationChannelParams[synTypeToQuantType(dtype)];
}

QuantizationData::QuantizationData(synDataType dataType) : m_qDataType(synTypeToQuantType(dataType))
{
    setDefaultExpBias();
}

QuantizationData::QuantizationData(eQuantDataType dataType) : m_qDataType(dataType)
{
    setDefaultExpBias();
}

QuantizationData::QuantizationData(const synQuantizationParams& quantInfo) :
        m_isUserQuantInfo(true),
        m_qDataType(synTypeToQuantType(quantInfo.m_qDataType))
{
    m_quantParams.resize(m_numChannels);
    m_quantParams[0].zp      = quantInfo.m_zp;
    m_quantParams[0].scale   = quantInfo.m_scale;
    m_quantParams[0].expBias = quantInfo.m_expBias;
}

QuantizationData::QuantizationData(const synQuantMetadata& quantInfo)
: m_isUserQuantInfo(true),
  m_isUserPCQuantInfo(true),
  m_qDataType(synTypeToQuantType(quantInfo.dataType)),
  m_numChannels(quantInfo.numZPScales)
{
    m_quantParams.resize(m_numChannels);
    for (unsigned i = 0; i < m_numChannels; i++)
    {
        m_quantParams[i].zp    = quantInfo.zpScales[i].zp;
        m_quantParams[i].scale = quantInfo.zpScales[i].scale;
    }
}

QuantizationData::QuantizationData(const synFpQuantMetadata& quantInfo)
: m_isUserQuantInfo(true),
  m_isUserPCQuantInfo(true),
  m_qDataType(synTypeToQuantType(quantInfo.dataType)),
  m_numChannels(quantInfo.numFpQuantParams)
{
    m_quantParams.resize(m_numChannels);
    for (unsigned i = 0; i < m_numChannels; i++)
    {
        m_quantParams[i].scale   = quantInfo.fpQuantParams[i].scale;
        m_quantParams[i].expBias = quantInfo.fpQuantParams[i].expBias;
    }
}

QuantizationData::QuantizationData(const synPerChannelQuantizationParams& quantInfo) :
        m_isUserPCQuantInfo(true),
        m_numChannels(quantInfo.m_numChannels)
{
    m_qDataType = synTypeToQuantType(quantInfo.m_qDataType);
    m_quantParams.resize(m_numChannels);
    for (unsigned i = 0; i < m_numChannels; i++)
    {
        m_quantParams[i].zp      = quantInfo.m_pcZps[i];
        m_quantParams[i].scale   = quantInfo.m_pcScales[i];
        m_quantParams[i].expBias = quantInfo.m_pcExpBias[i];
    }
}

void QuantizationData::reset(unsigned numChannels, eQuantDataType dtype)
{
    m_numChannels = numChannels;
    m_qDataType = dtype;
    m_quantParams.resize(numChannels);
    const unsigned defaultExpBias = defaultQuantizationChannelParams[dtype].expBias;
    for (auto& param : m_quantParams)
    {
        param.expBias = defaultExpBias;
    }
}

void QuantizationData::reset(unsigned numChannels, synDataType dtype)
{
    m_numChannels = numChannels;
    m_qDataType = synTypeToQuantType(dtype);
    m_quantParams.resize(numChannels);
    const unsigned defaultExpBias = defaultQuantizationChannelParams[m_qDataType].expBias;
    for (auto& param : m_quantParams)
    {
        param.expBias = defaultExpBias;
    }
}

bool QuantizationData::isPerChannel() const
{
    return m_numChannels > 1;
}

double QuantizationData::zp(unsigned index) const
{
    if (index == 0 && m_quantParams.empty()) return 0;
    HB_ASSERT(m_quantParams.size() > index, "index out of range");
    return m_quantParams[index].zp;
}

void QuantizationData::setZp(double zp, unsigned index)
{
    if (index == 0 && m_quantParams.empty()) m_quantParams.resize(1);
    HB_ASSERT(m_quantParams.size() > index, "index out of range");
    m_quantParams[index].zp = zp;
}

double QuantizationData::scale(unsigned index) const
{
    if (index == 0 && m_quantParams.empty()) return 1;
    HB_ASSERT(m_quantParams.size() > index, "index out of range");
    return m_quantParams[index].scale;
}

void QuantizationData::setScale(double scale, unsigned index)
{
    if (index == 0 && m_quantParams.empty()) m_quantParams.resize(1);
    HB_ASSERT(m_quantParams.size() > index, "index out of range");
    m_quantParams[index].scale = scale;
}

unsigned QuantizationData::expBias(unsigned index) const
{
    if (index == 0 && m_quantParams.empty())
    {
        return defaultQuantizationChannelParams[m_qDataType].expBias;
    }
    HB_ASSERT(m_quantParams.size() > index, "index out of range");
    return m_quantParams[index].expBias;
}

void QuantizationData::setExpBias(unsigned expBias, unsigned index)
{
    if (index == 0 && m_quantParams.empty()) m_quantParams.resize(1);
    HB_ASSERT(m_quantParams.size() > index, "index out of range");
    m_quantParams[index].expBias = expBias;
}

void QuantizationData::setDefaultExpBias(unsigned index)
{
    setExpBias(defaultQuantizationChannelParams[m_qDataType].expBias, index);
}

synDataType QuantizationData::getSynDataType() const
{
    return quantTypeToSynType(m_qDataType);
}

bool QuantizationData::operator==(const QuantizationData& other) const
{
    return m_numChannels == other.m_numChannels && m_qDataType == other.m_qDataType &&
           m_quantParams == other.m_quantParams;
}

const std::vector<double> QuantizationData::getZpVector() const
{
    uint32_t size = m_quantParams.size();
    if (size == 0)
    {
        std::vector<double> zpVec = {0};
        return zpVec;
    }
    else
    {
        std::vector<double> zpVec(size);
        for (unsigned i = 0; i < size; i++)
        {
            zpVec[i] = m_quantParams[i].zp;
        }
        return zpVec;
    }
}

const std::vector<double> QuantizationData::getScaleVector() const
{
    uint32_t size = m_quantParams.size();
    if (size == 0)
    {
        std::vector<double> scaleVec = {1};
        return scaleVec;
    }
    else
    {
        std::vector<double> scaleVec(size);
        for (unsigned i = 0; i < size; i++)
        {
            scaleVec[i] = m_quantParams[i].scale;
        }
        return scaleVec;
    }
}

const std::vector<unsigned> QuantizationData::getExpBiasVector() const
{
    std::vector<unsigned> expBiasVec = {1};
    if (uint32_t size = m_quantParams.size() ; size > 0)
    {
        expBiasVec.resize(size);
        for (unsigned i = 0; i < size; i++)
        {
            expBiasVec[i] = m_quantParams[i].expBias;
        }
    }
    return expBiasVec;
}

QuantizationData::QuantizationParams& QuantizationData::getChannelParams(unsigned index)
{
    HB_ASSERT(m_quantParams.size() > index, "index out of range");
    return m_quantParams[index];
}

const QuantizationData::QuantizationParams& QuantizationData::getChannelParams(unsigned index) const
{
    if (index == 0 && m_quantParams.empty())
    {
        return defaultQuantizationChannelParams[m_qDataType];
    }
    return m_quantParams[index];
}
