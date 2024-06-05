#pragma once

#include "types.h"
#include <cstdint>

enum eQuantDataType : uint8_t
{
    quant_type_int4,
    quant_type_uint4,
    quant_type_int8,
    quant_type_uint8,
    quant_type_int16,
    quant_type_uint16,
    quant_type_int8fp,
    quant_type_int16ltd,
    quant_type_int32,
    quant_type_uint32,
    quant_type_float32,
    quant_type_bf16,
    quant_type_fp16,
    quant_type_fp8_143,
    quant_type_fp8_152,
    quant_type_int64,
    quant_type_uint64,
    quant_type_ufp16,
    // max
    quant_type_max,
    // invalid
    quant_type_na
};
// needs to be equal to largest possible value of eQuantDataType
static constexpr unsigned quant_type_max_including_na = quant_type_na + 1;

class QuantizationData
{
public:
    struct QuantizationParams
    {
        double    zp       = 0;
        double    scale    = 1;
        unsigned  expBias  = S_EXP_BIAS_143_DEFAULT;

        bool operator==(const QuantizationParams& other) const { return zp      == other.zp    &&
                                                                        scale   == other.scale &&
                                                                        expBias == other.expBias; }
    };

    // avoid allocations for default first channel
    using QuantizationParamsVector = llvm_vecsmall::SmallVector<QuantizationParams, 1>;

    static std::string getDataTypeString(eQuantDataType t);
    static eQuantDataType synTypeToQuantType(synDataType t);
    static synDataType quantTypeToSynType(eQuantDataType t);
    static unsigned                                    getDefaultExpBias(synDataType dtype);
    static unsigned                                    getDefaultExpBias(eQuantDataType dtype);
    static const QuantizationData::QuantizationParams& getDefaultQuantizationChannelParams(synDataType dtype);

    QuantizationData() = default;
    QuantizationData(synDataType dataType);
    QuantizationData(eQuantDataType dataType);
    QuantizationData(const synQuantizationParams& quantInfo);
    QuantizationData(const synPerChannelQuantizationParams& quantInfo);
    QuantizationData(const synQuantMetadata& quantInfo);
    QuantizationData(const synFpQuantMetadata& quantInfo);
    QuantizationData(const QuantizationData& other) = default;

    void reset(unsigned numChannels, eQuantDataType dtype);
    void reset(unsigned numChannels, synDataType dtype);
    bool isPerChannel() const;
    double zp(unsigned index = 0) const;
    void setZp(double zp, unsigned index = 0);
    double scale(unsigned index = 0) const;
    void setScale(double scale, unsigned index = 0);
    unsigned expBias(unsigned index = 0) const;
    void setExpBias(unsigned expBias, unsigned index = 0);
    void setDefaultExpBias(unsigned index = 0);
    synDataType getSynDataType() const;
    uint32_t                       numOfParams() const { return m_quantParams.size(); }
    const std::vector<double>      getZpVector() const;
    const std::vector<double>      getScaleVector() const;
    const std::vector<unsigned>   getExpBiasVector() const;
    const QuantizationParamsVector&        getQuantParamsVector() const { return m_quantParams; };
    const QuantizationParams&              getChannelParams(unsigned index = 0) const;
    QuantizationParams&                    getChannelParams(unsigned index = 0);
    void setQuantParamsVector(const QuantizationParamsVector& params) { m_quantParams = params; }
    bool operator==(const QuantizationData& other) const;
    bool operator!=(const QuantizationData& other) const { return !((*this) == other); }

    bool m_isUserQuantInfo = false;   // true if quantization info was provided by the user, otherwise false
    bool m_isUserPCQuantInfo = false; // true if per channel quantization info was provided by the user, otherwise false
    eQuantDataType m_qDataType = quant_type_na;
    unsigned       m_numChannels = 1;

    static constexpr unsigned S_EXP_BIAS_143_DEFAULT  = 7;  // default value for fp8_143
    static constexpr unsigned S_EXP_BIAS_152_DEFAULT  = 15; // default value for fp8_152
    static constexpr unsigned S_EXP_BIAS_FP16_DEFAULT = 15; // default value for fp16

    static const std::array<QuantizationData, quant_type_max_including_na>   defaultQuantizationData;
    static const std::array<QuantizationParams, quant_type_max_including_na> defaultQuantizationChannelParams;

private:
    QuantizationParamsVector m_quantParams;
};

typedef std::vector<synQuantDynamicRange> RangesVector;
struct PerChannelDynamicRange
{
    RangesVector ranges;
    unsigned     numChannels;
    bool         isSet = false;
};

struct ConvQuantizationParams
{
    QuantizationData x;
    QuantizationData w;
    QuantizationData residual;
    QuantizationData out;
};