#include "node_factory.h"
#include "habana_graph.h"
#include "data_type_utils.h"
#include "quant_info_calculator.h"
#include "synapse_common_types.h"

std::string_view getStringFromSynDataType(synDataType type)
{
    switch (type)
    {
        case syn_type_int4:
            return "int4";
        case syn_type_uint4:
            return "uint4";
        case syn_type_fixed:
            return "int8";
        case syn_type_uint8:
            return "uint8";
        case syn_type_bf16:
            return "bf16";
        case syn_type_fp16:
            return "float16";
        case syn_type_single:
            return "float32";
        case syn_type_int16:
            return "int16";
        case syn_type_uint16:
            return "uint16";
        case syn_type_int32:
            return "int32";
        case syn_type_uint32:
            return "uint32";
        case syn_type_fp8_152:
            return "float8";
        case syn_type_fp8_143:
            return "hfloat8";
        case syn_type_tf32:
            return "tf32";
        case syn_type_hb_float:
            return "tf32";
        case syn_type_int64:
            return "int64";
        case syn_type_uint64:
            return "uint64";
        default:
            return "invalid";
    }
}

synDataType getSynDataTypeFromString(const std::string& type)
{
    std::string lower_type(type);
    std::transform(lower_type.begin(), lower_type.end(), lower_type.begin(), ::tolower);

    if (lower_type == "int4")
    {
        return syn_type_int4;
    }
    else if (lower_type == "uint4")
    {
        return syn_type_uint4;
    }
    else if (lower_type == "hf8")
    {
        return syn_type_fp8_143;
    }
    else if (lower_type == "f8")
    {
        return syn_type_fp8_152;
    }
    else if (lower_type == "int8")
    {
        return syn_type_fixed;
    }
    else if (lower_type == "uint8")
    {
        return syn_type_uint8;
    }
    else if (lower_type == "bf16" || lower_type == "bfloat16")
    {
        return syn_type_bf16;
    }
    else if (lower_type == "float16")
    {
        return syn_type_fp16;
    }
    else if (lower_type == "float32")
    {
        return syn_type_single;
    }
    else if (lower_type == "int16")
    {
        return syn_type_int16;
    }
    else if (lower_type == "uint16")
    {
        return syn_type_uint16;
    }
    else if (lower_type == "int32")
    {
        return syn_type_int32;
    }
    else if (lower_type == "uint32")
    {
        return syn_type_uint32;
    }
    else if (lower_type == "int64")
    {
        return syn_type_int64;
    }
    else if (lower_type == "uint64")
    {
        return syn_type_uint64;
    }
    return syn_type_na;
}

std::string_view getDtypeSuffixFromSynDataType(synDataType type)
{
    switch (type)
    {
        case syn_type_int4:
            return "i4";
        case syn_type_uint4:
            return "u4";
        case syn_type_fixed:
            return "i8";
        case syn_type_uint8:
            return "u8";
        case syn_type_bf16:
            return "bf16";
        case syn_type_fp16:
            return "f16";
        case syn_type_single:
            return "f32";
        case syn_type_int16:
            return "i16";
        case syn_type_uint16:
            return "u16";
        case syn_type_int32:
            return "i32";
        case syn_type_uint32:
            return "u32";
        case syn_type_int64:
            return "i64";
        case syn_type_uint64:
            return "u64";
        case syn_type_fp8_143:
            return "hf8";
        case syn_type_fp8_152:
            return "f8";
        case syn_type_hb_float:
            return "f32";
        case syn_type_na:
        default:
            break;
    }
    return "";
}

std::string_view extractDtypeFromGUID(std::string_view guid)
{
    std::size_t found = guid.find_last_of('_');
    if (found != std::string::npos)
    {
        std::string_view dtype = guid.substr(found + 1);
        // make sure data type is valid and return it
        if (isGUIDDataTypeSupported(dtype))
        {
            return dtype;
        }
    }
    return "";
}

bool isSameBitRepresentation(synDataType typeA, synDataType typeB)
{
    constexpr auto validDatatypesMask = syn_type_uint8 | syn_type_int8 | syn_type_uint16 | syn_type_int16 |
                                        syn_type_uint32 | syn_type_int32 | syn_type_uint64 | syn_type_int64;
    if (typeA == typeB)
    {
        return true;
    }

    bool isValidTypes = (validDatatypesMask & typeB) != 0 && (validDatatypesMask & typeA) != 0;
    return isValidTypes ? dataTypeSizeInBytes(typeA) == dataTypeSizeInBytes(typeB) : false;
}

std::string_view extractDtypeFromCastGUID(std::string_view guid)
{
    // cast guid is of one of the following forms:
    // 1. "cast_to_<dtype>"
    // 2. "cast_<precision>_to_<dtype>"
    // here we extract the user node precision (if present) from the guid
    auto found = guid.rfind("_to_");
    if (found != std::string_view::npos)
    {
        std::string_view castGuid = guid.substr(0, found);
        return extractDtypeFromGUID(castGuid);
    }
    return "";
}

std::string_view extractGUIDFromFullGUID(std::string_view guid)
{
    std::size_t dTypeStart = guid.find('_');
    if (dTypeStart == std::string_view::npos) return guid;
    std::size_t dTypeEnd;
    do
    {
        dTypeEnd                      = guid.find('_', dTypeStart + 1);
        auto                   length = (dTypeEnd == std::string_view::npos ? guid.length() : dTypeEnd) - dTypeStart - 1;
        const std::string_view dtype(guid.data() + dTypeStart + 1, length);
        if (isGUIDDataTypeSupported(dtype))
        {
            return guid.substr(0, dTypeStart);
        }
        dTypeStart = dTypeEnd;
    } while (dTypeStart != std::string_view::npos);
    return guid;
}

std::string getCastGUID(synDataType from, synDataType to)
{
    return fmt::format("cast_{}_to_{}", getDtypeSuffixFromSynDataType(from), getDtypeSuffixFromSynDataType(to));
}

synDataType getQuantDataType(HabanaGraph& g, const TensorPtr& tensor, const NodePtr& consumer)
{
    // check for specific guids and their input index requirements
    const std::string_view guidWithoutDType = extractGUIDFromFullGUID(consumer->getGUID());
    const unsigned    inputIndex       = consumer->getInputIndexOfTensor(tensor);

    synDataType ret = getRequiredInputDataTypeByIndex(guidWithoutDType, inputIndex);
    if (ret != syn_type_na)
    {
        return ret;
    }

    // check for node precision or node type min precision
    synDataType precision = consumer->getNodePrecision();
    if (precision != syn_type_na)
    {
        return precision;
    }
    return g.getNodeTypeMinPrecision(consumer);
}

// TODO: [SW-38168] Query this information from the glue code when perf lib support for the subject will be achieved
synDataType getRequiredInputDataTypeByIndex(std::string_view guid, const unsigned index)
{
    std::string_view guidWithoutDType = extractGUIDFromFullGUID(guid);
    std::string lower_guid(guidWithoutDType);
    std::transform(lower_guid.begin(), lower_guid.end(), lower_guid.begin(), ::tolower);

    LOG_TRACE(DATA_TYPES, "{}: guid {} index: {}", HLLOG_FUNC, lower_guid, index);

    // roi pooling's rois input is always int16, since it contains indices
    if (lower_guid == NodeFactory::maxPoolRoiNodeTypeName && index == 1)
    {
        return syn_type_int16;
    }
    // indices tensor is always int32
    else if (lower_guid == "max_unpool_2d" && index == 1)
    {
        return syn_type_int32;
    }
    // the model-parameters must stay in FP32
    else if (lower_guid == NodeFactory::batchNormNodeTypeName && (index == 1 || index == 2 || index == 3 || index == 4))
    {
        return syn_type_single;
    }
    // indices tensor is always int32
    else if (lower_guid == "roialign" && index == 2)
    {
        return syn_type_int32;
    }
    // indices tensor is always int32
    else if (lower_guid == "take" && index == 1)
    {
        return syn_type_int32;
    }
    // embedding's IFM is always int16, since it contains indices
    else if (lower_guid == NodeFactory::embeddingNodeTypeName && index == 1)
    {
        return syn_type_int32;
    }
    // sequence reverse's SEQUENCE_LENS input is always int16, since it contains indices
    else if (lower_guid == NodeFactory::sequenceReverseNodeTypeName && index == 1)
    {
        return syn_type_int16;
    }
    // indices tensor is always int32
    else if (lower_guid == "scatter" && index == 1)
    {
        return syn_type_int32;
    }
    // indices tensor is always int32
    else if (lower_guid == "scatter_nd" && index == 1)
    {
        return syn_type_int32;
    }
    // indices tensor is always int32
    else if (lower_guid == "gather_elements" && index == 1)
    {
        return syn_type_int32;
    }
    // indices tensor is always int32
    else if (lower_guid == "gather_nd" && index == 1)
    {
        return syn_type_int32;
    }
    else if (lower_guid == "one_hot")
    {
        HB_ASSERT(index == 0, "input tensor is always int32 since it's indices");
        return syn_type_int32;
    }
    // indices tensor is always int32
    else if (lower_guid == "crop_and_resize" && index == 2)
    {
        return syn_type_int32;
    }
    else if (lower_guid == NodeFactory::cropMirorNormNodeTypeName && (index == 1 || index == 2 ))
    {
        return syn_type_float;
    }
    return syn_type_na;
}

// TODO: [SW-38168] Query this information from the glue code when perf lib support for the subject will be achieved
synDataType getRequiredOutputDataTypeByIndex(const std::string& guid, const unsigned index)
{
    const unsigned BEAM_SEARCH_OUTPUT_INDICES = 1;

    std::string lowerGuid(guid);
    std::transform(lowerGuid.begin(), lowerGuid.end(), lowerGuid.begin(), ::tolower);

    LOG_TRACE(DATA_TYPES, "{}: guid {} index: {}", HLLOG_FUNC, lowerGuid, index);

    std::string_view guidWithoutDType = extractGUIDFromFullGUID(lowerGuid);

    if (guidWithoutDType == "argmax" || guidWithoutDType == "argmin")
    {
        return syn_type_int32;
    }
    else if (guidWithoutDType == "isnan" || guidWithoutDType == "isinf")
    {
        return syn_type_int8;
    }
    else if (guidWithoutDType == NodeFactory::beamSearchNodeTypeName && index == BEAM_SEARCH_OUTPUT_INDICES)
    {
        return syn_type_int16;
    }
    else if (lowerGuid.find("cast_") == 0)
    {
        LOG_TRACE(DATA_TYPES, "setting output of cast kernel according to its guid-specified to-dtype: {}", guid);
        // return the "to" dtype
        return getSynDataTypeFromDtypeSuffix(extractDtypeFromGUID(guid));
    }
    return syn_type_na;
}

bool isQuantDtype(std::string_view dtype)
{
    return isQuantDtype(getSynDataTypeFromDtypeSuffix(dtype));
}

bool isQuantDtype(synDataType dtype)
{
    eQuantDataType        type                = QuantizationData::synTypeToQuantType(dtype);
    const size_t          supportedDTypesSize = sizeof(QuantInfoCalculator::supportedDTypes) / sizeof(eQuantDataType);
    const eQuantDataType* end                 = QuantInfoCalculator::supportedDTypes + supportedDTypesSize;

    if (std::find(QuantInfoCalculator::supportedDTypes, end, type) != end)
    {
        return true;
    }

    return false;
}

// TODO: [SW-166081] Remove when fuser is moved to protocolIR
synDataType translateTensorDataType(gcapi::TensorDataType_t type, const synDataType defaultType)
{
    switch (type)
    {
        case gcapi::TensorDataType_t::DATA_I4:
            return syn_type_int4;

        case gcapi::TensorDataType_t::DATA_I8:
            return syn_type_fixed;

        case gcapi::TensorDataType_t::DATA_U4:
            return syn_type_uint4;

        case gcapi::TensorDataType_t::DATA_U8:
            return syn_type_uint8;

        case gcapi::TensorDataType_t::DATA_I16:
            return syn_type_int16;

        case gcapi::TensorDataType_t::DATA_U16:
            return syn_type_uint16;

        case gcapi::TensorDataType_t::DATA_I32:
            return syn_type_int32;

        case gcapi::TensorDataType_t::DATA_I64:
            return syn_type_int64;

        case gcapi::TensorDataType_t::DATA_F8_152:
            return syn_type_fp8_152;

        case gcapi::TensorDataType_t::DATA_F8_143:
            return syn_type_fp8_143;

        case gcapi::TensorDataType_t::DATA_U32:
            return syn_type_uint32;

        case gcapi::TensorDataType_t::DATA_U64:
            return syn_type_uint64;

        case gcapi::TensorDataType_t::DATA_F32:
            return syn_type_single;

        case gcapi::TensorDataType_t::DATA_BF16:
            return syn_type_bf16;

        case gcapi::TensorDataType_t::DATA_F16:
            return syn_type_fp16;

        default:
            return defaultType;
    }
}

synDataType translateTensorDataType(tpc_lib_api::TensorDataType type, const synDataType defaultType)
{
    switch (type)
    {
        case tpc_lib_api::TensorDataType::DATA_I4:
            return syn_type_int4;

        case tpc_lib_api::TensorDataType::DATA_I8:
            return syn_type_fixed;

        case tpc_lib_api::TensorDataType::DATA_U4:
            return syn_type_uint4;

        case tpc_lib_api::TensorDataType::DATA_U8:
            return syn_type_uint8;

        case tpc_lib_api::TensorDataType::DATA_I16:
            return syn_type_int16;

        case tpc_lib_api::TensorDataType::DATA_U16:
            return syn_type_uint16;

        case tpc_lib_api::TensorDataType::DATA_I32:
            return syn_type_int32;

        case tpc_lib_api::TensorDataType::DATA_I64:
            return syn_type_int64;

        case tpc_lib_api::TensorDataType::DATA_F8_152:
            return syn_type_fp8_152;

        case tpc_lib_api::TensorDataType::DATA_F8_143:
            return syn_type_fp8_143;

        case tpc_lib_api::TensorDataType::DATA_U32:
            return syn_type_uint32;

        case tpc_lib_api::TensorDataType::DATA_U64:
            return syn_type_uint64;

        case tpc_lib_api::TensorDataType::DATA_F32:
            return syn_type_single;

        case tpc_lib_api::TensorDataType::DATA_BF16:
            return syn_type_bf16;

        case tpc_lib_api::TensorDataType::DATA_F16:
            return syn_type_fp16;

        default:
            return defaultType;
    }
}

float* bf16BufferTofloatBuffer(bf16_t* bf16Buffer, unsigned numElements)
{
    auto floatBuffer = new float[numElements];
    // foreach element in bf16 buffer - convert to float representation
    for (unsigned i = 0; i < numElements; i++)
    {
        // Perform the conversion
        floatBuffer[i] = (float)(bf16Buffer[i]);
    }
    return floatBuffer;
}

float* float16BufferToFloatBuffer(fp16_t* fp16Buffer, unsigned numElements)
{
    auto floatBuffer = new float[numElements];
    // foreach element in float buffer - convert to fp16 representation
    for (unsigned i = 0; i < numElements; i++)
    {
        // Perform the conversion
        floatBuffer[i] = float(fp16Buffer[i]);
    }
    return floatBuffer;
}

synDataType getSynDataTypeFromDtypeSuffix(std::string_view guidSuffix)
{
    static constexpr std::string_view dtypeSuffix[] =
        {"bf16", "f16", "f32", "hf8", "f8", "i16", "i32", "i4", "i64", "i8", "u16", "u32", "u4", "u64", "u8"};
    static constexpr synDataType dtypeVal[] = {syn_type_bf16,
                                               syn_type_fp16,
                                               syn_type_single,
                                               syn_type_fp8_143,
                                               syn_type_fp8_152,
                                               syn_type_int16,
                                               syn_type_int32,
                                               syn_type_int4,
                                               syn_type_int64,
                                               syn_type_fixed,
                                               syn_type_uint16,
                                               syn_type_uint32,
                                               syn_type_uint4,
                                               syn_type_uint64,
                                               syn_type_uint8};
    static constexpr auto        dtypeSuffixBegin = std::begin(dtypeSuffix);
    static constexpr auto        dtypeSuffixEnd   = std::end(dtypeSuffix);

    auto dtypeSuffixIter = std::find(dtypeSuffixBegin, dtypeSuffixEnd, guidSuffix);
    return (dtypeSuffixIter == dtypeSuffixEnd) ? syn_type_na
                                               : dtypeVal[std::distance(dtypeSuffixBegin, dtypeSuffixIter)];
}

bool isGUIDDataTypeSupported(std::string_view guidSuffix)
{
    return getSynDataTypeFromDtypeSuffix(guidSuffix) != syn_type_na;
}
