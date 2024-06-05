#pragma once

// TODO: [SW-166081] Remove when fuser is moved to protocolIR
#include "gc_interface.h"
#include <tpc_kernel_lib_interface.h>
#include "synapse_common_types.h"
#include "types.h"
#include "utils.h"
#include "mme_reference/data_types/non_standard_dtypes.h"

class HabanaGraph;
class HalReader;

// data type related utils
std::string_view getStringFromSynDataType(synDataType type);
synDataType      getSynDataTypeFromString(const std::string& type);
std::string_view getDtypeSuffixFromSynDataType(synDataType type);
bool             isSameBitRepresentation(synDataType typeA, synDataType typeB);

// guid related utils
std::string_view extractDtypeFromGUID(std::string_view guid);
std::string_view extractDtypeFromCastGUID(std::string_view guid);
std::string_view extractGUIDFromFullGUID(std::string_view guid);

std::string getCastGUID(synDataType from, synDataType to);
static inline bool isCastGUID(std::string_view guid)
{
    // cast guid starts with "cast_" and contains "_to_"
    return startsWith(guid, "cast_") && guid.find("_to_") != std::string::npos;
}
static inline bool isConstantGUID(std::string_view guid)
{
    return startsWith(guid, "constant_");
}

synDataType getQuantDataType(HabanaGraph& g, const TensorPtr& tensor, const NodePtr& node);
synDataType getRequiredInputDataTypeByIndex(std::string_view guid, const unsigned index);
synDataType getRequiredOutputDataTypeByIndex(const std::string& guid, const unsigned index);
// TODO: [SW-166081] Remove when fuser is moved to protocolIR
synDataType translateTensorDataType(gcapi::TensorDataType_t type, const synDataType defaultType = syn_type_na);
synDataType translateTensorDataType(tpc_lib_api::TensorDataType type, const synDataType defaultType = syn_type_na);

bool isQuantDtype(std::string_view dtype);
bool isQuantDtype(synDataType dtype);

float* float16BufferToFloatBuffer(fp16_t* fp16Buffer, unsigned numElements);

float* bf16BufferTofloatBuffer(bf16_t* bf16Buffer, unsigned numElements);

template<typename T, typename U>
T* convertBuffer(U* origBuffer, unsigned numElements)
{
    T* convertedBuffer = new T[numElements];
    // foreach element in the original buffer - convert to desired representation
    for (unsigned i = 0; i < numElements; i++)
    {
        // Perform the conversion
        convertedBuffer[i] = T(origBuffer[i]);
    }
    return convertedBuffer;
}

synDataType getSynDataTypeFromDtypeSuffix(std::string_view guidSuffix);

bool isGUIDDataTypeSupported(std::string_view guidSuffix);