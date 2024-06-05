#include "defs.h"
#include "common_type_utils.h"
#include "utils.h"

std::string tensorTypeToString(synTensorType tensorType)
{
    switch (tensorType)
    {
        TRANSLATE_ENUM_TO_STRING(DATA_TENSOR)
        TRANSLATE_ENUM_TO_STRING(OUTPUT_DESCRIBING_SHAPE_TENSOR)
        TRANSLATE_ENUM_TO_STRING(INPUT_DESCRIBING_SHAPE_TENSOR)
        TRANSLATE_ENUM_TO_STRING(DATA_TENSOR_DYNAMIC)
        TRANSLATE_ENUM_TO_STRING(DEVICE_SHAPE_TENSOR)
        TRANSLATE_ENUM_TO_STRING(HOST_SHAPE_TENSOR)
        TRANSLATE_ENUM_TO_STRING(HOST_TO_DEVICE_TENSOR)
        default:
            HB_ASSERT(false, "Failed to get tensor type string, tensor type: {}", tensorType);
            return "";
    }
}

synTensorType tensorTypeFromString(const std::string& str)
{
    if (str == "DATA_TENSOR") return DATA_TENSOR;
    if (str == "OUTPUT_DESCRIBING_SHAPE_TENSOR") return OUTPUT_DESCRIBING_SHAPE_TENSOR;
    if (str == "INPUT_DESCRIBING_SHAPE_TENSOR") return INPUT_DESCRIBING_SHAPE_TENSOR;
    if (str == "DATA_TENSOR_DYNAMIC") return DATA_TENSOR_DYNAMIC;
    if (str == "DEVICE_SHAPE_TENSOR") return DEVICE_SHAPE_TENSOR;
    if (str == "HOST_SHAPE_TENSOR") return HOST_SHAPE_TENSOR;
    if (str == "HOST_TO_DEVICE_TENSOR") return HOST_TO_DEVICE_TENSOR;
    HB_ASSERT(false, "Invalid tensor type string: {}", str);
    return TENSOR_TYPE_MAX;
}
