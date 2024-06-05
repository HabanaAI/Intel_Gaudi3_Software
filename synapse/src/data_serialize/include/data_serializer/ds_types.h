#pragma once

#include "synapse_common_types.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace data_serialize
{
static const std::string TENSORS_TABLE = "TENSORS";

struct GraphInfo
{
    std::string id;
    uint16_t    recipeId;
    uint64_t    group;
};

enum class Compression
{
    NO_COMP,
    LZ4,
    LZ4_HC
};

enum class TensorValidation
{
    VALID,
    INVALID_ID,
    DUPLICATE,
    INVALID_DATA
};

struct TensorMetadata
{
    uint64_t                  id;
    std::string               name;
    synTensorType             type;
    synDataType               dataType;
    std::vector<TSize>        shape;
    std::vector<uint8_t>      permutation;
    uint64_t                  dataSize;
    std::shared_ptr<uint64_t> data;
    Compression               compression;
    std::vector<uint8_t>      compressedData;
    TensorValidation          validation;
    bool                      constTensor;
    uint64_t                  address;
    uint64_t                  launchIndex;
    std::optional<size_t>     index;
    bool                      input;
};
}  // namespace data_serialize