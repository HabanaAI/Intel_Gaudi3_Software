#pragma once

#include <cstdlib>
#include <synapse_types.h>
#include <list>
#include <string>
#include <string_view>
#include "infra/defs.h"

enum Direction
{
    FWD,
    BWD
};

enum BnOps
{
    BN_OPS_BN = 0,
    BN_OPS_BN_ACTIVATION,
    BN_OPS_BN_ADD_ACTIVATION
};


constexpr std::string_view dir2Str(Direction direction)
{
    std::string_view dirStr;

    switch (direction)
    {
        case FWD:
            dirStr = "_fwd";
            break;
        case BWD:
            dirStr= "_bwd";
            break;
        default:
            HB_ASSERT(false, "direction should be forward or backward");
    }
    return dirStr;
}

constexpr std::string_view type2Str(synDataType type)
{
    std::string_view typeStr;

    switch (type)
    {
        case syn_type_bf16:
            typeStr = "_bf16";
            break;
        case syn_type_fp8_152:
            typeStr = "_f8";
            break;
        case syn_type_float:
            typeStr = "_f32";
            break;
        case syn_type_na:
        default:
            HB_ASSERT(false, "type should be f32 or bf16");
    }
    return typeStr;
}

constexpr std::string_view bnOp2Str(BnOps op)
{
    std::string_view opStr;

    switch (op)
    {
        case BN_OPS_BN:
            opStr = "";
            break;
        case BN_OPS_BN_ACTIVATION:
            opStr = "_relu";
            break;
        case BN_OPS_BN_ADD_ACTIVATION:
            opStr = "_add_relu";
            break;
        default:
            HB_ASSERT(false, "op should be empty, relu or add_relu");
    }
    return opStr;
}

inline std::string getBN1Guid(Direction direction, synDataType data_type)
{
    return fmt::format("batch_norm_stage1{}{}", dir2Str(direction), type2Str(data_type));
}

inline std::string getBN2Guid(BnOps operations, Direction direction, synDataType data_type)
{

    return fmt::format("batch_norm_stage2{}{}{}", bnOp2Str(operations), dir2Str(direction), type2Str(data_type));
}

inline std::string getMomentsGuid(int stage, Direction direction, synDataType data_type)
{

    return fmt::format("bn_get_moments_stage{}{}{}", std::to_string(stage), dir2Str(direction), type2Str(data_type));
}

std::vector<std::string> getBN1Guids(Direction direction);

std::vector<std::string> getBN2Guids(Direction direction);
