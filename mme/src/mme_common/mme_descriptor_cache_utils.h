#pragma once

#include "include/mme_common/mme_common_enum.h"
#include "include/mme_common/descriptor_cache.h"
#include <string_view>
#include <functional>

template<>
struct std::hash<MmeCommon::MmeLayerParams>
{
    std::size_t operator()(const MmeCommon::MmeLayerParams& params) const noexcept
    {
        // hashing depends on fields ordering in MmeLayerParams so we enforce it here with static assert
        // This trait makes it possible to determine whether a type can be correctly hashed by hashing its object
        // representation as a byte array.
        static_assert(std::has_unique_object_representations_v<MmeCommon::MmeTensorView>);
        static_assert(offsetof(MmeCommon::MmeLayerParams, x) ==
                      offsetof(MmeCommon::MmeLayerParams, opType) + sizeof(MmeCommon::MmeLayerParams::opType));
        static_assert(offsetof(MmeCommon::MmeLayerParams, y) ==
                      offsetof(MmeCommon::MmeLayerParams, x) + sizeof(MmeCommon::MmeLayerParams::x));
        static_assert(offsetof(MmeCommon::MmeLayerParams, w) ==
                      offsetof(MmeCommon::MmeLayerParams, y) + sizeof(MmeCommon::MmeLayerParams::y));

        // It is sufficient to only take the op type and operand tensor views under account
        // to reduce the amount of colisions as it is unlikely to have keys with same values for those
        // but different values for other MmeLayerParams fields (apart from the node name).
        const auto* const START_PTR = reinterpret_cast<const char*>(&params.opType);
        std::string_view view(START_PTR,
                              offsetof(MmeCommon::MmeLayerParams, w) + sizeof(MmeCommon::MmeLayerParams::w) -
                                  offsetof(MmeCommon::MmeLayerParams, opType));
        return std::hash<std::string_view> {}(view);
    }
};

template<>
struct std::hash<MmeCommon::KeyAndHash<MmeCommon::MmeLayerParams>>
{
    std::size_t operator()(const MmeCommon::KeyAndHash<MmeCommon::MmeLayerParams>& paramsAndHash) const noexcept
    {
        return paramsAndHash.getHash();
    }
};
