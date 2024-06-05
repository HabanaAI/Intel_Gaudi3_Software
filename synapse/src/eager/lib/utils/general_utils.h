#pragma once

// eager includes (relative to src/eager/lib/)
#include "utils/general_defs.h"

// synapse api (relative to include/)
#include "internal/habana_device_types.h"
#include "synapse_common_types.h"

// Macro to extract type of pointer variable VAR at STRUCT
#define POINTER_TYPE(STRUCT, VAR) std::remove_reference<decltype(*STRUCT::VAR)>::type

namespace eager_mode
{
// Convert between two enums
constexpr HabanaDeviceType engineType2HabanaDeviceType(EngineType engineType)
{
    switch (engineType)
    {
        case EngineType::TPC:
            return HabanaDeviceType::DEVICE_TPC;
        case EngineType::MME:
            return HabanaDeviceType::DEVICE_MME;
        case EngineType::DMA:
            return HabanaDeviceType::DEVICE_EDMA;
        default:
            EAGER_ASSERT_0;
            break;
    }
    return LAST_HABANA_DEVICE;
}

// Convert between two enums
constexpr EngineType habanaDeviceType2EngineType(HabanaDeviceType devType)
{
    switch (devType)
    {
        case HabanaDeviceType::DEVICE_TPC:
            return EngineType::TPC;
        case HabanaDeviceType::DEVICE_MME:
            return EngineType::MME;
        case HabanaDeviceType::DEVICE_EDMA:
            return EngineType::DMA;
        default:
            EAGER_ASSERT_0;
            break;
    }
    return EngineType::INVALID;
}

// Convert between two enums
constexpr synDeviceType chipType2SynDeviceType(ChipType chipType)
{
    switch (chipType)
    {
        case ChipType::GAUDI2:
            return synDeviceGaudi2;
        case ChipType::GAUDI3:
            return synDeviceGaudi3;
        default:
            EAGER_ASSERT_0;
            break;
    }
    return synDeviceTypeInvalid;
}

// Convert between two enums
constexpr ChipType synDeviceType2ChipType(synDeviceType devType)
{
    switch (devType)
    {
        case synDeviceGaudi2:
            return ChipType::GAUDI2;
        case synDeviceGaudi3:
            return ChipType::GAUDI3;
        default:
            EAGER_ASSERT_0;
            break;
    }
    return ChipType::INVALID;
}

constexpr std::string_view engineToStr(EngineType engine)
{
#define ADD_ENGINE_STR(ENGINE)                                                                                         \
    case EngineType::ENGINE:                                                                                           \
        do                                                                                                             \
        {                                                                                                              \
            return #ENGINE;                                                                                            \
        } while (false)

    switch (engine)
    {
        ADD_ENGINE_STR(TPC);
        ADD_ENGINE_STR(MME);
        ADD_ENGINE_STR(DMA);
        ADD_ENGINE_STR(ROT);
        ADD_ENGINE_STR(CME);
        default:
            EAGER_ASSERT_0;
            return {};
    }

#undef ADD_ENGINE_STR
}

}  // namespace eager_mode