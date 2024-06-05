#pragma once

#include "recipe_tensor_info.hpp"
#include "recipe_staged_submission_info.hpp"
#include "device_agnostic_recipe_static_info.hpp"
#include "synapse_common_types.h"

#include "runtime/scal/common/recipe_static_info_scal.hpp"

#include <unordered_map>
#include <deque>

struct RecipeDsdStaticInfo
{
    std::vector<uint64_t> m_recipeInputs;
    std::vector<uint64_t> m_recipeOutputs;
    std::vector<bool>     m_isStaticTensors;
    std::vector<bool>     m_inputAndStaticTensors;
    uint16_t              m_maxNodeOutputs    = 0;
    uint16_t              m_fuserMaxIn        = 0;
    uint16_t              m_fuserMaxOut       = 0;
    uint16_t              m_fuserMaxDbTensors = 0;
};

struct SignalFromGraphInfo
{
    uint32_t getExtTensorExeOrderByExtTensorIdx(uint32_t tensorIdx) const;

    size_t getNumberOfExternalTensors() const { return m_sfgExtTensorIdxToExeOrder.size(); }

    const std::deque<uint64_t>& getTensorExecutionOrder() const { return m_sfgExtTensorIdxSortedByExeOrder; }

    std::deque<uint64_t>                   m_sfgExtTensorIdxSortedByExeOrder;
    std::unordered_map<uint32_t, uint32_t> m_sfgExtTensorIdxToExeOrder;

    static const uint32_t TENSOR_EXE_ORDER_INVALID = UINT32_MAX;

    // gaudi only
    uint32_t m_lkdSobLowPartAddress = TENSOR_EXE_ORDER_INVALID;
};

struct DeviceAgnosticRecipeInfo
{
    synDeviceType        m_deviceType    = synDeviceTypeInvalid;
    uint64_t             m_workspaceSize = 0;
    RecipeTensorsInfo    m_recipeTensorInfo;
    RecipeDsdStaticInfo  m_recipeDsdStaticInfo;
    RecipeStaticInfoScal m_recipeStaticInfoScal;
    SignalFromGraphInfo  m_signalFromGraphInfo;
    RecipeStageSubmisssionInfo m_recipeStageInfo;
    DeviceAgnosticRecipeStaticInfo m_recipeStaticInfo;
    bool                 m_isInitialized;
};
