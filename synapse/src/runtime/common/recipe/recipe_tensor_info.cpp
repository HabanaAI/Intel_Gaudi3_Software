#include "runtime/common/recipe/recipe_tensor_info.hpp"
#include "defs.h"
#include "defenders.h"
#include "recipe.h"
#include "synapse_common_types.h"

bool isInternalTensor(const tensor_info_t& tInfo)
{
    return tInfo.tensor_type == tensor_info_t::INTERNAL_TENSOR || tInfo.tensor_db_index == INVALID_TENSOR_INDEX;
}

synStatus RecipeTensorsInfo::tensorRetrieveIds(uint64_t* tensorIds, const uint32_t numOfTensors) const
{
    uint32_t tensorsNr = getTensorAmount();
    if (tensorsNr != numOfTensors)
    {
        LOG_ERR(SYN_RECIPE,
                "{}: requested number of tensors ({}) is different from recipe launch tensor ({})",
                HLLOG_FUNC,
                numOfTensors,
                tensorsNr);
        return synFail;
    }

    if (m_shapePlanRecipe != nullptr)
    {
        uint32_t arrayIdx = 0;
        for (uint64_t i = 0; i < m_shapePlanRecipe->sp_tensors_nr; i++)
        {
            tensor_info_t& curr = m_shapePlanRecipe->sp_tensors[i];
            if (isInternalTensor(curr)) continue;
            uint64_t tensorIdx    = curr.tensor_db_index;
            tensorIds[arrayIdx++] = GET_TENSOR_INFO(tensorIdx, curr.user_tensor_type);
        }
    }
    else
    {
        if (m_recipe == nullptr)
        {
            LOG_WARN(SYN_RECIPE, "{}: Can't get tensor ids, recipe not processed", HLLOG_FUNC);
            return synFail;
        }
        for (uint32_t index = 0; index < m_recipe->persist_tensors_nr; ++index)
        {
            tensorIds[index] = GET_TENSOR_INFO(index, DATA_TENSOR);
        }
    }
    return synSuccess;
}

synStatus
RecipeTensorsInfo::tensorRetrieveIds(const char** tensorNames, uint64_t* tensorIds, const uint32_t numOfTensors) const
{
    CHECK_POINTER(SYN_RECIPE, tensorNames, "tensorNames", synInvalidArgument);
    CHECK_POINTER(SYN_RECIPE, tensorIds, "tensorIds", synInvalidArgument);

    for (unsigned i = 0; i < numOfTensors; i++)
    {
        if (tensorNames[i] == nullptr)
        {
            LOG_ERR(SYN_RECIPE, "{}: input tensor {}, name is invalid ", HLLOG_FUNC, i);
            return synInvalidArgument;
        }

        tensorRetrieveId(tensorNames[i], tensorIds + i);
    }

    return synSuccess;
}

synStatus RecipeTensorsInfo::tensorRetrieveId(const char* tensorName, uint64_t* tensorId) const
{
    if (m_recipe == nullptr)
    {
        LOG_WARN(SYN_RECIPE, "{}: Can't get tensor {} id, recipe not processed", HLLOG_FUNC, tensorName);
        return synFail;
    }

    if (m_shapePlanRecipe != nullptr)
    {
        for (uint64_t i = 0; i < m_shapePlanRecipe->sp_tensors_nr; i++)
        {
            tensor_info_t& curr      = m_shapePlanRecipe->sp_tensors[i];
            if (isInternalTensor(curr)) continue;
            uint64_t       tensorIdx = curr.tensor_db_index;
            const char* name = curr.tensor_type == tensor_info_t::PERSISTENT_TENSOR ?
                                    m_recipe->tensors[tensorIdx].name : m_shapePlanRecipe->shape_tensors[tensorIdx].name;
            if (strcmp(name, tensorName) == 0)
            {
                *tensorId = GET_TENSOR_INFO(tensorIdx, curr.user_tensor_type);
                LOG_DEBUG(SYN_RECIPE,
                          "{}: DSD recipe: tensorName {} tensorId {} tensorType {} user_tensor_type {} dataType {} recipe {}",
                          HLLOG_FUNC,
                          tensorName,
                          *tensorId,
                          curr.tensor_type,
                          curr.user_tensor_type,
                          curr.data_type,
                          TO64(m_recipe));
                return synSuccess;
            }
        }
    }
    else
    {
        const uint64_t numPersistTensors = m_recipe->persist_tensors_nr;
        for (uint64_t i = 0; i < numPersistTensors; i++)
        {
            auto& curr = m_recipe->tensors[i];
            if (strcmp(curr.name, tensorName) == 0)
            {
                *tensorId = GET_TENSOR_INFO(i, DATA_TENSOR);
                LOG_DEBUG(SYN_RECIPE,
                          "{}: Static recipe: tensorName {} tensorId {}, dataType {}, sectionIdx {}, recipe {}",
                          HLLOG_FUNC,
                          tensorName,
                          *tensorId,
                          curr.elementType,
                          curr.section_idx,
                          TO64(m_recipe));
                return synSuccess;
            }
        }
    }
    *tensorId = TENSOR_INVALID_ID;
    LOG_DEBUG(SYN_RECIPE, "{}: Given tensor not in persist/shape list: {}", HLLOG_FUNC, tensorName);

    return synSuccess;
}

uint32_t getDsdTensorsAnount(const shape_plane_graph_t& spr)
{
    uint32_t ret = 0;
    for (uint64_t i = 0; i < spr.sp_tensors_nr; i++)
    {
        if (isInternalTensor(spr.sp_tensors[i])) continue;
        ++ret;
    }
    return ret;
}

uint32_t RecipeTensorsInfo::getTensorAmount() const
{
    return m_shapePlanRecipe != nullptr ? getDsdTensorsAnount(*m_shapePlanRecipe) : m_recipe->persist_tensors_nr;
}
