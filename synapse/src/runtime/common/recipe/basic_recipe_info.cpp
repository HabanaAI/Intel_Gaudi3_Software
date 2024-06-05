#include "basic_recipe_info.hpp"

#include "runtime/common/recipe/recipe_serializer.hpp"
#include "defenders.h"
#include "utils.h"

synStatus _tryGetTensorFromRecipe(uint64_t tensorId, const shape_plane_graph_t* pShapePlaneGraph, tensor_info_t& tensor)
{
    for (size_t i = 0; i < pShapePlaneGraph->sp_tensors_nr; i++)
    {
        auto     curr      = pShapePlaneGraph->sp_tensors[i];
        uint64_t recipeIdx = GET_TENSOR_INFO(curr.tensor_db_index, curr.user_tensor_type);
        if (curr.tensor_type == tensor_info_t::INTERNAL_TENSOR) continue;
        if (tensorId == recipeIdx)
        {
            tensor = curr;
            return synSuccess;
        }
    }

    return synInvalidArgument;
}

// for static recipe only
synStatus _tensorRetrieveLaunchInfo(const recipe_t*                  pRecipe,
                                    const persist_tensor_info_t&     tensorInfo,
                                    synRetrievedLaunchTensorInfoExt* tensorsLaunchInfo)
{
    VERIFY_IS_NULL_POINTER(SYN_RECIPE, pRecipe, "pRecipe")
    tensorsLaunchInfo->tensorType            = synTensorType(tensorInfo.tensorType);
    tensorsLaunchInfo->tensorDataType        = synDataType(tensorInfo.elementType);
    tensorsLaunchInfo->tensorDims            = tensorInfo.dimensions;
    tensorsLaunchInfo->isInput               = tensorInfo.isInput ? 1 : 0;
    tensorsLaunchInfo->tensorSectionId       = tensorInfo.section_idx;
    tensorsLaunchInfo->tensorOffsetInSection = tensorInfo.offset_in_section;

    VERIFY_IS_NULL_POINTER(SYN_RECIPE, tensorInfo.name, "tensorInfo.name")
    std::strcpy(tensorsLaunchInfo->tensorName, tensorInfo.name);
    memcpy(tensorsLaunchInfo->tensorMaxSize, tensorInfo.dimensionsSize, sizeof(tensorsLaunchInfo->tensorMaxSize));
    memcpy(tensorsLaunchInfo->tensorMinSize, tensorInfo.dimensionsSize, sizeof(tensorsLaunchInfo->tensorMinSize));
    memcpy(tensorsLaunchInfo->tensorPermutation, tensorInfo.permutation, sizeof(tensorsLaunchInfo->tensorPermutation));
    return synSuccess;
}

// for dynamic recipe only
synStatus _tensorRetrieveLaunchInfo(const recipe_t*                  pRecipe,
                                    const tensor_info_t&             tensorInfo,
                                    const shape_plane_graph_t*       pShapePlaneGraph,
                                    synRetrievedLaunchTensorInfoExt* tensorsLaunchInfo)
{
    VERIFY_IS_NULL_POINTER(SYN_RECIPE, pRecipe, "pRecipe")
    VERIFY_IS_NULL_POINTER(SYN_RECIPE, pShapePlaneGraph, "pShapePlaneGraph")
    VERIFY_IS_NULL_POINTER(SYN_RECIPE, tensorsLaunchInfo, "tensorsLaunchInfo")
    switch (tensorInfo.tensor_type)
    {
        case tensor_info_t::PERSISTENT_TENSOR:
        {
            persist_tensor_info_t persistTensorInfo  = pRecipe->tensors[tensorInfo.tensor_db_index];
            tensorsLaunchInfo->tensorType            = tensorInfo.user_tensor_type;
            tensorsLaunchInfo->tensorDataType        = synDataType(persistTensorInfo.elementType);
            tensorsLaunchInfo->tensorDims            = persistTensorInfo.dimensions;
            tensorsLaunchInfo->isInput               = persistTensorInfo.isInput ? 1 : 0;
            tensorsLaunchInfo->tensorSectionId       = persistTensorInfo.section_idx;
            tensorsLaunchInfo->tensorOffsetInSection = persistTensorInfo.offset_in_section;

            VERIFY_IS_NULL_POINTER(SYN_RECIPE, persistTensorInfo.name, "persistTensorInfo.name")
            std::strcpy(tensorsLaunchInfo->tensorName, persistTensorInfo.name);
            memcpy(tensorsLaunchInfo->tensorMaxSize, persistTensorInfo.dimensionsSize, sizeof(persistTensorInfo.dimensionsSize));

            if (persistTensorInfo.dimensions <= sizeof(tensorInfo.min_dims)/sizeof(tensorInfo.min_dims[0]))
            {
                // This is a low-dimensioned tensor, all sizes are present in tensorInfo.min_dims
                // Copy tensorsLaunchInfo->tensorMinSize from tensorInfo.min_dims
                // TODO if/when tensor_info_t gets upgraded to ndims, this should be unified with the else branch
                memcpy(tensorsLaunchInfo->tensorMinSize, tensorInfo.min_dims, sizeof(tensorInfo.min_dims));
            }
            else
            {
                // This is a high-dimensioned tensor. tensor_info_t does not have all ot its shape
                // but it is static, so minSize == maxSize, and we can copy
                // tensorsLaunchInfo->tensorMinSize from persistTensorInfo.dimensionsSize
                memcpy(tensorsLaunchInfo->tensorMinSize, persistTensorInfo.dimensionsSize, sizeof(persistTensorInfo.dimensionsSize));
            }
            memcpy(tensorsLaunchInfo->tensorPermutation, persistTensorInfo.permutation, sizeof(persistTensorInfo.permutation));
            break;
        }
        case tensor_info_t::SHAPE_TENSOR:
        {
            tensorsLaunchInfo->tensorType            = tensorInfo.user_tensor_type;
            tensorsLaunchInfo->tensorDataType        = synDataType(tensorInfo.data_type);
            tensorsLaunchInfo->tensorDims            = tensorInfo.infer_info.geometry.dims;
            tensorsLaunchInfo->isInput               = 1;
            tensorsLaunchInfo->tensorSectionId       = 0;
            tensorsLaunchInfo->tensorOffsetInSection = 0;

            shape_tensor_info_t& shapeTensor = pShapePlaneGraph->shape_tensors[tensorInfo.tensor_db_index];

            VERIFY_IS_NULL_POINTER(SYN_RECIPE, shapeTensor.name, "shapeTensor.name")
            std::strcpy(tensorsLaunchInfo->tensorName, shapeTensor.name);
            memcpy(tensorsLaunchInfo->tensorMaxSize, tensorInfo.max_dims, sizeof(tensorInfo.max_dims));
            memcpy(tensorsLaunchInfo->tensorMinSize, tensorInfo.min_dims, sizeof(tensorInfo.min_dims));
            memcpy(tensorsLaunchInfo->tensorPermutation, tensorInfo.permutation, sizeof(tensorInfo.permutation));
            break;
        }
        case tensor_info_t::INTERNAL_TENSOR:
        {
            LOG_ERR(SYN_RECIPE, "{}: invalid tensor type: {}", HLLOG_FUNC, tensorInfo.tensor_type);
            return synFail;
        }
    }
    return synSuccess;
}

synStatus _tensorGetIndexFromRecipe(const char* tensorName, const recipe_t* pRecipe, unsigned& tensorIndex)
{
    if (tensorName == nullptr)
    {
        LOG_ERR(SYN_RECIPE, "{}: tensor-name is null", HLLOG_FUNC);
        return synFail;
    }

    for (size_t i = 0; i < pRecipe->persist_tensors_nr; i++)
    {
        const char* currentTensorName = pRecipe->tensors[i].name;

        // match the tensor name the user provided  with the tensor name in the recipe
        if (strcmp(currentTensorName, tensorName) == 0)
        {
            tensorIndex = i;
            return synSuccess;
        }
    }
    LOG_ERR(SYN_RECIPE, "{}: Can not find tensor-name {} in TensorMD DB", HLLOG_FUNC, tensorName);

    for (size_t i = 0; i < pRecipe->persist_tensors_nr; i++)
    {
        const char* currentTensorName = pRecipe->tensors[i].name;
        LOG_ERR(SYN_RECIPE, "{}: currentTensorName {} in TensorMD DB", HLLOG_FUNC, currentTensorName);
    }

    return synFail;
}

void _tensorRetrieveMetadataInfo(const persist_tensor_info_t& tensorMetadata,
                                 TensorMetadataInfoExt*       tensorsMetadataInfo)
{
    tensorsMetadataInfo->zp              = tensorMetadata.zp;
    tensorsMetadataInfo->scale           = tensorMetadata.scale;
    tensorsMetadataInfo->elementType     = tensorMetadata.elementType;
    tensorsMetadataInfo->batchSize       = (uint16_t)tensorMetadata.batchSize;
    tensorsMetadataInfo->dimensions      = tensorMetadata.dimensions;
    tensorsMetadataInfo->isInput         = tensorMetadata.isInput ? 1 : 0;
    tensorsMetadataInfo->sectionId       = tensorMetadata.section_idx;
    tensorsMetadataInfo->offsetInSection = tensorMetadata.offset_in_section;

    memcpy(tensorsMetadataInfo->dimensionsSize,
           tensorMetadata.dimensionsSize,
           sizeof(*(tensorMetadata.dimensionsSize)) * tensorsMetadataInfo->dimensions);

    memset(&tensorsMetadataInfo->layout[0], '\0', MAX_LAYOUT_SIZE);

    std::string layoutStr("NotAvailable");
    if (tensorMetadata.layout != nullptr)
    {
        layoutStr = std::string(tensorMetadata.layout);
    }
    memcpy(tensorsMetadataInfo->layout, layoutStr.c_str(), sizeof(char) * layoutStr.size());
}

synStatus basicRecipeInfo::tensorRetrieveMetadatasInfosByName(const uint32_t         numOfTensors,
                                                              TensorMetadataInfoExt* tensorsMetadataInfo) const
{
    LOG_TRACE(SYN_RECIPE, "{}", HLLOG_FUNC);

    VERIFY_IS_NULL_POINTER(SYN_RECIPE, tensorsMetadataInfo, "tensorsMetadataInfo");

    synStatus status = synSuccess;
    for (unsigned tensorIndex = 0; tensorIndex < numOfTensors; tensorIndex++)
    {
        TensorMetadataInfoExt* pSingleTensorMDInfo = &tensorsMetadataInfo[tensorIndex];
        unsigned               metadataIndex       = 0;
        const recipe_t*        pRecipe             = recipe;

        status = _tensorGetIndexFromRecipe(pSingleTensorMDInfo->tensorName, pRecipe, metadataIndex);
        if (status != synSuccess)
        {
            LOG_ERR(SYN_RECIPE,
                    "{}: Can not find tensor-name {} in TensorMD DB",
                    HLLOG_FUNC,
                    pSingleTensorMDInfo->tensorName);
            return status;
        }
        _tensorRetrieveMetadataInfo(pRecipe->tensors[metadataIndex], pSingleTensorMDInfo);
    }

    return synSuccess;
}

synStatus basicRecipeInfo::tensorRetrievePersistentAmount(uint32_t& numOfTensors) const
{
    LOG_TRACE(SYN_RECIPE, "{}", HLLOG_FUNC);

    const recipe_t* pRecipe = recipe;
    if (pRecipe == nullptr)
    {
        LOG_ERR(SYN_RECIPE, "{}: input recipe argument is invalid ", HLLOG_FUNC);
        return synInvalidArgument;
    }
    numOfTensors = (uint32_t)pRecipe->persist_tensors_nr;
    return synSuccess;
}

synStatus basicRecipeInfo::tensorRetrieveNames(char           tensorsName[][ENQUEUE_TENSOR_NAME_MAX_SIZE],
                                               const uint32_t numOfTensors) const
{
    LOG_TRACE(SYN_RECIPE, "{}", HLLOG_FUNC);

    synStatus       status  = synSuccess;
    const recipe_t* pRecipe = recipe;
    if (pRecipe == nullptr)
    {
        LOG_ERR(SYN_RECIPE, "{}: input recipe argument is invalid ", HLLOG_FUNC);
        return synInvalidArgument;
    }
    if (numOfTensors > pRecipe->persist_tensors_nr)
    {
        LOG_ERR(SYN_RECIPE,
                "{}: numOfTensors {} exceeds recipe persist tensors nr {} ",
                HLLOG_FUNC,
                numOfTensors,
                pRecipe->persist_tensors_nr);
        return synFail;
    }

    for (size_t i = 0; i < numOfTensors; i++)
    {
        // Todo [SW-52885] Synapse should not crop the 1024 character in tensor name
        std::strncpy((char*)tensorsName[i], pRecipe->tensors[i].name, ENQUEUE_TENSOR_NAME_MAX_SIZE);
        tensorsName[i][ENQUEUE_TENSOR_NAME_MAX_SIZE - 1] = '\0';
        LOG_DEBUG(SYN_RECIPE, "{}: Tensor {} name {}", HLLOG_FUNC, i, tensorsName[i]);
    }
    return status;
}

synStatus basicRecipeInfo::tensorRetrieveLaunchInfoById(const uint32_t                   numOfTensors,
                                                        synRetrievedLaunchTensorInfoExt* tensorsLaunchInfo) const
{
    LOG_TRACE(SYN_RECIPE, "{}", HLLOG_FUNC);
    if (numOfTensors > 0)
    {
        VERIFY_IS_NULL_POINTER(SYN_RECIPE, tensorsLaunchInfo, "tensorsLaunchInfo");
    }

    for (unsigned i = 0; i < numOfTensors; i++)
    {
        synRetrievedLaunchTensorInfoExt* pInfo = &tensorsLaunchInfo[i];
        if (pInfo->tensorId == TENSOR_INVALID_ID)
        {
            LOG_DEBUG(SYN_RECIPE, "{}: skipping tensor index {} with invalid Tensor-ID", HLLOG_FUNC, i);
            pInfo->tensorType = TENSOR_TYPE_INVALID;
            continue;
        }

        if (shape_plan_recipe == nullptr)
        {
            if (pInfo->tensorId >= recipe->persist_tensors_nr)
            {
                LOG_DEBUG(SYN_RECIPE,
                          "{}: Tensor index {} Tensor ID {} exceeds recipe persistent tensors count: {}",
                          HLLOG_FUNC,
                          i,
                          pInfo->tensorId,
                          recipe->persist_tensors_nr);
                pInfo->tensorType = TENSOR_TYPE_INVALID;
                continue;
            }
            const persist_tensor_info_t& rTensorInfo = recipe->tensors[pInfo->tensorId];
            synStatus                    status      = _tensorRetrieveLaunchInfo(recipe, rTensorInfo, pInfo);
            if (status != synSuccess) return status;
        }
        else
        {
            tensor_info_t tensorInfo {};
            synStatus     status = _tryGetTensorFromRecipe(pInfo->tensorId, shape_plan_recipe, tensorInfo);
            if (status != synSuccess)
            {
                pInfo->tensorType = TENSOR_TYPE_INVALID;
                continue;
            }
            status = _tensorRetrieveLaunchInfo(recipe, tensorInfo, shape_plan_recipe, pInfo);
            if (status != synSuccess) return status;
        }
    }

    return synSuccess;
}

synStatus basicRecipeInfo::getHostMemorySize(uint64_t& retVal) const
{
    const recipe_t*            pRecipe = recipe;
    const shape_plane_graph_t* pSpr    = shape_plan_recipe;
    ParamsSizeManager          params;
    synStatus                  status = RecipeSerializer::serialize(pRecipe, pSpr, &params);
    if (status == synSuccess)
    {
        uint32_t currentSize = params.getCurrentSize();
        LOG_TRACE(SYN_RECIPE, "{}: Received size {} from recipe", HLLOG_FUNC, currentSize);
        retVal = currentSize;
    }
    else
    {
        LOG_ERR(SYN_RECIPE, "Failed to receive size from recipe");
        return status;
    }
    return status;
}

synStatus basicRecipeInfo::getPersistentTensorsMemorySize(uint64_t& retVal) const
{
    const recipe_t* pRecipe    = recipe;
    uint64_t        memorySize = 0;

    // first, add total struct sizes - constant
    memorySize += (pRecipe->persist_tensors_nr) * sizeof(persist_tensor_info_t);

    // then, calculate variable length members
    for (uint32_t tensor_index = 0; tensor_index < pRecipe->persist_tensors_nr; tensor_index++)
    {
        persist_tensor_info_t* pTensor = &pRecipe->tensors[tensor_index];

        memorySize += (strlen(pTensor->name) + 1);

        if (pTensor->layout != nullptr)
        {
            memorySize += (strlen(pTensor->layout) + 1);
        }

        if (pTensor->multi_views_indices != nullptr)
        {
            memorySize += pTensor->multi_views_indices_nr * sizeof(uint32_t);
        }
    }

    retVal = memorySize;
    return synSuccess;
}

synStatus basicRecipeInfo::getConstSectionMemorySize(uint64_t& retVal) const
{
    const recipe_t* pRecipe    = recipe;
    uint64_t        memorySize = 0;

    // first, add total struct sizes - constant
    memorySize += (pRecipe->const_sections_nr) * sizeof(const_section_t);

    // then, calculate variable length members
    for (uint32_t section_index = 0; section_index < pRecipe->const_sections_nr; section_index++)
    {
        memorySize += pRecipe->const_sections[section_index].size;
    }

    retVal = memorySize;
    return synSuccess;
}

synStatus basicRecipeInfo::getRecipeHbmMemorySize(uint64_t& retVal) const
{
    const recipe_t* pRecipe    = recipe;
    uint64_t        memorySize = 0;

    memorySize += pRecipe->program_data_blobs_size;
    memorySize += pRecipe->dynamic_blobs_buffer_size;

    // first, add total struct sizes - constant
    memorySize += (pRecipe->programs_nr) * sizeof(program_t);

    // then, calculate variable length members
    for (uint32_t program_index = 0; program_index < pRecipe->programs_nr; program_index++)
    {
        memorySize += pRecipe->programs[program_index].program_length;
    }

    retVal = memorySize;
    return synSuccess;
}
