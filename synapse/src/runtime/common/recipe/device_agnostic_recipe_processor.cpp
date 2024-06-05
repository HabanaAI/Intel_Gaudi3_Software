#include "device_agnostic_recipe_processor.hpp"
#include "defenders.h"
#include "global_statistics.hpp"
#include "habana_global_conf_runtime.h"
#include "recipe.h"
#include "recipe_handle_impl.hpp"
#include "recipe_verification.hpp"
#include "synapse_runtime_logging.h"
#include "device_agnostic_recipe_static_processor.hpp"
#include "runtime/common/recipe/recipe_utils.hpp"
#include "runtime/common/recipe/recipe_tensor_processor.hpp"
#include "runtime/common/recipe/patching/host_address_patching_executer.hpp"
#include "runtime/common/recipe/recipe_staged_submission_processor.hpp"
#include "runtime/qman/common/qman_types.hpp"
#include "runtime/scal/common/recipe_static_processor_scal.hpp"
#include "types.h"

synStatus DeviceAgnosticRecipeProcessor::process(const basicRecipeInfo&    rBasicRecipeInfo,
                                                 DeviceAgnosticRecipeInfo& rDeviceAgnosticRecipeInfo)
{
    VERIFY_IS_NULL_POINTER(SYN_RECIPE, rBasicRecipeInfo.recipe, "rBasicRecipeInfo.recipe");


    RECIPE_STATS_START(parse);
    parseRecipe(rBasicRecipeInfo, rDeviceAgnosticRecipeInfo.m_deviceType);
    LOG_RECIPE_STATS("parse         {:10d} ns", RECIPE_STATS_END(parse));

    if (!GCFG_RUNTIME_SKIP_RECIPE_VALIDATION.value())
    {
        RECIPE_STATS_START(verify);
        const synStatus status = verifyRecipe(rBasicRecipeInfo);
        if (status != synSuccess)
        {
            LOG_ERR(SYN_RECIPE, "{}: Can not verifyRecipe {}", HLLOG_FUNC, status);
            return status;
        }
        LOG_RECIPE_STATS("verify        {:10d} ns", RECIPE_STATS_END(verify));
    }

    RECIPE_STATS_START(tensorProc);
    // fills table
    synStatus status = RecipeTensorsProcessor::process(rBasicRecipeInfo,
                                                       rDeviceAgnosticRecipeInfo.m_recipeTensorInfo,
                                                       rDeviceAgnosticRecipeInfo.m_recipeDsdStaticInfo);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_RECIPE, "{}: RecipeTensorsProcessor failed {}", HLLOG_FUNC, status);
        return status;
    }
    LOG_RECIPE_STATS("TensorProc    {:10d} ns", RECIPE_STATS_END(tensorProc));

    if (!GCFG_RUNTIME_SKIP_RECIPE_VALIDATION.value())
    {
        // Note: HostAddressPatchingExecuter::verify has to be done after RecipeTensorsProcessor::process
        status =
            patching::HostAddressPatchingExecuter::verify(*rBasicRecipeInfo.recipe,
                                                          rDeviceAgnosticRecipeInfo.m_recipeTensorInfo.m_maxSectionId);
        if (status != synSuccess)
        {
            LOG_ERR(SYN_RECIPE, "{}: HostAddressPatchingExecuter failed {}", HLLOG_FUNC, status);
            return status;
        }
    }

    getTopologyWorkspaceSize(rBasicRecipeInfo,
                             rDeviceAgnosticRecipeInfo.m_deviceType,
                             rDeviceAgnosticRecipeInfo.m_workspaceSize);

    const bool isScalArch = isScalArchitecture(rDeviceAgnosticRecipeInfo.m_deviceType);

    if (!isScalArch)
    {
        RecipeStagedSubmissionProcessor::process(rBasicRecipeInfo, rDeviceAgnosticRecipeInfo.m_recipeStageInfo);

        status = DeviceAgnosticRecipeStaticProcessor::process(*rBasicRecipeInfo.recipe,
                                                              rDeviceAgnosticRecipeInfo.m_recipeStaticInfo,
                                                              rDeviceAgnosticRecipeInfo.m_deviceType);
        if (status != synSuccess)
        {
            LOG_ERR(SYN_RECIPE,
                    "{}: DeviceAgnosticRecipeStaticProcessor::process failed with status {}",
                    HLLOG_FUNC,
                    status);
            return status;
        }
    }
    else
    {
        RECIPE_STATS_START(scalProc);
        status = DeviceAgnosticRecipeStaticProcessorScal::process(rDeviceAgnosticRecipeInfo.m_deviceType,
                                                                  rBasicRecipeInfo,
                                                                  rDeviceAgnosticRecipeInfo.m_recipeStaticInfoScal);
        if (status != synSuccess)
        {
            LOG_ERR(SYN_RECIPE, "{}: DeviceAgnosticRecipeStaticProcessorScal failed {}", HLLOG_FUNC, status);
            return status;
        }
        LOG_RECIPE_STATS("scalProc      {:10d} ns", RECIPE_STATS_END(scalProc));

        if (!GCFG_RUNTIME_SKIP_RECIPE_VALIDATION.value())
        {
            RECIPE_STATS_START(verify);
            bool ret = RecipeVerification::verifyScalMemorySizes(rDeviceAgnosticRecipeInfo.m_recipeStaticInfoScal);
            if (!ret)
            {
                LOG_ERR(SYN_RECIPE, "{}: Can't verifyRecipe {}", HLLOG_FUNC, status);
                return synFail;
            }
            LOG_RECIPE_STATS("verify verifyScalMemorySizes {:10d} ns", RECIPE_STATS_END(verify));
        }
    }

    rDeviceAgnosticRecipeInfo.m_isInitialized = _extractSfgInfo(rDeviceAgnosticRecipeInfo.m_recipeTensorInfo,
                                                                rBasicRecipeInfo.recipe,
                                                                rDeviceAgnosticRecipeInfo.m_signalFromGraphInfo);

    return synSuccess;
}

bool DeviceAgnosticRecipeProcessor::_extractSfgInfo(const RecipeTensorsInfo& recipeTensorsInfo,
                                                    const recipe_t*          recipe,
                                                    SignalFromGraphInfo&     rSignalFromGraphInfo)
{
    HB_ASSERT_PTR(recipe);

    rSignalFromGraphInfo.m_sfgExtTensorIdxSortedByExeOrder.clear();
    rSignalFromGraphInfo.m_sfgExtTensorIdxToExeOrder.clear();
    struct ExtTensorOrderWithIdx
    {
        ExtTensorOrderWithIdx(uint32_t _tensorExeOrder, uint32_t _tensorIdx)
        : tensorExeOrder(_tensorExeOrder), tensorIdx(_tensorIdx)
        {
        }
        uint32_t tensorExeOrder;
        uint32_t tensorIdx;
    };
    SmallVector<ExtTensorOrderWithIdx, 16> orderWithOffset;
    orderWithOffset.reserve(recipe->persist_tensors_nr);
    for (uint32_t tensorIdx = 0; tensorIdx < recipe->persist_tensors_nr; ++tensorIdx)
    {
        if (recipe->tensors[tensorIdx].isExternal)
        {
            rSignalFromGraphInfo.m_sfgExtTensorIdxToExeOrder[tensorIdx] = recipe->tensors[tensorIdx].extTensorExeOrder;
            orderWithOffset.emplace_back(recipe->tensors[tensorIdx].extTensorExeOrder, tensorIdx);
        }
    }

    std::sort(orderWithOffset.begin(),
              orderWithOffset.end(),
              [](ExtTensorOrderWithIdx const& lhs, ExtTensorOrderWithIdx const& rhs) {
                  return lhs.tensorExeOrder < rhs.tensorExeOrder;
              });

    for (ExtTensorOrderWithIdx const& item : orderWithOffset)
    {
        uint64_t tensorId;
        recipeTensorsInfo.tensorRetrieveId(recipe->tensors[item.tensorIdx].name, &tensorId);
        rSignalFromGraphInfo.m_sfgExtTensorIdxSortedByExeOrder.push_back(tensorId);
    }

    patch_point_t* pCurrentPatchedPoint = recipe->patch_points;
    for (uint32_t ppIdx = 0; ppIdx < recipe->patch_points_nr; ppIdx++, pCurrentPatchedPoint++)
    {
        if (pCurrentPatchedPoint->type != patch_point_t::SOB_PATCH_POINT)
        {
            continue;
        }
        blob_t*   pCurrentBlob         = recipe->blobs + pCurrentPatchedPoint->blob_idx;
        uint32_t* pPatchedBlobLocation = ((uint32_t*)pCurrentBlob->data) + pCurrentPatchedPoint->dw_offset_in_blob;

        if (rSignalFromGraphInfo.m_lkdSobLowPartAddress == SignalFromGraphInfo::TENSOR_EXE_ORDER_INVALID)
        {
            rSignalFromGraphInfo.m_lkdSobLowPartAddress = *pPatchedBlobLocation;
        }
        else if (rSignalFromGraphInfo.m_lkdSobLowPartAddress != *pPatchedBlobLocation)
        {
            LOG_ERR(SYN_RECIPE,
                    "{}: lkd sob low part address in recipe patch points don't match {} != {}, current patch point "
                    "index {}",
                    HLLOG_FUNC,
                    rSignalFromGraphInfo.m_lkdSobLowPartAddress,
                    *pPatchedBlobLocation,
                    ppIdx);
            return false;
        }
    }
    return true;
}

void DeviceAgnosticRecipeProcessor::parseRecipe(const basicRecipeInfo& rBasicRecipeInfo, synDeviceType& rDeviceType)
{
    auto val = RecipeUtils::getConfVal(rBasicRecipeInfo.recipe, gc_conf_t::DEVICE_TYPE);
    if (!val.is_set())
    {
        rDeviceType = synDeviceTypeInvalid;
    }
    else
    {
        rDeviceType = static_cast<synDeviceType>(val.value());
    }
}

synStatus DeviceAgnosticRecipeProcessor::verifyRecipe(const basicRecipeInfo& rBasicRecipeInfo)
{
    if (!RecipeVerification::verifyRecipe(rBasicRecipeInfo.recipe, rBasicRecipeInfo.shape_plan_recipe))
    {
        return synFail;
    }

    return synSuccess;
}

void DeviceAgnosticRecipeProcessor::getTopologyWorkspaceSize(const basicRecipeInfo& rBasicRecipeInfo,
                                                             synDeviceType          deviceType,
                                                             uint64_t&              rWorkspaceSize)
{
    rWorkspaceSize = rBasicRecipeInfo.recipe->workspace_sizes[MEMORY_ID_RESERVED_FOR_WORKSPACE];

    const bool isScalArch = isScalArchitecture(deviceType);

    if (isScalArch && (RecipeUtils::isKernelPrintf(rBasicRecipeInfo)))
    {
        // For gaudi2 scal, add room in the workspace for the kernels-printf. The prints are copied
        // from the program-data to the workspace after the launch is done
        rWorkspaceSize += RecipeUtils::getTotalKernelPrintfSize(rBasicRecipeInfo);
    }

    LOG_DEBUG(SYN_RECIPE,
              "{}: Scratch pad size: {}, Program Data size: {}, Program Code size: {} return {} isScalArch {}",
              HLLOG_FUNC,
              rBasicRecipeInfo.recipe->workspace_sizes[MEMORY_ID_RESERVED_FOR_WORKSPACE],
              rBasicRecipeInfo.recipe->workspace_sizes[MEMORY_ID_RESERVED_FOR_PROGRAM_DATA],
              MANDATORY_KERNEL_ALIGNMENT + rBasicRecipeInfo.recipe->workspace_sizes[MEMORY_ID_RESERVED_FOR_PROGRAM],
              rWorkspaceSize,
              isScalArch);
}

synStatus DeviceAgnosticRecipeProcessor::recipeGetAttribute(uint64_t*                 retVal,
                                                            const synRecipeAttribute* recipeAttr,
                                                            const unsigned            querySize,
                                                            const synRecipeHandle     recipeHandle)
{
    if (retVal == nullptr)
    {
        LOG_ERR(SYN_RECIPE, "{}: Received null recipe ptr value", HLLOG_FUNC);
        return synFail;
    }

    synStatus status = synSuccess;
    uint64_t  tmpValue;
    for (unsigned queryIndex = 0; queryIndex < querySize; queryIndex++)
    {
        switch (recipeAttr[queryIndex])
        {
            case RECIPE_ATTRIBUTE_WORKSPACE_SIZE:
            {
                retVal[queryIndex] = recipeHandle->deviceAgnosticRecipeHandle.m_workspaceSize;
                break;
            }
            case RECIPE_ATTRIBUTE_NUM_PERSISTENT_TENSORS:
            {
                uint32_t persistentAmount;
                status = recipeHandle->basicRecipeHandle.tensorRetrievePersistentAmount(persistentAmount);
                if (status == synSuccess)
                {
                    retVal[queryIndex] = persistentAmount;
                }
                else
                {
                    return status;
                }
                break;
            }
            case RECIPE_ATTRIBUTE_HOST_MEM_SIZE:
            {
                status = recipeHandle->basicRecipeHandle.getHostMemorySize(tmpValue);
                if (status == synSuccess)
                {
                    retVal[queryIndex] = tmpValue;
                }
                else
                {
                    LOG_ERR(SYN_RECIPE, "Failed to receive size from recipe, query index {}", queryIndex);
                    return status;
                }

                break;
            }
            case RECIPE_ATTRIBUTE_NUM_EXTERNAL_TENSORS:
            {
                retVal[queryIndex] =
                    recipeHandle->deviceAgnosticRecipeHandle.m_signalFromGraphInfo.getNumberOfExternalTensors();
                break;
            }
            case RECIPE_ATTRIBUTE_PERSISTENT_TENSORS_SIZE:
            {
                status = recipeHandle->basicRecipeHandle.getPersistentTensorsMemorySize(tmpValue);
                if (status == synSuccess)
                {
                    retVal[queryIndex] = tmpValue;
                }
                else
                {
                    LOG_ERR(SYN_RECIPE, "Failed to receive size from recipe, query index {}", queryIndex);
                    return status;
                }

                break;
            }
            case RECIPE_ATTRIBUTE_CONST_SECTIONS_SIZE:
            {
                status = recipeHandle->basicRecipeHandle.getConstSectionMemorySize(tmpValue);
                if (status == synSuccess)
                {
                    retVal[queryIndex] = tmpValue;
                }
                else
                {
                    LOG_ERR(SYN_RECIPE, "Failed to receive size from recipe, query index {}", queryIndex);
                    return status;
                }

                break;
            }
            case RECIPE_ATTRIBUTE_DEVICE_MEMORY_SIZE:
            {
                status = recipeHandle->basicRecipeHandle.getRecipeHbmMemorySize(tmpValue);
                if (status == synSuccess)
                {
                    retVal[queryIndex] = tmpValue;
                }
                else
                {
                    LOG_ERR(SYN_RECIPE, "Failed to receive size from recipe, query index {}", queryIndex);
                    return status;
                }

                break;
            }
            default:
            {
                LOG_ERR(SYN_RECIPE,
                        "Got unsupported recipe attribute {} in query index {}",
                        recipeAttr[queryIndex],
                        queryIndex);
                return synUnsupported;
            }
        }
    }
    return synSuccess;
}

bool DeviceAgnosticRecipeProcessor::isScalArchitecture(synDeviceType deviceType)
{
    return (deviceType == synDeviceGaudi2) || (deviceType == synDeviceGaudi3);
}
