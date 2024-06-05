#include "runtime/common/recipe/recipe_patch_processor.hpp"
#include "runtime/common/recipe/recipe_tensor_processor.hpp"
#include "runtime/common/recipe/basic_recipe_info.hpp"
#include "runtime/common/recipe/patching/host_address_patcher.hpp"
#include "utils.h"
#include "habana_global_conf_runtime.h"
#include "host_to_virtual_address_mapper.hpp"
#include "device_agnostic_recipe_info.hpp"
#include "runtime/common/device/device_mem_alloc.hpp"

static uint16_t const NUM_TENSOR_TYPES = 2;

// Verify dynamic tensor size: if element size is 4 bit, then dim0 must be even
static bool
sanityCheckTensor(const recipe_t* recipeHandle, const synLaunchTensorInfoExt* launchTensorsInfo, uint64_t idx)
{
    if (launchTensorsInfo->tensorType == DATA_TENSOR_DYNAMIC)
    {
        if ((recipeHandle->tensors[idx].elementType == syn_type_int4) ||
            (recipeHandle->tensors[idx].elementType == syn_type_uint4))
        {
            if (launchTensorsInfo->tensorSize[0] % 2 == 1)
            {
                LOG_ERR_T(SYN_RECIPE,
                          "Tensor is 4 bit, but dim0 is not even. Tensor name {}, dim0 {}",
                          launchTensorsInfo->tensorName,
                          launchTensorsInfo->tensorSize[0]);
                return false;
            }
        }
    }
    return true;
}

static bool
getMmuVirtualAddress(DevMemoryAllocInterface& rDevMemAlloc, const uint64_t hostAddress, uint64_t* o_deviceVirtAddr)
{
    uint64_t                            deviceVirtAddr = 0;
    eHostAddrToVirtualAddrMappingStatus mappingStatus;
    mappingStatus = rDevMemAlloc.getDeviceVirtualAddress(true, (void*)hostAddress, 0, &deviceVirtAddr);
    if (mappingStatus != HATVA_MAPPING_STATUS_FOUND)
    {
        LOG_ERR(SYN_API, "Can not find a source VA for host-address 0x{:x} status {})", hostAddress, mappingStatus);
        return false;
    }

    *o_deviceVirtAddr = deviceVirtAddr;
    return true;
}

synStatus RecipePatchProcessor::process(const basicRecipeInfo&                    rBasicRecipeInfo,
                                        const RecipeTensorsInfo&                  rRecipeTensorsInfo,
                                        const synLaunchTensorInfoExt*             enqueueTensorsInfo,
                                        uint32_t                                  enqueueTensorsInfoAmount,
                                        uint32_t                                  enqueueFlags,
                                        const WorkSpacesInformation&              rWorkSpacesInformation,
                                        patching::HostAddressPatchingInformation& rPatchInfo,
                                        DevMemoryAllocInterface&                  rDevMemAlloc,
                                        std::vector<uint32_t>*                    tensorIdx2userIdx,
                                        bool                                      isInitAndCompletionRequired,
                                        bool                                      shouldResolveTensorsIndices,
                                        const ValidSectionAddresses*              pValidSectionAddresses)
{
    synStatus status = synSuccess;
    if (isInitAndCompletionRequired)
    {
        rPatchInfo.initialize(rRecipeTensorsInfo.m_maxSectionId, rRecipeTensorsInfo.m_numSectionsToPatch);
    }

    bool res = _setWorkspacesSectionAddress(rBasicRecipeInfo,
                                            rWorkSpacesInformation,
                                            rPatchInfo,
                                            &rRecipeTensorsInfo.m_sectionToSectionType);
    if (!res)
    {
        return synFail;
    }

    status = _resolveSectionsAddressesAndTensorsIndices(rBasicRecipeInfo,
                                                        rRecipeTensorsInfo,
                                                        enqueueTensorsInfoAmount,
                                                        enqueueTensorsInfo,
                                                        enqueueFlags,
                                                        rPatchInfo,
                                                        tensorIdx2userIdx,
                                                        &rRecipeTensorsInfo.m_sectionToSectionType,
                                                        rDevMemAlloc,
                                                        shouldResolveTensorsIndices,
                                                        rWorkSpacesInformation,
                                                        pValidSectionAddresses);

    if (isInitAndCompletionRequired)
    {
        if (status != synSuccess)
        {
           rPatchInfo.patchingAbort();
        }
        else
        {
            rPatchInfo.patchingCompletion();
        }
    }

    return status;
}

bool RecipePatchProcessor::_setWorkspacesSectionAddress(
    const basicRecipeInfo&                    rBasicRecipeInfo,
    const WorkSpacesInformation&              workspaceInfo,
    patching::HostAddressPatchingInformation& hostAddressPatchingInfo,
    const SectionToSectionType*               pSectionToSectionType)
{
    uint64_t* pCurrWorkspaceSize   = rBasicRecipeInfo.recipe->workspace_sizes;
    uint64_t  currWorkspaceAddress = 0;
    const bool isDsd                = rBasicRecipeInfo.shape_plan_recipe != nullptr;

    LOG_TRACE(SYN_RECIPE, "{}", HLLOG_FUNC);

    for (uint64_t i = 0; i < rBasicRecipeInfo.recipe->workspace_nr; i++, pCurrWorkspaceSize++)
    {
        bool     addToPatchingInformation = true;
        uint64_t sectionSize              = rBasicRecipeInfo.recipe->workspace_sizes[i];
        bool     isZeroSectionSize        = (sectionSize == 0);
        if (i == MEMORY_ID_RESERVED_FOR_PROGRAM_DATA)
        {
            currWorkspaceAddress     =  round_to_multiple(workspaceInfo.programDataAddress, (1 << 13));
            addToPatchingInformation &= !isZeroSectionSize;
        }
        else if (i == MEMORY_ID_RESERVED_FOR_WORKSPACE)
        {
            currWorkspaceAddress     =  workspaceInfo.scratchPadAddress;
            addToPatchingInformation &= !isZeroSectionSize;
        }
        else if (i == MEMORY_ID_RESERVED_FOR_PROGRAM)
        {
            currWorkspaceAddress     =  workspaceInfo.programCodeAddress;
            addToPatchingInformation &= false;
        }
        else
        {
            LOG_ERR(SYN_RECIPE, "{}: Failed to update workspace index {}", HLLOG_FUNC, i);
            hostAddressPatchingInfo.patchingAbort();
            return false;
        }

        if (addToPatchingInformation)
        {
            uint64_t sectionTypeId = (pSectionToSectionType == nullptr) ? 0 : (*pSectionToSectionType)[i];

            bool res = hostAddressPatchingInfo.setSectionHostAddress(i,
                                                                     sectionTypeId,
                                                                     currWorkspaceAddress,
                                                                     isZeroSectionSize,
                                                                     isDsd,
                                                                     (i == MEMORY_ID_RESERVED_FOR_WORKSPACE));
            if (!res)
            {
                LOG_ERR(SYN_RECIPE,
                        "{}: Failed to setSectionHostAddress for section {} with address {:#x}",
                        HLLOG_FUNC,
                        i,
                        currWorkspaceAddress);
                hostAddressPatchingInfo.patchingAbort();
                return false;
            }
        }
    }

    // Assuming that assertAsyncMappedAddress must have a non zero address to be set
    bool res = hostAddressPatchingInfo.setSectionHostAddress(
        MEMORY_ID_RESERVED_FOR_ASSERT_ASYNC,
        0,
        workspaceInfo.assertAsyncMappedAddress,
        (workspaceInfo.assertAsyncMappedAddress == 0) ? true : false,
        isDsd);
    if (!res)
    {
        LOG_ERR(SYN_RECIPE,
                "{}: Failed to setSectionHostAddress for section {} with address {:#x}",
                HLLOG_FUNC,
                MEMORY_ID_RESERVED_FOR_ASSERT_ASYNC,
                workspaceInfo.assertAsyncMappedAddress);
        hostAddressPatchingInfo.patchingAbort();
        return false;
    }

    return true;
}

// Note, pValidSectionAddresses is optional. If passed as nullptr, range check is not done
synStatus RecipePatchProcessor::validateSectionsInfo(const basicRecipeInfo&          rBasicRecipeInfo,
                                                     const RecipeTensorsInfo&        rRecipeInfo,
                                                     const uint32_t                  launchTensorsAmount,
                                                     const synLaunchTensorInfoExt*   launchTensorsInfo,
                                                     uint32_t                        flags,
                                                     const WorkSpacesInformation&    workspaceInfo,
                                                     const SectionSizeInfoVec&       sectionsSizeInfo,
                                                     DevMemoryAllocInterface&        rDevMemAlloc)
{
    const uint64_t maxSectionId = sectionsSizeInfo.size();
    uint64_t       currSectionAddr;
    const char*    tensorName;
    IdxAndType     idxAndType;
    uint64_t       tensorInfoIndex;
    synTensorType  tensorInfoType;

    HB_ASSERT(static_cast<uint32_t>(tensor_info_t::PERSISTENT_TENSOR) == 0, "Used for array access");
    HB_ASSERT(static_cast<uint32_t>(tensor_info_t::SHAPE_TENSOR) == 1, "Used for array access");

    LOG_TRACE(SYN_RECIPE,
              "{}: Start validate new tensors for Launch amount {} pointer 0x{:x}",
              HLLOG_FUNC,
              launchTensorsAmount,
              (uint64_t)launchTensorsInfo);

    SortedSectionsAddressInfo sortedSectionStart;  // [address, [section_id,size]]
    SectionsAddressDB         sectionAddresses;    // [section_id, address]

    if (!_addRecipeReservedSectionsWorkspaceAddresses(sortedSectionStart, rBasicRecipeInfo, workspaceInfo))
    {
        return synFail;
    }

    bool checkSectionsOverlap = GCFG_CHECK_SECTION_OVERLAP.value();

    const shape_plane_graph_t* pShapePlainGraph = rBasicRecipeInfo.shape_plan_recipe;
    bool                       isDsd            = (pShapePlainGraph != nullptr);

    // We need to support Persistent and Shape Tensors (2)
    const persist_tensor_info_t* persistentTensors    = rBasicRecipeInfo.recipe->tensors;
    const uint64_t               numberPersistTensors = rBasicRecipeInfo.recipe->persist_tensors_nr;

    const shape_tensor_info_t* shapeTensors      = isDsd ? pShapePlainGraph->shape_tensors : nullptr;
    uint64_t                   numOfShapeTensors = isDsd ? pShapePlainGraph->shape_tensors_list_nr : 0;

    uint64_t expectedNumTensors[NUM_TENSOR_TYPES] = {numberPersistTensors, numOfShapeTensors};
    uint64_t tensorCnt[NUM_TENSOR_TYPES] {};

    // tensorGiven holds tensors supplied by the User and tensors marked as not needed (zero section size)
    TensorsValidationInfoDB tensorGiven[NUM_TENSOR_TYPES];
    for (int i = 0; i < NUM_TENSOR_TYPES; i++)
    {
        tensorGiven[i].resize(expectedNumTensors[i], false);
    }

    sectionAddresses.resize(maxSectionId + 1);

    for (uint32_t loop = 0; loop < launchTensorsAmount; loop++)
    {
        const synLaunchTensorInfoExt& currTensorLaunchInfo = launchTensorsInfo[loop];

        uint64_t launchTensorId = currTensorLaunchInfo.tensorId;

        // skip on invalid tensors, they can be delivered by launch
        if (IS_TENSOR_INVALID(launchTensorId))
        {
            continue;
        }

        tensorName      = currTensorLaunchInfo.tensorName;
        tensorInfoType  = TENSOR_INFO_TO_TYPE(launchTensorId);
        tensorInfoIndex = TENSOR_INFO_TO_INDEX(launchTensorId);

        if (!_validateTensorType(isDsd, tensorInfoIndex, tensorName, currTensorLaunchInfo.tensorType, tensorInfoType))
        {
            return synInvalidTensorProperties;
        }

        if (!_shouldHandleValidTensor(tensorGiven, tensorCnt, tensorInfoIndex, tensorName, tensorInfoType))
        {
            // In case already handled or not a PT we can continue to the next tensor
            continue;
        }

        if (sanityCheckTensor(rBasicRecipeInfo.recipe, &currTensorLaunchInfo, tensorInfoIndex) == false)
        {
            return synInvalidTensorProperties;
        }

        uint64_t currSectionIdx = persistentTensors[tensorInfoIndex].section_idx;
        HB_ASSERT(currSectionIdx <= maxSectionId,
                  "sectionId {} is larger than masSectionID {}",
                  currSectionIdx,
                  maxSectionId);

        bool     isTensorZeroAddress   = currTensorLaunchInfo.pTensorAddress == 0;
        uint64_t tensorOffsetInSection = persistentTensors[tensorInfoIndex].offset_in_section;

        if (isTensorZeroAddress)
        {
            // the section address can be 0 if the tensor is alone on the section.
            // or deduce it from another tensor included in the same section
            continue;
        }

        currSectionAddr = currTensorLaunchInfo.pTensorAddress - tensorOffsetInSection;

        if (currTensorLaunchInfo.tensorType == HOST_TO_DEVICE_TENSOR)
        {
            uint64_t deviceSectionAddr = 0;
            if (!getMmuVirtualAddress(rDevMemAlloc, currSectionAddr, &deviceSectionAddr))
            {
                return synMappingNotFound;
            }
            LOG_DEBUG(SYN_PATCH_INFO,
                      "{}: Found section virtual address 0x{:x} for host address index: 0x{:x} for tensor {}",
                      HLLOG_FUNC,
                      deviceSectionAddr,
                      currSectionAddr,
                      (tensorName != nullptr) ? tensorName : "???");
            continue;
        }

        if (!_addTensorSectionAddressToDB(sectionAddresses, currSectionIdx, currSectionAddr))
        {
            return synFailedSectionValidation;
        }

        if (checkSectionsOverlap)
        {
            bool insertSectionStatus = _insertSectionSize(sortedSectionStart,
                                                          currSectionIdx,
                                                          tensorInfoIndex,
                                                          currSectionAddr,
                                                          sectionsSizeInfo,
                                                          currTensorLaunchInfo,
                                                          tensorInfoType,
                                                          rBasicRecipeInfo.recipe->tensors,
                                                          rBasicRecipeInfo.recipe->persist_tensors_nr);

            if (!insertSectionStatus)
            {
                _logLaunchTensorsInfo(sortedSectionStart,
                                      launchTensorsInfo,
                                      rBasicRecipeInfo.recipe->tensors,
                                      rBasicRecipeInfo.recipe->persist_tensors_nr,
                                      launchTensorsAmount,
                                      maxSectionId);

                return synFailedSectionValidation;
            }
        }
    }

    _markConstZeroSizeTensors(rRecipeInfo.m_constZeroSizeTensors, tensorGiven, tensorCnt);

    for (uint8_t i = 0; i < NUM_TENSOR_TYPES; i++)
    {
        bool tensorsGivenStatus = _validateAllTensorsGiven(i,
                                                           tensorCnt[i],
                                                           expectedNumTensors[i],
                                                           tensorGiven[i],
                                                           rBasicRecipeInfo.recipe->tensors,
                                                           shapeTensors,
                                                           rBasicRecipeInfo.recipe->persist_tensors_nr,
                                                           numOfShapeTensors);

        if (!tensorsGivenStatus)
        {
            return synFailedSectionValidation;
        }
    }
    if (checkSectionsOverlap && sortedSectionStart.size() > 0)
    {
        // All sections' addresses are known. check if there's an overlap.
        if (!_checkSectionsOverlap(sortedSectionStart))
        {
            return synFailedSectionValidation;
        }
    }

    return synSuccess;
}

WorkSpacesInformation::WorkSpacesInformation()
{
    scratchPadAddress  = 0;
    programCodeAddress = 0;
    programDataAddress = 0;
    programCodeInCache = false;
    programDataInCache = false;
}

bool RecipePatchProcessor::resolveTensorsIndices(std::vector<uint32_t>*        tensorIdx2userIdx,
                                                 const basicRecipeInfo&        rBasicRecipeInfo,
                                                 const uint32_t                launchTensorsAmount,
                                                 const synLaunchTensorInfoExt* launchTensorsInfo)
{
    LOG_TRACE(SYN_RECIPE,
              "{}: Launch tensors amount {} pointer 0x{:x}",
              HLLOG_FUNC,
              launchTensorsAmount,
              (uint64_t)launchTensorsInfo);

    const uint64_t numberPersistTensors = rBasicRecipeInfo.recipe->persist_tensors_nr;
    const uint64_t numberShapeTensors   = rBasicRecipeInfo.shape_plan_recipe != nullptr ?
                                          rBasicRecipeInfo.shape_plan_recipe->shape_tensors_list_nr : 0;

    // Init DB
    uint64_t expectedNumTensors[NUM_TENSOR_TYPES] = {numberPersistTensors, numberShapeTensors};
    for (int i = 0; i < NUM_TENSOR_TYPES; i++)
    {
        tensorIdx2userIdx[i].clear();
        tensorIdx2userIdx[i].resize(expectedNumTensors[i], INVALID_TENSOR_INDEX);
    }

    tensor_info_t::ETensorType tensorRecipeType = tensor_info_t::PERSISTENT_TENSOR;
    synTensorType              type             = TENSOR_TYPE_INVALID;
    uint64_t                   tensorIndex      = 0;
    //
    // Resolve DB
    for (uint32_t i = 0; i < launchTensorsAmount; i++)
    {
        uint64_t tensorId = launchTensorsInfo[i].tensorId;

        // skip on invalid tensors, they can be delivered by launch
        if (IS_TENSOR_INVALID(tensorId))
        {
            continue;
        }

        _retrieveTensorInfo(tensorRecipeType, type, tensorIndex, tensorId);

        if (!_resolveSingleTensorIdToTensorIndex(tensorRecipeType,
                                                 tensorIndex,
                                                 numberShapeTensors,
                                                 numberPersistTensors))
        {
            return false;
        }

        tensorIdx2userIdx[tensorRecipeType][tensorIndex] = i;
    }

    return true;
}

bool RecipePatchProcessor::_checkWorkspaceAddresses(const basicRecipeInfo&                    rBasicRecipeInfo,
                                                    const WorkSpacesInformation&              rWorkSpacesInformation,
                                                    const ValidSectionAddresses*              pValidSectionAddresses)
{
    // check workspace addresses
    uint64_t* pReservedSectionsSizes  = rBasicRecipeInfo.recipe->workspace_sizes;
    uint64_t size = pReservedSectionsSizes[MEMORY_ID_RESERVED_FOR_WORKSPACE];
    if ( size > 0)
    {
        if (_isOutOfRange(rWorkSpacesInformation.scratchPadAddress, size, pValidSectionAddresses))
        {
            LOG_ERR(SYN_RECIPE,
                    "{}: Out of Range.  scratchPadAddress address 0x{:x} size 0x{:x}",
                    HLLOG_FUNC,
                    rWorkSpacesInformation.scratchPadAddress,
                    size);
            return false;
        }
    }
    size = pReservedSectionsSizes[MEMORY_ID_RESERVED_FOR_PROGRAM_DATA];
    if ( size > 0 && (!rWorkSpacesInformation.programDataInCache))
    {
        uint64_t baseAddr = round_to_multiple(rWorkSpacesInformation.programDataAddress, (1 << 13));
        if (_isOutOfRange(baseAddr, size, pValidSectionAddresses))
        {
            LOG_ERR(SYN_RECIPE,
                    "{}: Out of Range.  programDataAddress address 0x{:x} size 0x{:x}",
                    HLLOG_FUNC,
                    baseAddr,
                    size);
            return false;
        }
    }
    size = pReservedSectionsSizes[MEMORY_ID_RESERVED_FOR_PROGRAM];
    if ( size > 0 && (!rWorkSpacesInformation.programCodeInCache))
    {
        uint64_t baseAddr  = rWorkSpacesInformation.programCodeAddress;
        if (_isOutOfRange(baseAddr, size, pValidSectionAddresses))
        {
            LOG_ERR(SYN_RECIPE,
                    "{}: Out of Range.  programCodeAddress address 0x{:x} size 0x{:x}",
                    HLLOG_FUNC,
                    baseAddr,
                    size);
            return false;
        }
    }
    return true;
}

synStatus RecipePatchProcessor::_checkTensorAddresses(const basicRecipeInfo&                    rBasicRecipeInfo,
                                                      const synLaunchTensorInfoExt*             launchTensorsInfo,
                                                      DevMemoryAllocInterface&                  rDevMemAlloc,
                                                      uint32_t                                  tensorIdx,
                                                      uint64_t                                  tensorId,
                                                      const RecipeTensorsInfo&                  rRecipeTensorInfo,
                                                      const ValidSectionAddresses*              pValidSectionAddresses)
{
    tensorId = TENSOR_INFO_TO_INDEX(tensorId);
    if(tensorId >= rBasicRecipeInfo.recipe->persist_tensors_nr)
    {
        return synFail;
    }

    uint64_t currTensorAddr = launchTensorsInfo[tensorIdx].pTensorAddress;
    if (currTensorAddr == 0)
    {
        // No reason to perform address-related verification
        //
        // It is allowed that some of the tensors in a section (but not all) will have zero-address
        //
        // In case all tensors in a section have zero address, none of them will be tested below,
        // but the is-section-set verification will fail
        return synSuccess;
    }

    uint64_t currSectionIdx        = rBasicRecipeInfo.recipe->tensors[tensorId].section_idx;
    uint64_t tensorOffsetInSection = rBasicRecipeInfo.recipe->tensors[tensorId].offset_in_section;
    uint64_t currSectionAddr       = currTensorAddr - tensorOffsetInSection;
    if (launchTensorsInfo[tensorIdx].tensorType == HOST_TO_DEVICE_TENSOR)
    {
        uint64_t deviceSectionAddr = 0;
        if (!getMmuVirtualAddress(rDevMemAlloc, currSectionAddr, &deviceSectionAddr))
        {
            LOG_ERR(SYN_RECIPE,
                    "{}: HOST_TO_DEVICE_TENSOR {} address 0x{:x} failed to get mmu addr",
                    HLLOG_FUNC,
                    currSectionIdx,
                    currSectionAddr);
            return synMappingNotFound;
        }
        return synSuccess; // skip the boundaries check
    }

    // Sadly, we check this per tensor and not per section...
    uint64_t sectionSize = rRecipeTensorInfo.m_sectionsInfo[currSectionIdx].sectionSize;
    if (_isOutOfRange(currSectionAddr, sectionSize, pValidSectionAddresses))
    {
        LOG_ERR(SYN_RECIPE,
                "{}: Out of Range. section {} address 0x{:x} size 0x{:x}",
                HLLOG_FUNC,
                currSectionIdx,
                currSectionAddr,
                sectionSize);
        return synFail;
    }

    return synSuccess;
}

synStatus RecipePatchProcessor::resolveSectionsAddresses(const basicRecipeInfo&                    rBasicRecipeInfo,
                                                         const uint32_t                            launchTensorsAmount,
                                                         const synLaunchTensorInfoExt*             launchTensorsInfo,
                                                         patching::HostAddressPatchingInformation& hostAddressPatchingInfo,
                                                         const SectionToSectionType*               sectionToSectionType,
                                                         DevMemoryAllocInterface&                  rDevMemAlloc,
                                                         const WorkSpacesInformation&              rWorkSpacesInformation,
                                                         const RecipeTensorsInfo&                  rRecipeTensorInfo,
                                                         const ValidSectionAddresses*              pValidSectionAddresses)
{
    LOG_TRACE(SYN_RECIPE,
              "{}: Launch tensors amount {} pointer 0x{:x}",
              HLLOG_FUNC,
              launchTensorsAmount,
              (uint64_t)launchTensorsInfo);

    tensor_info_t::ETensorType tensorRecipeType = tensor_info_t::PERSISTENT_TENSOR;
    synTensorType              tensorType       = TENSOR_TYPE_INVALID;
    uint64_t                   tensorIndex      = 0;

    if (!_checkWorkspaceAddresses(rBasicRecipeInfo, rWorkSpacesInformation, pValidSectionAddresses))
    {
        return synFail;
    }

    bool checkSectionsOverlap = GCFG_CHECK_SECTION_OVERLAP.value();

    for (uint32_t i = 0; i < launchTensorsAmount; i++)
    {
        uint64_t tensorId = launchTensorsInfo[i].tensorId;

        // skip on invalid tensors, they can be delivered by launch
        if (IS_TENSOR_INVALID(tensorId))
        {
            continue;
        }

        _retrieveTensorInfo(tensorRecipeType, tensorType, tensorIndex, tensorId);

        if (tensorRecipeType != tensor_info_t::PERSISTENT_TENSOR)  // code below relevant only for persistent Tensors
        {
            continue;
        }

        synStatus status = _resolveSingleTensorSectionsAddresses(rBasicRecipeInfo,
                                                                 &launchTensorsInfo[i],
                                                                 hostAddressPatchingInfo,
                                                                 sectionToSectionType,
                                                                 rRecipeTensorInfo,
                                                                 rDevMemAlloc,
                                                                 tensorIndex);
        if (status != synSuccess)
        {
            return status;
        }

        if (checkSectionsOverlap)
        {
            status = _checkTensorAddresses(rBasicRecipeInfo, launchTensorsInfo,
                                           rDevMemAlloc, i, tensorId,
                                           rRecipeTensorInfo, pValidSectionAddresses);

            if (status != synSuccess)
            {
                return status;
            }
        }
    }

    if (!hostAddressPatchingInfo.validateAllSectionsAddressSet())
    {
        LOG_ERR(SYN_RECIPE, "{}: Invalid patching information, some sections were not set", HLLOG_FUNC);
        hostAddressPatchingInfo.patchingAbort();
        return synFail;
    }

    return synSuccess;
}


synStatus RecipePatchProcessor::_resolveBothSectionsAddressesAndTensorsIndices(
    const basicRecipeInfo&                    rBasicRecipeInfo,
    const uint32_t                            launchTensorsAmount,
    const synLaunchTensorInfoExt*             launchTensorsInfo,
    std::vector<uint32_t>*                    tensorIdx2userIdx,
    patching::HostAddressPatchingInformation& hostAddressPatchingInfo,
    const SectionToSectionType*               sectionToSectionType,
    DevMemoryAllocInterface&                  rDevMemAlloc,
    const WorkSpacesInformation&              rWorkSpacesInformation,
    const RecipeTensorsInfo&                  rRecipeTensorInfo,
    const ValidSectionAddresses*              pValidSectionAddresses)
{
    const uint64_t numberPersistTensors = rBasicRecipeInfo.recipe->persist_tensors_nr;
    const uint64_t numberShapeTensors   = rBasicRecipeInfo.shape_plan_recipe != nullptr ?
                                          rBasicRecipeInfo.shape_plan_recipe->shape_tensors_list_nr : 0;

    if (!_checkWorkspaceAddresses(rBasicRecipeInfo, rWorkSpacesInformation, pValidSectionAddresses))
    {
        return synFail;
    }
    tensor_info_t::ETensorType tensorRecipeType = tensor_info_t::PERSISTENT_TENSOR;
    synTensorType              tensorType       = TENSOR_TYPE_INVALID;
    uint64_t                   tensorIndex      = 0;
    bool checkSectionsOverlap = GCFG_CHECK_SECTION_OVERLAP.value();

    for (uint32_t i = 0; i < launchTensorsAmount; i++)
    {
        uint64_t tensorId = launchTensorsInfo[i].tensorId;

        // skip on invalid tensors, they can be delivered by launch
        if (IS_TENSOR_INVALID(tensorId))
        {
            continue;
        }

        _retrieveTensorInfo(tensorRecipeType, tensorType, tensorIndex, tensorId);

        if (!_resolveSingleTensorIdToTensorIndex(tensorRecipeType,
                                                 tensorIndex,
                                                 numberShapeTensors,
                                                 numberPersistTensors))
        {
            return synFail;
        }

        tensorIdx2userIdx[tensorRecipeType][tensorIndex] = i;

        if (tensorRecipeType != tensor_info_t::PERSISTENT_TENSOR)  // code below relevant only for persistent Tensors
        {
            continue;
        }

        synStatus status = _resolveSingleTensorSectionsAddresses(rBasicRecipeInfo,
                                                                 &launchTensorsInfo[i],
                                                                 hostAddressPatchingInfo,
                                                                 sectionToSectionType,
                                                                 rRecipeTensorInfo,
                                                                 rDevMemAlloc,
                                                                 tensorIndex);
        if (status != synSuccess)
        {
            return status;
        }

        if (checkSectionsOverlap)
        {
            status = _checkTensorAddresses(rBasicRecipeInfo, launchTensorsInfo,
                                           rDevMemAlloc, i, tensorId,
                                           rRecipeTensorInfo, pValidSectionAddresses);

            if (status != synSuccess)
            {
                return status;
            }
        }
    }


    return synSuccess;
}

synStatus RecipePatchProcessor::_resolveSectionsAddressesAndTensorsIndices(
    const basicRecipeInfo&                    rBasicRecipeInfo,
    const RecipeTensorsInfo&                  rRecipeTensorInfo,
    const uint32_t                            launchTensorsAmount,
    const synLaunchTensorInfoExt*             launchTensorsInfo,
    uint32_t                                  flags,
    patching::HostAddressPatchingInformation& hostAddressPatchingInfo,
    std::vector<uint32_t>*                    tensorIdx2userIdx,
    const SectionToSectionType*               sectionToSectionType,
    DevMemoryAllocInterface&                  rDevMemAlloc,
    bool                                      shouldResolveTensorsIndices,
    const WorkSpacesInformation&              rWorkSpacesInformation,
    const ValidSectionAddresses*              pValidSectionAddresses)
{
    LOG_TRACE(SYN_RECIPE,
              "{}: Launch tensors amount {} pointer 0x{:x}",
              HLLOG_FUNC,
              launchTensorsAmount,
              (uint64_t)launchTensorsInfo);

    synStatus status(synSuccess);

    if (shouldResolveTensorsIndices)
    {
        const uint64_t numberPersistTensors = rBasicRecipeInfo.recipe->persist_tensors_nr;
        const uint64_t numberShapeTensors =
            rBasicRecipeInfo.shape_plan_recipe != nullptr ? rBasicRecipeInfo.shape_plan_recipe->shape_tensors_list_nr : 0;
        uint64_t expectedNumTensors[NUM_TENSOR_TYPES] = {numberPersistTensors, numberShapeTensors};
        for (int i = 0; i < NUM_TENSOR_TYPES; i++)
        {
            tensorIdx2userIdx[i].clear();
            tensorIdx2userIdx[i].resize(expectedNumTensors[i], INVALID_TENSOR_INDEX);
        }

        status = _resolveBothSectionsAddressesAndTensorsIndices(rBasicRecipeInfo,
                                                                launchTensorsAmount,
                                                                launchTensorsInfo,
                                                                tensorIdx2userIdx,
                                                                hostAddressPatchingInfo,
                                                                sectionToSectionType,
                                                                rDevMemAlloc,
                                                                rWorkSpacesInformation,
                                                                rRecipeTensorInfo,
                                                                pValidSectionAddresses);
    }
    else
    {
        status = resolveSectionsAddresses(rBasicRecipeInfo,
                                          launchTensorsAmount,
                                          launchTensorsInfo,
                                          hostAddressPatchingInfo,
                                          sectionToSectionType,
                                          rDevMemAlloc,
                                          rWorkSpacesInformation,
                                          rRecipeTensorInfo,
                                          pValidSectionAddresses);
    }

    if (status != synSuccess)
    {
        return status;
    }

    // some secions may be 0 size, and user didn't gave tensorInfo for therm
    for (auto sectionid : rRecipeTensorInfo.m_constZeroSizeSections)
    {
        bool res = hostAddressPatchingInfo.markConstZeroSizeSection(sectionid);
        if (!res)
        {
            return synFail;
        }
    }

    if (!hostAddressPatchingInfo.validateAllSectionsAddressSet())
    {
        LOG_ERR(SYN_RECIPE, "{}: Invalid patching information, some sections were not set", HLLOG_FUNC);
        hostAddressPatchingInfo.patchingAbort();
        return synFail;
    }

    return synSuccess;
}

tensor_info_t::ETensorType RecipePatchProcessor::userTypeToRecipeType(synTensorType type)
{
    switch (type)
    {
        case DATA_TENSOR:
        case DATA_TENSOR_DYNAMIC:
        case DEVICE_SHAPE_TENSOR:
        case HOST_TO_DEVICE_TENSOR:
            return tensor_info_t::PERSISTENT_TENSOR;

        case OUTPUT_DESCRIBING_SHAPE_TENSOR:
        case INPUT_DESCRIBING_SHAPE_TENSOR:
        case HOST_SHAPE_TENSOR:
            return tensor_info_t::SHAPE_TENSOR;

        case TENSOR_TYPE_MAX:
            HB_ASSERT(0, "Should never be here");
            return tensor_info_t::PERSISTENT_TENSOR;  // just return something
    }
    return tensor_info_t::PERSISTENT_TENSOR;  // we should never get here, just to make the compiler happy
}

bool RecipePatchProcessor::_addRecipeReservedSectionsWorkspaceAddresses(SortedSectionsAddressInfo&   sortedSections,
                                                                        const basicRecipeInfo&       rBasicRecipeInfo,
                                                                        const WorkSpacesInformation& workspaceInfo)
{
    uint64_t  currentWorkspaceAddress = 0;
    uint64_t* pReservedSectionsSizes  = rBasicRecipeInfo.recipe->workspace_sizes;

    for (uint64_t i = 0; i < rBasicRecipeInfo.recipe->workspace_nr; i++, pReservedSectionsSizes++)
    {
        uint64_t currentWorkspaceSize      = *pReservedSectionsSizes;
        bool     addToUserSectionsAnalysis = (rBasicRecipeInfo.recipe->workspace_sizes[i] != 0);

        if (i == MEMORY_ID_RESERVED_FOR_PROGRAM_DATA)
        {
            currentWorkspaceAddress = round_to_multiple(workspaceInfo.programDataAddress, (1 << 13));
            addToUserSectionsAnalysis &= (!workspaceInfo.programDataInCache);
        }
        else if (i == MEMORY_ID_RESERVED_FOR_WORKSPACE)
        {
            currentWorkspaceAddress = workspaceInfo.scratchPadAddress;
        }
        else if (i == MEMORY_ID_RESERVED_FOR_PROGRAM)
        {
            currentWorkspaceAddress = workspaceInfo.programCodeAddress;
            addToUserSectionsAnalysis &= (!workspaceInfo.programCodeInCache);
        }
        else
        {
            LOG_ERR(SYN_RECIPE, "{}: Failed to update workspace index {}", HLLOG_FUNC, i);
            return false;
        }

        // Insert only the sections placed in WS, not in cache
        if (addToUserSectionsAnalysis)
        {
            sortedSections[currentWorkspaceAddress].sectionSize = currentWorkspaceSize;
            sortedSections[currentWorkspaceAddress].sectionId   = i;
        }
    }

    return true;
}

bool RecipePatchProcessor::_validateTensorType(bool                 isDsd,
                                               uint64_t             tensorInfoIndex,
                                               const char*&         tensorName,
                                               const synTensorType& launchTensorType,
                                               const synTensorType& expectedType)
{
    // For DSD only - need to validate tensor-type
    if (!isDsd)
    {
        return true;
    }

    // type is set only for DSD recipe
    if (launchTensorType != expectedType)  // Tensor found, but wrong type
    {
        LOG_ERR_T(SYN_RECIPE,
                  "Wrong tensor type. Tensor {} {} given type {} {} expected {} {}",
                  tensorInfoIndex,
                  (tensorName) ? tensorName : "",
                  (uint8_t)launchTensorType,
                  synTensorType2Txt(launchTensorType),
                  expectedType,
                  synTensorType2Txt(expectedType));

        return false;
    }

    return true;
}

bool RecipePatchProcessor::_shouldHandleValidTensor(TensorsValidationInfoDB* tensorsValidationInfo,
                                                    uint64_t*                tensorCntDB,
                                                    uint64_t                 tensorInfoIndex,
                                                    const char*&             tensorName,
                                                    const synTensorType&     launchTensorType)
{
    HB_ASSERT_PTR(tensorsValidationInfo);
    HB_ASSERT_PTR(tensorCntDB);

    tensor_info_t::ETensorType tensorRecipeType = userTypeToRecipeType(launchTensorType);

    if (tensorsValidationInfo[tensorRecipeType][tensorInfoIndex] != false)  // Already got this tensor
    {
        LOG_DEBUG(SYN_RECIPE,
                  "Tensor already handled: {} type {} {}",
                  tensorName ? tensorName : "",
                  launchTensorType,
                  synTensorType2Txt(launchTensorType));

        return false;
    }

    tensorsValidationInfo[tensorRecipeType][tensorInfoIndex] = true;
    tensorCntDB[tensorRecipeType]++;

    if (tensorRecipeType != tensor_info_t::PERSISTENT_TENSOR)  // code below relevant only for persistent Tensors
    {
        return false;
    }

    return true;
}

bool RecipePatchProcessor::_addTensorSectionAddressToDB(SectionsAddressDB& sectionAddresses,
                                                        uint64_t           currSectionIdx,
                                                        uint64_t           currSectionAddr)
{
    bool firstSeen = sectionAddresses[currSectionIdx] == 0;

    if (!firstSeen)
    {
        if (sectionAddresses[currSectionIdx] == currSectionAddr)
        {
            // Allowed
            if (LOG_LEVEL_AT_LEAST_TRACE(SYN_RECIPE))
            {
                LOG_TRACE(SYN_RECIPE,
                          "Patching-info-ID was already set for section {} patching address 0x{:x}",
                          currSectionIdx,
                          currSectionAddr);
            }
        }
        else
        {
            LOG_ERR(SYN_RECIPE,
                    "Patching-info-ID was already set for section {} with address 0x{:x}, expected 0x{:x}",
                    currSectionIdx,
                    currSectionAddr,
                    sectionAddresses[currSectionIdx]);
            return false;
        }
    }
    else
    {
        sectionAddresses[currSectionIdx] = currSectionAddr;
    }

    return true;
}

bool RecipePatchProcessor::_insertSectionSize(SortedSectionsAddressInfo&    sortedSections,
                                              uint64_t                      currSectionIdx,
                                              uint64_t                      tensorInfoIndex,
                                              uint64_t                      currSectionAddr,
                                              const SectionSizeInfoVec&     sectionsSizeInfo,
                                              const synLaunchTensorInfoExt& launchTensorInfo,
                                              const synTensorType&          launchTensorType,
                                              const persist_tensor_info_t*  recipePersistentTensors,
                                              uint64_t                      numOfPersistentTensors)
{
    LOG_DEBUG(SYN_RECIPE, "{}: tensorId {} tensorName {} currSectionIdx {} tensorInfoIndex {} currSectionAddr 0x{:x} launchTensorType {}",
              HLLOG_FUNC, launchTensorInfo.tensorId, launchTensorInfo.tensorName, currSectionIdx, tensorInfoIndex, currSectionAddr, launchTensorType);


    if (currSectionIdx >= sectionsSizeInfo.size())
    {
        LOG_ERR(SYN_RECIPE, "{}: sectionId {} is out of sectionsSizeInfo DB bound", HLLOG_FUNC, currSectionIdx);
        return false;
    }

    uint64_t sectionSize = sectionsSizeInfo[currSectionIdx].sectionSize;
    bool     isConstSection = sectionsSizeInfo[currSectionIdx].isConstSection;
    uint64_t lastTensorRecipeidx = sectionsSizeInfo[currSectionIdx].lastTensorRecipeidx;

    LOG_DEBUG(SYN_RECIPE, "{}: currSectionIdx {} tensorInfoIndex {} currSectionAddr 0x{:x}, sectionSize {}, isConstSection {} lastTensorRecipeidx {}",
            HLLOG_FUNC, currSectionIdx, tensorInfoIndex, currSectionAddr, sectionSize, isConstSection, lastTensorRecipeidx);

    if (sectionSize == 0)
    {

        if (isConstSection)
        {
            LOG_DEBUG(SYN_RECIPE,
                      "{}: sectionId {} has 0 size, DB size {}, isConstSection {}",
                      HLLOG_FUNC,
                      currSectionIdx,
                      sectionsSizeInfo.size(),
                      isConstSection);
            return true;
        }
        else
        {
            LOG_ERR(SYN_RECIPE,
                    "{}: sectionId {} not found in sectionDB. DB size {}, isConstSection {}",
                    HLLOG_FUNC,
                    currSectionIdx,
                    sectionsSizeInfo.size(),
                    isConstSection);
            HB_ASSERT(0, "invalid section id in db");
            return false;
        }
    }

    if (!isConstSection && launchTensorType == DATA_TENSOR_DYNAMIC)  // might need to adjust the actual size
    {
        if (lastTensorRecipeidx >= numOfPersistentTensors)
        {
            LOG_ERR(SYN_RECIPE,
                    "{}: Invalid last tensor-index {} (limit {}) for sectionId {}",
                    HLLOG_FUNC,
                    lastTensorRecipeidx,
                    numOfPersistentTensors,
                    currSectionIdx);

            return false;
        }

        if (lastTensorRecipeidx == tensorInfoIndex)  // if this tensor is the last in section, need to fix the section size
        {
            uint64_t actualTensorSize = getActualTensorSize<TSize>(recipePersistentTensors[lastTensorRecipeidx].dimensions,
                                                                   launchTensorInfo.tensorSize,
                                                                   recipePersistentTensors[lastTensorRecipeidx].elementType);

            sectionSize = recipePersistentTensors[lastTensorRecipeidx].offset_in_section + actualTensorSize;
            LOG_DEBUG(SYN_RECIPE, "{}: update section size for dynamic tensor: tensorId {}  actualTensorSize {} sectionSize {}",
                                 HLLOG_FUNC, launchTensorInfo.tensorId, actualTensorSize, sectionSize);
        }
    }

    ValidateSectionInfo newInsert = {currSectionIdx, sectionSize, isConstSection};

    auto ret = sortedSections.insert(std::pair<uint64_t, ValidateSectionInfo>(currSectionAddr, newInsert));

    // If the last tensor in the section is dynamic, we need to adjust the section size. However, other tensors
    // in that section already set the section size to its maximum. So after inserting, if there is already an
    // entry for this section, we set the size to the minimum between what we already have and the new tensor
    if (!ret.second)  // already in, verify same sectionId
    {
        ValidateSectionInfo& existing = ret.first->second;

        LOG_DEBUG(SYN_RECIPE, "{} update exsiting section, existingSectionSize {} newSectionSize {}",
            HLLOG_FUNC, existing.sectionSize, sectionSize);


        // we update for the minimum between the current one and the new one
        existing.sectionSize = std::min(existing.sectionSize, sectionSize);

        if (existing.sectionId != currSectionIdx)
        {
            LOG_ERR(SYN_RECIPE,
                    "{}: overlap; trying to set two sections ({} and {}) to start on the same address (isConst {}, "
                    "isConst {})",
                    HLLOG_FUNC,
                    currSectionIdx,
                    existing.sectionId,
                    isConstSection,
                    existing.isConstSection);
            return false;
        }
    }

    return true;
}

bool RecipePatchProcessor::_markConstZeroSizeTensors(const std::unordered_set<uint64_t>& constZeroSizeTensors,
                                                     TensorsValidationInfoDB*            tensorGiven,
                                                     uint64_t*                           tensorCnt)
{
    // user doesn't have to supply zero sized const tensorsInfo
    for (auto tensorInfoIndex : constZeroSizeTensors)
    {
        tensor_info_t::ETensorType tensorRecipeType = userTypeToRecipeType(synTensorType::DATA_TENSOR);
        if (tensorGiven[tensorRecipeType][tensorInfoIndex] == true)  // Already got this tensor
        {
            LOG_DEBUG_T(SYN_RECIPE, "const Tensor already handled: tensorIndex {}", tensorInfoIndex);
        }
        else
        {
            tensorGiven[tensorRecipeType][tensorInfoIndex] = true;
            tensorCnt[tensorRecipeType]++;
        }
    }
    return true;
}

bool RecipePatchProcessor::_validateAllTensorsGiven(uint16_t                       tensorTypeIndex,
                                                    uint64_t                       tensorCnt,
                                                    uint64_t                       expectedNumTensors,
                                                    const TensorsValidationInfoDB& tensorValidationInfo,
                                                    const persist_tensor_info_t*   persistentTensorsDB,
                                                    const shape_tensor_info_t*     shapeTensorsDB,
                                                    uint64_t                       numOfPersistentTensors,
                                                    uint64_t                       numOfShapeTensors)
{
    if (tensorCnt != expectedNumTensors)
    {  // Some tensors are missing
        bool        isPersistentTensor = (tensorTypeIndex == 0);
        const char* tensorsTypeName    = isPersistentTensor ? "persistent" : "shape";

        LOG_ERR_T(SYN_RECIPE,
                  "Expected {} tensors, only {} given for {} tensors-type",
                  expectedNumTensors,
                  tensorCnt,
                  tensorsTypeName);

        uint64_t numOfTensorsOnDB = isPersistentTensor ? numOfPersistentTensors : numOfShapeTensors;
        if (expectedNumTensors > numOfTensorsOnDB)
        {
            LOG_ERR_T(SYN_RECIPE,
                      "Expected num of tensors in {} exceeds DB size {} for {} tensors-type",
                      expectedNumTensors,
                      numOfTensorsOnDB,
                      tensorsTypeName);

            return false;
        }

        for (int loop = 0; loop < expectedNumTensors; loop++)
        {
            if (tensorValidationInfo[loop] == false)
            {
                const char* name = isPersistentTensor ? persistentTensorsDB[loop].name : shapeTensorsDB[loop].name;

                LOG_ERR_T(SYN_RECIPE, "Tensor {} missing for {} tensors-type", name, tensorsTypeName);
            }
        }
        return false;
    }

    return true;
}

bool RecipePatchProcessor::_checkSectionsOverlap(const SortedSectionsAddressInfo& sortedSections)
{
    auto last = --sortedSections.end();
    for (auto it = sortedSections.begin(); it != last;)
    {
        auto     nextEntry  = std::next(it);
        uint64_t endSection = it->first + it->second.sectionSize;
        if (nextEntry->first < endSection)
        {
            LOG_ERR(SYN_RECIPE,
                    "{}: overlap; trying to set 2 sections with overlap: sections {} and {},"
                    " section-{}: start 0x{:x} end 0x{:x}, section-{}: start 0x{:x} end 0x{:x}",
                    HLLOG_FUNC,
                    it->second.sectionId,
                    nextEntry->second.sectionId,
                    it->second.sectionId,
                    it->first,
                    it->first + it->second.sectionSize,
                    nextEntry->second.sectionId,
                    nextEntry->first,
                    nextEntry->first + nextEntry->second.sectionSize);
            return false;
        }
        it = nextEntry;
    }

    return true;
}

void RecipePatchProcessor::_logLaunchTensorsInfo(const SortedSectionsAddressInfo& sortedSections,
                                                 const synLaunchTensorInfoExt*    launchTensorsInfo,
                                                 const persist_tensor_info_t*     persistentTensorsDB,
                                                 uint64_t                         numOfPersistentTensors,
                                                 uint32_t                         launchTensorsAmount,
                                                 uint64_t                         maxSectionId)
{
    for (uint32_t loop = 0; loop < launchTensorsAmount; loop++)
    {
        const synLaunchTensorInfoExt& currTensorLaunchInfo = launchTensorsInfo[loop];

        uint64_t launchTensorId = currTensorLaunchInfo.tensorId;
        if (IS_TENSOR_INVALID(launchTensorId))
        {
            // Skip on invalid tensors, they can be delivered by launch
            continue;
        }

        synTensorType              tensorInfoType   = TENSOR_INFO_TO_TYPE(launchTensorId);
        tensor_info_t::ETensorType tensorRecipeType = userTypeToRecipeType(tensorInfoType);

        if (tensorRecipeType != tensor_info_t::PERSISTENT_TENSOR)
        {
            // Skip non-PTs
            continue;
        }

        const char* tensorName = (currTensorLaunchInfo.tensorName != nullptr) ? currTensorLaunchInfo.tensorName
                                                                              : "(tensor-name was not supplied)";

        uint64_t tensorInfoIndex = TENSOR_INFO_TO_INDEX(launchTensorId);
        HB_ASSERT(tensorInfoIndex < numOfPersistentTensors,
                  "tensorInfoIndex {} is out of PT DB bound (size {})",
                  tensorInfoIndex,
                  numOfPersistentTensors);

        uint64_t actualTensorSize = getActualTensorSize<TSize>(persistentTensorsDB[tensorInfoIndex].dimensions,
                                                               currTensorLaunchInfo.tensorSize,
                                                               persistentTensorsDB[tensorInfoIndex].elementType);

        uint64_t currSectionIdx = persistentTensorsDB[tensorInfoIndex].section_idx;
        HB_ASSERT(currSectionIdx <= maxSectionId,
                  "sectionId {} is larger than masSectionID {}",
                  currSectionIdx,
                  maxSectionId);

        LOG_ERR(SYN_RECIPE,
                "Tensor {} {} (address 0x{:x} size {}) at sectionIdx {} (deducted section-address 0x{:x})",
                tensorInfoIndex,
                tensorName,
                currTensorLaunchInfo.pTensorAddress,
                actualTensorSize,
                currSectionIdx,
                currTensorLaunchInfo.pTensorAddress - persistentTensorsDB[tensorInfoIndex].offset_in_section);
    }

    uint64_t sectionIndex = 0;
    for (auto sectionInfo : sortedSections)
    {
        LOG_ERR(SYN_RECIPE,
                "Section (Index {} ID {}) address 0x{:x} size 0x{:x}",
                sectionIndex,
                sectionInfo.second.sectionId,
                sectionInfo.first,
                sectionInfo.second.sectionSize);

        sectionIndex++;
    }
}

bool RecipePatchProcessor::_isOutOfRange(uint64_t                     baseAddr,
                                         uint64_t                     size,
                                         const ValidSectionAddresses* pValidSectionAddr)
{
    if (size == 0 || !pValidSectionAddr)
    {
        return false;
    }
    uint64_t lastAddr = baseAddr + size - 1;

    bool outOfRange = ((baseAddr < pValidSectionAddr->lowestValidAddress) || (baseAddr > pValidSectionAddr->highestValidAddress) ||
            (lastAddr < pValidSectionAddr->lowestValidAddress) || (lastAddr > pValidSectionAddr->highestValidAddress));
    if (outOfRange)
    {
        LOG_ERR(SYN_RECIPE,
                "{}: section base 0x{:x} size 0x{:x} is out of range [0x{:x} - 0x{:x}]",
                HLLOG_FUNC,
                baseAddr,
                size,
                pValidSectionAddr->lowestValidAddress,
                pValidSectionAddr->highestValidAddress);
    }
    return outOfRange;
}

void RecipePatchProcessor::_retrieveTensorInfo(tensor_info_t::ETensorType& tensorRecipeType,
                                               synTensorType&              tensorType,
                                               uint64_t&                   tensorIndex,
                                               uint64_t                    tensorId)
{
    tensorType       = TENSOR_INFO_TO_TYPE(tensorId);
    tensorIndex      = TENSOR_INFO_TO_INDEX(tensorId);
    tensorRecipeType = userTypeToRecipeType(tensorType);
}

bool RecipePatchProcessor::_resolveSingleTensorIdToTensorIndex(tensor_info_t::ETensorType& tensorRecipeType,
                                                               uint64_t&                   tensorIndex,
                                                               uint64_t                    numberShapeTensors,
                                                               uint64_t                    numberPersistTensors)
{
    if (tensorRecipeType == tensor_info_t::SHAPE_TENSOR)
    {
        if (tensorIndex >= numberShapeTensors)
        {
            LOG_ERR(SYN_RECIPE,
                    "{}: invalid shape tensorIndex {} max allowed {}",
                    HLLOG_FUNC,
                    tensorIndex,
                    numberShapeTensors);
            return false;
        }
    }
    else if (tensorRecipeType == tensor_info_t::PERSISTENT_TENSOR)
    {
        if (tensorIndex >= numberPersistTensors)
        {
            LOG_ERR(SYN_RECIPE,
                    "{}: invalid persist tensorIndex {} max allowed {}",
                    HLLOG_FUNC,
                    tensorIndex,
                    numberPersistTensors);
            return false;
        }
    }
    else
    {
        HB_ASSERT(0, "Should never get here, userTypeToRecipeType returns only PERSISTENT/SHAPE");
    }

    return true;
}

synStatus RecipePatchProcessor::_resolveSingleTensorSectionsAddresses(
    const basicRecipeInfo&                    rBasicRecipeInfo,
    const synLaunchTensorInfoExt*             pLaunchTensorsInfo,
    patching::HostAddressPatchingInformation& hostAddressPatchingInfo,
    const SectionToSectionType*               sectionToSectionType,
    const RecipeTensorsInfo&                  rRecipeTensorInfo,
    DevMemoryAllocInterface&                  rDevMemAlloc,
    uint64_t                                  tensorIndex)
{
    HB_ASSERT_DEBUG_ONLY((pLaunchTensorsInfo != nullptr), "Nullptr tensor-info");

    uint64_t currSectionIdx        = rBasicRecipeInfo.recipe->tensors[tensorIndex].section_idx;
    uint64_t tensorOffsetInSection = rBasicRecipeInfo.recipe->tensors[tensorIndex].offset_in_section;
    uint64_t tensorAddress         = pLaunchTensorsInfo->pTensorAddress;

    if (tensorOffsetInSection > tensorAddress)
    {
        if (tensorAddress == 0)
        {
            // we can't deduce the section address from this tensorInfo,
            LOG_WARN(SYN_RECIPE,
                     "{}: tensorIndex {} (sectionIndex {}) has offsetInSection {} when address is {}",
                     HLLOG_FUNC,
                     tensorIndex,
                     currSectionIdx,
                     tensorOffsetInSection,
                     tensorAddress);
            return synSuccess;
        }
        else
        {
            // otherwise currSectionAddr will become negative
            LOG_ERR(SYN_RECIPE,
                    "{}: Failed, tensorIndex {} (sectionIndex {}) has OffsetInSection {} is bigger than pTensorAddress {}",
                    HLLOG_FUNC,
                    tensorIndex,
                    currSectionIdx,
                    tensorOffsetInSection,
                    tensorAddress);
            hostAddressPatchingInfo.patchingAbort();
            return synFail;
        }
    }

    uint64_t currSectionAddr = tensorAddress - tensorOffsetInSection;

    // we have the tensor OS address, we want to patch with the mmu address
    if (pLaunchTensorsInfo->tensorType == HOST_TO_DEVICE_TENSOR)
    {
        uint64_t deviceSectionAddr = 0;
        if (!getMmuVirtualAddress(rDevMemAlloc, currSectionAddr, &deviceSectionAddr))
        {
            LOG_ERR(SYN_PATCH_INFO, "failed to get VA for HOST_TO_DEVICE_TENSOR idx {} currSectionAddr {:x}",
                    tensorIndex, currSectionAddr);
            return synMappingNotFound;
        }
        LOG_DEBUG(SYN_PATCH_INFO,
                  "{}: Match section virtual address 0x{:x} to host address index: 0x{:x} for tensorIndex {}",
                  HLLOG_FUNC,
                  deviceSectionAddr,
                  currSectionAddr,
                  tensorIndex);
        currSectionAddr = deviceSectionAddr;
    }

    if (LOG_LEVEL_AT_LEAST_DEBUG(SYN_RECIPE))
    {
        const char* tensorName = pLaunchTensorsInfo->tensorName;
        LOG_DEBUG(
            SYN_PATCH_INFO,
            "{}: Match section address 0x{:x} to section index: {}, tensorIndex {} tensor_name {} with offset 0x{:x}",
            HLLOG_FUNC,
            currSectionAddr,
            currSectionIdx,
            tensorIndex,
            (tensorName != nullptr) ? tensorName : "",
            tensorOffsetInSection);
    }

    uint64_t   sectionSize   = rRecipeTensorInfo.m_sectionsInfo[currSectionIdx].sectionSize;
    uint64_t   sectionTypeId = (sectionToSectionType == nullptr) ? 0 : (*sectionToSectionType)[currSectionIdx];
    const bool isDsd         = rBasicRecipeInfo.shape_plan_recipe != nullptr;
    bool res = hostAddressPatchingInfo.setSectionHostAddress(currSectionIdx,
                                                             sectionTypeId,
                                                             currSectionAddr,
                                                             (sectionSize == 0),
                                                             isDsd);
    if (!res)
    {
        LOG_ERR(SYN_RECIPE,
                "{}: Failed to setSectionHostAddress for section {} with address {:#x}",
                HLLOG_FUNC,
                currSectionIdx,
                currSectionAddr);
        hostAddressPatchingInfo.patchingAbort();

        return synFail;
    }

    return synSuccess;
}
