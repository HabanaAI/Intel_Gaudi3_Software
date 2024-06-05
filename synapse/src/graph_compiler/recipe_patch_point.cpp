#include <string>

#include "recipe_patch_point.h"

#include "habana_global_conf.h"
#include "habana_graph.h"
#include "log_manager.h"
#include "node.h"
#include "utils.h"

#include "recipe_allocator.h"

#include "define_synapse_common.hpp"

#include "infra/defs.h"

//-----------------------------------------------------------------------------
//                               RecipePatchPoint
//-----------------------------------------------------------------------------

RecipePatchPoint::RecipePatchPoint(uint64_t offsetInBlob, uint32_t nodeExecIndex = 0)
  : m_blobIndex(0),
    m_offsetInBlob(offsetInBlob),
    m_nodeExecIndex(nodeExecIndex)
{
}

RecipePatchPoint::~RecipePatchPoint()
{
}


//-----------------------------------------------------------------------------
//                                MemoryPatchPoint
//-----------------------------------------------------------------------------

MemoryPatchPoint::MemoryPatchPoint(FieldType addrPartType,
                                   uint64_t  virtualAddr,
                                   uint64_t  sectionIndex,
                                   uint64_t  offsetInBlob,
                                   uint32_t  nodeExecIndex)
: RecipePatchPoint(offsetInBlob, nodeExecIndex), m_virtualAddr(virtualAddr), m_sectionIndex(sectionIndex)
{

    switch (addrPartType)
    {
    case FIELD_ADDRESS_PART_FULL:
        setType(patch_point_t::SIMPLE_DDW_MEM_PATCH_POINT);
        break;
    case FIELD_ADDRESS_PART_LOW:
        setType(patch_point_t::SIMPLE_DW_LOW_MEM_PATCH_POINT);
        break;
    case FIELD_ADDRESS_PART_HIGH:
        setType(patch_point_t::SIMPLE_DW_HIGH_MEM_PATCH_POINT);
        break;
    default:
        HB_ASSERT(0, "unhandled case");
    }
}

MemoryPatchPoint::~MemoryPatchPoint()
{
}

void MemoryPatchPoint::serialize(patch_point_t* pPatchPoints) const
{
    pPatchPoints->type                                  = m_type;
    pPatchPoints->blob_idx                              = m_blobIndex;
    pPatchPoints->dw_offset_in_blob                     = m_offsetInBlob;
    pPatchPoints->memory_patch_point.effective_address  = m_virtualAddr;
    pPatchPoints->memory_patch_point.section_idx        = m_sectionIndex;
    pPatchPoints->node_exe_index                        = m_nodeExecIndex;
}

void MemoryPatchPoint::print() const
{
    if (!LOG_LEVEL_AT_LEAST_DEBUG(RECIPE_GEN)) return;

    LOG_DEBUG(RECIPE_GEN, "      type = {}", m_type);
    LOG_DEBUG(RECIPE_GEN, "      blob index = {}", m_blobIndex);
    LOG_DEBUG(RECIPE_GEN, "      offset in blob = {}", m_offsetInBlob);
    LOG_DEBUG(RECIPE_GEN, "      effective (virtual) address = 0x{:x}", m_virtualAddr);
    LOG_DEBUG(RECIPE_GEN, "      section index = {}", m_sectionIndex);
}



//-----------------------------------------------------------------------------
//                                DynamicPatchPoint
//-----------------------------------------------------------------------------

DynamicPatchPoint::DynamicPatchPoint(EFieldType           type,
                                     uint32_t             offsetInBlob,
                                     std::vector<uint8_t> metadata,
                                     const NodeROI*       roi,
                                     ShapeFuncID          functionId,
                                     uint64_t             patchSizeInDw,
                                     uint32_t             highPatchPointIndex,
                                     uint32_t             lowPatchPointIndex,
                                     bool                 isUnskippable,
                                     unsigned             blobId)
: RecipePatchPoint(offsetInBlob),
  m_smppType(type),
  m_isUnskippable(isUnskippable),
  m_metadata(std::move(metadata)),
  m_roi(roi),
  m_roiIndex(0),
  m_functionId(functionId),
  m_patchSizeInDw(patchSizeInDw),
  m_highPatchPointIndex(highPatchPointIndex),
  m_lowPatchPointIndex(lowPatchPointIndex),
  m_blobId(blobId)  // associate the dynamic patch point with the blob unique identifier
{
}

void DynamicPatchPoint::serialize(sm_patch_point_t* pPatchPoint, RecipeAllocator* pRecipeAlloc) const
{
    pPatchPoint->patch_point_type     = m_smppType;
    if (m_smppType == EFieldType::FIELD_DYNAMIC_ADDRESS)
    {
        pPatchPoint->patch_point_idx_low  = m_lowPatchPointIndex;
        pPatchPoint->patch_point_idx_high = m_highPatchPointIndex;
    }
    else
    {
        pPatchPoint->blob_idx             = m_blobIndex;
        pPatchPoint->dw_offset_in_blob    = m_offsetInBlob;
    }
    pPatchPoint->patch_size_dw        = m_patchSizeInDw;
    pPatchPoint->roi_idx              = m_roiIndex;
    pPatchPoint->smf_id.sm_funcid     = m_functionId;
    pPatchPoint->smf_id.sm_tableid    = LIB_ID_RESERVED_FOR_GC_SMF;
    pPatchPoint->is_unskippable       = m_isUnskippable;

    // serialize metadata
    // round up the size
    auto sizeInElements               = (m_metadata.size() + sizeof(uint64_t) - 1)/sizeof(uint64_t);
    pPatchPoint->metadata             = (uint64_t*)pRecipeAlloc->allocate(sizeInElements * sizeof(uint64_t));
    pPatchPoint->metadata_size        = sizeInElements;
    // copy the data
    std::copy(m_metadata.begin(), m_metadata.end(), reinterpret_cast<uint8_t*>(pPatchPoint->metadata));
}

void DynamicPatchPoint::print() const
{
    if (!LOG_LEVEL_AT_LEAST_DEBUG(GC)) return;

    LOG_DEBUG(GC, "      patch_point_type = {}", m_smppType);
    LOG_DEBUG(GC, "      smf id = {}", m_functionId);

    sm_function_id_t fid;
    fid.sm_funcid  = m_functionId;
    fid.sm_tableid = LIB_ID_RESERVED_FOR_GC_SMF;
    LOG_DEBUG(GC, "      smf name = {}", ShapeFuncRegistry::instance().getSmfName(fid));

    LOG_DEBUG(GC, "      blob index = {}", m_blobIndex);
    LOG_DEBUG(GC, "      offset in blob = {}", m_offsetInBlob);
    LOG_DEBUG(GC, "      metadata = {}", toString(m_metadata.begin(), m_metadata.end(), ' ', [](char s) {
                  return isprint(s) ? std::string {s} : fmt::format("\\x{:02x}", (int)s);
              }));
    LOG_DEBUG(GC, "      roiIndex = {}", m_roiIndex);
    LOG_DEBUG(GC, "      patch_point_idx_low = {}", m_lowPatchPointIndex);
    LOG_DEBUG(GC, "      patch_point_idx_high = {}", m_highPatchPointIndex);
}


//-----------------------------------------------------------------------------
//                            RecipePatchPointContainer
//-----------------------------------------------------------------------------

RecipePatchPointContainer::~RecipePatchPointContainer()
{
    for (RecipePatchPoint* pPatchPoint : m_patchPoints)
    {
        delete pPatchPoint;
    }
}

RecipePatchPoint& RecipePatchPointContainer::operator[](int index)
{
    return *m_patchPoints[index];
}

size_t RecipePatchPointContainer::addMemoryPatchPoint(AddressFieldInfo& fieldInfo, uint64_t cmdOffsetWithinBlobInBytes)
{
    m_patchPoints.push_back(
        new MemoryPatchPoint(fieldInfo.getAddressPart(),
                             fieldInfo.getTargetAddress(),
                             fieldInfo.getMemorySectionId(),
                             fieldInfo.getFieldIndexOffset() + cmdOffsetWithinBlobInBytes / sizeof(uint32_t),
                             fieldInfo.getNodeExecutionIndex()));

    m_numOfMemoryPatchPoints++;
    if (fieldInfo.getMemorySectionId() != MEMORY_ID_RESERVED_FOR_PROGRAM_DATA)
    {
        fieldInfo.setIndex(m_numOfMemoryPatchPoints - 1);
    }
    return m_patchPoints.size() - 1;
}

void RecipePatchPointContainer::addDynamicPatchPoint(const BasicFieldInfo& fieldInfo,
                                                     uint64_t              cmdOffsetWithinBlobInBytes,
                                                     unsigned              blobId)
{
    if (!fieldInfo.isDynamicShape())
    {
        return;
    }
    auto& dynamicFieldInfo = reinterpret_cast<const DynamicShapeFieldInfo&>(fieldInfo);

    auto metadata = dynamicFieldInfo.getMetadata();

    uint64_t lowIndex = -1;
    uint64_t highIndex = -1;
    bool     ppNeedUpdate = false;

    auto* asAddressPatchPoint = dynamic_cast<const DynamicAddressFieldInfo*>(&dynamicFieldInfo);
    if (asAddressPatchPoint != nullptr)
    {
        if (asAddressPatchPoint->m_addressFieldHigh != nullptr)
        {
            highIndex = asAddressPatchPoint->m_addressFieldHigh->getIndex();
            ppNeedUpdate = true;
        }
        if (asAddressPatchPoint->m_addressFieldLow != nullptr)
        {
            lowIndex = asAddressPatchPoint->m_addressFieldLow->getIndex();
            ppNeedUpdate = true;
        }
    }

    auto dynamicPatchPoint = std::make_shared<DynamicPatchPoint>(dynamicFieldInfo.getType(),
                                                                 dynamicFieldInfo.getFieldIndexOffset() +
                                                                     cmdOffsetWithinBlobInBytes / sizeof(uint32_t),
                                                                 metadata,
                                                                 dynamicFieldInfo.getRoi(),
                                                                 dynamicFieldInfo.getSmfID(),
                                                                 dynamicFieldInfo.getSize(),
                                                                 highIndex,
                                                                 lowIndex,
                                                                 dynamicFieldInfo.isUnskippable(),
                                                                 blobId);

    if (ppNeedUpdate && GCFG_ENABLE_STAGED_SUBMISSION.value())
    {
        m_dynamicPatchPointsToUpdate.push_back(dynamicPatchPoint);
    }

    m_dynamicPatchPoints.push_back(dynamicPatchPoint);
    dynamicFieldInfo.getOrigin()->getShapeNode()->addPatchPoint(std::move(dynamicPatchPoint));
}

MemoryPatchPoint* RecipePatchPointContainer::getNeighboringLowerCounterPart(const AddressFieldInfo& highMemoryFieldInfo, uint64_t cmdOffsetWithinBlobInBytes, const std::unordered_map<uint64_t, uint64_t>& ppIndicesMap)
{
    auto ppIndexItr = ppIndicesMap.find(highMemoryFieldInfo.getEngineFieldId());
    if (ppIndexItr == ppIndicesMap.end())
    {
        return nullptr;
    }

    uint64_t highFieldIndexOffset = highMemoryFieldInfo.getFieldIndexOffset() + cmdOffsetWithinBlobInBytes / sizeof(uint32_t);

    uint64_t ppIndex = ppIndexItr->second;
    HB_ASSERT(ppIndex < m_patchPoints.size(), "Patch-Point index out of bound");
    auto lowCounterpart = m_patchPoints[ppIndex];

    if (lowCounterpart->getOffsetInBlob() + 1 != highFieldIndexOffset) // are LOW and HIGH neighbors in the blob?
    {
        return nullptr;
    }
    // Do not insert a new patch point, just update the LOW part to be FULL
    return dynamic_cast<MemoryPatchPoint*>(lowCounterpart);
}

std::list<uint64_t> RecipePatchPointContainer::insertPatchPoints(const BasicFieldsContainerInfo& bfci,
                                                                 uint64_t cmdOffsetWithinBlobInBytes,
                                                                 unsigned blobId)
{
    // Extract all patch-point from the afci and convert them to recipe memory patch points and return their indices.
    // In the afci there can be either FULL, LOW, or HIGH patch points indicating 64-bit patch, 32-bit patch of low
    // part and 32-bit patch of high part respectively. We will try to optimize the LOW and HIGH part such that if
    // they located adjacently in the blob, we will joint them to a single FULL patch point.

    std::unordered_map<uint64_t, uint64_t> ppIndices;
    std::vector<AddressFieldInfoSharedPtr> highPartPP;
    std::list<uint64_t>                    ret;  // returns the indices of all the newly created recipe patch points

    // Go through all patch points in the afci and make a recipe patch point for the FULL and LOW ones, keep aside the HIGH ones
    for (auto& addressInfoPair : bfci.retrieveAddressFieldInfoSet())
    {
        auto& pAddressInfo = addressInfoPair.second;

        if (pAddressInfo->getAddressPart() == FIELD_ADDRESS_PART_HIGH)
        {
            highPartPP.push_back(pAddressInfo);
        }
        else // FULL or LOW
        {
            auto newIndex = addMemoryPatchPoint(*pAddressInfo, cmdOffsetWithinBlobInBytes);
            ret.push_back(newIndex);
            ppIndices[pAddressInfo->getEngineFieldId()] = newIndex;
        }
    }

    // Go through all the HIGH ones and check if they can be joined with their LOW counterparts to form a FULL patch point
    for (auto& pAddressInfo : highPartPP)
    {
        auto neighboringLowCounterpart = getNeighboringLowerCounterPart(*pAddressInfo, cmdOffsetWithinBlobInBytes, ppIndices);

        if (neighboringLowCounterpart) // are LOW and HIGH neighbors in the blob?
        {
            // Do not insert a new patch point, just update the LOW part to be FULL
            neighboringLowCounterpart->setType(patch_point_t::SIMPLE_DDW_MEM_PATCH_POINT);
            LOG_TRACE(RECIPE_GEN,
                      "{}: Optimizing patch-points, joining LOW and HIGH to a single FULL at index {}",
                      HLLOG_FUNC,
                      ppIndices[pAddressInfo->getEngineFieldId()]);
        }
        else // LOW and HIGH don't sit adjacently in the blob, so add patch point to the HIGH
        {
            ret.push_back(addMemoryPatchPoint(*pAddressInfo, cmdOffsetWithinBlobInBytes));
        }
    }

    for (auto& basicFieldInfoPair : bfci.retrieveBasicFieldInfoSet())
    {
        if (basicFieldInfoPair.second->isDynamicShape())
        {
            addDynamicPatchPoint(*basicFieldInfoPair.second, cmdOffsetWithinBlobInBytes, blobId);
        }
        else
        {
            ret.push_back(addSyncObjectPatchPoint(*basicFieldInfoPair.second, cmdOffsetWithinBlobInBytes));
        }
    }

    return ret;
}

void RecipePatchPointContainer::setPatchPointsBlobIndex(uint64_t patchingBlobIndex, unsigned blobId)
{
    auto it = m_dynamicPatchPoints.begin();
    while (it != m_dynamicPatchPoints.end())
    {
        if ((*it)->getBlobId() == blobId)
        {
            // set blob index and remove element only if the patch point is associated with the blobId
            (*it)->setBlobIndex(patchingBlobIndex);
            it = m_dynamicPatchPoints.erase(it);
        }
        else
        {
            ++it;
        }
    }
}

void RecipePatchPointContainer::erasePatchPointByIndex(unsigned idx)
{
    HB_ASSERT(idx < m_patchPoints.size(), "Patch-point index out of bound");

    if (dynamic_cast<MemoryPatchPoint*>(m_patchPoints[idx]))
    {
        m_numOfMemoryPatchPoints--;
    }

    delete m_patchPoints[idx];
    m_patchPoints.erase(m_patchPoints.begin() + idx);
}

uint8_t RecipePatchPointContainer::getSectionTypeForSectionId(const std::map<uint64_t, uint8_t>& sectionIdToSectionType,
                                                              uint64_t                           sectionId) const
{
    if (sectionIdToSectionType.find(sectionId) != sectionIdToSectionType.end())
    {
        return sectionIdToSectionType.at(sectionId);
    }

    if (sectionId == MEMORY_ID_RESERVED_FOR_PROGRAM_DATA)
    {
        return PROGRAM_DATA_SECTION_TYPE;
    }

    return DEFAULT_SECTION_TYPE;
}

void RecipePatchPointContainer::getPatchPointsByNodeId(
    std::map<uint32_t, std::vector<RecipePatchPoint*>>& nodeToPatchPointList,
    std::map<RecipePatchPoint*, uint64_t>&              patchPointCurrentPos) const
{
    uint32_t nodeExecIndex;
    uint64_t index = 0;

    for (RecipePatchPoint* pPatchPoint : m_patchPoints)
    {
        // save the old position of the patch point before changing it
        patchPointCurrentPos[pPatchPoint] = index;
        nodeExecIndex                     = pPatchPoint->getNodeExecIndex();
        nodeToPatchPointList[nodeExecIndex].push_back(pPatchPoint);
        index++;
    }
}

void RecipePatchPointContainer::calcAccumulatedPatchPointsPerNode(
    const std::map<uint32_t, std::vector<RecipePatchPoint*>>& nodeToPatchPointList) const
{
    uint64_t currentTotal = 0;

    // For each node execution index - accumulate the number of patch points up until and including the current node.
    // We will later need to serialize it in node_exe_list
    for (auto it = nodeToPatchPointList.begin(); it != nodeToPatchPointList.end(); it++)
    {
        currentTotal += it->second.size();
        m_nodeToPatchPointsCount[it->first] = currentTotal;
    }

}

void RecipePatchPointContainer::updateDynamicPatchPoints(
    std::map<uint64_t, uint64_t>& patchPointNewPosToCurrentPos) const
{
    LOG_DEBUG(GC, "Number of dynamic patch point to update: {}", m_dynamicPatchPointsToUpdate.size());

    uint64_t currentPatchPointIndex;

    for (std::shared_ptr<DynamicPatchPoint> pPatchPoint : m_dynamicPatchPointsToUpdate)
    {
        currentPatchPointIndex = pPatchPoint->getHighPatchPointIndex();
        if (patchPointNewPosToCurrentPos.find(currentPatchPointIndex) != patchPointNewPosToCurrentPos.end())
        {
            pPatchPoint->setHighPatchPointIndex(patchPointNewPosToCurrentPos[currentPatchPointIndex]);
            LOG_DEBUG(GC,
                      "Dynamic patch point high index was modified. Old: {}, New: {}",
                      currentPatchPointIndex,
                      patchPointNewPosToCurrentPos[currentPatchPointIndex]);
        }
        currentPatchPointIndex = pPatchPoint->getLowPatchPointIndex();
        if (patchPointNewPosToCurrentPos.find(currentPatchPointIndex) != patchPointNewPosToCurrentPos.end())
        {
            pPatchPoint->setLowPatchPointIndex(patchPointNewPosToCurrentPos[currentPatchPointIndex]);
            LOG_DEBUG(GC,
                      "Dynamic patch point low index was modified. Old: {}, New: {}",
                      currentPatchPointIndex,
                      patchPointNewPosToCurrentPos[currentPatchPointIndex]);
        }
    }
}

void RecipePatchPointContainer::serialize(uint32_t*                          pNumPatchPoints,
                                          uint32_t*                          pNumActivatePatchPoints,
                                          patch_point_t**                    ppPatchPoints,
                                          uint32_t*                          pNumSectionTypes,
                                          section_group_t**                  pSectionTypePatchPoints,
                                          const std::map<uint64_t, uint8_t>& sectionIdToSectionType,
                                          uint32_t*                          pNumOfSectionIds,
                                          section_blobs_t**                  pSectionIdBlobIndices,
                                          section_group_t*                   pSignalOutSection,
                                          RecipeAllocator*                   pRecipeAlloc,
                                          const TensorSet&                   persistTensors) const
{
    HB_ASSERT(pNumPatchPoints != nullptr && ppPatchPoints != nullptr, "got input null pointers");

    // include all patch points
    *pNumPatchPoints = getNumOfMemoryPatchPoints();
    *pNumActivatePatchPoints = getNumOfActivatePatchPoints();
    LOG_DEBUG(GC, "Total Patch Points: {}, Activation Patch Points: {}", *pNumPatchPoints, *pNumActivatePatchPoints);

    *ppPatchPoints         = (patch_point_t*)pRecipeAlloc->allocate(*pNumPatchPoints * sizeof(patch_point_t));
    patch_point_t* pFiller = *ppPatchPoints;

    std::map<uint8_t, std::vector<uint64_t>> sectionListOfPatchPoints;
    uint8_t                                  currentSectionType;
    uint64_t                                 index = 0;

    // section id to blobs indices map
    std::map<uint64_t, std::set<uint32_t> > sectionIdToBlobIndices;

    // In case we have sync object patch points, we prepare the tensors index map once
    // later to be used when serializing
    std::map<uint32_t, uint32_t> tIdToPresistentTIdx;
    if (m_bIsUsingSyncObjectPatchPoints)
    {
        uint32_t tIndex = 0;
        for (pTensor t : persistTensors)
        {
            tIdToPresistentTIdx[t->getId()] = tIndex++;
        }
    }

    if (GCFG_ENABLE_STAGED_SUBMISSION.value())
    {
        // First get patch points sorted by node execution order.
        std::map<uint32_t, std::vector<RecipePatchPoint*>> nodeToPatchPointList;
        std::map<RecipePatchPoint*, uint64_t>              patchPointCurrentPos;

        getPatchPointsByNodeId(nodeToPatchPointList, patchPointCurrentPos);

        calcAccumulatedPatchPointsPerNode(nodeToPatchPointList);

        std::map<uint64_t, uint64_t> patchPointNewPosToCurrentPos;

        for (auto it = nodeToPatchPointList.begin(); it != nodeToPatchPointList.end(); it++)
        {
            for (int j = 0; j < it->second.size(); j++)
            {
                auto memoryPatchPoint = dynamic_cast<MemoryPatchPoint*>(it->second[j]);
                if (memoryPatchPoint)
                {
                    sectionIdToBlobIndices[memoryPatchPoint->getSectionId()].insert(memoryPatchPoint->getBlobIndex());

                    currentSectionType =
                        getSectionTypeForSectionId(sectionIdToSectionType, memoryPatchPoint->getSectionId());
                    sectionListOfPatchPoints[currentSectionType].push_back(index);

                    memoryPatchPoint->serialize(pFiller);
                    pFiller++;
                }
                else
                {
                    auto* syncPatchPoint = dynamic_cast<SyncObjectPatchPoint*>(it->second[j]);
                    HB_ASSERT(syncPatchPoint, "Unexpected patchpoint type");
                    syncPatchPoint->serialize(pFiller, tIdToPresistentTIdx);
                    pFiller++;
                }
                // update patch point new position in the patch points list
                patchPointNewPosToCurrentPos[patchPointCurrentPos[memoryPatchPoint]] = index;

                index++;
            }
        }
        // update dynamic patch points - if needed
        if (m_dynamicPatchPointsToUpdate.size())
        {
            updateDynamicPatchPoints(patchPointNewPosToCurrentPos);
        }
    }
    else  // this is the old mode - using the unordered patch points list
    {
        for (RecipePatchPoint* pPatchPoint : m_patchPoints)
        {
            auto* memoryPatchPoint = dynamic_cast<MemoryPatchPoint*>(pPatchPoint);
            if (memoryPatchPoint)
            {
                sectionIdToBlobIndices[memoryPatchPoint->getSectionId()].insert(memoryPatchPoint->getBlobIndex());

                currentSectionType =
                    getSectionTypeForSectionId(sectionIdToSectionType, memoryPatchPoint->getSectionId());
                sectionListOfPatchPoints[currentSectionType].push_back(index);
                index++;

                memoryPatchPoint->serialize(pFiller);
                pFiller++;
            }
            else
            {
                auto* syncPatchPoint = dynamic_cast<SyncObjectPatchPoint*>(pPatchPoint);
                if (!syncPatchPoint) continue;
                syncPatchPoint->serialize(pFiller, tIdToPresistentTIdx);
                pFiller++;
            }
        }
    }

    pSignalOutSection->section_group   = -1;
    pSignalOutSection->patch_points_nr = 0;
    for (int i = 0; i < *pNumPatchPoints; i++)
    {
        if ((*ppPatchPoints)[i].type == patch_point_t::SOB_PATCH_POINT)
        {
            pSignalOutSection->patch_points_nr++;
        }
    }
    pSignalOutSection->patch_points_index_list =
        (uint32_t*)pRecipeAlloc->allocate(pSignalOutSection->patch_points_nr * sizeof(uint32_t));
    auto writerIterator = pSignalOutSection->patch_points_index_list;
    for (int i = 0; i < *pNumPatchPoints; i++)
    {
        if ((*ppPatchPoints)[i].type == patch_point_t::SOB_PATCH_POINT)
        {
            *writerIterator = i;
            writerIterator++;
        }
    }

    // handle section id to blob indices list
    *pNumOfSectionIds = sectionIdToBlobIndices.size();
    *pSectionIdBlobIndices = (section_blobs_t*)pRecipeAlloc->allocate(*pNumOfSectionIds * sizeof(section_blobs_t));
    uint64_t sectionIdx = 0;

    for (auto it = sectionIdToBlobIndices.begin(); it != sectionIdToBlobIndices.end(); it++)
    {
        (*pSectionIdBlobIndices)[sectionIdx].section_idx = it->first;

        uint64_t numBlobs = it->second.size();
        (*pSectionIdBlobIndices)[sectionIdx].blobs_nr = numBlobs;
        (*pSectionIdBlobIndices)[sectionIdx].blob_indices =
            (uint32_t*)pRecipeAlloc->allocate(numBlobs * sizeof(uint32_t));

        int i = 0;
        for (auto itt = it->second.begin(); itt != it->second.end(); itt++)
        {
            (*pSectionIdBlobIndices)[sectionIdx].blob_indices[i] = *itt;
            i++;
        }

        sectionIdx++;
    }

    *pNumSectionTypes = sectionListOfPatchPoints.size();
    *pSectionTypePatchPoints = (section_group_t*)pRecipeAlloc->allocate(*pNumSectionTypes * sizeof(section_group_t));
    sectionIdx = 0;

    for (auto iter = sectionListOfPatchPoints.begin(); iter != sectionListOfPatchPoints.end(); ++iter)
    {
        uint64_t numOfPPForSection                                     = iter->second.size();
        (*pSectionTypePatchPoints)[sectionIdx].patch_points_nr         = numOfPPForSection;
        (*pSectionTypePatchPoints)[sectionIdx].section_group           = iter->first;
        (*pSectionTypePatchPoints)[sectionIdx].patch_points_index_list =
            (uint32_t*)pRecipeAlloc->allocate(numOfPPForSection * sizeof(uint32_t));

        LOG_DEBUG(GC, "Section type: {}, Num of patch points: {}", iter->first, numOfPPForSection);
        for (int i = 0; i < numOfPPForSection; i++)
        {
            (*pSectionTypePatchPoints)[sectionIdx].patch_points_index_list[i] = (iter->second)[i];
        }
        sectionIdx++;
    }

    HB_ASSERT(pFiller - *ppPatchPoints == *pNumPatchPoints, "Incorrect number of address patch points");
}

void RecipePatchPointContainer::print() const
{
    if (!LOG_LEVEL_AT_LEAST_DEBUG(RECIPE_GEN) && !LOG_LEVEL_AT_LEAST_DEBUG(GC)) return;

    LOG_DEBUG(RECIPE_GEN, "  Patch-point Container Dump:");
    LOG_DEBUG(RECIPE_GEN,
              "    Number of patch-points = {} (first {} in activate program)",
              m_patchPoints.size(),
              getNumOfActivatePatchPoints());
    uint64_t i = 0;
    for (auto& patchPoint : m_patchPoints)
    {
        LOG_DEBUG(RECIPE_GEN, "    Patch-point {}:", i++);
        patchPoint->print();
    }
}

size_t RecipePatchPointContainer::addSyncObjectPatchPoint(const BasicFieldInfo& fieldInfo,
                                                          uint64_t              cmdOffsetWithinBlobInBytes)
{
    m_bIsUsingSyncObjectPatchPoints = true;

    auto sobFieldInfo = dynamic_cast<const SyncObjectAddressFieldInfo*>(&fieldInfo);
    m_patchPoints.push_back(
        new SyncObjectPatchPoint(sobFieldInfo->getTensorId(),
                                 fieldInfo.getFieldIndexOffset() + cmdOffsetWithinBlobInBytes / sizeof(uint32_t),
                                 fieldInfo.getNodeExecutionIndex()));

    m_numOfMemoryPatchPoints++;
    return m_patchPoints.size() - 1;
}
void SyncObjectPatchPoint::print() const
{
    LOG_DEBUG(RECIPE_GEN, "      type = {}", m_type);
    LOG_DEBUG(RECIPE_GEN, "      blob index = {}", m_blobIndex);
    LOG_DEBUG(RECIPE_GEN, "      offset in blob = {}", m_offsetInBlob);
}

void SyncObjectPatchPoint::serialize(patch_point_t*                pPatchPoints,
                                     std::map<uint32_t, uint32_t>& tIdToPresistentTIdx) const
{
    pPatchPoints->type              = m_type;
    pPatchPoints->blob_idx          = m_blobIndex;
    pPatchPoints->dw_offset_in_blob = m_offsetInBlob;
    int dbIndex                     = 0;

    HB_ASSERT(tIdToPresistentTIdx.find(this->m_tensorId) != tIdToPresistentTIdx.end(),
              "Failed to find tensor id: {} in persistent tensors list",
              this->m_tensorId);

    dbIndex = tIdToPresistentTIdx[this->m_tensorId];

    pPatchPoints->sob_patch_point.tensor_db_index = dbIndex;
    pPatchPoints->node_exe_index                  = m_nodeExecIndex;
}
