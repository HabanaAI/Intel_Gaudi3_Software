#pragma once

#include "address_fields_container_info.h"

#include "define_synapse_common.hpp"

#include <cstdint>
#include <list>
#include <memory>
#include <unordered_map>
#include <vector>

class HabanaGraph;

static const uint8_t DEFAULT_SECTION_TYPE      = 0;
static const uint8_t PROGRAM_DATA_SECTION_TYPE = 1;

class RecipeAllocator;
class RecipeBlob;

class RecipePatchPoint
{
public:
    RecipePatchPoint(uint64_t offsetInBlob, uint32_t nodeExecIndex);
    virtual ~RecipePatchPoint();

    void         setBlobIndex(uint64_t blobIdx)               { m_blobIndex = blobIdx; }
    uint64_t     getOffsetInBlob()                            { return m_offsetInBlob; }
    uint64_t     getBlobIndex() const                         { return m_blobIndex;    }
    patch_point_t::EPatchPointType getType() const            { return m_type;         }
    uint32_t     getNodeExecIndex()                           { return m_nodeExecIndex; }
    void         setNodeExecIndex(uint32_t nodeExecIndex)     { m_nodeExecIndex = nodeExecIndex; }

    virtual void print() const = 0;
    RecipePatchPoint()         = delete;

protected:
    uint32_t                        m_blobIndex;
    uint32_t                        m_offsetInBlob; // offset in 32bit word units to the word in the blob that requires patching
    patch_point_t::EPatchPointType  m_type;
    uint32_t                        m_nodeExecIndex;

private:
};


class MemoryPatchPoint : public RecipePatchPoint
{
public:
    MemoryPatchPoint(FieldType addrPartType, uint64_t virtualAddr, uint64_t sectionIndex, uint64_t offsetInBlob, uint32_t nodeExecIndex);

    virtual ~MemoryPatchPoint();

    void serialize(patch_point_t* pPatchPoints) const;
    void         setType(patch_point_t::EPatchPointType type) { m_type = type; }
    virtual void print() const override;
    uint64_t getSectionId() { return m_sectionIndex; }
    MemoryPatchPoint() = delete;

private:

    uint64_t m_virtualAddr;
    uint64_t m_sectionIndex; // section ID and section index are synonymous - we don't expect to have gaps in the section ID
};

class SyncObjectPatchPoint : public RecipePatchPoint
{
public:
    SyncObjectPatchPoint(uint32_t tensorId, uint64_t offsetInBlob, uint32_t nodeExecIndex)
    : RecipePatchPoint(offsetInBlob, nodeExecIndex), m_tensorId(tensorId)
    {
        setType(patch_point_t::EPatchPointType::SOB_PATCH_POINT);
    }

    void         serialize(patch_point_t* pPatchPoints, std::map<uint32_t, uint32_t>& tIdToPresistentTIdx) const;
    void         setType(patch_point_t::EPatchPointType type) { m_type = type; }
    virtual void print() const override;

private:
    uint32_t m_tensorId;
};

class DynamicPatchPoint : public RecipePatchPoint
{
public:
    DynamicPatchPoint(EFieldType           type,
                      uint32_t             offsetInBlob,
                      std::vector<uint8_t> metadata,
                      const NodeROI*       roi,
                      ShapeFuncID          functionId,
                      uint64_t             patchSizeInDw,
                      uint32_t             highPatchPointIndex,
                      uint32_t             lowPatchPointIndex,
                      bool                 isUnskippable,
                      unsigned             blobId);

    const NodeROI* getRoi() { return m_roi; }
    void           serialize(sm_patch_point_t* pPatchPoint, RecipeAllocator* pRecipeAlloc) const;
    void print() const override;
    void setRoiIndex(uint32_t roiIndex) { m_roiIndex = roiIndex; }
    uint32_t getRoiIndex() const { return m_roiIndex; }
    EFieldType getFieldType() const { return m_smppType; }
    uint64_t       getHighPatchPointIndex() { return m_highPatchPointIndex; }
    uint64_t       getLowPatchPointIndex() { return m_lowPatchPointIndex; }
    void           setHighPatchPointIndex(uint64_t index) { m_highPatchPointIndex = index; }
    void           setLowPatchPointIndex(uint64_t index) { m_lowPatchPointIndex = index; }
    unsigned       getBlobId() { return m_blobId; }

private:
    EFieldType                        m_smppType;
    bool                              m_isUnskippable;
    std::vector<uint8_t>              m_metadata;
    const NodeROI*                    m_roi;
    uint32_t                          m_roiIndex;
    ShapeFuncID                       m_functionId;
    uint64_t                          m_patchSizeInDw;
    uint32_t                          m_highPatchPointIndex;
    uint32_t                          m_lowPatchPointIndex;
    unsigned                          m_blobId;
};
using DynamicPatchPointPtr = std::shared_ptr<DynamicPatchPoint>;

class RecipePatchPointContainer
{
public:
    RecipePatchPointContainer() = default;
    virtual ~RecipePatchPointContainer();

    RecipePatchPoint&    operator[](int index);
    std::list<uint64_t>
    insertPatchPoints(const BasicFieldsContainerInfo& afci, uint64_t cmdOffsetWithinBlobInBytes, unsigned blobId = 0);
    void                 erasePatchPointByIndex(unsigned idx);
    void                 serialize(uint32_t*                          pNumPatchPoints,
                                   uint32_t*                          pNumActivatePatchPoints,
                                   patch_point_t**                    ppPatchPoints,
                                   uint32_t*                          pNumSectionTypes,
                                   section_group_t**                  pSectionTypePatchPoints,
                                   const std::map<uint64_t, uint8_t>& sectionIdToSectionType,
                                   uint32_t*                          pNumOfSectionIds,
                                   section_blobs_t**                  pSectionIdBlobIndices,
                                   section_group_t*                   pSignalOutSection,
                                   RecipeAllocator*                   pRecipeAlloc,
                                   const TensorSet&                   persistTensors) const;
    void                 print() const;
    unsigned             getNumOfMemoryPatchPoints() const { return m_numOfMemoryPatchPoints; }
    unsigned             getNumOfActivatePatchPoints() const { return m_numOfActivatePatchPointes; }

    const std::vector<RecipePatchPoint*>& getPatchPoints() const { return m_patchPoints; }

    void setPatchPointsBlobIndex(uint64_t patchingBlobIndex, unsigned blobId);

    uint8_t getSectionTypeForSectionId(const std::map<uint64_t, uint8_t>& sectionIdToSectionType,
                                       uint64_t                           sectionId) const;

    const std::unordered_map<uint32_t, uint64_t>& getAccumulatedPatchPointsPerNode() const { return m_nodeToPatchPointsCount; }
    void markExistingPatchPointsAsActivateProgram() { m_numOfActivatePatchPointes = getPatchPoints().size(); }

private:
    using DynamicPatchPointVector = std::vector<std::shared_ptr<DynamicPatchPoint>>;

    RecipePatchPointContainer(const RecipePatchPointContainer&) = delete;
    RecipePatchPointContainer& operator=(const RecipePatchPointContainer&) = delete;

    // Returns the index
    size_t addMemoryPatchPoint(AddressFieldInfo& fieldInfo, uint64_t cmdOffsetWithinBlobInBytes);
    void   addDynamicPatchPoint(const BasicFieldInfo& fieldInfo, uint64_t cmdOffsetWithinBlobInBytes, unsigned blobId);
    size_t addSyncObjectPatchPoint(const BasicFieldInfo& fieldInfo, uint64_t cmdOffsetWithinBlobInBytes);

    MemoryPatchPoint* getNeighboringLowerCounterPart(const AddressFieldInfo& highMemoryFieldInfo, uint64_t cmdOffsetWithinBlobInBytes, const std::unordered_map<uint64_t, uint64_t>& ppIndicesMap);

    void getPatchPointsByNodeId(std::map<uint32_t, std::vector<RecipePatchPoint*>>& nodeToPatchPointList,
                                std::map<RecipePatchPoint*, uint64_t>&              patchPointCurrentPos) const;

    void updateDynamicPatchPoints(std::map<uint64_t, uint64_t>& patchPointCurrentPos) const;

    void calcAccumulatedPatchPointsPerNode(
        const std::map<uint32_t, std::vector<RecipePatchPoint*>>& nodeToPatchPointList) const;

    std::vector<RecipePatchPoint*>                 m_patchPoints;
    DynamicPatchPointVector                        m_dynamicPatchPoints;
    DynamicPatchPointVector                        m_dynamicPatchPointsToUpdate;
    mutable std::unordered_map<uint32_t, uint64_t> m_nodeToPatchPointsCount;
    unsigned                                       m_numOfMemoryPatchPoints        = 0;
    bool                                           m_bIsUsingSyncObjectPatchPoints = false;
    unsigned                                       m_numOfActivatePatchPointes     = 0;
};
