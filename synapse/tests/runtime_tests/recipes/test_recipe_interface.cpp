#include "test_recipe_interface.hpp"

#include "test_tensors_container.hpp"

#include "../infra/test_types.hpp"
#include "synapse_api.h"

#include "utils.h"

synStatus TestRecipeInterface::getConstSectionProperties(bool&              isConst,
                                                         uint64_t&          sectionSize,
                                                         uint64_t&          hostBuffer,
                                                         const synSectionId sectionId) const
{
    getIsConstProperty(isConst, sectionId);

    if (!isConst)
    {
        return synSuccess;
    }

    getSectionSizeProperty(sectionSize, sectionId);

    return getSectionHostBufferProperty(hostBuffer, sectionId);
}

void TestRecipeInterface::createUniqueSections(TestSectionsContainer& sections, const synGraphHandle& graphHandle)
{
    const SectionInfoVec& uniqueSectionsInfo = getUniqueSectionsInfo();

    unsigned sectionIndex = 0;
    for (auto singleSectionInfo : uniqueSectionsInfo)
    {
        synSectionHandle sectionHandle;

        createSection(&sectionHandle, graphHandle, true /* isPersist */, singleSectionInfo.m_isConstSection);

        sections.setSection(sectionIndex, sectionHandle);
        sectionIndex++;
    }
}

void TestRecipeInterface::createSection(synSectionHandle*    pSectionHandle,
                                        const synGraphHandle graphHandle,
                                        bool                 isPersist,
                                        bool                 isConstSection)
{
    ASSERT_NE(pSectionHandle, nullptr) << "Failed to create section handle (nullptr)";

    uint64_t sectionDescriptor = 0;

    synStatus status = synSectionCreate(pSectionHandle, sectionDescriptor, graphHandle);
    ASSERT_EQ(status, synSuccess) << "Failed to create section";

    status = synSectionSetPersistent(*pSectionHandle, isPersist);
    ASSERT_EQ(status, synSuccess) << "Failed to set section persist attribute";

    if (isConstSection)
    {
        status = synSectionSetConst(*pSectionHandle, isConstSection);
        ASSERT_EQ(status, synSuccess) << "Failed to set section as const";
    }
}

void TestRecipeInterface::createTrainingTensor(TestTensorsContainer& tensors,
                                               unsigned              tensorIndex,
                                               const TensorInfo&     tensorInfo,
                                               bool                  isPersist,
                                               const std::string&    name,
                                               const synGraphHandle  graphHandle,
                                               synSectionHandle*     pGivenSectionHandle,
                                               void*                 hostBuffer)
{
    synTensor tensor;
    synStatus status = synTensorHandleCreate(&tensor, graphHandle, tensorInfo.m_tensorType, name.c_str());
    ASSERT_EQ(status, synSuccess) << "Failed to create tensor handle";

    if (tensorInfo.m_isSfg)
    {
        status = synTensorSetExternal(tensor, true);
        ASSERT_EQ(status, synSuccess) << "Failed to set tensor as external (SFG)";
    }

    synSectionHandle sectionHandle = pGivenSectionHandle ? *pGivenSectionHandle : nullptr;
    switch (tensorInfo.m_sectionType)
    {
        case TestSectionType::NON_CONST_SECTION:
        case TestSectionType::CONST_SECTION:
        {
            if (!sectionHandle && isPersist)
            {
                createSection(&sectionHandle, graphHandle, isPersist, false /* isConstSection */);
            }
            if (sectionHandle)
            {
                status = synTensorAssignToSection(tensor, sectionHandle, tensorInfo.m_sectionOffset);
                ASSERT_EQ(status, synSuccess) << "Failed to assign tensor section";
            }
            break;
        }
        case TestSectionType::CONST_TENSOR_SECTION:
        {
            break;
        }
        default:
            ASSERT_TRUE(false) << "Unsupported TestSectionType";
            break;
    }

    if (hostBuffer != nullptr)
    {
        status = synTensorSetHostPtr(tensor,
                                     hostBuffer,
                                     tensorInfo.m_tensorSize,
                                     tensorInfo.m_dataType,
                                     true /* copyBuffer */);
        ASSERT_EQ(status, synSuccess) << "Failed to assign tensor section";
    }

    synTensorGeometry geometry;
    geometry.dims = tensorInfo.m_dimsAmount;
    std::copy(tensorInfo.m_tensorDimsSize, tensorInfo.m_tensorDimsSize + tensorInfo.m_dimsAmount, geometry.sizes);

    status = synTensorSetGeometry(tensor, &geometry, synGeometryMaxSizes);
    ASSERT_EQ(status, synSuccess) << "Failed to set tensor geometry";

    if (tensorInfo.m_tensorType == DATA_TENSOR_DYNAMIC)
    {
        std::copy(tensorInfo.m_tensorMinDimsSize,
                  tensorInfo.m_tensorMinDimsSize + tensorInfo.m_dimsAmount,
                  geometry.sizes);
        status = synTensorSetGeometry(tensor, &geometry, synGeometryMinSizes);
        ASSERT_EQ(status, synSuccess) << "Failed to set tensor geometry (min)";
    }

    synTensorDeviceFullLayout deviceLayout;
    deviceLayout.deviceDataType = tensorInfo.m_dataType;

    std::fill_n(deviceLayout.strides, ARRAY_SIZE(deviceLayout.strides), 0);
    status = synTensorSetDeviceFullLayout(tensor, &deviceLayout);
    ASSERT_EQ(status, synSuccess) << "Failed to set tensor layout";

    if (pGivenSectionHandle != nullptr)
    {
        // return null in case we didn't create a new section
        sectionHandle = nullptr;
    }
    tensors.setTensor(tensorIndex, name, tensor, sectionHandle);
}

void TestRecipeInterface::createTrainingTensor(TestTensorsContainer& tensors,
                                               unsigned              tensorIndex,
                                               TSize                 dims,
                                               synDataType           dataType,
                                               const TSize*          tensorSize,
                                               bool                  isPersist,
                                               const std::string&    name,
                                               const synGraphHandle  graphHandle,
                                               synSectionHandle*     pGivenSectionHandle,
                                               bool                  isConstSection,
                                               uint64_t              offset,
                                               void*                 hostBuffer,
                                               synTensorType         tensorType,
                                               const TSize*          minTensorSize)
{
    synTensor tensor;
    synStatus status = synTensorHandleCreate(&tensor, graphHandle, tensorType, name.c_str());
    ASSERT_EQ(status, synSuccess) << "Failed to create tensor handle";

    synSectionHandle sectionHandle = pGivenSectionHandle ? *pGivenSectionHandle : nullptr;
    if (!sectionHandle && isPersist)
    {
        createSection(&sectionHandle, graphHandle, isPersist, false /* isConstSection */);
    }

    if (sectionHandle)
    {
        status = synTensorAssignToSection(tensor, sectionHandle, offset);
        ASSERT_EQ(status, synSuccess) << "Failed to assign tensor section";
    }

    synTensorGeometry geometry;
    geometry.dims = dims;
    std::copy(tensorSize, tensorSize + dims, geometry.sizes);

    status = synTensorSetGeometry(tensor, &geometry, synGeometryMaxSizes);
    ASSERT_EQ(status, synSuccess) << "Failed to set tensor geometry";

    if (tensorType == DATA_TENSOR_DYNAMIC)
    {
        ASSERT_NE(minTensorSize, nullptr) << "Minimal tensor-size is nullptr for a dynamic-tensor";

        std::copy(minTensorSize, minTensorSize + dims, geometry.sizes);
        status = synTensorSetGeometry(tensor, &geometry, synGeometryMinSizes);
        ASSERT_EQ(status, synSuccess) << "Failed to set tensor geometry (min)";
    }

    synTensorDeviceFullLayout deviceLayout;
    deviceLayout.deviceDataType = dataType;

    std::fill_n(deviceLayout.strides, ARRAY_SIZE(deviceLayout.strides), 0);
    status = synTensorSetDeviceFullLayout(tensor, &deviceLayout);
    ASSERT_EQ(status, synSuccess) << "Failed to set tensor layout";

    if (pGivenSectionHandle != nullptr)
    {
        // return null in case we didn't create a new section
        sectionHandle = nullptr;
    }
    tensors.setTensor(tensorIndex, name, tensor, sectionHandle);
}

void TestRecipeInterface::clearResourceFiles()
{
    fs::path p = TEST_RESOURCE_PATH;
    fs::remove_all(p);
}

unsigned TestRecipeInterface::getSectionID(TensorInfo& tensorInfo) const
{
    unsigned ret = -1;
    if (!tensorInfo.m_isConst)
    {
        TensorMetadataInfo tensorMetadataInfo;
        tensorMetadataInfo.tensorName = tensorInfo.m_tensorName.c_str();
        synStatus status = synTensorRetrieveInfosByName(getRecipe(), 1, &tensorMetadataInfo);
        if (status != synSuccess)
        {
            LOG_WARN(SYN_RECIPE, "synTensorRetrieveInfosByName failed. tensorName={} size={} flags={}",
                tensorInfo.m_tensorName, tensorInfo.m_tensorSize , tensorInfo.m_tensorFlags);
        }
        else
        {
            ret = tensorMetadataInfo.sectionId;
        }
    }
    return ret;
}
