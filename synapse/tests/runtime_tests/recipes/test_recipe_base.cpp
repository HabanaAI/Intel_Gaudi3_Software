#include "test_recipe_base.hpp"

#include "filesystem.h"

#include "test_tensors_container.hpp"
#include "test_event.hpp"
#include "test_utils.h"

#include "../infra/test_types.hpp"
#include "synapse_api.h"

std::mutex TestRecipeBase::m_threadMutex;

TestRecipeBase::TestRecipeBase(std::string const& uniqueRecipeName,
                               synDeviceType      deviceType,
                               unsigned           inputTensorsAmount,
                               unsigned           innerTensorsAmount,
                               unsigned           outputTensorsAmount,
                               unsigned           uniqueSectionsAmount,
                               bool               isEagerMode)
: m_deviceType(deviceType),
  m_recipeHandle(nullptr),
  m_uniqueSectionsInfo(uniqueSectionsAmount),
  m_eagerMode(isEagerMode),
  m_graphHandle(nullptr),
  m_inputTensorsContainer(inputTensorsAmount),
  m_innerTensorsContainer(innerTensorsAmount),
  m_outputTensorsContainer(outputTensorsAmount),
  m_uniqueSections(uniqueSectionsAmount),
  m_numOfExternalTensors(0),
  m_filename(uniqueRecipeName)
{
    m_tensorInfoVecInputs.resize(m_inputTensorsContainer.size());
    m_tensorInfoVecOutputs.resize(m_outputTensorsContainer.size());
}

TestRecipeBase::TestRecipeBase(TestRecipeBase&& other) noexcept
: m_deviceType(other.m_deviceType),
  m_recipeHandle(other.m_recipeHandle),
  m_tensorIds(other.m_tensorIds),
  m_orderedTensorIds(other.m_orderedTensorIds),
  m_tensorInfoVecInputs(other.m_tensorInfoVecInputs),
  m_tensorInfoVecOutputs(other.m_tensorInfoVecOutputs),
  m_uniqueSectionsInfo(other.m_uniqueSectionsInfo),
  m_uniqueSectionsMemory(other.m_uniqueSectionsMemory),
  m_eagerMode(other.m_eagerMode),
  m_graphHandle(other.m_graphHandle),
  m_inputTensorsContainer(other.m_inputTensorsContainer),
  m_innerTensorsContainer(other.m_innerTensorsContainer),
  m_outputTensorsContainer(other.m_outputTensorsContainer),
  m_uniqueSections(other.m_uniqueSections),
  m_numOfExternalTensors(other.m_numOfExternalTensors),
  m_filename(other.m_filename)
{
    other.m_graphHandle  = nullptr;
    other.m_recipeHandle = nullptr;
}

TestRecipeBase::~TestRecipeBase()
{
    // Workaround for ASSERT_EQ usage
    destroy();

    if (m_graphHandle != nullptr)
    {
        _destroyGraphHandle();
    }
}

void TestRecipeBase::getIsConstProperty(bool& isConst, const synSectionId sectionId) const
{
    uint64_t  property = 0;
    synStatus status   = synRecipeSectionGetProp(m_recipeHandle, sectionId, IS_CONST, &property);
    ASSERT_EQ(status, synSuccess) << "Failed to retrieve tensor's property (is const)";

    isConst = (property == 1);
}

void TestRecipeBase::getSectionSizeProperty(uint64_t& sectionSize, const synSectionId sectionId) const
{
    synStatus status = synRecipeSectionGetProp(m_recipeHandle, sectionId, SECTION_SIZE, &sectionSize);
    ASSERT_EQ(status, synSuccess) << "Failed to get const-section's data-size";
}

synStatus TestRecipeBase::getSectionHostBufferProperty(uint64_t& hostBuffer, const synSectionId sectionId) const
{
    return synRecipeSectionGetProp(m_recipeHandle, sectionId, SECTION_DATA, &hostBuffer);
}

void TestRecipeBase::recipeSerialize()
{
    std::string fileFullPath(getFileFullPath());
    synStatus   status = synRecipeSerialize(m_recipeHandle, fileFullPath.c_str());
    ASSERT_EQ(status, synSuccess) << "Failed to synRecipeSerialize";
}

bool TestRecipeBase::recipeDeserialize()
{
    std::string fileFullPath(getFileFullPath());
    if (!fs::exists(fileFullPath))
    {
        return false;
    }

    synStatus status = synRecipeDeSerialize(&m_recipeHandle, fileFullPath.c_str());
    ASSERT_EQ(status, synSuccess) << "Failed to deserialize recipe";

    _extractTensorsId();
    _extractExecutionOrderedTensorIds();

    return true;
}

void TestRecipeBase::generateRecipe()
{
    std::lock_guard<std::mutex> lock(m_threadMutex);
    if (recipeDeserialize())
    {
        return;
    }

    compileGraph();

    recipeSerialize();
}

uint64_t TestRecipeBase::getWorkspaceSize() const
{
    uint64_t        workspaceSize;
    const synStatus status = synWorkspaceGetSize(&workspaceSize, m_recipeHandle);
    ASSERT_EQ(status, synSuccess) << "synWorkspaceGetSize failed";
    return workspaceSize;
}

uint64_t TestRecipeBase::getTensorSizeInput(unsigned tensorIndex) const
{
    const TensorInfo* pTensorInfo = getTensorInfoInput(tensorIndex);

    if (pTensorInfo == nullptr)
    {
        return 0;
    }

    return pTensorInfo->m_tensorSize;
}

uint64_t TestRecipeBase::getTensorSizeOutput(unsigned tensorIndex) const
{
    const TensorInfo* pTensorInfo = getTensorInfoOutput(tensorIndex);

    if (pTensorInfo == nullptr)
    {
        return 0;
    }

    return pTensorInfo->m_tensorSize;
}

uint64_t TestRecipeBase::getDynamicTensorMaxDimSizeInput(unsigned tensorIndex, unsigned dimIndex) const
{
    const TensorInfo* pTensorInfo = getTensorInfoInput(tensorIndex);
    if (pTensorInfo == nullptr)
    {
        return 0;
    }

    if (dimIndex >= pTensorInfo->m_dimsAmount)
    {
        return 0;
    }

    return pTensorInfo->m_tensorDimsSize[dimIndex];
}

uint64_t TestRecipeBase::getDynamicTensorMaxDimSizeOutput(unsigned tensorIndex, unsigned dimIndex) const
{
    const TensorInfo* pTensorInfo = getTensorInfoOutput(tensorIndex);
    if (pTensorInfo == nullptr)
    {
        return 0;
    }

    if (dimIndex >= pTensorInfo->m_dimsAmount)
    {
        return 0;
    }

    return pTensorInfo->m_tensorDimsSize[dimIndex];
}

void TestRecipeBase::_createGraphHandle()
{
    synStatus status =
        m_eagerMode ? synGraphCreateEager(&m_graphHandle, m_deviceType) : synGraphCreate(&m_graphHandle, m_deviceType);
    ASSERT_EQ(status, synSuccess) << "Failed to synGraphCreate";
}

void TestRecipeBase::_destroyGraphHandle()
{
    synStatus status = synGraphDestroy(m_graphHandle);
    ASSERT_EQ(status, synSuccess) << "Failed to synGraphDestroy";

    m_graphHandle = nullptr;

    m_inputTensorsContainer.destroy();
    m_innerTensorsContainer.destroy();
    m_outputTensorsContainer.destroy();
    m_uniqueSections.destroy();
}

void TestRecipeBase::_graphCompile()
{
    // Compile the graph to get an executable recipe
    synStatus status = synGraphCompile(&m_recipeHandle, m_graphHandle, m_filename.c_str(), nullptr);
    ASSERT_EQ(status, synSuccess) << "Failed to synGraphCompile";

    _extractTensorsId();
    _extractExecutionOrderedTensorIds();
}

void TestRecipeBase::_extractTensorsId()
{
    // Associate the tensors with the device memory so compute knows where to read from / write to
    uint64_t numberOfTensors = getTensorInfoVecSize();

    m_tensorIds.resize(numberOfTensors);
    synStatus status = synTensorRetrieveIds(m_recipeHandle,
                                            (const char**)getTensorsName().data(),
                                            m_tensorIds.data(),
                                            numberOfTensors);
    ASSERT_EQ(status, synSuccess) << "Failed to retrieve tensor ids";
}

void TestRecipeBase::_extractExecutionOrderedTensorIds()
{
    m_orderedTensorIds.resize(m_numOfExternalTensors);
    synStatus status =
        synTensorExtExtractExecutionOrder(m_recipeHandle, m_numOfExternalTensors, m_orderedTensorIds.data());
    ASSERT_EQ(status, synSuccess) << "Failed to extract external-tensors' execution-order";
}

std::string TestRecipeBase::getFileFullPath() const
{
    return TEST_RESOURCE_PATH + m_filename;
}

std::string TestRecipeBase::getRecipeName() const
{
    return m_filename;
}

const void* TestRecipeBase::getHostBuffer(bool                         isInput,
                                          unsigned                     tensorIndex,
                                          const TensorMemoryVec&       tensorInfoVec,
                                          const std::vector<uint64_t>& rConstSectionsHostBuffers) const
{
    const TensorInfo* pTensorInfo = getTensorInfo(tensorIndex, isInput);
    ASSERT_NE(pTensorInfo, nullptr) << "Invalid tensor-info (nullptr)";

    if (pTensorInfo->m_tensorFlags != (uint32_t)TensorFlag::CONST_SECTION)
    {
        return tensorInfoVec[tensorIndex].getTestHostBuffer().getBuffer();
    }
    else
    {
        ASSERT_LT(pTensorInfo->m_sectionIndex, rConstSectionsHostBuffers.size()) << "Invalid tensor's section-index";
        return (const void*)((uint8_t*)rConstSectionsHostBuffers[pTensorInfo->m_sectionIndex] +
                             pTensorInfo->m_sectionOffset);
    }
}

const TensorInfo* TestRecipeBase::getTensorInfo(unsigned tensorIndex, bool isInput) const
{
    const TensorInfoVec& tensorInfoVec = isInput ? m_tensorInfoVecInputs : m_tensorInfoVecOutputs;

    if (tensorIndex >= tensorInfoVec.size())
    {
        LOG_ERR(SYN_RT_TEST, "Invalid {}'s tensor-index", isInput ? "input" : "output");
        return nullptr;
    }

    return &tensorInfoVec[tensorIndex];
}

bool TestRecipeBase::isOnConstSection(unsigned tensorIndex, bool isInput) const
{
    const TensorInfoVec& tensorInfoVec = isInput ? m_tensorInfoVecInputs : m_tensorInfoVecOutputs;

    ASSERT_LT(tensorIndex, tensorInfoVec.size()) << "Invalid {}'s tensor-index" << (isInput ? "input" : "output");

    return (tensorInfoVec[tensorIndex].m_tensorFlags == (uint32_t)TensorFlag::CONST_SECTION);
}

const TensorInfo* TestRecipeBase::getTensorInfo(const std::string& tensorName) const
{
    for (unsigned tensorIndex = 0; tensorIndex < m_tensorInfoVecInputs.size(); tensorIndex++)
    {
        if (m_tensorInfoVecInputs[tensorIndex].m_tensorName != tensorName)
        {
            continue;
        }
        return &m_tensorInfoVecInputs[tensorIndex];
    }

    for (unsigned tensorIndex = 0; tensorIndex < m_tensorInfoVecOutputs.size(); tensorIndex++)
    {
        if (m_tensorInfoVecOutputs[tensorIndex].m_tensorName != tensorName)
        {
            continue;
        }
        return &m_tensorInfoVecOutputs[tensorIndex];
    }

    LOG_ERR(SYN_RT_TEST, "Tensor name '{}' not found", tensorName);
    return nullptr;
}

synTensorDescriptor TestRecipeBase::getTensorDescriptor(synDataType   dataType,
                                                        const TSize*  tensorSizes,
                                                        TSize         dims,
                                                        const char*   name,
                                                        unsigned*     strides,
                                                        void*         ptr,
                                                        bool          isQuantized,
                                                        const TSize*  minSizes,
                                                        synTensorType tensorType)
{
    synTensorDescriptor desc;

    desc.m_dataType    = dataType;
    desc.m_dims        = dims;
    desc.m_name        = name;
    desc.m_ptr         = ptr;
    desc.m_isQuantized = isQuantized;
    desc.m_tensorType  = tensorType;

    memset(desc.m_strides, 0, sizeof(desc.m_strides));
    memset(desc.m_sizes, 1, sizeof(desc.m_sizes));
    for (size_t i = 0; i < dims; i++)
        desc.m_sizes[i] = tensorSizes[i];

    if (minSizes != nullptr)
    {
        memset(desc.m_minSizes, 1, sizeof(desc.m_sizes));
        for (size_t i = 0; i < dims; i++)
            desc.m_minSizes[i] = minSizes[i];
    }

    if (strides)
    {
        memcpy(desc.m_strides, strides, (dims + 1) * sizeof(unsigned));
    }

    return desc;
}

const std::vector<const char*> TestRecipeBase::getTensorsName() const
{
    std::vector<const char*> namesVec;
    for (auto& tensorInfo : m_tensorInfoVecInputs)
    {
        namesVec.push_back(tensorInfo.m_tensorName.c_str());
    }

    for (auto& tensorInfo : m_tensorInfoVecOutputs)
    {
        namesVec.push_back(tensorInfo.m_tensorName.c_str());
    }

    return namesVec;
}

void TestRecipeBase::extractLaunchTensorsInfo(LaunchTensorsInfoDB& launchTensorsInfo) const
{
    synStatus status(synSuccess);

    unsigned numOfTensors = getTensorInfoVecSize();

    launchTensorsInfo.clear();
    launchTensorsInfo.resize(numOfTensors);

    // Associate the tensors with the device memory so compute knows where to read from / write to
    const synRecipeHandle&       recipeHandle = getRecipe();
    const std::vector<uint64_t>& tensorsId    = retrieveTensorsId();

    for (unsigned i = 0; i < numOfTensors; i++)
    {
        launchTensorsInfo[i].tensorId = tensorsId[i];
    }

    status = synTensorRetrieveLaunchInfoByIdExt(recipeHandle, numOfTensors, launchTensorsInfo.data());
    ASSERT_EQ(status, synSuccess) << "Failed to retrieve tensor-info";
}

void TestRecipeBase::getConstSectionsIds(std::vector<synSectionId>& constSectionIdDB) const
{
    synStatus status(synSuccess);

    LaunchTensorsInfoDB launchTensorsInfo;
    extractLaunchTensorsInfo(launchTensorsInfo);

    const synRecipeHandle& recipeHandle = getRecipe();

    for (auto singleLaunchInfo : launchTensorsInfo)
    {
        synSectionId   sectionId   = singleLaunchInfo.tensorSectionId;
        synSectionProp sectionProp = IS_CONST;
        uint64_t       property    = 0;

        status = synRecipeSectionGetProp(recipeHandle, sectionId, sectionProp, &property);
        ASSERT_EQ(status, synSuccess) << "Failed to retrieve tensor's property (is const)";
        if (!property)
        {
            continue;
        }

        LOG_TRACE(SYN_TEST, "sectionId {} is const-section", sectionId);
        constSectionIdDB.push_back(sectionId);
    }
}

void TestRecipeBase::clearConstSectionHostBuffers(const std::vector<synSectionId>& constSectionIds) const
{
    LOG_DEBUG(SYN_RT_TEST, "Free const-section's host-buffer");
    synStatus status = synRecipeSectionHostBuffersClear(getRecipe(), constSectionIds.data(), constSectionIds.size());
    ASSERT_EQ(status, synSuccess) << "Failed to free host-buffer of recipe's const-section";
}

void TestRecipeBase::getConstSectionInfo(uint64_t& sectionSize, void*& hostSectionData, synSectionId sectionId) const
{
    synStatus status(synSuccess);

    status = synRecipeSectionGetProp(m_recipeHandle, sectionId, SECTION_SIZE, &sectionSize);
    ASSERT_EQ(status, synSuccess) << "Failed to retrieve const-section's size";

    status = synRecipeSectionGetProp(m_recipeHandle, sectionId, SECTION_DATA, (uint64_t*)&hostSectionData);
    ASSERT_EQ(status, synSuccess) << "Failed to retrieve const-section's data (buffer)";
}

void TestRecipeBase::compileGraph()
{
    _createGraphHandle();

    _graphCreation();

    _graphCompile();

    _destroyGraphHandle();
}

void TestRecipeBase::retrieveAmountOfExternalTensors(uint64_t& amountOfExternalTensors)
{
    static const unsigned numOfAttributes = 1;
    synRecipeAttribute    recipeAttr(RECIPE_ATTRIBUTE_NUM_EXTERNAL_TENSORS);

    synStatus status = synRecipeGetAttribute(&amountOfExternalTensors, &recipeAttr, numOfAttributes, m_recipeHandle);
    ASSERT_EQ(status, synSuccess) << "Failed to retrieve num of external tensors attribute";
}

const std::vector<uint64_t>& TestRecipeBase::retrieveTensorsId() const
{
    return m_tensorIds;
}

const std::vector<uint64_t>& TestRecipeBase::retrieveOrderedTensorsId() const
{
    return m_orderedTensorIds;
}

void TestRecipeBase::mapEventToExternalTensor(TestEvent& rEvent, const synLaunchTensorInfoExt* launchTensorsInfo) const
{
    static const size_t externalTensorsMappingAmount = 1;
    rEvent.mapTensor(externalTensorsMappingAmount, launchTensorsInfo, m_recipeHandle);
}

unsigned TestRecipeBase::getNumberInputConstTensors() const
{
    unsigned numOfTensors = 0;
    for (const auto& tensorInfoInput : m_tensorInfoVecInputs)
    {
        if (tensorInfoInput.m_isConst)
        {
            numOfTensors++;
        }
    }
    return numOfTensors;
}

void TestRecipeBase::destroy()
{
    if (m_recipeHandle != 0)
    {
        synStatus status = synRecipeDestroy(m_recipeHandle);
        ASSERT_EQ(status, synSuccess);
        m_recipeHandle = 0;
    }
}