#include "recipe_static_information.hpp"
#include "recipe.h"

RecipeStaticInfo::RecipeStaticInfo() : m_isInitialized(false) {}

RecipeStaticInfo::~RecipeStaticInfo()
{
    for (size_t i = 0; i < EXECUTION_STAGE_LAST; i++)
    {
        for (auto& v : m_pPatchPointsDataChunksInfo[i])
        {
            delete v.second;
        }
    }
}

void RecipeStaticInfo::initDBs(const recipe_t& rRecipe)
{
    uint64_t blobsNum = rRecipe.blobs_nr;
    m_programCodeBlobsToDeviceAddress.resize(blobsNum, {});
    m_programCodeBlobsToCpDmaAddress.resize(blobsNum);
    m_programCodeBlobIndicesToCpDmaMappedAddress.resize(blobsNum, 0);
    m_patchableBlobsOffsetInDc.resize(blobsNum);
    m_pSobjPatchPointsDataChunksInfo = nullptr;
    for (uint8_t executionStage = EXECUTION_STAGE_ACTIVATE; executionStage < EXECUTION_STAGE_LAST; executionStage++)
    {
        m_pPatchPointsDataChunksInfo[executionStage].reserve(rRecipe.section_groups_nr + 1);  // +1 for "all_types"
    }
}

void RecipeStaticInfo::clearDBs()
{
    m_programCodeBlobsToDeviceAddress.clear();
    m_programCodeBlobsToCpDmaAddress.clear();
    m_programCodeBlobIndicesToCpDmaMappedAddress.clear();
    m_patchableBlobsOffsetInDc.clear();
    m_pSmPatchPointsDataChunksInfo.reset();
}

void RecipeStaticInfo::setInitialized(bool isInitialized)
{
    m_isInitialized = isInitialized;
}

bool RecipeStaticInfo::isInitialized()
{
    return m_isInitialized;
}

void RecipeStaticInfo::setArbitrationSetCommand(uint64_t arbSetCmdHostAddress)
{
    m_arbSetCmdHostAddress.set(arbSetCmdHostAddress);
}

void RecipeStaticInfo::setArbitrationReleaseCommand(uint64_t arbReleaseCmdHostAddress)
{
    m_arbReleaseCmdHostAddress.set(arbReleaseCmdHostAddress);
}

void RecipeStaticInfo::setCpDmaChunksAmount(eExecutionStage stage, CachedAndNot cpDmaChunksAmount)
{
    HB_ASSERT(stage != EXECUTION_STAGE_LAST, "Illegal execution stage");
    m_cpDmaChunksAmount[stage].set(cpDmaChunksAmount);
}

bool RecipeStaticInfo::addProgramCodeBlockMapping(uint64_t handle, uint64_t mappedAddress, uint64_t size)
{
    return m_programCodeBlocksMapping.addMapping(handle,
                                                 mappedAddress,
                                                 size,
                                                 AddressRangeMapper::ARM_MAPPING_TYPE_RANGE);
}

void RecipeStaticInfo::clearProgramCodeBlockMapping()
{
    m_programCodeBlocksMapping.clear();
}

void RecipeStaticInfo::setCpDmaStaticBlobsBuffer(char* pCpDmaBufferAddress)
{
    m_pCpDmaBufferAddress.set(pCpDmaBufferAddress);
}

bool RecipeStaticInfo::getCpDmaStaticBlobsBuffer(char*& rpCpDmaBufferAddress) const
{
    if (m_pCpDmaBufferAddress.is_set())
    {
        rpCpDmaBufferAddress = m_pCpDmaBufferAddress.value();
        return true;
    }
    return false;
}

void RecipeStaticInfo::clearCpDmaStaticBlobsBuffer()
{
    m_pCpDmaBufferAddress.unset();
}

void RecipeStaticInfo::addProgramCodeBlobDeviceAddress(uint64_t        blobIdx,
                                                       blobAddressType blobAddress,
                                                       uint64_t        blobDeviceAddress,
                                                       uint64_t        blobPartialSize)
{
    m_programCodeBlobsToDeviceAddress[blobIdx].hostAddr = blobAddress;
    if (m_programCodeBlobsToDeviceAddress[blobIdx].devAddrAndSize.size == 0)  // first one, 99% of the cases
    {
        m_programCodeBlobsToDeviceAddress[blobIdx].devAddrAndSize = {blobDeviceAddress, blobPartialSize};
    }
    else
    {
        m_programCodeBlobsToDeviceAddress[blobIdx].extraDevAddrAndSize.push_back({blobDeviceAddress, blobPartialSize});
    }
}

void RecipeStaticInfo::setProgramCodeBlobCpDmaAddress(uint64_t blobIdx, uint64_t cpDmaPacketHostAddress)
{
    m_programCodeBlobsToCpDmaAddress[blobIdx].push_back(cpDmaPacketHostAddress);
}

void RecipeStaticInfo::setPatchingPointsDcLocation(eExecutionStage           executionStage,
                                                   uint64_t                  patchPointsTypeId,
                                                   DataChunkPatchPointsInfo* pPatchPointsDataChunksInfo,
                                                   uint64_t                  amountOfPps,
                                                   bool                      isSobj)
{
    HB_ASSERT(executionStage != EXECUTION_STAGE_LAST, "Illegal execution stage");

    HB_ASSERT_PTR(pPatchPointsDataChunksInfo);

    if (isSobj)
    {
        HB_ASSERT(m_pSobjPatchPointsDataChunksInfo == nullptr, "m_pSobjPatchPointsDataChunksInfo is already set");
        m_pSobjPatchPointsDataChunksInfo = pPatchPointsDataChunksInfo;
    }
    else
    {
        HB_ASSERT(m_pPatchPointsDataChunksInfo[executionStage].find(patchPointsTypeId) ==
                      m_pPatchPointsDataChunksInfo[executionStage].end(),
                  "m_pPatchPointsDataChunksInfo is already set");
        m_pPatchPointsDataChunksInfo[executionStage][patchPointsTypeId] = pPatchPointsDataChunksInfo;
        m_pPatchPointsAmount[executionStage][patchPointsTypeId]         = amountOfPps;
    }
}

bool RecipeStaticInfo::allocateSmPatchingPointsDcLocation(uint64_t patchPointsAmount, uint32_t dcSizeCommand)
{
    m_pSmPatchPointsDataChunksInfo = std::make_unique<DataChunkSmPatchPointsInfo>();
    try
    {
        m_pSmPatchPointsDataChunksInfo->m_dataChunkSmPatchPoints = new data_chunk_sm_patch_point_t[patchPointsAmount];
        m_pSmPatchPointsDataChunksInfo->m_singleChunkSize        = dcSizeCommand;
    }
    catch (std::bad_alloc& err)
    {
        m_pSmPatchPointsDataChunksInfo.reset();
        LOG_ERR(SYN_RECIPE, "Failed to allocate PPs per DCs due to {}", err.what());
        return false;
    }
    return true;
}

bool RecipeStaticInfo::getArbitrationSetHostAddress(uint64_t& arbSetCmdHostAddress) const
{
    if (!m_arbSetCmdHostAddress.is_set())
    {
        return false;
    }

    arbSetCmdHostAddress = m_arbSetCmdHostAddress.value();

    return true;
}

bool RecipeStaticInfo::getArbitrationReleaseHostAddress(uint64_t& arbReleaseCmdHostAddress) const
{
    if (!m_arbReleaseCmdHostAddress.is_set())
    {
        return false;
    }

    arbReleaseCmdHostAddress = m_arbReleaseCmdHostAddress.value();

    return true;
}

bool RecipeStaticInfo::getCpDmaChunksAmount(eExecutionStage stage, uint64_t& cpDmaChunksAmount, bool inCache) const
{
    HB_ASSERT(stage != EXECUTION_STAGE_LAST, "Illegal execution stage");
    if (!m_cpDmaChunksAmount[stage].is_set())
    {
        return false;
    }

    cpDmaChunksAmount =
        inCache ? m_cpDmaChunksAmount[stage].value().cached : m_cpDmaChunksAmount[stage].value().notCached;
    return true;
}

const AddressRangeMapper& RecipeStaticInfo::getProgramCodeBlocksMapping() const
{
    return m_programCodeBlocksMapping;
}

void RecipeStaticInfo::setProgramDataMappedAddress(blobAddressType mappedAddr)
{
    m_programDataMappedAddress.set(mappedAddr);
}

void RecipeStaticInfo::setProgramCodeMappedAddress(blobAddressType mappedAddr)
{
    m_programCodeMappedAddress.set(mappedAddr);
}

bool RecipeStaticInfo::getProgramDataMappedAddress(blobAddressType& mappedAddr) const
{
    if (!m_programDataMappedAddress.is_set())
    {
        return false;
    }

    mappedAddr = m_programDataMappedAddress.value();
    return true;
}

bool RecipeStaticInfo::getProgramCodeMappedAddress(blobAddressType& mappedAddr) const
{
    if (!m_programCodeMappedAddress.is_set())
    {
        return false;
    }

    mappedAddr = m_programCodeMappedAddress.value();
    return true;
}

void RecipeStaticInfo::clearProgramDataMappedAddress()
{
    m_programDataMappedAddress.unset();
}

void RecipeStaticInfo::clearProgramCodeMappedAddress()
{
    m_programCodeMappedAddress.unset();
}

bool RecipeStaticInfo::getProgramCodeBlobCpDmaAmount(uint64_t blobIdx, uint8_t& cpDmaAmount)
{
    if (blobIdx >= m_programCodeBlobsToDeviceAddress.size())
    {
        LOG_ERR(SYN_RECIPE, "Trying to get program code blob cp dma amount from non exist blob index, idx = {}", blobIdx);
        return false;
    }

    HostAndDevAddr& hostAndDevAddr = m_programCodeBlobsToDeviceAddress[blobIdx];

    if (hostAndDevAddr.devAddrAndSize.size == 0)  // an indication this entry wasn't set (can use the addr as well)
    {
        LOG_WARN(SYN_RECIPE, "{}: Blob's index {} in static part of reciepe is with size 0", HLLOG_FUNC, blobIdx);
        cpDmaAmount = 0;
    }
    else
    {
        cpDmaAmount = 1 + hostAndDevAddr.extraDevAddrAndSize
                            .size();  // the addr/size in the struct + the number of addr/size in the vector
    }

    return true;
}

bool RecipeStaticInfo::getProgramCodeBlobCpDmaAddress(uint64_t                       blobIdx,
                                                      const BlobCpDmaHostAddresses*& cpDmaPacketHostAddress) const
{
    cpDmaPacketHostAddress = &(m_programCodeBlobsToCpDmaAddress.at(blobIdx));
    return true;
}

const std::vector<HostAndDevAddr>& RecipeStaticInfo::getProgramCodeBlobsToDeviceAddress() const
{
    return m_programCodeBlobsToDeviceAddress;
}

const std::array<DataChunksDB, RecipeStaticInfo::EXTERNAL_QMANS_AMOUNT>&
RecipeStaticInfo::retrieveExternalQueueBlobsDataChunks() const
{
    return m_externalQueueDataChunks;
}

const DataChunkPatchPointsInfo* RecipeStaticInfo::getPatchingPointsDcLocation(eExecutionStage executionStage,
                                                                              uint64_t        patchPointsTypeId) const
{
    DataChunkPatchPointsInfo* pDataChunkPatchPointsInfo = nullptr;
    HB_ASSERT(executionStage != EXECUTION_STAGE_LAST, "Illegal execution stage");

    auto itr = m_pPatchPointsDataChunksInfo[executionStage].find(patchPointsTypeId);
    if (itr != m_pPatchPointsDataChunksInfo[executionStage].end())
    {
        pDataChunkPatchPointsInfo = itr->second;
    }

    return pDataChunkPatchPointsInfo;
}

uint64_t RecipeStaticInfo::getPatchingPointsDcAmount(eExecutionStage executionStage,
                                                     uint64_t        patchPointsTypeId) const
{
    HB_ASSERT(executionStage != EXECUTION_STAGE_LAST, "Illegal execution stage");

    auto itr = m_pPatchPointsAmount[executionStage].find(patchPointsTypeId);
    if (itr != m_pPatchPointsAmount[executionStage].end())
    {
        return itr->second;
    }

    return 0;
}

const DataChunkPatchPointsInfo* RecipeStaticInfo::getSobjPatchingPointsDcLocation() const
{
    return m_pSobjPatchPointsDataChunksInfo;
}

DataChunkSmPatchPointsInfo* RecipeStaticInfo::refSmPatchingPointsDcLocation() const
{
    return m_pSmPatchPointsDataChunksInfo.get();
}

const DataChunkSmPatchPointsInfo* RecipeStaticInfo::getSmPatchingPointsDcLocation() const
{
    return m_pSmPatchPointsDataChunksInfo.get();
}

void RecipeStaticInfo::clearArbitrationSetCommand()
{
    m_arbSetCmdHostAddress.unset();
}

void RecipeStaticInfo::clearArbitrationReleaseCommand()
{
    m_arbReleaseCmdHostAddress.unset();
}

void RecipeStaticInfo::clearProgramCodeBlobsDeviceAddressDatabase(uint64_t blobsNum)
{
    m_programCodeBlobsToDeviceAddress.clear();
    m_programCodeBlobsToDeviceAddress.resize(blobsNum, {});
}

void RecipeStaticInfo::clearProgramCodeToCpDmaAddressDatabase(uint64_t blobsNum)
{
    m_programCodeBlobsToCpDmaAddress.clear();
    m_programCodeBlobsToCpDmaAddress.resize(blobsNum);
}

void RecipeStaticInfo::deleteStagePatchingPointsDcs(eExecutionStage executionStage)
{
    HB_ASSERT(executionStage != EXECUTION_STAGE_LAST, "Illegal execution stage");

    for (auto& element : m_pPatchPointsDataChunksInfo[executionStage])
    {
        delete element.second;
    }
    m_pPatchPointsDataChunksInfo[executionStage].clear();
    if (executionStage == EXECUTION_STAGE_ENQUEUE && m_pSobjPatchPointsDataChunksInfo != nullptr)
    {
        delete m_pSobjPatchPointsDataChunksInfo;
        m_pSobjPatchPointsDataChunksInfo = nullptr;
    }
}

const patchableBlobOffsetInDc& RecipeStaticInfo::getPatchableBlobOffset(uint64_t blobIndex) const
{
    return m_patchableBlobsOffsetInDc[blobIndex];
}
