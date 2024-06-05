#pragma once

#include "address_range_mapper.hpp"
#include "settable.h"
#include "synapse_api_types.h"
#include "recipe_package_types.hpp"
#include "runtime/common/recipe/patching/define.hpp"
#include "runtime/qman/common/qman_types.hpp"
#include "utils.h"

#include <map>
#include <unordered_map>
#include <vector>

// This class holds memory information about a given recipe, that are in recipe-level.
// Specifically:
//  - Normal-mode
//      Program-Data - Create a section and map it once per recipe
//  - Gaudi-Demo
//      Program-Data - Mapping the recipe's (program-data) blobs
//      Program-Code - Mapping the recipe's (program-code) blobs

// Remark (stream level information):
//  - Normal-mode:
//      Program-Code - Patched per stream and mapped (using DataChunks)
//      CP-DMA       - Pointing to the above Program-Code (mapped - using DataChunks) host addresses
//      CS           - Pointing to the above mapped CP-DMA addresses
//  - Gaudi-Demo
//      CP-DMA       - Pointing to the device-address that the program-code had been downloaded into
//      CS           - Pointing to the above mapped CP-DMA addresses

// This class is NOT thread-safe. It is up for the user of this class to perform any thread-safeness operation required

struct blob_t;
struct data_chunk_sm_patch_point_t;
struct recipe_t;
struct shape_plane_graph_t;
struct sm_patch_point_t;
struct DataChunkPatchPointsInfo;
struct DataChunkSmPatchPointsInfo;

using namespace patching;


struct patchableBlobOffsetInDc
{
    uint32_t offsetInDc;
    uint32_t dcIndex;
};

using BlobCpDmaHostAddresses = std::vector<uint64_t>;

class RecipeStaticInfo
{
public:
    RecipeStaticInfo();

    virtual ~RecipeStaticInfo();

    static const unsigned EXTERNAL_QMANS_AMOUNT = gaudi_queue_id::GAUDI_QUEUE_ID_SIZE;

    void initDBs(const recipe_t& rRecipe);

    void clearDBs();

    // Simple methods that informs whether the information had been initialized or not
    // TODO - add assertion (Error-handling seems to big of an overhead) that block update after initializing
    void setInitialized(bool isInitialized);

    bool isInitialized();

    void setArbitrationSetCommand(uint64_t arbSetCmdHostAddress);

    void setArbitrationReleaseCommand(uint64_t arbReleaseCmdHostAddress);

    void setCpDmaChunksAmount(eExecutionStage stage, CachedAndNot cpDmaChunksAmount);

    // Will hold the final device addresses - whether it is in cache or not
    bool addProgramCodeBlockMapping(uint64_t handle, uint64_t mappedAddress, uint64_t size);

    void setProgramDataMappedAddress(blobAddressType mappedAddr);

    void setProgramCodeMappedAddress(blobAddressType mappedAddr);

    bool getProgramDataMappedAddress(blobAddressType& mappedAddr) const;

    bool getProgramCodeMappedAddress(blobAddressType& mappedAddr) const;

    void clearProgramDataMappedAddress();

    void clearProgramCodeMappedAddress();

    // The following set methods and their equivalent get methods will be used in GaudiDemo,
    // where the PRG-Code is static in the device.
    //
    // This is also a future solution for the non-patchable program-code,
    // when that will be handled by Synapse (and not affected by workspace-address)

    // blobDeviceAddress - Address in the device and not mapped address
    void addProgramCodeBlobDeviceAddress(uint64_t        blobIdx,
                                         blobAddressType blobAddress,
                                         uint64_t        blobDeviceAddress,
                                         uint64_t        blobPartialSize);

    void setProgramCodeBlobCpDmaAddress(uint64_t blobIdx, uint64_t cpDmaPacketHostAddress);

    void setPatchingPointsDcLocation(eExecutionStage           executionStage,
                                     uint64_t                  patchPointsTypeId,
                                     DataChunkPatchPointsInfo* pPatchPointsDataChunksInfo,
                                     uint64_t                  amountOfPps, // Not required for SOBJ' PPs
                                     bool                      isSobj);

    bool allocateSmPatchingPointsDcLocation(uint64_t patchPointsAmount, uint32_t dcSizeCommand);

    bool getArbitrationSetHostAddress(uint64_t& arbSetHostAddress) const;

    bool getArbitrationReleaseHostAddress(uint64_t& arbReleaseCmdHostAddress) const;

    bool getCpDmaChunksAmount(eExecutionStage stage, uint64_t& cpDmaChunksAmount, bool inCache) const;

    const AddressRangeMapper& getProgramCodeBlocksMapping() const;

    bool getProgramCodeBlobCpDmaAmount(uint64_t blobIdx, uint8_t& cpDmaAmount);

    bool getProgramCodeBlobCpDmaAddress(uint64_t blobIdx, const BlobCpDmaHostAddresses*& cpDmaPacketHostAddress) const;

    const std::vector<HostAndDevAddr>& getProgramCodeBlobsToDeviceAddress() const;

    const std::array<DataChunksDB, EXTERNAL_QMANS_AMOUNT>& retrieveExternalQueueBlobsDataChunks() const;

    const DataChunkPatchPointsInfo* getPatchingPointsDcLocation(eExecutionStage executionStage,
                                                                uint64_t        patchPointsTypeId) const;

    uint64_t getPatchingPointsDcAmount(eExecutionStage executionStage,
                                       uint64_t        patchPointsTypeId) const;

    const DataChunkPatchPointsInfo* getSobjPatchingPointsDcLocation() const;

    const DataChunkSmPatchPointsInfo* getSmPatchingPointsDcLocation() const;

    DataChunkSmPatchPointsInfo* refSmPatchingPointsDcLocation() const;

    void clearArbitrationSetCommand();

    void clearArbitrationReleaseCommand();

    void clearProgramCodeBlobsDeviceAddressDatabase(uint64_t blobsNum);

    void clearProgramCodeToCpDmaAddressDatabase(uint64_t blobsNum);

    void clearProgramCodeBlockMapping();

    void deleteStagePatchingPointsDcs(eExecutionStage executionStage);

    void setCpDmaStaticBlobsBuffer(char* pCpDmaBufferAddress);
    bool getCpDmaStaticBlobsBuffer(char*& rpCpDmaBufferAddress) const;
    void clearCpDmaStaticBlobsBuffer();

    std::vector<patchableBlobOffsetInDc>& getPatchableBlobsOffsetsDB() { return m_patchableBlobsOffsetInDc; }
    const patchableBlobOffsetInDc&        getPatchableBlobOffset(uint64_t blobIndex) const;

private:
    using blobAddrAndblobCpDmaHostAddresses = BlobCpDmaHostAddresses;

    bool m_isInitialized;

    Settable<uint64_t> m_arbSetCmdHostAddress;
    Settable<uint64_t> m_arbReleaseCmdHostAddress;

    // Chunks amount is affected from the chunk-size
    // The chunks-size is define upon recipe static processing
    Settable<CachedAndNot> m_cpDmaChunksAmount[EXECUTION_STAGE_LAST];

    AddressRangeMapper m_programCodeBlocksMapping;

    Settable<blobAddressType> m_programDataMappedAddress;
    Settable<blobAddressType> m_programCodeMappedAddress;

    Settable<char*> m_pCpDmaBufferAddress;

    // blobAddr->map(blob's part (device address) to the size of this part)

    std::vector<HostAndDevAddr>                    m_programCodeBlobsToDeviceAddress;
    std::vector<blobAddrAndblobCpDmaHostAddresses> m_programCodeBlobsToCpDmaAddress;
    std::vector<uint64_t>                          m_programCodeBlobIndicesToCpDmaMappedAddress;

    std::array<DataChunksDB, EXTERNAL_QMANS_AMOUNT> m_externalQueueDataChunks;

    std::unordered_map<uint64_t /*SectionGroupType*/, DataChunkPatchPointsInfo*>
                                                m_pPatchPointsDataChunksInfo[EXECUTION_STAGE_LAST];
    // Intentionally split from the above DB
    std::unordered_map<uint64_t /*SectionGroupType*/, uint64_t>
                                                m_pPatchPointsAmount[EXECUTION_STAGE_LAST];

    DataChunkPatchPointsInfo*                   m_pSobjPatchPointsDataChunksInfo = nullptr;
    std::unique_ptr<DataChunkSmPatchPointsInfo> m_pSmPatchPointsDataChunksInfo;

    std::unique_ptr<uint8_t[]> m_patchingBlobsBuffer;
    std::unique_ptr<blob_t[]>  m_blobs;

    std::vector<patchableBlobOffsetInDc> m_patchableBlobsOffsetInDc;
};
