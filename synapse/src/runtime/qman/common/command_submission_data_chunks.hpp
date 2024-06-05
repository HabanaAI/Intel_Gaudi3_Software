#pragma once

#include "address_range_mapper.hpp"
#include "debug_define.hpp"
#include "graph_compiler/debug_define.hpp"
#include "hal_reader/gaudi1/hal_reader.h"
#include "recipe_package_types.hpp"
#include "runtime/common/recipe/patching/host_address_patching_executer.hpp"
#include "runtime/qman/common/qman_types.hpp"
#include "settable.h"

#include <deque>
#include <drm/habanalabs_accel.h>
#include <limits>
#include <memory>
#include <unordered_map>

class DataChunk;
class CommandSubmission;
class InflightCsParserHelper;

class RecipeProgramBuffer;
typedef std::shared_ptr<RecipeProgramBuffer> SpRecipeProgramBuffer;

struct recipe_t;
struct StagedInfo;

// Define three types of DBs;
// 1) A database of DataChunks that contains the Commands-Buffer
typedef DataChunksDB commandsDataChunksDB;

// 2) A database of DataChunks that contains the CP-DMAs
typedef DataChunksDB cpDmaDataChunksDB;

// 3) A mapping between blobIndex and a database.
struct cpDmaSingleDataChunkInfo
{
    uint64_t hostAddress;
    uint64_t deviceVirtualAddress;
    uint64_t size;  // Multiple consecutive CP-DMAs of a single blob, inside given DataChunk
};

// Aa single blob may be spread over multiple DataChunks (and hence multiple CP-DMA commands)
typedef std::deque<cpDmaSingleDataChunkInfo> cpDmaSingleEngineInfo;
typedef std::vector<cpDmaSingleEngineInfo>   cpDmaDataChunksMappingDB;

#define CS_DC_INVALID_EXECUTION_HANDLE            0
#define CS_DC_RESERVED_EXECUTION_HANDLE           1
#define CS_DC_FIRST_NON_RESERVED_EXECUTION_HANDLE 2

#define INVALID_RECIPE_ID         0xFFFFFFFFFFFFFFFF

enum eCommandSumissionDataChunkType : uint8_t
{
    CS_DC_TYPE_COMPUTE,
    CS_DC_TYPE_MEMCOPY
};

struct HostRangeMapping
{
    uint64_t hostAddress;
    uint64_t mappedAddress;
    uint64_t bufferSize;
};

typedef std::unique_ptr<HostRangeMapping[]> upHostRangeMappingDB;
typedef std::vector<HostRangeMapping>       HostRangeMappingVector;

class CommandSubmissionDataChunks
{
    using eAddressRangeMappingType = AddressRangeMapper::eAddressRangeMappingType;

public:
    // For a non recipe-launch's CS
    CommandSubmissionDataChunks(eCommandSumissionDataChunkType usageType,
                                synDeviceType                  deviceType,
                                size_t                         csDcMappingDbSize);

    CommandSubmissionDataChunks(eCommandSumissionDataChunkType usageType,
                                synDeviceType                  deviceType,
                                size_t                         csDcMappingDbSize,
                                uint64_t                       recipeId,
                                const recipe_t&                rRecipe,
                                InternalRecipeHandle*          pRecipeHandle,
                                eExecutionStage                stage,
                                uint64_t                       maxSectionId,
                                uint64_t                       numOfSectionsToPatch);

    ~CommandSubmissionDataChunks();

    void addSingleProgramBlobsDataChunk(DataChunk* singleDataChunk);

    void addProgramBlobsDataChunks(const commandsDataChunksDB& dataChunks);

    void addCpDmaDataChunks(cpDmaDataChunksDB cpDmaDataChunks);

    void setRecipeProgramBuffer(SpRecipeProgramBuffer pRecipeProgramBuffer);

    const SpRecipeProgramBuffer getRecipeProgramBuffer() const;

    void setProgramDataSubTypeAllocationId(uint64_t subTypeAllocationId) { m_prgDataSubTypeAllocationId = subTypeAllocationId; };

    uint64_t getPrgDataSubTypeAllocationId() { return m_prgDataSubTypeAllocationId; };

    // * REMARK * - It is up for the caller to supply a valid mapping. It is not verified on this class (TODO ?)
    void addSingleCpDmaDataChunksMapping(uint32_t engineId,
                                         uint64_t hostAddress,
                                         uint64_t deviceVirtualAddress,
                                         uint64_t size);

    bool setCommandSubmissionInstance(CommandSubmission* pCommandSubmission);

    CommandSubmission* getCommandSubmissionInstance();

    const CommandSubmission* getCommandSubmissionInstance() const;

    void addWaitForEventHandle(uint64_t waitForEventHandle);

    bool popWaitForEventHandle(uint64_t waitForEventHandle);

    bool getWaitForEventHandle(uint64_t& waitForEventHandle, bool isOldset) const;

    const std::deque<uint64_t>& getAllWaitForEventHandles() const;

    bool isWaitForEventHandleSet() const;

    bool destroyCommandSubmission();

    // May throw out_of_range exception
    const cpDmaSingleEngineInfo& getSingleEngineCpDmaMapping(uint32_t engineId) const;

    commandsDataChunksDB&       getCommandsBufferDataChunks();
    const commandsDataChunksDB& getCommandsBufferDataChunks() const;

    void clearCommandsBufferDataChunks();

    const cpDmaDataChunksMappingDB& getBlobsCpDmaMapping() const;

    cpDmaDataChunksDB&       getCpDmaDataChunks();
    const cpDmaDataChunksDB& getCpDmaDataChunks() const;

    eExecutionStage getExecutionStage() const { return m_stage; }
    inline bool     isCompute() const { return (getUsageType() == CS_DC_TYPE_COMPUTE); }
    inline bool     isActivate() const { return (isCompute() && (getExecutionStage() == EXECUTION_STAGE_ACTIVATE)); }
    inline bool     isEnqueue() const { return (isCompute() && (getExecutionStage() == EXECUTION_STAGE_ENQUEUE)); }
    uint64_t getRecipeId() const;

    inline InternalRecipeHandle* getRecipeHandle() const { return m_pRecipeHandle; }

    void     setExecutionHandle(uint64_t executionHandle);
    uint64_t getExecutionHandle() const;

    eCommandSumissionDataChunkType getUsageType() const;

    uint64_t getProgramCodeHandle() const;

    void setProgramCodeHandle(uint64_t programCodeHandle);

    void getProgramDataHandle(uint64_t& prgDataHandle) const;

    void setProgramDataHandle(uint64_t prgDataHandle);

    bool isNewProgramDataHandle(uint64_t newPrgDataHandle) const;

    void getScratchpadHandle(uint64_t& scratchpadHandle) const;

    void setScratchpadHandle(uint64_t scratchpadHandle);

    bool isNewScratchpadHandle(uint64_t newScratchpadHandle) const;

    void setSobjAddress(uint32_t sobjAddress);

    void getSobjAddress(uint32_t& sobjAddress) const;

    bool isNewSobjAddress(uint32_t newSobjAddress) const;

    eCsDcExecutionType getCsDcExecutionType() const;

    void setCsDcExecutionType(eCsDcExecutionType csDcExecutionType);

    void setDataChunksStagesInfo(const std::vector<uint32_t>& stagesNodes, const std::vector<StagedInfo>& stagesInfo);

    const std::vector<uint32_t>&   getDataChunksStagesNodes();
    const std::vector<StagedInfo>& getDataChunksStagesInfo();

    patching::HostAddressPatchingInformation& getHostAddressInDcInformation();

    void setCopiedToDc() { m_isCopiedToDc = true; };
    bool isCopiedToDc() const { return m_isCopiedToDc; };

    void hostRangeMappingReserveSize(uint64_t size);

    void hostRangeMappingAddEntry(uint64_t hostAddress, uint64_t mappedAddress, uint64_t bufferSize);

    const HostRangeMappingVector& getHostRangeMapping() const;

    inline synDeviceType getDeviceType() const { return m_deviceType; };

    bool containsHandle(uint64_t csSeq);
    std::string dfaGetCsList();

    // Overrides AddressRangeMapper
    // Used when PRG-Code is in Cache [imported from the (launch-agnostic) static-info]
    void setProgramCodeBlocksMappingInCache(const AddressRangeMapper& programCodeBlocksMapping);

    // Overrides AddressRangeMapper
    // Used when PRG-Code is in Workspace
    bool setProgramCodeBlocksMappingInWorkspace(uint64_t                 handle,
                                                uint64_t                 mappedAddress,
                                                uint64_t                 size,
                                                eAddressRangeMappingType mappingType);

    const AddressRangeMapper& getProgramCodeBlocksMapping() const;

    void updateWithActivatePqEntries(InflightCsParserHelper& csParserHelper);

private:
    synDeviceType m_deviceType;

    const uint64_t        m_recipeId;
    const recipe_t* m_pRecipe;
    InternalRecipeHandle* m_pRecipeHandle;

    uint64_t m_executionHandle;

    commandsDataChunksDB     m_commandsBuffersChunksDatabase;
    cpDmaDataChunksDB        m_cpDmaChunksDatabase;
    cpDmaDataChunksMappingDB m_cpDmaDataChunksMappingDatabase;  // engine->cpDma info

    CommandSubmission*   m_pCommandSubmission;
    std::deque<uint64_t> m_waitForEventHandle;

    uint64_t m_cpDmaCommandSize;
    // Each CSDC has it's execution handle and the prgCodeHandle, so when in future
    // We will want to reuse the CSDC we would know what kind of reuse we need to do
    uint64_t m_ExecutionHandle;
    uint64_t m_prgCodeHandle;
    uint64_t m_prgDataHandle;
    uint64_t m_scratchpadHandle;
    uint32_t m_sfgSobjLowPartAddress;

    eCsDcExecutionType             m_csDcExecutionType;
    const eCommandSumissionDataChunkType m_usageType;

    const eExecutionStage m_stage;
    bool m_isCopiedToDc;

    // Used for setting mappings that contains packets, while not part of the DCs of this CS-DC
    // Specifically, used for copy operations of program-code buffers
    HostRangeMappingVector m_hostRangeMapping;

    std::vector<uint32_t>   m_stagesNodes;
    std::vector<StagedInfo> m_stagesInfo;

    AddressRangeMapper m_programCodeBlocksMapping;

    SpRecipeProgramBuffer m_spRecipeProgramBuffer;

    uint64_t m_prgDataSubTypeAllocationId;

    patching::HostAddressPatchingInformation m_hostAddressPatchingOnDataChunksInfo;
};

// TODO - at the moment, valid only for Gaudi
//
// pCommandSubmissionInput   - Used when the CS-DC is not yet containing the CS itself
// amountOfEnginesInArbGroup - In case zero (0), ARB-validation is disabled
bool checkForCsUndefinedOpcode(const CommandSubmissionDataChunks* pCommandSubmissionDataChunks,
                               const CommandSubmission*           pCommandSubmissionInput,
                               uint32_t                           amountOfEnginesInArbGroup,
                               bool                               returnUponFailure = false);