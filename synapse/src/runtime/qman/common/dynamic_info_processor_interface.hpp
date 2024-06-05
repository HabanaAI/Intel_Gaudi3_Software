#pragma once

#include "synapse_common_types.h"
#include "recipe.h"
#include "synapse_api_types.h"
#include "recipe_package_types.hpp"

#include <cstdint>
#include <deque>
#include <map>
#include <vector>

// vector of pairs of blobIndex to its device address (in WS)
typedef std::map<uint64_t, uint64_t> staticBlobsDeviceAddresses;

struct basicRecipeInfo;
struct DeviceAgnosticRecipeInfo;
struct QmanEvent;

class CommandSubmissionDataChunks;
class DataChunk;
class DynamicRecipe;
class RecipeStaticInfo;

class IDynamicInfoProcessor
{
public:
    typedef std::deque<CommandSubmissionDataChunks*>  CommandSubmissionDataChunksDB;
    typedef std::vector<CommandSubmissionDataChunks*> CommandSubmissionDataChunksVec;

    virtual ~IDynamicInfoProcessor() = default;

    virtual synStatus enqueue(const synLaunchTensorInfoExt* launchTensorsInfo,
                              uint32_t                      launchTensorsAmount,
                              CommandSubmissionDataChunks*& pCsDataChunks,
                              uint64_t                      scratchPadAddress,
                              uint64_t                      programDataDeviceAddress,
                              uint64_t                      programCodeHandle,
                              uint64_t                      programDataHandle,
                              bool                          programCodeInCache,
                              bool                          programDataInCache,
                              uint64_t                      assertAsyncMappedAddress,
                              uint32_t                      flags,
                              uint64_t&                     csHandle,
                              eAnalyzeValidateStatus        analyzeValidateStatus,
                              uint32_t                      sigHandleId,
                              uint32_t                      sigHandleSobjBaseAddressOffset,
                              eCsDcProcessorStatus&         csDcProcessingStatus) = 0;

    // Releasing and retrieving CS-DC elements of any handle
    // The "current" execution-handle items should be last option -> TBD
    virtual uint32_t releaseCommandSubmissionDataChunks(uint32_t                        numOfElementsToRelease,
                                                        CommandSubmissionDataChunksVec& releasedElements,
                                                        bool                            keepOne) = 0;

    // Release and retrieve a CS-DC element of current execution-handle
    // Todo remove isCurrent option along with CommandSubmissionDataChunks::getExecutionHandle
    virtual CommandSubmissionDataChunks* getAvailableCommandSubmissionDataChunks(eExecutionStage executionStage,
                                                                                 bool            isCurrent) = 0;

    virtual bool
    notifyCsCompleted(CommandSubmissionDataChunks* pCsDataChunks, uint64_t waitForEventHandle, bool csFailed) = 0;

    virtual void incrementExecutionHandle() = 0;

    virtual uint64_t getExecutionHandle() = 0;

    virtual void getProgramCodeHandle(uint64_t& programCodeHandle) const = 0;

    virtual void getProgramDataHandle(uint64_t& programDataHandle) const = 0;

    virtual void setProgramCodeHandle(uint64_t programCodeHandle) = 0;

    virtual void setProgramCodeAddrInWS(uint64_t programCodeAddrInWS) = 0;

    virtual void setProgramDataHandle(uint64_t programDataHandle) = 0;

    virtual bool isAnyInflightCsdc() = 0;

    virtual const char* getRecipeName() = 0;
    virtual uint64_t    getRecipeId() const = 0;

    virtual const basicRecipeInfo&          getRecipeBasicInfo()                    = 0;
    virtual const DeviceAgnosticRecipeInfo& getDevAgnosticInfo()                    = 0;
    virtual std::vector<tensor_info_t>      getDynamicShapesTensorInfoArray() const = 0;

    virtual DynamicRecipe* getDsdPatcher() = 0;

    virtual bool resolveTensorsIndices(std::vector<uint32_t>*&       tensorIdx2userIdx,
                                       const uint32_t                launchTensorsAmount,
                                       const synLaunchTensorInfoExt* launchTensorsInfo) = 0;
};