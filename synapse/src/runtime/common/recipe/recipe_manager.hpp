#pragma once

#include "synapse_api_types.h"
#include "synapse_common_types.h"
#include "log_manager.h"
#include <deque>
#include <mutex>
#include <atomic>

struct InternalRecipeHandle;
struct recipe_t;
class HabanaGraph;
struct basicRecipeInfo;
struct SignalFromGraphInfo;
/**
 * brief about Recipe Singleton
 *
 * Responsible to hold and handle multi recipe-processors.
 * recipeHandle->recipeProcessor
 */
class RecipeManager
{
public:
    RecipeManager() = default;

    virtual ~RecipeManager() = default;

    static synStatus recipeSerialize(const InternalRecipeHandle* pRecipeHandle, const char* recipeFileName);

    synStatus recipeDeSerialize(InternalRecipeHandle*& rpRecipeHandle, const char* recipeFileName);

    synStatus addRecipeHandleAndCompileGraph(HabanaGraph*                graph,
                                             bool                        isProtobuf,
                                             const CompilationAttribute* compileParams,
                                             uint32_t                    sizeCompileParams,
                                             const char*                 fileName,
                                             const char*                 buildLog,
                                             InternalRecipeHandle*&      rpRecipeHandle);

    static synStatus compileGraph(HabanaGraph*                graph,
                                  bool                        isProtobuf,
                                  const CompilationAttribute* compileParams,
                                  uint32_t                    sizeCompileParams,
                                  const char*                 fileName,
                                  const char*                 buildLog,
                                  InternalRecipeHandle**      ppRecipeHandle);

    bool removeRecipeHandle(InternalRecipeHandle* pRecipeHandle);

    bool removeAllRecipeHandle();

    static void
    logRecipeStats(const basicRecipeInfo& recipeInfo, size_t nbExternalTensors, synapse::LogManager::LogType logType);

    static void dfaLogRecipeInfo(const InternalRecipeHandle& rInternalRecipeHandle);

    static void notifyRecipeLaunchFailure(const InternalRecipeHandle*   pRecipeHandle,
                                          const synLaunchTensorInfoExt* enqueueTensorsInfo,
                                          uint32_t                      enqueueTensorsAmount,
                                          uint32_t                      flags);

    static synStatus recipeGetAttribute(uint64_t*                 retVal,
                                        const synRecipeAttribute* recipeAttr,
                                        const unsigned            querySize,
                                        const synRecipeHandle     recipeHandle);

private:
    // static recipes - not linked to any device
    typedef std::deque<InternalRecipeHandle*> recipeHandlesDB;

    bool addRecipeHandle(InternalRecipeHandle*& rpRecipeHandle);

    synStatus processRecipe(InternalRecipeHandle* pRecipeHandle, const std::string& name);

    void logRecipeStatsHeader(const InternalRecipeHandle* pInternalRecipeHandle, const char* name) const;

    static void printRecipePersistentTensors(const recipe_t& rRecipe);

    static void serializeFailedRecipe(const InternalRecipeHandle* pRecipeHandle);

    std::atomic<uint64_t> m_currentRecipeSeqNum {0};

    // This DB allows recipe's static-parts cleanup
    recipeHandlesDB m_recipes;

    mutable std::mutex m_recipesMutex;

    static constexpr uint64_t           M_S_MAX_DUMP_BAD_RECIPES = 1;
    inline static std::atomic<uint64_t> m_numOfRecipesDumped {0};
};
