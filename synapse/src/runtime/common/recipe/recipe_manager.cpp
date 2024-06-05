#include "recipe_manager.hpp"
#include "defenders.h"
#include "synapse_runtime_logging.h"
#include "recipe_handle_impl.hpp"
#include "recipe_allocator.h"
#include "recipe_serializer.hpp"
#include "recipe_logger.hpp"
#include "habana_graph.h"
#include "device_agnostic_recipe_processor.hpp"
#include "habana_global_conf_runtime.h"
#include "global_statistics.hpp"
#include <sys/stat.h>  //  stat operation

#include <algorithm>
#include <deque>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <queues/queue_compute_utils.hpp>
#include "runtime/common/recipe/recipe_utils.hpp"

// #define PRINT_PERSISTENT_TENSORS

synStatus RecipeManager::processRecipe(InternalRecipeHandle* pRecipeHandle, const std::string& name)
{
#ifdef PRINT_PERSISTENT_TENSORS
    printRecipePersistentTensors(*pRecipeHandle->basicRecipeHandle.recipe);
#endif

    RECIPE_STATS_START(agnosticTime);
    const synStatus status = DeviceAgnosticRecipeProcessor::process(pRecipeHandle->basicRecipeHandle,
                                                                    pRecipeHandle->deviceAgnosticRecipeHandle);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_RECIPE,
                "{}: DeviceAgnosticRecipeProcessor failed processing {:#x}",
                HLLOG_FUNC,
                TO64(pRecipeHandle));
        RecipeLogger::dumpRecipe(pRecipeHandle, true, false, synapse::LogManager::LogType::SYN_API);
        bool operStatus = removeRecipeHandle(pRecipeHandle);
        if (!operStatus)
        {
            LOG_ERR(SYN_RECIPE, "{}: Failed to remove recipe handle {:#x}", HLLOG_FUNC, TO64(pRecipeHandle));
        }
        return status;
    }
    logRecipeStats(pRecipeHandle->basicRecipeHandle,
                   pRecipeHandle->deviceAgnosticRecipeHandle.m_signalFromGraphInfo.getNumberOfExternalTensors(),
                   synapse::LogManager::LogType::RECIPE_STATS);

    if (GCFG_RUNTIME_DUMP_RECIPE.value())
    {
        RecipeLogger::dumpRecipe(pRecipeHandle, true /*log*/, false /*screen*/, synapse::LogManager::LogType::SYN_API);
    }

    LOG_RECIPE_STATS("agnostic proc (total) {:10d} ns", RECIPE_STATS_END(agnosticTime));

    return synSuccess;
}

bool RecipeManager::addRecipeHandle(InternalRecipeHandle*& rpRecipeHandle)
{
    // Create new recipe handle
    InternalRecipeHandle::createRecipeHandle(rpRecipeHandle);

    std::unique_lock<std::mutex> lock(m_recipesMutex);
    m_recipes.push_back(rpRecipeHandle);
    return true;
}

bool RecipeManager::removeRecipeHandle(InternalRecipeHandle* pRecipeHandle)
{
    CHECK_POINTER(SYN_RECIPE, pRecipeHandle, "pRecipeHandle", false);

    LOG_RECIPE_STATS("Removing ===== recipe {} InternalRecipeHandle {:x} =====",
                     pRecipeHandle->recipeSeqNum,
                     TO64(pRecipeHandle));

    bool removed = false;
    {
        std::unique_lock<std::mutex> lock(m_recipesMutex);

        auto it = std::find(m_recipes.begin(), m_recipes.end(), pRecipeHandle);
        if (it != m_recipes.end())
        {
            m_recipes.erase(it);
            removed = true;
        }
    }
    if (removed)
    {
        InternalRecipeHandle::destroyRecipeHandle(pRecipeHandle);
        return true;
    }

    // Todo Return false as it indicates inconsistency
    LOG_ERR(SYN_RECIPE, "{} pRecipeHandle {:#x} was not found", HLLOG_FUNC, TO64(pRecipeHandle));
    return true;
}

bool RecipeManager::removeAllRecipeHandle()
{
    std::unique_lock<std::mutex> lock(m_recipesMutex);
    for (InternalRecipeHandle* pRecipeHandle : m_recipes)
    {
        InternalRecipeHandle::destroyRecipeHandle(pRecipeHandle);
    }
    m_recipes.clear();
    return true;
}

synStatus RecipeManager::recipeSerialize(const InternalRecipeHandle* pRecipeHandle, const char* recipeFileName)
{
    VERIFY_IS_NULL_POINTER(SYN_RECIPE, pRecipeHandle, "pRecipeHandle");
    VERIFY_IS_NULL_POINTER(SYN_RECIPE, recipeFileName, "recipeFileName");

    const recipe_t*            pRecipe = pRecipeHandle->basicRecipeHandle.recipe;
    const shape_plane_graph_t* pSpr    = pRecipeHandle->basicRecipeHandle.shape_plan_recipe;

    ParamsManager params;
    params.setFileName(recipeFileName, false);

    synStatus status = RecipeSerializer::serialize(pRecipe, pSpr, &params);
    if (status != synSuccess)
    {
        LOG_ERR(SYN_RECIPE, "{}: Can not serialize file {}", HLLOG_FUNC, recipeFileName);
        return status;
    }

    return synSuccess;
}

synStatus RecipeManager::recipeDeSerialize(InternalRecipeHandle*& rpRecipeHandle, const char* recipeFileName)
{
    LOG_TRACE(SYN_RECIPE, "{}", HLLOG_FUNC);

    VERIFY_IS_NULL_POINTER(SYN_RECIPE, recipeFileName, "recipeFileName");

    InternalRecipeHandle* pInternalRecipeHandle;
    bool                  operStatus = addRecipeHandle(pInternalRecipeHandle);
    if (!operStatus)
    {
        LOG_ERR(SYN_RECIPE, "{}: Failed to add recipe handle into Recipe-Singleton", HLLOG_FUNC);
        return synFail;
    }

    basicRecipeInfo& basicRecipeHandle = pInternalRecipeHandle->basicRecipeHandle;
    basicRecipeHandle.recipeAllocator  = new RecipeAllocator();
    basicRecipeHandle.recipe           = (recipe_t*)basicRecipeHandle.recipeAllocator->allocate(sizeof(recipe_t));

    memset(pInternalRecipeHandle->basicRecipeHandle.recipe, 0, sizeof(recipe_t));
    ParamsManager params;
    params.setFileName(recipeFileName, false);
    RecipeSerializer recipeSerializer;

    synStatus status = synSuccess;
    try
    {
        status = RecipeSerializer::deserialize(basicRecipeHandle.recipe,
                                               basicRecipeHandle.shape_plan_recipe,
                                               &params,
                                               basicRecipeHandle.recipeAllocator);
    }
    catch (...)
    {
        LOG_ERR(SYN_RECIPE,
                "{}: Can not deserialize file {} recipeSerializer.deserialize threw exception",
                HLLOG_FUNC,
                recipeFileName);
        basicRecipeHandle.recipeAllocator->freeAll();
        status = synFail;
    }

    if (status != synSuccess)
    {
        LOG_ERR(SYN_RECIPE, "{}: Can not deserialize file {}", HLLOG_FUNC, recipeFileName);
        operStatus = removeRecipeHandle(pInternalRecipeHandle);
        if (!operStatus)
        {
            LOG_ERR(SYN_RECIPE, "{}: Failed to remove recipe handle {:#x}", HLLOG_FUNC, TO64(pInternalRecipeHandle));
        }
        return status;
    }

    pInternalRecipeHandle->recipeSeqNum = ++m_currentRecipeSeqNum;

    logRecipeStatsHeader(pInternalRecipeHandle, pInternalRecipeHandle->basicRecipeHandle.recipe->name);
    LOG_RECIPE_STATS("Recipe was deserialized from file {}", recipeFileName ? recipeFileName : "NO-FILE-NAME");

    status = processRecipe(pInternalRecipeHandle, "unknown(deserialize)");
    if (status != synSuccess)
    {
        LOG_ERR(SYN_RECIPE, "{}: processRecipe failed {:#x}", HLLOG_FUNC, TO64(pInternalRecipeHandle));
        return status;
    }

    rpRecipeHandle = pInternalRecipeHandle;

    return synSuccess;
}

synStatus RecipeManager::addRecipeHandleAndCompileGraph(HabanaGraph*                graph,
                                                        bool                        isProtobuf,
                                                        const CompilationAttribute* compileParams,
                                                        uint32_t                    sizeCompileParams,
                                                        const char*                 fileName,
                                                        const char*                 buildLog,
                                                        InternalRecipeHandle*&      rpRecipeHandle)
{
    InternalRecipeHandle* pInternalRecipeHandle = nullptr;
    bool                  operStatus            = addRecipeHandle(pInternalRecipeHandle);
    if (!operStatus)
    {
        LOG_ERR(SYN_RECIPE, "{}: Failed to add recipe handle into Recipe-Singleton", HLLOG_FUNC);
        return synFail;
    }

    RECIPE_STATS_START(compileTime);
    synStatus status = compileGraph(graph, false, nullptr, 0, fileName, buildLog, &pInternalRecipeHandle);
    uint64_t  diffNs = RECIPE_STATS_END(compileTime);

    if (status != synSuccess)
    {
        LOG_ERR(SYN_RECIPE, "{}: Can not compile", HLLOG_FUNC);
        operStatus = removeRecipeHandle(pInternalRecipeHandle);
        if (!operStatus)
        {
            LOG_ERR(SYN_RECIPE, "{}: Failed to remove recipe handle {:#x}", HLLOG_FUNC, TO64(pInternalRecipeHandle));
        }
        return status;
    }

    pInternalRecipeHandle->recipeSeqNum = ++m_currentRecipeSeqNum;

    logRecipeStatsHeader(pInternalRecipeHandle, graph->getRecipeName().c_str());
    LOG_RECIPE_STATS("Compilation   {:10d} ns", diffNs);

    status = processRecipe(pInternalRecipeHandle, graph->getRecipeName());
    if (status != synSuccess)
    {
        LOG_ERR(SYN_RECIPE, "{}: processRecipe failed {:#x}", HLLOG_FUNC, TO64(pInternalRecipeHandle));
        return status;
    }

    rpRecipeHandle = pInternalRecipeHandle;

    return synSuccess;
}

void RecipeManager::logRecipeStatsHeader(const InternalRecipeHandle* pInternalRecipeHandle, const char* name) const
{
    uint64_t               recipeSeqNum = pInternalRecipeHandle->recipeSeqNum;
    const basicRecipeInfo& recipeInfo   = pInternalRecipeHandle->basicRecipeHandle;

    LOG_RECIPE_STATS("===== recipe {} name {} InternalRecipeHandle {:x} recipe {:x} shape_plan_recipe {:x} =====",
                     recipeSeqNum,
                     name ? name : "NO-NAME",
                     TO64(pInternalRecipeHandle),
                     TO64(recipeInfo.recipe),
                     TO64(recipeInfo.shape_plan_recipe));
}

synStatus RecipeManager::compileGraph(HabanaGraph*                graph,
                                      bool                        isProtobuf,
                                      const CompilationAttribute* compileParams,
                                      uint32_t                    sizeCompileParams,
                                      const char*                 fileName,
                                      const char*                 buildLog,
                                      InternalRecipeHandle**      ppRecipeHandle)
{
    LOG_TRACE(SYN_RECIPE, "{} thread ID {:#x}", HLLOG_FUNC, TO64(pthread_self()));
    if (buildLog != nullptr)
    {
        SET_LOGGER_SINK(GC, std::string(buildLog), USER_LOG_LEVEL, LOG_SIZE, LOG_AMOUNT);
    }

    if (graph == nullptr)
    {
        LOG_ERR(SYN_RECIPE, "{}: Current graph is nullptr", HLLOG_FUNC);
        return synFail;
    }

    graph->setRecipeName(fileName ? fileName : "");

    graph->dumpGraphToJson(graph_serializer::GraphState::PRE_COMPILE);

    bool ret = true;

    // measure compilation time only if logger is in debug level
    if (LOG_LEVEL_AT_LEAST_DEBUG(SYN_RECIPE))
    {
        graph->m_timer.start(graph->getRecipeName());
        ret = graph->compile();
        graph->m_timer.stop(graph->getRecipeName());
        LOG_DEBUG(SYN_RECIPE,
                  "{}: compilation time: {}",
                  graph->getRecipeName(),
                  graph->m_timer.getTotalTimeStr(graph->getRecipeName()));
    }
    else
    {
        ret = graph->compile();
    }

    graph->dumpTpcNodesDataToJson();

    if (!ret)
    {
        LOG_ERR(SYN_RECIPE, "{}: Can not compile graph", HLLOG_FUNC);
        return synFail;
    }

    graph->dumpGraphToJson(graph_serializer::GraphState::POST_COMPILE);

    if (fileName != nullptr)
    {
        if (!isProtobuf)
        {
            HB_ASSERT_PTR(ppRecipeHandle);

            basicRecipeInfo& basicRecipeHandle = (*ppRecipeHandle)->basicRecipeHandle;

            RecipeAllocator* compositeTemplateAllocator = graph->consumeEagerCompositeTemplateRecipeAllocator();
            basicRecipeHandle.recipeAllocator =
                compositeTemplateAllocator != nullptr ? compositeTemplateAllocator : new RecipeAllocator {};

            basicRecipeHandle.recipe = graph->serializeDataPlane(basicRecipeHandle.recipeAllocator);
            if (basicRecipeHandle.recipe == nullptr)
            {
                LOG_ERR(SYN_API, "{}: Failed to serializeDataPlan", HLLOG_FUNC);
                return synFail;
            }

            if (graph->isDynamicShape())
            {
                basicRecipeHandle.shape_plan_recipe = graph->serializeShapePlane(basicRecipeHandle.recipeAllocator);
            }

            ETL_DEBUG(EVENT_LOGGER_LOG_TYPE_CHECK_OPCODES,
                      SYN_RECIPE,
                      "Recipe Name {} Recipe ID {:#x}",
                      graph->getRecipeName(),
                      (uint64_t)basicRecipeHandle.recipe);
        }
        else
        {
            LOG_INFO(SYN_RECIPE, "Requested to save compiled output to file {}", fileName);

            std::stringstream input;
            ParamsManager     params;
            params.setFileName(fileName, true);
            ret = graph->serialize(input, &params);
            if (!ret)
            {
                LOG_ERR(SYN_RECIPE, "{}: Can not serialize file {}", HLLOG_FUNC, fileName);
                return synFail;
            }

            std::ofstream file(fileName);
            file << input.str();
            file.close();
            params.saveToDisk();
        }
    }
    return synSuccess;
}

/**
 * logs recipe stats in info level
 * @param recipeInfo
 * @param logType which log will be used
 */
void RecipeManager::logRecipeStats(const basicRecipeInfo&       recipeInfo,
                                   size_t                       nbExternalTensors,
                                   synapse::LogManager::LogType logType)
{
    int logLevel = SPDLOG_LEVEL_INFO;

    // if the given log level is higher than info, we can skip logging
    if (!log_level_at_least(logType, logLevel))
    {
        return;
    }
    const recipe_t* recipe = recipeInfo.recipe;
    SYN_LOG(logType, logLevel, SEPARATOR_STR);
    SYN_LOG(logType, logLevel, "| Recipe stats:");
    SYN_LOG(logType, logLevel, "| program_data_blobs_size:      {:#x}", recipe->program_data_blobs_size);
    SYN_LOG(logType, logLevel, "| execution_blobs_buffer_size:  {:#x}", recipe->execution_blobs_buffer_size);
    SYN_LOG(logType, logLevel, "| patching_blobs_buffer_size:   {:#x}", recipe->patching_blobs_buffer_size);
    SYN_LOG(logType, logLevel, "| dynamic_blobs_buffer_size:    {:#x}", recipe->dynamic_blobs_buffer_size);
    SYN_LOG(logType, logLevel, "| sections_nr:                  {:#x}", recipe->sections_nr);
    SYN_LOG(logType, logLevel, "| section_ids_nr:               {:#x}", recipe->section_ids_nr);
    SYN_LOG(logType, logLevel, "| section_groups_nr:            {:#x}", recipe->section_groups_nr);
    SYN_LOG(logType, logLevel, "| node_nr:                      {:#x}", recipe->node_nr);
    SYN_LOG(logType, logLevel, "| h2di_tensors_nr:              {:#X}", recipe->h2di_tensors_nr);

    int numOfJobs = recipe->arc_jobs_nr;
    for (int i = 0; i < numOfJobs; i++)
    {
        SYN_LOG(logType,
                logLevel,
                "| arc job dynamic/static:       {:#x} / {:#x} (for {})",
                recipe->arc_jobs[i].dynamic_ecb.cmds_size,
                recipe->arc_jobs[i].static_ecb.cmds_size,
                recipe->arc_jobs[i].logical_engine_id);
    }

    SYN_LOG(logType, logLevel, "| persist_tensors_nr:           {:#x}", recipe->persist_tensors_nr);
    SYN_LOG(logType, logLevel, "| external tensors nr           {:#x}", nbExternalTensors);
    SYN_LOG(logType, logLevel, "| permute_tensors_views_nr:     {:#x}", recipe->permute_tensors_views_nr);
    SYN_LOG(logType, logLevel, "| patch_points_nr:              {:#x}", recipe->patch_points_nr);
    SYN_LOG(logType, logLevel, "| activate_patch_points_nr      {:#x}", recipe->activate_patch_points_nr);
    SYN_LOG(logType, logLevel, "| blobs_nr:                     {:#x}", recipe->blobs_nr);

    shape_plane_graph_t* spr = recipeInfo.shape_plan_recipe;
    if (spr)
    {
        SYN_LOG(logType, logLevel, "| shape_tensors_list_nr:        {:#x}", spr->shape_tensors_list_nr);
        SYN_LOG(logType, logLevel, "| sp_node_nr:                   {:#x}", spr->sp_node_nr);
        SYN_LOG(logType, logLevel, "| sp_tensors_nr:                {:#x}", spr->sp_tensors_nr);
    }
    SYN_LOG(logType, logLevel, SEPARATOR_STR);
}

void RecipeManager::dfaLogRecipeInfo(const InternalRecipeHandle& rInternalRecipeHandle)
{
    size_t numExternalTensors =
        rInternalRecipeHandle.deviceAgnosticRecipeHandle.m_signalFromGraphInfo.getNumberOfExternalTensors();
    const basicRecipeInfo& recipeInfo = rInternalRecipeHandle.basicRecipeHandle;
    recipe_t*              recipe     = recipeInfo.recipe;
    const char*            recipeName = recipe->name ? recipe->name : "NO-RECIPE-NAME";

    LOG_INFO(SYN_DEV_FAIL, SEPARATOR_STR);
    LOG_INFO(SYN_DEV_FAIL, "| Recipe name:                  {}", recipeName);
    LOG_INFO(SYN_DEV_FAIL, "| ID:                           {:#x}", TO64(recipe));
    LOG_INFO(SYN_DEV_FAIL, "| seqNumber:                    {}", rInternalRecipeHandle.recipeSeqNum);
    LOG_INFO(SYN_DEV_FAIL, "| Num of successful launches:   {}", recipeInfo.recipeStats.numbSuccessfulLaunch.load());
    LOG_INFO(SYN_DEV_FAIL, "| Shape plan recipe:            {:#x}", TO64(recipeInfo.shape_plan_recipe));
    LOG_INFO(SYN_DEV_FAIL, SEPARATOR_STR);

    RecipeManager::logRecipeStats(recipeInfo, numExternalTensors, synapse::LogManager::LogType::SYN_DEV_FAIL);
}

void RecipeManager::serializeFailedRecipe(const InternalRecipeHandle* pRecipeHandle)
{
    if (GCFG_LOG_LAUNCH_INFO_UPON_FAILURE.value() == false)
    {
        return;
    }
    uint64_t dumpSoFar = m_numOfRecipesDumped++;

    if (dumpSoFar >= M_S_MAX_DUMP_BAD_RECIPES)
    {
        LOG_INFO(SYN_API, "{}: Already dumped {} bad recipes, skipping", HLLOG_FUNC, M_S_MAX_DUMP_BAD_RECIPES);
        return;
    }

    HB_ASSERT_PTR(pRecipeHandle);
    const basicRecipeInfo& rBasicRecipeHandle = pRecipeHandle->basicRecipeHandle;
    HB_ASSERT_PTR(rBasicRecipeHandle.recipe);
    const recipe_t& rRecipe = *rBasicRecipeHandle.recipe;

    std::string fullFilePath;
    synapse::LogManager::getLogsFolderPath(fullFilePath);

    fullFilePath += "/faulty_recipe";
    if (rRecipe.name != nullptr)
    {
        fullFilePath += "_";
        fullFilePath += rRecipe.name;
    }

    // Ensure only done once
    struct stat       buffer;
    uint32_t          fileCounter = 0;
    std::stringstream specificFilePath;
    do
    {
        specificFilePath.str(std::string());
        specificFilePath << fullFilePath << "_" << fileCounter;

        if (stat(specificFilePath.str().c_str(), &buffer) != 0)
        {
            break;
        }

        fileCounter++;
    } while (1);

    LOG_ERR_T(SYN_API,
              "{}: Failed to launch. Serializing recipe {} into {}",
              HLLOG_FUNC,
              rRecipe.name ? rRecipe.name : "Recipe-name-is-null",
              specificFilePath.str());
    recipeSerialize(pRecipeHandle, specificFilePath.str().c_str());
}

void RecipeManager::notifyRecipeLaunchFailure(const InternalRecipeHandle*   pRecipeHandle,
                                              const synLaunchTensorInfoExt* enqueueTensorsInfo,
                                              uint32_t                      enqueueTensorsAmount,
                                              uint32_t                      flags)
{
    serializeFailedRecipe(pRecipeHandle);

    const basicRecipeInfo& rBasicRecipeHandle = pRecipeHandle->basicRecipeHandle;
    const recipe_t&        rRecipe            = *rBasicRecipeHandle.recipe;
    LOG_ERR(SYN_API, "Failed recipe is{} DSD", rBasicRecipeHandle.shape_plan_recipe ? "" : " not");
    printRecipePersistentTensors(rRecipe);

    LOG_ERR(SYN_API, "---User tensors---");
    QueueComputeUtils::logLaunchTensors(enqueueTensorsInfo, enqueueTensorsAmount, HLLOG_LEVEL_ERROR, flags);
}

synStatus RecipeManager::recipeGetAttribute(uint64_t*                 retVal,
                                            const synRecipeAttribute* recipeAttr,
                                            const unsigned            querySize,
                                            const synRecipeHandle     recipeHandle)
{
    return DeviceAgnosticRecipeProcessor::recipeGetAttribute(retVal, recipeAttr, querySize, recipeHandle);
}

void RecipeManager::printRecipePersistentTensors(const recipe_t& rRecipe)
{
    persist_tensor_info_t* pCurrentPersistentTensor = rRecipe.tensors;
    uint64_t               numOfPersistTensors      = rRecipe.persist_tensors_nr;

    LOG_ERR(SYN_RECIPE, "{}: Printing persistent-tensors' name of recipe {:#x}", HLLOG_FUNC, (uint64_t)&rRecipe);

    for (uint64_t i = 0; i < numOfPersistTensors; i++, pCurrentPersistentTensor++)
    {
        // TODO (?) - Dimensions will not be printed
        LOG_ERR(SYN_RECIPE,
                "Tensor id {} - {}: section [type {} index {} offset {} Bytes]"
                " size {} Bytes element-type {} zp {} scale {} batchSize {} type {}",
                i,
                pCurrentPersistentTensor->name,
                pCurrentPersistentTensor->section_type,
                pCurrentPersistentTensor->section_idx,
                pCurrentPersistentTensor->offset_in_section,
                pCurrentPersistentTensor->size,
                pCurrentPersistentTensor->elementType,
                pCurrentPersistentTensor->zp,
                pCurrentPersistentTensor->scale,
                pCurrentPersistentTensor->batchSize,
                pCurrentPersistentTensor->tensorType);
    }

    const_section_t* pCurrentConstSection = rRecipe.const_sections;
    uint64_t         numOfConstSections   = rRecipe.const_sections_nr;
    uint64_t         numOfSections        = rRecipe.sections_nr;

    LOG_ERR(SYN_RECIPE,
            "{}: Sections-amount: regular {} const {}",
            HLLOG_FUNC,
            numOfSections,
            numOfConstSections);

    LOG_ERR(SYN_RECIPE, "{}: Printing const-sections' info", HLLOG_FUNC);
    for (uint64_t i = 0; i < numOfConstSections; i++, pCurrentConstSection++)
    {
        LOG_ERR(SYN_RECIPE,
                "Const section {} - index {} size {:#x}",
                i,
                pCurrentConstSection->section_idx,
                pCurrentConstSection->size);
    }
}