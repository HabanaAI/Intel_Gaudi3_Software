#include "recipe_staged_submission_processor.hpp"
#include "runtime/common/habana_global_conf_runtime.h"
#include "runtime/common/recipe/device_agnostic_recipe_info.hpp"
#include "runtime/common/recipe/basic_recipe_info.hpp"
#include "runtime/common/recipe/recipe_utils.hpp"
#include "recipe.h"
#include "defs.h"
#include <sstream>

void RecipeStagedSubmissionProcessor::process(const basicRecipeInfo&      rBasicRecipeInfo,
                                              RecipeStageSubmisssionInfo& rRecipeStageSubmissionInfo)
{
    if (GCFG_ENABLE_STAGED_SUBMISSION.value())
    {
        const bool isDsd = RecipeUtils::isDsd(rBasicRecipeInfo);
        _setupStagedSubmissionStages(*rBasicRecipeInfo.recipe, rRecipeStageSubmissionInfo, isDsd);
    }
}

void RecipeStagedSubmissionProcessor::_setupStagedSubmissionStages(
    const recipe_t&             rRecipe,
    RecipeStageSubmisssionInfo& rRecipeStageSubmissionInfo,
    const bool                  isDsd)
{
    uint32_t nodesPerStage = isDsd ? GCFG_DSD_STAGED_SUBMISSION_NODES_PER_STAGE.value()
                                   : GCFG_STAGED_SUBMISSION_NODES_PER_STAGE.value();
    HB_ASSERT(nodesPerStage > 1, "STAGED_SUBMISSION_NODES_PER_STAGE must be bigger than 1");
    float nodesPerStageFactor = isDsd ? GCFG_DSD_STAGED_SUBMISSION_NODES_PER_STAGE_INCREMENT_FACTOR.value()
                                      : GCFG_STAGED_SUBMISSION_NODES_PER_STAGE_INCREMENT_FACTOR.value();
    HB_ASSERT(nodesPerStageFactor >= 1.0, "STAGED_SUBMISSION_NODES_PER_STAGE_FACTOR must be bigger or equal to 1.0");

    std::vector<uint32_t> stagesNodes;

    if (rRecipe.node_nr > 0)
    {
        for (unsigned nodesIndex = nodesPerStage - 1; nodesIndex < rRecipe.node_nr - 1; nodesIndex += nodesPerStage)
        {
            stagesNodes.push_back(nodesIndex);
            nodesPerStage *= nodesPerStageFactor;
        }
        stagesNodes.push_back(rRecipe.node_nr - 1);
        if (LOG_LEVEL_AT_LEAST_DEBUG(SYN_RECIPE))
        {
            std::stringstream stageStringStream;
            stageStringStream << stagesNodes.size() << " Stages: ";
            for (auto stageNodeIndex : stagesNodes)
            {
                stageStringStream << " " << stageNodeIndex;
            }
            LOG_DEBUG(SYN_RECIPE, "{}", stageStringStream.str());
        }
    }

    rRecipeStageSubmissionInfo.m_stagesNodes = std::move(stagesNodes);
}