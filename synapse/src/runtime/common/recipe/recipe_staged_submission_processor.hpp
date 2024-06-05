#pragma once

struct basicRecipeInfo;
struct RecipeStageSubmisssionInfo;
struct recipe_t;

class RecipeStagedSubmissionProcessor
{
public:
    static void process(const basicRecipeInfo&      rBasicRecipeInfo,
                        RecipeStageSubmisssionInfo& rRecipeStageSubmissionInfo);

private:
    static void _setupStagedSubmissionStages(const recipe_t&             rRecipe,
                                             RecipeStageSubmisssionInfo& rRecipeStageSubmissionInfo,
                                             const bool                  isDsd);
};
