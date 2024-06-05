#pragma once

#include "synapse_common_types.h"
struct recipe_t;
struct shape_plane_graph_t;
struct RecipeStaticInfoScal;
namespace RecipeVerification
{
bool verifyProgramCodeBlobs(const recipe_t* pRecipe);
bool verifyProgramDataBlobs(const recipe_t* pRecipe, bool testMode = false);
bool verifyTpc(const recipe_t* pRecipe);
bool verifyPatching(const recipe_t* pRecipe);
bool verifyDynamicRecipe(const recipe_t* pRecipe, const shape_plane_graph_t* spg);
bool verifyStagesSubmissionNodeExe(const recipe_t* pRecipe, const shape_plane_graph_t* spg);
bool verifyStagedInfo(const recipe_t* pRecipe, const shape_plane_graph_t* spg);
bool verifyRecipeCacheSize(const recipe_t* pRecipe);
bool verifyScalRecipe(const recipe_t* recipe);
// internally calls all the above verifications
bool verifyRecipe(const recipe_t* pRecipe, const shape_plane_graph_t* spg);
bool verifyScalMemorySizes(const RecipeStaticInfoScal& recipeStaticInfoScal);
}  // namespace RecipeVerification
