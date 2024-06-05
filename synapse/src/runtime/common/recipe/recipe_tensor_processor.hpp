#pragma once

#include "synapse_api_types.h"

struct basicRecipeInfo;
struct RecipeTensorsInfo;
struct RecipeDsdStaticInfo;
struct tensor_info_t;

class RecipeTensorsProcessor
{
public:
    static synStatus process(const basicRecipeInfo& rBasicRecipeInfo,
                             RecipeTensorsInfo&     rRecipeTensorsInfo,
                             RecipeDsdStaticInfo&   rRecipeDsdStaticInfo);

    static bool testOnlyProcessShapePlanRecipe(const basicRecipeInfo& rBasicRecipeInfo,
                                               RecipeTensorsInfo&     rRecipeTensorsInfo,
                                               RecipeDsdStaticInfo&   rRecipeDsdStaticInfo);

private:
    static void setSectionsInfo(const basicRecipeInfo& rBasicRecipeInfo, RecipeTensorsInfo& rRecipeTensorsInfo);
    static void setSectionTypesInfo(const basicRecipeInfo& rBasicRecipeInfo, RecipeTensorsInfo& rRecipeTensorsInfo);
    static void initTensorInfo(const basicRecipeInfo& rBasicRecipeInfo, RecipeTensorsInfo& rRecipeTensorsInfo);

    static bool processShapePlanRecipe(const basicRecipeInfo& rBasicRecipeInfo,
                                       RecipeTensorsInfo&     rRecipeTensorsInfo,
                                       RecipeDsdStaticInfo&   rRecipeDsdStaticInfo);

    static void setSectionDb(const basicRecipeInfo& rBasicRecipeInfo, RecipeTensorsInfo& rRecipeTensorsInfo);
    static bool checkIsStaticTensor(const tensor_info_t& tensor);

    friend class UTrecipePatchProcTest;
};
