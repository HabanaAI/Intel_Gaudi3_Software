#pragma once

#include "synapse_common_types.h"

#include "runtime/common/recipe/basic_recipe_info.hpp"

struct recipe_t;
struct RecipeStaticInfoScal;

/************************************************************************/
//
// Below is how the sections organized in mapped memory and HBM memory
//
/************************************************************************/

// if(!DSD)
//---------
//
// In mapped patchable memory
//-----------
//|PATCHABLE|
//-----------
//
// In mapped not-patchable memory
//--------------------------------------------------------------------------------------
//|PRG DATA|ALIGN|NON-PATCHABLE|ALIGN|DYNAMIC|ALIGN|ECB-DYNAMIC0|ALIGN|ECB-STATIC0|....|
//--------------------------------------------------------------------------------------
//
//
// In Arc-HBM
//---------------------------------------------------
//|DYNAMIC|ALIGN|ECB-DYNAMIC0|ALIGN|ECB-STATIC0|....|
//---------------------------------------------------
//
// In Global-HBM
//----------------------------------------------
//|PRG DATA|ALIGN|NON-PATCHABLE|ALIGN|PATCHABLE|
//----------------------------------------------


// if(DSD)
//--------
//
// In mapped patchable memory
//-------------------------
//|PATCHABLE|ALIGN|DYNAMIC|
//-------------------------
//
// In mapped not-patchable memory
//------------------------------------------------------------------------
//|PRG DATA|ALIGN|NON-PATCHABLE|ALIGN|ECB-DYNAMIC0|ALIGN|ECB-STATIC0|....|
//------------------------------------------------------------------------
//
//
// In Arc-HBM
//---------------------------------------------------
//|DYNAMIC|ALIGN|ECB-DYNAMIC0|ALIGN|ECB-STATIC0|....|
//---------------------------------------------------
//
// In Global-HBM
//----------------------------------------------
//|PRG DATA|ALIGN|NON-PATCHABLE|ALIGN|PATCHABLE|
//----------------------------------------------


// if(IH2D && DSD)
//----------------
//
// In mapped patchable memory
//----------------------------------------
//|PATCHABLE|ALIGN|DYNAMIC|ALIGN|PRG DATA|
//----------------------------------------
//
// In mapped not-patchable memory
//---------------------------------------------------------
//|NON-PATCHABLE|ALIGN|ECB-DYNAMIC0|ALIGN|ECB-STATIC0|....|
//---------------------------------------------------------
//
//
// In Arc-HBM
//---------------------------------------------------
//|DYNAMIC|ALIGN|ECB-DYNAMIC0|ALIGN|ECB-STATIC0|....|
//---------------------------------------------------
//
// In Global-HBM
//----------------------------------------------
//|PRG DATA|ALIGN|NON-PATCHABLE|ALIGN|PATCHABLE|
//----------------------------------------------

class DeviceAgnosticRecipeStaticProcessorScal
{
public:
    static synStatus process(synDeviceType          deviceType,
                             const basicRecipeInfo& rBasicRecipeInfo,
                             RecipeStaticInfoScal&  rRecipeStaticInfoScal);

    static const uint32_t PRG_DATA_ALIGN = (1 << 13);

private:
    static synStatus processtRecipeSections(synDeviceType         deviceType,
                                            const recipe_t&       rRecipe,
                                            RecipeStaticInfoScal& rRecipeStaticInfoScal,
                                            bool                  isDsd,
                                            bool                  isIH2DRecipe);
};
