#pragma once

#include "runtime/scal/common/recipe_static_info_scal.hpp"

struct MemorySectionsScal;

class TestDummyRecipe;

struct RecipeAndSections
{
    TestDummyRecipe*      pDummyRecipe;
    RecipeStaticInfoScal* pRecipeStaticInfoScal;
    MemorySectionsScal*   pSections;
};

class MappedMemMgrTestUtils
{
public:
    static void testingFillMappedPatchable(const RecipeSingleSectionVec& rRecipeSections,
                                           uint64_t                      id,
                                           const MemorySectionsScal&     rSections);

    static void testingFillMappedDsdPatchable(const RecipeSingleSectionVec& rRecipeSections,
                                              uint64_t                      id,
                                              const MemorySectionsScal&     rSections);

    static bool testingCompareWithRecipePatchable(const RecipeSingleSectionVec& rRecipeSections,
                                                  uint64_t                      id,
                                                  const MemorySectionsScal&     rSections);

    static bool testingCompareWithRecipeDynamicPatchable(const RecipeSingleSectionVec& rRecipeSections,
                                                         uint64_t                      id,
                                                         const MemorySectionsScal&     rSections);

    static bool testingCompareWithRecipeNonPatchable(const RecipeSingleSectionVec& rRecipeSections,
                                                     int                           sectionId,
                                                     const MemorySectionsScal&     rSections);

    static bool testingCompareWithRecipeSimulatedPatch(const RecipeSingleSectionVec& rRecipeSections,
                                                       uint64_t                      id,
                                                       const MemorySectionsScal&     rSections,
                                                       bool                          isDsd,
                                                       bool                          isIH2DRecipe);

    static bool testingVerifyMappedToDevOffset(uint64_t offset, const MemorySectionsScal& rSections);

    static void testingCopyToArc(const RecipeSingleSectionVec& rRecipeSections, const MemorySectionsScal& rSections);

    static bool testingCheckArcSections(const RecipeSingleSectionVec& rRecipeSections,
                                        uint64_t                      offset,
                                        const MemorySectionsScal&     rSections);

    static void testingCopyToGlb(const RecipeSingleSectionVec& rRecipeSections, const MemorySectionsScal& rSections);

    static bool testingCheckGlbSections(const RecipeSingleSectionVec& rRecipeSections,
                                        const MemorySectionsScal&     rSections);
};
