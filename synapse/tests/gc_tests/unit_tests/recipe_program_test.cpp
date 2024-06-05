#include "recipe_test_base.h"
#include "graph_compiler/recipe_program.h"
#include "recipe_allocator.h"

void RecipeTestBase::program_basic()
{
    RecipeProgramContainer testProgContainer;
    unsigned maxEngines = 10;
    unsigned maxBlobs = 5;
    unsigned programIndex;

    for (unsigned i = 0 ; i < maxEngines; i++)
    {
        RecipeProgram& testProgram = testProgContainer.getProgram(i, DEVICE_TPC, programIndex);

        for (unsigned blob = 0 ; blob < maxBlobs; blob++)
        {
            testProgram.insertBlobIndex(blob*(i+1));
        }
    }

    ASSERT_EQ(testProgContainer.getNumPrograms(), maxEngines) << "Wrong number of programs in the container!";

    RecipeProgram& testProgram = testProgContainer.getProgramByIndex(0);
    ASSERT_EQ(testProgram.getEngineId(), 0) << "Wrong engine id for the first program in the container!";

    testProgContainer.eraseProgramByIndex(0);
    testProgram = testProgContainer.getProgramByIndex(0);
    ASSERT_NE(testProgram.getEngineId(), 0) << "Wrong engine id for the first program in the container!";
    ASSERT_EQ(testProgContainer.getNumPrograms(), maxEngines-1) << "Wrong number of programs in the container!";
    testProgContainer.print();
}

void RecipeTestBase::program_serialize()
{
    RecipeProgramContainer testProgContainer;
    unsigned maxEngines = 10;
    unsigned maxBlobs = 5;
    unsigned programIndex;

    for (unsigned i = 0 ; i < maxEngines; i++)
    {
        RecipeProgram& testProgram = testProgContainer.getProgram(i, DEVICE_TPC, programIndex);

        for (unsigned blob = 0 ; blob < maxBlobs; blob++)
        {
            testProgram.insertBlobIndex((blob+1)*(i+1));
        }
    }

    program_t *pSerPrograms;
    uint32_t NumOfPrograms;
    RecipeAllocator recipeAlloc;
    testProgContainer.serialize(&NumOfPrograms, &pSerPrograms, &recipeAlloc);

    ASSERT_EQ(NumOfPrograms, maxEngines) << "Wrong number of serialized programs!";

    for (unsigned i = 0 ; i < maxEngines; i++)
    {
        ASSERT_EQ(pSerPrograms[i].program_length, maxBlobs) << "Wrong serialized program length!";

        for (unsigned blob = 0 ; blob < maxBlobs; blob++)
        {
            ASSERT_EQ(pSerPrograms[i].blob_indices[blob], (blob+1)*(i+1)) << "Wrong serialized program length!";
        }

        delete[] pSerPrograms[i].blob_indices;
    }

    // Need to delete all
    delete[] pSerPrograms;
}
