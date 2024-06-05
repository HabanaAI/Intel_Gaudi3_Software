#include "recipe_test_base.h"
#include "recipe_generator.h"
#include "recipe_allocator.h"
#include "graph_compiler/graph_traits.h"
#include "graph_compiler/compilation_hal_reader.h"
#include "graph_compiler/habana_graph.h"

void RecipeTestBase::unique_hash_system_base_test(HabanaGraph *g)
{
    CompilationHalReaderSetter compHalReaderSetter(g);
    makeQueues(2, g, true, true);
    RecipeGenerator generator(g);
    generator.generateRecipes(false);
    RecipeAllocator recipeAlloc;
    recipe_t*       recipe = generator.serializeDataPlaneGraph(&recipeAlloc);
    makeQueues(4, g, true, true);
    RecipeGenerator generator2(g);
    generator2.generateRecipes(false);
    recipe_t* recipe2 = generator.serializeDataPlaneGraph(&recipeAlloc);

    uint32_t blobsNum = 1;

    ASSERT_EQ(recipe->program_data_blobs_nr, blobsNum);
    ASSERT_EQ(recipe2->program_data_blobs_nr, blobsNum);

    // No matter how many queues, as long as the blobs are the same, it should compress them.
    EXPECT_EQ(recipe->blobs_nr, recipe2->blobs_nr);

}
