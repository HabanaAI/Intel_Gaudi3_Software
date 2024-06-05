#include "gc_gaudi_test_infra.h"
#include "node_factory.h"
#include "recipe.h"
#include "scoped_configuration_change.h"
#include "infra/gc_synapse_test.h"
#include "runtime/common/recipe/recipe_handle_impl.hpp"

class SynGaudiNodeExecutionTest : public SynGaudiTestInfra
{
    protected:
        virtual void SetUpTest() override;
        void         afterSynInitialize() override
        {
            synConfigurationSet("ENABLE_EXPERIMENTAL_FLAGS", "true");
            synConfigurationSet("ENABLE_STAGED_SUBMISSION", "true");
            SynGaudiTestInfra::afterSynInitialize();
        }
};

void SynGaudiNodeExecutionTest::SetUpTest()
{
    SynGaudiTestInfra::SetUpTest();
}

TEST_F_GC(SynGaudiNodeExecutionTest, node_execution_test)
{
    HB_ASSERT(GCFG_ENABLE_STAGED_SUBMISSION.value() == true, "ENABLE_STAGED_SUBMISSION should be true for this test");
    ScopedConfigurationChange scc("ENABLE_TPC_ICACHE_PREFETCH", "true");  // this GCFG affects the number of patchpoints
    ScopedConfigurationChange scalarPipeCfg("MIN_SCALAR_PIPE_INPUT_BYTES_FOR_SRAM_PLACEMENT",
                                            "0");  // Enable small scalar-pipe tensors sram placement

    ns_ScatterKernel::Params params;
    params.axis = 1;

    float inValues[4] = {1.0, 2.0, 3.0, 4.0};

    float indices[2] = {0.0, 0.0};
    float updates[2] = {3.1, 2.1};

    unsigned dataDims[4] = {1, 2, 2, 1};
    unsigned idxDims[4] = {1, 1, 2, 1};

    unsigned inputData = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, inValues, dataDims, 4 ,syn_type_single);
    unsigned inputIndices = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, indices, idxDims, 4,syn_type_int32);
    unsigned inputUpdates = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, updates, idxDims, 4 ,syn_type_single);

    unsigned outputTensor   = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, dataDims, 4,  syn_type_single);

    addNodeToGraph("scatter_fwd_f32", {inputData, inputIndices, inputUpdates}, {outputTensor}, &params, sizeof(ns_ScatterKernel::Params));

    compileTopology();

    synRecipeHandle recipeHandle = getRecipeHandle();
    recipe_t* currRecipe = recipeHandle->basicRecipeHandle.recipe;

    // verify that each blob count in node is greater or equal to the equivalent in previous node
    for (unsigned i=0; i < currRecipe->programs_nr; i++)
    {
        for (unsigned nodeIdx = 1; nodeIdx < currRecipe->node_nr; nodeIdx++)
        {
            ASSERT_GE(currRecipe->node_exe_list[nodeIdx].program_blobs_nr[i],
                      currRecipe->node_exe_list[nodeIdx - 1].program_blobs_nr[i]);
        }
    }
}
