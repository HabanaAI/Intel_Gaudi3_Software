#include <memory>

#include "defs.h"
#include "gc_gaudi_test_infra.h"
#include "infra/gc_synapse_test.h"
#include "node_factory.h"
#include "scoped_configuration_change.h"
#include "synapse_api_types.h"
#include "runtime/common/recipe/recipe_handle_impl.hpp"
#include "infra/global_conf_manager.h"


class SynGaudiTestStagedSubmission : public SynGaudiTestInfra
{
protected:
    void afterSynInitialize() override
    {
        synConfigurationSet("ENABLE_EXPERIMENTAL_FLAGS", "true");
        synConfigurationSet("ENABLE_STAGED_SUBMISSION", "true");
        SynGaudiTestInfra::afterSynInitialize();
    }
};

TEST_F_GC(SynGaudiTestStagedSubmission, patch_point_node_exection_order_L2, {synDeviceGaudi})
{
    HB_ASSERT(GCFG_ENABLE_STAGED_SUBMISSION.value() == true, "ENABLE_STAGED_SUBMISSION should be true for this test");
    ScopedConfigurationChange replace_copy_config("ENABLE_REMOVE_REDUNDANT_MEMCPY", "false");   // don't remove the memcopy
    // Graph will have three nodes: [relu_fwd]->[memcpy]->[relu_bwd]
    unsigned fwdIn  = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE);
    unsigned fwdOut = createTensor(OUTPUT_TENSOR);
    addNodeToGraph("relu_fwd_f32");

    unsigned dmaIn  = connectOutputTensorToInputTensor(fwdOut);
    unsigned dmaOut = createTensor(OUTPUT_TENSOR);
    addNodeToGraph("memcpy", {dmaIn}, {dmaOut});

    unsigned bwdIn1 = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE);
    unsigned bwdIn2 = connectOutputTensorToInputTensor(dmaOut);
    unsigned bwdOut = createPersistTensor(OUTPUT_TENSOR);
    addNodeToGraph("relu_bwd_f32", {bwdIn1, bwdIn2}, {bwdOut});

    compileTopology();

    // validate node/patch points execution order
    recipe_t* currRecipe = m_graphs[0].recipeHandle->basicRecipeHandle.recipe;

    uint32_t           nodeExeId;
    uint32_t           prevNodeExeId = 0;
    bool               bFirstNode    = true;
    std::set<uint32_t> nodeExeSet;

    // iterate all over patch points and verify they are ordered by node execution index
    for (uint64_t patch_index = 0; patch_index < currRecipe->patch_points_nr; patch_index++)
    {
        nodeExeId = currRecipe->patch_points[patch_index].node_exe_index;
        nodeExeSet.insert(nodeExeId);

        if (nodeExeId == 0)
        {
            // we would like to verify that all patch points with node execution index = 0 appears at the beginning
            if (!bFirstNode)
            {
                assert(0);
            }
            continue;
        }
        bFirstNode = false;

        // verify they are ordered in ascending order
        ASSERT_GE(nodeExeId, prevNodeExeId);

        if (nodeExeId > prevNodeExeId)
        {
            prevNodeExeId = nodeExeId;
        }
    }

    ASSERT_EQ(nodeExeSet.size(), 4);

    runTopology();

    float* pFwdInput  = (float*) m_hostBuffers[fwdIn];
    float* pBwdInput  = (float*) m_hostBuffers[bwdIn1];
    float* pBwdOutput = (float*) m_hostBuffers[bwdOut];

    for (uint64_t i = 0; i < getDefaultNumberOfElements(); i++)
    {
        float expectedResult = (*pFwdInput > 0) ? *pBwdInput : 0;
        ASSERT_EQ(expectedResult, *pBwdOutput)
            << "Mismatch at index " << i << " Expected:" << expectedResult << " BwdOutput: " << *pBwdOutput
            << " FwdInput: " << *pFwdInput << " BwdInput " << *pBwdInput;
        pFwdInput++;
        pBwdInput++;
        pBwdOutput++;
    }
}
