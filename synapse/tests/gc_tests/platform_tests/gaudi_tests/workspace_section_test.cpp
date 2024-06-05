#include "synapse_api.h"
#include "infra/gc_synapse_test.h"
#include "gc_gaudi_test_infra.h"
#include "node_factory.h"

// A simple test initiating half a workspace section with 0 and the other half with 1
// Todo- fix and enable the following test: [SW-137638] https://jira.habana-labs.com/browse/SW-137638
//  start failing after applying eliminateRedundantNodes
TEST_F_GC(SynGaudiTestInfra, DISABLED_three_tensors_in_section)
{
    unsigned inSize  = 128;
    unsigned outSize = 256;

    auto inAIdx =
        createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, &inSize, 1, syn_type_single, nullptr, "inA");
    auto sectionIdx = createNonPersistentSection(0, false);

    auto memcopyAOutIdx = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, /*isPersistent*/
                                        "inAOut",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr,
                                        &inSize,
                                        1,
                                        syn_type_float,
                                        nullptr,
                                        0,
                                        0, /*offsetInSection*/
                                        &sectionIdx /*sectionIndex*/)[0];

    synNodeId memcopyANodeId;

    addNodeToGraph("memcpy", {inAIdx}, {memcopyAOutIdx}, nullptr, 0, "memcopyA", 0, &memcopyANodeId);

    auto inBIdx =
        createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ONES, nullptr, &inSize, 1, syn_type_single, nullptr, "inB");

    auto      memcopyBOutIdx = createTensors(1,
                                        OUTPUT_TENSOR,
                                        false, /*isPersistent*/
                                        "inBout",
                                        MEM_INIT_ALL_ZERO,
                                        nullptr,
                                        &inSize,
                                        1,
                                        syn_type_float,
                                        nullptr,
                                        0,
                                        inSize * sizeof(float), /*offsetInSection*/
                                        &sectionIdx /*sectionIndex*/)[0];
    synNodeId memcopyBNodeId;
    addNodeToGraph("memcpy", {inBIdx}, {memcopyBOutIdx}, nullptr, 0, "memcopyB", 0, &memcopyBNodeId);

    auto fullSectionMemcopyInIdx = createTensors(1,
                                                 OUTPUT_TENSOR,
                                                 false, /*isPersistent*/
                                                 "full_section",
                                                 MEM_INIT_ALL_ZERO,
                                                 nullptr,
                                                 &outSize,
                                                 1,
                                                 syn_type_float,
                                                 nullptr,
                                                 0,
                                                 0, /*offsetInSection*/
                                                 &sectionIdx /*sectionIndex*/)[0];

    auto outFullSectionIdx =
        createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, &outSize, 1, syn_type_single, nullptr, "input");

    synNodeId memcopyFullNodeId;
    addNodeToGraph("memcpy",
                   {fullSectionMemcopyInIdx},
                   {outFullSectionIdx},
                   nullptr,
                   0,
                   "memcopy_full",
                   0,
                   &memcopyFullNodeId);

    synNodeDependencySet(m_graphs[0].graphHandle, &memcopyANodeId, &memcopyFullNodeId, 1, 1);
    synNodeDependencySet(m_graphs[0].graphHandle, &memcopyBNodeId, &memcopyFullNodeId, 1, 1);

    compileTopology();
    runTopology();

    float* pOutput = castHostBuffer<float>(outFullSectionIdx);

    for (auto i = 0; i < outSize; i++)
    {
        if (i < inSize)
        {
            ASSERT_EQ(pOutput[i], 0);
        }
        else
        {
            ASSERT_EQ(pOutput[i], 1);
        }
    }
}
