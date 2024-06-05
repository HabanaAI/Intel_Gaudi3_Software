#include "syn_gaudi_two_run_compare_test.h"
#include "syn_singleton.hpp"

static TSize getBNChannelSize(const synGraphHandle& handle)
{
    const HabanaGraph* graph = synSingleton::getInstanceInternal()->getGraph(handle);
    for (const NodePtr& n : graph->getNodes())
    {
        if (!n || n->getNodeType() != Node::TYPE_USER) continue;
        if (n->getGUID().find("batch_norm_stage1_fwd") != std::string::npos)
        {
            return n->getInput(0)->getSizeInElements(0);
        }
    }
    return 0;
}

TEST_F_GC(SynGaudiTwoRunCompareTest, low_fcd_batch_norm_test, {synDeviceGaudi2})  // not relevant for g3 which uses HN
{
    const unsigned C_SIZE       = 64;
    const unsigned WH_SIZE      = 163840;
    const auto     dtype        = syn_type_bf16;
    unsigned       bnIFMSizes[] = {C_SIZE, WH_SIZE, 1, 1};
    unsigned       bnCSizes[]   = {C_SIZE};

    unsigned xIn =
        createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, bnIFMSizes, 4, dtype, nullptr, "xIn");
    unsigned beta          = createPersistTensor(INPUT_TENSOR,
                                        MEM_INIT_RANDOM_WITH_NEGATIVE,
                                        nullptr,
                                        bnCSizes,
                                        1,
                                        syn_type_float,
                                        nullptr,
                                        "betta");
    unsigned gamma         = createPersistTensor(INPUT_TENSOR,
                                         MEM_INIT_RANDOM_WITH_NEGATIVE,
                                         nullptr,
                                         bnCSizes,
                                         1,
                                         syn_type_float,
                                         nullptr,
                                         "gamma");
    unsigned runningMeanIn = createPersistTensor(INPUT_TENSOR,
                                                 MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                 nullptr,
                                                 bnCSizes,
                                                 1,
                                                 syn_type_float,
                                                 nullptr,
                                                 "runningMeanIn");
    unsigned runningVarIn  = createPersistTensor(INPUT_TENSOR,
                                                MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                nullptr,
                                                bnCSizes,
                                                1,
                                                syn_type_float,
                                                nullptr,
                                                "runningVarIn");

    unsigned xOut =
        createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, bnIFMSizes, 4, dtype, nullptr, "xOut");
    unsigned meanOut =
        createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, bnCSizes, 1, syn_type_float, nullptr, "meanOut");
    unsigned iStdOut =
        createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, bnCSizes, 1, syn_type_float, nullptr, "iStdOut");
    unsigned runningMeanOut = createPersistTensor(OUTPUT_TENSOR,
                                                  MEM_INIT_ALL_ZERO,
                                                  nullptr,
                                                  bnCSizes,
                                                  1,
                                                  syn_type_float,
                                                  nullptr,
                                                  "runningMeanOut");
    unsigned runningVarOut  = createPersistTensor(OUTPUT_TENSOR,
                                                 MEM_INIT_ALL_ZERO,
                                                 nullptr,
                                                 bnCSizes,
                                                 1,
                                                 syn_type_float,
                                                 nullptr,
                                                 "runningVarOut");

    ns_BatchNormKernel::ParamsV2 params;
    params.isTraining  = true;
    params.momentum    = 0.1;
    params.threshold.f = 1e-05;
    params.epsilon     = 1e-05;
    addNodeToGraph("batch_norm_fwd_bf16",
                   {xIn, beta, gamma, runningMeanIn, runningVarIn},
                   {xOut, meanOut, iStdOut, runningMeanOut, runningVarOut},
                   &params,
                   sizeof(params),
                   "batch_norm");

    addConfigurationToRun(FIRST_RUN, "ENABLE_BN_FCD_PACKING", "false");
    addConfigurationToRun(SECOND_RUN, "ENABLE_BN_FCD_PACKING", "true");
    compareRunsResults({xOut, meanOut, iStdOut, runningMeanOut, runningVarOut});

    // check optimization was really done -
    TSize fcdSize0 = getBNChannelSize(getGraph(0).graphHandle);
    TSize fcdSize1 = getBNChannelSize(getGraph(1).graphHandle);
    ASSERT_NE(fcdSize0, 0);
    ASSERT_NE(fcdSize0, fcdSize1);
}
