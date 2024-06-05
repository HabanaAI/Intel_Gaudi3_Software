#include "syn_gaudi_two_run_compare_test.h"
#include "syn_singleton.hpp"

class SynFuseBNTest : public SynGaudiTwoRunCompareTest
{
protected:
    unsigned createBN();
    bool     findGuid(unsigned graphIndex, std::string_view guid);
    void     verifyResults(std::string_view guidToFind);

    synDataType               m_dtype           = syn_type_float;
    static constexpr unsigned DIM               = 4;
    unsigned                  m_ifmSizes[DIM]   = {64, 64, 64, 1};
    unsigned                  m_reshapeSizes[1] = {multiplyElements(m_ifmSizes, m_ifmSizes + DIM)};
    std::vector<unsigned>     m_validateOutputs;
};

void SynFuseBNTest::verifyResults(std::string_view guidToFind)
{
    /* compare results accuracy */
    addConfigurationToRun(FIRST_RUN, "ENABLE_FUSE_BATCH_NORM", "false");
    addConfigurationToRun(SECOND_RUN, "ENABLE_FUSE_BATCH_NORM", "true");
    compareRunsResults(m_validateOutputs);
    /* check fusion */
    ASSERT_TRUE(findGuid(1, guidToFind));
}

bool SynFuseBNTest::findGuid(unsigned graphIndex, std::string_view guid)
{
    const HabanaGraph* graph = synSingleton::getInstanceInternal()->getGraph(getGraph(graphIndex).graphHandle);
    for (const NodePtr& n : graph->getNodes())
    {
        if (!n || n->getNodeType() != Node::TYPE_USER) continue;
        if (n->getGUID().find(guid) != std::string::npos)
        {
            return true;
        }
    }
    return false;
}

unsigned SynFuseBNTest::createBN()
{
    unsigned stateSize[1]  = {m_ifmSizes[0]};
    unsigned xIn           = createPersistTensor(INPUT_TENSOR,
                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                       nullptr,
                                       m_ifmSizes,
                                       DIM,
                                       m_dtype,
                                       nullptr,
                                       "xIn");
    unsigned beta          = createPersistTensor(INPUT_TENSOR,
                                        MEM_INIT_RANDOM_WITH_NEGATIVE,
                                        nullptr,
                                        stateSize,
                                        1,
                                        syn_type_float,
                                        nullptr,
                                        "betta");
    unsigned gamma         = createPersistTensor(INPUT_TENSOR,
                                         MEM_INIT_RANDOM_WITH_NEGATIVE,
                                         nullptr,
                                         stateSize,
                                         1,
                                         syn_type_float,
                                         nullptr,
                                         "gamma");
    unsigned runningMeanIn = createPersistTensor(INPUT_TENSOR,
                                                 MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                 nullptr,
                                                 stateSize,
                                                 1,
                                                 syn_type_float,
                                                 nullptr,
                                                 "runningMeanIn");
    unsigned runningVarIn  = createPersistTensor(INPUT_TENSOR,
                                                MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                nullptr,
                                                stateSize,
                                                1,
                                                syn_type_float,
                                                nullptr,
                                                "runningVarIn");

    unsigned xOut           = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, m_ifmSizes, DIM, m_dtype);
    unsigned meanOut        = createPersistTensor(OUTPUT_TENSOR,
                                           MEM_INIT_ALL_ZERO,
                                           nullptr,
                                           stateSize,
                                           1,
                                           syn_type_float,
                                           nullptr,
                                           "meanOut");
    unsigned iStdOut        = createPersistTensor(OUTPUT_TENSOR,
                                           MEM_INIT_ALL_ZERO,
                                           nullptr,
                                           stateSize,
                                           1,
                                           syn_type_float,
                                           nullptr,
                                           "iStdOut");
    unsigned runningMeanOut = createPersistTensor(OUTPUT_TENSOR,
                                                  MEM_INIT_ALL_ZERO,
                                                  nullptr,
                                                  stateSize,
                                                  1,
                                                  syn_type_float,
                                                  nullptr,
                                                  "runningMeanOut");
    unsigned runningVarOut  = createPersistTensor(OUTPUT_TENSOR,
                                                 MEM_INIT_ALL_ZERO,
                                                 nullptr,
                                                 stateSize,
                                                 1,
                                                 syn_type_float,
                                                 nullptr,
                                                 "runningVarOut");

    ns_BatchNormKernel::ParamsV2 params;
    params.isTraining  = true;
    params.momentum    = 0.1;
    params.threshold.f = 1e-05;
    params.epsilon     = 1e-05;
    const auto bnGuid  = fmt::format("batch_norm_fwd_{}", getDtypeSuffixFromSynDataType(m_dtype));
    addNodeToGraph(bnGuid.c_str(),
                   {xIn, beta, gamma, runningMeanIn, runningVarIn},
                   {xOut, meanOut, iStdOut, runningMeanOut, runningVarOut},
                   &params,
                   sizeof(params),
                   "batch_norm");
    m_validateOutputs.push_back(meanOut);
    m_validateOutputs.push_back(iStdOut);
    m_validateOutputs.push_back(runningMeanOut);
    m_validateOutputs.push_back(runningVarOut);
    return xOut;
}

TEST_F_GC(SynFuseBNTest, test_fuse_bn_relu)
{
    unsigned    bnOut   = createBN();
    unsigned    reluOut = createPersistTensor(OUTPUT_TENSOR,
                                           MEM_INIT_ALL_ZERO,
                                           nullptr,
                                           m_ifmSizes,
                                           ARRAY_SIZE(m_ifmSizes),
                                           syn_type_float,
                                           nullptr,
                                           "out");
    std::string guid    = fmt::format("relu_fwd_{}", getDtypeSuffixFromSynDataType(m_dtype));
    addNodeToGraph(guid.c_str(), {bnOut}, {reluOut}, nullptr, 0, "relu");
    m_validateOutputs.push_back(reluOut);

    verifyResults("batch_norm_stage2_relu_fwd");
}

TEST_F_GC(SynFuseBNTest, test_fuse_bn_reshape_relu)
{
    unsigned bnOut = createBN();
    unsigned reshapeOut =
        createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, m_reshapeSizes, ARRAY_SIZE(m_reshapeSizes), m_dtype);
    addNodeToGraph("reshape", {bnOut}, {reshapeOut}, nullptr, 0, "reshape");
    unsigned    reluOut = createPersistTensor(OUTPUT_TENSOR,
                                           MEM_INIT_ALL_ZERO,
                                           nullptr,
                                           m_reshapeSizes,
                                           1,
                                           syn_type_float,
                                           nullptr,
                                           "out");
    std::string guid    = fmt::format("relu_fwd_{}", getDtypeSuffixFromSynDataType(m_dtype));
    addNodeToGraph(guid.c_str(), {reshapeOut}, {reluOut}, nullptr, 0, "relu");
    m_validateOutputs.push_back(reluOut);

    verifyResults("batch_norm_stage2_relu_fwd");
}

TEST_F_GC(SynFuseBNTest, test_fuse_bn_add_relu)
{
    unsigned bnOut = createBN();
    unsigned addOut =
        createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, m_ifmSizes, ARRAY_SIZE(m_ifmSizes), m_dtype);
    unsigned    addIn   = createPersistTensor(INPUT_TENSOR,
                                         MEM_INIT_RANDOM_WITH_NEGATIVE,
                                         nullptr,
                                         m_ifmSizes,
                                         ARRAY_SIZE(m_ifmSizes),
                                         m_dtype);
    std::string addGuid = fmt::format("add_fwd_{}", getDtypeSuffixFromSynDataType(m_dtype));
    addNodeToGraph(addGuid.c_str(), {addIn, bnOut}, {addOut}, nullptr, 0, "reshape");
    unsigned    reluOut  = createPersistTensor(OUTPUT_TENSOR,
                                           MEM_INIT_ALL_ZERO,
                                           nullptr,
                                           m_ifmSizes,
                                           ARRAY_SIZE(m_ifmSizes),
                                           syn_type_float,
                                           nullptr,
                                           "out");
    std::string reluGuid = fmt::format("relu_fwd_{}", getDtypeSuffixFromSynDataType(m_dtype));
    addNodeToGraph(reluGuid.c_str(), {addOut}, {reluOut}, nullptr, 0, "relu");
    m_validateOutputs.push_back(reluOut);

    verifyResults("batch_norm_stage2_add_relu_fwd");
}

TEST_F_GC(SynFuseBNTest, test_fuse_bn_reshape_add_relu)
{
    unsigned bnOut = createBN();
    unsigned reshapeOut =
        createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, m_reshapeSizes, ARRAY_SIZE(m_reshapeSizes), m_dtype);
    addNodeToGraph("reshape", {bnOut}, {reshapeOut}, nullptr, 0, "reshape");
    unsigned addOut =
        createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, m_reshapeSizes, ARRAY_SIZE(m_reshapeSizes), m_dtype);
    unsigned    addIn   = createPersistTensor(INPUT_TENSOR,
                                         MEM_INIT_RANDOM_WITH_NEGATIVE,
                                         nullptr,
                                         m_reshapeSizes,
                                         ARRAY_SIZE(m_reshapeSizes),
                                         m_dtype);
    std::string addGuid = fmt::format("add_fwd_{}", getDtypeSuffixFromSynDataType(m_dtype));
    addNodeToGraph(addGuid.c_str(), {addIn, reshapeOut}, {addOut}, nullptr, 0, "reshape");
    unsigned    reluOut  = createPersistTensor(OUTPUT_TENSOR,
                                           MEM_INIT_ALL_ZERO,
                                           nullptr,
                                           m_reshapeSizes,
                                           ARRAY_SIZE(m_reshapeSizes),
                                           syn_type_float,
                                           nullptr,
                                           "out");
    std::string reluGuid = fmt::format("relu_fwd_{}", getDtypeSuffixFromSynDataType(m_dtype));
    addNodeToGraph(reluGuid.c_str(), {addOut}, {reluOut}, nullptr, 0, "relu");
    m_validateOutputs.push_back(reluOut);

    verifyResults("batch_norm_stage2_add_relu_fwd");
}