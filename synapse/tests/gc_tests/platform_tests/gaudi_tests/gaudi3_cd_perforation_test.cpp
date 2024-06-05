#include "gaudi_tests/gc_gaudi_test_infra.h"
#include "node_factory.h"
#include "synapse_common_types.h"
#include "tensor.h"
#include "test_types.hpp"
#include <vector>
#include "syn_gaudi_two_run_compare_test.h"
#include "utils.h"
#include "syn_singleton.hpp"

static bool verifyCDParallel(const synGraphHandle& handle)
{
    const HabanaGraph* graph = synSingleton::getInstanceInternal()->getGraph(handle);
    bool               ret   = true;
    for (const NodePtr& n : graph->getNodes())
    {
        if (graph->runsOnMME(n))
        {
            MMENodePtr mmeNode = std::dynamic_pointer_cast<MmeNode>(n);
            ret &= mmeNode->isCdPerforated();
        }
    }
    return ret;
}

class SynTrainingCDParallelGemmTest
: public SynGaudiTwoRunCompareTest
, public testing::WithParamInterface<std::tuple<int /* cd */, int /* h */, int /* w */, int /* batch*/>>
{
};

TEST_P_GC(SynTrainingCDParallelGemmTest, gemm_with_large_cd, {synDeviceGaudi3})
{
    const unsigned        commonDim = std::get<0>(GetParam());
    const unsigned        height    = std::get<1>(GetParam());
    const unsigned        width     = std::get<2>(GetParam());
    const unsigned        batch     = std::get<3>(GetParam());
    std::vector<unsigned> aSizes    = {commonDim, height};
    std::vector<unsigned> bSizes    = {width, commonDim};
    std::vector<unsigned> outSizes  = {width, height};

    if (batch > 1)
    {
        aSizes.push_back(batch);
        bSizes.push_back(batch);
        outSizes.push_back(batch);
    }
    unsigned a = createTensors(1,
                               INPUT_TENSOR,
                               true,
                               "A",
                               MEM_INIT_RANDOM_WITH_NEGATIVE,
                               nullptr,
                               aSizes.data(),
                               aSizes.size(),
                               syn_type_single,
                               nullptr,
                               0,
                               0,
                               nullptr,
                               false,
                               aSizes.data(),
                               synTensorType::DATA_TENSOR)[0];

    unsigned b = createTensors(1,
                               INPUT_TENSOR,
                               true,
                               "B",
                               MEM_INIT_RANDOM_WITH_NEGATIVE,
                               nullptr,
                               bSizes.data(),
                               bSizes.size(),
                               syn_type_single,
                               nullptr,
                               0,
                               0,
                               nullptr,
                               false,
                               bSizes.data(),
                               synTensorType::DATA_TENSOR)[0];

    unsigned out = createTensors(1,
                                 OUTPUT_TENSOR,
                                 true,
                                 "out",
                                 MEM_INIT_ALL_ZERO,
                                 nullptr,
                                 outSizes.data(),
                                 outSizes.size(),
                                 syn_type_single,
                                 nullptr,
                                 0,
                                 0,
                                 nullptr,
                                 false,
                                 outSizes.data(),
                                 synTensorType::DATA_TENSOR)[0];

    synGEMMParams gemmParams(false, false);

    if (batch > 1)
    {
        addNodeToGraph(NodeFactory::batchGemmNodeTypeName, {a, b}, {out}, nullptr, 0, "BGEMM");
    }
    else
    {
        addNodeToGraph(NodeFactory::gemmNodeTypeName, {a, b}, {out}, &gemmParams, sizeof(gemmParams), "GEMM");
    }

    // First run configuration
    addConfigurationToRun(FIRST_RUN, "ENABLE_CD_PARALLEL", "false");

    // Second run configuration
    addConfigurationToRun(SECOND_RUN, "ENABLE_CD_PARALLEL", "true");
    addConfigurationToRun(SECOND_RUN, "LAYERED_BRAIN_BUNDLE_MIN_NOF_SLICES", "1");

    compareRunsResults({out});
    // verify cd parallel strategy was chosen
    ASSERT_EQ(verifyCDParallel(getGraph(1).graphHandle), true) << "mme node is expected to be perforated on cd";
}

INSTANTIATE_TEST_SUITE_P(test,
                         SynTrainingCDParallelGemmTest,
                         ::testing::Values(std::make_tuple(8200, 256, 512, 1),
                                           std::make_tuple(8200, 128, 128, 1),
                                           std::make_tuple(8200, 256, 256, 1),
                                           std::make_tuple(8200, 64, 64, 2),
                                           std::make_tuple(8200, 256, 128, 2),
                                           std::make_tuple(8200, 32, 32, 4)));

class SynTrainingCDParallelConvTest
: public SynGaudiTwoRunCompareTest
, public testing::WithParamInterface<std::tuple<int /* c */,
                                                int /* k */,
                                                int /* w */,
                                                int /* h */,
                                                int /* batch */,
                                                int /* kW */,
                                                int /* kH */,
                                                const char* /*opType*/>>
{
public:
    typedef struct
    {
        unsigned a;
        unsigned b;
        unsigned c;
    } TensorsData;

    static TensorsData setTensorsByOp(unsigned x, unsigned w, unsigned y, const char* opType)
    {
        TensorsData tensors;

        if (!strcmp(opType, NodeFactory::convolutionNodeTypeName))
        {
            tensors.a = x;
            tensors.b = w;
            tensors.c = y;
        }
        else if (!strcmp(opType, NodeFactory::deDwNodeTypeName))
        {
            tensors.a = y;
            tensors.b = x;
            tensors.c = w;
        }
        else
        {
            tensors.a = y;
            tensors.b = w;
            tensors.c = x;
        }

        return tensors;
    }
};

TEST_P_GC(SynTrainingCDParallelConvTest, conv_with_large_cd, {synDeviceGaudi3})
{
    synConvolutionParams convParams;
    const unsigned       c     = std::get<0>(GetParam());
    const unsigned       k     = std::get<1>(GetParam());
    const unsigned       wOFM  = std::get<2>(GetParam());
    const unsigned       hOFM  = std::get<3>(GetParam());
    const unsigned       batch = std::get<4>(GetParam());
    convParams.kW              = std::get<5>(GetParam());
    convParams.kH              = std::get<6>(GetParam());
    const char* opType         = std::get<7>(GetParam());

    const unsigned wIFM =
        convInputDimSize(wOFM, convParams.kW, convParams.dW, convParams.padL + convParams.padR, convParams.dilW);
    const unsigned hIFM =
        convInputDimSize(hOFM, convParams.kH, convParams.dH, convParams.padT + convParams.padB, convParams.dilH);
    TestSizes xSize = {c, wIFM, hIFM, batch};
    TestSizes wSize = {k, c, convParams.kW, convParams.kH};
    TestSizes ySize = {k, wOFM, hOFM, batch};

    unsigned x = createPersistTensor(!strcmp(opType, NodeFactory::deDxNodeTypeName) ? OUTPUT_TENSOR : INPUT_TENSOR,
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,
                                     xSize.data(),
                                     DEFAULT_SIZES,
                                     syn_type_single);

    unsigned w = createPersistTensor(!strcmp(opType, NodeFactory::deDwNodeTypeName) ? OUTPUT_TENSOR : INPUT_TENSOR,
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,
                                     wSize.data(),
                                     DEFAULT_SIZES,
                                     syn_type_single);

    unsigned y =
        createPersistTensor(!strcmp(opType, NodeFactory::convolutionNodeTypeName) ? OUTPUT_TENSOR : INPUT_TENSOR,
                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                            nullptr,
                            ySize.data(),
                            DEFAULT_SIZES,
                            syn_type_single);

    TensorsData tensors = setTensorsByOp(x, w, y, opType);
    addNodeToGraph(opType, {tensors.a, tensors.b}, {tensors.c}, &convParams, sizeof(convParams));

    // First run configuration
    addConfigurationToRun(FIRST_RUN, "ENABLE_CD_PARALLEL", "false");

    // Second run configuration
    addConfigurationToRun(SECOND_RUN, "ENABLE_CD_PARALLEL", "true");
    addConfigurationToRun(SECOND_RUN, "LAYERED_BRAIN_BUNDLE_MIN_NOF_SLICES", "1");
    addConfigurationToRun(SECOND_RUN,
                          "ENABLE_LB_MME_CONCURRENCY_OPT",
                          "false");  // Disable concurrency solutions until cd-parallel & cd-concurrency will be
                                     // supported together [SW-174762] [SW-144531]

    compareRunsResults({tensors.c});

    // verify cd parallel strategy was chosen
    ASSERT_EQ(verifyCDParallel(getGraph(1).graphHandle), true) << "mme node is expected to be perforated on cd";
}

INSTANTIATE_TEST_SUITE_P(
    test,
    SynTrainingCDParallelConvTest,
    ::testing::Values(std::make_tuple(8200, 2, 7, 7, 1, 3, 3, NodeFactory::convolutionNodeTypeName),
                      std::make_tuple(8200, 2, 7, 7, 4, 3, 3, NodeFactory::convolutionNodeTypeName),
                      std::make_tuple(2, 8200, 7, 7, 1, 3, 3, NodeFactory::deDxNodeTypeName),
                      std::make_tuple(2, 8200, 7, 7, 4, 3, 3, NodeFactory::deDxNodeTypeName),
                      std::make_tuple(256, 512, 1, 1, 8192, 1, 1, NodeFactory::deDwNodeTypeName),
                      std::make_tuple(256, 512, 1, 1, 8192, 2, 1, NodeFactory::deDwNodeTypeName),
                      std::make_tuple(256, 512, 1, 1, 8192, 1, 2, NodeFactory::deDwNodeTypeName),
                      std::make_tuple(256, 512, 1, 1, 8192, 2, 2, NodeFactory::deDwNodeTypeName),
                      std::make_tuple(256, 512, 1, 1, 8192, 3, 1, NodeFactory::deDwNodeTypeName),
                      std::make_tuple(256, 512, 1, 1, 8192, 1, 3, NodeFactory::deDwNodeTypeName),
                      std::make_tuple(256, 512, 1, 1, 8192, 2, 3, NodeFactory::deDwNodeTypeName)));