#include "gaudi_dual_execution_test_infra.h"
#include "node_factory.h"
class SynTrainingConvDualExecutionTest : public SynDualExecutionGaudiTestInfra
{
public:
    static const unsigned NUM_DIMS = 4;
    using TensorLayoutParams       = std::pair<std::string, bool>;

    SynTrainingConvDualExecutionTest()
    {
    }

    void basicGroupedConvolution(const TensorLayoutParams& ifmDataLayout,
                                 const TensorLayoutParams& wghDataLayout,
                                 const TensorLayoutParams& ofmDataLayout);
};

void SynTrainingConvDualExecutionTest::basicGroupedConvolution(const TensorLayoutParams& ifmDataLayout,
                                                            const TensorLayoutParams& wghDataLayout,
                                                            const TensorLayoutParams& ofmDataLayout)
{
    synConvolutionParams params;

    params.kH   = 3;
    params.kW   = 3;
    params.padT    = 1;
    params.padB    = 1;
    params.padL    = 1;
    params.padR    = 1;
    params.nGroups = 11;

    const unsigned batch = 4;
    const unsigned nIFM  = 11;
    const unsigned nOFM  = 1;
    const unsigned wOFM  = 56;
    const unsigned hOFM  = 56;
    const unsigned wIFM  = 56;
    const unsigned hIFM  = 56;

    auto generateDimArray = [](const std::string&                    userLayout,
                               const std::array<unsigned, NUM_DIMS>& expectedLayoutDims) {
        std::array<unsigned, NUM_DIMS> userLayoutDims;
        for (int i = 0; i < NUM_DIMS; i++)
        {
            switch (toupper(userLayout[i]))
            {
                case 'C':
                    userLayoutDims[i] = expectedLayoutDims[DIM_C];
                    break;
                case 'W':
                    userLayoutDims[i] = expectedLayoutDims[DIM_W];
                    break;
                case 'H':
                    userLayoutDims[i] = expectedLayoutDims[DIM_H];
                    break;
                case 'N':
                    userLayoutDims[i] = expectedLayoutDims[DIM_B];
                    break;
                default:
                    throw std::runtime_error("wrong frame layout passed");
                    break;
            }
        }
        return userLayoutDims;
    };

    auto generateWghDimArray = [](const std::string&                    userLayout,
                                  const std::array<unsigned, NUM_DIMS>& expectedLayoutDims) {
        std::array<unsigned, NUM_DIMS> userLayoutDims;
        for (int i = 0; i < NUM_DIMS; i++)
        {
            switch (toupper(userLayout[i]))
            {
                case 'K':
                    userLayoutDims[i] = expectedLayoutDims[WEIGHT_DIM_K];
                    break;
                case 'C':
                    userLayoutDims[i] = expectedLayoutDims[WEIGHT_DIM_C];
                    break;
                case 'S':
                    userLayoutDims[i] = expectedLayoutDims[WEIGHT_DIM_S];
                    break;
                case 'R':
                    userLayoutDims[i] = expectedLayoutDims[WEIGHT_DIM_R];
                    break;
                default:
                    throw std::runtime_error("wrong weights layout passed");
                    break;
            }
        }
        return userLayoutDims;
    };

    // create_tensor's layout
    std::array<unsigned, NUM_DIMS> ifmDimSizes = generateDimArray(ifmDataLayout.first, {nIFM, wIFM, hIFM, batch});
    std::array<unsigned, NUM_DIMS> wghDimSizes =
        generateWghDimArray(wghDataLayout.first, {params.nGroups, nOFM, params.kW, params.kH});
    std::array<unsigned, NUM_DIMS> ofmDimSizes =
        generateDimArray(ofmDataLayout.first, {params.nGroups, wOFM, hOFM, batch});

    createPersistTensors(INPUT_TENSOR,
                         MEM_INIT_RANDOM_POSITIVE,
                         nullptr,
                         ifmDimSizes.data(),
                         ifmDimSizes.size(),
                         syn_type_single,
                         ifmDataLayout.second);
    createPersistTensors(INPUT_TENSOR,
                         MEM_INIT_RANDOM_POSITIVE,
                         nullptr,
                         wghDimSizes.data(),
                         wghDimSizes.size(),
                         syn_type_single,
                         wghDataLayout.second);
    auto yTensorIndexPair = createPersistTensors(OUTPUT_TENSOR,
                                                 MEM_INIT_ALL_ZERO,
                                                 nullptr,
                                                 ofmDimSizes.data(),
                                                 ofmDimSizes.size(),
                                                 syn_type_single,
                                                 ofmDataLayout.second);

    const char* conv2D_in_layouts[]  = {ifmDataLayout.first.c_str(), wghDataLayout.first.c_str()};
    const char* conv2D_out_layouts[] = {ofmDataLayout.first.c_str()};

    addNodesToGraphs(NodeFactory::convolutionNodeTypeName,
                     (void*)&params,
                     sizeof(synConvolutionParams),
                     nullptr,
                     {DEFAULT_GRAPH_MODE_INDEX, DEFAULT_EAGER_MODE_INDEX},
                     nullptr,
                     conv2D_in_layouts,
                     conv2D_out_layouts);

    compileAndRun();

    auto pOutputBufferGraphMode = static_cast<float*>(m_hostBuffers[yTensorIndexPair.graph]);
    auto pOutputBufferEagerMode = static_cast<float*>(m_hostBuffers[yTensorIndexPair.eager]);
    for (uint64_t i = 0; i < getNumberOfElements(ofmDimSizes.data()); i++)
    {
        ASSERT_EQ(pOutputBufferGraphMode[i], pOutputBufferEagerMode[i])
            << "Graph mode mismatch at index " << i << " Graph mode:" << pOutputBufferGraphMode[i]
            << " Eager mode: " << pOutputBufferEagerMode[i];
    }
}

TEST_F_GC(SynTrainingConvDualExecutionTest, basic_L2_expected_layout)
{
    basicGroupedConvolution({"CWHN", false}, {"KCSR", false}, {"CWHN", false});
}

TEST_F_GC(SynTrainingConvDualExecutionTest, basic_L2_different_ifm_layout_allow_permutation)
{
    basicGroupedConvolution({"WHCN", true}, {"KCSR", false}, {"CWHN", false});
}

TEST_F_GC(SynTrainingConvDualExecutionTest, basic_L2_different_ifm_layout)
{
    basicGroupedConvolution({"WHCN", false}, {"KCSR", false}, {"CWHN", false});
}

TEST_F_GC(SynTrainingConvDualExecutionTest, basic_L2_different_wgh_layout_allow_permutation)
{
    basicGroupedConvolution({"CWHN", false}, {"SRCK", true}, {"CWHN", false});
}

TEST_F_GC(SynTrainingConvDualExecutionTest, basic_L2_different_wgh_layout)
{
    basicGroupedConvolution({"CWHN", false}, {"SRCK", false}, {"CWHN", false});
}

TEST_F_GC(SynTrainingConvDualExecutionTest, basic_L2_different_output_layout_allow_permutation)
{
    basicGroupedConvolution({"CWHN", false}, {"KCSR", false}, {"WHCN", true});
}

TEST_F_GC(SynTrainingConvDualExecutionTest, basic_L2_different_output_layout)
{
    basicGroupedConvolution({"CWHN", false}, {"KCSR", false}, {"WHCN", false});
}

TEST_F_GC(SynTrainingConvDualExecutionTest, basic_L2_all_different_layout_allow_permutation)
{
    basicGroupedConvolution({"WHNC", true}, {"RCKS", true}, {"NHCW", true});
}

TEST_F_GC(SynTrainingConvDualExecutionTest, basic_L2_all_different)
{
    basicGroupedConvolution({"WHNC", false}, {"RCKS", false}, {"NHCW", false});
}

TEST_F_GC(SynTrainingConvDualExecutionTest, basic_L2_all_different_layout_allow_permutation_output)
{
    basicGroupedConvolution({"WHNC", false}, {"RCKS", false}, {"NHCW", true});
}

TEST_F_GC(SynTrainingConvDualExecutionTest, basic_L2_all_different_layout_allow_permutation_one_input)
{
    basicGroupedConvolution({"WHNC", true}, {"RCKS", false}, {"NHCW", false});
}

TEST_F_GC(SynTrainingConvDualExecutionTest, basic_grouped_convolution2)
{
    synConvolutionParams params;

    params.kH      = 3;
    params.kW      = 3;
    params.padT    = 1;
    params.padB    = 1;
    params.padL    = 1;
    params.padR    = 1;
    params.nGroups = 384;

    const unsigned batch = 1;
    const unsigned nIFM  = 384;
    const unsigned nOFM  = 1;
    const unsigned wOFM  = 30;
    const unsigned hOFM  = 40;
    const unsigned wIFM  = 30;
    const unsigned hIFM  = 40;

    // create_tensor's layout
    std::array<unsigned, NUM_DIMS> ifmDimSizes = {nIFM, wIFM, hIFM, batch};
    std::array<unsigned, NUM_DIMS> wghDimSizes = {params.nGroups, nOFM, params.kW, params.kH};
    std::array<unsigned, NUM_DIMS> ofmDimSizes = {params.nGroups, wOFM, hOFM, batch};

    createPersistTensors(INPUT_TENSOR,
                         MEM_INIT_RANDOM_POSITIVE,
                         nullptr,
                         ifmDimSizes.data(),
                         ifmDimSizes.size(),
                         syn_type_single);
    createPersistTensors(INPUT_TENSOR,
                         MEM_INIT_RANDOM_POSITIVE,
                         nullptr,
                         wghDimSizes.data(),
                         wghDimSizes.size(),
                         syn_type_single);
    auto yTensorIndexPair = createPersistTensors(OUTPUT_TENSOR,
                                                 MEM_INIT_ALL_ZERO,
                                                 nullptr,
                                                 ofmDimSizes.data(),
                                                 ofmDimSizes.size(),
                                                 syn_type_single);

    addNodesToGraphs(NodeFactory::convolutionNodeTypeName, (void*)&params, sizeof(synConvolutionParams));

    compileAndRun();

    auto pOutputBufferGraphMode = static_cast<float*>(m_hostBuffers[yTensorIndexPair.graph]);
    auto pOutputBufferEagerMode = static_cast<float*>(m_hostBuffers[yTensorIndexPair.eager]);
    for (uint64_t i = 0; i < getNumberOfElements(ofmDimSizes.data()); i++)
    {
        ASSERT_LE(abs(pOutputBufferGraphMode[i] - pOutputBufferEagerMode[i]), 0.00001)
            << "Graph mode mismatch at index " << i << " Graph mode:" << pOutputBufferGraphMode[i]
            << " Eager mode: " << pOutputBufferEagerMode[i];
    }
}