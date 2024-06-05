#include "gc_gaudi_test_infra.h"
#include "syn_gaudi_two_run_compare_test.h"

TEST_F_GC(SynTrainingTestInfra, remove_casts_without_reshape_nodes)
{
    unsigned inputSize[]  = {128, 128, 1};
    unsigned middle1Size[] = {128, 128};
    unsigned middle2Size[] = {128, 128, 1, 1, 1};
    unsigned outputSize[] = {128, 512, 1, 1, 1};

    unsigned input          = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, inputSize, 3, syn_type_float, nullptr);
    unsigned castOutput1    = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, inputSize, 3, syn_type_bf16, nullptr);
    unsigned reshapeOutput1 = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, middle1Size, 2, syn_type_bf16, nullptr);
    unsigned castOutput2    = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, middle1Size, 2, syn_type_float, nullptr);
    unsigned reshapeOutput2 = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, middle2Size, 5, syn_type_float, nullptr);
    unsigned output         = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, outputSize, 5, syn_type_float, nullptr);

    addNodeToGraph("cast_f32_to_bf16", {input}, {castOutput1}, nullptr, 0, "cast1");
    addNodeToGraph("reshape", {castOutput1}, {reshapeOutput1}, nullptr, 0, "reshape1");
    addNodeToGraph("cast_bf16_to_f32", {reshapeOutput1}, {castOutput2}, nullptr, 0, "cast2");
    addNodeToGraph("reshape", {castOutput2}, {reshapeOutput2}, nullptr, 0, "reshape2");
    ns_PadKernelEx::Params params {};
    params.pads[1] = 384;
    addNodeToGraph("pad_fwd_f32", {reshapeOutput2}, {output}, (void*)&params, sizeof(params), "pad");

    compileAndRun();
}

enum class OptimizeCastsTestMode
{
    NO_LOGICAL                         = 0,
    PACKING                            = 1,
    LOGICAL_NODES_BETWEEN              = 2,
    PACKING_WITH_LOGICAL_NODES_BETWEEN = 3
};
class SynTrainingOptimizeCastsTest
: public SynGaudiTwoRunCompareTest
, public testing::WithParamInterface<
      std::tuple<OptimizeCastsTestMode, std::string, std::string, synDataType, synDataType, synDataType, bool>>
{
public:
    SynTrainingOptimizeCastsTest()
    : m_mode(std::get<0>(GetParam())),
      m_cast1Guid(std::get<1>(GetParam())),
      m_cast2Guid(std::get<2>(GetParam())),
      m_origDt(std::get<3>(GetParam())),
      m_cast1OutDt(std::get<4>(GetParam())),
      m_cast2OutDt(std::get<5>(GetParam())),
      m_enableTpcOptimizer(std::get<6>(GetParam()))
    {
    }

    struct NodeInGraph
    {
        NodeInGraph(const char*                  guid,
                    const std::vector<unsigned>& inputs,
                    const std::vector<unsigned>& outputs,
                    void*                        params       = nullptr,
                    unsigned                     sizeOfParams = 0)
        : m_guid(guid), m_inputs(inputs), m_outputs(outputs), m_params(params), m_sizeOfParams(sizeOfParams)
        {
        }

        const char*           m_guid;
        std::vector<unsigned> m_inputs;
        std::vector<unsigned> m_outputs;
        void*                 m_params;
        unsigned              m_sizeOfParams;
    };

    struct TensorInGraph
    {
        TensorInGraph(const std::vector<unsigned>& sizes, synDataType dataType, bool isPersistent, bool isInput)
        : m_sizes(sizes), m_dataType(dataType), m_isPersistent(isPersistent), m_isInput(isInput)
        {
        }

        std::vector<unsigned> m_sizes;
        const synDataType     m_dataType;
        const bool            m_isPersistent;
        const bool            m_isInput;
    };

    std::string createNodeName(const char* guid) const
    {
        static unsigned counter = 0;
        return std::string(guid) + "_" + std::to_string(counter++);
    }

    std::string createTensorName() const
    {
        static unsigned counter = 0;
        return "Tensor_" + std::to_string(counter++);
    }

    std::vector<unsigned> init(const std::vector<NodeInGraph>& nodes, std::vector<TensorInGraph>& tensors)
    {
        std::vector<unsigned> generatedTensorsIdx;
        for (auto& tensor : tensors)
        {
            unsigned tensorIdx = createTensors(1,
                                               tensor.m_isInput ? INPUT_TENSOR : OUTPUT_TENSOR,
                                               tensor.m_isPersistent,
                                               createTensorName().c_str(),
                                               tensor.m_isInput ? MEM_INIT_RANDOM_WITH_NEGATIVE : MEM_INIT_ALL_ZERO,
                                               nullptr,
                                               tensor.m_sizes.data(),
                                               tensor.m_sizes.size(),
                                               tensor.m_dataType,
                                               nullptr,
                                               0,
                                               0,
                                               nullptr,
                                               false)[0];
            generatedTensorsIdx.push_back(tensorIdx);
        }
        for (const auto& node : nodes)
        {
            TensorIndices inputs;
            TensorIndices outputs;
            for (const auto& input : node.m_inputs)
            {
                inputs.push_back(generatedTensorsIdx.at(input));
            }
            for (const auto& output : node.m_outputs)
            {
                outputs.push_back(generatedTensorsIdx.at(output));
            }
            addNodeToGraph(node.m_guid,
                           inputs,
                           outputs,
                           node.m_params,
                           node.m_sizeOfParams,
                           createNodeName(node.m_guid).c_str(),
                           0);
        }
        return generatedTensorsIdx;
    }

    void runSingleTest()
    {
        switch (m_mode)
        {
            case OptimizeCastsTestMode::NO_LOGICAL:
                runNoLogicalCase();
                break;
            case OptimizeCastsTestMode::PACKING:
                runPackingCase();
                break;
            case OptimizeCastsTestMode::LOGICAL_NODES_BETWEEN:
                runLogicalNodesBetweenCase();
                break;
            case OptimizeCastsTestMode::PACKING_WITH_LOGICAL_NODES_BETWEEN:
                runPackingWithLogicalNodesBetweenCase();
                break;
            default:
                ASSERT_TRUE(false);
        }
    }

    void runAndCheckResults(const std::vector<unsigned>& outputToCompareIdx)
    {
        GlobalConfTestSetter tpcOptimizer("ENABLE_TPC_TENSOR_SHAPE_MANIPULATION",
                                          m_enableTpcOptimizer ? "true" : "false");
        GlobalConfTestSetter tpcFuser("RUN_TPC_FUSER", "false");  // Avoid casts fusing by fuser
        addConfigurationToRun(FIRST_RUN, "ENABLE_CONTIGUOUS_CAST_REMOVAL", "false");
        addConfigurationToRun(SECOND_RUN, "ENABLE_CONTIGUOUS_CAST_REMOVAL", "true");
        compareRunsResults(outputToCompareIdx);
    }

    void runNoLogicalCase()
    {
        const std::vector<unsigned>& sizes   = {1, 256, 16};
        std::vector<TensorInGraph>   tensors = {TensorInGraph(sizes, m_origDt, true, true),        // Tensor 0
                                              TensorInGraph(sizes, m_origDt, false, false),      // Tensor 1
                                              TensorInGraph(sizes, m_cast1OutDt, false, false),  // Tensor 2
                                              TensorInGraph(sizes, m_cast2OutDt, false, false),  // Tensor 3
                                              TensorInGraph(sizes, m_cast2OutDt, true, false),   // Tensor 4
                                              TensorInGraph(sizes, m_cast2OutDt, true, false)};  // Tensor 5
        std::vector<NodeInGraph>     nodes;
        nodes.push_back(NodeInGraph("memcpy", {0}, {1}));
        nodes.push_back(NodeInGraph(m_cast1Guid.c_str(), {1}, {2}));
        nodes.push_back(NodeInGraph(m_cast2Guid.c_str(), {2}, {3}));
        nodes.push_back(NodeInGraph("memcpy", {3}, {4}));
        nodes.push_back(NodeInGraph("memcpy", {3}, {5}));
        std::vector<unsigned> generatedTensorsIdx = init(nodes, tensors);
        runAndCheckResults({generatedTensorsIdx[4], generatedTensorsIdx[5]});
    }

    void runPackingCase()
    {
        const std::vector<unsigned>& sizes         = {1, 256, 16};
        const std::vector<unsigned>& reshapedSizes = {64, 4, 16};
        std::vector<TensorInGraph>   tensors       = {TensorInGraph(sizes, m_origDt, true, true),            // Tensor 0
                                              TensorInGraph(sizes, m_origDt, false, false),          // Tensor 1
                                              TensorInGraph(reshapedSizes, m_origDt, false, false),  // Tensor 2
                                              TensorInGraph(reshapedSizes, m_cast1OutDt, false, false),  // Tensor 3
                                              TensorInGraph(sizes, m_cast1OutDt, false, false),  // Tensor 4
                                              TensorInGraph(reshapedSizes, m_cast1OutDt, false, false),  // Tensor 5
                                              TensorInGraph(reshapedSizes, m_cast2OutDt, false, false),  // Tensor 6
                                              TensorInGraph(sizes, m_cast2OutDt, false, false),  // Tensor 7
                                              TensorInGraph(sizes, m_cast2OutDt, true, false),   // Tensor 8
                                              TensorInGraph(sizes, m_cast2OutDt, true, false)};  // Tensor 9
        std::vector<NodeInGraph>     nodes;
        nodes.push_back(NodeInGraph("memcpy", {0}, {1}));
        nodes.push_back(NodeInGraph("reshape", {1}, {2}));
        nodes.push_back(NodeInGraph(m_cast1Guid.c_str(), {2}, {3}));
        nodes.push_back(NodeInGraph("reshape", {3}, {4}));
        nodes.push_back(NodeInGraph("reshape", {4}, {5}));
        nodes.push_back(NodeInGraph(m_cast2Guid.c_str(), {5}, {6}));
        nodes.push_back(NodeInGraph("reshape", {6}, {7}));
        nodes.push_back(NodeInGraph("memcpy", {7}, {8}));
        nodes.push_back(NodeInGraph("memcpy", {7}, {9}));
        std::vector<unsigned> generatedTensorsIdx = init(nodes, tensors);
        runAndCheckResults({generatedTensorsIdx[8], generatedTensorsIdx[9]});
    }

    void runLogicalNodesBetweenCase()
    {
        const std::vector<unsigned>& sizes   = {1, 256, 16};
        std::vector<TensorInGraph>   tensors = {TensorInGraph(sizes, m_origDt, true, true),        // Tensor 0
                                              TensorInGraph(sizes, m_origDt, false, false),      // Tensor 1
                                              TensorInGraph(sizes, m_cast1OutDt, false, false),  // Tensor 2
                                              TensorInGraph(sizes, m_cast1OutDt, false, false),  // Tensor 3
                                              TensorInGraph(sizes, m_cast1OutDt, false, false),  // Tensor 4
                                              TensorInGraph(sizes, m_cast1OutDt, false, false),  // Tensor 5
                                              TensorInGraph(sizes, m_cast2OutDt, false, false),  // Tensor 6
                                              TensorInGraph(sizes, m_cast2OutDt, true, false),   // Tensor 7
                                              TensorInGraph(sizes, m_cast2OutDt, true, false)};  // Tensor 8
        std::vector<NodeInGraph>     nodes;
        nodes.push_back(NodeInGraph("memcpy", {0}, {1}));
        nodes.push_back(NodeInGraph(m_cast1Guid.c_str(), {1}, {2}));
        nodes.push_back(NodeInGraph("reshape", {2}, {3}));
        nodes.push_back(NodeInGraph("identity", {3}, {4}));
        nodes.push_back(NodeInGraph("reshape", {4}, {5}));
        nodes.push_back(NodeInGraph(m_cast2Guid.c_str(), {5}, {6}));
        nodes.push_back(NodeInGraph("memcpy", {6}, {7}));
        nodes.push_back(NodeInGraph("memcpy", {6}, {8}));
        std::vector<unsigned> generatedTensorsIdx = init(nodes, tensors);
        runAndCheckResults({generatedTensorsIdx[7], generatedTensorsIdx[8]});
    }

    void runPackingWithLogicalNodesBetweenCase()
    {
        const std::vector<unsigned>& sizes          = {1, 256, 16};
        const std::vector<unsigned>& reshapedSizes  = {64, 4, 16};
        const std::vector<unsigned>& reshaped2Sizes = {64, 8, 8};
        std::vector<TensorInGraph>   tensors        = {TensorInGraph(sizes, m_origDt, true, true),    // Tensor 0
                                              TensorInGraph(sizes, m_origDt, false, false),  // Tensor 1
                                              TensorInGraph(reshapedSizes, m_origDt, false, false),  // Tensor 2
                                              TensorInGraph(reshapedSizes, m_cast1OutDt, false, false),  // Tensor 3
                                              TensorInGraph(sizes, m_cast1OutDt, false, false),  // Tensor 4
                                              TensorInGraph(reshaped2Sizes, m_cast1OutDt, false, false),  // Tensor 5
                                              TensorInGraph(reshaped2Sizes, m_cast1OutDt, false, false),  // Tensor 6
                                              TensorInGraph(sizes, m_cast1OutDt, false, false),  // Tensor 7
                                              TensorInGraph(reshapedSizes, m_cast1OutDt, false, false),  // Tensor 8
                                              TensorInGraph(reshapedSizes, m_cast2OutDt, false, false),  // Tensor 9
                                              TensorInGraph(sizes, m_cast2OutDt, false, false),  // Tensor 10
                                              TensorInGraph(sizes, m_cast2OutDt, true, false),   // Tensor 11
                                              TensorInGraph(sizes, m_cast2OutDt, true, false)};  // Tensor 12
        std::vector<NodeInGraph>     nodes;
        nodes.push_back(NodeInGraph("memcpy", {0}, {1}));
        nodes.push_back(NodeInGraph("reshape", {1}, {2}));
        nodes.push_back(NodeInGraph(m_cast1Guid.c_str(), {2}, {3}));
        nodes.push_back(NodeInGraph("reshape", {3}, {4}));
        nodes.push_back(NodeInGraph("reshape", {4}, {5}));
        nodes.push_back(NodeInGraph("identity", {5}, {6}));
        nodes.push_back(NodeInGraph("reshape", {6}, {7}));
        nodes.push_back(NodeInGraph("reshape", {7}, {8}));
        nodes.push_back(NodeInGraph(m_cast2Guid.c_str(), {8}, {9}));
        nodes.push_back(NodeInGraph("reshape", {9}, {10}));
        nodes.push_back(NodeInGraph("memcpy", {10}, {11}));
        nodes.push_back(NodeInGraph("memcpy", {10}, {12}));
        std::vector<unsigned> generatedTensorsIdx = init(nodes, tensors);
        runAndCheckResults({generatedTensorsIdx[11], generatedTensorsIdx[12]});
    }

protected:
    const OptimizeCastsTestMode m_mode;
    const std::string           m_cast1Guid;
    const std::string           m_cast2Guid;
    const synDataType           m_origDt;
    const synDataType           m_cast1OutDt;
    const synDataType           m_cast2OutDt;
    const bool                  m_enableTpcOptimizer;
};

TEST_P_GC(SynTrainingOptimizeCastsTest, optimize_casts_test)
{
    runSingleTest();
}

INSTANTIATE_TEST_SUITE_P(remove_opposite_casts,
                         SynTrainingOptimizeCastsTest,
                         testing::Combine(testing::Values(OptimizeCastsTestMode::NO_LOGICAL,
                                                          OptimizeCastsTestMode::PACKING,
                                                          OptimizeCastsTestMode::LOGICAL_NODES_BETWEEN,
                                                          OptimizeCastsTestMode::PACKING_WITH_LOGICAL_NODES_BETWEEN),
                                          testing::Values("cast_f32_to_bf16"),
                                          testing::Values("cast_bf16_to_f32"),
                                          testing::Values(syn_type_float),
                                          testing::Values(syn_type_bf16),
                                          testing::Values(syn_type_float),
                                          testing::Values(true, false)));

INSTANTIATE_TEST_SUITE_P(merge_casts,
                         SynTrainingOptimizeCastsTest,
                         testing::Combine(testing::Values(OptimizeCastsTestMode::NO_LOGICAL,
                                                          OptimizeCastsTestMode::PACKING,
                                                          OptimizeCastsTestMode::LOGICAL_NODES_BETWEEN,
                                                          OptimizeCastsTestMode::PACKING_WITH_LOGICAL_NODES_BETWEEN),
                                          testing::Values("cast_i8_to_f32"),
                                          testing::Values("cast_f32_to_bf16"),
                                          testing::Values(syn_type_int8),
                                          testing::Values(syn_type_float),
                                          testing::Values(syn_type_bf16),
                                          testing::Values(true, false)));

TEST_F_GC(SynGaudiTwoRunCompareTest, remove_casts_after_sram_reduction_ASIC_CI)
{
    GlobalConfTestSetter tpcFuser("RUN_TPC_FUSER", "false");  // Disable fuser to avoid fusing of cast+relu

    unsigned dySizes[] = {512, 80, 40, 16};
    unsigned xSizes[]  = {512, 80, 40, 16};
    unsigned dwSizes[] = {512, 512, 3, 3};

    unsigned dy =
        createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, dySizes, 4, syn_type_bf16, nullptr);
    unsigned x =
        createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, xSizes, 4, syn_type_bf16, nullptr);
    unsigned dw      = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, dwSizes, 4, syn_type_bf16, nullptr);
    unsigned castOut = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, dwSizes, 4, syn_type_single, nullptr);
    unsigned reluOut =
        createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, dwSizes, 4, syn_type_single, nullptr);

    synConvolutionParams params(3, 3, 1, 1, 1, 1, 1, 1, 1, 1);

    addNodeToGraph("dedw", {dy, x}, {dw}, &params, sizeof(params), "DEDW");
    addNodeToGraph("cast_bf16_to_f32", {dw}, {castOut}, nullptr, 0, "CAST");
    addNodeToGraph("relu_fwd_f32", {castOut}, {reluOut}, nullptr, 0, "RELU");

    // The DEDW will be sliced on common-dim and use SRAM reduction in F32.
    // Since the DEDW output should be BF16 the sram-manager will add F32->BF16 cast.
    // This cast will be eliminated together with the user cast_bf16_to_f32 node (opposite casts).
    // Make sure the results are the same when this optimization is disabled.
    addConfigurationToRun(FIRST_RUN, "ENABLE_CONTIGUOUS_CAST_REMOVAL", "false");
    addConfigurationToRun(SECOND_RUN, "ENABLE_CONTIGUOUS_CAST_REMOVAL", "true");

    compareRunsResults({reluOut});
}