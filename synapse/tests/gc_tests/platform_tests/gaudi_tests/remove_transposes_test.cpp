#include "gc_gaudi_test_infra.h"
#include "syn_gaudi_two_run_compare_test.h"
#include "synapse_common_types.h"

class SynTrainingOptimizeTransposesTest : public SynGaudiTwoRunCompareTest
{
public:
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
        TensorInGraph(const std::vector<unsigned>& sizes, bool isPersistent, bool isInput)
        : m_sizes(sizes), m_isPersistent(isPersistent), m_isInput(isInput)
        {
        }

        std::vector<unsigned> m_sizes;
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
                                               syn_type_single,
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

    void runAndCheckResults(const std::vector<unsigned>& outputToCompareIdx)
    {
        addConfigurationToRun(FIRST_RUN, "ENABLE_CONTIGUOUS_TRANSPOSE_REMOVAL", "false");
        addConfigurationToRun(SECOND_RUN, "ENABLE_CONTIGUOUS_TRANSPOSE_REMOVAL", "true");
        compareRunsResults(outputToCompareIdx);
    }
};

// input: T0 -> T1 is identity.
// +---+    +----+    +----+    +---+
// | A |--->| T0 |--->| T1 |--->| B |
// +---+    +----+    +----+    +---+
// output:
// +---+    +---+
// | A |--->| B |
// +---+    +---+
TEST_F_GC(SynTrainingOptimizeTransposesTest, basic_opposite_transposes)
{
    const std::vector<unsigned>& sizes           = {32, 32};
    std::vector<TensorInGraph>   tensors         = {TensorInGraph(sizes, true, true),    // Tensor 0
                                          TensorInGraph(sizes, false, false),  // Tensor 1
                                          TensorInGraph(sizes, false, false),  // Tensor 2
                                          TensorInGraph(sizes, false, false),  // Tensor 3
                                          TensorInGraph(sizes, true, false)};  // Tensor 4
    synTransposeParams           transposeParams = {{TPD_Width, TPD_Channel}, 2};
    std::vector<NodeInGraph>     nodes;
    nodes.push_back(NodeInGraph("relu_fwd_f32", {0}, {1}));
    nodes.push_back(NodeInGraph("transpose", {1}, {2}, &transposeParams, sizeof(transposeParams)));
    nodes.push_back(NodeInGraph("transpose", {2}, {3}, &transposeParams, sizeof(transposeParams)));
    nodes.push_back(NodeInGraph("relu_fwd_f32", {3}, {4}));
    std::vector<unsigned> generatedTensorsIdx = init(nodes, tensors);
    runAndCheckResults({generatedTensorsIdx[4]});
}

// input: T0 -> T1 is identity
// +---+    +----+    +----+
// | A |--->| T0 |--->| T1 |
// +---+    +----+    +----+
// output:
// +---+
// | A |
// +---+
TEST_F_GC(SynTrainingOptimizeTransposesTest, seq_output_is_graph_output)
{
    const std::vector<unsigned>& sizes   = {32, 32};
    std::vector<TensorInGraph>   tensors = {TensorInGraph(sizes, true, true),    // Tensor 0
                                          TensorInGraph(sizes, false, false),  // Tensor 1
                                          TensorInGraph(sizes, false, false),  // Tensor 2
                                          TensorInGraph(sizes, true, false)};  // Tensor 3
    std::vector<NodeInGraph>     nodes;
    synTransposeParams           transposeParams = {{TPD_Width, TPD_Channel}, 2};
    nodes.push_back(NodeInGraph("relu_fwd_f32", {0}, {1}));
    nodes.push_back(NodeInGraph("transpose", {1}, {2}, &transposeParams, sizeof(transposeParams)));
    nodes.push_back(NodeInGraph("transpose", {2}, {3}, &transposeParams, sizeof(transposeParams)));
    std::vector<unsigned> generatedTensorsIdx = init(nodes, tensors);
    runAndCheckResults({generatedTensorsIdx[3]});
}

// input: T0 -> T1 is identity
// +----+    +----+
// | T0 |--->| T1 |
// +----+    +----+
// output:
// +--------+
// | memcpy |
// +--------+
TEST_F_GC(SynTrainingOptimizeTransposesTest, seq_io_are_graph_io)
{
    const std::vector<unsigned>& sizes   = {32, 32};
    std::vector<TensorInGraph>   tensors = {TensorInGraph(sizes, true, true),    // Tensor 0
                                          TensorInGraph(sizes, false, false),  // Tensor 1
                                          TensorInGraph(sizes, true, false)};  // Tensor 2
    std::vector<NodeInGraph>     nodes;
    synTransposeParams           transposeParams = {{TPD_Width, TPD_Channel}, 2};
    nodes.push_back(NodeInGraph("transpose", {0}, {1}, &transposeParams, sizeof(transposeParams)));
    nodes.push_back(NodeInGraph("transpose", {1}, {2}, &transposeParams, sizeof(transposeParams)));
    std::vector<unsigned> generatedTensorsIdx = init(nodes, tensors);
    runAndCheckResults({generatedTensorsIdx[2]});
}

// input: T0 -> T1 is identity
// +----+    +----+    +----+   +----+   +----+
// | A  +--->| T0 +--->| T1 +-->| T2 +-->| B  |
// +----+    +----+    +-+--+   +----+   +----+
//                       |
//                       v
//                     +----+
//                     | C  |
//                     +----+
// output:
//   +-----------------------------+
//   |                             |
// +-+--+    +----+    +----+   +--v-+
// | A  +--->| T0 +--->| T1 |   | B  |
// +----+    +----+    +-+--+   +----+
//                       |
//                       v
//                     +----+
//                     | C  |
//                     +----+
TEST_F_GC(SynTrainingOptimizeTransposesTest, middle_node_with_multiple_consumers)
{
    const std::vector<unsigned>& sizes   = {32, 32, 32, 32};
    std::vector<TensorInGraph>   tensors = {TensorInGraph(sizes, true, true),    // Tensor 0
                                          TensorInGraph(sizes, false, false),  // Tensor 1
                                          TensorInGraph(sizes, false, false),  // Tensor 2
                                          TensorInGraph(sizes, false, false),  // Tensor 3
                                          TensorInGraph(sizes, false, false),  // Tensor 4
                                          TensorInGraph(sizes, true, false),   // Tensor 5
                                          TensorInGraph(sizes, true, false)};  // Tensor 6
    std::vector<NodeInGraph>     nodes;
    synTransposeParams           transposeParamsT0 = {{TPD_Width, TPD_Channel, TPD_4Dim_Batch, TPD_Height}, 4};
    synTransposeParams           transposeParamsT1 = {{TPD_Width, TPD_Channel, TPD_Height, TPD_4Dim_Batch}, 4};
    synTransposeParams           transposeParamsT2 = {{TPD_Channel, TPD_Width, TPD_4Dim_Batch, TPD_Height}, 4};
    nodes.push_back(NodeInGraph("relu_fwd_f32", {0}, {1}));                                              // A
    nodes.push_back(NodeInGraph("transpose", {1}, {2}, &transposeParamsT0, sizeof(transposeParamsT0)));  // T0
    nodes.push_back(NodeInGraph("transpose", {2}, {3}, &transposeParamsT1, sizeof(transposeParamsT1)));  // T1
    nodes.push_back(NodeInGraph("transpose", {3}, {4}, &transposeParamsT2, sizeof(transposeParamsT2)));  // T2
    nodes.push_back(NodeInGraph("relu_fwd_f32", {3}, {5}));                                              // C
    nodes.push_back(NodeInGraph("relu_fwd_f32", {4}, {6}));                                              // B
    std::vector<unsigned> generatedTensorsIdx = init(nodes, tensors);
    runAndCheckResults({generatedTensorsIdx[6], generatedTensorsIdx[5]});
}

// input: T0 -> T1 is identity, T0 output is persistent
// +---+    +----+    +----+    +---+
// | A |--->| T0 |--->| T1 |--->| B |
// +---+    +----+    +----+    +---+
// output:
//   +-------------------+
//   |                   |
// +-+--+    +----+   +--v-+
// | A  +--->| T0 |   | B  |
// +----+    +-+--+   +----+
TEST_F_GC(SynTrainingOptimizeTransposesTest, middle_node_with_persistent_output)
{
    const std::vector<unsigned>& sizes   = {32, 32, 32, 32};
    std::vector<TensorInGraph>   tensors = {TensorInGraph(sizes, true, true),    // Tensor 0
                                          TensorInGraph(sizes, false, false),  // Tensor 1
                                          TensorInGraph(sizes, true, false),   // Tensor 2
                                          TensorInGraph(sizes, false, false),  // Tensor 3
                                          TensorInGraph(sizes, true, false)};  // Tensor 4
    std::vector<NodeInGraph>     nodes;
    synTransposeParams           transposeParamsT0 = {{TPD_Width, TPD_Channel, TPD_Height, TPD_4Dim_Batch}, 4};
    synTransposeParams           transposeParamsT1 = {{TPD_Width, TPD_Channel, TPD_Height, TPD_4Dim_Batch}, 4};
    nodes.push_back(NodeInGraph("relu_fwd_f32", {0}, {1}));                                              // A
    nodes.push_back(NodeInGraph("transpose", {1}, {2}, &transposeParamsT0, sizeof(transposeParamsT0)));  // T9
    nodes.push_back(NodeInGraph("transpose", {2}, {3}, &transposeParamsT1, sizeof(transposeParamsT1)));  // T1
    nodes.push_back(NodeInGraph("relu_fwd_f32", {3}, {4}));                                              // B
    std::vector<unsigned> generatedTensorsIdx = init(nodes, tensors);
    runAndCheckResults({generatedTensorsIdx[2], generatedTensorsIdx[4]});
}

// input: T0 -> T1 is identity, T1 output is persistent
// +----+    +----+    +---+
// | T0 |--->| T1 |--->| A |
// +----+    +----+    +---+
// output:
// +--------+    +---+
// | memcpy |--->| A |
// +--------+    +---+
TEST_F_GC(SynTrainingOptimizeTransposesTest, last_node_with_persistent_output)
{
    const std::vector<unsigned>& sizes   = {32, 32, 32, 32};
    std::vector<TensorInGraph>   tensors = {TensorInGraph(sizes, true, true),    // Tensor 0
                                          TensorInGraph(sizes, false, false),  // Tensor 1
                                          TensorInGraph(sizes, true, false),   // Tensor 2
                                          TensorInGraph(sizes, true, false)};  // Tensor 3
    std::vector<NodeInGraph>     nodes;
    synTransposeParams           transposeParamsT0 = {{TPD_Width, TPD_Channel, TPD_Height, TPD_4Dim_Batch}, 4};
    synTransposeParams           transposeParamsT1 = {{TPD_Width, TPD_Channel, TPD_Height, TPD_4Dim_Batch}, 4};
    nodes.push_back(NodeInGraph("transpose", {0}, {1}, &transposeParamsT0, sizeof(transposeParamsT0)));  // T0
    nodes.push_back(NodeInGraph("transpose", {1}, {2}, &transposeParamsT1, sizeof(transposeParamsT1)));  // T1
    nodes.push_back(NodeInGraph("relu_fwd_f32", {2}, {3}));                                              // A
    std::vector<unsigned> generatedTensorsIdx = init(nodes, tensors);
    runAndCheckResults({generatedTensorsIdx[3], generatedTensorsIdx[2]});
}

// input: T0 -> T1 is identity
//  +----+    +----+    +----+    +----+
//  | A  +--->| T0 +--->| T1 +--->| C  |
//  +----+    +----+    +-+--+    +----+
//                        |
//                        |
//                      +-v--+
//                      | B  |
//                      +----+
// output:
//   +----+     +----+
//   | A  +---> | C  |
//   +-+--+     +----+
//     |
//     v
//   +----+
//   | B  |
//   +----+
TEST_F_GC(SynTrainingOptimizeTransposesTest, last_node_with_multiple_consumers)
{
    const std::vector<unsigned>& sizes   = {32, 32, 32, 32};
    std::vector<TensorInGraph>   tensors = {TensorInGraph(sizes, true, true),    // Tensor 0
                                          TensorInGraph(sizes, false, false),  // Tensor 1
                                          TensorInGraph(sizes, false, false),  // Tensor 2
                                          TensorInGraph(sizes, false, false),  // Tensor 3
                                          TensorInGraph(sizes, true, false),   // Tensor 4
                                          TensorInGraph(sizes, true, false)};  // Tensor 5
    std::vector<NodeInGraph>     nodes;
    synTransposeParams           transposeParamsT0 = {{TPD_Width, TPD_Channel, TPD_Height, TPD_4Dim_Batch}, 4};
    synTransposeParams           transposeParamsT1 = {{TPD_Width, TPD_Channel, TPD_Height, TPD_4Dim_Batch}, 4};
    nodes.push_back(NodeInGraph("relu_fwd_f32", {0}, {1}));                                              // A
    nodes.push_back(NodeInGraph("transpose", {1}, {2}, &transposeParamsT0, sizeof(transposeParamsT0)));  // T0
    nodes.push_back(NodeInGraph("transpose", {2}, {3}, &transposeParamsT1, sizeof(transposeParamsT1)));  // T1
    nodes.push_back(NodeInGraph("relu_fwd_f32", {3}, {4}));                                              // B
    nodes.push_back(NodeInGraph("relu_fwd_f32", {3}, {5}));                                              // C
    std::vector<unsigned> generatedTensorsIdx = init(nodes, tensors);
    runAndCheckResults({generatedTensorsIdx[4], generatedTensorsIdx[5]});
}

// input: T1 -> T2 and T0 -> T1 -> T3 are identity sequences.
// +----+    +----+    +----+    +----+    +----+
// | A  +--->| T0 +--->| T1 +--->| T2 +--->| B  |
// +----+    +----+    +-+--+    +----+    +----+
//                       |
//                       |
//                     +-v--+    +----+    +----+
//                     | T3 +--->| T4 +--->| C  |
//                     +----+    +----+    +----+
// output:
//  +----+    +-+--+    +----+
//  | A  +--->| T0 +--->| B  |
//  +--+-+    +----+    +----+
//     |
//     v
//  +----+    +----+
//  | T4 +--->| C  |
//  +----+    +----+
TEST_F_GC(SynTrainingOptimizeTransposesTest, two_subsequences_are_identity_no_overlap)
{
    const std::vector<unsigned>& sizes   = {32, 32, 32, 32};
    std::vector<TensorInGraph>   tensors = {TensorInGraph(sizes, true, true),    // Tensor 0
                                          TensorInGraph(sizes, false, false),  // Tensor 1
                                          TensorInGraph(sizes, false, false),  // Tensor 2
                                          TensorInGraph(sizes, false, false),  // Tensor 3
                                          TensorInGraph(sizes, false, false),  // Tensor 4
                                          TensorInGraph(sizes, true, false),   // Tensor 5
                                          TensorInGraph(sizes, false, false),  // Tensor 6
                                          TensorInGraph(sizes, false, false),  // Tensor 7
                                          TensorInGraph(sizes, true, false)};  // Tensor 8

    std::vector<NodeInGraph> nodes;
    synTransposeParams       transposeParamsT0 = {{TPD_Width, TPD_Channel, TPD_4Dim_Batch, TPD_Height}, 4};
    synTransposeParams       transposeParamsT1 = {{TPD_Width, TPD_Channel, TPD_Height, TPD_4Dim_Batch}, 4};
    synTransposeParams       transposeParamsT2 = {{TPD_Width, TPD_Channel, TPD_Height, TPD_4Dim_Batch}, 4};
    synTransposeParams       transposeParamsT3 = {{TPD_Channel, TPD_Width, TPD_4Dim_Batch, TPD_Height}, 4};

    nodes.push_back(NodeInGraph("relu_fwd_f32", {0}, {1}));                                               // A
    nodes.push_back(NodeInGraph("transpose", {1}, {2}, &transposeParamsT0, sizeof(synTransposeParams)));  // T0
    nodes.push_back(NodeInGraph("transpose", {2}, {3}, &transposeParamsT1, sizeof(synTransposeParams)));  // T1
    nodes.push_back(NodeInGraph("transpose", {3}, {4}, &transposeParamsT2, sizeof(synTransposeParams)));  // T2
    nodes.push_back(NodeInGraph("relu_fwd_f32", {4}, {5}));                                               // B
    nodes.push_back(NodeInGraph("transpose", {3}, {6}, &transposeParamsT3, sizeof(synTransposeParams)));  // T3
    nodes.push_back(NodeInGraph("transpose", {6}, {7}, &transposeParamsT2, sizeof(synTransposeParams)));  // T4
    nodes.push_back(NodeInGraph("relu_fwd_f32", {7}, {8}));                                               // C
    std::vector<unsigned> generatedTensorsIdx = init(nodes, tensors);
    runAndCheckResults({generatedTensorsIdx[8], generatedTensorsIdx[5]});
}

// input: T0 is identity.
// +---+    +----+    +---+
// | A |--->| T0 |--->| B |
// +---+    +----+    +---+
// output:
// +---+    +---+
// | A |--->| B |
// +---+    +---+
TEST_F_GC(SynTrainingOptimizeTransposesTest, single_identity_transpose)
{
    const std::vector<unsigned>& sizes           = {32, 32};
    std::vector<TensorInGraph>   tensors         = {TensorInGraph(sizes, true, true),    // Tensor 0
                                          TensorInGraph(sizes, false, false),  // Tensor 1
                                          TensorInGraph(sizes, false, false),  // Tensor 2
                                          TensorInGraph(sizes, true, false)};  // Tensor 3
    synTransposeParams           transposeParams = {{TPD_Channel, TPD_Width}, 2};
    std::vector<NodeInGraph>     nodes;
    nodes.push_back(NodeInGraph("relu_fwd_f32", {0}, {1}));                                          // A
    nodes.push_back(NodeInGraph("transpose", {1}, {2}, &transposeParams, sizeof(transposeParams)));  // T0
    nodes.push_back(NodeInGraph("relu_fwd_f32", {2}, {3}));                                          // B
    std::vector<unsigned> generatedTensorsIdx = init(nodes, tensors);
    runAndCheckResults({generatedTensorsIdx[3]});
}

// input: T0-> T1, and T1->T2->T3 is identity.
// +----+    +----+    +----+    +----+    +----+    +----+
// | A  +--->| T0 +--->| T1 +--->| T2 +--->| T3 +--->| B  |
// +----+    +----+    +----+    +----+    +----+    +----+
// output:
// +----+    +----+    +----+
// | A  +--->| T0 +--->| B  |
// +----+    +----+    +----+
TEST_F_GC(SynTrainingOptimizeTransposesTest, overlapping_sequences)
{
    const std::vector<unsigned>& sizes   = {32, 32, 32, 32};
    std::vector<TensorInGraph>   tensors = {TensorInGraph(sizes, true, true),    // Tensor 0
                                          TensorInGraph(sizes, false, false),  // Tensor 1
                                          TensorInGraph(sizes, false, false),  // Tensor 2
                                          TensorInGraph(sizes, false, false),  // Tensor 3
                                          TensorInGraph(sizes, false, false),  // Tensor 4
                                          TensorInGraph(sizes, false, false),  // Tensor 5
                                          TensorInGraph(sizes, true, false)};  // Tensor 6

    std::vector<NodeInGraph> nodes;
    synTransposeParams       transposeParamsT0 = {{TPD_Width, TPD_Channel, TPD_Height, TPD_4Dim_Batch}, 4};
    synTransposeParams       transposeParamsT1 = {{TPD_Width, TPD_Channel, TPD_Height, TPD_4Dim_Batch}, 4};
    synTransposeParams       transposeParamsT2 = {{TPD_Width, TPD_Channel, TPD_4Dim_Batch, TPD_Height}, 4};
    synTransposeParams       transposeParamsT3 = {{TPD_Channel, TPD_Width, TPD_4Dim_Batch, TPD_Height}, 4};

    nodes.push_back(NodeInGraph("relu_fwd_f32", {0}, {1}));                                               // A
    nodes.push_back(NodeInGraph("transpose", {1}, {2}, &transposeParamsT0, sizeof(synTransposeParams)));  // T0
    nodes.push_back(NodeInGraph("transpose", {2}, {3}, &transposeParamsT1, sizeof(synTransposeParams)));  // T1
    nodes.push_back(NodeInGraph("transpose", {3}, {4}, &transposeParamsT2, sizeof(synTransposeParams)));  // T2
    nodes.push_back(NodeInGraph("transpose", {4}, {5}, &transposeParamsT3, sizeof(synTransposeParams)));  // T3
    nodes.push_back(NodeInGraph("relu_fwd_f32", {5}, {6}));                                               // B
    std::vector<unsigned> generatedTensorsIdx = init(nodes, tensors);
    runAndCheckResults({generatedTensorsIdx[6]});
}