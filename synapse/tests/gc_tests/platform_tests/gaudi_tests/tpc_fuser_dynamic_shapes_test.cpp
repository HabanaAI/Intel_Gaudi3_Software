#include "gc_dynamic_shapes_infra.h"
#include "syn_singleton.hpp"

class SynGaudiTPCFuserDynamicTest : public SynGaudiTestInfra
{
public:
    SynGaudiTPCFuserDynamicTest()
    {
        setTestPackage(TEST_PACKAGE_DSD);
    }

protected:
    void SetUp()
    {
        SynGaudiTestInfra::SetUp();
    }

    void TearDown()
    {
        SynGaudiTestInfra::TearDown();
    };
};

TEST_F_GC(SynGaudiTPCFuserDynamicTest, tpc_fuser_fuse_dynamic_relu_nodes)
{
    // Graph have two TPC nodes:  [relu_fwd]->[relu_bwd]
    // that will be fused to one TPC node
    // All tensors ae dynamic and will get actual size during run-time
    // The test verify the output

    unsigned maxSizes[] = {10, 10};
    unsigned minSizes[] = {0, 0};

    ns_ReluKernel::Params reluParams = {0};
    reluParams.threshold.i = 0;

    unsigned inTensor  = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, maxSizes, 2,
                                             syn_type_float, nullptr, nullptr, 0, 0,
                                             nullptr, minSizes);

    unsigned fwdOut = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, maxSizes, 2,
                                   syn_type_float, nullptr, minSizes);

    auto outTensor = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, maxSizes, 2,
                                         syn_type_float, nullptr, nullptr, 0, 0,
                                         nullptr, minSizes);

    addNodeToGraph("relu_fwd_f32", {inTensor}, {fwdOut}, &reluParams, sizeof(reluParams));

    unsigned bwdIn1 = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, maxSizes, 2,
                                          syn_type_float, nullptr, nullptr, 0, 0,
                                          nullptr, minSizes);

    unsigned bwdIn2 = connectOutputTensorToInputTensor(fwdOut);

    reluParams.threshold.f = 0.;
    addNodeToGraph("relu_bwd_f32", {bwdIn1, bwdIn2}, {outTensor}, &reluParams, sizeof(reluParams));

    compileTopology();

    unsigned inActualSize[] = {8, 8};
    setActualSizes(inTensor, inActualSize);
    setActualSizes(fwdOut, inActualSize);
    setActualSizes(bwdIn1, inActualSize);
    setActualSizes(outTensor, inActualSize);

    runTopology();

    float* pFwdInput  = (float*)m_hostBuffers[inTensor];
    float* pBwdInput  = (float*)m_hostBuffers[bwdIn1];
    float* pBwdOutput = (float*)m_hostBuffers[outTensor];

    unsigned outputNumElements = inActualSize[0]*inActualSize[1];

    for (unsigned i = 0; i < outputNumElements; i++)
    {
        float expectedResult = (*pFwdInput > 0.) ? *pBwdInput : 0;
        ASSERT_EQ(expectedResult, *pBwdOutput) << "Mismatch for at index " << i
                                               << " Expected:"             << expectedResult
                                               << " BwdOutput: "           << *pBwdOutput
                                               << " FwdInput: "            << *pFwdInput
                                               << " BwdInput "             << *pBwdInput;
        pFwdInput++;
        pBwdInput++;
        pBwdOutput++;
    }
}

TEST_F_GC(SynGaudiTPCFuserDynamicTest, tpc_fuser_fuse_3_add_dynamic)
{
    unsigned maxSizes[] = {400, 400};
    unsigned minSizes[] = {0, 0};

    unsigned in1_1 = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, maxSizes, 2,
                                         syn_type_float, nullptr, nullptr, 0, 0,
                                         nullptr, minSizes);
    unsigned in1_2 = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, maxSizes, 2,
                                         syn_type_float, nullptr, nullptr, 0, 0,
                                         nullptr, minSizes);
    unsigned out1  = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, maxSizes, 2,
                                  syn_type_float, nullptr, minSizes);

    unsigned in2_1 = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, maxSizes, 2,
                                         syn_type_float, nullptr, nullptr, 0, 0,
                                         nullptr, minSizes);
    unsigned in2_2 = connectOutputTensorToInputTensor(out1);
    unsigned out2  = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, maxSizes, 2,
                                  syn_type_float, nullptr, minSizes);

    unsigned in3_1 = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, maxSizes, 2,
                                         syn_type_float, nullptr, nullptr, 0, 0,
                                         nullptr, minSizes);
    unsigned in3_2 = connectOutputTensorToInputTensor(out2);
    unsigned out3  = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, maxSizes, 2,
                                         syn_type_float, nullptr, nullptr, 0, 0,
                                         nullptr, minSizes);

    addNodeToGraph("add_fwd_f32", {in1_1, in1_2}, {out1});
    addNodeToGraph("add_fwd_f32", {in2_1, in2_2}, {out2});
    addNodeToGraph("add_fwd_f32", {in3_1, in3_2}, {out3});

    HabanaGraph* graph = synSingleton::getInstanceInternal()->getGraph(m_graphs.front().graphHandle);

    compileTopology();

    bool isThereFusedKernel = false;
    for (auto node : graph->getExeSortedNodes())
    {
        auto nodeGuid = node->getGUID();
        if (nodeGuid.find("fused_kernel") != std::string::npos)
        {
            isThereFusedKernel = true;
        }
    }
    ASSERT_TRUE(isThereFusedKernel) << "Expected fused kernel";

    ASSERT_NE(m_graphs[0].recipeHandle->basicRecipeHandle.recipe, nullptr);
    shape_plane_graph_t* recipe = m_graphs[0].recipeHandle->basicRecipeHandle.shape_plan_recipe;
    ASSERT_NE(recipe, nullptr);

    unsigned inActualSize[] = {256, 256};
    setActualSizes(in1_1, inActualSize);
    setActualSizes(in1_2, inActualSize);
    setActualSizes(out1, inActualSize);
    setActualSizes(in2_1, inActualSize);
    setActualSizes(in2_2, inActualSize);
    setActualSizes(out2, inActualSize);
    setActualSizes(in3_1, inActualSize);
    setActualSizes(in3_2, inActualSize);
    setActualSizes(out3, inActualSize);

    runTopology();

    float* input1 = (float*)m_hostBuffers[in1_1];
    float* input2 = (float*)m_hostBuffers[in1_2];
    float* input3 = (float*)m_hostBuffers[in2_1];
    float* input4 = (float*)m_hostBuffers[in3_1];

    float* output3 = (float*)m_hostBuffers[out3];

    unsigned outputNumElements = inActualSize[0]*inActualSize[1];

    for (unsigned idx = 0; idx < outputNumElements; idx++)
    {
        float expected_out = *input1 + *input2 + *input3 + *input4;

        ASSERT_EQ(expected_out, *output3) << "OUTPUT3: Mismatch for at index " << idx
                                          << " |Expected:" << expected_out
                                          << " |Result: " << *output3
                                          << " |Operands: "
                                          << *input1 << ", " << *input2 << ", " << *input3 << ", " << *input4;


        input1++;
        input2++;
        input3++;
        input4++;

        output3++;
    }
}
