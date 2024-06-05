#include "../gaudi_tests/syn_gaudi_two_run_compare_test.h"
#include "gaudi_tests/gc_gaudi_test_infra.h"
#include "node_factory.h"
#include "scoped_configuration_change.h"
#include "synapse_common_types.h"
#include "tensor.h"
#include <numeric>
#include "transpose_utils.h"
#include "syn_singleton.hpp"
#include <vector>
#include "64_bit_slice_test_huge_tensors.h"

class SynTrainingBroadcastHugeTensorTest
: public SynTrainingTwoRunCompareTest
, public testing::WithParamInterface<std::tuple<TestSizeVec, TestSizeVec, bool>>
{
protected:
    void run(bool dynamicShapeTest)
    {
        ScopedConfigurationChange conf("ENABLE_HUGE_TENSOR_SLICING", "true");
        // Since there is no MME node in the graph we need to disable manually the max path scheduler
        ScopedConfigurationChange conf2("ENABLE_MAX_PATH_SCHEDULE", "false");
        auto [inputSizes, outputSizes, runAndValidate] = GetParam();

        runAndValidate &= (m_deviceType != synDeviceGaudi);  // [SW-145665] remove this line

        unsigned in  = createPersistTensor(INPUT_TENSOR,
                                          runAndValidate ? MEM_INIT_RANDOM_WITH_NEGATIVE : MEM_INIT_COMPILATION_ONLY,
                                          nullptr,
                                          inputSizes.data(),
                                          inputSizes.size());
        unsigned out   = createPersistTensor(OUTPUT_TENSOR,
                                           runAndValidate ? MEM_INIT_ALL_ZERO : MEM_INIT_COMPILATION_ONLY,
                                           nullptr,
                                           outputSizes.data(),
                                           outputSizes.size(),
                                           syn_type_single,
                                           nullptr,
                                           nullptr,
                                           0,
                                           0,
                                           nullptr,
                                           dynamicShapeTest ? inputSizes.data() : outputSizes.data());
        unsigned shape = createShapeTensor(OUTPUT_TENSOR,
                                           outputSizes.data(),
                                           dynamicShapeTest ? inputSizes.data() : outputSizes.data(),
                                           outputSizes.size());

        TensorIndices inputs = dynamicShapeTest ? TensorIndices({in, shape}) : TensorIndices({in});
        addNodeToGraph("broadcast", inputs, {out}, nullptr, 0);

        addConfigurationToRun(FIRST_RUN, "ENABLE_HUGE_TENSOR_SLICING", "true");
        addConfigurationToRun(FIRST_RUN, "MAKE_BROADCAST_PHYSICAL", "true");
        addConfigurationToRun(SECOND_RUN, "ENABLE_HUGE_TENSOR_SLICING", "false");
        addConfigurationToRun(SECOND_RUN, "MAKE_BROADCAST_PHYSICAL", "false");

        if (runAndValidate)
        {
            setActualSizes(out, outputSizes.data());
            setActualSizes(shape, outputSizes.data());
            compareRunsResults({out});
        }
        else
        {
            compileTopology("", FIRST_RUN);
        }
    }
};

TEST_P_GC(SynTrainingBroadcastHugeTensorTest,
          broadcast_huge_tensor_ASIC_CI,
          {synDeviceGaudi3, synDeviceGaudi2, synDeviceGaudi})
{
    run(false /* Dynamic shapes test*/);
}

TEST_P_GC(SynTrainingBroadcastHugeTensorTest, broadcast_huge_tensor_dsd_ASIC_CI, {synDeviceGaudi2, synDeviceGaudi})
{
    run(true /* Dynamic shapes test*/);
}

INSTANTIATE_TEST_SUITE_P(
    ,
    SynTrainingBroadcastHugeTensorTest,
    ::testing::Values(
        std::make_tuple(TestSizeVec({9216, 9216, 1}), TestSizeVec({9216, 9216, 30}), false /* run & compare results */
                        ),
        std::make_tuple(TestSizeVec({9216, 1, 30}), TestSizeVec({9216, 9216, 30}), false /* run & compare results */
                        ),
        std::make_tuple(TestSizeVec({1, 9216, 30}), TestSizeVec({9216, 9216, 30}), false /* run & compare results */
                        ),
        std::make_tuple(TestSizeVec({1, 1, 1}), TestSizeVec({9216, 9216, 30}), false /* run & compare results */
                        ),
        std::make_tuple(TestSizeVec({1, 9216, 1, 2}), TestSizeVec({9216, 9216, 15, 2}), true /* run & compare results */
                        ),
        std::make_tuple(TestSizeVec({9216, 9216, 15, 1}),
                        TestSizeVec({9216, 9216, 15, 2}),
                        false /* run & compare results */
                        )));

class SynTrainingTransposeHugeTensorTest
: public SynTrainingTestInfra
, public testing::WithParamInterface<std::tuple<TestSizeVec, TransposePermutationArray, bool>>
{
public:
    // these tests may take a lot of device memory, and might become unstable due to memory fragmentation
    // on the device. [SW-151831][SW-153419]
    SynTrainingTransposeHugeTensorTest() { ReleaseDevice(); }
protected:
    void run()
    {
        ScopedConfigurationChange conf("ENABLE_HUGE_TENSOR_SLICING", "true");
        // Since there is no MME node in the graph we need to disable manually the max path scheduler
        ScopedConfigurationChange conf2("ENABLE_MAX_PATH_SCHEDULE", "false");
        auto [inputSizes, permutation, runAndValidate] = GetParam();
        HB_ASSERT(permutation.size() == inputSizes.size(), "size mismatch");
        TestSizeVec outputSize(permutation.size());
        applyPermutation(inputSizes.data(), permutation, outputSize.data());

        unsigned in  = createPersistTensor(INPUT_TENSOR,
                                          runAndValidate ? MEM_INIT_RANDOM_WITH_NEGATIVE : MEM_INIT_COMPILATION_ONLY,
                                          nullptr,
                                          inputSizes.data(),
                                          inputSizes.size());
        unsigned out = createPersistTensor(OUTPUT_TENSOR,
                                           runAndValidate ? MEM_INIT_ALL_ZERO : MEM_INIT_COMPILATION_ONLY,
                                           nullptr,
                                           outputSize.data(),
                                           outputSize.size());

        synTransposeParamsNDims params = permutationToParams(permutation);

        addNodeToGraph("transpose", {in}, {out}, (void*)&params, sizeof(params));

        compileTopology();
        if (runAndValidate)
        {
            HB_ASSERT(permutation.size() == 4, "compare results is implemented only for 4 dim tensor");
            runTopology();

            float* iPtr = castHostBuffer<float>(in);
            float* oPtr = castHostBuffer<float>(out);

            uint64_t iStrides[4] = {1, 0, 0, 0}, oStrides[4] = {1, 0, 0, 0};
            for (int i = 1; i < 4; ++i)
            {
                iStrides[i] = iStrides[i - 1] * inputSizes[i - 1];
                oStrides[i] = oStrides[i - 1] * outputSize[i - 1];
            }

            const auto& p = permutation;
            for (unsigned i = 0; i < outputSize[3]; ++i)
            {
                for (unsigned j = 0; j < outputSize[2]; ++j)
                {
                    for (unsigned k = 0; k < outputSize[1]; ++k)
                    {
                        for (unsigned l = 0; l < outputSize[0]; ++l)
                        {
                            ASSERT_FLOAT_EQ(
                                iPtr[i * iStrides[p[3]] + j * iStrides[p[2]] + k * iStrides[p[1]] + l * iStrides[p[0]]],
                                oPtr[i * oStrides[3] + j * oStrides[2] + k * oStrides[1] + l * oStrides[0]]);
                        }
                    }
                }
            }
        }
    }
};

TEST_P_GC(SynTrainingTransposeHugeTensorTest,
          transpose_huge_tensor_ASIC_CI,
          {synDeviceGaudi3, synDeviceGaudi2, synDeviceGaudi})
{
    if (m_deviceType == synDeviceGaudi)
    {
        TSize size = multiplyElements(std::get<0>(GetParam())) * 4 /*float*/ * 2 /* live operands */;
        if (size > TSize(32) * 1024 * 1024 * 1024)
        {
            GTEST_SKIP();  // skip super large tests since we don't have enough memory on gaudi1
        }
    }
    run();
}

using D = TransposePermutationDim;

INSTANTIATE_TEST_SUITE_P(,
                         SynTrainingTransposeHugeTensorTest,
                         ::testing::Values(std::make_tuple(TestSizeVec({9216, 9216, 30}),
                                                           TransposePermutationArray({(D)0, (D)2, (D)1}),
                                                           false /* run & compare results */
                                                           ),
                                           std::make_tuple(TestSizeVec({9216, 9216 * 30}),
                                                           TransposePermutationArray({(D)1, (D)0}),
                                                           false /* run & compare results */
                                                           ),
                                           std::make_tuple(TestSizeVec({9216, 9216, 15, 2}),
                                                           TransposePermutationArray({(D)2, (D)1, (D)0, (D)3}),
                                                           false /* run & compare results */
                                                           ),
                                           std::make_tuple(TestSizeVec({4500, 2, 7000, 64, 2}),
                                                           TransposePermutationArray({(D)2, (D)3, (D)1, (D)0, (D)4}),
                                                           false /* run & compare results */
                                                           ),
                                           std::make_tuple(TestSizeVec({9216, 9216, 15, 2}),
                                                           TransposePermutationArray({(D)1, (D)2, (D)3, (D)0}),
                                                           true /* run & compare results */
                                                           ),
                                            std::make_tuple(TestSizeVec({25, 1024, 576, 128, 1}),
                                                            TransposePermutationArray({(D)3, (D)1, (D)2, (D)0, (D)4}),
                                                            false /* run & compare results */)));
class SynTrainingHugeTensorTest
: public SynGaudiTestInfra
, public testing::WithParamInterface<std::tuple<TestSizeVec, TestSizeVec, TestSizeVec, bool>>
{
public:
    static unsigned countDmaNodes(const synGraphHandle& handle)
    {
        const HabanaGraph* graph = synSingleton::getInstanceInternal()->getGraph(handle);
        unsigned           ret   = 0;
        for (const NodePtr& n : graph->getNodes())
        {
            if (n && n->isDma())
            {
                ret++;
            }
        }
        return ret;
    }

    void runSingleMmeTest(const char* guid)
    {
        ScopedConfigurationChange conf("ENABLE_HUGE_TENSOR_SLICING", "true");
        // disable slicing so that we really check the rest of the stack can handle huge tensors
        ScopedConfigurationChange sramSliceConf("SRAM_SLICER_MAX_CAPACITY_BYTES", "0");
        ScopedConfigurationChange disableLayeredBrain("ENABLE_LAYERED_PIPELINE_BRAIN", "false");
        auto [opASizes, opBSizes, outSizes, runAndValidate] = GetParam();
        unsigned opA = createPersistTensor(INPUT_TENSOR,
                                           runAndValidate ? MEM_INIT_ALL_ONES : MEM_INIT_COMPILATION_ONLY,
                                           nullptr,
                                           opASizes.data(),
                                           opASizes.size());
        unsigned opB = createPersistTensor(INPUT_TENSOR,
                                           runAndValidate ? MEM_INIT_ALL_ONES : MEM_INIT_COMPILATION_ONLY,
                                           nullptr,
                                           opBSizes.data(),
                                           opBSizes.size());
        unsigned out = createPersistTensor(OUTPUT_TENSOR,
                                           runAndValidate ? MEM_INIT_ALL_ZERO : MEM_INIT_COMPILATION_ONLY,
                                           nullptr,
                                           outSizes.data(),
                                           outSizes.size());

        synGEMMParams params = {false, false};
        addNodeToGraph(guid, {opA, opB}, {out}, (void*)&params, sizeof(params));

        compileTopology();

        if (runAndValidate)
        {
            runTopology();
            float* outPtr   = castHostBuffer<float>(out);
            float  resValue = opASizes[0];  // Since all inputs are 1's the values of the output is the common dim
            TSize  elements = std::accumulate(outSizes.begin(), outSizes.end(), TSize(1), std::multiplies<TSize>());
            for (TSize i = 0; i < elements; ++i)
            {
                ASSERT_FLOAT_EQ(resValue, outPtr[i])
                    << "mismatch at index " << i << "\nexpected: " << resValue << "\nresult: " << outPtr[i];
            }
        }
    }

    void runSingleBiasedMmeTest(const char* guid)
    {
        ScopedConfigurationChange conf("ENABLE_HUGE_TENSOR_SLICING", "true");
        // disable slicing so that we really check the rest of the stack can handle huge tensors
        ScopedConfigurationChange sramSliceConf("SRAM_SLICER_MAX_CAPACITY_BYTES", "0");
        ScopedConfigurationChange disableLayeredBrain("ENABLE_LAYERED_PIPELINE_BRAIN", "false");
        auto [opASizes, opBSizes, outSizes, runAndValidate] = GetParam();

        unsigned     opA = createPersistTensor(INPUT_TENSOR,
                                           runAndValidate ? MEM_INIT_ALL_ONES : MEM_INIT_COMPILATION_ONLY,
                                           nullptr,
                                           opASizes.data(),
                                           opASizes.size());
        unsigned     opB = createPersistTensor(INPUT_TENSOR,
                                           runAndValidate ? MEM_INIT_ALL_ONES : MEM_INIT_COMPILATION_ONLY,
                                           nullptr,
                                           opBSizes.data(),
                                           opBSizes.size());
        unsigned int biasSizes[1];
        biasSizes[0]  = outSizes.data()[0];
        unsigned bias = createPersistTensor(INPUT_TENSOR,
                                            runAndValidate ? MEM_INIT_ALL_ONES : MEM_INIT_COMPILATION_ONLY,
                                            nullptr,
                                            biasSizes,
                                            1);
        unsigned out  = createPersistTensor(OUTPUT_TENSOR,
                                           runAndValidate ? MEM_INIT_ALL_ZERO : MEM_INIT_COMPILATION_ONLY,
                                           nullptr,
                                           outSizes.data(),
                                           outSizes.size());

        synGEMMParams params = {false, false};
        addNodeToGraph(guid, {opA, opB, bias}, {out}, (void*)&params, sizeof(params));

        compileTopology();

        if (runAndValidate)
        {
            runTopology();
            float* outPtr = castHostBuffer<float>(out);
            float  resValue =
                opASizes[0] +
                1;  // Since all inputs are 1's the values of the output is the common dim (+1 because of the bias)
            TSize elements = std::accumulate(outSizes.begin(), outSizes.end(), TSize(1), std::multiplies<TSize>());
            for (TSize i = 0; i < elements; ++i)
            {
                ASSERT_FLOAT_EQ(resValue, outPtr[i])
                    << "mismatch at index " << i << "\nexpected: " << resValue << "\nresult: " << outPtr[i];
            }
        }
    }
};

class SynTrainingBgemmHugeTensorTest : public SynTrainingHugeTensorTest
{
};

class SynTrainingBgemmHugeTensorWithDegeneratedDimsTest : public SynTrainingHugeTensorTest
{
};

TEST_P_GC(SynTrainingBgemmHugeTensorWithDegeneratedDimsTest,
          bgemm_huge_tensor_with_degenerated_dims_ASIC_CI,
          {synDeviceGaudi3, synDeviceGaudi2, synDeviceGaudi})
{
    runSingleMmeTest(NodeFactory::batchGemmNodeTypeName);
    ASSERT_EQ(countDmaNodes(getGraph(0).graphHandle),
              0);  // slicing huge mme node should avoid solving it by dma engine
}

TEST_P_GC(SynTrainingBgemmHugeTensorTest, bgemm_huge_tensor_ASIC_CI, {synDeviceGaudi3, synDeviceGaudi2, synDeviceGaudi})
{
    runSingleMmeTest(NodeFactory::batchGemmNodeTypeName);
}

TEST_P_GC(SynTrainingBgemmHugeTensorTest,
          biased_bgemm_huge_tensor_ASIC_CI,
          {synDeviceGaudi3, synDeviceGaudi2, synDeviceGaudi})
{
    runSingleBiasedMmeTest(NodeFactory::batchGemmNodeTypeName);
}

class SynTrainingGemmHugeTensorTest : public SynTrainingHugeTensorTest
{
};

TEST_P_GC(SynTrainingGemmHugeTensorTest, gemm_huge_tensor_ASIC_CI, {synDeviceGaudi3, synDeviceGaudi2, synDeviceGaudi})
{
    if (m_deviceType == synDeviceGaudi)
    {
        TSize outSize = multiplyElements(std::get<2>(GetParam()));
        if (outSize == TSize(9216) * 9216 * 30)
        {
            GTEST_SKIP();  // TODO [SW-151461] - enable after fixing gemm flow in gaudi
        }
    }
    runSingleMmeTest(NodeFactory::gemmNodeTypeName);
}

TEST_P_GC(SynTrainingGemmHugeTensorTest,
          biased_gemm_huge_tensor_ASIC_CI,
          {synDeviceGaudi3, synDeviceGaudi2, synDeviceGaudi})
{
    if (m_deviceType == synDeviceGaudi)
    {
        TSize outSize = multiplyElements(std::get<2>(GetParam()));
        if (outSize == TSize(9216) * 9216 * 30)
        {
            GTEST_SKIP();  // TODO [SW-151461] - enable after fixing gemm flow in gaudi
        }
    }
    runSingleBiasedMmeTest(NodeFactory::gemmNodeTypeName);
}

INSTANTIATE_TEST_SUITE_P(,
                         SynTrainingGemmHugeTensorTest,
                         ::testing::Values(std::make_tuple(  // op A non common dim is huge
                                               TestSizeVec({9216, 9216 * 30}) /*opA*/,
                                               TestSizeVec({64, 9216}) /*opB*/,
                                               TestSizeVec({64, 9216 * 30}) /*output*/,
                                               false),
                                           std::make_tuple(  // op B non common dim is huge
                                               TestSizeVec({9216, 9216}) /*opA*/,
                                               TestSizeVec({9216 * 30, 9216}) /*opB*/,
                                               TestSizeVec({9216 * 30, 9216}) /*output*/,
                                               true),
                                           std::make_tuple(  // common dim is huge
                                               TestSizeVec({9216 * 30, 64}) /*opA*/,
                                               TestSizeVec({64, 9216 * 30}) /*opB*/,
                                               TestSizeVec({64, 64}) /*output*/,
                                               true)));

INSTANTIATE_TEST_SUITE_P(huge_input_should_slice_batch,
                         SynTrainingBgemmHugeTensorTest,
                         ::testing::Values(std::make_tuple(  // huge op A - should slice batch
                                               TestSizeVec({9216, 9216, 30}) /*opA*/,
                                               TestSizeVec({64, 9216, 30}) /*opB*/,
                                               TestSizeVec({64, 9216, 30}) /*output*/,
                                               false),
                                           std::make_tuple(  // all ops are huge - should slice batch
                                               TestSizeVec({9216, 9216, 30}) /*opA*/,
                                               TestSizeVec({9216, 9216, 30}) /*opB*/,
                                               TestSizeVec({9216, 9216, 30}) /*output*/,
                                               false)));

INSTANTIATE_TEST_SUITE_P(huge_output_should_slice_batch,
                         SynTrainingBgemmHugeTensorTest,
                         ::testing::Values(std::make_tuple(  // huge output - should slice batch
                                               TestSizeVec({64, 9216, 30}) /*opA*/,
                                               TestSizeVec({9216, 64, 30}) /*opB*/,
                                               TestSizeVec({9216, 9216, 30}) /*output*/,
                                               true),
                                           std::make_tuple(  // broadcast operand B
                                               TestSizeVec({64, 9216, 30}) /*opA*/,
                                               TestSizeVec({9216, 64, 1}) /*opB*/,
                                               TestSizeVec({9216, 9216, 30}) /*output*/,
                                               true)));

INSTANTIATE_TEST_SUITE_P(,
                         SynTrainingBgemmHugeTensorWithDegeneratedDimsTest,
                         ::testing::Values(std::make_tuple(  // huge output - should slice batch
                             TestSizeVec({128, 16384, 8, 1, 1}) /*opA*/,
                             TestSizeVec({16384, 128, 1, 1, 1}) /*opB*/,
                             TestSizeVec({16384, 16384, 8, 1, 1}) /*output*/,
                             true)));

INSTANTIATE_TEST_SUITE_P(huge_asymmetric,
                         SynTrainingBgemmHugeTensorTest,
                         ::testing::Values(std::make_tuple(TestSizeVec({14336, 1, 1}) /*opA*/,
                                                           TestSizeVec({250880, 14336}) /*opB*/,
                                                           TestSizeVec({250880, 1, 1}) /*output*/,
                                                           true),
                                           std::make_tuple(TestSizeVec({64, 9216}) /*opA*/,
                                                           TestSizeVec({9216, 64, 30}) /*opB*/,
                                                           TestSizeVec({9216, 9216, 30}) /*output*/,
                                                           true)));

INSTANTIATE_TEST_SUITE_P(
    DISABLED_should_slice_non_common_dim,  // TODO [SW-143103] - remove this block and enable this test for gaudi1.
    SynTrainingBgemmHugeTensorTest,
    ::testing::Values(std::make_tuple(  // broadcast operand B + Huge non common dim operand A -
                                        // should slice non common dim (because of output)
        TestSizeVec({64, 9216 * 30, 2}) /*opA*/,
        TestSizeVec({9216, 64, 1}) /*opB*/,
        TestSizeVec({9216, 9216 * 30, 2}) /*output*/,
        false)));

INSTANTIATE_TEST_SUITE_P(
    DISABLED_should_slice_common_dim,  // TODO [SW-143103] - remove this block and enable this test for gaudi1.
    SynTrainingBgemmHugeTensorTest,
    ::testing::Values(std::make_tuple(  // broadcast operand B + Huge common dim - should slice non
                                        // common dim (because of opB)
        TestSizeVec({9216 * 30, 64, 2}) /*opA*/,
        TestSizeVec({9216, 9216 * 30, 1}) /*opB*/,
        TestSizeVec({9216, 64, 2}) /*output*/,
        true)));

// masked bgemm is not supported in gaudi1.
TEST_F_GC(SynGaudiTestInfra, huge_masked_bgemm, {synDeviceGaudi2})
{
    /*************
     * g_0__masked_batch_gemm_7_0 node
     * inputs:
     *     g_0_tensor_15_id_60_aten__permute[9216 , 9216 * 30, 1, 1] (dtype=float32)
     *     g_0_tensor_13_id_64_aten__permute[128, 9216 , 1, 1] (dtype=float32)
     *     g_0_tensor_16[13, 9216, 1, 1] (dtype=float32)
     *     g_0_tensor_17[128, 13, 1, 1] (dtype=float32)
     * outputs:
     *     g_0_tensor_18_id_76_hpu__masked_batch_gemm[128, 9216 * 30, 1, 1] (dtype=float32)
     * ctrl inputs:
     * ctrl outputs:
     *************/
    // create g_0_tensor_15_id_60_aten__permute tensor
    unsigned g_0_tensor_15_id_60_aten__permute_max_sizes[] = {9216, 9216 * 30, 1, 1};
    unsigned g_0_tensor_15_id_60_aten__permute_min_sizes[] = {9216, 9216 * 30, 1, 1};
    unsigned g_0_tensor_15_id_60_aten__permute             = createTensors(1,
                                                               INPUT_TENSOR,
                                                               true,
                                                               "g_0_tensor_15_id_60_aten__permute",
                                                               MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                               nullptr,
                                                               g_0_tensor_15_id_60_aten__permute_max_sizes,
                                                               4,
                                                               syn_type_single,
                                                               nullptr,
                                                               0,
                                                               0,
                                                               nullptr,
                                                               false,
                                                               g_0_tensor_15_id_60_aten__permute_min_sizes,
                                                               synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_13_id_64_aten__permute tensor
    unsigned g_0_tensor_13_id_64_aten__permute_max_sizes[] = {128, 9216, 1, 1};
    unsigned g_0_tensor_13_id_64_aten__permute_min_sizes[] = {128, 9216, 1, 1};
    unsigned g_0_tensor_13_id_64_aten__permute             = createTensors(1,
                                                               INPUT_TENSOR,
                                                               true,
                                                               "g_0_tensor_13_id_64_aten__permute",
                                                               MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                               nullptr,
                                                               g_0_tensor_13_id_64_aten__permute_max_sizes,
                                                               4,
                                                               syn_type_single,
                                                               nullptr,
                                                               0,
                                                               0,
                                                               nullptr,
                                                               false,
                                                               g_0_tensor_13_id_64_aten__permute_min_sizes,
                                                               synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_16 tensor
    unsigned g_0_tensor_16_max_sizes[] = {13, 9216 * 30, 1, 1};
    unsigned g_0_tensor_16_min_sizes[] = {13, 9216 * 30, 1, 1};
    unsigned g_0_tensor_16             = createTensors(1,
                                           INPUT_TENSOR,
                                           true,
                                           "g_0_tensor_16",
                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                           nullptr,
                                           g_0_tensor_16_max_sizes,
                                           4,
                                           syn_type_single,
                                           nullptr,
                                           0,
                                           0,
                                           nullptr,
                                           false,
                                           g_0_tensor_16_min_sizes,
                                           synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_17 tensor
    unsigned g_0_tensor_17_max_sizes[] = {128, 13, 1, 1};
    unsigned g_0_tensor_17_min_sizes[] = {128, 13, 1, 1};
    unsigned g_0_tensor_17             = createTensors(1,
                                           INPUT_TENSOR,
                                           true,
                                           "g_0_tensor_17",
                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                           nullptr,
                                           g_0_tensor_17_max_sizes,
                                           4,
                                           syn_type_single,
                                           nullptr,
                                           0,
                                           0,
                                           nullptr,
                                           false,
                                           g_0_tensor_17_min_sizes,
                                           synTensorType::DATA_TENSOR)[0];

    // create g_0_tensor_18_id_76_hpu__masked_batch_gemm tensor
    unsigned g_0_tensor_18_id_76_hpu__masked_batch_gemm_max_sizes[] = {128, 9216 * 30, 1, 1};
    unsigned g_0_tensor_18_id_76_hpu__masked_batch_gemm_min_sizes[] = {128, 9216 * 30, 1, 1};
    unsigned g_0_tensor_18_id_76_hpu__masked_batch_gemm =
        createTensors(1,
                      OUTPUT_TENSOR,
                      true,
                      "g_0_tensor_18_id_76_hpu__masked_batch_gemm",
                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                      nullptr,
                      g_0_tensor_18_id_76_hpu__masked_batch_gemm_max_sizes,
                      4,
                      syn_type_single,
                      nullptr,
                      0,
                      0,
                      nullptr,
                      false,
                      g_0_tensor_18_id_76_hpu__masked_batch_gemm_min_sizes,
                      synTensorType::DATA_TENSOR)[0];
    synNodeId     g_0__masked_batch_gemm_7_0_id;
    unsigned char g_0__masked_batch_gemm_7_0_params[] = {0, 0};
    addNodeToGraph("masked_batch_gemm",
                   {g_0_tensor_15_id_60_aten__permute, g_0_tensor_13_id_64_aten__permute, g_0_tensor_16, g_0_tensor_17},
                   {g_0_tensor_18_id_76_hpu__masked_batch_gemm},
                   (void*)g_0__masked_batch_gemm_7_0_params,
                   2,
                   "g_0__masked_batch_gemm_7_0",
                   0 /*graphIndex*/,
                   &g_0__masked_batch_gemm_7_0_id);

    compileTopology();
}

// compile only test. Gaudi1 treats the add output as a huge tensor and thus should not be tested
TEST_F_GC(SynGaudiTestInfra, split_transpose_with_hint, {synDeviceGaudi3, synDeviceGaudi2})
{
    ScopedConfigurationChange conf("ENABLE_HUGE_TENSOR_SLICING", "true");
    static constexpr int      NUM_SPLITS    = 8;
    static constexpr int      DIM           = 2;
    unsigned                  splitSizes[]  = {16777216, 64};
    unsigned                  fullSizes[]   = {16777216, 64 * NUM_SPLITS};
    unsigned                  splitSizesT[] = {splitSizes[1], splitSizes[0]};
    unsigned                  fullSizesT[]  = {fullSizes[1], fullSizes[0]};

    unsigned input      = createPersistTensor(INPUT_TENSOR, MEM_INIT_COMPILATION_ONLY, nullptr, splitSizes, DIM);
    unsigned output     = createPersistTensor(INPUT_TENSOR, MEM_INIT_COMPILATION_ONLY, nullptr, splitSizesT, DIM);
    unsigned catOutput  = createTensor(OUTPUT_TENSOR, MEM_INIT_COMPILATION_ONLY, nullptr, fullSizes, DIM);
    unsigned splitInput = createTensor(OUTPUT_TENSOR, MEM_INIT_COMPILATION_ONLY, nullptr, fullSizesT, DIM);

    std::vector<unsigned> splitOutputs(NUM_SPLITS);
    std::vector<unsigned> catInputs(NUM_SPLITS);
    synTransposeParams    transposeParams;
    transposeParams.tensorDim = 2;
    for (unsigned d = 0; d < ARRAY_SIZE(transposeParams.permutation); d++)
    {
        transposeParams.permutation[d] = static_cast<TransposePermutationDim>(d);
    }
    transposeParams.permutation[0] = static_cast<TransposePermutationDim>(1);
    transposeParams.permutation[1] = static_cast<TransposePermutationDim>(0);
    unsigned catDim                = 1;
    unsigned splitDim              = 0;

    for (unsigned i = 0; i < NUM_SPLITS; i++)
    {
        splitOutputs[i] = createTensor(OUTPUT_TENSOR, MEM_INIT_COMPILATION_ONLY, nullptr, splitSizesT, DIM);
        catInputs[i]    = input;
    }

    std::vector<unsigned> addInputs = splitOutputs;
    for (unsigned numAdds = NUM_SPLITS / 2; numAdds >= 1; numAdds /= 2)
    {
        std::vector<unsigned> addOutputs(numAdds);
        for (unsigned i = 0; i < numAdds; i++)
        {
            addOutputs[i] = numAdds == 1
                                ? output
                                : createTensor(OUTPUT_TENSOR, MEM_INIT_COMPILATION_ONLY, nullptr, splitSizesT, DIM);
            addNodeToGraph("add_fwd_f32", {addInputs[i * 2], addInputs[i * 2 + 1]}, {addOutputs[i]});
        }
        addInputs = addOutputs;
    }

    addNodeToGraph("concat", catInputs, {catOutput}, &catDim, sizeof(catDim), "concat");
    addNodeToGraph("transpose", {catOutput}, {splitInput}, &transposeParams, sizeof(transposeParams), "transpose");
    addNodeToGraph("split", {splitInput}, splitOutputs, &splitDim, sizeof(splitDim), "split");

    compileTopology("");

    // validate expected graph output
    auto               handle     = getGraph(0).graphHandle;
    const HabanaGraph* graph      = synSingleton::getInstanceInternal()->getGraph(handle);
    unsigned           numSplits  = 0;
    unsigned           numConcats = 0;
    for (const NodePtr& n : graph->getNodes())
    {
        if (!n) continue;
        numSplits += static_cast<unsigned>(n->getNodeType() == Node::eNodeType::TYPE_INTERNAL_SPLIT);
        numConcats += static_cast<unsigned>(n->getNodeType() == Node::eNodeType::TYPE_INTERNAL_CONCAT);
    }
    ASSERT_EQ(numConcats, 0);
    ASSERT_EQ(numSplits, 0);
}

TEST_F_GC(SynGaudiHugeTensors, huge_tensors_test_gaudi3_ASIC_CI, {synDeviceGaudi3})
{
    // This optimization has to be disabled, or otherwise the huge tensor copy would not be reproduced due to
    // splitting of the tpc memcpy node
    ScopedConfigurationChange conf("ENABLE_STRIDED_OP_DECODING", "false");

    TSize inputSizes[]  = {TSize(1 << 5), TSize(1 << 20), TSize(1 << 7), TSize(1 << 1)};
    TSize outputSizes[] = {TSize(1 << 5), TSize(1 << 20), TSize(1 << 1), TSize(1 << 7)};

    unsigned inputTensor = createHugeTensors(1,
                                             INPUT_TENSOR,
                                             true,
                                             "input_tensor",
                                             MEM_INIT_COMPILATION_ONLY,
                                             nullptr,
                                             inputSizes,
                                             4,
                                             syn_type_int8)[0];

    unsigned outputTensor = createHugeTensors(1,
                                              OUTPUT_TENSOR,
                                              true,
                                              "output_tensor",
                                              MEM_INIT_COMPILATION_ONLY,
                                              nullptr,
                                              outputSizes,
                                              4,
                                              syn_type_int8)[0];

    synStridedOpParams svParams = {0};

    svParams.strides[0] = 1;
    svParams.strides[1] = svParams.strides[0] * inputSizes[0];
    svParams.strides[2] = svParams.strides[1] * inputSizes[1];
    svParams.strides[3] = svParams.strides[2] * inputSizes[2];

    std::swap(svParams.strides[2], svParams.strides[3]);

    addNodeToGraph("strided_view", {inputTensor}, {outputTensor}, &svParams, sizeof(svParams), "strided_view");

    compileTopology();
}

class DynamicHugeTensorReluTest : public SynTrainingTestInfra
{
  public:

    void compileTest(unsigned  tensorDim,
                     TSize     minSizes[SYN_MAX_TENSOR_DIM],
                     TSize     maxSizes[SYN_MAX_TENSOR_DIM],
                     unsigned& inTensor,
                     unsigned& outTensor)
    {
        inTensor = createHugeTensors(1,
                                     INPUT_TENSOR,
                                     true,
                                     "input_tensor_name",
                                     MEM_INIT_RANDOM_WITH_NEGATIVE,
                                     nullptr,
                                     maxSizes,
                                     tensorDim,
                                     syn_type_single,
                                     nullptr,
                                     0,
                                     0,
                                     nullptr,
                                     false,
                                     minSizes)[0];

        outTensor = createHugeTensors(1,
                                      OUTPUT_TENSOR,
                                      true,
                                      "output_tensor_name",
                                      MEM_INIT_ALL_ONES,
                                      nullptr,
                                      maxSizes,
                                      tensorDim,
                                      syn_type_single,
                                      nullptr,
                                      0,
                                      0,
                                      nullptr,
                                      false,
                                      minSizes)[0];

        addNodeToGraph("relu_fwd_f32", {inTensor}, {outTensor});

        compileTopology();

        ASSERT_NE(m_graphs[0].recipeHandle->basicRecipeHandle.recipe, nullptr);
        shape_plane_graph_t *recipe = m_graphs[0].recipeHandle->basicRecipeHandle.shape_plan_recipe;
        ASSERT_NE(recipe, nullptr);
    }

    void runTest(unsigned  dynamicDim,
                 TSize     actualSize,
                 TSize     maxSizes[SYN_MAX_TENSOR_DIM],
                 unsigned& inTensor,
                 unsigned& outTensor)
    {
        TSize actualSizes[SYN_MAX_TENSOR_DIM];
        memcpy(actualSizes, maxSizes, sizeof(actualSizes));

        actualSizes[dynamicDim] = actualSize;
        setActualSizes(inTensor, actualSizes);
        setActualSizes(outTensor, actualSizes);

        runTopology(0, true);
    }
};

TEST_F_GC(DynamicHugeTensorReluTest, DISABLED_dynamic_dim0_with_huge_tensor_ASIC, {synDeviceGaudi2})
{
    unsigned inTensor, outTensor;

    const unsigned tensorDim  = 2;
    const unsigned dynamicDim = 0;
    const TSize    minW       = 1ULL * 1024ULL * 1024ULL * 1024ULL;
    const TSize    maxW       = 5ULL * 1024ULL * 1024ULL * 1024ULL;
    const TSize    actualW    = 4ULL * 1024ULL * 1024ULL * 1024ULL + 128ULL;
    const TSize    H          = 1;

    TSize maxSizes[SYN_MAX_TENSOR_DIM] = {maxW, H};
    TSize minSizes[SYN_MAX_TENSOR_DIM] = {minW, H};

    compileTest(tensorDim, minSizes, maxSizes, inTensor, outTensor);
    runTest(dynamicDim, actualW, maxSizes, inTensor, outTensor);

    TSize  actualSizes[] = {actualW, H};
    float* inBuffer = castHostInBuffer<float>(inTensor);
    float* outBuffer = castHostOutBuffer<float>(outTensor);

    for (uint64_t i = 0 ; i < actualSizes[0] * actualSizes[1]; i++)
    {
        float expected = inBuffer[i] > 0 ? inBuffer[i] : 0;
        ASSERT_EQ(expected, outBuffer[i]);
    }
}
