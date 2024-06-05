#include "gc_gaudi_test_infra.h"
#include "synapse_common_types.h"
#include "infra/gc_synapse_test.h"
#include "node_factory.h"
#include "syn_gaudi_two_run_compare_test.h"
#include "utils.h"
#include "scoped_configuration_change.h"
#include "syn_singleton.hpp"

// TODO [SW-97543] - Refactor SynTrainingInplaceReuseBindingTest
class SynTrainingInplaceReuseBindingTest : public SynTrainingTestInfra
{};

TEST_F_GC(SynTrainingInplaceReuseBindingTest, scatter_in_place_input_reuse)
{
    ns_ScatterKernel::Params params;
    params.axis = 1;

    float inValues[4] = {1.0, 2.0, 3.0, 4.0};

    float indices[2] = {0.0, 0.0};
    float updates[2] = {3.1, 2.1};

    unsigned dataDims[4] = {1, 2, 2, 1};
    unsigned idxDims[4] = {1, 1, 2, 1};

    unsigned inputData = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, inValues, dataDims, 4 ,syn_type_bf16);
    unsigned inputIndices = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, indices, idxDims, 4,syn_type_int32);
    unsigned inputUpdates = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, updates, idxDims, 4 ,syn_type_bf16);

    unsigned outputTensor   = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, dataDims, 4,  syn_type_bf16);

    addNodeToGraph("scatter_fwd_bf16", {inputData, inputIndices, inputUpdates}, {outputTensor}, &params, sizeof(ns_ScatterKernel::Params));

    compileTopology();
    runTopology();

    bfloat16* pFmOutput = (bfloat16*)m_hostBuffers[outputTensor];

    bfloat16 outRef[4] = {3.1, 2.0, 2.1, 4.0};
    validateResult(outRef, pFmOutput, 4);
}

TEST_F_GC(SynTrainingInplaceReuseBindingTest,
          unsorted_scatter_add_in_place_input_reuse_rmw,
          {synDeviceGaudi2, synDeviceGaudi3})
{
    ns_ScatterKernel::Params params;
    params.axis = 0;

    int32_t indices[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    unsigned dataDims[1] = {20};
    unsigned idxDims[1] = {10};

    unsigned inputData = createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ONES, nullptr, dataDims, 1 ,syn_type_float);
    unsigned inputIndices = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, (float*)indices, idxDims, 1,syn_type_int32);
    unsigned inputUpdates = createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ONES, nullptr, idxDims, 1 ,syn_type_float);

    unsigned outputTensor   = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, dataDims, 1,  syn_type_float);

    addNodeToGraph("unsorted_scatter_add_fwd_f32", {inputData, inputIndices, inputUpdates}, {outputTensor}, &params, sizeof(ns_ScatterKernel::Params));

    compileTopology();
    runTopology();

    float* pFmOutput = castHostOutBuffer<float>(outputTensor);

    float outRef[20] = {
        2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    };
    validateResult(outRef, pFmOutput, 4);
}

TEST_F_GC(SynTrainingInplaceReuseBindingTest, sorted_scatter_add_in_place_input_reuse)
{
    ns_ScatterKernel::Params params;
    params.axis = 0;

    float inValues[4] = {1.0, 1.0, 1.0, 1.0};

    int32_t indices[2] = {0, 1};
    float updates[2] = {3.0, 3.0};

    unsigned dataDims[1] = {4};
    unsigned idxDims[1] = {2};

    unsigned inputData = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, inValues, dataDims, 1 ,syn_type_float);
    unsigned inputIndices = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, (float*)indices, idxDims, 1,syn_type_int32);
    unsigned inputUpdates = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, updates, idxDims, 1 ,syn_type_float);

    unsigned outputTensor   = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, dataDims, 1,  syn_type_float);

    addNodeToGraph("scatter_add_fwd_f32", {inputData, inputIndices, inputUpdates}, {outputTensor}, &params, sizeof(ns_ScatterKernel::Params));

    compileTopology();
    runTopology();

    float* pFmOutput = castHostOutBuffer<float>(outputTensor);
    float outRef[4] = {4.0, 4.0, 1.0, 1.0};
    validateResult(outRef, pFmOutput, 4);
}

TEST_F_GC(SynTrainingInplaceReuseBindingTest, gather_bwd_in_place_input_reuse)
{
    ns_ScatterKernel::Params params;
    params.axis = 1;

    float inValues[4] = {1.0, 2.0, 3.0, 4.0};

    float indices[2] = {0.0, 0.0};
    float updates[2] = {3.1, 2.1};

    unsigned dataDims[4] = {1, 2, 2, 1};
    unsigned idxDims[4] = {1, 1, 2, 1};

    unsigned inputData = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, inValues, dataDims, 4 ,syn_type_bf16);
    unsigned inputIndices = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, indices, idxDims, 4,syn_type_int32);
    unsigned inputUpdates = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, updates, idxDims, 4 ,syn_type_bf16);

    unsigned outputTensor   = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, dataDims, 4,  syn_type_bf16);

    addNodeToGraph("gather_bwd_bf16", {inputData, inputIndices, inputUpdates}, {outputTensor}, &params, sizeof(ns_ScatterKernel::Params));

    compileTopology();
    runTopology();

    bfloat16* pFmOutput = (bfloat16*)m_hostBuffers[outputTensor];

    bfloat16 outRef[4] = {3.1, 2.0, 2.1, 4.0};
    validateResult(outRef, pFmOutput, 4);
}

TEST_F_GC(SynTrainingInplaceReuseBindingTest, gather_bwd_input_output_sharing_memory)
{
    ns_ScatterKernel::Params params;
    params.axis = 1;

    float inValues[4] = {1.0, 2.0, 3.0, 4.0};

    float indices[2] = {0.0, 0.0};
    float updates[2] = {3.1, 2.1};

    unsigned dataDims[4] = {1, 2, 2, 1};
    unsigned idxDims[4] = {1, 1, 2, 1};

    uint64_t memSize    = getMemorySize(dataDims, syn_type_bf16, 4);
    unsigned sectionIdx = createSection(memSize);

    unsigned inputData = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, inValues, dataDims, 4,
                                             syn_type_bf16, nullptr, "inputData", 0, 0, &sectionIdx);
    unsigned inputIndices = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, indices, idxDims, 4,syn_type_int32);
    unsigned inputUpdates = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, updates, idxDims, 4 ,syn_type_bf16);

    unsigned outputTensor   = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, dataDims, 4, syn_type_bf16,
                                                  nullptr, "output", 0, 0, &sectionIdx);

    addNodeToGraph("gather_bwd_bf16", {inputData, inputIndices, inputUpdates}, {outputTensor}, &params, sizeof(ns_ScatterKernel::Params));

    compileTopology();
    runTopology();

    bfloat16* pFmOutput = (bfloat16*)m_hostBuffers[outputTensor];

    bfloat16 outRef[4] = {3.1, 2.0, 2.1, 4.0};
    validateResult(outRef, pFmOutput, 4);
}

TEST_F_GC(SynTrainingTwoRunCompareTest, block_reuse_for_multibuffered_tensors)
{
    // Graph #0

    /*************
     * add_fwd_bf16_n1817 node
     * inputs:
     *     t3863[1024, 9216] (dtype=bf16)
     *     t3865[1024, 1] (dtype=bf16)
     * outputs:
     *     t3864[1024, 9216] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create t3863 tensor
    unsigned t3863_max_sizes[] = {1024, 9216};
    unsigned t3863_min_sizes[] = {1024, 9216};
    unsigned t3863             = createTensors(1,
                                   INPUT_TENSOR,
                                   true,
                                   "t3863",
                                   MEM_INIT_RANDOM_POSITIVE,
                                   nullptr,
                                   t3863_max_sizes,
                                   2,
                                   syn_type_bf16,
                                   nullptr,
                                   0,
                                   0,
                                   nullptr,
                                   false,
                                   t3863_min_sizes,
                                   synTensorType::DATA_TENSOR)[0];

    // create t3865 tensor
    unsigned t3865_max_sizes[] = {1024, 1};
    unsigned t3865_min_sizes[] = {1024, 1};
    unsigned t3865             = createTensors(1,
                                   OUTPUT_TENSOR,
                                   false,
                                   "t3865",
                                   MEM_INIT_RANDOM_POSITIVE,
                                   nullptr,
                                   t3865_max_sizes,
                                   2,
                                   syn_type_bf16,
                                   nullptr,
                                   0,
                                   0,
                                   nullptr,
                                   false,
                                   t3865_min_sizes,
                                   synTensorType::DATA_TENSOR)[0];

    // create t3864 tensor
    unsigned  t3864_max_sizes[] = {1024, 9216};
    unsigned  t3864_min_sizes[] = {1024, 9216};
    unsigned  t3864             = createTensors(1,
                                   OUTPUT_TENSOR,
                                   false,
                                   "t3864",
                                   MEM_INIT_RANDOM_POSITIVE,
                                   nullptr,
                                   t3864_max_sizes,
                                   2,
                                   syn_type_bf16,
                                   nullptr,
                                   0,
                                   0,
                                   nullptr,
                                   false,
                                   t3864_min_sizes,
                                   synTensorType::DATA_TENSOR)[0];
    synNodeId add_fwd_bf16_n1817_id;
    addNodeToGraph("add_fwd_bf16",
                   {t3863, t3865},
                   {t3864},
                   nullptr,
                   0,
                   "add_fwd_bf16_n1817",
                   0 /*graphIndex*/,
                   &add_fwd_bf16_n1817_id);

    /*************
     * dropout_fwd_bf16_n1818 node
     * inputs:
     *     t3864[1024, 9216] (dtype=bf16)
     *     t1847[1] (dtype=int32)
     * outputs:
     *     t3867[1024, 9216] (dtype=bf16)
     *     t3868[1024, 9216] (dtype=int8)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create t1847 tensor
    unsigned t1847_max_sizes[] = {1};
    unsigned t1847_min_sizes[] = {1};
    unsigned t1847             = createTensors(1,
                                   INPUT_TENSOR,
                                   true,
                                   "t1847",
                                   MEM_INIT_RANDOM_POSITIVE,
                                   nullptr,
                                   t1847_max_sizes,
                                   1,
                                   syn_type_int32,
                                   nullptr,
                                   0,
                                   0,
                                   nullptr,
                                   false,
                                   t1847_min_sizes,
                                   synTensorType::DATA_TENSOR)[0];

    // create t3867 tensor
    unsigned t3867_max_sizes[] = {1024, 9216};
    unsigned t3867_min_sizes[] = {1024, 9216};
    unsigned t3867             = createTensors(1,
                                   OUTPUT_TENSOR,
                                   false,
                                   "t3867",
                                   MEM_INIT_RANDOM_POSITIVE,
                                   nullptr,
                                   t3867_max_sizes,
                                   2,
                                   syn_type_bf16,
                                   nullptr,
                                   0,
                                   0,
                                   nullptr,
                                   false,
                                   t3867_min_sizes,
                                   synTensorType::DATA_TENSOR)[0];

    // create t3868 tensor
    unsigned      t3868_max_sizes[] = {1024, 9216};
    unsigned      t3868_min_sizes[] = {1024, 9216};
    unsigned      t3868             = createTensors(1,
                                   OUTPUT_TENSOR,
                                   true,
                                   "t3868",
                                   MEM_INIT_RANDOM_POSITIVE,
                                   nullptr,
                                   t3868_max_sizes,
                                   2,
                                   syn_type_int8,
                                   nullptr,
                                   0,
                                   0,
                                   nullptr,
                                   false,
                                   t3868_min_sizes,
                                   synTensorType::DATA_TENSOR)[0];
    synNodeId     dropout_fwd_bf16_n1818_id;
    unsigned char dropout_fwd_bf16_n1818_params[] = {205, 204, 204, 61, 0, 0, 0, 0};
    addNodeToGraph("dropout_fwd_bf16",
                   {t3864, t1847},
                   {t3867, t3868},
                   (void*)dropout_fwd_bf16_n1818_params,
                   8,
                   "dropout_fwd_bf16_n1818",
                   0 /*graphIndex*/,
                   &dropout_fwd_bf16_n1818_id);

    /*************
     * add_fwd_bf16_n1819 node
     * inputs:
     *     t1823[1024, 9216] (dtype=bf16)
     *     t3867[1024, 9216] (dtype=bf16)
     * outputs:
     *     t3869[1024, 9216] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create t1823 tensor
    unsigned t1823_max_sizes[] = {1024, 9216};
    unsigned t1823_min_sizes[] = {1024, 9216};
    unsigned t1823             = createTensors(1,
                                   OUTPUT_TENSOR,
                                   false,
                                   "t1823",
                                   MEM_INIT_RANDOM_POSITIVE,
                                   nullptr,
                                   t1823_max_sizes,
                                   2,
                                   syn_type_bf16,
                                   nullptr,
                                   0,
                                   0,
                                   nullptr,
                                   false,
                                   t1823_min_sizes,
                                   synTensorType::DATA_TENSOR)[0];

    // create t3869 tensor
    unsigned  t3869_max_sizes[] = {1024, 9216};
    unsigned  t3869_min_sizes[] = {1024, 9216};
    unsigned  t3869             = createTensors(1,
                                   OUTPUT_TENSOR,
                                   false,
                                   "t3869",
                                   MEM_INIT_RANDOM_POSITIVE,
                                   nullptr,
                                   t3869_max_sizes,
                                   2,
                                   syn_type_bf16,
                                   nullptr,
                                   0,
                                   0,
                                   nullptr,
                                   false,
                                   t3869_min_sizes,
                                   synTensorType::DATA_TENSOR)[0];
    synNodeId add_fwd_bf16_n1819_id;
    addNodeToGraph("add_fwd_bf16",
                   {t1823, t3867},
                   {t3869},
                   nullptr,
                   0,
                   "add_fwd_bf16_n1819",
                   0 /*graphIndex*/,
                   &add_fwd_bf16_n1819_id);

    /*************
     * gemm_n1825 node
     * inputs:
     *     t3870[1024, 9216] (dtype=bf16)
     *     t1217[4096, 1024] (dtype=bf16)
     * outputs:
     *     t3881[4096, 9216] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create t1217 tensor
    unsigned t1217_max_sizes[] = {4096, 1024};
    unsigned t1217_min_sizes[] = {4096, 1024};
    unsigned t1217             = createTensors(1,
                                   INPUT_TENSOR,
                                   true,
                                   "t1217",
                                   MEM_INIT_RANDOM_POSITIVE,
                                   nullptr,
                                   t1217_max_sizes,
                                   2,
                                   syn_type_bf16,
                                   nullptr,
                                   0,
                                   0,
                                   nullptr,
                                   false,
                                   t1217_min_sizes,
                                   synTensorType::DATA_TENSOR)[0];

    // create t3881 tensor
    unsigned      t3881_max_sizes[] = {4096, 9216};
    unsigned      t3881_min_sizes[] = {4096, 9216};
    unsigned      t3881             = createTensors(1,
                                   OUTPUT_TENSOR,
                                   true,
                                   "t3881",
                                   MEM_INIT_RANDOM_POSITIVE,
                                   nullptr,
                                   t3881_max_sizes,
                                   2,
                                   syn_type_bf16,
                                   nullptr,
                                   0,
                                   0,
                                   nullptr,
                                   false,
                                   t3881_min_sizes,
                                   synTensorType::DATA_TENSOR)[0];
    synNodeId     gemm_n1825_id;
    unsigned char gemm_n1825_params[] = {0, 0};
    addNodeToGraph("gemm",
                   {t3869, t1217},
                   {t3881},
                   (void*)gemm_n1825_params,
                   2,
                   "gemm_n1825",
                   0 /*graphIndex*/,
                   &gemm_n1825_id);

    /*************
     * reshape_n323 node
     * inputs:
     *     t1821[1024, 384, 24] (dtype=bf16)
     *     t1824[1024, 9216] (dtype=uint32) (shape tensor)
     * outputs:
     *     t1823[1024, 9216] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create t1821 tensor
    unsigned t1821_max_sizes[] = {1024, 384, 24};
    unsigned t1821_min_sizes[] = {1024, 384, 24};
    unsigned t1821             = createTensors(1,
                                   INPUT_TENSOR,
                                   true,
                                   "t1821",
                                   MEM_INIT_RANDOM_POSITIVE,
                                   nullptr,
                                   t1821_max_sizes,
                                   3,
                                   syn_type_bf16,
                                   nullptr,
                                   0,
                                   0,
                                   nullptr,
                                   false,
                                   t1821_min_sizes,
                                   synTensorType::DATA_TENSOR)[0];

    // create t1824 tensor
    unsigned  t1824_max_sizes[] = {1024, 9216};
    unsigned  t1824_min_sizes[] = {1024, 9216};
    unsigned  t1824             = createTensors(1,
                                   INPUT_TENSOR,
                                   false,
                                   "t1824",
                                   MEM_INIT_RANDOM_POSITIVE,
                                   nullptr,
                                   t1824_max_sizes,
                                   2,
                                   syn_type_uint32,
                                   nullptr,
                                   0,
                                   0,
                                   nullptr,
                                   false,
                                   t1824_min_sizes,
                                   synTensorType::SHAPE_TENSOR)[0];
    synNodeId reshape_n323_id;
    addNodeToGraph("reshape", {t1821, t1824}, {t1823}, nullptr, 0, "reshape_n323", 0 /*graphIndex*/, &reshape_n323_id);

    /*************
     * reshape_n1816 node
     * inputs:
     *     t1215[1024] (dtype=bf16)
     *     t3866[1024, 1] (dtype=uint32) (shape tensor)
     * outputs:
     *     t3865[1024, 1] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create t1215 tensor
    unsigned t1215_max_sizes[] = {1024};
    unsigned t1215_min_sizes[] = {1024};
    unsigned t1215             = createTensors(1,
                                   INPUT_TENSOR,
                                   true,
                                   "t1215",
                                   MEM_INIT_RANDOM_POSITIVE,
                                   nullptr,
                                   t1215_max_sizes,
                                   1,
                                   syn_type_bf16,
                                   nullptr,
                                   0,
                                   0,
                                   nullptr,
                                   false,
                                   t1215_min_sizes,
                                   synTensorType::DATA_TENSOR)[0];

    // create t3866 tensor
    unsigned  t3866_max_sizes[] = {1024, 1};
    unsigned  t3866_min_sizes[] = {1024, 1};
    unsigned  t3866             = createTensors(1,
                                   INPUT_TENSOR,
                                   false,
                                   "t3866",
                                   MEM_INIT_RANDOM_POSITIVE,
                                   nullptr,
                                   t3866_max_sizes,
                                   2,
                                   syn_type_uint32,
                                   nullptr,
                                   0,
                                   0,
                                   nullptr,
                                   false,
                                   t3866_min_sizes,
                                   synTensorType::SHAPE_TENSOR)[0];
    synNodeId reshape_n1816_id;
    addNodeToGraph("reshape",
                   {t1215, t3866},
                   {t3865},
                   nullptr,
                   0,
                   "reshape_n1816",
                   0 /*graphIndex*/,
                   &reshape_n1816_id);

    /*************
     * reshape_n6838 node
     * inputs:
     *     t3869[1024, 9216] (dtype=bf16)
     *     t11497[1024, 1, 1, 9216] (dtype=uint32) (shape tensor)
     * outputs:
     *     t11496[1024, 1, 1, 9216] (dtype=bf16)
     * ctrl inputs:
     * ctrl outputs:
     *************/

    // create t11497 tensor
    unsigned t11497_max_sizes[] = {1024, 1, 1, 9216};
    unsigned t11497_min_sizes[] = {1024, 1, 1, 9216};
    unsigned t11497             = createTensors(1,
                                    INPUT_TENSOR,
                                    false,
                                    "t11497",
                                    MEM_INIT_ALL_ZERO,
                                    nullptr,
                                    t11497_max_sizes,
                                    4,
                                    syn_type_uint32,
                                    nullptr,
                                    0,
                                    0,
                                    nullptr,
                                    false,
                                    t11497_min_sizes,
                                    synTensorType::SHAPE_TENSOR)[0];

    // create t11496 tensor
    unsigned  t11496_max_sizes[] = {1024, 1, 1, 9216};
    unsigned  t11496_min_sizes[] = {1024, 1, 1, 9216};
    unsigned  t11496             = createTensors(1,
                                    OUTPUT_TENSOR,
                                    true,
                                    "t11496",
                                    MEM_INIT_ALL_ZERO,
                                    nullptr,
                                    t11496_max_sizes,
                                    4,
                                    syn_type_bf16,
                                    nullptr,
                                    0,
                                    0,
                                    nullptr,
                                    false,
                                    t11496_min_sizes,
                                    synTensorType::DATA_TENSOR)[0];
    synNodeId reshape_n6838_id;
    addNodeToGraph("reshape",
                   {t3869, t11497},
                   {t11496},
                   nullptr,
                   0,
                   "reshape_n6838",
                   0 /*graphIndex*/,
                   &reshape_n6838_id);

    addConfigurationToRun(FIRST_RUN, "ENABLE_INPLACE_REUSE_FOR_SUGGESTIONS", "true");
    addConfigurationToRun(SECOND_RUN, "ENABLE_INPLACE_REUSE_FOR_SUGGESTIONS", "false");
    compareRunsResults({t3868, t3881});
}

class SynTrainingInplaceReuseSuggestionTest : public SynTrainingTestInfra
{};

// There's a matching GC test with the same name validates that reluOutput is aliased to addInput. This test is for
// accuracy validation.
TEST_F_GC(SynTrainingInplaceReuseSuggestionTest, basic_inplace_suggestion)
{
    // Configurations for the test:
    // Disable Shape manipulation suggestions - it adds logical nodes and which adds "noise" to the test and could prevent inplace.
    // Disable tpc fuser - it fuses the tpc nodes to one which misses the point of the test (inplace won't be allowed)
    GlobalConfTestSetter conf1("ENABLE_TPC_TENSOR_SHAPE_MANIPULATION", "false");
    GlobalConfTestSetter conf2("RUN_TPC_FUSER", "false");
    GlobalConfTestSetter conf3("ENABLE_INPLACE_REUSE_FOR_SUGGESTIONS", "true");

    unsigned  dims   = 4;
    TestSizes sizes  = {64, 64, 64, 64, 1};
    unsigned  addIn1 = createPersistTensor(INPUT_TENSOR,
                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                          nullptr,
                                          sizes.data(),
                                          dims,
                                          syn_type_float,
                                          nullptr,
                                          "addIn1");
    unsigned  addIn2 = createPersistTensor(INPUT_TENSOR,
                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                          nullptr,
                                          sizes.data(),
                                          dims,
                                          syn_type_float,
                                          nullptr,
                                          "addIn2");
    unsigned  addOut = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, sizes.data(), dims, syn_type_float);
    addNodeToGraph("add_fwd_f32", {addIn1, addIn2}, {addOut});

    unsigned relu2Out = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, sizes.data(), dims, syn_type_float);
    addNodeToGraph("relu_fwd_f32", {addOut}, {relu2Out});

    unsigned relu2OutCopy = createPersistTensor(OUTPUT_TENSOR,
                                                MEM_INIT_ALL_ZERO,
                                                nullptr,
                                                sizes.data(),
                                                dims,
                                                syn_type_float,
                                                nullptr,
                                                "relu2OutCopy");
    addNodeToGraph(NodeFactory::memcpyNodeTypeName, {relu2Out}, {relu2OutCopy});

    compileAndRun();

    unsigned numElements = multiplyElements(sizes.begin(), sizes.end());
    float*   addIn1Data  = castHostInBuffer<float>(addIn1);
    float*   addIn2Data  = castHostInBuffer<float>(addIn2);
    float*   out         = castHostOutBuffer<float>(relu2OutCopy);

    for (unsigned i = 0; i < numElements; i++)
    {
        float expected = std::max(0.0f, addIn1Data[i] + addIn2Data[i]);
        ASSERT_FLOAT_EQ(expected, out[i])
            << "Mismatch at index " << i << " Expected: " << expected << " Result: " << out[i];
    }
}

// There's a matching GC test with the same name validates the aliasing correctness. This test is for
// accuracy validation.
TEST_F_GC(SynTrainingInplaceReuseSuggestionTest, allow_reuse_only_for_last_consumer)
{
    GlobalConfTestSetter conf1("ENABLE_TPC_TENSOR_SHAPE_MANIPULATION", "false");
    GlobalConfTestSetter conf2("RUN_TPC_FUSER", "false");
    GlobalConfTestSetter conf3("ENABLE_INPLACE_REUSE_FOR_SUGGESTIONS", "true");

    unsigned  dims  = 2;
    TestSizes sizes = {64, 64, 1, 1, 1};

    unsigned reluIn  = createPersistTensor(INPUT_TENSOR,
                                          MEM_INIT_RANDOM_WITH_NEGATIVE,
                                          nullptr,
                                          sizes.data(),
                                          dims,
                                          syn_type_float,
                                          nullptr,
                                          "reluIn");
    unsigned reluOut = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, sizes.data(), dims, syn_type_float);
    addNodeToGraph("relu_fwd_f32", {reluIn}, {reluOut});

    unsigned  add1In1 = createPersistTensor(INPUT_TENSOR,
                                           MEM_INIT_RANDOM_WITH_NEGATIVE,
                                           nullptr,
                                           sizes.data(),
                                           dims,
                                           syn_type_float,
                                           nullptr,
                                           "add1In1");
    unsigned  add1Out = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, sizes.data(), dims, syn_type_float);
    synNodeId add1Id;
    addNodeToGraph("add_fwd_f32", {add1In1, reluOut}, {add1Out}, nullptr, 0, "add1", 0, &add1Id);

    unsigned  add2Out = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, sizes.data(), dims, syn_type_float);
    synNodeId add2Id;
    addNodeToGraph("add_fwd_f32", {add1Out, reluOut}, {add2Out}, nullptr, 0, "add2", 0, &add2Id);

    unsigned add1OutCopy = createPersistTensor(OUTPUT_TENSOR,
                                               MEM_INIT_ALL_ZERO,
                                               nullptr,
                                               sizes.data(),
                                               dims,
                                               syn_type_float,
                                               nullptr,
                                               "add1OutCopy");
    addNodeToGraph(NodeFactory::memcpyNodeTypeName, {add1Out}, {add1OutCopy});
    unsigned  add2OutCopy = createPersistTensor(OUTPUT_TENSOR,
                                               MEM_INIT_ALL_ZERO,
                                               nullptr,
                                               sizes.data(),
                                               dims,
                                               syn_type_float,
                                               nullptr,
                                               "add2OutCopy");
    synNodeId add1CopyId;
    addNodeToGraph(NodeFactory::memcpyNodeTypeName, {add2Out}, {add2OutCopy}, nullptr, 0, "add1Copy", 0, &add1CopyId);

    // Force the add nodes to run together, to prevent a scenario such as: add1 -> memcopy1 -> add2 -> memcopy2, which
    // would make the test pass and can hide a bug
    setNodeDependency(&add1Id, &add2Id, 1, 1);
    setNodeDependency(&add2Id, &add1CopyId, 1, 1);

    compileAndRun();

    float*   reluInData      = castHostInBuffer<float>(reluIn);
    float*   add1In1Data     = castHostInBuffer<float>(add1In1);
    float*   add1OutCopyData = castHostOutBuffer<float>(add1OutCopy);
    float*   add2OutCopyData = castHostOutBuffer<float>(add2OutCopy);
    unsigned numElements     = multiplyElements(sizes.begin(), sizes.end());

    for (unsigned i = 0; i < numElements; i++)
    {
        float reluOutData = std::max(0.0f, reluInData[i]);
        float expectedAddOut1 = reluOutData + add1In1Data[i];
        float expectedAddOut2 = expectedAddOut1 + reluOutData;
        ASSERT_EQ(expectedAddOut2, add2OutCopyData[i])
            << "Mismatch at index " << i << " Expected: " << expectedAddOut2 << " Result: " << add2OutCopyData[i];
        ASSERT_EQ(expectedAddOut1, add1OutCopyData[i])
            << "Mismatch at index " << i << " Expected: " << expectedAddOut1 << " Result: " << add1OutCopyData[i];
    }
}

TEST_F_GC(SynTrainingInplaceReuseBindingTest, scatter_with_transposes_basic)
{
    ScopedConfigurationChange useReuseAsLogical("ENABLE_INPUT_REUSE_AS_LOGICAL_NODE", "true");
    // we want to test the logical op flow without the interference of slicer or fuser
    ScopedConfigurationChange disableSramSlicer("SRAM_SLICER_MAX_CAPACITY_BYTES", "0");
    ScopedConfigurationChange disableLayeredBrain("ENABLE_LAYERED_PIPELINE_BRAIN", "false");
    ScopedConfigurationChange disableTpcFuser("RUN_TPC_FUSER", "false");

    unsigned constexpr dim0        = 128;
    unsigned constexpr dim1        = 448;
    unsigned constexpr dim2        = 512;
    unsigned constexpr elementSize = 2;

    unsigned dataSizes[]           = {dim0, dim1, dim2};
    unsigned dataSizesTransposed[] = {dim0, dim2, dim1};
    unsigned indexSizes[]          = {1, 1};
    unsigned updateSizes[]         = {dim0, dim1, 1};

    auto section = createSection(dim0 * dim1 * dim2 * elementSize);

    // clang-format off
    unsigned in     = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, dataSizesTransposed, 3, syn_type_bf16, nullptr, nullptr, 0, 0, &section);
    unsigned out    = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, dataSizesTransposed, 3, syn_type_bf16, nullptr, nullptr, 0, 0, &section);
    unsigned data   = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, dataSizes, 3, syn_type_bf16);
    unsigned index  = createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, indexSizes, 2, syn_type_int32);
    unsigned update = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, updateSizes, 3, syn_type_bf16);
    unsigned sOut   = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, dataSizes, 3, syn_type_bf16);
    // clang-format on

    synTransposeParamsNDims tParams;
    tParams.tensorDim      = 3;
    tParams.permutation[0] = 0;
    tParams.permutation[1] = 2;
    tParams.permutation[2] = 1;

    ns_ScatterKernel::Params sParams;
    sParams.axis = 2;

    addNodeToGraph("transpose", {in}, {data}, (void*)&tParams, sizeof(tParams));
    addNodeToGraph("scatter_nd_update_fwd_bf16", {data, index, update}, {sOut}, (void*)&sParams, sizeof(sParams));
    addNodeToGraph("transpose", {sOut}, {out}, (void*)&tParams, sizeof(tParams));

    compileAndRun();

    HabanaGraph* graph = synSingleton::getInstanceInternal()->getGraph(m_graphs.front().graphHandle);
    HB_ASSERT_PTR(graph);
    unsigned counter = 0;
    for (const auto& n : graph->getNodes())
    {
        counter += isMemcpy(*n);
    }
    ASSERT_EQ(counter, 0) << "Expect to find zero memcpies, but " << counter << " found.";
}

TEST_F_GC(SynTrainingInplaceReuseBindingTest, scatter_with_transposes_no_memcpy)
{
    ScopedConfigurationChange useReuseAsLogical("ENABLE_INPUT_REUSE_AS_LOGICAL_NODE", "true");
    // we want to test the logical op flow without the interference of slicer or fuser
    ScopedConfigurationChange disableSramSlicer("SRAM_SLICER_MAX_CAPACITY_BYTES", "0");
    ScopedConfigurationChange disableLayeredBrain("ENABLE_LAYERED_PIPELINE_BRAIN", "false");
    ScopedConfigurationChange disableTpcFuser("RUN_TPC_FUSER", "false");

    unsigned constexpr dim0 = 128;
    unsigned constexpr dim1 = 448;
    unsigned constexpr dim2 = 512;

    unsigned dataSizes[]           = {dim0, dim1, dim2};
    unsigned dataSizesTransposed[] = {dim0, dim2, dim1};
    unsigned indexSizes[]          = {1, 1};
    unsigned updateSizes[]         = {dim0, dim1, 1};

    // clang-format off
    unsigned in     = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, dataSizesTransposed, 3, syn_type_bf16);
    unsigned out1   = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, dataSizesTransposed, 3, syn_type_bf16);
    unsigned out2   = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, dataSizesTransposed, 3, syn_type_bf16);
    unsigned dataT  = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, dataSizesTransposed, 3, syn_type_bf16);
    unsigned data   = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, dataSizes, 3, syn_type_bf16);
    unsigned index  = createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, indexSizes, 2, syn_type_int32);
    unsigned update = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, updateSizes, 3, syn_type_bf16);
    unsigned sOut   = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, dataSizes, 3, syn_type_bf16);
    // clang-format on

    synTransposeParamsNDims tParams;
    tParams.tensorDim      = 3;
    tParams.permutation[0] = 0;
    tParams.permutation[1] = 2;
    tParams.permutation[2] = 1;

    ns_ScatterKernel::Params sParams;
    sParams.axis = 2;

    addNodeToGraph("relu_fwd_bf16", {in}, {dataT});
    // path 1
    addNodeToGraph("transpose", {dataT}, {data}, (void*)&tParams, sizeof(tParams));
    addNodeToGraph("scatter_nd_update_fwd_bf16", {data, index, update}, {sOut}, (void*)&sParams, sizeof(sParams));
    addNodeToGraph("transpose", {sOut}, {out1}, (void*)&tParams, sizeof(tParams));
    // path 2
    addNodeToGraph("neg_fwd_bf16", {dataT}, {out2});

    compileAndRun();

    HabanaGraph* graph = synSingleton::getInstanceInternal()->getGraph(m_graphs.front().graphHandle);
    HB_ASSERT_PTR(graph);
    const auto& nodes = graph->getNodes();

    const auto negIt = std::find_if(nodes.begin(), nodes.end(), [](const NodePtr& n) {
        return n->getGUID().find("neg") != std::string::npos;
    });
    HB_ASSERT(negIt != nodes.end(), "can't find neg node");
    const auto scatterIt = std::find_if(nodes.begin(), nodes.end(), [](const NodePtr& n) {
        return n->getGUID().find("scatter") != std::string::npos;
    });
    HB_ASSERT(scatterIt != nodes.end(), "can't find scatter node");

    ASSERT_EQ(graph->getNumberOfPaths(*negIt, *scatterIt, Node::TENSOR_TYPE_ALL), 1)
        << "Expect to find one path, but " << graph->getNumberOfPaths(*scatterIt, *negIt, Node::TENSOR_TYPE_ALL)
        << " found.";

    unsigned counter = 0;
    for (const auto& n : graph->getNodes())
    {
        counter += isMemcpy(*n);
    }
    ASSERT_EQ(counter, 0) << "Expect to find zero memcpies, but " << counter << " found.";
}

TEST_F_GC(SynTrainingInplaceReuseBindingTest, scatter_with_transposes_one_memcpy)
{
    ScopedConfigurationChange useReuseAsLogical("ENABLE_INPUT_REUSE_AS_LOGICAL_NODE", "true");
    // we want to test the logical op flow without the interference of slicer or fuser
    ScopedConfigurationChange disableSramSlicer("SRAM_SLICER_MAX_CAPACITY_BYTES", "0");
    ScopedConfigurationChange disableLayeredBrain("ENABLE_LAYERED_PIPELINE_BRAIN", "false");
    ScopedConfigurationChange disableTpcFuser("RUN_TPC_FUSER", "false");

    unsigned constexpr dim0 = 128;
    unsigned constexpr dim1 = 448;
    unsigned constexpr dim2 = 512;

    unsigned dataSizes[]           = {dim0, dim1, dim2};
    unsigned dataSizesTransposed[] = {dim0, dim2, dim1};
    unsigned indexSizes[]          = {1, 1};
    unsigned updateSizes[]         = {dim0, dim1, 1};

    // clang-format off
    unsigned in     = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, dataSizesTransposed, 3, syn_type_bf16);
    unsigned out    = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, dataSizesTransposed, 3, syn_type_bf16);
    unsigned dataT  = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, dataSizesTransposed, 3, syn_type_bf16);
    unsigned data   = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, dataSizes, 3, syn_type_bf16);
    unsigned index  = createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, indexSizes, 2, syn_type_int32);
    unsigned update = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, updateSizes, 3, syn_type_bf16);
    unsigned sOut   = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, dataSizes, 3, syn_type_bf16);
    unsigned sOutT  = createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, dataSizesTransposed, 3, syn_type_bf16);
    // clang-format on

    synTransposeParamsNDims tParams;
    tParams.tensorDim      = 3;
    tParams.permutation[0] = 0;
    tParams.permutation[1] = 2;
    tParams.permutation[2] = 1;

    ns_ScatterKernel::Params sParams;
    sParams.axis = 2;

    addNodeToGraph("relu_fwd_bf16", {in}, {dataT});
    // path 1
    addNodeToGraph("transpose", {dataT}, {data}, (void*)&tParams, sizeof(tParams));
    addNodeToGraph("scatter_nd_update_fwd_bf16", {data, index, update}, {sOut}, (void*)&sParams, sizeof(sParams));
    addNodeToGraph("transpose", {sOut}, {sOutT}, (void*)&tParams, sizeof(tParams));
    // path 2
    addNodeToGraph("add_fwd_bf16", {dataT, sOutT}, {out});

    compileAndRun();

    HabanaGraph* graph = synSingleton::getInstanceInternal()->getGraph(m_graphs.front().graphHandle);
    HB_ASSERT_PTR(graph);
    const auto& nodes = graph->getNodes();

    const auto addIt = std::find_if(nodes.begin(), nodes.end(), [](const NodePtr& n) {
        return n->getGUID().find("add") != std::string::npos;
    });
    HB_ASSERT(addIt != nodes.end(), "can't find add node");
    const auto scatterIt = std::find_if(nodes.begin(), nodes.end(), [](const NodePtr& n) {
        return n->getGUID().find("scatter") != std::string::npos;
    });
    HB_ASSERT(scatterIt != nodes.end(), "can't find scatter node");

    ASSERT_EQ(graph->getNumberOfPaths(*addIt, *scatterIt, Node::TENSOR_TYPE_ALL), 0)
        << "Expect to not find path, but " << graph->getNumberOfPaths(*scatterIt, *addIt, Node::TENSOR_TYPE_ALL)
        << " found.";

    unsigned counter = 0;
    for (const auto& n : graph->getNodes())
    {
        counter += isMemcpy(*n);
    }
    ASSERT_EQ(counter, 1) << "Expect to find one memcpy, but " << counter << " found.";
}