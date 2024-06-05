#include "event_triggered_logger.hpp"
#include "gc_gaudi_test_infra.h"
#include "global_conf_test_setter.h"
#include "graph_compiler/sim_graph.h"
#include "graph_compiler/habana_nodes/node_factory.h"
#include "data_type_utils.h"
#include "supported_devices_macros.h"
#include "synapse_api.h"
#include "synapse_common_types.h"
#include "synapse_common_types.hpp"
#include <tuple>
#include <vector>
#include "transpose_utils.h"
#include "syn_gaudi_two_run_compare_test.h"
#include "scoped_configuration_change.h"

// Introduction:
// Compare Gaudi graph compilation and execution result with SimGraph, where the transpose operation is
//      performed on the cpu.
//
// The test plan coverage is divided into few classes:
// 1. Tpc/Mme dataType = { float/bfloat16/int32}
// 2. Transpose type = {logical only, physical only, composed = logical + physical}
//
// Using the following naming convention: transpose_<fp32/bp16>_<newAxis: HBCW>
//
// Generated new class SynGaudiTransposeTest which implements template datatype functions:
// - executeTransposeOperation - performs the transpose operation using the Gaudi graph
// - validateTransposeOperation - performs the transpose operation on the cpu and validate accuracy with the execution
// result

class SynGaudiTransposeSystemTest : public SynGaudiTestInfra, public ::testing::WithParamInterface<std::tuple<std::vector<unsigned int>, TransposePermutationArray, synDataType>>
{
public:
    SynGaudiTransposeSystemTest();
    void runTransposeTest();
    template<typename DTYPE>
    void runTrasposeTest(unsigned int*             input_dimensions,
                         const unsigned            inputSize,
                         TransposePermutationArray permutation,
                         synDataType               dataType);
    template<typename DTYPE>
    void validateTransposeOperation(unsigned int*      input_dimensions,
                                    DTYPE*             inputArray,
                                    unsigned int*      output_dimensions,
                                    const unsigned int dim_num,
                                    synDataType        dataType,
                                    DTYPE*             pOutputBuffer);

    template<typename DTYPE>
    void validateResults(unsigned* dimensions, unsigned dimNum, DTYPE* output, DTYPE* outputRef);

    std::pair<unsigned, unsigned> executeTransposeOperation(unsigned int*             input_dimensions,
                                                            unsigned int*             output_dimensions,
                                                            synDataType               dataType,
                                                            unsigned                  dim,
                                                            TransposePermutationArray permutation);

    void initInputs(TransposePermutationArray permutation,
                    unsigned int*             input_dimensions,
                    const unsigned            inputSize,
                    unsigned int*             output_dimensions,
                    synDataType               dataType);

    struct PrintToStringParamName
    {
        template<class ParamType>
        std::string operator()(const ::testing::TestParamInfo<ParamType>& info) const
        {
            std::stringstream ss;
            auto inputDimensions = std::get<0>(info.param);
            TransposePermutationArray permutation = std::get<1>(info.param);
            synDataType synType = std::get<2>(info.param);
            for (auto& dim : inputDimensions)
            {
                ss << dim << "_";
            }
            static char translate[] = "CWHB56789";
            for (auto& dim : permutation)
            {
                ss << translate[static_cast<int>(dim)];
            }
            ss << "_" << getStringFromSynDataType(synType);
            if (multiplyElements(inputDimensions) >= 5000)
            {
                ss << "_ASIC";
            }
            return ss.str();
        }
    };

protected:
    synTransposeParamsNDims transposeParams;
    std::optional<bool>     m_inputPermuted;
    std::optional<bool>     m_inputAllowPermuted;
};

class SynGaudiHighRankTransposeSystemTest : public SynGaudiTransposeSystemTest
{
};

TEST_P_GC(SynGaudiTransposeSystemTest, transpose, {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    runTransposeTest();
};

TEST_P_GC(SynGaudiHighRankTransposeSystemTest, high_rank_transpose, {synDeviceGaudi, synDeviceGaudi3})
{
    runTransposeTest();
};

INSTANTIATE_TEST_SUITE_P(
    Utilization,
    SynGaudiTransposeSystemTest,
    ::testing::Values(std::make_tuple(std::vector<unsigned int>{4, 32, 2, 1}, TransposePermutationArray{TPD_Width, TPD_Height, TPD_Channel, TPD_Depth}, syn_type_bf16),        /* single, physical transpose. Requires flattening */
                      std::make_tuple(std::vector<unsigned int>{4, 128, 2, 1}, TransposePermutationArray{TPD_Width, TPD_Height, TPD_Channel, TPD_Depth}, syn_type_bf16),
                      std::make_tuple(std::vector<unsigned int>{4, 64, 2, 1}, TransposePermutationArray{TPD_Width, TPD_Height, TPD_Channel, TPD_Depth}, syn_type_bf16),
                      std::make_tuple(std::vector<unsigned int>{4, 32, 2, 1}, TransposePermutationArray{TPD_Width, TPD_Channel, TPD_Height, TPD_Depth}, syn_type_bf16),        /* physical transpose -> logical. Requires flattening */
                      std::make_tuple(std::vector<unsigned int>{4, 320, 2, 1}, TransposePermutationArray{TPD_Width, TPD_Channel, TPD_Height, TPD_Depth}, syn_type_bf16),        /* physical transpose -> logical. Requires flattening */
                      std::make_tuple(std::vector<unsigned int>{4, 2, 32, 1}, TransposePermutationArray{TPD_Height, TPD_Channel, TPD_Width, TPD_Depth}, syn_type_bf16),        /* logical -> physical transpose -> logical. Requires flattening */
                      std::make_tuple(std::vector<unsigned int>{4, 2, 32 * 10, 1}, TransposePermutationArray{TPD_Height, TPD_Channel, TPD_Width, TPD_Depth}, syn_type_bf16),        /* logical -> physical transpose -> logical. Requires flattening */
                      std::make_tuple(std::vector<unsigned int>{1, 1, 64,256}, TransposePermutationArray{TPD_Depth, TPD_Height, TPD_Channel, TPD_Width}, syn_type_single),         /* float32_WBCH */
                      std::make_tuple(std::vector<unsigned int>{56,56,64,64}, TransposePermutationArray{TPD_Height, TPD_Channel, TPD_Width, TPD_Depth}, syn_type_single),         /* float32_WBCH */
                      std::make_tuple(std::vector<unsigned int>{256,56,56,64}, TransposePermutationArray{TPD_Width, TPD_Height, TPD_Channel, TPD_Depth}, syn_type_single),         /* float32_WBCH */
                      std::make_tuple(std::vector<unsigned int>{56,56,256,64}, TransposePermutationArray{TPD_Height, TPD_Channel, TPD_Width, TPD_Depth}, syn_type_single),         /* float32_WBCH */
                      std::make_tuple(std::vector<unsigned int>{64,56,56,64}, TransposePermutationArray{TPD_Width, TPD_Height, TPD_Channel, TPD_Depth}, syn_type_single),         /* float32_WBCH */
                      std::make_tuple(std::vector<unsigned int>{256,64,1,1}, TransposePermutationArray{TPD_Height, TPD_Depth, TPD_Width, TPD_Channel}, syn_type_single)         /* float32_WBCH */
                      ),
                      SynGaudiTransposeSystemTest::PrintToStringParamName{});


INSTANTIATE_TEST_SUITE_P(
    DoubleTranspose,
    SynGaudiTransposeSystemTest,
    ::testing::Values(std::make_tuple(std::vector<unsigned int>{64, 12, 384, 64}, TransposePermutationArray{TPD_Width, TPD_Channel, TPD_Height, TPD_Depth}, syn_type_bf16)),
                      SynGaudiTransposeSystemTest::PrintToStringParamName{});

INSTANTIATE_TEST_SUITE_P(
    WithPhysical,
    SynGaudiTransposeSystemTest,
    ::testing::Values( // std::make_tuple(std::vector<unsigned int>{4, 32, 2, 1}, TransposePermutationArray{TPD_Width, TPD_Height, TPD_Channel, TPD_Depth}, syn_type_bf16),        /* single, physical transpose. Requires flattening */
                      std::make_tuple(std::vector<unsigned int>{4, 32, 2, 1}, TransposePermutationArray{TPD_Width, TPD_Channel, TPD_Height, TPD_Depth}, syn_type_bf16),        /* physical transpose -> logical. Requires flattening */
                      std::make_tuple(std::vector<unsigned int>{4, 64, 5, 1}, TransposePermutationArray{TPD_Width, TPD_Channel, TPD_Height, TPD_Depth}, syn_type_bf16),        /* float32_WBCH */
                      std::make_tuple(std::vector<unsigned int>{2, 2, 2, 2}, TransposePermutationArray{TPD_Width, TPD_Depth, TPD_Channel, TPD_Height}, syn_type_float),           /* float32_WBCH */
                      std::make_tuple(std::vector<unsigned int>{6, 6, 1, 1}, TransposePermutationArray{TPD_Height, TPD_4Dim_Batch, TPD_Channel, TPD_Width}, syn_type_float),      /* float32_HBCW */
                      std::make_tuple(std::vector<unsigned int>{3, 2, 3, 1}, TransposePermutationArray{TPD_4Dim_Batch, TPD_Channel, TPD_Height, TPD_Width}, syn_type_float),      /* float32_BCHW */
                      std::make_tuple(std::vector<unsigned int>{3, 2, 3, 1}, TransposePermutationArray{TPD_Height, TPD_Width, TPD_Channel, TPD_4Dim_Batch}, syn_type_int32),      /* int32_HWCB */
                      std::make_tuple(std::vector<unsigned int>{3, 2, 3, 1}, TransposePermutationArray{TPD_4Dim_Batch, TPD_Width, TPD_Channel, TPD_Height}, syn_type_int32),        /* int32_BWCH */
                      std::make_tuple(std::vector<unsigned int>{2, 2, 2, 2}, TransposePermutationArray{TPD_Width, TPD_Depth, TPD_Channel, TPD_Height}, syn_type_int8),            /* int8_WBCH */
                      std::make_tuple(std::vector<unsigned int>{2, 2, 2, 2}, TransposePermutationArray{TPD_Width, TPD_Depth, TPD_Channel, TPD_Height}, syn_type_uint8),            /* uint8_WBCH */
                      std::make_tuple(std::vector<unsigned int>{6, 6, 1, 1}, TransposePermutationArray{TPD_Height, TPD_4Dim_Batch, TPD_Channel, TPD_Width}, syn_type_int8),      /* float32_HBCW */
                      std::make_tuple(std::vector<unsigned int>{3, 2, 3, 1}, TransposePermutationArray{TPD_Width, TPD_Height, TPD_4Dim_Batch, TPD_Channel}, syn_type_bf16),       /* bf16_WHBC */
                      std::make_tuple(std::vector<unsigned int>{3, 2, 3, 1}, TransposePermutationArray{TPD_Width, TPD_Channel, TPD_Height, TPD_4Dim_Batch}, syn_type_bf16),       /* bf16_WCHB */
                      std::make_tuple(std::vector<unsigned int>{5000, 5000, 1, 1}, TransposePermutationArray{TPD_Width, TPD_Channel, TPD_Height, TPD_4Dim_Batch}, syn_type_bf16), /* bf16_WCHB_perf */
                      std::make_tuple(std::vector<unsigned int>{6, 6, 1, 1}, TransposePermutationArray{TPD_4Dim_Batch, TPD_Channel, TPD_Width, TPD_Height}, syn_type_bf16),        /* bf16_BCWH */
                      std::make_tuple(std::vector<unsigned int>{3, 2, 3, 1}, TransposePermutationArray{TPD_Height, TPD_4Dim_Batch, TPD_Width, TPD_Channel}, syn_type_bf16)        /* bf16_HBWC */
                      ),
                      SynGaudiTransposeSystemTest::PrintToStringParamName{});

INSTANTIATE_TEST_SUITE_P(
    BugTest,
    SynGaudiTransposeSystemTest,
    ::testing::Values(std::make_tuple(std::vector<unsigned int> {50, 50, 1, 1},
                                      TransposePermutationArray {TPD_Depth, TPD_Height, TPD_Width, TPD_Channel},
                                      syn_type_float),
                      std::make_tuple(std::vector<unsigned int> {4, 13, 13, 1137},
                                      TransposePermutationArray {TPD_Channel, TPD_Depth, TPD_Height, TPD_Width},
                                      syn_type_float)),
    SynGaudiTransposeSystemTest::PrintToStringParamName {});

INSTANTIATE_TEST_SUITE_P(
    6Dims,
    SynGaudiHighRankTransposeSystemTest,
    ::testing::Values(
                      std::make_tuple(std::vector<unsigned int> {4, 4, 4, 4, 4, 4},TransposePermutationArray {TPD_Depth,TPD_Height,TPD_Width,TPD_Channel,TransposePermutationDim(4),TransposePermutationDim(5)},syn_type_float), /* float32_WBCH */
                      std::make_tuple(std::vector<unsigned int> {4, 4, 4, 4, 4, 4},TransposePermutationArray {TPD_Channel,TPD_Width, TPD_Height, TPD_Depth, TransposePermutationDim(5), TransposePermutationDim(4)}, syn_type_float), /* float32_WBCH */
                      std::make_tuple(std::vector<unsigned int> {2, 2, 2, 2, 2, 10}, TransposePermutationArray {TransposePermutationDim(0), TransposePermutationDim(1), TransposePermutationDim(3), TransposePermutationDim(5), TransposePermutationDim(2), TransposePermutationDim(4)}, syn_type_float), /* float32_WBCH */
                      std::make_tuple(std::vector<unsigned int> {2, 2, 2, 2, 2, 12}, TransposePermutationArray {TransposePermutationDim(0), TransposePermutationDim(1), TransposePermutationDim(3), TransposePermutationDim(5), TransposePermutationDim(2), TransposePermutationDim(4)}, syn_type_float), /* float32_WBCH */
                      std::make_tuple(std::vector<unsigned int> {3, 3, 3, 3, 3, 7}, TransposePermutationArray {TransposePermutationDim(0), TransposePermutationDim(1), TransposePermutationDim(3), TransposePermutationDim(5), TransposePermutationDim(2), TransposePermutationDim(4)}, syn_type_float), /* float32_WBCH */
                      std::make_tuple(std::vector<unsigned int> {3, 3, 3, 3, 3, 13}, TransposePermutationArray {TransposePermutationDim(0), TransposePermutationDim(1), TransposePermutationDim(3), TransposePermutationDim(5), TransposePermutationDim(2), TransposePermutationDim(4)}, syn_type_float), /* float32_WBCH */
                      std::make_tuple(std::vector<unsigned int> {3, 3, 3, 3, 3, 15}, TransposePermutationArray {TransposePermutationDim(0), TransposePermutationDim(1), TransposePermutationDim(3), TransposePermutationDim(5), TransposePermutationDim(2), TransposePermutationDim(4)}, syn_type_float), /* float32_WBCH */
                      std::make_tuple(std::vector<unsigned int> {2, 2, 2, 2, 2, 1}, TransposePermutationArray {TransposePermutationDim(0), TransposePermutationDim(1), TransposePermutationDim(2), TransposePermutationDim(3), TransposePermutationDim(4), TransposePermutationDim(5)}, syn_type_float), /* float32_WBCH */
                      std::make_tuple(std::vector<unsigned int> {2, 1, 1, 1, 1, 2}, TransposePermutationArray {TransposePermutationDim(0), TransposePermutationDim(1), TransposePermutationDim(2), TransposePermutationDim(3), TransposePermutationDim(4), TransposePermutationDim(5)}, syn_type_float), /* float32_WBCH */
                      std::make_tuple(std::vector<unsigned int> {2, 2, 2, 2, 2, 2}, TransposePermutationArray {TransposePermutationDim(0), TransposePermutationDim(1), TransposePermutationDim(2), TransposePermutationDim(3), TransposePermutationDim(4), TransposePermutationDim(5)}, syn_type_float) /* float32_WBCH */
                      ),
                      SynGaudiTransposeSystemTest::PrintToStringParamName {});

INSTANTIATE_TEST_SUITE_P(
    7Dims,
    SynGaudiHighRankTransposeSystemTest,
    ::testing::Values(
                      std::make_tuple(std::vector<unsigned int> {4, 5, 12, 3, 7, 6, 2},TransposePermutationArray {TPD_Depth,TPD_Height,TPD_Width,TransposePermutationDim(6),TPD_Channel,TransposePermutationDim(4),TransposePermutationDim(5)},syn_type_float), /* float32_WBCH */
                      std::make_tuple(std::vector<unsigned int> {40, 4, 12, 2, 18, 4, 17},TransposePermutationArray {TPD_Channel,TPD_Width, TPD_Height, TPD_Depth, TransposePermutationDim(5), TransposePermutationDim(4), TransposePermutationDim(6)}, syn_type_float), /* float32_WBCH */
                      std::make_tuple(std::vector<unsigned int> {12, 12, 7, 18, 15, 10, 2}, TransposePermutationArray {TransposePermutationDim(0), TransposePermutationDim(1), TransposePermutationDim(3), TransposePermutationDim(5), TransposePermutationDim(6), TransposePermutationDim(2), TransposePermutationDim(4)}, syn_type_float), /* float32_WBCH */
                      std::make_tuple(std::vector<unsigned int> {2, 2, 2, 2, 2, 12, 12}, TransposePermutationArray {TransposePermutationDim(3), TransposePermutationDim(1), TransposePermutationDim(6), TransposePermutationDim(0), TransposePermutationDim(5), TransposePermutationDim(2), TransposePermutationDim(4)}, syn_type_float), /* float32_WBCH */
                      std::make_tuple(std::vector<unsigned int> {3, 3, 3, 3, 3, 17, 7}, TransposePermutationArray {TransposePermutationDim(1), TransposePermutationDim(6), TransposePermutationDim(0), TransposePermutationDim(3), TransposePermutationDim(5), TransposePermutationDim(2), TransposePermutationDim(4)}, syn_type_float), /* float32_WBCH */
                      std::make_tuple(std::vector<unsigned int> {3, 13, 3, 3, 3, 8, 13}, TransposePermutationArray {TransposePermutationDim(6), TransposePermutationDim(0), TransposePermutationDim(1), TransposePermutationDim(3), TransposePermutationDim(5), TransposePermutationDim(2), TransposePermutationDim(4)}, syn_type_float), /* float32_WBCH */
                      std::make_tuple(std::vector<unsigned int> {3, 3, 3, 3, 3, 5, 15}, TransposePermutationArray {TransposePermutationDim(3), TransposePermutationDim(1), TransposePermutationDim(0), TransposePermutationDim(5), TransposePermutationDim(6), TransposePermutationDim(2), TransposePermutationDim(4)}, syn_type_float), /* float32_WBCH */
                      std::make_tuple(std::vector<unsigned int> {2, 2, 2, 2, 2, 2, 1}, TransposePermutationArray {TransposePermutationDim(0), TransposePermutationDim(1), TransposePermutationDim(2), TransposePermutationDim(3), TransposePermutationDim(4), TransposePermutationDim(5), TransposePermutationDim(6)}, syn_type_float), /* float32_WBCH */
                      std::make_tuple(std::vector<unsigned int> {2, 1, 1, 1, 1, 1, 2}, TransposePermutationArray {TransposePermutationDim(4), TransposePermutationDim(1), TransposePermutationDim(2), TransposePermutationDim(3), TransposePermutationDim(6), TransposePermutationDim(0), TransposePermutationDim(5)}, syn_type_float), /* float32_WBCH */
                      std::make_tuple(std::vector<unsigned int> {2, 2, 2, 2, 2, 2, 2}, TransposePermutationArray {TransposePermutationDim(0), TransposePermutationDim(1), TransposePermutationDim(2), TransposePermutationDim(6), TransposePermutationDim(3), TransposePermutationDim(4), TransposePermutationDim(5)}, syn_type_float) /* float32_WBCH */
                      ),
                      SynGaudiTransposeSystemTest::PrintToStringParamName {});

INSTANTIATE_TEST_SUITE_P(
    8Dims,
    SynGaudiHighRankTransposeSystemTest,
    ::testing::Values(
                      std::make_tuple(std::vector<unsigned int> {4, 5, 12, 3, 7, 6, 2, 20},TransposePermutationArray {TPD_Depth,TPD_Height,TPD_Width,TransposePermutationDim(7),TransposePermutationDim(6),TPD_Channel,TransposePermutationDim(4),TransposePermutationDim(5)},syn_type_float), /* float32_WBCH */
                      std::make_tuple(std::vector<unsigned int> {20, 4, 10, 2, 18, 4, 17, 2},TransposePermutationArray {TPD_Channel,TPD_Width, TPD_Height, TPD_Depth, TransposePermutationDim(5), TransposePermutationDim(7), TransposePermutationDim(4), TransposePermutationDim(6)}, syn_type_float), /* float32_WBCH */
                      std::make_tuple(std::vector<unsigned int> {12, 12, 7, 18, 5, 10, 2, 31}, TransposePermutationArray {TransposePermutationDim(7),TransposePermutationDim(0), TransposePermutationDim(1), TransposePermutationDim(3), TransposePermutationDim(5), TransposePermutationDim(6), TransposePermutationDim(2), TransposePermutationDim(4)}, syn_type_float), /* float32_WBCH */
                      std::make_tuple(std::vector<unsigned int> {18, 2, 2, 20, 2, 2, 7, 12, 40}, TransposePermutationArray {TransposePermutationDim(3), TransposePermutationDim(1), TransposePermutationDim(7), TransposePermutationDim(6), TransposePermutationDim(0), TransposePermutationDim(5), TransposePermutationDim(2), TransposePermutationDim(4)}, syn_type_float), /* float32_WBCH */
                      std::make_tuple(std::vector<unsigned int> {3, 3, 13, 5, 3, 3, 4, 7, 5}, TransposePermutationArray {TransposePermutationDim(1), TransposePermutationDim(7), TransposePermutationDim(6), TransposePermutationDim(0), TransposePermutationDim(3), TransposePermutationDim(5), TransposePermutationDim(2), TransposePermutationDim(4)}, syn_type_float), /* float32_WBCH */
                      std::make_tuple(std::vector<unsigned int> {3, 13, 3, 3, 10, 3, 8, 13, 20}, TransposePermutationArray {TransposePermutationDim(6), TransposePermutationDim(0), TransposePermutationDim(1), TransposePermutationDim(3), TransposePermutationDim(5), TransposePermutationDim(2),TransposePermutationDim(7), TransposePermutationDim(4)}, syn_type_float), /* float32_WBCH */
                      std::make_tuple(std::vector<unsigned int> {50, 3, 3, 3, 3, 3, 5, 5, 15}, TransposePermutationArray {TransposePermutationDim(3), TransposePermutationDim(1), TransposePermutationDim(0), TransposePermutationDim(5), TransposePermutationDim(6), TransposePermutationDim(7), TransposePermutationDim(2), TransposePermutationDim(4)}, syn_type_float), /* float32_WBCH */
                      std::make_tuple(std::vector<unsigned int> {2, 2, 2, 2, 2, 2, 2, 2, 1}, TransposePermutationArray {TransposePermutationDim(0), TransposePermutationDim(1), TransposePermutationDim(2), TransposePermutationDim(3), TransposePermutationDim(4), TransposePermutationDim(5), TransposePermutationDim(6), TransposePermutationDim(7)}, syn_type_float), /* float32_WBCH */
                      std::make_tuple(std::vector<unsigned int> {2, 1, 1, 1, 1, 1, 1, 1, 2}, TransposePermutationArray {TransposePermutationDim(4), TransposePermutationDim(7), TransposePermutationDim(1), TransposePermutationDim(2), TransposePermutationDim(3), TransposePermutationDim(6), TransposePermutationDim(0), TransposePermutationDim(5)}, syn_type_float), /* float32_WBCH */
                      std::make_tuple(std::vector<unsigned int> {2, 2, 2, 2, 2, 2, 10, 17, 15}, TransposePermutationArray {TransposePermutationDim(0), TransposePermutationDim(1), TransposePermutationDim(2), TransposePermutationDim(3), TransposePermutationDim(6), TransposePermutationDim(7), TransposePermutationDim(4), TransposePermutationDim(5)}, syn_type_float) /* float32_WBCH */
                      ),
                      SynGaudiTransposeSystemTest::PrintToStringParamName {});

INSTANTIATE_TEST_SUITE_P(
    OnlyLogical, // OnlyLogical/SynGaudiTransposeSystemTest*
    SynGaudiTransposeSystemTest,
    ::testing::Combine(
        ::testing::Values(std::vector<unsigned int>{3, 2, 3, 5}, std::vector<unsigned int>{32, 4, 2, 1}),
        ::testing::Values(TransposePermutationArray{TPD_Channel, TPD_Width, TPD_4Dim_Batch, TPD_Height}, /* CWBH */
                          TransposePermutationArray{TPD_Channel, TPD_Height, TPD_Width, TPD_4Dim_Batch}, /* CHWB */
                          TransposePermutationArray{TPD_Channel, TPD_Height, TPD_4Dim_Batch, TPD_Width}, /* CHBW */
                          TransposePermutationArray{TPD_Channel, TPD_4Dim_Batch, TPD_Width, TPD_Height}, /* CBWH */
                          TransposePermutationArray{TPD_Channel, TPD_4Dim_Batch, TPD_Height, TPD_Width}  /* CBHW */
                          )
        ,
        ::testing::Values(syn_type_float, syn_type_bf16, syn_type_uint8)),
    SynGaudiTransposeSystemTest::PrintToStringParamName{});

INSTANTIATE_TEST_SUITE_P(
    FiveDims,
    SynGaudiTransposeSystemTest,
    ::testing::Values(
        std::make_tuple(std::vector<unsigned int> {4, 4, 4, 4, 4}, TransposePermutationArray {TPD_Depth, TPD_Height, TPD_Width, TPD_Channel, TransposePermutationDim(4)}, syn_type_float), /* float32_WBCH */
        std::make_tuple(std::vector<unsigned int> {4, 4, 4, 4, 4}, TransposePermutationArray {TPD_Channel, TPD_Width, TPD_Height, TPD_Depth, TransposePermutationDim(4)}, syn_type_float), /* float32_WBCH */
        std::make_tuple(std::vector<unsigned int> {2, 2, 2, 2, 10}, TransposePermutationArray {TransposePermutationDim(0), TransposePermutationDim(1), TransposePermutationDim(3), TransposePermutationDim(2), TransposePermutationDim(4)}, syn_type_float), /* float32_WBCH */
        std::make_tuple(std::vector<unsigned int> {2, 2, 2, 2, 12}, TransposePermutationArray {TransposePermutationDim(0),TransposePermutationDim(1), TransposePermutationDim(3), TransposePermutationDim(2), TransposePermutationDim(4)}, syn_type_float), /* float32_WBCH */
        std::make_tuple(std::vector<unsigned int> {3, 3, 3, 3, 7}, TransposePermutationArray {TransposePermutationDim(0), TransposePermutationDim(1), TransposePermutationDim(3), TransposePermutationDim(2), TransposePermutationDim(4)}, syn_type_float), /* float32_WBCH */
        std::make_tuple(std::vector<unsigned int> {3, 3, 3, 3, 13}, TransposePermutationArray {TransposePermutationDim(0), TransposePermutationDim(1), TransposePermutationDim(3), TransposePermutationDim(2), TransposePermutationDim(4)}, syn_type_float), /* float32_WBCH */
        std::make_tuple(std::vector<unsigned int> {3, 3, 3, 3, 15}, TransposePermutationArray {TransposePermutationDim(0), TransposePermutationDim(1), TransposePermutationDim(3), TransposePermutationDim(2), TransposePermutationDim(4)}, syn_type_float), /* float32_WBCH */
        std::make_tuple(std::vector<unsigned int> {81, 19, 19, 6, 64}, TransposePermutationArray {TransposePermutationDim(4), TransposePermutationDim(2), TransposePermutationDim(0), TransposePermutationDim(1), TransposePermutationDim(3)}, syn_type_float) /* float32_WBCH */
        ),
    SynGaudiTransposeSystemTest::PrintToStringParamName {});

INSTANTIATE_TEST_SUITE_P(
    wide_64b_transpose_high_rank,
    SynGaudiHighRankTransposeSystemTest,
    ::testing::Values(
                      std::make_tuple(std::vector<unsigned int> {4, 4, 4, 4, 4, 4},TransposePermutationArray {TPD_Depth,TPD_Height,TPD_Width,TPD_Channel,TransposePermutationDim(4),TransposePermutationDim(5)}, syn_type_int64), /* float32_WBCH */
                      std::make_tuple(std::vector<unsigned int> {4, 4, 4, 4, 4, 4},TransposePermutationArray {TPD_Channel,TPD_Width, TPD_Height, TPD_Depth, TransposePermutationDim(5), TransposePermutationDim(4)}, syn_type_int64), /* float32_WBCH */
                      std::make_tuple(std::vector<unsigned int> {2, 2, 2, 2, 2, 10}, TransposePermutationArray {TransposePermutationDim(0), TransposePermutationDim(1), TransposePermutationDim(3), TransposePermutationDim(5), TransposePermutationDim(2), TransposePermutationDim(4)}, syn_type_int64), /* float32_WBCH */
                      std::make_tuple(std::vector<unsigned int> {2, 2, 2, 2, 2, 12}, TransposePermutationArray {TransposePermutationDim(0), TransposePermutationDim(1), TransposePermutationDim(3), TransposePermutationDim(5), TransposePermutationDim(2), TransposePermutationDim(4)}, syn_type_int64), /* float32_WBCH */
                      std::make_tuple(std::vector<unsigned int> {3, 3, 3, 3, 3, 7}, TransposePermutationArray {TransposePermutationDim(0), TransposePermutationDim(1), TransposePermutationDim(3), TransposePermutationDim(5), TransposePermutationDim(2), TransposePermutationDim(4)}, syn_type_int64), /* float32_WBCH */
                      std::make_tuple(std::vector<unsigned int> {3, 3, 3, 3, 3, 13}, TransposePermutationArray {TransposePermutationDim(0), TransposePermutationDim(1), TransposePermutationDim(3), TransposePermutationDim(5), TransposePermutationDim(2), TransposePermutationDim(4)}, syn_type_int64), /* float32_WBCH */
                      std::make_tuple(std::vector<unsigned int> {3, 3, 3, 3, 3, 15}, TransposePermutationArray {TransposePermutationDim(0), TransposePermutationDim(1), TransposePermutationDim(3), TransposePermutationDim(5), TransposePermutationDim(2), TransposePermutationDim(4)}, syn_type_int64), /* float32_WBCH */
                      std::make_tuple(std::vector<unsigned int> {2, 2, 2, 2, 2, 1}, TransposePermutationArray {TransposePermutationDim(0), TransposePermutationDim(1), TransposePermutationDim(2), TransposePermutationDim(3), TransposePermutationDim(4), TransposePermutationDim(5)}, syn_type_int64), /* float32_WBCH */
                      std::make_tuple(std::vector<unsigned int> {2, 1, 1, 1, 1, 2}, TransposePermutationArray {TransposePermutationDim(0), TransposePermutationDim(1), TransposePermutationDim(2), TransposePermutationDim(3), TransposePermutationDim(4), TransposePermutationDim(5)}, syn_type_int64), /* float32_WBCH */
                      std::make_tuple(std::vector<unsigned int> {2, 2, 2, 2, 2, 2}, TransposePermutationArray {TransposePermutationDim(0), TransposePermutationDim(1), TransposePermutationDim(2), TransposePermutationDim(3), TransposePermutationDim(4), TransposePermutationDim(5)}, syn_type_int64) /* float32_WBCH */
                      ),
                      SynGaudiTransposeSystemTest::PrintToStringParamName {});

INSTANTIATE_TEST_SUITE_P(
    wide_64b_transpose_low_rank,
    SynGaudiTransposeSystemTest,
    ::testing::Values(
        std::make_tuple(std::vector<unsigned int> {4, 4, 4, 4}, TransposePermutationArray {TPD_Depth, TPD_Height, TPD_Width, TPD_Channel}, syn_type_int64),
        std::make_tuple(std::vector<unsigned int> {4, 4, 4, 4}, TransposePermutationArray {TPD_Channel, TPD_Width, TPD_Height, TPD_Depth}, syn_type_int64),
        std::make_tuple(std::vector<unsigned int> {2, 2, 2, 2, 10}, TransposePermutationArray {TransposePermutationDim(0), TransposePermutationDim(1), TransposePermutationDim(3), TransposePermutationDim(2), TransposePermutationDim(4)}, syn_type_int64),
        std::make_tuple(std::vector<unsigned int> {2, 2, 2, 2, 12}, TransposePermutationArray {TransposePermutationDim(0),TransposePermutationDim(1), TransposePermutationDim(3), TransposePermutationDim(2), TransposePermutationDim(4)}, syn_type_int64),
        std::make_tuple(std::vector<unsigned int> {3, 3, 3, 3, 7}, TransposePermutationArray {TransposePermutationDim(0), TransposePermutationDim(1), TransposePermutationDim(3), TransposePermutationDim(2), TransposePermutationDim(4)}, syn_type_int64),
        std::make_tuple(std::vector<unsigned int> {3, 3, 3, 3, 13}, TransposePermutationArray {TransposePermutationDim(0), TransposePermutationDim(1), TransposePermutationDim(3), TransposePermutationDim(2), TransposePermutationDim(4)}, syn_type_int64),
        std::make_tuple(std::vector<unsigned int> {3, 3, 3, 3, 15}, TransposePermutationArray {TransposePermutationDim(0), TransposePermutationDim(1), TransposePermutationDim(3), TransposePermutationDim(2), TransposePermutationDim(4)}, syn_type_int64),
        std::make_tuple(std::vector<unsigned int> {81, 19, 19, 6, 64}, TransposePermutationArray {TransposePermutationDim(4), TransposePermutationDim(2), TransposePermutationDim(0), TransposePermutationDim(1), TransposePermutationDim(3)}, syn_type_int64)
        ),
    SynGaudiTransposeSystemTest::PrintToStringParamName {});

SynGaudiTransposeSystemTest::SynGaudiTransposeSystemTest()
{
    setTestPackage(TEST_PACKAGE_TRANSPOSE);
}

void SynGaudiTransposeSystemTest::runTransposeTest()
{
    auto                      inputDimensions = std::get<0>(GetParam());
    TransposePermutationArray permutation     = std::get<1>(GetParam());
    synDataType               synType         = std::get<2>(GetParam());

    const unsigned inputSize =
        std::accumulate(std::begin(inputDimensions), std::end(inputDimensions), 1, std::multiplies<int>());
    if (synType == syn_type_float)
    {
        runTrasposeTest<float>(inputDimensions.data(), inputSize, permutation, synType);
    }
    else if (synType == syn_type_int32)
    {
        runTrasposeTest<int32_t>(inputDimensions.data(), inputSize, permutation, synType);
    }
    else if (synType == syn_type_bf16)
    {
        runTrasposeTest<bfloat16>(inputDimensions.data(), inputSize, permutation, synType);
    }
    else if (synType == syn_type_int64)
    {
        runTrasposeTest<int64_t>(inputDimensions.data(), inputSize, permutation, synType);
    }
    else
    {
        runTrasposeTest<int8_t>(inputDimensions.data(), inputSize, permutation, synType);
    }
}

template<typename DTYPE>
void SynGaudiTransposeSystemTest::runTrasposeTest(unsigned int*             input_dimensions,
                                                  const unsigned            inputSize,
                                                  TransposePermutationArray permutation,
                                                  synDataType               dataType)
{
    const unsigned int dim_num = permutation.size();
    std::vector<unsigned> output_dimensions(dim_num);

    initInputs(permutation, input_dimensions, inputSize, output_dimensions.data(), dataType);

    auto [in, out] =
        executeTransposeOperation(input_dimensions, output_dimensions.data(), dataType, dim_num, permutation);
    if (m_inputPermuted.has_value() || m_inputAllowPermuted.has_value())
    {
        validateResults(input_dimensions, dim_num, (DTYPE*)m_hostBuffers[out], (DTYPE*)m_hostBuffers[in]);
    }
    else
    {
        validateTransposeOperation(input_dimensions,
                                   (DTYPE*)m_hostBuffers[in],
                                   output_dimensions.data(),
                                   dim_num,
                                   dataType,
                                   (DTYPE*)m_hostBuffers[out]);
    }
}

template<typename DTYPE>
void SynGaudiTransposeSystemTest::validateTransposeOperation(unsigned int*      input_dimensions,
                                                             DTYPE*             inputArray,
                                                             unsigned int*      output_dimensions,
                                                             const unsigned int dim_num,
                                                             synDataType        dataType,
                                                             DTYPE*             pOutputBuffer)
{
    TSize inSizes[transposeParams.tensorDim];
    TSize outSizes[transposeParams.tensorDim];
    castNcopy(inSizes, input_dimensions, transposeParams.tensorDim);
    castNcopy(outSizes, output_dimensions, transposeParams.tensorDim);
    TensorPtr IFM = TensorPtr( new Tensor(transposeParams.tensorDim, inSizes, dataType));
    IFM->setTensorBuffer(inputArray, IFM->getDenseSizeInBytes(), dataType);
    TensorPtr OFM_ref = TensorPtr(new Tensor(transposeParams.tensorDim, outSizes, dataType));

    NodePtr ref_n = NodeFactory::createNode({IFM}, {OFM_ref}, &transposeParams, NodeFactory::transposeNodeTypeName, "");
    ref_n->RunOnCpu();
    DTYPE *ref_resultArray = (DTYPE *) OFM_ref->map();

    validateResults(input_dimensions, dim_num, pOutputBuffer, ref_resultArray);
}

template<typename DTYPE>
void SynGaudiTransposeSystemTest::validateResults(unsigned* dimensions,
                                                  unsigned  dimNum,
                                                  DTYPE*    output,
                                                  DTYPE*    outputRef)
{
    const int SIZE = std::accumulate(dimensions,
                                     dimensions + dimNum,
                                     1,  // initial acc
                                     [](unsigned int acc, unsigned int dim) { return acc * dim; });

    for (unsigned int i = 0; i < SIZE; ++i)
    {
        ASSERT_EQ(outputRef[i], output[i]) << "Wrong result at cell " << i;
    }
}

std::pair<unsigned, unsigned>
SynGaudiTransposeSystemTest::executeTransposeOperation(unsigned int*             input_dimensions,
                                                       unsigned int*             output_dimensions,
                                                       synDataType               dataType,
                                                       unsigned                  dim,
                                                       TransposePermutationArray permutation)
{
    unsigned transpose_in =
        createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, input_dimensions, dim, dataType);

    unsigned transpose_out =
        createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, output_dimensions, dim, dataType);

    if (m_inputPermuted.has_value())
    {
        synTensorPermutation perm;
        perm.dims = permutation.size();
        if (m_inputPermuted.value())
        {
            for (int i = 0; i < perm.dims; ++i)
            {
                perm.permutation[i] = permutation[i];
            }
            synTensorSetPermutation(m_tensors[transpose_in], &perm);
        }
        else
        {
            for (int i = 0; i < perm.dims; ++i)
            {
                perm.permutation[permutation[i]] = i;
            }
            synTensorSetPermutation(m_tensors[transpose_out], &perm);
        }
    }
    else if (m_inputAllowPermuted.has_value())
    {
        if (m_inputAllowPermuted.value())
        {
            synTensorSetAllowPermutation(m_tensors[transpose_in], 1);
        }
        else
        {
            synTensorSetAllowPermutation(m_tensors[transpose_out], 1);
        }
    }

    addNodeToGraph("transpose", {transpose_in}, {transpose_out}, (void*)&transposeParams, sizeof(transposeParams));

    compileAndRun();

    return std::make_pair(transpose_in, transpose_out);
}

void fillDenseStrides(unsigned* size, unsigned dimNum, unsigned sizeOfElement, unsigned* strides)
{
    strides[0] = sizeOfElement;
    for (int i = 0; i < dimNum - 1; ++i)
    {
        strides[i + 1] = strides[i] * size[i];
    }
}

void SynGaudiTransposeSystemTest::initInputs(TransposePermutationArray permutation,
                                             unsigned int*             input_dimensions,
                                             const unsigned            inputSize,
                                             unsigned int*             output_dimensions,
                                             synDataType               dataType)
{

    // Set output dimensions according to the transpose permutation
    const unsigned int dim_num = permutation.size();
    transposeParams.tensorDim = dim_num;
    for (unsigned int index = 0; index < dim_num; ++index)
    {
        output_dimensions[index] = input_dimensions[permutation[index]];
        transposeParams.permutation[index] = permutation[index];
    }
}

class SynGaudiLogicalStridedTranspose : public SynGaudiTransposeSystemTest
{
};

INSTANTIATE_TEST_SUITE_P(
    StridedLogical,
    SynGaudiLogicalStridedTranspose,
    ::testing::Values(std::make_tuple(std::vector<unsigned int> {4, 32, 2, 1},
                                      TransposePermutationArray {TPD_Height, TPD_Channel, TPD_Depth, TPD_Width},
                                      syn_type_bf16)),
    SynGaudiTransposeSystemTest::PrintToStringParamName {});

TEST_P_GC(SynGaudiLogicalStridedTranspose, logical_strided_input)
{
    m_inputPermuted.emplace(true);
    runTransposeTest();
}

TEST_P_GC(SynGaudiLogicalStridedTranspose, logical_strided_output)
{
    m_inputPermuted.emplace(false);
    runTransposeTest();
}

TEST_F_GC(SynGaudiTwoRunCompareTest, strided_dma_with_transpose_engine)
{
    unsigned inSizes[]  = {4, 242991};
    unsigned outSizes[] = {1, 242991};

    unsigned in  = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, inSizes, 2);
    unsigned out = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, outSizes, 2);

    synSliceParams p;
    p.axes[0] = 0;
    p.axes[1] = 1;

    p.steps[0] = 1;
    p.steps[1] = 1;

    p.starts[0] = 0;
    p.starts[1] = 0;

    p.ends[0] = 1;
    p.ends[1] = 242991;

    addNodeToGraph("slice", {in}, {out}, (void*)&p, sizeof(p));

    addConfigurationToRun(FIRST_RUN, "ENABLE_STRIDED_DMA_WITH_TRANSPOSE_ENGINE", "false");
    addConfigurationToRun(SECOND_RUN, "ENABLE_STRIDED_DMA_WITH_TRANSPOSE_ENGINE", "true");

    compareRunsResults({out});
}

class SynGaudiAllowPermuteLogicalTranspose : public SynGaudiTransposeSystemTest
{
};

INSTANTIATE_TEST_SUITE_P(
    StridedLogical,
    SynGaudiAllowPermuteLogicalTranspose,
    ::testing::Values(std::make_tuple(std::vector<unsigned int> {4, 32, 2, 1},
                                      TransposePermutationArray {TPD_Height, TPD_Channel, TPD_Depth, TPD_Width},
                                      syn_type_bf16)),
    SynGaudiTransposeSystemTest::PrintToStringParamName {});

TEST_P_GC(SynGaudiAllowPermuteLogicalTranspose, allow_input_permuted)
{
    // TODO [SW-89250] remove global configuration setting
    GlobalConfTestSetter setter("ALLOW_PERMUTATION_ON_USER_TRANSPOSE", "1");
    m_inputAllowPermuted.emplace(true);
    runTransposeTest();
}

TEST_P_GC(SynGaudiAllowPermuteLogicalTranspose, allow_output_permuted)
{
    // TODO [SW-89250] remove global configuration setting
    GlobalConfTestSetter setter("ALLOW_PERMUTATION_ON_USER_TRANSPOSE", "1");
    m_inputAllowPermuted.emplace(false);
    runTransposeTest();
}


class TransposeViaGemm : public SynGaudiTransposeSystemTest {};

INSTANTIATE_TEST_SUITE_P(
    Transpose_via_gemm,
    TransposeViaGemm,
    ::testing::Combine(
        ::testing::Values(std::vector<unsigned int> {550, 6, 1, 1},    // smaller then unit-matrix size on one dim
                          std::vector<unsigned int> {120, 550, 1, 1},  // smaller then unit-matrix size on one dim
                          std::vector<unsigned int> {550, 300, 1, 1},  // bigger then unit-matrix size
                          std::vector<unsigned int> {550, 300, 3, 1},  // 3D
                          std::vector<unsigned int> {550, 300, 10, 3}  // 4D
                          ),
        ::testing::Values(TransposePermutationArray {TPD_Width, TPD_Channel, TPD_4Dim_Batch, TPD_Height} /* WCBH */),
        ::testing::Values(syn_type_float,
                          syn_type_int32,
                          syn_type_uint32,
                          syn_type_bf16,
                          syn_type_fp16,
                          syn_type_int16,
                          syn_type_uint16,
                          syn_type_fp8_152,
                          syn_type_fp8_143,
                          syn_type_int8,
                          syn_type_uint8)),
    SynGaudiTransposeSystemTest::PrintToStringParamName {});

TEST_P_GC(TransposeViaGemm, dtype_test, {synDeviceGaudi3})
{
    ScopedConfigurationChange experimental_flags("EXP_FLAGS", "true");
    ScopedConfigurationChange xposeViaGemm("ENABLE_TRANSPOSE_VIA_GEMM", "1");
    runTransposeTest();
}

// Base class for testing performance of transpose nodes that were taken from various models
// Template params are: data_type, input_shape, permutation
class TestTransposePerformance
: public SynGaudiTestInfra
, public testing::WithParamInterface<std::tuple<synDataType, TestSizeVec, TestSizeVec>>
{
public:
    void runNode(const std::tuple<synDataType, TestSizeVec, TestSizeVec>& testParams)
    {
        const auto& [dataType, inSizes, permutation] = testParams;
        ASSERT_EQ(inSizes.size(), permutation.size());
        TestSizeVec outSizes(inSizes.size(), 0);
        synTransposeParams params;
        params.tensorDim = inSizes.size();
        for (size_t i = 0; i < params.tensorDim; ++i)
        {
            params.permutation[i] = TransposePermutationDim(permutation[i]);
            outSizes[i] = inSizes[permutation[i]];
        }

        unsigned inTensor = createPersistTensor(INPUT_TENSOR,
                                                MEM_INIT_RANDOM_POSITIVE,
                                                nullptr,
                                                const_cast<unsigned*>(inSizes.data()),
                                                inSizes.size(),
                                                dataType);

        unsigned outTensor = createPersistTensor(OUTPUT_TENSOR,
                                                 MEM_INIT_ALL_ZERO,
                                                 nullptr,
                                                 outSizes.data(),
                                                 outSizes.size(),
                                                 dataType);

        addNodeToGraph(NodeFactory::transposeNodeTypeName, {inTensor}, {outTensor}, &params, sizeof(params));
        compileTopology();
        ASSERT_FALSE(HasFailure()) << "Compilation failed";
        runTopology();
    }
};

TEST_P_GC(TestTransposePerformance, run_me_on_ASIC, {synDeviceGaudi3})
{
    runNode(GetParam());
}

// Data was taken from model: pt_resnet_lars_no_media_1step_bs128
INSTANTIATE_TEST_SUITE_P(
    pt_resnet_lars_no_media_1step_bs128,
    TestTransposePerformance,
    ::testing::Values(// data_type,      input_shape,               permutation
        std::make_tuple(syn_type_bf16,   TestSizeVec {50176,3,128}, TestSizeVec {1,0,2}),
        std::make_tuple(syn_type_bf16,   TestSizeVec {64,128,4,3},  TestSizeVec {1,0,2,3}),
        std::make_tuple(syn_type_bf16,   TestSizeVec {64,128,2,1},  TestSizeVec {1,0,2,3}),
        std::make_tuple(syn_type_single, TestSizeVec {192,7,7},     TestSizeVec {1,0,2}),
        std::make_tuple(syn_type_single, TestSizeVec {4096,3,3},    TestSizeVec {1,0,2}),
        std::make_tuple(syn_type_single, TestSizeVec {16384,3,3},   TestSizeVec {1,0,2}),
        std::make_tuple(syn_type_single, TestSizeVec {65536,3,3},   TestSizeVec {1,0,2}),
        std::make_tuple(syn_type_single, TestSizeVec {262144,3,3},  TestSizeVec {1,0,2})));

// Data was taken from model: pt_unet3d-md-ptl_8x_g2
INSTANTIATE_TEST_SUITE_P(
    pt_unet3d_md_ptl_8x_g2,
    TestTransposePerformance,
    ::testing::Values(  // data_type,      input_shape,                   permutation
        std::make_tuple(syn_type_single, TestSizeVec {8388608, 2}, TestSizeVec {1, 0}),
        std::make_tuple(syn_type_single, TestSizeVec {131072, 128}, TestSizeVec {1, 0}),
        std::make_tuple(syn_type_bf16, TestSizeVec {2097152, 4, 2}, TestSizeVec {1, 0, 2}),
        std::make_tuple(syn_type_bf16, TestSizeVec {4, 128, 128, 128, 2}, TestSizeVec {1, 2, 3, 0, 4}),
        std::make_tuple(syn_type_single, TestSizeVec {4, 128, 128, 128, 2}, TestSizeVec {1, 2, 3, 0, 4}),
        std::make_tuple(syn_type_single, TestSizeVec {2097152, 4, 2, 1}, TestSizeVec {1, 0, 2, 3}),
        std::make_tuple(syn_type_single, TestSizeVec {4, 2097152, 2, 1}, TestSizeVec {1, 0, 2, 3}),
        std::make_tuple(syn_type_bf16, TestSizeVec {4, 128, 4, 1, 1}, TestSizeVec {1, 0, 2, 3, 4}),
        std::make_tuple(syn_type_bf16, TestSizeVec {32, 128, 6, 3, 3}, TestSizeVec {1, 0, 2, 3, 4}),
        std::make_tuple(syn_type_bf16, TestSizeVec {32, 256, 6, 3, 3}, TestSizeVec {1, 0, 2, 3, 4}),
        std::make_tuple(syn_type_bf16, TestSizeVec {64, 128, 4, 3, 3}, TestSizeVec {1, 0, 2, 3, 4}),
        std::make_tuple(syn_type_bf16, TestSizeVec {64, 256, 4, 3, 3}, TestSizeVec {1, 0, 2, 3, 4}),
        std::make_tuple(syn_type_bf16, TestSizeVec {8192, 32768}, TestSizeVec {1, 0}),
        std::make_tuple(syn_type_bf16, TestSizeVec {32768, 4096}, TestSizeVec {1, 0}),
        std::make_tuple(syn_type_single, TestSizeVec {1024, 3, 3, 3}, TestSizeVec {1, 2, 0, 3}),
        std::make_tuple(syn_type_single, TestSizeVec {2048, 3, 3, 3}, TestSizeVec {1, 2, 0, 3}),
        std::make_tuple(syn_type_single, TestSizeVec {2048, 2, 2, 2}, TestSizeVec {1, 2, 0, 3}),
        std::make_tuple(syn_type_single, TestSizeVec {4096, 3, 3, 3}, TestSizeVec {1, 2, 0, 3}),
        std::make_tuple(syn_type_single, TestSizeVec {8192, 3, 3, 3}, TestSizeVec {1, 2, 0, 3}),
        std::make_tuple(syn_type_single, TestSizeVec {8192, 2, 2, 2}, TestSizeVec {1, 2, 0, 3}),
        std::make_tuple(syn_type_single, TestSizeVec {32768, 3, 3, 3}, TestSizeVec {1, 2, 0, 3}),
        std::make_tuple(syn_type_single, TestSizeVec {81920, 2, 2, 2}, TestSizeVec {1, 2, 0, 3}),
        std::make_tuple(syn_type_single, TestSizeVec {81920, 3, 3, 3}, TestSizeVec {1, 2, 0, 3}),
        std::make_tuple(syn_type_single, TestSizeVec {128, 3, 3, 3}, TestSizeVec {1, 2, 0, 3}),
        std::make_tuple(syn_type_single, TestSizeVec {32, 4, 1, 1, 1}, TestSizeVec {1, 0, 2, 3, 4}),
        std::make_tuple(syn_type_single, TestSizeVec {864, 32}, TestSizeVec {1, 0}),
        std::make_tuple(syn_type_single, TestSizeVec {1728, 32}, TestSizeVec {1, 0}),
        std::make_tuple(syn_type_single, TestSizeVec {256, 64}, TestSizeVec {1, 0}),
        std::make_tuple(syn_type_single, TestSizeVec {1728, 64}, TestSizeVec {1, 0}),
        std::make_tuple(syn_type_single, TestSizeVec {3456, 64}, TestSizeVec {1, 0}),
        std::make_tuple(syn_type_single, TestSizeVec {512, 128}, TestSizeVec {1, 0}),
        std::make_tuple(syn_type_single, TestSizeVec {3456, 128}, TestSizeVec {1, 0}),
        std::make_tuple(syn_type_single, TestSizeVec {6912, 128}, TestSizeVec {1, 0}),
        std::make_tuple(syn_type_single, TestSizeVec {1024, 256}, TestSizeVec {1, 0}),
        std::make_tuple(syn_type_single, TestSizeVec {6912, 256}, TestSizeVec {1, 0}),
        std::make_tuple(syn_type_single, TestSizeVec {13824, 256}, TestSizeVec {1, 0}),
        std::make_tuple(syn_type_single, TestSizeVec {2048, 320}, TestSizeVec {1, 0}),
        std::make_tuple(syn_type_single, TestSizeVec {8640, 320}, TestSizeVec {1, 0}),
        std::make_tuple(syn_type_single, TestSizeVec {17280, 320}, TestSizeVec {1, 0}),
        std::make_tuple(syn_type_single, TestSizeVec {2560, 320}, TestSizeVec {1, 0}),
        std::make_tuple(syn_type_single, TestSizeVec {6912, 320}, TestSizeVec {1, 0}),
        std::make_tuple(syn_type_single, TestSizeVec {3456, 256}, TestSizeVec {1, 0}),
        std::make_tuple(syn_type_single, TestSizeVec {1728, 128}, TestSizeVec {1, 0}),
        std::make_tuple(syn_type_single, TestSizeVec {864, 64}, TestSizeVec {1, 0}),
        std::make_tuple(syn_type_single, TestSizeVec {108, 32}, TestSizeVec {1, 0}),
        std::make_tuple(syn_type_int32, TestSizeVec {2097152, 3}, TestSizeVec {1, 0}),
        std::make_tuple(syn_type_bf16, TestSizeVec {4, 128, 128, 128}, TestSizeVec {1, 2, 3, 0}),
        std::make_tuple(syn_type_single, TestSizeVec {3648832, 4, 1, 1}, TestSizeVec {1, 0, 2, 3}),
        std::make_tuple(syn_type_single, TestSizeVec {3706200, 4, 1, 1}, TestSizeVec {1, 0, 2, 3}),
        std::make_tuple(syn_type_single, TestSizeVec {16384, 3, 3, 3}, TestSizeVec {1, 2, 0, 3}),
        std::make_tuple(syn_type_single, TestSizeVec {32768, 2, 2, 2}, TestSizeVec {1, 2, 0, 3}),
        std::make_tuple(syn_type_single, TestSizeVec {65536, 3, 3, 3}, TestSizeVec {1, 2, 0, 3}),
        std::make_tuple(syn_type_single, TestSizeVec {131072, 3, 3, 3}, TestSizeVec {1, 2, 0, 3}),
        std::make_tuple(syn_type_single, TestSizeVec {102400, 3, 3, 3}, TestSizeVec {1, 2, 0, 3}),
        std::make_tuple(syn_type_single, TestSizeVec {204800, 3, 3, 3}, TestSizeVec {1, 2, 0, 3}),
        std::make_tuple(syn_type_single, TestSizeVec {102400, 2, 2, 2}, TestSizeVec {1, 2, 0, 3})));
