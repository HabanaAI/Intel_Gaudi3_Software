#include "gc_gaudi_test_infra.h"
#include "node_factory.h"
#include "recipe.h"
#include "infra/gc_synapse_test.h"
#include "runtime/common/recipe/recipe_handle_impl.hpp"

#include <cmath>

class SynGaudiTensorOverlapNodeOrderTest
: public SynGaudiTestInfra
, public testing::WithParamInterface<std::tuple<int, int, int, int>>
{
public:
    struct GetName
    {
        std::string operator()(const ::testing::TestParamInfo<std::tuple<int, int, int, int>> & info) const
        {
            ::std::stringstream ss;
            ss << "tensor_overlap_node_order_offsets_"
               << std::get<0>(info.param) << "_"
               << std::get<1>(info.param) << "_"
               << std::get<2>(info.param) << "_"
               << std::get<3>(info.param);
            return ss.str();
        }
    };

    void runTest(const std::tuple<int, int, int, int>& params)
    {
        const int tensor0Offset = std::get<0>(params);
        const int tensor1Offset = std::get<1>(params);
        const int tensor2Offset = std::get<2>(params);
        const int tensor3Offset = std::get<3>(params);

        unsigned sectionIdx    = createSection(s_tensorSize * 4);
        unsigned inputTensor1  = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, s_inValues, dataDims, 4 ,syn_type_single, nullptr, nullptr, 0, tensor0Offset, &sectionIdx, nullptr);
        unsigned inputTensor2  = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, s_inValues, dataDims, 4 ,syn_type_single, nullptr, nullptr, 0, tensor1Offset, &sectionIdx, nullptr);
        unsigned outputTensor1 = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, dataDims, 4,  syn_type_single, nullptr, nullptr, 0, tensor2Offset, &sectionIdx, nullptr);
        unsigned outputTensor2 = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, dataDims, 4,  syn_type_single, nullptr, nullptr, 0, tensor3Offset, &sectionIdx, nullptr);

        addNodeToGraph("add_fwd_f32", {inputTensor1, inputTensor2}, {outputTensor1}, nullptr, 0);
        addNodeToGraph("add_fwd_f32", {inputTensor1, inputTensor2}, {outputTensor2}, nullptr, 0);

        auto graphData = m_graphs[0];
        synStatus status = synGraphCompile(&graphData.recipeHandle,
                                           graphData.graphHandle,
                                           GetTestFileName().c_str(),
                                           nullptr);

        synStatus expectedResult = synSuccess;
        if (std::abs((int)tensor2Offset - (int)tensor3Offset) < s_tensorSize ||
            std::abs((int)tensor2Offset - (int)tensor0Offset) < s_tensorSize ||
            std::abs((int)tensor2Offset - (int)tensor1Offset) < s_tensorSize ||
            std::abs((int)tensor3Offset - (int)tensor0Offset) < s_tensorSize ||
            std::abs((int)tensor3Offset - (int)tensor1Offset) < s_tensorSize)
        {
            expectedResult = synFail;
        }

        ASSERT_EQ(status, expectedResult);
    }

    unsigned dataDims[4] = {1, 2, 2, 1};
    constexpr static float s_inValues[4] = {1.0, 2.0, 3.0, 4.0};
    constexpr static unsigned s_tensorSize = sizeof(s_inValues);
};

constexpr float SynGaudiTensorOverlapNodeOrderTest::s_inValues[4];

std::vector<std::tuple<int, int, int, int>> generateOffsets()
{
    std::vector<std::tuple<int, int, int, int>> paramsToTest;
    const unsigned tensorSize = SynGaudiTensorOverlapNodeOrderTest::s_tensorSize;
    const int offset0 = 0;
    for (int offset1 = 0; offset1 <= tensorSize; offset1 += tensorSize / 2)
    {
        for (int offset2 = 0; offset2 <= tensorSize; offset2 += tensorSize / 2)
        {
            for (int offset3 = 0; offset3 <= tensorSize; offset3 += tensorSize / 2)
            {
                    paramsToTest.emplace_back(std::make_tuple(offset0,
                                                              offset0 + offset1,
                                                              offset0 + offset1 + offset2,
                                                              offset0 + offset1 + offset2 + offset3));
            }
        }
    }

    return paramsToTest;
}

INSTANTIATE_TEST_SUITE_P(, SynGaudiTensorOverlapNodeOrderTest,
                         ::testing::ValuesIn(generateOffsets()),
                         SynGaudiTensorOverlapNodeOrderTest::GetName());

TEST_P_GC(SynGaudiTensorOverlapNodeOrderTest, tensor_overlap_node_order_test)
{
    std::tuple<int, int, int, int> params = GetParam();
    runTest(params);
}
