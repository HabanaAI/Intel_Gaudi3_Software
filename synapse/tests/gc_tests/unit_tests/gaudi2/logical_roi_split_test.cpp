#include <gtest/gtest.h>
#include <iostream>
#include "graph_optimizer_test.h"
#include "tensor.h"
#include "data_type_utils.h"
#include "scoped_configuration_change.h"
#include "platform/gaudi2/graph_compiler/gaudi2_graph.h"
#include <graph_compiler/habana_nodes/node_factory.h>

class LogicalRoiTest
: public GraphOptimizerTest
, public testing::WithParamInterface<std::tuple<TSize, unsigned, unsigned, std::vector<unsigned>>>
{
};

TEST_P(LogicalRoiTest, logical_roi_split_according_to_num_of_physical_engines)
{
    Gaudi2Graph g;

    auto numSamples     = std::get<0>(GetParam());
    auto numOfChunks     = std::get<1>(GetParam());
    auto numOfPhysicalEngs = std::get<2>(GetParam());
    auto expectedLogicalRois = std::get<3>(GetParam());

    /************************* Get optimized logical ROIs *************************/

    std::vector<TSize> logicalRois = splitToChunks(numSamples, numOfChunks, 0, numOfPhysicalEngs);

    /*************************** Validate logical ROIs ****************************/

    ASSERT_EQ(numOfChunks, logicalRois.size()) << "Num of expected logical rois " << numOfChunks
                                               << " is different than actual num of logical rois " << logicalRois.size();

    for (unsigned roiId = 0; roiId < logicalRois.size(); roiId++)
    {
        ASSERT_EQ(logicalRois[roiId], expectedLogicalRois[roiId]) << "Expected logical roi[" << roiId << "] " << expectedLogicalRois[roiId]
                                                                  << " is different than actual logical roi " << logicalRois[roiId];
    }

}

INSTANTIATE_TEST_SUITE_P(
    _,
    LogicalRoiTest,
    ::testing::Values(std::make_tuple(683, 3, 24, std::vector<unsigned>{240, 216, 227}),
                      std::make_tuple(40, 3, 24, std::vector<unsigned>{14, 13, 13}),
                      std::make_tuple(8000, 8, 24, std::vector<unsigned>{1008, 1008, 1008, 1008, 1008, 984, 984, 992}),
                      std::make_tuple(1000, 2, 24, std::vector<unsigned>{504, 496})));