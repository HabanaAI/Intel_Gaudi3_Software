#include "graph_compiler/passes/sram_management/bundle.h"
#include "graph_compiler/passes/sram_management/slice_mapping.h"
#include "graph_optimizer_test.h"

class ReshapeMapperTest : public GraphOptimizerTest
{
};

TEST_F(ReshapeMapperTest, test_bwd_mapping)
{
    SizeArray inSize = {10, 6, 6, 28};
    SizeArray outSize = {360, 28};

    pTensor reshapeIn = pTensor(new Tensor(4, inSize.data(), syn_type_bf16));
    pTensor reshapeOut = pTensor(new Tensor(2, outSize.data(), syn_type_bf16));

    pSlicedOperand inputOperand  = std::make_shared<SlicedOperand>(reshapeIn);
    pSlicedOperand outputOperand = std::make_shared<SlicedOperand>(reshapeOut);

    pBackwardSliceMapping mapper = ReshapeSliceMapper::mapOutputToInput(inputOperand, outputOperand);

    pSliceReference outRef = std::make_shared<SliceReference>(outputOperand);
    outRef->coordinates = {0,1,0,0,0};

    auto inRefList = mapper->getInputs(std::make_pair(outRef, 0));
    ASSERT_EQ(inRefList.size(), 1);
    auto inRef = inRefList.front();

    CoordArray expectedCoors = {0,0,0,1,0};
    ASSERT_EQ(inRef->coordinates, expectedCoors);

    // Check it the other way around
    pBackwardSliceMapping reverseMapper = ReshapeSliceMapper::mapOutputToInput(outputOperand, inputOperand);
    auto outRefList = reverseMapper->getInputs(std::make_pair(inRef, 0));
    expectedCoors = {0,1,0,0,0};
    ASSERT_EQ(outRefList.front()->coordinates, expectedCoors);
}
