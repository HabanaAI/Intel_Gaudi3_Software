#include "operand_access.h"

#include "mme_access_pattern.h"
#include "mme_unit_test.h"
#include <variant>

using namespace MmeCommon;

using OperandEnumerations = std::tuple<EMmeInternalOperand, EMmeOperand>;

class MmeUTOperandAccessTest
: public MMEUnitTest
, public ::testing::WithParamInterface<OperandEnumerations>
{
public:
    constexpr static uint64_t DIM01_GRANULARITY = 7;
    constexpr static uint64_t DIM2_GRANULARITY = 51;
    constexpr static uint64_t DIM34_GRANULARITY = 1;

    constexpr static uint64_t DIM0_BASE_SZ = 1;
    constexpr static uint64_t DIM1_BASE_SZ = 2;
    constexpr static uint64_t DIM2_BASE_SZ = 3;
    constexpr static uint64_t DIM3_BASE_SZ = 4;
    constexpr static uint64_t DIM4_BASE_SZ = 5;

    // Generate some arbitrary access pattern for one of the operands -
    // tensor dims 0,1 mapped to idx space dim 0
    // tensor dim 2 mapped to idx space dim 1
    // tensor dims 3,4 mapped to idx space dim 2
    AccessPattern generateAP(EMmeInternalOperand internalOperand, EMmeOperand externalOperand)
    {
        // Empty access pattern
        AccessPattern ap;

        auto role = externalOperand == e_mme_op_x   ? OperandRole::X
                    : externalOperand == e_mme_op_w ? OperandRole::W
                                                    : OperandRole::Y;

        auto& tensorAP = ap.operandAccessPatterns[role];

        AccessPattern::TensorAccessPattern::DimAccessPattern dimAP {};
        dimAP.indexSpaceDim = 0;
        dimAP.size = DIM0_BASE_SZ;
        tensorAP.dimsAccessPattern.push_back(dimAP);
        dimAP.indexSpaceDim = 0;
        dimAP.size = DIM1_BASE_SZ;
        tensorAP.dimsAccessPattern.push_back(dimAP);
        dimAP.indexSpaceDim = 1;
        dimAP.size = DIM2_BASE_SZ;
        tensorAP.dimsAccessPattern.push_back(dimAP);
        dimAP.indexSpaceDim = 2;
        dimAP.size = DIM3_BASE_SZ;
        tensorAP.dimsAccessPattern.push_back(dimAP);
        dimAP.indexSpaceDim = 2;
        dimAP.size = DIM4_BASE_SZ;
        tensorAP.dimsAccessPattern.push_back(dimAP);

        switch (internalOperand)
        {
            case e_mme_op_a:
                ap.roleA = role;
                break;
            case e_mme_op_b:
                ap.roleB = role;
                break;
            default:
                ap.roleC = role;
                break;
        }

        return ap;
    }
};

TEST_P(MmeUTOperandAccessTest, operand_access_should_supply_granularity_of_operands)
{
    auto [internalOperand, externalOperand] = GetParam();

    // Generate some arbitrary access pattern for the output -
    // tensor dims 0,1 mapped to idx space dim 0
    // tensor dim 2 mapped to idx space dim 1
    AccessPattern ap = generateAP(internalOperand, externalOperand);

    // Generate an atbitrary granularity vector
    AccessPattern::IndexSpaceVector granularity = {DIM01_GRANULARITY, DIM2_GRANULARITY, DIM34_GRANULARITY};

    // When asking for the granularity of operation's output
    OperandAccess operandAccess(ap, granularity);

    // Then the granularity of the output should be the same as the granularity vector
    EXPECT_EQ(DIM01_GRANULARITY * DIM0_BASE_SZ, operandAccess.granularityByTensorDim(internalOperand, 0));
    EXPECT_EQ(DIM01_GRANULARITY * DIM1_BASE_SZ, operandAccess.granularityByTensorDim(internalOperand, 1));
    EXPECT_EQ(DIM2_GRANULARITY * DIM2_BASE_SZ, operandAccess.granularityByTensorDim(internalOperand, 2));
    EXPECT_EQ(DIM34_GRANULARITY * DIM3_BASE_SZ, operandAccess.granularityByTensorDim(internalOperand, 3));
    EXPECT_EQ(DIM34_GRANULARITY * DIM4_BASE_SZ, operandAccess.granularityByTensorDim(internalOperand, 4));

    EXPECT_EQ(DIM01_GRANULARITY * DIM0_BASE_SZ, operandAccess.granularityByTensorDim(externalOperand, 0));
    EXPECT_EQ(DIM01_GRANULARITY * DIM1_BASE_SZ, operandAccess.granularityByTensorDim(externalOperand, 1));
    EXPECT_EQ(DIM2_GRANULARITY * DIM2_BASE_SZ, operandAccess.granularityByTensorDim(externalOperand, 2));
    EXPECT_EQ(DIM34_GRANULARITY * DIM3_BASE_SZ, operandAccess.granularityByTensorDim(externalOperand, 3));
    EXPECT_EQ(DIM34_GRANULARITY * DIM4_BASE_SZ, operandAccess.granularityByTensorDim(externalOperand, 4));
}

INSTANTIATE_TEST_CASE_P(_,
                        MmeUTOperandAccessTest,
                        ::testing::Values(OperandEnumerations {e_mme_op_a, e_mme_op_x},
                                          OperandEnumerations {e_mme_op_b, e_mme_op_w},
                                          OperandEnumerations {e_mme_op_c, e_mme_op_y}));
