#include "compilation_hal_reader.h"
#include "platform/gaudi3/graph_compiler/gaudi3_graph.h"
#include "platform/gaudi3/graph_compiler/passes.h"
#include "batch_norm_eviction_fuser_test_common.h"

class BatchNormStage1FwdEvictionFuserTestGaudi3 : public BatchNormStage1FwdEvictionFuserTestCommon<Gaudi3Graph>
{
protected:
    virtual void preBundleHandling() override
    {
        using namespace gaudi3;
        ASSERT_TRUE(loadTpcKernels(m_graph));
    }
};
class BatchNormStage2FwdEvictionFuserTestGaudi3 : public BatchNormStage2FwdEvictionFuserTestCommon<Gaudi3Graph>
{
protected:
    virtual void preBundleHandling() override
    {
        using namespace gaudi3;
        ASSERT_TRUE(loadTpcKernels(m_graph));
    }
};

TEST_P(BatchNormStage1FwdEvictionFuserTestGaudi3, stage1_fwd_consumer_eviction_test)
{
    runBnEvictionFuserTest();  // Test parameters are handled inside.
}

INSTANTIATE_TEST_SUITE_P(BNEvictionFuser, BatchNormStage1FwdEvictionFuserTestGaudi3,
                         ::testing::ValuesIn({ std::tuple<bool, bool>
                             {true, false},  // Expect fusion because the mme output is persistent
                             {false, true},  // Expect fusion because the mme output is consumed outside the bundle
                             {true, true},   // Expect fusion because both conditions apply
                             {false, false}  // Expect no fusion
                         }));

TEST_P(BatchNormStage2FwdEvictionFuserTestGaudi3, stage2_fwd_consumer_eviction_test)
{
    runBnEvictionFuserTest();  // Test parameters are handled inside.
}

INSTANTIATE_TEST_SUITE_P(BNEvictionFuser2, BatchNormStage2FwdEvictionFuserTestGaudi3,
                         ::testing::ValuesIn({ std::tuple<bool, bool>
                              {true, false},  // Expect fusion because the mme output is persistent
                              {false, true},  // Expect fusion because the mme output is consumed outside the bundle
                              {true, true},   // Expect fusion because both conditions apply
                             {false, false}  // Expect no fusion
                         }));
