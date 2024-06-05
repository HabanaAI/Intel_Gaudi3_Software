#include "mme_test_base.h"
#include "include/gaudi/mme_descriptor_generator.h"

class MMEGaudiSanityTests : public MMEGaudiVerification
{
    std::string getSubFolder() override { return "sanity_tests/"; }
};
class MMEGaudiRegressionTests : public MMEGaudiVerification
{
    std::string getSubFolder() override { return "regression_tests/"; }
};
class MMEGaudiFullTests : public MMEGaudiVerification
{
    std::string getSubFolder() override { return "full_tests/"; }
};
class MMEGaudiNetworkTests : public MMEGaudiVerification
{
    std::string getSubFolder() override { return "network_tests/"; }
};

//============= Sanity tests ======================
TEST_F(MMEGaudiSanityTests, sanity_bgemm_tests)
{
    runTest(test_info_->name());
}
TEST_F(MMEGaudiSanityTests, sanity_conv_tests)
{
    runTest(test_info_->name());
}
TEST_F(MMEGaudiSanityTests, sanity_dedx_tests)
{
    runTest(test_info_->name());
}
TEST_F(MMEGaudiSanityTests, sanity_dedw_tests)
{
    runTest(test_info_->name());
}
//============= Regression tests ======================
TEST_F(MMEGaudiRegressionTests, bgemm_mini_reg)
{
    runTest(test_info_->name());
}
TEST_F(MMEGaudiRegressionTests, bgemm_zero_cd_ab_tests)
{
    runTest(test_info_->name());
}
TEST_F(MMEGaudiRegressionTests, bgemm_zero_cd_abt_tests)
{
    runTest(test_info_->name());
}
TEST_F(MMEGaudiRegressionTests, bgemm_zero_cd_atb_tests)
{
    runTest(test_info_->name());
}
TEST_F(MMEGaudiRegressionTests, bgemm_zero_cd_atbt_tests)
{
    runTest(test_info_->name());
}
TEST_F(MMEGaudiRegressionTests, bgemm_atb_tests)
{
    runTest(test_info_->name());
}
TEST_F(MMEGaudiRegressionTests, bgemm_abt_tests)
{
    runTest(test_info_->name());
}
TEST_F(MMEGaudiRegressionTests, bgemm_atbt_tests)
{
    runTest(test_info_->name());
}
TEST_F(MMEGaudiRegressionTests, bgemm_broadcast_tests)
{
    runTest(test_info_->name());
}
TEST_F(MMEGaudiRegressionTests, conv_tests)
{
    runTest(test_info_->name());
}
TEST_F(MMEGaudiRegressionTests, conv_zero_cd_tests)
{
    runTest(test_info_->name());
}
TEST_F(MMEGaudiRegressionTests, dedw_mini_reg)
{
    runTest(test_info_->name());
}
TEST_F(MMEGaudiRegressionTests, dedw_sbreuse_tests)
{
    runTest(test_info_->name());
}
TEST_F(MMEGaudiRegressionTests, dedw_as_bgemm)
{
    runTest(test_info_->name());
}
TEST_F(MMEGaudiRegressionTests, dedw_unroll)
{
    runTest(test_info_->name());
}
TEST_F(MMEGaudiRegressionTests, sbreuse_tests)
{
    runTest(test_info_->name());
}
TEST_F(MMEGaudiRegressionTests, dedw_dilations)
{
    runTest(test_info_->name());
}
//================ Complementary list of tests =======================
TEST_F(MMEGaudiFullTests, bgemm_broadcast)
{
    runTest(test_info_->name());
}

TEST_F(MMEGaudiFullTests, amos_test)
{
    runTest(test_info_->name());
}
TEST_F(MMEGaudiFullTests, bgemm_1w4h_tests)
{
    runTest(test_info_->name());
}
TEST_F(MMEGaudiFullTests, bgemm_2wx2h_2x_tests)
{
    runTest(test_info_->name());
}
TEST_F(MMEGaudiFullTests, bgemm_2wx2h_tests)
{
    runTest(test_info_->name());
}
TEST_F(MMEGaudiFullTests, bgemm_2xw_ab)
{
    runTest(test_info_->name());
}
TEST_F(MMEGaudiFullTests, bgemm_2xw_abt)
{
    runTest(test_info_->name());
}
TEST_F(MMEGaudiFullTests, bgemm_2xw_atb)
{
    runTest(test_info_->name());
}
TEST_F(MMEGaudiFullTests, bgemm_4w1h_tests)
{
    runTest(test_info_->name());
}
TEST_F(MMEGaudiFullTests, bgemm_ab_bert_tests)
{
    runTest(test_info_->name());
}
TEST_F(MMEGaudiFullTests, bgemm_large_gemm)
{
    runTest(test_info_->name());
}
TEST_F(MMEGaudiFullTests, conv3d_padding)
{
    runTest(test_info_->name());
}
TEST_F(MMEGaudiFullTests, conv3x3_bf16_basic)
{
    runTest(test_info_->name());
}
TEST_F(MMEGaudiFullTests, conv3x3_bf16_lowering)
{
    runTest(test_info_->name());
}
TEST_F(MMEGaudiFullTests, conv3x3_fp32_basic)
{
    runTest(test_info_->name());
}
TEST_F(MMEGaudiFullTests, dedw_complex_tests)
{
    runTest(test_info_->name());
}
TEST_F(MMEGaudiFullTests, dedw_converted_from_gaudi2)
{
    runTest(test_info_->name());
}
TEST_F(MMEGaudiFullTests, dedw_lowering)
{
    runTest(test_info_->name());
}
TEST_F(MMEGaudiFullTests, dedw_tests)
{
    runTest(test_info_->name());
}
TEST_F(MMEGaudiFullTests, gemm_121x88_float2float)
{
    runTest(test_info_->name());
}
TEST_F(MMEGaudiFullTests, gemm_128x1536_repeat_11)
{
    runTest(test_info_->name());
}
TEST_F(MMEGaudiFullTests, gemm_128x512_repeat_2)
{
    runTest(test_info_->name());
}
TEST_F(MMEGaudiFullTests, gemm_128x512_repeat_32)
{
    runTest(test_info_->name());
}
TEST_F(MMEGaudiFullTests, gemm_128x512_repeat_90)
{
    runTest(test_info_->name());
}
TEST_F(MMEGaudiFullTests, gemm_64x256_bfloat2bfloat)
{
    runTest(test_info_->name());
}
TEST_F(MMEGaudiFullTests, gemm_64x4096_bfloat2bfloat)
{
    runTest(test_info_->name());
}
TEST_F(MMEGaudiFullTests, gemm_64x4096_bfloat2bfloat_hbm)
{
    runTest(test_info_->name());
}
TEST_F(MMEGaudiFullTests, gemm_tests)
{
    runTest(test_info_->name());
}
TEST_F(MMEGaudiFullTests, resnet50_L0)
{
    runTest(test_info_->name());
}
TEST_F(MMEGaudiFullTests, stripes_reuse)
{
    runTest(test_info_->name());
}

//=============== Network tests ==================
TEST_F(MMEGaudiNetworkTests, dedw_nodes_v4_syn_resnet_b32)
{
    runTest(test_info_->name());
}
TEST_F(MMEGaudiNetworkTests, dedw_nodes_v4_tf_bert_finetuning_squat_b24)
{
    runTest(test_info_->name());
}
TEST_F(MMEGaudiNetworkTests, dedw_nodes_v4_tf_unet2d)
{
    runTest(test_info_->name());
}
TEST_F(MMEGaudiNetworkTests, dedw_nodes_v4_tf_unet3d)
{
    runTest(test_info_->name());
}
TEST_F(MMEGaudiNetworkTests, dedw_nodes_v4_tf_resnext_b128)
{
    runTest(test_info_->name());
}
