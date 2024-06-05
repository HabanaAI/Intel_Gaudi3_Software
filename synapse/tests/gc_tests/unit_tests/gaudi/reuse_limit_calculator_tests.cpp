#include <graph_compiler/types.h>
#include <platform/gaudi/graph_compiler/gaudi_graph.h>
#include "passes/sram_management/reuse_limit_calculator.h"
#include "graph_optimizer_test.h"
#include "utils.h"
#include "tensor.h"
#include "node_factory.h"
#include "hal_reader/gaudi1/hal_reader.h"
#include "graph_compiler/passes/sram_management/slicing_brain.h"

class RLCTest : public GraphOptimizerTest
{
protected:
    void SetUp()
    {
        GraphOptimizerTest::SetUp();
        GaudiGraph g;
        MMESlicingBrain dummyBrain(g);  // init brain knobs

        // Get some more substance from the tests
        setGlobalConfForTest(GCFG_SRAM_SLICER_REUSE_LIMIT_FACTOR, "2.0");
    }
};

TEST_F(RLCTest, rlc_should_return_full_tensor_sizes_for_small_common_dim_fwd)
{
    TSize xSize[] = {16, 160000};
    TSize wSize[] = {65536, 16};
    TSize ySize[] = {65536, 160000};

    pTensor x = std::make_shared<Tensor>(ARRAY_SIZE(xSize), xSize, syn_type_float);
    pTensor w = std::make_shared<Tensor>(ARRAY_SIZE(wSize), wSize, syn_type_float);
    pTensor y = std::make_shared<Tensor>(ARRAY_SIZE(ySize), ySize, syn_type_float);

    synGEMMParams params{};
    pNode gemm = NodeFactory::createNode({x, w}, {y}, &params, NodeFactory::gemmNodeTypeName, "gemm");

    ReuseLimitCalculator rlc(*GaudiHalReader::instance(synDeviceGaudi), gemm);

    ASSERT_EQ(rlc.getLimit(), 160000);
}

TEST_F(RLCTest, rlc_should_return_full_tensor_sizes_for_small_common_dim_dedx)
{
    TSize xSize[] = {160000, 65536};
    TSize wSize[] = {16, 160000};
    TSize ySize[] = {16, 65536};

    pTensor dx = std::make_shared<Tensor>(ARRAY_SIZE(xSize), xSize, syn_type_float);
    pTensor w  = std::make_shared<Tensor>(ARRAY_SIZE(wSize), wSize, syn_type_float);
    pTensor dy = std::make_shared<Tensor>(ARRAY_SIZE(ySize), ySize, syn_type_float);

    synConvolutionParams params{};
    pNode gemm = NodeFactory::createNode({dy, w}, {dx}, &params, NodeFactory::deDxNodeTypeName, "dedx");

    ReuseLimitCalculator rlc(*GaudiHalReader::instance(synDeviceGaudi), gemm);

    ASSERT_EQ(rlc.getLimit(), 160000);
}

TEST_F(RLCTest, rlc_should_return_8_mme_vector_sizes_for_huge_common_dim)
{
    TSize xSize[] = {8096, 4096};
    TSize wSize[] = {1400, 8096};
    TSize ySize[] = {1400, 4096};

    pTensor x = std::make_shared<Tensor>(ARRAY_SIZE(xSize), xSize, syn_type_bf16);
    pTensor w = std::make_shared<Tensor>(ARRAY_SIZE(wSize), wSize, syn_type_bf16);
    pTensor y = std::make_shared<Tensor>(ARRAY_SIZE(ySize), ySize, syn_type_bf16);

    synGEMMParams params{};
    pNode gemm = NodeFactory::createNode({x, w}, {y}, &params, NodeFactory::gemmNodeTypeName, "gemm");

    ReuseLimitCalculator rlc(*GaudiHalReader::instance(synDeviceGaudi), gemm);

    auto mmeVecSizeInElements =
        GaudiHalReader::instance(synDeviceGaudi)->getMmeVectorSize() / y->getElementSizeInBytes();
    ASSERT_EQ(rlc.getLimit(), 8 * mmeVecSizeInElements);
}

TEST_F(RLCTest, rlc_should_calculate_processing_time_to_traffic_time_ratio)
{
    // OPs: 256*256*16
    // Cycles: Ops/128^2
    // Processing time [usec]: Cycles/Freq
    double PrTime = (256. * 256. * 16. / (128. * 128.)) / GaudiHalReader::instance(synDeviceGaudi)->getClockFreqMHz();

    // Traffic: bytesize(x) + bytesize(w) + bytesize(y)
    // HBM BW: 1000*10^9 bytes/sec = 950 kbytes/usec (theoretical)
    // Traffic Time: Traffic / BW
    constexpr double TrTime = (16.*256. + 256.*16. + 256.*256.)*2./(950e3);

    TSize xSize[] = {16, 256};
    TSize wSize[] = {256, 16};
    TSize ySize[] = {256, 256};

    pTensor x = std::make_shared<Tensor>(ARRAY_SIZE(xSize), xSize, syn_type_bf16);
    pTensor w = std::make_shared<Tensor>(ARRAY_SIZE(wSize), wSize, syn_type_bf16);
    pTensor y = std::make_shared<Tensor>(ARRAY_SIZE(ySize), ySize, syn_type_bf16);

    synGEMMParams params{};
    pNode gemm = NodeFactory::createNode({x, w}, {y}, &params, NodeFactory::gemmNodeTypeName, "gemm");

    ReuseLimitCalculator rlc(*GaudiHalReader::instance(synDeviceGaudi), gemm);

    ASSERT_DOUBLE_EQ(rlc.getPrTrRatio(256, 256), PrTime / TrTime);
}

TEST_F(RLCTest, rlc_should_return_intermediate_value_for_medium_cd_size)
{
    TSize xSize[] = {256, 32768};
    TSize wSize[] = {32768, 256};
    TSize ySize[] = {32768, 32768};

    pTensor x = std::make_shared<Tensor>(ARRAY_SIZE(xSize), xSize, syn_type_bf16);
    pTensor w = std::make_shared<Tensor>(ARRAY_SIZE(wSize), wSize, syn_type_bf16);
    pTensor y = std::make_shared<Tensor>(ARRAY_SIZE(ySize), ySize, syn_type_bf16);

    synGEMMParams params{};
    pNode gemm = NodeFactory::createNode({x, w}, {y}, &params, NodeFactory::gemmNodeTypeName, "gemm");

    ReuseLimitCalculator rlc(*GaudiHalReader::instance(synDeviceGaudi), gemm);

    LOG_INFO(GO_TEST, "Limit: {}", rlc.getLimit());
    LOG_INFO(GO_TEST, "Ratio: {}", rlc.getPrTrRatio(rlc.getLimit(), rlc.getLimit()));

    auto mmeVecSizeInElements =
        GaudiHalReader::instance(synDeviceGaudi)->getMmeVectorSize() / y->getElementSizeInBytes();
    ASSERT_GT(rlc.getLimit(), 8 * mmeVecSizeInElements);
    ASSERT_LT(rlc.getLimit(), 32768);
}

TEST_F(RLCTest, rlc_should_return_intermediate_value_for_medium_cd_size_fwd_4d)
{
    TSize xSize[] = {256, 32768, 1, 1};
    TSize wSize[] = {32768, 256, 1, 1};
    TSize ySize[] = {32768, 32768, 1, 1};

    pTensor x = std::make_shared<Tensor>(ARRAY_SIZE(xSize), xSize, syn_type_bf16);
    pTensor w = std::make_shared<Tensor>(ARRAY_SIZE(wSize), wSize, syn_type_bf16);
    pTensor y = std::make_shared<Tensor>(ARRAY_SIZE(ySize), ySize, syn_type_bf16);

    synConvolutionParams params{};
    pNode conv = NodeFactory::createNode({x, w}, {y}, &params, NodeFactory::convolutionNodeTypeName, "conv");

    ReuseLimitCalculator rlc(*GaudiHalReader::instance(synDeviceGaudi), conv);

    LOG_INFO(GO_TEST, "Limit: {}", rlc.getLimit());
    LOG_INFO(GO_TEST, "Ratio: {}", rlc.getPrTrRatio(rlc.getLimit(), rlc.getLimit()));

    auto mmeVecSizeInElements =
        GaudiHalReader::instance(synDeviceGaudi)->getMmeVectorSize() / y->getElementSizeInBytes();
    ASSERT_GT(rlc.getLimit(), 8 * mmeVecSizeInElements);
    ASSERT_LT(rlc.getLimit(), 32768);
}

TEST_F(RLCTest, rlc_should_return_intermediate_value_for_medium_cd_size_dedx_4d)
{
    TSize xSize[] = {32768, 32768, 1, 1};
    TSize wSize[] = {256, 32768, 1, 1};
    TSize ySize[] = {256, 32768, 1, 1};

    pTensor dx = std::make_shared<Tensor>(ARRAY_SIZE(xSize), xSize, syn_type_bf16);
    pTensor w  = std::make_shared<Tensor>(ARRAY_SIZE(wSize), wSize, syn_type_bf16);
    pTensor dy = std::make_shared<Tensor>(ARRAY_SIZE(ySize), ySize, syn_type_bf16);

    synConvolutionParams params{};
    pNode dedx = NodeFactory::createNode({dy, w}, {dx}, &params, NodeFactory::deDxNodeTypeName, "dedx");

    ReuseLimitCalculator rlc(*GaudiHalReader::instance(synDeviceGaudi), dedx);

    LOG_INFO(GO_TEST, "Limit: {}", rlc.getLimit());
    LOG_INFO(GO_TEST, "Ratio: {}", rlc.getPrTrRatio(rlc.getLimit(), rlc.getLimit()));

    auto mmeVecSizeInElements =
        GaudiHalReader::instance(synDeviceGaudi)->getMmeVectorSize() / dx->getElementSizeInBytes();
    ASSERT_GT(rlc.getLimit(), 8 * mmeVecSizeInElements);
    ASSERT_LT(rlc.getLimit(), 32768);
}
