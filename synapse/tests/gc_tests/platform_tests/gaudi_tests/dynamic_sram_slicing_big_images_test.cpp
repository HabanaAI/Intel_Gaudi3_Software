#include "gaudi_tests/gc_gaudi_test_infra.h"
#include "sram_slicing_test.h"
#include "gc_dynamic_shapes_infra.h"

class SynGaudiFwdConvSpatialSRAMSlicingDynamicTest :
    public SynGaudiFwdConvSpatialSRAMSlicingTest
{
public:
    SynGaudiFwdConvSpatialSRAMSlicingDynamicTest()
    {
        setTestPackage(TEST_PACKAGE_DSD);
    }

    void setActualSizes() override;
};

void SynGaudiFwdConvSpatialSRAMSlicingDynamicTest::setActualSizes()
{
    SynTrainingTwoRunCompareTest::setActualSizes(m_tensorXIdx, m_xSizes.actual.data());
    SynTrainingTwoRunCompareTest::setActualSizes(m_tensorYIdx, m_ySizes.actual.data());

    ASSERT_FALSE(HasFailure());
}


class SynGaudiDedwSpatialSRAMSlicingDynamicTest : public SynGaudiDedwSpatialSRAMSlicingTest
{
public:
    SynGaudiDedwSpatialSRAMSlicingDynamicTest()
    {
        setTestPackage(TEST_PACKAGE_DSD);
    }

    void setActualSizes() override;
};

void SynGaudiDedwSpatialSRAMSlicingDynamicTest::setActualSizes()
{
    SynTrainingTwoRunCompareTest::setActualSizes(m_tensorXIdx, m_xSizes.actual.data());
    SynTrainingTwoRunCompareTest::setActualSizes(m_tensorYIdx, m_ySizes.actual.data());

    ASSERT_FALSE(HasFailure());
}

class SynGaudiDedxSpatialSRAMSlicingDynamicTest : public SynGaudiDedxSpatialSRAMSlicingTest
{
public:
    SynGaudiDedxSpatialSRAMSlicingDynamicTest()
    {
        setTestPackage(TEST_PACKAGE_DSD);
    }

    void     setActualSizes() override;
    unsigned addNode(synConvolutionParams& params, ShapeSizes& xSizes, ShapeSizes& ySizes, ShapeSizes& wSizes) override;

protected:
    unsigned m_tensorDxShapeIdx;
};

void SynGaudiDedxSpatialSRAMSlicingDynamicTest::setActualSizes()
{
    SynTrainingTwoRunCompareTest::setActualSizes(m_tensorXIdx, m_xSizes.actual.data());
    SynTrainingTwoRunCompareTest::setActualSizes(m_tensorYIdx, m_ySizes.actual.data());
    SynTrainingTwoRunCompareTest::setActualSizes(m_tensorDxShapeIdx, m_xSizes.actual.data());

    ASSERT_FALSE(HasFailure());
}

unsigned SynGaudiDedxSpatialSRAMSlicingDynamicTest::addNode(synConvolutionParams& params,
                                                            ShapeSizes&           xSizes,
                                                            ShapeSizes&           ySizes,
                                                            ShapeSizes&           wSizes)
{
    m_tensorYIdx = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, ySizes.max.data(),
                                       ySizes.actual.size(), syn_type_float, nullptr, "dY", 0, 0, nullptr, ySizes.min.data());
    m_tensorWIdx = createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, wSizes.actual.data(),
                                       wSizes.actual.size(), syn_type_float, nullptr, "W");
    m_tensorXIdx = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, xSizes.max.data(),
                                       xSizes.max.size(), syn_type_float, nullptr, "dX", 0, 0, nullptr, xSizes.min.data());
    m_tensorDxShapeIdx = createShapeTensor(INPUT_TENSOR, xSizes.max.data(), xSizes.min.data(), xSizes.max.size(), syn_type_single, "dXshape", 0);

    addNodeToGraph(NodeFactory::deDxNodeTypeName, {m_tensorYIdx, m_tensorWIdx, m_tensorDxShapeIdx}, {m_tensorXIdx}, &params, sizeof(params), "dEdX");

    return m_tensorXIdx;
}

INSTANTIATE_TEST_SUITE_P(dynamic_big_image_fwd_conv_single,
                        SynGaudiFwdConvSpatialSRAMSlicingDynamicTest,
                        // batch, height, width, nIFM, nOFM, filter, stride, dilation, padBefore, padAfter
                        ::testing::Values(std::make_tuple(DimSizes(1), DimSizes(64, 128, 64), DimSizes(128), 128, 8, 3, 2, 1, 0, 0),
                                          std::make_tuple(DimSizes(1), DimSizes(64, 128, 96), DimSizes(128), 128, 8, 3, 2, 1, 0, 0),
                                          std::make_tuple(DimSizes(1), DimSizes(64, 128, 128), DimSizes(128), 128, 8, 3, 2, 1, 0, 0),
                                          std::make_tuple(DimSizes(1), DimSizes(64, 128, 64), DimSizes(128), 128, 8, 3, 1, 1, 0, 0),
                                          std::make_tuple(DimSizes(1), DimSizes(64, 128, 96), DimSizes(128), 128, 8, 3, 1, 1, 0, 0),
                                          std::make_tuple(DimSizes(1), DimSizes(64, 128, 128), DimSizes(128), 128, 8, 3, 1, 1, 0, 0),
                                          std::make_tuple(DimSizes(1), DimSizes(64, 128, 64), DimSizes(128), 128, 8, 4, 3, 2, 2, 0),
                                          std::make_tuple(DimSizes(1), DimSizes(64, 128, 96), DimSizes(128), 128, 8, 4, 3, 2, 2, 0),
                                          std::make_tuple(DimSizes(1), DimSizes(64, 128, 128), DimSizes(128), 128, 8, 4, 3, 2, 2, 0),
                                          std::make_tuple(DimSizes(1), DimSizes(64, 128, 64), DimSizes(128), 128, 8, 3, 2, 2, 2, 3),
                                          std::make_tuple(DimSizes(1), DimSizes(64, 128, 96), DimSizes(128), 128, 8, 3, 2, 2, 2, 3),
                                          std::make_tuple(DimSizes(1), DimSizes(64, 128, 128), DimSizes(128), 128, 8, 3, 2, 2, 2, 3)));

INSTANTIATE_TEST_SUITE_P(
    dynamic_big_image_dedw_single,
    SynGaudiDedwSpatialSRAMSlicingDynamicTest,
    ::testing::Values(std::make_tuple(DimSizes(1), DimSizes(64, 128, 64), DimSizes(128), 256, 8, 3, 2, 1, 0, 0),
                      std::make_tuple(DimSizes(1), DimSizes(64, 128, 97), DimSizes(128), 256, 8, 3, 2, 1, 0, 0),
                      std::make_tuple(DimSizes(1), DimSizes(64, 128, 128), DimSizes(128), 256, 8, 3, 2, 1, 0, 0),
                      std::make_tuple(DimSizes(1), DimSizes(64, 128, 64), DimSizes(128), 256, 8, 3, 1, 1, 0, 0),
                      std::make_tuple(DimSizes(1), DimSizes(64, 128, 96), DimSizes(128), 256, 8, 3, 1, 1, 0, 0),
                      std::make_tuple(DimSizes(1), DimSizes(64, 128, 128), DimSizes(128), 256, 8, 3, 1, 1, 0, 0),
                      std::make_tuple(DimSizes(1), DimSizes(64, 128, 64), DimSizes(128), 256, 8, 4, 3, 2, 2, 0),
                      std::make_tuple(DimSizes(1), DimSizes(64, 128, 96), DimSizes(128), 256, 8, 4, 3, 2, 2, 0),
                      std::make_tuple(DimSizes(1), DimSizes(64, 128, 128), DimSizes(128), 256, 8, 4, 3, 2, 2, 0),
                      std::make_tuple(DimSizes(1), DimSizes(5, 10, 5), DimSizes(256), 2000, 8, 3, 1, 1, 0, 0),
                      std::make_tuple(DimSizes(1), DimSizes(5, 10, 7), DimSizes(128), 2000, 8, 3, 1, 1, 0, 0),
                      std::make_tuple(DimSizes(1), DimSizes(5, 10, 10), DimSizes(256), 2000, 8, 3, 1, 1, 0, 0),
                      std::make_tuple(DimSizes(1), DimSizes(64, 128, 64), DimSizes(64), 256, 8, 3, 2, 2, 2, 3),
                      std::make_tuple(DimSizes(1), DimSizes(64, 128, 96), DimSizes(64), 256, 8, 3, 2, 2, 2, 3),
                      std::make_tuple(DimSizes(1), DimSizes(64, 128, 128), DimSizes(64), 256, 8, 3, 2, 2, 2, 3),
                      // b,h,w are sliced
                      std::make_tuple(DimSizes(2), DimSizes(2, 6, 6), DimSizes(8192), 32, 32, 3, 1, 1, 0, 0),
                      // b,h,w,non-common are sliced
                      std::make_tuple(DimSizes(2), DimSizes(2, 10, 7), DimSizes(16384), 512, 16, 3, 2, 1, 0, 0)));

INSTANTIATE_TEST_SUITE_P(dynamic_big_image_dedx_single,
                        SynGaudiDedxSpatialSRAMSlicingDynamicTest,
                        ::testing::Values(std::make_tuple(DimSizes(1), DimSizes(64, 128, 64), DimSizes(128), 256, 8, 3, 1, 1, 0, 0),
                                          std::make_tuple(DimSizes(1), DimSizes(64, 128, 96), DimSizes(128), 256, 8, 3, 1, 1, 0, 0),
                                          std::make_tuple(DimSizes(1), DimSizes(64, 128, 128), DimSizes(128), 256, 8, 3, 1, 1, 0, 0),
                                          std::make_tuple(DimSizes(1), DimSizes(64, 128, 64), DimSizes(128), 256, 32, 3, 2, 1, 0, 0),
                                          std::make_tuple(DimSizes(1), DimSizes(64, 128, 96), DimSizes(128), 256, 32, 3, 2, 1, 0, 0),
                                          std::make_tuple(DimSizes(1), DimSizes(64, 128, 128), DimSizes(128), 256, 32, 3, 2, 1, 0, 0),
                                          std::make_tuple(DimSizes(1), DimSizes(128, 256, 128), DimSizes(256), 256, 40, 4, 3, 2, 2, 0),
                                          std::make_tuple(DimSizes(1), DimSizes(128, 256, 192), DimSizes(256), 256, 40, 4, 3, 2, 2, 0),
                                          std::make_tuple(DimSizes(1), DimSizes(128, 256, 256), DimSizes(256), 256, 40, 4, 3, 2, 2, 0),
                                          std::make_tuple(DimSizes(1), DimSizes(128, 256, 128), DimSizes(256), 256, 16, 3, 2, 3, 0, 0),
                                          std::make_tuple(DimSizes(1), DimSizes(128, 256, 192), DimSizes(256), 256, 16, 3, 2, 3, 0, 0),
                                          std::make_tuple(DimSizes(1), DimSizes(128, 256, 256), DimSizes(256), 256, 16, 3, 2, 3, 0, 0),
                                          std::make_tuple(DimSizes(1), DimSizes(128, 256, 128), DimSizes(256), 256, 40, 3, 3, 4, 2, 0),
                                          std::make_tuple(DimSizes(1), DimSizes(128, 256, 192), DimSizes(256), 256, 40, 3, 3, 4, 2, 0),
                                          std::make_tuple(DimSizes(1), DimSizes(128, 256, 256), DimSizes(256), 256, 40, 3, 3, 4, 2, 0)));

TEST_P_GC(SynGaudiFwdConvSpatialSRAMSlicingDynamicTest, dynamic_big_image_fwd_conv_ASIC_CI)
{
    runSingleTest();
}

TEST_P_GC(SynGaudiDedwSpatialSRAMSlicingDynamicTest, dynamic_big_image_dedw_ASIC_CI)
{
    runSingleTest();
}

TEST_P_GC(SynGaudiDedxSpatialSRAMSlicingDynamicTest, dynamic_big_image_dedx_ASIC_CI)
{
    runSingleTest();
}
