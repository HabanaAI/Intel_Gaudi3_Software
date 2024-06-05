#pragma once

#include "gaudi_tests/gc_gaudi_test_infra.h"
#include "gc_autogen_test.h"
#include "dynamic_shapes_types.h"
#include "syn_gaudi_two_run_compare_test.h"

class SynGaudiSRAMSlicingTest : public SynTrainingTwoRunCompareTest
{
public:

    void SetUpTest() override
    {
        SynTrainingTwoRunCompareTest::SetUpTest();

        synConfigurationGet("SRAM_SLICER_MAX_CAPACITY_BYTES", m_origSramCapStr, sizeof(m_origSramCapStr));
        synConfigurationGet("SRAM_SLICER_BUNDLE_EXPANSION_ENABLED",
                            m_origTpcSlicingEnabled,
                            sizeof(m_origTpcSlicingEnabled));

        synConfigurationSet("SRAM_SLICER_MAX_CAPACITY_BYTES", "4000000");
        synConfigurationSet("SRAM_SLICER_BUNDLE_EXPANSION_ENABLED", "True");
    }
    void TearDownTest() override
    {
        synConfigurationSet("SRAM_SLICER_MAX_CAPACITY_BYTES", m_origSramCapStr);
        synConfigurationSet("SRAM_SLICER_BUNDLE_EXPANSION_ENABLED", m_origTpcSlicingEnabled);
        SynTrainingTwoRunCompareTest::TearDownTest();
    }

    void testNonCDSlicing(unsigned hChunks, unsigned kChunks, unsigned inCD);
    void testReluConvSlicing(const char* sramCapStr = nullptr, bool checkIntermediateTensor = false);
    void testReluConvWithEvictionSlicing(const char* sramCapStr = nullptr);
    void testReluConvWithEvictionSlicingConsistency(const char* sramCapStr = nullptr);
    void testBigConvReluX5(unsigned batchSize);

protected:
    char m_origSramCapStr[128];
    char m_origTpcSlicingEnabled[128];
};

class SynGaudiSpatialSRAMSlicingTest : public SynGaudiSRAMSlicingTest,
                                       public testing::WithParamInterface<std::tuple<DimSizes, DimSizes, DimSizes, int, int, int, int, int, int, int>>
                                                          // batch, height, width, nIFM, nOFM, filter, stride, dilation, padBefore, padAfter
{
public:
    SynGaudiSpatialSRAMSlicingTest();

    void runSingleTest();
    virtual bool blockTestForConvParams();

    // Return Value: output tensor index
    virtual unsigned
                 addNode(synConvolutionParams& params, ShapeSizes& xSizes, ShapeSizes& ySizes, ShapeSizes& wSizes) = 0;
    virtual void setActualSizes() {}

protected:
    DimSizes m_batch;
    DimSizes m_xHeight;
    DimSizes m_xWidth;
    unsigned m_nIFM;
    unsigned m_nOFM;
    unsigned m_filter;
    unsigned m_stride;
    unsigned m_dilation;
    unsigned m_padBefore;
    unsigned m_padAfter;

    unsigned m_tensorXIdx;
    unsigned m_tensorYIdx;
    unsigned m_tensorWIdx;

    DimSizes m_yHeight;
    DimSizes m_yWidth;

    ShapeSizes m_xSizes;
    ShapeSizes m_wSizes;
    ShapeSizes m_ySizes;

    synConvolutionParams m_params;
};

class SynGaudiFwdConvSpatialSRAMSlicingTest : public SynGaudiSpatialSRAMSlicingTest
{
public:

    SynGaudiFwdConvSpatialSRAMSlicingTest() : SynGaudiSpatialSRAMSlicingTest() {}

    unsigned addNode(synConvolutionParams& params, ShapeSizes& xSizes, ShapeSizes& ySizes, ShapeSizes& wSizes) override;
};

class SynGaudiDedwSpatialSRAMSlicingTest : public SynGaudiSpatialSRAMSlicingTest
{
public:

    SynGaudiDedwSpatialSRAMSlicingTest() : SynGaudiSpatialSRAMSlicingTest() {}

    unsigned addNode(synConvolutionParams& params, ShapeSizes& xSizes, ShapeSizes& ySizes, ShapeSizes& wSizes) override;
};

class SynGaudiDedxSpatialSRAMSlicingTest : public SynGaudiSpatialSRAMSlicingTest
{
public:

    SynGaudiDedxSpatialSRAMSlicingTest() : SynGaudiSpatialSRAMSlicingTest() {}

    bool blockTestForConvParams() override;
    unsigned addNode(synConvolutionParams& params, ShapeSizes& xSizes, ShapeSizes& ySizes, ShapeSizes& wSizes) override;
};

class SynGaudiSRAMSlicingConsistency : public SynGaudiTestInfra
{
public:
    void testReluConvWithEvictionSlicingConsistency(const char* sramCapStr);
};