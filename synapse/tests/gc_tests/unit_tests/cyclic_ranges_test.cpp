#include "defs.h"
#include "gaudi_graph.h"
#include "graph_optimizer_test.h"
#include "habana_pass.h"
#include "synapse_common_types.h"
#include "include/sync/data_range.h"
#include "passes/calculate_tensor_roi_linear_ranges.h"
#include "node_factory.h"
#include "tensor.h"
#include "types.h"
#include "gtest/gtest-param-test.h"
#include "gtest/gtest.h"
#include "gtest/internal/gtest-param-util.h"
#include "compilation_hal_reader.h"
#include "hal_reader/gaudi1/hal_reader.h"
#include <algorithm>
#include <fstream>
#include <limits>
#include <sstream>
#include <vector>
#include <numeric>

class CyclicRangeTest : public GraphOptimizerTest
{
protected:
    static std::string toStr(const CyclicDataRange& cr)
    {
        std::stringstream ss;
        ss << "{" << cr.start() << ", " << cr.end() << ", " << cr.stride() << "}";
        return ss.str();
    }

    template<class T>
    static std::string toStr(const DataRange<T>& r)
    {
        std::stringstream ss;
        ss << "[" << r.start() << ", " << r.end() << "]";
        return ss.str();
    }

    static bool isOverlap(const std::vector<DataRange<uint64_t>>& a, const std::vector<DataRange<uint64_t>>& b)
    {
        int ib = 0;
        for (const auto& ra : a)
        {
            for (; ib < b.size(); ib++)
            {
                const auto& rb = b[ib];
                if (ra.isOverlap(rb)) return true;
                if (ra.end() <= rb.start()) break;
            }
        }
        return false;
    }
};

TEST_F(CyclicRangeTest, test1)
{
    CyclicDataRange r1(0, 128, 256);
    CyclicDataRange r2 = r1;

    EXPECT_TRUE(r1.isOverlap(r2));
    EXPECT_TRUE(r2.isOverlap(r1));
    r2.shift(64);
    EXPECT_TRUE(r1.isOverlap(r2));
    EXPECT_TRUE(r2.isOverlap(r1));
    r2.shift(64);
    EXPECT_TRUE(!r1.isOverlap(r2));
    EXPECT_TRUE(!r2.isOverlap(r1));
}

TEST_F(CyclicRangeTest, toLinear)
{
    CyclicDataRange r1(0, 128, 256);

    std::vector<DataRange<uint64_t>> linearRanges = CalculateTensorROIsLinearRanges::toLinearRanges(r1, 0, 256);
    EXPECT_EQ(linearRanges.size(), 1);
    EXPECT_EQ(linearRanges[0].start(), 0);
    EXPECT_EQ(linearRanges[0].end(), 128);

    linearRanges = CalculateTensorROIsLinearRanges::toLinearRanges(r1, 10, 512 + 10);
    EXPECT_EQ(linearRanges.size(), 3);
    EXPECT_EQ(linearRanges[0].start(), 10);
    EXPECT_EQ(linearRanges[0].end(), 128);
    EXPECT_EQ(linearRanges[1].start(), 256);
    EXPECT_EQ(linearRanges[1].end(), 256 + 128);
    EXPECT_EQ(linearRanges[2].start(), 512);
    EXPECT_EQ(linearRanges[2].end(), 512 + 10);

    CyclicDataRange r2(128, 256, 256);
    linearRanges = CalculateTensorROIsLinearRanges::toLinearRanges(r2, 0, 256);
    EXPECT_EQ(linearRanges.size(), 1);
    EXPECT_EQ(linearRanges[0].start(), 128);
    EXPECT_EQ(linearRanges[0].end(), 256);

    linearRanges = CalculateTensorROIsLinearRanges::toLinearRanges(r2, 10, 512 + 10);
    EXPECT_EQ(linearRanges.size(), 2);
    EXPECT_EQ(linearRanges[0].start(), 128);
    EXPECT_EQ(linearRanges[0].end(), 256);
    EXPECT_EQ(linearRanges[1].start(), 256 + 128);
    EXPECT_EQ(linearRanges[1].end(), 512);

    CyclicDataRange r3(64, 192, 256);
    linearRanges = CalculateTensorROIsLinearRanges::toLinearRanges(r3, 0, 256);
    EXPECT_EQ(linearRanges.size(), 1);
    EXPECT_EQ(linearRanges[0].start(), 64);
    EXPECT_EQ(linearRanges[0].end(), 192);

    linearRanges = CalculateTensorROIsLinearRanges::toLinearRanges(r3, 10, 512 + 10);
    EXPECT_EQ(linearRanges.size(), 2);
    EXPECT_EQ(linearRanges[0].start(), 64);
    EXPECT_EQ(linearRanges[0].end(), 192);
    EXPECT_EQ(linearRanges[1].start(), 256 + 64);
    EXPECT_EQ(linearRanges[1].end(), 256 + 192);

    CyclicDataRange r4(192, 256 + 64, 256);
    linearRanges = CalculateTensorROIsLinearRanges::toLinearRanges(r4, 0, 256);
    EXPECT_EQ(linearRanges.size(), 2);
    EXPECT_EQ(linearRanges[0].start(), 0);
    EXPECT_EQ(linearRanges[0].end(), 64);
    EXPECT_EQ(linearRanges[1].start(), 192);
    EXPECT_EQ(linearRanges[1].end(), 256);

    linearRanges = CalculateTensorROIsLinearRanges::toLinearRanges(r4, 10, 512 + 10);
    EXPECT_EQ(linearRanges.size(), 3);
    EXPECT_EQ(linearRanges[0].start(), 10);
    EXPECT_EQ(linearRanges[0].end(), 64);
    EXPECT_EQ(linearRanges[1].start(), 192);
    EXPECT_EQ(linearRanges[1].end(), 256 + 64);
    EXPECT_EQ(linearRanges[2].start(), 256 + 192);
    EXPECT_EQ(linearRanges[2].end(), 512 + 10);

    CyclicDataRange r5(128, 256, 512);
    EXPECT_EQ(CalculateTensorROIsLinearRanges::toLinearRanges(r5, 100, 120).size(), 0);
    linearRanges = CalculateTensorROIsLinearRanges::toLinearRanges(r5, 100, 200);
    EXPECT_EQ(linearRanges.size(), 1);
    EXPECT_EQ(linearRanges[0].start(), 128);
    EXPECT_EQ(linearRanges[0].end(), 200);
    linearRanges = CalculateTensorROIsLinearRanges::toLinearRanges(r5, 100, 300);
    EXPECT_EQ(linearRanges.size(), 1);
    EXPECT_EQ(linearRanges[0].start(), 128);
    EXPECT_EQ(linearRanges[0].end(), 256);
    linearRanges = CalculateTensorROIsLinearRanges::toLinearRanges(r5, 200, 300);
    EXPECT_EQ(linearRanges.size(), 1);
    EXPECT_EQ(linearRanges[0].start(), 200);
    EXPECT_EQ(linearRanges[0].end(), 256);
    EXPECT_EQ(CalculateTensorROIsLinearRanges::toLinearRanges(r5, 300, 400).size(), 0);
}

TEST_F(CyclicRangeTest, isOverlap)
{
    CyclicDataRange r1(0, 128, 512);
    CyclicDataRange r2(128, 256, 256);
    EXPECT_FALSE(r1.isOverlap(r2, 0, 2048));

    r1 = CyclicDataRange(0, 200, 512);
    r2 = CyclicDataRange(128, 256, 256);
    EXPECT_TRUE(r1.isOverlap(r2, 200, 2048));

    r1 = CyclicDataRange(0, 64, 260);
    r2 = CyclicDataRange(128, 256, 256);
    EXPECT_FALSE(r1.isOverlap(r2, 0, 1024));

    r1 = CyclicDataRange(0, 64, 260);
    r2 = CyclicDataRange(128, 256, 256);
    EXPECT_TRUE(r1.isOverlap(r2, 0, 5000));

    EXPECT_TRUE(r2.isOverlap(256, 256 + 128 + 1));
    EXPECT_FALSE(r2.isOverlap(256, 256 + 128));

    r1 = CyclicDataRange(32, 100, 1024);
    EXPECT_FALSE(r1.isOverlap(r2, 1000, 10000000));
    EXPECT_FALSE(r1.isOverlap(r2, 1000, 1300));

    r1 = CyclicDataRange(100, 150, 1024);
    EXPECT_FALSE(r1.isOverlap(r2, 300, 384));
    EXPECT_TRUE(r1.isOverlap(r2, 300, 2000000));
    EXPECT_TRUE(r1.isOverlap(r2, 1100, 2000));

    r1 = CyclicDataRange(100, 200, 1024);
    r2 = CyclicDataRange(180, 800, 1024);
    EXPECT_FALSE(r1.isOverlap(r2, 1300, 1500));
    EXPECT_TRUE(r1.isOverlap(r2, 1200, 1500));
    EXPECT_TRUE(r1.isOverlap(r2, 1300, 10000000));

    r1 = CyclicDataRange(800, 1024 + 100, 1024);
    r2 = CyclicDataRange(700, 1024 + 150, 1024);
    EXPECT_FALSE(r1.isOverlap(r2, 2148, 2800));
    EXPECT_TRUE(r1.isOverlap(r2, 2148, 2900));
    EXPECT_TRUE(r1.isOverlap(r2, 1300, 10000000));

    r1 = CyclicDataRange(223, 325, 1024);
    r2 = CyclicDataRange(507, 512 + 384, 512);
    EXPECT_TRUE(r1.isOverlap(r2, 0, 1024));
}

class CyclicRangeOverlapTest
: public CyclicRangeTest
, public testing::WithParamInterface<std::tuple<int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int64_t, int64_t>>
{
};

INSTANTIATE_TEST_SUITE_P(CyclicRangeTest,
                         CyclicRangeOverlapTest,
                         ::testing::Combine(::testing::ValuesIn({0, -1, -1}),
                                            ::testing::ValuesIn({0, -1, -1}),
                                            ::testing::ValuesIn({1, 128, -1}),
                                            ::testing::ValuesIn({1, 256, -1}),
                                            ::testing::ValuesIn({32, 256, 1024, -1}),
                                            ::testing::ValuesIn({512, 1024, -1}),
                                            ::testing::ValuesIn({int64_t(0), int64_t(-1)}),
                                            ::testing::ValuesIn({int64_t(1), int64_t(128), int64_t(-1)})));

TEST_P(CyclicRangeOverlapTest, isOverlapParametrized)
{
    int32_t startA, startB, sizeA, sizeB, strideA, strideB;
    int64_t start, size;
    std::tie(startA, startB, sizeA, sizeB, strideA, strideB, start, size) = GetParam();

    startA  = (startA < 0) ? rand() % 1000 : startA;
    startB  = (startB < 0) ? rand() % 1000 : startB;
    sizeA   = (sizeA < 0) ? rand() % 500 + 1 : sizeA;
    sizeB   = (sizeB < 0) ? rand() % 500 + 1 : sizeB;
    strideA = (strideA < 0) ? rand() % 3000 + 1 : strideA;
    strideB = (strideB < 0) ? rand() % 3000 + 1 : strideB;
    strideA = std::max(strideA, sizeA);
    strideB = std::max(strideB, sizeB);
    start   = (start < 0) ? rand() % 10000 : start;
    size    = (size < 0) ? rand() % 1000 + 1 : size;

    int64_t end = start + size;

    CyclicDataRange a(startA, startA + sizeA, strideA);
    CyclicDataRange b(startB, startB + sizeB, strideB);

    bool res     = a.isOverlap(b, start, end);
    bool realRes = isOverlap(CalculateTensorROIsLinearRanges::toLinearRanges(a, start, end),
                             CalculateTensorROIsLinearRanges::toLinearRanges(b, start, end));

    EXPECT_EQ(res, realRes) << "crA: " << toStr(a) << " crB: " << toStr(b) << " in: [" << start << ", " << end << "]";
}

class ConvertToCyclicTest : public CyclicRangeTest
{
protected:
    static std::tuple<TensorROI, NodePtr>
    getTensorRoiAndNode(TSize* sizes, TStride* strides, uint32_t dim, uint64_t offset = 0)
    {
        GaudiGraph g;
        TensorPtr  in1(new Tensor(dim, sizes, syn_type_single));
        TensorPtr  in2(new Tensor(dim, sizes, syn_type_single));
        in1->reshape(dim, sizes, strides);
        TensorPtr out(new Tensor(dim, sizes, syn_type_single));
        NodePtr   n = NodeFactory::createNode({in1, in2}, {out}, nullptr, "add_f32", "add");
        GraphEditor::addNode(g, n);
        dynamic_cast<TPCNode*>(n.get())->init(tpc_lib_api::DEVICE_ID_GAUDI, nullptr, g.getNextTPCKernelUniqueId());
        g.GetNodeROIs(n)->push_back(n->generateRoi());
        EXPECT_TRUE(projectNodeROIs(g));

        TensorROI& tRoi1              = g.GetNodeROIs(n)->back().inputRois.front();
        tRoi1.getLayout().baseAddress = offset;

        return {tRoi1, n};
    }

    static TensorROI calcTensorRoiLegacy(TSize* sizes, TStride* strides, uint32_t dim, uint64_t offset = 0)
    {
        auto [tRoi1, n] = getTensorRoiAndNode(sizes, strides, dim, offset);
        CalculateTensorROIsLinearRanges::calculateLinearRangesLegacy(tRoi1, n, true);
        return tRoi1;
    }

    static void checkCyclicToLinearRanges(const std::vector<DataRange<uint64_t>>& linearRanges,
                                          const std::vector<DataRange<uint64_t>>& cyclicRangeBounds,
                                          const std::vector<CyclicDataRange>&     cyclicRanges)
    {
        std::vector<DataRange<uint64_t>> originalRanges = linearRanges;
        std::vector<DataRange<uint64_t>> convertedFromCyclicRanges;
        for (int i = 0; i < cyclicRangeBounds.size(); i++)
        {
            auto newRanges = CalculateTensorROIsLinearRanges::toLinearRanges(cyclicRanges[i],
                                                                             cyclicRangeBounds[i].start(),
                                                                             cyclicRangeBounds[i].end());
            convertedFromCyclicRanges.reserve(newRanges.size());
            convertedFromCyclicRanges.insert(convertedFromCyclicRanges.end(), newRanges.begin(), newRanges.end());
        }
        auto pred = [](const DataRange<uint64_t>& r1, const DataRange<uint64_t>& r2) {
            return (r1.start() == r2.start()) ? r1.end() < r2.end() : r1.start() < r2.start();
        };
        std::sort(convertedFromCyclicRanges.begin(), convertedFromCyclicRanges.end(), pred);
        std::sort(originalRanges.begin(), originalRanges.end(), pred);

        // Merge togehter overlapping original ranges
        std::vector<DataRange<uint64_t>> mergedOriginalRanges;
        uint64_t                         currStart = originalRanges[0].start();
        for (int i = 1; i < originalRanges.size(); ++i)
        {
            if (originalRanges[i - 1].end() < originalRanges[i].start())
            {
                mergedOriginalRanges.emplace_back(currStart, originalRanges[i - 1].end());
                currStart = originalRanges[i].start();
            }
        }
        mergedOriginalRanges.emplace_back(currStart, originalRanges.back().end());

        EXPECT_EQ(convertedFromCyclicRanges, mergedOriginalRanges);
    }
};

TEST_F(ConvertToCyclicTest, conversion_simple)
{
    uint32_t elemSize = 4;
    TSize    sizes[3] = {2, 64, 64};
    TStride  strides[4];
    strides[0] = elemSize;
    strides[1] = strides[0] * sizes[0] * 2;
    strides[2] = strides[1] * sizes[1];
    strides[3] = strides[2] * sizes[2];
    CompilationHalReader::setHalReader(GaudiHalReader::instance(synDeviceGaudi));
    TensorROI tRoi = calcTensorRoiLegacy(sizes, strides, 3);

    EXPECT_EQ(tRoi.m_overlapRoi.subRois->size(), 1);
    EXPECT_EQ(tRoi.m_overlapRoi.subRois->front().ranges.size(), sizes[1] * sizes[2]);

    std::vector<DataRange<uint64_t>> originalRanges = tRoi.m_overlapRoi.subRois->front().ranges;
    CalculateTensorROIsLinearRanges::calculateCyclicRangesFromLinearRanges(tRoi);
    EXPECT_EQ(tRoi.m_overlapRoi.subRois->size(), 1);
    const std::vector<CyclicDataRange>& cyclicRanges = tRoi.m_overlapRoi.subRois->front().cyclicRanges;
    EXPECT_EQ(cyclicRanges.size(), 1);
    EXPECT_EQ(tRoi.m_overlapRoi.subRois->front().ranges.size(), 1);
    EXPECT_EQ(tRoi.m_overlapRoi.subRois->front().ranges.front(),
              DataRange<uint64_t>(0, sizes[0] * sizes[1] * sizes[2] * elemSize * 2 - elemSize * sizes[0]));
    EXPECT_EQ(cyclicRanges.front(), CyclicDataRange(0, elemSize * sizes[0], elemSize * sizes[0] * 2));
    std::vector<DataRange<uint64_t>> ranges =
        CalculateTensorROIsLinearRanges::toLinearRanges(cyclicRanges.front(),
                                                        0,
                                                        sizes[0] * sizes[1] * sizes[2] * elemSize * 2);
    EXPECT_EQ(ranges, originalRanges);
}

TEST_F(ConvertToCyclicTest, conversion_offset)
{
    uint32_t elemSize   = 4;
    TSize    sizes[3]   = {2, 64, 64};
    TStride  strides[4] = {1};
    strides[0]          = elemSize;
    strides[1]          = strides[0] * sizes[0] * 2;
    strides[2]          = strides[1] * sizes[1];
    strides[3]          = strides[2] * sizes[2];
    CompilationHalReader::setHalReader((GaudiHalReader::instance(synDeviceGaudi)));

    TensorROI tRoi = calcTensorRoiLegacy(sizes, strides, 3, 125);

    std::vector<DataRange<uint64_t>> originalRanges = tRoi.m_overlapRoi.subRois->front().ranges;
    CalculateTensorROIsLinearRanges::calculateCyclicRangesFromLinearRanges(tRoi);
    const std::vector<CyclicDataRange>&     cyclicRanges      = tRoi.m_overlapRoi.subRois->front().cyclicRanges;
    const std::vector<DataRange<uint64_t>>& cyclicRangeBounds = tRoi.m_overlapRoi.subRois->front().ranges;
    checkCyclicToLinearRanges(originalRanges, cyclicRangeBounds, cyclicRanges);
}

TEST_F(ConvertToCyclicTest, conversion_double_stride)
{
    uint32_t elemSize   = 4;
    TSize    sizes[3]   = {2, 64, 64};
    TStride  strides[4] = {1};
    strides[0]          = elemSize;
    strides[1]          = strides[0] * sizes[0] * 2;
    strides[2]          = strides[1] * sizes[1] * 3;
    strides[3]          = strides[2] * sizes[2];
    CompilationHalReader::setHalReader((GaudiHalReader::instance(synDeviceGaudi)));

    TensorROI tRoi = calcTensorRoiLegacy(sizes, strides, 3, 0);

    std::vector<DataRange<uint64_t>> originalRanges = tRoi.m_overlapRoi.subRois->front().ranges;
    CalculateTensorROIsLinearRanges::calculateCyclicRangesFromLinearRanges(tRoi);
    const std::vector<CyclicDataRange>&     cyclicRanges      = tRoi.m_overlapRoi.subRois->front().cyclicRanges;
    const std::vector<DataRange<uint64_t>>& cyclicRangeBounds = tRoi.m_overlapRoi.subRois->front().ranges;
    checkCyclicToLinearRanges(originalRanges, cyclicRangeBounds, cyclicRanges);
}

TEST_F(ConvertToCyclicTest, conversion_double_stride_offset)
{
    uint32_t elemSize   = 4;
    TSize    sizes[3]   = {2, 64, 64};
    TStride  strides[4] = {1};
    strides[0]          = elemSize;
    strides[1]          = strides[0] * sizes[0] * 2;
    strides[2]          = strides[1] * sizes[1] * 3;
    strides[3]          = strides[2] * sizes[2];
    CompilationHalReader::setHalReader((GaudiHalReader::instance(synDeviceGaudi)));

    TensorROI tRoi = calcTensorRoiLegacy(sizes, strides, 3, 125);

    std::vector<DataRange<uint64_t>> originalRanges = tRoi.m_overlapRoi.subRois->front().ranges;
    CalculateTensorROIsLinearRanges::calculateCyclicRangesFromLinearRanges(tRoi);
    const std::vector<CyclicDataRange>&     cyclicRanges      = tRoi.m_overlapRoi.subRois->front().cyclicRanges;
    const std::vector<DataRange<uint64_t>>& cyclicRangeBounds = tRoi.m_overlapRoi.subRois->front().ranges;
    checkCyclicToLinearRanges(originalRanges, cyclicRangeBounds, cyclicRanges);
}

class CompareNewLegacyMemoryRanges : public ConvertToCyclicTest
{
private:
    static constexpr int MIN_RANGES_FOR_CYCLIC = 8;
    void SetUp() override
    {
        ConvertToCyclicTest::SetUp();
        setGlobalConfForTest(GCFG_ENABLE_DIRECT_CYCLIC_RANGE_CALC, "true");
    }

protected:
    static TensorROI calcTensorRoiIndirectly(TSize* sizes, TStride* strides, uint32_t dim, uint64_t offset = 0)
    {
        auto [tRoi1, n] = getTensorRoiAndNode(sizes, strides, dim, offset);
        CalculateTensorROIsLinearRanges::calculateLinearRangesLegacy(tRoi1, n, true);
        CalculateTensorROIsLinearRanges::calculateCyclicRangesFromLinearRanges(tRoi1);
        return tRoi1;
    }

    static TensorROI calcTensorRoiDirectly(TSize* sizes, TStride* strides, uint32_t dim, uint64_t offset = 0)
    {
        auto [tRoi1, n] = getTensorRoiAndNode(sizes, strides, dim, offset);
        CalculateTensorROIsLinearRanges::calculateMemoryRanges(tRoi1, n, true);
        return tRoi1;
    }

    static void checkNewAndOldMemoryRangesCollide(const TensorROI& directTRoi, const TensorROI& indirectTRoi)
    {
        if (isCyclicConvert(directTRoi))  // Direct calculation is cyclic
        {
            if (isCyclicConvert(indirectTRoi))  // Both direct and indirect calculations are cyclic
            {
                const auto& directCyclicRangeBounds   = directTRoi.m_overlapRoi.subRois->front().ranges;
                const auto& indirectCyclicRangeBounds = indirectTRoi.m_overlapRoi.subRois->front().ranges;
                EXPECT_EQ(directCyclicRangeBounds, indirectCyclicRangeBounds);

                const auto& directCyclicRanges   = directTRoi.m_overlapRoi.subRois->front().cyclicRanges;
                const auto& indirectCyclicRanges = indirectTRoi.m_overlapRoi.subRois->front().cyclicRanges;
                EXPECT_EQ(directCyclicRanges, indirectCyclicRanges);
            }
            else  // Indirect calculation is not cyclic
            {
                const auto& indirectOriginalRanges  = indirectTRoi.m_overlapRoi.subRois->front().ranges;
                const auto& directCyclicRangeBounds = directTRoi.m_overlapRoi.subRois->front().ranges;
                const auto& direcetCyclicRanges     = directTRoi.m_overlapRoi.subRois->front().cyclicRanges;
                checkCyclicToLinearRanges(indirectOriginalRanges, directCyclicRangeBounds, direcetCyclicRanges);
            }
        }
        else  // Direct calculation is not cyclic
        {
            if (!isCyclicConvert(indirectTRoi))  // Both direct and indirect calculations are not cyclic
            {
                const auto& directRanges   = directTRoi.m_overlapRoi.subRois->front().ranges;
                const auto& indirectRanges = indirectTRoi.m_overlapRoi.subRois->front().ranges;

                // Merge togehter overlapping indirect linear ranges
                std::vector<DataRange<uint64_t>> mergedIndirectRanges;
                uint64_t                         currStart = indirectRanges[0].start();
                for (int i = 1; i < indirectRanges.size(); ++i)
                {
                    if (indirectRanges[i - 1].end() < indirectRanges[i].start())
                    {
                        mergedIndirectRanges.emplace_back(currStart, indirectRanges[i - 1].end());
                        currStart = indirectRanges[i].start();
                    }
                }
                mergedIndirectRanges.emplace_back(currStart, indirectRanges.back().end());

                EXPECT_EQ(directRanges, mergedIndirectRanges);
            }
            else  // Indirect is cyclic but direct is linear
            {
                const auto& directOriginalRanges      = directTRoi.m_overlapRoi.subRois->front().ranges;
                const auto& indirectCyclicRangeBounds = indirectTRoi.m_overlapRoi.subRois->front().ranges;
                const auto& indirecetCyclicRanges     = indirectTRoi.m_overlapRoi.subRois->front().cyclicRanges;
                EXPECT_GE(indirecetCyclicRanges.size(), directOriginalRanges.size() / MIN_RANGES_FOR_CYCLIC);
                checkCyclicToLinearRanges(directOriginalRanges, indirectCyclicRangeBounds, indirecetCyclicRanges);
            }
        }
    }

private:
    static bool isCyclicConvert(const TensorROI& tRoi)
    {
        return !(tRoi.m_overlapRoi.subRois->front().cyclicRanges.empty() &&
                 !tRoi.m_overlapRoi.subRois->front().ranges.empty());
    }
};

TEST_F(CompareNewLegacyMemoryRanges, basic_check)
{
    uint32_t elemSize   = 4;
    TSize    sizes[4]   = {3, 3, 3, 2};
    TStride  strides[5] = {1};
    strides[0]          = elemSize;
    strides[1]          = strides[0] * sizes[0] * 2;
    strides[2]          = strides[1] * sizes[1];
    strides[3]          = strides[2] * sizes[2] * 2;
    strides[4]          = strides[3] * sizes[3];
    CompilationHalReader::setHalReader((GaudiHalReader::instance(synDeviceGaudi)));

    TensorROI directTRoi   = calcTensorRoiDirectly(sizes, strides, 4, 0);
    TensorROI indirectTRoi = calcTensorRoiIndirectly(sizes, strides, 4, 0);

    checkNewAndOldMemoryRangesCollide(directTRoi, indirectTRoi);
}

TEST_F(CompareNewLegacyMemoryRanges, basic_check2)
{
    uint32_t elemSize   = 4;
    TSize sizes[4]      = {3, 4, 2, 4};
    TStride strides[5]  = {1};
    strides[0]          = elemSize;
    strides[1]          = strides[0] * sizes[0];
    strides[2]          = strides[1] * sizes[1] * 2;
    strides[3]          = strides[2] * sizes[2];
    strides[4]          = strides[3] * sizes[3];
    CompilationHalReader::setHalReader((GaudiHalReader::instance(synDeviceGaudi)));

    TensorROI directTRoi   = calcTensorRoiDirectly(sizes, strides, 4, 0);
    TensorROI indirectTRoi = calcTensorRoiIndirectly(sizes, strides, 4, 0);

    checkNewAndOldMemoryRangesCollide(directTRoi, indirectTRoi);
}

TEST_F(CompareNewLegacyMemoryRanges, zero_sized_troi)
{
    uint32_t elemSize   = 4;
    TSize    sizes[4]   = {3, 3, 0, 2};
    uint64_t strides[5] = {1};
    strides[0]          = elemSize;
    strides[1]          = strides[0] * sizes[0] * 2;
    strides[2]          = strides[1] * sizes[1];
    strides[3]          = strides[2] * sizes[2] * 2;
    strides[4]          = strides[3] * sizes[3];
    CompilationHalReader::setHalReader((GaudiHalReader::instance(synDeviceGaudi)));

    TensorROI directTRoi = calcTensorRoiDirectly(sizes, strides, 4, 0);

    ASSERT_TRUE(directTRoi.m_overlapRoi.subRois->empty());
}

TEST_F(CompareNewLegacyMemoryRanges, unordered_strides)
{
    uint32_t elemSize   = 4;
    TSize sizes[4]      = {3, 4, 2, 4};
    TStride strides[5]  = {1};
    strides[0]          = elemSize;
    strides[1]          = strides[0] * sizes[0];
    strides[2]          = strides[1] * sizes[1] * 2;
    strides[3]          = strides[2] * sizes[2];
    strides[4]          = strides[3] * sizes[3];

    std::swap(strides[1], strides[2]);
    std::swap(sizes[1], sizes[2]);

    CompilationHalReader::setHalReader((GaudiHalReader::instance(synDeviceGaudi)));

    TensorROI directTRoi   = calcTensorRoiDirectly(sizes, strides, 4, 0);
    TensorROI indirectTRoi = calcTensorRoiIndirectly(sizes, strides, 4, 0);

    checkNewAndOldMemoryRangesCollide(directTRoi, indirectTRoi);
}

TEST_F(CompareNewLegacyMemoryRanges, unaligned_strides)
{
    uint32_t elemSize   = 4;
    TSize sizes[3]      = {5, 16, 256};
    TStride strides[4]  = {1};
    strides[0]          = elemSize;
    strides[1]          = 124;
    strides[2]          = 1184;
    strides[3]          = 303104;
    CompilationHalReader::setHalReader((GaudiHalReader::instance(synDeviceGaudi)));

    TensorROI directTRoi   = calcTensorRoiDirectly(sizes, strides, 3, 0);
    TensorROI indirectTRoi = calcTensorRoiIndirectly(sizes, strides, 3, 0);

    checkNewAndOldMemoryRangesCollide(directTRoi, indirectTRoi);
}

TEST_F(CompareNewLegacyMemoryRanges, conversion_double_stride)
{
    uint32_t elemSize   = 4;
    TSize sizes[3]      = {2, 64, 64};
    TStride strides[4]  = {1};
    strides[0]          = elemSize;
    strides[1]          = strides[0] * sizes[0] * 2;
    strides[2]          = strides[1] * sizes[1] * 3;
    strides[3]          = strides[2] * sizes[2];
    CompilationHalReader::setHalReader((GaudiHalReader::instance(synDeviceGaudi)));

    TensorROI directTRoi   = calcTensorRoiDirectly(sizes, strides, 3, 0);
    TensorROI indirectTRoi = calcTensorRoiIndirectly(sizes, strides, 3, 0);

    checkNewAndOldMemoryRangesCollide(directTRoi, indirectTRoi);
}

TEST_F(CompareNewLegacyMemoryRanges, conversion_offset)
{
    uint32_t elemSize   = 4;
    TSize sizes[3]      = {2, 64, 64};
    TStride strides[4]  = {1};
    strides[0]          = elemSize;
    strides[1]          = strides[0] * sizes[0] * 2;
    strides[2]          = strides[1] * sizes[1];
    strides[3]          = strides[2] * sizes[2];
    CompilationHalReader::setHalReader((GaudiHalReader::instance(synDeviceGaudi)));

    TensorROI directTRoi   = calcTensorRoiDirectly(sizes, strides, 3, 125);
    TensorROI indirectTRoi = calcTensorRoiIndirectly(sizes, strides, 3, 125);

    checkNewAndOldMemoryRangesCollide(directTRoi, indirectTRoi);
}

TEST_F(CompareNewLegacyMemoryRanges, conversion_double_stride_offset)
{
    uint32_t elemSize   = 4;
    TSize sizes[3]      = {2, 64, 64};
    TStride strides[4]  = {1};
    strides[0]          = elemSize;
    strides[1]          = strides[0] * sizes[0] * 2;
    strides[2]          = strides[1] * sizes[1] * 3;
    strides[3]          = strides[2] * sizes[2];
    CompilationHalReader::setHalReader((GaudiHalReader::instance(synDeviceGaudi)));

    TensorROI directTRoi   = calcTensorRoiDirectly(sizes, strides, 3, 125);
    TensorROI indirectTRoi = calcTensorRoiIndirectly(sizes, strides, 3, 125);

    checkNewAndOldMemoryRangesCollide(directTRoi, indirectTRoi);
}

TEST_F(CompareNewLegacyMemoryRanges, non_trivial_base_cyclic_dim)
{
    uint32_t elemSize   = 4;
    TSize sizes[4]      = {4, 8, 1, 11};
    TStride strides[5]  = {elemSize, 512, 128, 16, 4096};

    CompilationHalReader::setHalReader((GaudiHalReader::instance(synDeviceGaudi)));

    TensorROI directTRoi   = calcTensorRoiDirectly(sizes, strides, 4, 125);
    TensorROI indirectTRoi = calcTensorRoiIndirectly(sizes, strides, 4, 125);

    checkNewAndOldMemoryRangesCollide(directTRoi, indirectTRoi);
}

TEST_F(CompareNewLegacyMemoryRanges, overlapping_dense_ranges)
{
    uint32_t elemSize   = 4;
    TSize sizes[2]      = {4, 2};
    TStride strides[3]  = {elemSize, 8, 24};

    CompilationHalReader::setHalReader((GaudiHalReader::instance(synDeviceGaudi)));

    TensorROI directTRoi   = calcTensorRoiDirectly(sizes, strides, 2, 125);
    TensorROI indirectTRoi = calcTensorRoiIndirectly(sizes, strides, 2, 125);

    checkNewAndOldMemoryRangesCollide(directTRoi, indirectTRoi);
}

TEST_F(CompareNewLegacyMemoryRanges, overlapping_dense_ranges_2)
{
    uint32_t elemSize   = 4;
    TSize sizes[2]      = {4, 2};
    TStride strides[3]  = {elemSize, 8, 20};

    CompilationHalReader::setHalReader((GaudiHalReader::instance(synDeviceGaudi)));

    TensorROI directTRoi   = calcTensorRoiDirectly(sizes, strides, 2, 125);
    TensorROI indirectTRoi = calcTensorRoiIndirectly(sizes, strides, 2, 125);

    checkNewAndOldMemoryRangesCollide(directTRoi, indirectTRoi);
}

TEST_F(CompareNewLegacyMemoryRanges, overlapping_dense_ranges_3)
{
    uint32_t elemSize   = 4;
    TSize sizes[3]      = {4, 2, 32};
    TStride strides[4]  = {elemSize, 8, 30, 768};

    CompilationHalReader::setHalReader((GaudiHalReader::instance(synDeviceGaudi)));

    TensorROI directTRoi   = calcTensorRoiDirectly(sizes, strides, 3, 125);
    TensorROI indirectTRoi = calcTensorRoiIndirectly(sizes, strides, 3, 125);

    checkNewAndOldMemoryRangesCollide(directTRoi, indirectTRoi);
}
