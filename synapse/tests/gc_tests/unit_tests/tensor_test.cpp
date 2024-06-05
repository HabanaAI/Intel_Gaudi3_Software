#include <cstdio>
#include <fstream>
#include <gtest/gtest.h>
#include <utils/dense_coord_iter.h>
#include "graph_optimizer_test.h"
#include "syn_data_type_type_conversions.h"
#include "synapse_common_types.h"
#include "types.h"
#include "tensor.h"
#include "graph_compiler/habana_global_conf.h"

class TensorTests
: public GraphOptimizerTest
, public ::testing::WithParamInterface<int>
{
};

TEST_P(TensorTests, sanity)
{
    setGlobalConfForTest(GCFG_ENABLE_DYNAMIC_SHAPE_IN_HIGH_DIMENSION, "true");
    auto         dim = GetParam();
    NSizeArray   sizes;
    NSizeArray   minSizes;
    NStrideArray strides;

    strides[0] = 4;
    for (int i = 0; i < dim; i++)
    {
        sizes[i]    = i % 4 + 2;
        minSizes[i] = i % 4 + 1;
        strides[i + 1] = sizes[i] * strides[i];
    }

    Tensor t(dim,
             sizes.data(),
             synDataType::syn_type_float,
             nullptr,
             strides.data(),
             false,
             false,
             INVALID_BATCH_POS,
             minSizes.data());

    auto sizesFromTensor    = t.getNSizesInElements();
    auto minSizesFromTensor = t.getNMinimalSizesInElements();

    for (int i = 0; i < dim; i++)
    {
        ASSERT_EQ(t.getSizeInElements(i), sizes[i]);
        ASSERT_EQ(sizesFromTensor[i], sizes[i]);
        ASSERT_EQ(t.getSizeInBytes(i), sizes[i] * t.getElementSizeInBytes());
        ASSERT_EQ(t.getStrideInBytes(i), strides[i]);
        ASSERT_EQ(t.getStrideInElements(i), strides[i] / t.getElementSizeInBytes());
        ASSERT_EQ(t.getMinimalSizeInElements(i), minSizes[i]);
        ASSERT_EQ(minSizesFromTensor[i], minSizes[i]);
        ASSERT_EQ(t.getMinimalSizeInBytes(i), minSizes[i] * t.getElementSizeInBytes());
    }

    ASSERT_FALSE(t.isZeroSizedDataTensor());
    sizes[GetParam() - 1] = 0;
    t.reshape(dim, sizes.data(), nullptr);
    ASSERT_TRUE(t.isZeroSizedDataTensor());
}

INSTANTIATE_TEST_SUITE_P(, TensorTests, ::testing::Range(1, HABANA_DIM_MAX));  // dimension
