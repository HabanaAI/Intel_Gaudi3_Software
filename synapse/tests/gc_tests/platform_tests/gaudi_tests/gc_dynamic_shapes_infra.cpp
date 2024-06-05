#include "gc_dynamic_shapes_infra.h"

TestSizes SynGaudiDynamicShapesTestsInfra::calculateStride(const unsigned* sizes, uint32_t dim)
{
    TestSizes stridesArray;
    stridesArray[0] = 1;

    for (int d = 1; d < SYN_MAX_TENSOR_DIM; d++)
    {
        if (d <= dim)
        {
            stridesArray[d] = stridesArray[d - 1] * sizes[d - 1];
        }
        else
        {
            stridesArray[d] = stridesArray[d - 1];
        }
    }
    return stridesArray;
}

TestSizes SynGaudiDynamicShapesTestsInfra::padSizes(const unsigned* sizes, uint32_t dim)
{
    TestSizes sizesArray;

    for (int d = 0; d < SYN_MAX_TENSOR_DIM; d++)
    {
        if (d < dim)
        {
            sizesArray[d] = sizes[d];
        }
        else
        {
            sizesArray[d] = 1;
        }
    }
    return sizesArray;
}

void SynGaudiDynamicShapesTestsInfra::TestStaticTensor(Tensor* testTensor)
{
    ASSERT_FALSE(testTensor->isDynamicShape());
    ASSERT_EQ(testTensor->getMinimalElements(), testTensor->getTotalElements());
    ASSERT_EQ(testTensor->getMinimalSizeInBytes(), testTensor->getTotalSizeInBytes());

    SizeArray minSizesArray = testTensor->getAllMinimalSizesInElements();
    TSize minSizes[SYN_MAX_TENSOR_DIM];
    testTensor->getAllMinimalSizesInElements(minSizes, SYN_MAX_TENSOR_DIM);

    SizeArray sizesArray = testTensor->getAllSizesInElements();
    TSize sizes[SYN_MAX_TENSOR_DIM];
    testTensor->getAllSizesInElements(sizes, SYN_MAX_TENSOR_DIM);

    for (int i = 0; i < testTensor->getDim(); i++)
    {
        ASSERT_EQ(minSizes[i], sizes[i]);
        ASSERT_EQ(minSizesArray[i], sizesArray[i]);
        ASSERT_EQ(testTensor->getMinimalSizeInBytes(i), testTensor->getSizeInBytes(i));
        ASSERT_EQ(testTensor->getMinimalSizeInElements(i), testTensor->getSizeInElements(i));
    }
}
