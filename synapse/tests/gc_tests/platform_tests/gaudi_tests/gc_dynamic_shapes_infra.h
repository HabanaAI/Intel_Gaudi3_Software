#pragma once

#include "synapse_api.h"
#include "node_factory.h"
#include <gtest/gtest.h>
#include "gc_gaudi_test_infra.h"
#include "runtime/common/recipe/recipe_handle_impl.hpp"
#include "scoped_configuration_change.h"
#include "smf/shape_func_registry.h"
#include "synapse_common_types.h"
#include <queue>

class SynGaudiDynamicShapesTestsInfra : public SynGaudiTestInfra
{
public:
    SynGaudiDynamicShapesTestsInfra()
    {
        setTestPackage(TEST_PACKAGE_DSD);
        setSupportedDevices({synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3});
    }

    void TestStaticTensor(Tensor* testTesnor);

    static TestSizes calculateStride(const unsigned* sizes, uint32_t dim);
    static TestSizes padSizes(const unsigned* sizes, uint32_t dim);

    template<typename T>
    std::queue<T> serializeBuffer(T* buffer, TestSizes origSizes, TestSizes actualSizes)
    {
        std::queue<T> serializedData;
        auto origStrides = calculateStride(origSizes.data(), origSizes.size());
        serializeBuffer(buffer, origSizes, origStrides, actualSizes, serializedData);
        return serializedData;
    }

    template<typename T>
    void serializeBuffer(const T*       buffer,
                         TestSizes      /*origSizes*/,
                         TestSizes      origStrides,
                         TestSizes      actualSizes,
                         std::queue<T>& serializedData)
    {
        uint32_t src_offset4 = 0;
        for(int di4 = 0; di4 < actualSizes[4]; di4++)
        {
            uint32_t src_offset3 = 0;
            for(int di3 = 0; di3 < actualSizes[3]; di3++)
            {
                uint32_t src_offset2 = 0;
                for(int di2 = 0; di2 < actualSizes[2]; di2++)
                {
                    uint32_t src_offset1 = 0;
                    for(int di1 = 0; di1 < actualSizes[1]; di1++)
                    {
                        for(int di0 = 0; di0 < actualSizes[0]; di0++)
                        {
                            serializedData.push(buffer[di0 + src_offset1 + src_offset2 + src_offset3 + src_offset4]);
                        }
                        src_offset1 += origStrides[1];
                    }
                    src_offset2 += origStrides[2];
                }
                src_offset3 += origStrides[3];
            }
            src_offset4 += origStrides[4];
        }
    }

    template<typename T>
    void deserializeBuffer(T* buffer, TestSizes srcSizes, TestSizes destStrides, std::queue<T>& serializedData)
    {
        uint32_t dst_offset4 = 0;
        for(int di4 = 0; di4 < srcSizes[4]; di4++)
        {
            uint32_t dst_offset3 = 0;
            for(int di3 = 0; di3 < srcSizes[3]; di3++)
            {
                uint32_t dst_offset2 = 0;
                for(int di2 = 0; di2 < srcSizes[2]; di2++)
                {
                    uint32_t dst_offset1 = 0;
                    for(int di1 = 0; di1 < srcSizes[1]; di1++)
                    {
                        for(int di0 = 0; di0 < srcSizes[0]; di0++)
                        {
                            buffer[di0 + dst_offset1 + dst_offset2 + dst_offset3 + dst_offset4] = serializedData.front();
                            serializedData.pop();
                        }
                        dst_offset1 += destStrides[1];
                    }
                    dst_offset2 += destStrides[2];
                }
                dst_offset3 += destStrides[3];
            }
            dst_offset4 += destStrides[4];
        }
    }


    template<typename T>
    void StridedToDenseBuffer(const T* src, T* dst, const unsigned* stridedSizes, const unsigned* denseSizes, unsigned dim)
    {
        TestSizes MaxSizes = padSizes(stridedSizes, dim);
        TestSizes ActualSizes = padSizes(denseSizes, dim);
        TestSizes stridedStrides = calculateStride(stridedSizes, dim);
        TestSizes denseStrides = calculateStride(denseSizes, dim);

        std::queue<T> queue;
        serializeBuffer<T>(src, MaxSizes, stridedStrides, ActualSizes, queue);
        memset(dst, 0, sizeof(T) * stridedStrides[4]);
        deserializeBuffer<T>(dst, ActualSizes, denseStrides, queue);
    }

    template <typename T>
    void StridedToDenseBuffer(T* buffer, const unsigned* stridedSizes, const unsigned* denseSizes, unsigned dim)
    {
        StridedToDenseBuffer<T>(buffer, buffer, stridedSizes, denseSizes, dim);
    }

    template<typename T>
    void setActualSizesAndSerialize(unsigned tensorIndex, const unsigned* tensorSizes)
    {
        SynGaudiTestInfra::setActualSizes(tensorIndex, tensorSizes);

        // Compact the input buffer to adjust for automatic serialize-deserialize added by the compiler.
        const auto& desc = m_tensorDescs[tensorIndex];
        StridedToDenseBuffer(castHostBuffer<T>(tensorIndex), desc.m_sizes, tensorSizes, desc.m_dims);
    }

    template <typename T>
    void DenseToStridedBuffer(T* buffer, unsigned* stridedSizes, unsigned* denseSizes, unsigned dim)
    {
        TestSizes ActualSizes = padSizes(denseSizes, dim);
        TestSizes stridedStrides = calculateStride(stridedSizes, dim);
        TestSizes denseStrides = calculateStride(denseSizes, dim);

        std::queue<T> queue;
        serializeBuffer<T>(buffer, ActualSizes, denseStrides, ActualSizes, queue);
        memset(buffer, 0, sizeof(T) * stridedStrides[4]);
        deserializeBuffer<T>(buffer, ActualSizes, stridedStrides, queue);
    }

    template <typename T>
    void CalcMaxpool2D_2by2(T* buffer, unsigned* sizes, unsigned dim, T* outBuffer)
    {
        TestSizes strides = calculateStride(sizes, dim);

        uint32_t batch_offset = 0;
        for(int b = 0; b < sizes[3]; b++)
        {
            for(int h = 0; h < sizes[2] / 2; h++)
            {
                for(int w = 0; w < sizes[1] / 2; w++)
                {
                    for(int c = 0; c < sizes[0]; c++)
                    {
                        // Test all 4 options for the 2 by 2 maxpool and get the max.
                        T max = buffer[c + (2 * w * strides[1]) + (2 * h * strides[2] + batch_offset)];
                        max = std::max(max, buffer[c + ((2 * w + 1) * strides[1]) + (2 * h * strides[2] + batch_offset)]);
                        max = std::max(max, buffer[c + (2 * w * strides[1]) + ((2 * h + 1) * strides[2] + batch_offset)]);
                        max = std::max(max, buffer[c + ((2 * w + 1) * strides[1]) + ((2 * h + 1) * strides[2] + batch_offset)]);

                        outBuffer[c + w * strides[1] + (h * strides[2]/2) + (batch_offset/4)] = max;
                    }
                }
            }

            batch_offset += strides[3];
        }
    }
};

class SynGaudiDynamicDMATestMemcpyBase : virtual public SynGaudiDynamicShapesTestsInfra
{
public:
    struct basic_dynamic_memcpy_s
    {
        const unsigned tensorDim = 4;
        const unsigned lastDim = tensorDim - 1;
        const unsigned minBatch = 2;
        const unsigned maxBatch = 10;
        unsigned H = 2;
        unsigned W = 64;
        unsigned C = 4;
        unsigned inMaxSize[4] = {C, W, H, maxBatch};
        unsigned inMinSize[4] = {C, W, H, minBatch};
        unsigned inTensor;
        unsigned outTensor;
    }  params;

    void createRecipe(unsigned graphIndex = 0);
    void runRecipe(unsigned actualBatch, unsigned graphIndex = 0);
    void checkResults(unsigned actualBatch);
};

class SynGaudiSimpleDynamicGemmAllDynamicBase : virtual public SynGaudiDynamicShapesTestsInfra
{
public:
    struct gemm_test_s
    {
        unsigned op1WMax   = 256;
        unsigned op1HMax   = 8 * 1024;
        unsigned op1HMin   = 64;
        unsigned op2WMax   = 256;
        unsigned op2WMin   = 128;
        unsigned op2HMax   = 256;

        size_t tensorDim = 2;
        unsigned op1MaxSizes[2] = {op1WMax, op1HMax};
        unsigned op1MinSizes[2] = {op1WMax, op1HMin};
        unsigned op2MaxSizes[2] = {op2WMax, op2HMax};
        unsigned op2MinSizes[2] = {op2WMin, op2HMax};

        unsigned outputMaxSizes[2] = {op2WMax, op1HMax};
        unsigned outputMinSizes[2] = {op2WMin, op1HMin};

        unsigned opA;
        unsigned opB;
        unsigned output;
        synGEMMParams gemmParams;
    } params;

    void createRecipe(unsigned graphIndex = 0);
    void runRecipe(unsigned actualBatch, unsigned graphIndex = 0, synStatus expected = synSuccess);
    void checkResults(unsigned actualBatch);
};
