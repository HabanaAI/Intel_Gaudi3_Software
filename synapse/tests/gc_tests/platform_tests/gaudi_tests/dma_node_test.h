#pragma once

#include <inttypes.h>
#include "gc_gaudi_test_infra.h"
#include "node_factory.h"
#include "synapse_common_types.h"

class SynGaudiTestDma : public SynGaudiTestInfra
{
public:
    SynGaudiTestDma() { setTestPackage(TEST_PACKAGE_DMA); }

    template<typename ValType = float>
    void linear_dma_node()
    {
        synDataType dtype = dataTypeToSynType<ValType>();
        auto        in    = createPersistTensor(INPUT_TENSOR,
                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                      nullptr,
                                      (unsigned*)getDefaultSizes32b(),
                                      DEFAULT_SIZES,
                                      dtype);
        auto        out   = createPersistTensor(OUTPUT_TENSOR,
                                       MEM_INIT_ALL_ZERO,
                                       nullptr,
                                       (unsigned*)getDefaultSizes32b(),
                                       DEFAULT_SIZES,
                                       dtype);
        addNodeToGraph("memcpy", {in}, {out});

        compileTopology();
        runTopology();

        int failed[2] {};
        for (int loop = 0; loop < 2; loop++)
        {
            ValType* pDmaInput  = (ValType*)m_hostBuffers[in];
            ValType* pDmaOutput = (ValType*)m_hostBuffers[out];

            failed[loop] = 0;
            for (uint64_t i = 0; i < getDefaultNumberOfElements(); i++)
            {
                if (*pDmaInput != *pDmaOutput)
                {
                    failed[loop]++;
                    if (failed[loop] < 0x10)
                    {
                        uint8_t* pDmaInput8  = (uint8_t*)pDmaInput;
                        uint8_t* pDmaOutput8 = (uint8_t*)pDmaOutput;

                        for (int j = 0; j < sizeof(ValType); j++)
                        {
                            printf("total elements %" PRIu64 " mismatch loop %d at index %" PRIu64 " in %X out %X\n",
                                   getDefaultNumberOfElements(),
                                   loop,
                                   i,
                                   pDmaInput8[j],
                                   pDmaOutput8[j]);
                        }
                    }
                }
                pDmaInput++;
                pDmaOutput++;
            }
            if (failed[0] == 0) break;  // all is good, no need to retry
            sleep(1);
        }
        ASSERT_EQ(failed[0], 0);
        ASSERT_EQ(failed[1], 0);
    }

    template<typename ValType = float>
    void relu_forward_and_backward_with_linear_dma()
    {
        // Graph will have three nodes: [relu_fwd]->[memcpy]->[relu_bwd]
        synDataType dtype = dataTypeToSynType<ValType>();
        unsigned    fwdIn = createPersistTensor(INPUT_TENSOR,
                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                             nullptr,
                                             (unsigned*)getDefaultSizes32b(),
                                             DEFAULT_SIZES,
                                             dtype);
        unsigned    fwdOut =
            createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, (unsigned*)getDefaultSizes32b(), DEFAULT_SIZES, dtype);
        addNodeToGraph("relu_fwd_f32");

        unsigned dmaIn = connectOutputTensorToInputTensor(fwdOut);
        unsigned dmaOut =
            createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, (unsigned*)getDefaultSizes32b(), DEFAULT_SIZES, dtype);
        addNodeToGraph("memcpy", {dmaIn}, {dmaOut});

        unsigned bwdIn1 = createPersistTensor(INPUT_TENSOR,
                                              MEM_INIT_RANDOM_WITH_NEGATIVE,
                                              nullptr,
                                              (unsigned*)getDefaultSizes32b(),
                                              DEFAULT_SIZES,
                                              dtype);
        unsigned bwdIn2 = connectOutputTensorToInputTensor(dmaOut);
        unsigned bwdOut = createPersistTensor(OUTPUT_TENSOR,
                                              MEM_INIT_ALL_ZERO,
                                              nullptr,
                                              (unsigned*)getDefaultSizes32b(),
                                              DEFAULT_SIZES,
                                              dtype);
        addNodeToGraph("relu_bwd_f32", {bwdIn1, bwdIn2}, {bwdOut});

        compileTopology();
        runTopology();

        ValType* pFwdInput  = (ValType*)m_hostBuffers[fwdIn];
        ValType* pBwdInput  = (ValType*)m_hostBuffers[bwdIn1];
        ValType* pBwdOutput = (ValType*)m_hostBuffers[bwdOut];

        for (uint64_t i = 0; i < getDefaultNumberOfElements(); i++)
        {
            ValType expectedResult = (*pFwdInput > 0) ? *pBwdInput : 0;
            ASSERT_EQ(expectedResult, *pBwdOutput);
            pFwdInput++;
            pBwdInput++;
            pBwdOutput++;
        }
    }

    template<typename ValType = float>
    void strided_dma_node()
    {
        synDataType dtype     = dataTypeToSynType<ValType>();
        unsigned    dim0      = 3 * 64;                    // elements
        unsigned    dim1      = 3 * 128;                   // elements
        uint64_t    stride0   = 3 * 96 * sizeof(ValType);  // bytes
        unsigned    sizes[]   = {dim0, dim1, 1, 1};
        unsigned    strides[] = {sizeof(ValType), stride0, stride0 * dim1, stride0 * dim1, stride0 * dim1};

        // Graph will have two nodes: [dma_dense_to_stride]->[dma_stride_to_dense]

        unsigned denseIn =
            createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, sizes, DEFAULT_SIZES, dtype);
        unsigned strideOut =
            createTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, sizes, DEFAULT_SIZES, dtype, strides);

        addNodeToGraph("memcpy", {denseIn}, {strideOut});

        unsigned strideIn = connectOutputTensorToInputTensor(strideOut);
        unsigned denseOut = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, sizes, DEFAULT_SIZES, dtype);
        addNodeToGraph("memcpy", {strideIn}, {denseOut});

        compileTopology();
        runTopology();

        int failed[2] {};
        for (int loop = 0; loop < 2; loop++)
        {
            ValType* pDmaInput  = (ValType*)m_hostBuffers[denseIn];
            ValType* pDmaOutput = (ValType*)m_hostBuffers[denseOut];

            failed[loop] = 0;
            for (uint64_t i = 0; i < getNumberOfElements(sizes); i++)
            {
                if (*pDmaInput != *pDmaOutput)
                {
                    failed[loop]++;
                    if (failed[loop] < 0x10)
                    {
                        uint8_t* pDmaInput8  = (uint8_t*)pDmaInput;
                        uint8_t* pDmaOutput8 = (uint8_t*)pDmaOutput;

                        for (int j = 0; j < sizeof(ValType); j++)
                        {
                            printf("total elements %" PRIu64 " mismatch loop %d at index %" PRIu64 " in %X out %X\n",
                                   getNumberOfElements(sizes),
                                   loop,
                                   i,
                                   pDmaInput8[j],
                                   pDmaOutput8[j]);
                        }
                    }
                }
                pDmaInput++;
                pDmaOutput++;
            }
            if (failed[0] == 0) break;  // all is good, no need to retry
            sleep(1);
        }
        ASSERT_EQ(failed[0], 0);
        ASSERT_EQ(failed[1], 0);
    }

    template<typename ValType = float>
    void three_dimensional_strided_dma_node()
    {
        synDataType dtype     = dataTypeToSynType<ValType>();
        unsigned    dim0      = 96;                          // elements
        unsigned    dim1      = 192;                         // elements
        unsigned    dim2      = 64;                          // elements
        uint64_t    stride0   = 3 * dim0 * sizeof(ValType);  // bytes
        uint64_t    stride1   = stride0 * dim1 + 8;          // bytes
        unsigned    sizes[]   = {dim0, dim1, dim2, 1};
        unsigned    strides[] = {sizeof(ValType), stride0, stride1, stride1 * dim2, stride1 * dim2};

        // Graph will have two nodes: [dma_dense_to_stride]->[dma_stride_to_dense]
        unsigned denseIn =
            createPersistTensor(INPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, sizes, DEFAULT_SIZES, dtype);
        unsigned strideOut =
            createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, sizes, DEFAULT_SIZES, dtype, strides);
        addNodeToGraph(NodeFactory::memcpyNodeTypeName, {denseIn}, {strideOut});

        unsigned strideIn = connectOutputTensorToInputTensor(strideOut);
        unsigned denseOut = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, sizes, DEFAULT_SIZES, dtype);
        addNodeToGraph(NodeFactory::memcpyNodeTypeName, {strideIn}, {denseOut});

        compileTopology();
        runTopology(0, true);

        int failed[2] {};
        for (int loop = 0; loop < 2; loop++)
        {
            ValType* pDmaInput  = (ValType*)m_hostBuffers[denseIn];
            ValType* pDmaOutput = (ValType*)m_hostBuffers[denseOut];

            failed[loop] = 0;
            for (uint64_t i = 0; i < getNumberOfElements(sizes); i++)
            {
                if (*pDmaInput != *pDmaOutput)
                {
                    failed[loop]++;
                    if (failed[loop] < 0x10)
                    {
                        uint8_t* pDmaInput8  = (uint8_t*)pDmaInput;
                        uint8_t* pDmaOutput8 = (uint8_t*)pDmaOutput;

                        for (int j = 0; j < sizeof(ValType); j++)
                        {
                            printf("total elements %" PRIu64 " mismatch loop %d at index %" PRIu64 " in %X out %X\n",
                                   getNumberOfElements(sizes),
                                   loop,
                                   i,
                                   pDmaInput8[j],
                                   pDmaOutput8[j]);
                        }
                    }
                }
                pDmaInput++;
                pDmaOutput++;
            }
            if (failed[0] == 0) break;  // all is good, no need to retry
            sleep(1);
        }
        ASSERT_EQ(failed[0], 0);
        ASSERT_EQ(failed[1], 0);
    }
};

class SynGaudiTestDmaMemset
: public SynGaudiTestInfra
, public testing::WithParamInterface<std::tuple<int, int>>

{
public:
    SynGaudiTestDmaMemset() { setTestPackage(TEST_PACKAGE_DMA); }

    template<typename ValType = float>
    void linear_memset_dma_node()
    {
        synDataType dtype = dataTypeToSynType<ValType>();
        auto        out   = createPersistTensor(OUTPUT_TENSOR,
                                       MEM_INIT_RANDOM_POSITIVE,
                                       nullptr,
                                       (unsigned*)getDefaultSizes32b(),
                                       DEFAULT_SIZES,
                                       dtype);
        addNodeToGraph("memset");
        compileTopology();
        runTopology(0, true);

        ValType* pOutput1 = (ValType*)m_hostBuffers[out];

        for (uint64_t i = 0; i < getDefaultNumberOfElements(); i++)
        {
            ValType expectedResult = (ValType)((float)0);
            ASSERT_EQ(expectedResult, *pOutput1) << "Mismatch at index " << i;
            pOutput1++;
        }
    }

    template<typename ValType = float>
    void linear_memset_2d_dma_node()
    {
        synDataType dtype   = dataTypeToSynType<ValType>();
        unsigned    dim0    = 3 * 64;   // elements
        unsigned    dim1    = 3 * 128;  // elements
        unsigned    sizes[] = {dim0, dim1, 1, 1, 1};

        unsigned out =
            createPersistTensor(OUTPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, sizes, DEFAULT_SIZES, dtype);
        addNodeToGraph("memset", {}, TensorIndices {out});

        compileTopology();
        runTopology(0, true);

        ValType* pOutput1 = (ValType*)m_hostBuffers[out];

        for (uint64_t i = 0; i < getNumberOfElements(sizes); i++)
        {
            ValType expectedResult = (ValType)((float)0);
            ASSERT_EQ(expectedResult, *pOutput1) << "Mismatch at index " << i;
            pOutput1++;
        }
    }

    template<typename ValType = float>
    void three_dimensional_memset_node()
    {
        synDataType dtype   = dataTypeToSynType<ValType>();
        unsigned    dim0    = 3 * 32;  // elements
        unsigned    dim1    = 3 * 64;  // elements
        unsigned    dim2    = 3 * 16;  // elements
        unsigned    sizes[] = {dim0, dim1, dim2, 1};

        // Graph will have two nodes: [memset_stride]->[dma_stride_to_dense]

        unsigned out =
            createPersistTensor(OUTPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, sizes, DEFAULT_SIZES, dtype);
        addNodeToGraph(NodeFactory::memsetNodeTypeName, {}, TensorIndices {out});

        unsigned in = connectOutputTensorToInputTensor(out);
        unsigned denseOut =
            createPersistTensor(OUTPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE, nullptr, sizes, DEFAULT_SIZES, dtype);
        addNodeToGraph(NodeFactory::memcpyNodeTypeName, {in}, {denseOut});

        compileTopology();
        runTopology(0, true);

        ValType* pDmaOutput     = (ValType*)m_hostBuffers[denseOut];
        ValType  expectedResult = (ValType)((float)0);
        for (uint64_t i = 0; i < getNumberOfElements(sizes); i++)
        {
            ASSERT_EQ(expectedResult, *pDmaOutput) << "Mismatch at index " << i;
            pDmaOutput++;
        }
    }

    void linear_memset_dma_node_single_test()
    {
        unsigned sizes[1]      = {0};
        unsigned memsetSize    = std::get<0>(GetParam());
        unsigned parallelLevel = std::get<1>(GetParam());

        GlobalConfTestSetter conf_parallelLevel("GAUDI_MEMSET_PARALLEL_LEVEL", std::to_string(parallelLevel));

        sizes[0] = memsetSize;
        auto out = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_RANDOM_POSITIVE, nullptr, sizes, 1, syn_type_single);

        addNodeToGraph("memset");
        compileTopology();
        runTopology(0, true);

        float* pOutput1 = (float*)m_hostBuffers[out];

        float expectedResult = 0;
        for (unsigned i = 0; i < memsetSize; i++)
        {
            ASSERT_EQ(expectedResult, *pOutput1) << "Mismatch at index " << i;
            pOutput1++;
        }
    }
};  // SynGaudiTestDmaMemset
