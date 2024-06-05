#include "gc_gaudi_test_infra.h"
#include "infra/gc_synapse_test.h"
#include "node_factory.h"

TEST_F_GC(SynTrainingTestInfra, strided_memset)
{
        unsigned memsetDim0 = 4;
        unsigned constsetDim0 = 5;
        unsigned dim1 = 5;
        unsigned dimsNum = 2;

        TestSizes memsetTensorSizes = {memsetDim0, dim1, 1, 1, 1};
        TestSizes constsetTensorSizes = {constsetDim0, dim1, 1, 1, 1}   ;
        TestSizes concatOutputSizes = {memsetDim0 + constsetDim0, dim1, 1, 1, 1};
        unsigned memsetOutputTensorId = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                            nullptr, memsetTensorSizes.data(), dimsNum);
        unsigned concatInputTensorId1 = createPersistTensor(INPUT_TENSOR, MEM_INIT_ALL_ONES, nullptr,
                                                            constsetTensorSizes.data(), dimsNum);
        unsigned concatInputTensor2 = connectOutputTensorToInputTensor(memsetOutputTensorId);
        addNodeToGraph(NodeFactory::memsetNodeTypeName, TensorIndices(), {memsetOutputTensorId});
        unsigned concatOutputTensorId = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                            nullptr, concatOutputSizes.data(), dimsNum);
        unsigned concatDim = 0;
        addNodeToGraph(NodeFactory::concatenateNodeTypeName, {concatInputTensorId1, concatInputTensor2},
                       {concatOutputTensorId}, &concatDim, sizeof(concatDim));

        compileAndRun();

        float* pOutputBuffer = (float*)m_hostBuffers[concatOutputTensorId];
        unsigned expectedValue = 0;
        for (unsigned row = 0; row < dim1; row++)
        {
            for (unsigned col = 0; col < memsetDim0 + constsetDim0; col++)
            {
                expectedValue = (col < constsetDim0) ? 1 : 0;
                ASSERT_EQ(expectedValue, *pOutputBuffer) << "Mismatch for at row, col: [" << row << ", " << col << "]"
                                                        << " Expected:" << expectedValue << " Result: " << *pOutputBuffer;
                pOutputBuffer++;
            }
        }
 }
