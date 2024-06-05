#include "gc_gaudi_test_infra.h"
#include "node_factory.h"
#include "utils.h"

class SynTrainingMemcpyToCastTest : public SynTrainingTestInfra
{
};
// Since the input and output data types are different, a tpc node of cast is inserted to the graph instead of the
// original (semantic) memcpy node during the selectMemcpyEngine pass.
// The test checks the cast results are correct
TEST_F_GC(SynTrainingMemcpyToCastTest, memcpy_to_cast)
{
    const unsigned FCD = 4, WIDTH = 1, HEIGHT = 1, BATCH = 1;
    unsigned int dimensions[] = { FCD, WIDTH, HEIGHT, BATCH };
    const unsigned numOfElements = FCD * WIDTH * HEIGHT * BATCH;
    float *inputArray = new float[numOfElements]{ 9999, 0, 1, 2019.1212 };

    unsigned tensorIn = createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, (float*)inputArray,
                                     (unsigned*)dimensions, DEFAULT_SIZES, syn_type_float);
    unsigned tensorOut = createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr,
                                     (unsigned*)dimensions, DEFAULT_SIZES, syn_type_bf16);

    addNodeToGraph(NodeFactory::memcpyNodeTypeName, {tensorIn}, {tensorOut});
    compileAndRun();

    //validate
    float* pInputBuffer = (float*)m_hostBuffers[tensorIn];
    int16_t* pOutputBuffer = (int16_t*)m_hostBuffers[tensorOut];

    for (unsigned i = 0; i < numOfElements; ++i)
    {
        int16_t expectedRes = castFloatToBFloat16(*pInputBuffer);
        ASSERT_EQ(expectedRes, *pOutputBuffer) << "Mismatch for at index " << i
                                               << " Expected:"             << expectedRes
                                               << " Result: "              << *pOutputBuffer
                                               << " operand "              << *pInputBuffer;
        ++pInputBuffer;
        ++pOutputBuffer;
    }

    delete[] inputArray;
}
