#include "gaudi_dual_execution_test_infra.h"
#include "node_factory.h"
#include "synapse_common_types.hpp"

#include <bitset>

class SynTrainingTransposeFusionDualExecutionTest : public SynDualExecutionGaudiTestInfra
{
public:
    static constexpr unsigned NUM_OPERANDS = 3;
    enum GemmOperands
    {
        OPERAND_A,
        OPERAND_B,
        OPERAND_OUT,
        OPERANDS_COUNT
    };

    void simpleGemmTest(std::bitset<OPERANDS_COUNT> transposedOperands, std::bitset<OPERANDS_COUNT> persistentOperands);
    void simpleBatchGemmTest(std::bitset<OPERANDS_COUNT> transposedOperands);
    void sameInputGemmTest(std::bitset<OPERANDS_COUNT> transposedOperands, bool sameTranspose);
};

void SynTrainingTransposeFusionDualExecutionTest::simpleGemmTest(std::bitset<OPERANDS_COUNT> transposedOperands,
                                                                 std::bitset<OPERANDS_COUNT> persistentOperands)
{
    static constexpr unsigned NUM_DIMS = 2;

    std::array<unsigned, NUM_DIMS>                             sizesA       = {90, 45};
    std::array<unsigned, NUM_DIMS>                             sizesB       = {35, 90};
    std::array<unsigned, NUM_DIMS>                             sizesC       = {35, 45};
    std::array<std::array<unsigned, NUM_DIMS>, OPERANDS_COUNT> operandSizes = {sizesA, sizesB, sizesC};
    std::array<TensorIndexPair, OPERANDS_COUNT>                operands     = {};
    TensorIndexPair                                            graphOutput  = {};

    GraphIndexPair graphIndexPair = createNewGraphPair();

    for (int i = 0; i < OPERANDS_COUNT; i++)
    {
        std::array<unsigned, NUM_DIMS> persistentTensorSizes = operandSizes[i];
        if (transposedOperands[i])
        {
            persistentTensorSizes = {operandSizes[i][1], operandSizes[i][0]};
        }
        TensorIndexPair persistentTensor = createPersistTensors(INPUT_TENSOR,
                                                                MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                nullptr,
                                                                persistentTensorSizes.data(),
                                                                persistentTensorSizes.size(),
                                                                syn_type_single,
                                                                false,
                                                                nullptr,
                                                                nullptr,
                                                                graphIndexPair);

        if (transposedOperands[i])
        {
            std::array<unsigned, NUM_DIMS> transposeDimSizes = {persistentTensorSizes[1], persistentTensorSizes[0]};
            TensorIndexPair                transposeOperand  = {};
            if (persistentOperands[i])
            {
                transposeOperand = createPersistTensors(INPUT_TENSOR,
                                                        MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                        nullptr,
                                                        transposeDimSizes.data(),
                                                        transposeDimSizes.size(),
                                                        syn_type_single,
                                                        false,
                                                        nullptr,
                                                        nullptr,
                                                        graphIndexPair);
            }
            else
            {
                transposeOperand = createTensors(INPUT_TENSOR,
                                                 MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                 nullptr,
                                                 transposeDimSizes.data(),
                                                 transposeDimSizes.size(),
                                                 syn_type_single,
                                                 nullptr,
                                                 graphIndexPair);
            }

            TensorIndicesPair inIndices;
            TensorIndicesPair outIndices;

            if (i == OPERAND_OUT)
            {
                inIndices  = {{transposeOperand.graph}, {transposeOperand.eager}};
                outIndices = {{persistentTensor.graph}, {persistentTensor.eager}};
            }
            else
            {
                inIndices  = {{persistentTensor.graph}, {persistentTensor.eager}};
                outIndices = {{transposeOperand.graph}, {transposeOperand.eager}};
            }
            operands[i]                        = transposeOperand;
            synTransposeParams transposeParams = {{TPD_Width, TPD_Channel}, NUM_DIMS};
            addNodesToGraphs(NodeFactory::transposeNodeTypeName,
                             inIndices,
                             outIndices,
                             &transposeParams,
                             sizeof(transposeParams),
                             nullptr,
                             graphIndexPair);
        }
        else
        {
            operands[i] = persistentTensor;
        }
    }

    synGEMMParams     gemmParams = {};
    TensorIndicesPair inIndices  = {{operands[OPERAND_A].graph, operands[OPERAND_B].graph},
                                   {operands[OPERAND_A].eager, operands[OPERAND_B].eager}};
    TensorIndicesPair outIndices = {{operands[OPERAND_OUT].graph}, {operands[OPERAND_OUT].eager}};
    addNodesToGraphs(NodeFactory::gemmNodeTypeName,
                     inIndices,
                     outIndices,
                     &gemmParams,
                     sizeof(gemmParams),
                     nullptr,
                     graphIndexPair);

    compileTopology("topology", graphIndexPair);
    runTopology(graphIndexPair);

    auto pOutputBufferGraphMode = static_cast<float*>(m_hostBuffers[graphOutput.graph]);
    auto pOutputBufferEagerMode = static_cast<float*>(m_hostBuffers[graphOutput.eager]);
    for (uint64_t i = 0; i < getNumberOfElements(sizesC.data(), sizesC.size()); i++)
    {
        ASSERT_EQ(pOutputBufferGraphMode[i], pOutputBufferEagerMode[i])
            << "Graph mode mismatch at index " << i << " Graph mode:" << pOutputBufferGraphMode[i]
            << " Eager mode: " << pOutputBufferEagerMode[i];
    }
}

TEST_F_GC(SynTrainingTransposeFusionDualExecutionTest, gemm_transpose_fusion_no_persistent_operands)
{
    for (int i = 0; i < pow(2, OPERANDS_COUNT); i++)
    {
        simpleGemmTest(i, 0);
    }
}

TEST_F_GC(SynTrainingTransposeFusionDualExecutionTest, gemm_transpose_fusion_with_persistent_operands)
{
    for (int i = 1; i < pow(2, OPERANDS_COUNT); i++)
    {
        std::bitset<OPERANDS_COUNT> transposedOperands(i);
        for (int j = 0; j < OPERANDS_COUNT; j++)
        {
            if (transposedOperands[j])
            {
                simpleGemmTest(i, (1 << j));
            }
        }
    }
    std::bitset<OPERANDS_COUNT> fullBitmap(0b111);
    simpleGemmTest(fullBitmap, fullBitmap);
}

void SynTrainingTransposeFusionDualExecutionTest::simpleBatchGemmTest(std::bitset<OPERANDS_COUNT> transposedOperands)
{
    static constexpr unsigned NUM_DIMS = 3;

    std::array<unsigned, NUM_DIMS>                             sizesA       = {90, 45, 3};
    std::array<unsigned, NUM_DIMS>                             sizesB       = {35, 90, 3};
    std::array<unsigned, NUM_DIMS>                             sizesC       = {35, 45, 3};
    std::array<std::array<unsigned, NUM_DIMS>, OPERANDS_COUNT> operandSizes = {sizesA, sizesB, sizesC};
    std::array<TensorIndexPair, OPERANDS_COUNT>                operands     = {};
    TensorIndexPair                                            graphOutput  = {};

    GraphIndexPair graphIndexPair = createNewGraphPair();

    for (int i = 0; i < OPERANDS_COUNT; i++)
    {
        std::array<unsigned, NUM_DIMS> persistentTensorSizes = operandSizes[i];
        if (transposedOperands[i])
        {
            persistentTensorSizes = {operandSizes[i][1], operandSizes[i][0], operandSizes[i][2]};
        }
        TensorIndexPair persistentTensor = createPersistTensors(INPUT_TENSOR,
                                                                MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                                nullptr,
                                                                persistentTensorSizes.data(),
                                                                persistentTensorSizes.size(),
                                                                syn_type_single,
                                                                false,
                                                                nullptr,
                                                                nullptr,
                                                                graphIndexPair);

        if (transposedOperands[i])
        {
            std::array<unsigned, NUM_DIMS> transposeDimSizes = {persistentTensorSizes[1],
                                                                persistentTensorSizes[0],
                                                                persistentTensorSizes[2]};
            TensorIndexPair                transposeOperand  = createTensors(INPUT_TENSOR,
                                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                             nullptr,
                                                             transposeDimSizes.data(),
                                                             transposeDimSizes.size(),
                                                             syn_type_single,
                                                             nullptr,
                                                             graphIndexPair);

            TensorIndicesPair inIndices;
            TensorIndicesPair outIndices;

            if (i == OPERAND_OUT)
            {
                inIndices  = {{transposeOperand.graph}, {transposeOperand.eager}};
                outIndices = {{persistentTensor.graph}, {persistentTensor.eager}};
            }
            else
            {
                inIndices  = {{persistentTensor.graph}, {persistentTensor.eager}};
                outIndices = {{transposeOperand.graph}, {transposeOperand.eager}};
            }
            operands[i]                        = transposeOperand;
            synTransposeParams transposeParams = {{TPD_Width, TPD_Channel, TPD_Height}, NUM_DIMS};
            addNodesToGraphs(NodeFactory::transposeNodeTypeName,
                             inIndices,
                             outIndices,
                             &transposeParams,
                             sizeof(transposeParams),
                             nullptr,
                             graphIndexPair);
        }
        else
        {
            operands[i] = persistentTensor;
        }
    }

    synGEMMParams     gemmParams = {};
    TensorIndicesPair inIndices  = {{operands[OPERAND_A].graph, operands[OPERAND_B].graph},
                                   {operands[OPERAND_A].eager, operands[OPERAND_B].eager}};
    TensorIndicesPair outIndices = {{operands[OPERAND_OUT].graph}, {operands[OPERAND_OUT].eager}};
    addNodesToGraphs(NodeFactory::batchGemmNodeTypeName,
                     inIndices,
                     outIndices,
                     &gemmParams,
                     sizeof(gemmParams),
                     nullptr,
                     graphIndexPair);

    compileTopology("topology", graphIndexPair);
    runTopology(graphIndexPair);

    auto pOutputBufferGraphMode = static_cast<float*>(m_hostBuffers[graphOutput.graph]);
    auto pOutputBufferEagerMode = static_cast<float*>(m_hostBuffers[graphOutput.eager]);
    for (uint64_t i = 0; i < getNumberOfElements(sizesC.data(), sizesC.size()); i++)
    {
        ASSERT_EQ(pOutputBufferGraphMode[i], pOutputBufferEagerMode[i])
            << "Graph mode mismatch at index " << i << " Graph mode:" << pOutputBufferGraphMode[i]
            << " Eager mode: " << pOutputBufferEagerMode[i];
    }
}

TEST_F_GC(SynTrainingTransposeFusionDualExecutionTest, batch_gemm_transpose_fusion)
{
    for (int i = 0; i < pow(2, OPERANDS_COUNT); i++)
    {
        simpleBatchGemmTest(i);
    }
}

void SynTrainingTransposeFusionDualExecutionTest::sameInputGemmTest(std::bitset<OPERANDS_COUNT> transposedOperands,
                                                                    bool                        sameTranspose)
{
    static constexpr unsigned NUM_DIMS = 2;

    std::array<unsigned, NUM_DIMS>              sizes    = {45, 45};
    std::array<TensorIndexPair, OPERANDS_COUNT> operands = {};

    GraphIndexPair graphIndexPair = createNewGraphPair();

    TensorIndexPair graphInput = createPersistTensors(INPUT_TENSOR,
                                                      MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                      nullptr,
                                                      sizes.data(),
                                                      sizes.size(),
                                                      syn_type_single,
                                                      false,
                                                      nullptr,
                                                      nullptr,
                                                      graphIndexPair);

    TensorIndexPair graphOutput = createPersistTensors(INPUT_TENSOR,
                                                       MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                       nullptr,
                                                       sizes.data(),
                                                       sizes.size(),
                                                       syn_type_single,
                                                       false,
                                                       nullptr,
                                                       nullptr,
                                                       graphIndexPair);

    for (int i = 0; i < OPERANDS_COUNT; i++)
    {
        if (transposedOperands[i])
        {
            if (sameTranspose && transposedOperands[OPERAND_A] && transposedOperands[OPERAND_B] && i == OPERAND_B)
            {
                operands[OPERAND_B] = operands[OPERAND_A];
                continue;
            }

            TensorIndexPair transposeOperand = createTensors(INPUT_TENSOR,
                                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                             nullptr,
                                                             sizes.data(),
                                                             sizes.size(),
                                                             syn_type_single,
                                                             nullptr,
                                                             graphIndexPair);

            TensorIndicesPair inIndices;
            TensorIndicesPair outIndices;

            if (i == OPERAND_OUT)
            {
                inIndices  = {{transposeOperand.graph}, {transposeOperand.eager}};
                outIndices = {{graphOutput.graph}, {graphOutput.eager}};
            }
            else
            {
                inIndices  = {{graphInput.graph}, {graphInput.eager}};
                outIndices = {{transposeOperand.graph}, {transposeOperand.eager}};
            }
            operands[i]                        = transposeOperand;
            synTransposeParams transposeParams = {{TPD_Width, TPD_Channel}, NUM_DIMS};
            addNodesToGraphs(NodeFactory::transposeNodeTypeName,
                             inIndices,
                             outIndices,
                             &transposeParams,
                             sizeof(transposeParams),
                             nullptr,
                             graphIndexPair);
        }
        else
        {
            operands[i] = (i == OPERAND_OUT) ? graphOutput : graphInput;
        }
    }

    synGEMMParams     gemmParams = {};
    TensorIndicesPair inIndices  = {{operands[OPERAND_A].graph, operands[OPERAND_B].graph},
                                   {operands[OPERAND_A].eager, operands[OPERAND_B].eager}};
    TensorIndicesPair outIndices = {{operands[OPERAND_OUT].graph}, {operands[OPERAND_OUT].eager}};
    addNodesToGraphs(NodeFactory::gemmNodeTypeName,
                     inIndices,
                     outIndices,
                     &gemmParams,
                     sizeof(gemmParams),
                     nullptr,
                     graphIndexPair);

    compileTopology("topology", graphIndexPair);
    runTopology(graphIndexPair);

    auto pOutputBufferGraphMode = static_cast<float*>(m_hostBuffers[graphOutput.graph]);
    auto pOutputBufferEagerMode = static_cast<float*>(m_hostBuffers[graphOutput.eager]);
    for (uint64_t i = 0; i < getNumberOfElements(sizes.data(), sizes.size()); i++)
    {
        ASSERT_EQ(pOutputBufferGraphMode[i], pOutputBufferEagerMode[i])
            << "Graph mode mismatch at index " << i << " Graph mode:" << pOutputBufferGraphMode[i]
            << " Eager mode: " << pOutputBufferEagerMode[i];
    }
}

TEST_F_GC(SynTrainingTransposeFusionDualExecutionTest, gemm_transpose_fusion_repeated_input)
{
    for (int i = 0; i < pow(2, OPERANDS_COUNT); i++)
    {
        sameInputGemmTest(i, false);
        sameInputGemmTest(i, true);
    }
}