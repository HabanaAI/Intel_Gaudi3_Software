#include "graph_optimizer_test.h"
#include "node.h"
#include "gaudi_graph.h"
#include "node_factory.h"
#include "synapse_common_types.h"
#include "tpc_node.h"
#include "perf_lib_layer_params.h"
#include "graph_compiler/passes/sram_management/sliced_operand_traversal.h"
#include "graph_compiler/passes/sram_management/slice_mapping.h"
#include "platform/gaudi/graph_compiler/passes.h"

class AccessPatternSliceMapperTest : public GraphOptimizerTest
{
protected:
    GaudiGraph     m_graph;
    NodePtr        m_tpcNode;
    const unsigned batchSliceSize = 16;
    const unsigned batchSize      = 256;

    pTensor createTensor(const std::vector<TSize>& shape, synDataType dataType)
    {
        return std::make_shared<Tensor>(shape.size(), shape.data(), dataType);
    }

    void createBnNode()
    {
        pTensor ifm               = createTensor({64, 56, 56, batchSize}, syn_type_bf16);
        pTensor k                 = createTensor({64}, syn_type_float);
        pTensor sigmas            = createTensor({64, 2}, syn_type_float);
        pTensor betaGamma         = createTensor({64, 2}, syn_type_float);
        pTensor runningMeanAndVar = createTensor({64, 2}, syn_type_float);

        pTensor ofm                  = createTensor({64, 56, 56, batchSize}, syn_type_bf16);
        pTensor runningMeanAndVarOut = createTensor({64, 2}, syn_type_float);
        pTensor meanAndStdOut        = createTensor({64, 2}, syn_type_float);

        m_tpcNode = NodeFactory::createNode({ifm, k, sigmas, betaGamma, runningMeanAndVar},
                                            {ofm, runningMeanAndVarOut, meanAndStdOut},
                                            nullptr,
                                            "batch_norm_stage2_fwd_bf16",
                                            "BN2");

        ns_BatchNormStage2Kernel::Params stage2Params;
        stage2Params.N = m_tpcNode->getInput(0)->getTotalElements() / m_tpcNode->getInput(0)->getSizeInElements(DIM_C);
        stage2Params.momentum                = 0;
        stage2Params.disable_runnings_update = 1;
        stage2Params.epsilon                 = 0.1;

        std::dynamic_pointer_cast<TPCNode>(m_tpcNode)->storeParamsInBuffer(&stage2Params,
                                                                           sizeof(ns_BatchNormStage2Kernel::Params));

        ASSERT_TRUE(GraphEditor::addNode(m_graph, m_tpcNode));
        ASSERT_TRUE(gaudi::loadTpcKernels(m_graph));
    }

    void createBnSlicedOperands(std::list<pSlicedOperand>& inputOperands, std::list<pSlicedOperand>& outputOperands)
    {
        for (const auto& in : m_tpcNode->getInputs())
        {
            inputOperands.push_back(std::make_shared<SlicedOperand>(in));
        }

        for (const auto& out : m_tpcNode->getOutputs())
        {
            outputOperands.push_back(std::make_shared<SlicedOperand>(out));
        }

        // Slice first input and first output on batch dim.
        inputOperands.front()->chunkDimensions[DIM_B] = outputOperands.front()->chunkDimensions[DIM_B] = batchSliceSize;
    }

    void validateMapping(const SliceReferenceList& inputs, const SliceReferenceList& outputs, unsigned sliceIdx) const
    {
        ASSERT_EQ(inputs.size(), m_tpcNode->getInputs().size());
        ASSERT_EQ(outputs.size(), m_tpcNode->getOutputs().size());

        std::vector<pSliceReference> inputsVec(inputs.begin(), inputs.end());
        std::vector<pSliceReference> outputsVec(outputs.begin(), outputs.end());

        CoordArray zeroCoord;
        zeroCoord.fill(0);

        CoordArray sliceCoord;
        sliceCoord.fill(0);
        sliceCoord[DIM_B] = sliceIdx;

        for (auto i = 0; i < m_tpcNode->getNumInputs(); i++)
        {
            if (i == 0)
            {
                ASSERT_EQ(inputsVec[i]->coordinates, sliceCoord);
            }
            else
            {
                ASSERT_EQ(inputsVec[i]->coordinates, zeroCoord);
            }
        }

        for (auto i = 0; i < m_tpcNode->getNumOutputs(); i++)
        {
            if (i == 0)
            {
                ASSERT_EQ(outputsVec[i]->coordinates, sliceCoord);
            }
            else
            {
                ASSERT_EQ(outputsVec[i]->coordinates, zeroCoord);
            }
        }
    }
};

TEST_F(AccessPatternSliceMapperTest, access_pattern_fwd_mapping)
{
    createBnNode();

    std::list<pSlicedOperand> inputOperands;
    std::list<pSlicedOperand> outputOperands;
    createBnSlicedOperands(inputOperands, outputOperands);

    pForwardSliceMapping mapping = AccessPatternSliceMapper::createFwdMapping(m_tpcNode, inputOperands, outputOperands);
    ASSERT_NE(mapping, nullptr);

    ASSERT_EQ(mapping->getInputs().size(), m_tpcNode->getInputs().size());
    ASSERT_EQ(mapping->getOutputs().size(), m_tpcNode->getOutputs().size());

    unsigned nofSlices = 0;
    for (auto inputSlice :
         SlicedOperandTraversalPattern(inputOperands.front(), SlicedOperandTraversalPattern::LEFT_TO_RIGHT_4D))
    {
        SliceReferenceList inputs, outputs;
        std::tie(inputs, outputs) = mapping->getInputsAndOutputs(inputSlice.first).front();

        validateMapping(inputs, outputs, nofSlices);

        nofSlices++;
    }
    ASSERT_EQ(nofSlices, (batchSize / batchSliceSize));
}

TEST_F(AccessPatternSliceMapperTest, access_pattern_bwd_mapping)
{
    createBnNode();

    std::list<pSlicedOperand> inputOperands;
    std::list<pSlicedOperand> outputOperands;
    createBnSlicedOperands(inputOperands, outputOperands);

    std::vector<pSlicedOperand> inputOperandsVec(inputOperands.begin(), inputOperands.end());
    std::vector<pSlicedOperand> outputOperandsVec(outputOperands.begin(), outputOperands.end());
    pBackwardSliceMapping       mapping =
        AccessPatternSliceMapper::createBwdMapping(m_tpcNode, inputOperandsVec, outputOperandsVec);
    ASSERT_NE(mapping, nullptr);

    ASSERT_EQ(mapping->getInOperands().size(), m_tpcNode->getInputs().size());

    unsigned nofSlices = 0;
    for (auto outputSlice :
         SlicedOperandTraversalPattern(outputOperands.front(), SlicedOperandTraversalPattern::LEFT_TO_RIGHT_4D))
    {
        SliceReferenceList inputs  = mapping->getInputs(outputSlice);
        SliceReferenceList outputs = mapping->getOutputs(outputSlice.first);

        validateMapping(inputs, outputs, nofSlices);

        nofSlices++;
    }
    ASSERT_EQ(nofSlices, (batchSize / batchSliceSize));
}

TEST_F(AccessPatternSliceMapperTest, access_pattern_mapping_with_zero_sized_shape_tensors)
{
    const std::vector<TSize> fmSize  = {1, 64, 64, 1, 1};
    unsigned padData[] = {0, 3, 3, 0, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    const std::vector<TSize> padSize = {10};

    std::vector<TSize> paddedFmSize(fmSize.size(), 0);
    for (auto idx = 0; idx < fmSize.size(); idx++)
    {
        paddedFmSize[idx] = fmSize[idx] + padData[idx] + padData[idx + MAX_DIMENSIONS_NUM];
    }

    pTensor padH2D = createTensor(padSize, syn_type_uint32);
    padH2D->setTensorType(HOST_TO_DEVICE_TENSOR);
    padH2D->setTensorBuffer(padData, sizeof(padData)/sizeof(padData[0]), syn_type_uint32);
    padH2D->setAsDataTypeMatchData();
    pTensor unpaddedIFM = createTensor(fmSize, syn_type_float);
    pTensor paddedIFM   = createTensor(paddedFmSize, syn_type_float);

    ns_PadKernelEx::Params padParams {};
    padParams.mode    = PAD_MODE_CONSTANT;
    padParams.value.f = 0.f;
    pNode padNode =
        NodeFactory::createNode({unpaddedIFM, padH2D}, {paddedIFM}, &padParams, "pad_fwd_f32", "PAD");
    ASSERT_TRUE(GraphEditor::addNode(m_graph, padNode));

    ASSERT_TRUE(gaudi::loadTpcKernels(m_graph));

    // Create operands for trivial slicing
    std::vector<pSlicedOperand> inputOperands;
    std::vector<pSlicedOperand> outputOperands;
    for (const auto& in : padNode->getInputs())
    {
        inputOperands.push_back(std::make_shared<SlicedOperand>(in));
    }
    for (const auto& out : padNode->getOutputs())
    {
        outputOperands.push_back(std::make_shared<SlicedOperand>(out));
    }

    pBackwardSliceMapping mapping = AccessPatternSliceMapper::createBwdMapping(padNode, inputOperands, outputOperands);
    ASSERT_NE(mapping, nullptr);

    unsigned nofSlices = 0;
    for (auto outputSlice :
         SlicedOperandTraversalPattern(outputOperands.front(), SlicedOperandTraversalPattern::LEFT_TO_RIGHT_4D))
    {
        const SliceReferenceList& inputs  = mapping->getInputs(outputSlice);
        const SliceReferenceList& outputs = mapping->getOutputs(outputSlice.first);

        ASSERT_EQ(inputs.size(), padNode->getInputs().size());
        ASSERT_EQ(outputs.size(), padNode->getOutputs().size());

        CoordArray sliceCoord;
        sliceCoord.fill(0);

        for (const auto& inSlice : inputs)
        {
            ASSERT_EQ(inSlice->coordinates, sliceCoord);
        }

        for (const auto& outSlice : outputs)
        {
            ASSERT_EQ(outSlice->coordinates, sliceCoord);
        }

        nofSlices++;
    }
    ASSERT_EQ(nofSlices, 1);
}