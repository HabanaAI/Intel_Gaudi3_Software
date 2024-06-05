#include "graph_editor.h"
#include "habana_graph.h"
#include "hal_reader/hal_reader.h"
#include "node_factory.h"

class BatchGemmConverter
{
public:
    void convert(HabanaGraph& graph, const pNode& batchGemm)
    {
        m_baseName = batchGemm->getNodeName();
        createOperandAReplaceNodes(batchGemm->getInput(TENSOR_IFM));
        createOperandBReplaceNodes(batchGemm->getInput(TENSOR_WEIGHT));
        createOperandCReplaceNodes(batchGemm->getOutput(TENSOR_OFM));
        createGemmOperation();
        auto status = GraphEditor::replaceNodes(graph, {batchGemm}, m_convertNodes);
        HB_ASSERT(status == REPLACE_NODE_SUCCESS,
                  "{}: failed to convert batch gemm node {} to gemm",
                  HLLOG_FUNC,
                  batchGemm->getNodeName());
    }

private:
    void createOperandAReplaceNodes(const pTensor& operandA)
    {
        // Reshape - From C,W,B... to C,W
        m_gemmOperandA = createCombineReshapeNode(operandA, DIM_W, Tensor::c_tensorMaxDim, "operand_a");
    }

    void createOperandBReplaceNodes(const pTensor& operandB)
    {
        pTensor nextInput = operandB;

        if (operandB->getDim() > DIM_GEMM_BATCH + 1)
        {
            // Reshape to combine all batch size, move from K,C,B1,B2... to K,C,B
            nextInput = createCombineReshapeNode(nextInput, DIM_GEMM_BATCH, Tensor::c_tensorMaxDim, "batch_b");
        }

        // Add transpose node, move from K,C,B to K,B,C
        nextInput = createTransposeNode(nextInput, "operand_b");

        // Reshape - combine K and B, move from K,B,C to KB, C
        m_gemmOperandB = createCombineReshapeNode(nextInput, WEIGHT_DIM_K, WEIGHT_DIM_C + 1, "batch_b");
    }

    void createGemmOperation()
    {
        synGEMMParams params;

        pNode gemmNode = NodeFactory::createNode({m_gemmOperandA, m_gemmOperandB},
                                                 {m_gemmOperandC},
                                                 &params,
                                                 NodeFactory::gemmNodeTypeName,
                                                 fmt::format("{}_gemm", m_baseName));
        m_convertNodes.push_back(gemmNode);
    }

    void createOperandCReplaceNodes(const pTensor& operandC)
    {
        //Create the operand C of the gemm
        m_gemmOperandC = cloneTensor(operandC);
        SizeArray outSize = m_gemmOperandC->getAllSizesInElements();
        outSize[DIM_C] = m_gemmOperandB->getSizeInElements(WEIGHT_DIM_K);
        outSize[DIM_W] = m_gemmOperandA->getSizeInElements(DIM_W);
        m_gemmOperandC->reshape(m_gemmOperandA->getDim(), outSize.data(), nullptr);

        SizeArray outputSize = operandC->getAllSizesInElements();
        uint32_t batch = multiplyElements(outputSize.begin() + DIM_GEMM_BATCH, outputSize.end());

        // Create reshape from C',W' to C,B,W,B
        SizeArray nextInputSize = m_gemmOperandC->getAllSizesInElements();
        nextInputSize[DIM_B] = batch;
        nextInputSize[DIM_H] = nextInputSize[DIM_W] / batch;
        nextInputSize[DIM_W] = batch;
        nextInputSize[DIM_C] /= batch;

        pTensor nextInput = createReshapeNode(m_gemmOperandC, nextInputSize, m_gemmOperandC->getDim() * 2, "gemm");

        // Add transpose node, move from C,B,W,B to C,W,B,B
        nextInput = createTransposeNode(nextInput, "operand_c");

        // Add reshape node, move from C,W,B,B to C,W,BB
        nextInput = createCombineReshapeNode(nextInput, DIM_GEMM_BATCH, Tensor::c_tensorMaxDim, "operand_c");

        nextInputSize = nextInput->getAllSizesInElements();
        // Add slice to reach C,W,B from C,W,BB
        SliceNode::SliceNodeStaticParams sliceParams = {};
        uint32_t dim = 0;
        std::fill(sliceParams.starts.begin(), sliceParams.starts.begin() + MAX_DIMENSIONS_NUM, 0);
        dim = 0;
        std::generate(sliceParams.ends.begin(),
                      sliceParams.ends.begin() + MAX_DIMENSIONS_NUM,
                      [&dim, &nextInputSize]() { return nextInputSize[dim++]; });
        std::fill(sliceParams.steps.begin(), sliceParams.steps.begin() + MAX_DIMENSIONS_NUM, 1);
        sliceParams.steps[DIM_GEMM_BATCH] = batch + 1;
        nextInputSize[DIM_GEMM_BATCH] = batch;

        pTensor sliceOutput = operandC;
        bool batchIsSplit = nextInputSize != operandC->getAllSizesInElements();
        if (batchIsSplit)
        {
            sliceOutput = cloneTensor(nextInput);
            sliceOutput->reshape(nextInput->getDim(), nextInputSize.data(), nullptr);
        }
        pNode sliceNode = NodeFactory::createNode({nextInput},
                                                  {sliceOutput},
                                                  &sliceParams,
                                                  NodeFactory::sliceNodeTypeName,
                                                  fmt::format("{}_slice_operand_c", m_baseName));
        m_convertNodes.push_back(sliceNode);

        if (batchIsSplit)
        {
            // Reshape to the original size
            pNode reshapeNode = NodeFactory::createNode({sliceOutput},
                                                        {operandC},
                                                        nullptr,
                                                        NodeFactory::reshapeNodeTypeName,
                                                        fmt::format("{}_reshape_final_operand_c", m_baseName));
            m_convertNodes.push_back(reshapeNode);
        }
    }

    pTensor createCombineReshapeNode(pTensor input, uint32_t startDim, uint32_t endDim, const std::string& name)
    {
        SizeArray nextInputSize = input->getAllSizesInElements();
        uint32_t newDimNum = input->getDim() - (std::min(endDim, input->getDim()) - startDim - 1);
        nextInputSize[startDim] = multiplyElements(nextInputSize.begin() + startDim, nextInputSize.begin() + endDim);
        for (--endDim; endDim < newDimNum; ++endDim)
        {
            nextInputSize[endDim] = nextInputSize[endDim + 1];
        }
        std::fill(nextInputSize.begin() + endDim, nextInputSize.end(), 1);

        return createReshapeNode(input, nextInputSize, newDimNum, name);
    }

    pTensor createReshapeNode(pTensor input, SizeArray outputShape, uint32_t outputDim, const std::string& name)
    {
        pTensor reshapeOut = cloneTensor(input);
        reshapeOut->reshape(outputDim, outputShape.data(), nullptr);
        reshapeOut->setName(fmt::format("{}_reshpe_{}_output", m_baseName, name));
        pNode reshapeNode = NodeFactory::createNode({input},
                                                    {reshapeOut},
                                                    nullptr,
                                                    NodeFactory::reshapeNodeTypeName,
                                                    fmt::format("{}_reshape_{}", m_baseName, name));
        m_convertNodes.push_back(reshapeNode);

        return reshapeOut;
    }

    TensorPtr createTransposeNode(TensorPtr input, std::string_view name)
    {
        pTensor   transposeOut  = cloneTensor(input);
        SizeArray nextInputSize = input->getAllSizesInElements();
        std::swap(nextInputSize[DIM_GEMM_BATCH], nextInputSize[DIM_W]);
        transposeOut->reshape(input->getDim(), nextInputSize.data(), nullptr);
        transposeOut->setName(fmt::format("{}_transpose_{}_output", m_baseName, name));
        synTransposeParams transposeParams;
        transposeParams.tensorDim      = transposeOut->getDim();
        transposeParams.permutation[0] = TPD_Channel;
        transposeParams.permutation[1] = TPD_Height;
        transposeParams.permutation[2] = TPD_Width;
        for (uint32_t dimIdx = 3; dimIdx < input->getDim(); ++dimIdx)
        {
            transposeParams.permutation[dimIdx] = (TransposePermutationDim)dimIdx;
        }

        pNode transposeNode = NodeFactory::createNode({input},
                                                      {transposeOut},
                                                      &transposeParams,
                                                      NodeFactory::transposeNodeTypeName,
                                                      fmt::format("{}_transpose_{}", m_baseName, name));
        m_convertNodes.push_back(transposeNode);
        return transposeOut;
    }

    pTensor cloneTensor(const pTensor& tensor)
    {
        pTensor cloned = tensor->clone();
        cloned->maskOutput();
        return cloned;
    }


    std::string m_baseName;
    pTensor m_gemmOperandA;
    pTensor m_gemmOperandB;
    pTensor m_gemmOperandC;
    NodeList m_convertNodes;
};

static bool shouldConvertBatchGemmToGemm(const pNode& batchGemm, const HabanaGraph& graph)
{
    {
        // TODO [SW-7607] - Batch gemm conversion support for transpose a and transpose b
        synGEMMParams gemmParams = static_cast<BatchGemmNode*>(batchGemm.get())->getGEMMParams();
        if (gemmParams.transpose_a || gemmParams.transpose_b)
        {
            return false;
        }
    }

    const pTensor& operandA = batchGemm->getInput(TENSOR_IFM);
    const pTensor& operandB = batchGemm->getInput(TENSOR_WEIGHT);
    const pTensor& operandC = batchGemm->getOutput(TENSOR_OFM);
    if (! operandA->isDenseLayout() ||
        ! operandB->isDenseLayout() ||
        ! operandC->isDenseLayout())
    {
        // Strided tensor as input or output may cause memcpy node
        // which will casuse a performance degradation
        return false;
    }

    if (operandC->getDim() > DIM_GEMM_BATCH + 1)
    {
        // If the batch is divided on several dimensions it may cause a memcpy
        // which will cause a performance degradation
        return false;
    }

    const SizeArray& operandCSize = operandC->getAllSizesInElements();
    uint32_t elementSizeInBytes = operandC->getElementSizeInBytes();
    uint32_t batch = multiplyElements(operandCSize.begin() + DIM_GEMM_BATCH, operandCSize.end());

    return (batch > 1 &&
            (operandCSize[DIM_C] * elementSizeInBytes < graph.getHALReader()->getMmeVectorSize() ||
             operandCSize[DIM_W] * elementSizeInBytes < graph.getHALReader()->getMmeVectorSize()));
}

bool convertBatchGemmToGemm(HabanaGraph& g)
{
    NodeVector allNodes = g.getExeSortedNodes();
    for (auto node : allNodes)
    {
        if (node->getNodeType() == Node::TYPE_BATCH_GEMM && shouldConvertBatchGemmToGemm(node, g))
        {
            BatchGemmConverter converter;
            converter.convert(g, node);
        }
    }
    return true;
}
