
#include "habana_graph.h"
#include "log_manager.h"
#include "passes.h"
#include "node_factory.h"
#include "graph_editor.h"
#include "graph_visualization.h"
#include "perf_lib_layer_params.h"
#include "layout.h"
#include "ragged_softmax_fuser.h"
#include "data_type_utils.h"

/* Input swapping variation for binary nodes in the Gelu pattern */
typedef enum {
    SWAP_BATCH_GEMM_MULT_INPUTS = 0,
    SWAP_SUB_GEMM_MULT_INPUTS,
    SWAP_ADD_INPUTS,
    NUM_INPUT_SWAPS_VARIATIONS /* Keep last */
}InputSwapsVariations;

bool RaggedSoftmaxFuser::constructRaggedSoftmaxPattern(Graph* pattern, unsigned int var)
{
    // todo SW-14510: check inputs order in symmetrical opeartors.

    bool patternStatus = true;

    /* right branch - add input port 0 */
    pTensor batchGemmIn1      = std::make_shared<Tensor>();
    pTensor batchGemmIn2      = std::make_shared<Tensor>();
    pTensor batchGemmOut      = std::make_shared<Tensor>();
    pTensor batchGemmMulOut   = std::make_shared<Tensor>();
    pTensor batchGemmMul2ndIn = std::make_shared<Tensor>();

    pNode batchGemmNode = NodeFactory::createNode({batchGemmIn1, batchGemmIn2}, {batchGemmOut}, nullptr,
                                                  NodeFactory::batchGemmNodeTypeName, "batch_gemm");
    patternStatus = patternStatus && pattern->addNode(batchGemmNode);
    pNode batchGemmMulNode;
    if ((var & (1 << SWAP_BATCH_GEMM_MULT_INPUTS)) == 0)
    {
        batchGemmMulNode = NodeFactory::createNode({batchGemmOut, batchGemmMul2ndIn}, {batchGemmMulOut}, nullptr,
                                                     "mult", "batch_gemm_mult");
    }
    else
    {
        batchGemmMulNode = NodeFactory::createNode({batchGemmMul2ndIn, batchGemmOut}, {batchGemmMulOut}, nullptr,
                                                     "mult", "batch_gemm_mult");
    }

    patternStatus = patternStatus && pattern->addNode(batchGemmMulNode);

    /* left branch add input port 1 */
    pTensor inputMask        = std::make_shared<Tensor>();
    pTensor inputMaskReshape = std::make_shared<Tensor>();
    pTensor subOut           = std::make_shared<Tensor>();
    pTensor sub1stInput      = std::make_shared<Tensor>();
    pTensor subMulOut        = std::make_shared<Tensor>();
    pTensor subMul2ndIn      = std::make_shared<Tensor>();

    pNode reshape1Node = NodeFactory::createNode({inputMask}, {inputMaskReshape}, nullptr,
                                                 NodeFactory::reshapeNodeTypeName, "reshape1");
    patternStatus = patternStatus && pattern->addNode(reshape1Node);

    pNode subNode = NodeFactory::createNode({sub1stInput, inputMaskReshape}, {subOut}, nullptr,
                                            "sub", "sub");
    patternStatus = patternStatus && pattern->addNode(subNode);
    pNode mulNode;
    if ((var & (1 << SWAP_SUB_GEMM_MULT_INPUTS)) == 0)
    {
        mulNode = NodeFactory::createNode({subOut, subMul2ndIn}, {subMulOut}, nullptr,
                                            "mult", "sub_mul");
    }
    else
    {
        mulNode = NodeFactory::createNode({subMul2ndIn, subOut}, {subMulOut}, nullptr,
                                            "mult", "sub_mul");
    }

    patternStatus = patternStatus && pattern->addNode(mulNode);

    /* main branch */
    pTensor addOut            = std::make_shared<Tensor>();
    pTensor flattenOut        = std::make_shared<Tensor>();
    pTensor softmaxOut        = std::make_shared<Tensor>();
    pTensor softmaxReshapeOut = std::make_shared<Tensor>();
    pNode addNode;

    if ((var & (1 << SWAP_ADD_INPUTS)) == 0)
    {
        addNode = NodeFactory::createNode({batchGemmMulOut, subMulOut}, {addOut}, nullptr,
                                            "add", "add");
    }
    else
    {
        addNode = NodeFactory::createNode({subMulOut, batchGemmMulOut}, {addOut}, nullptr,
                                            "add", "add");
    }

    patternStatus = patternStatus && pattern->addNode(addNode);

    synFlattenParams flattenParams = {1};
    pNode flattenNode = NodeFactory::createNode({addOut}, {flattenOut}, &flattenParams,
                                                NodeFactory::flattenNodeTypeName, "flatten");
    patternStatus = patternStatus && pattern->addNode(flattenNode);

    pNode softmaxNode = NodeFactory::createNode({flattenOut}, {softmaxOut}, nullptr,
                                                "softmax_f32", "softmax");
    patternStatus = patternStatus && pattern->addNode(softmaxNode);

    pNode reshape2Node = NodeFactory::createNode({softmaxOut}, {softmaxReshapeOut}, nullptr,
                                                 NodeFactory::reshapeNodeTypeName, "reshape2");
    patternStatus = patternStatus && pattern->addNode(reshape2Node);

    return patternStatus;
}

bool RaggedSoftmaxFuser::validatePattern(pNode               reshape2Node,
                                         NodeList&           patternNodes,
                                         pTensor&            tensorToNorm,
                                         pTensor&            normalizedTensor,
                                         ns_Softmax::Params& softmaxParams,
                                         pNode&              inputMaskMulNode,
                                         pNode&              inputMaskCastNode,
                                         pNode&              inputMaskReshapeNode,
                                         unsigned int        var)
{
    const unsigned EXPECTED_SUB_NODE_SCALAR_VAL     = 1;
    float          EXPECTED_MAX_MUL_NODE_SCALAR_VAL = GCFG_RAGGED_SOFTMAX_OPT_AMP_VAL.value();

    if (reshape2Node == nullptr || reshape2Node->getNodeType() != Node::TYPE_INTERNAL_RESHAPE) return false;
    patternNodes.push_back(reshape2Node);

    pTensor reshape2Output = reshape2Node->getOutput(0);
    if (reshape2Output == nullptr) return false;
    normalizedTensor = reshape2Output;

    pNode softmaxNode = m_graph.getTensorProducer(reshape2Node->getInput(0));
    if (softmaxNode == nullptr) return false;
    std::shared_ptr<TPCNode> tpcNode = std::dynamic_pointer_cast<TPCNode>(softmaxNode);
    if (tpcNode == nullptr || tpcNode->getGUIDWithoutDtype() != "softmax") return false;
    softmaxParams = *(ns_Softmax::Params*)(tpcNode->getParams());
    if (softmaxParams.dim != 0) return false;
    patternNodes.push_back(softmaxNode);

    pNode flattenNode = m_graph.getTensorProducer(softmaxNode->getInput(0));
    if (flattenNode == nullptr || flattenNode->getNodeType() != Node::TYPE_INTERNAL_FLATTEN) return false;
    patternNodes.push_back(flattenNode);

    unsigned int addInput0Index = ((var == 0) ? 0:1);
    unsigned int addInput1Index = ((var == 1) ? 0:1);

    pNode addNode = m_graph.getTensorProducer(flattenNode->getInput(addInput0Index));
    if (addNode == nullptr) return false;
    tensorToNorm = addNode->getInput(addInput0Index);
    tpcNode = std::dynamic_pointer_cast<TPCNode>(addNode);
    if (tpcNode == nullptr || tpcNode->getGUIDWithoutDtype() != "add") return false;
    patternNodes.push_back(addNode);

    pNode batchGemmMulNode = ((var & (1 << SWAP_ADD_INPUTS)) == 0) ? m_graph.getTensorProducer(addNode->getInput(0))
                                                            : m_graph.getTensorProducer(addNode->getInput(1));

    tensorToNorm = batchGemmMulNode->getOutput(0);
    if (batchGemmMulNode == nullptr) return false;
    tpcNode = std::dynamic_pointer_cast<TPCNode>(batchGemmMulNode);
    if (tpcNode == nullptr || tpcNode->getGUIDWithoutDtype() != "mult") return false;

    pNode batchGemmNode = ((var & (1 << SWAP_BATCH_GEMM_MULT_INPUTS)) == 0)
                              ? m_graph.getTensorProducer(batchGemmMulNode->getInput(0))
                              : m_graph.getTensorProducer(batchGemmMulNode->getInput(1));

    if (batchGemmNode == nullptr || batchGemmNode->getNodeType() != Node::TYPE_BATCH_GEMM) return false;

    float scalarVal;

    pNode subMulNode = m_graph.getTensorProducer(addNode->getInput(addInput1Index));
    if (subMulNode == nullptr) return false;
    tpcNode = std::dynamic_pointer_cast<TPCNode>(subMulNode);
    if (tpcNode == nullptr || tpcNode->getGUIDWithoutDtype() != "mult") return false;
    int subInputIndex = ((var & (1 << SWAP_SUB_GEMM_MULT_INPUTS)) == 0) ? 0 : 1;
    if (!extractScalarFromStaticTensor(subMulNode->getInput((subInputIndex+1)%2), scalarVal)) return false;
    if (scalarVal > EXPECTED_MAX_MUL_NODE_SCALAR_VAL) return false;
    patternNodes.push_back(subMulNode);

    pNode subNode = m_graph.getTensorProducer(subMulNode->getInput(subInputIndex));
    if (subNode == nullptr) return false;
    tpcNode = std::dynamic_pointer_cast<TPCNode>(subNode);
    if (tpcNode == nullptr || tpcNode->getGUIDWithoutDtype() != "sub") return false;
    if (!extractScalarFromStaticTensor(subNode->getInput(0), scalarVal)) return false;
    if (scalarVal != EXPECTED_SUB_NODE_SCALAR_VAL) return false;
    patternNodes.push_back(subNode);

    pNode reshape1Node = m_graph.getTensorProducer(subNode->getInput(1));
    if (reshape1Node == nullptr || reshape1Node->getNodeType() != Node::TYPE_INTERNAL_RESHAPE) return false;
    patternNodes.push_back(reshape1Node);

    inputMaskMulNode = m_graph.getTensorProducer(reshape1Node->getInput(0));
    tpcNode = std::dynamic_pointer_cast<TPCNode>(inputMaskMulNode);
    if (tpcNode == nullptr || tpcNode->getGUIDWithoutDtype() != "mult") return false;

    inputMaskReshapeNode = m_graph.getTensorProducer(inputMaskMulNode->getInput(1));
    if (inputMaskReshapeNode == nullptr) return false;

    if (inputMaskReshapeNode->getNodeType() != Node::TYPE_INTERNAL_RESHAPE)
    {
        if (!(inputMaskReshapeNode->getNodeType() == Node::TYPE_USER &&
              inputMaskReshapeNode->getGUID() == "cast_to_f32"))
            return false;

        inputMaskCastNode    = inputMaskReshapeNode;
        inputMaskReshapeNode = m_graph.getTensorProducer(inputMaskCastNode->getInput(0));
        if (inputMaskReshapeNode == nullptr || inputMaskReshapeNode->getNodeType() != Node::TYPE_INTERNAL_RESHAPE)
        {
            return false;
        }
    }

    return true;
}

void RaggedSoftmaxFuser::transposeRaggedSoftmaxTensors(pTensor& tensorToNorm,
                                                       pTensor& normalizedTensor,
                                                       unsigned numFusions,
                                                       NodeList& newNodes)
{

    // need to tranpose between dim0 and dim3:
    gc::Permutation perm({1,2,3,0});

    synTransposeParams transposeParams;
    perm.getValues(transposeParams.permutation, MAX_DIMENSIONS_NUM);
    transposeParams.tensorDim = 4;

    pTensor transposedInput = tensorToNorm->clone();
    transposedInput->setName(tensorToNorm->getName() + "_transposed");
    TSize newSizes[Tensor::c_tensorMaxDim];
    transposedInput->getAllSizesInElements(newSizes, Tensor::c_tensorMaxDim);
    perm.permuteShape<TSize>(newSizes, transposedInput->getDim());
    transposedInput->reshape(4U, newSizes, nullptr);

    pNode trBefore = NodeFactory::createNode({tensorToNorm},
                                            {transposedInput},
                                            &transposeParams,
                                            "transpose",
                                            "ragged_softmax_transpose_in" + std::to_string(numFusions));

    tensorToNorm = transposedInput;

    gc::Permutation invPerm(perm.getInversePermutation());
    synTransposeParams softmaxOutputTransposeParams;
    invPerm.getValues(softmaxOutputTransposeParams.permutation, MAX_DIMENSIONS_NUM);
    softmaxOutputTransposeParams.tensorDim = 4;

    pTensor  transposedOutput = transposedInput->clone();
    transposedOutput->setName(normalizedTensor->getName() + "_transposed");
    transposedOutput->setAllQuantizationParams(normalizedTensor->getAllQuantizationParams());
    transposedOutput->setDynamicRange(normalizedTensor->getDynamicRange());
    pNode trAfter = NodeFactory::createNode({transposedOutput},
                                            {normalizedTensor},
                                            &softmaxOutputTransposeParams,
                                            "transpose",
                                            "ragged_softmax_transpose_out" + std::to_string(numFusions));

    normalizedTensor = transposedOutput;

    newNodes.push_back(trBefore);
    newNodes.push_back(trAfter);
}

void RaggedSoftmaxFuser::createSeqLenTensor(NodeList& newNodes, pTensor inputMaskTensor, pTensor& seqLenTensor)
{

    TSize seqLenSizes[] = {inputMaskTensor->getSizeInElements(1)};
    TSize seqLen2dSizes[] = {1, inputMaskTensor->getSizeInElements(1)};

    synQuantizationParams seqLenTensorQuant;
    seqLenTensorQuant.m_scale     = 1;
    seqLenTensorQuant.m_zp        = 0;
    seqLenTensorQuant.m_qDataType = syn_type_int16;

    pTensor seqLenTensor2d = std::make_shared<Tensor>(2U, seqLen2dSizes, syn_type_int16);
    seqLenTensor2d->setName("sequence_len_2d");
    seqLenTensor2d->setQuantizationParams(seqLenTensorQuant);

    seqLenTensor = std::make_shared<Tensor>(1U, seqLenSizes, syn_type_int16);
    seqLenTensor->setName("sequence_len");
    seqLenTensor->setQuantizationParams(seqLenTensorQuant);

    ns_Reduction::Params params = {0};
    pNode reduceSumNode = NodeFactory::createNode({inputMaskTensor}, {seqLenTensor2d}, &params,
                                                "reduce_sum_i16", "reduce_sum_input_mask");


    if (!changeTensorElementTypeSafe(inputMaskTensor, syn_type_int16))
    {
        LOG_WARN(GC, "failed changing tensor {} dtype to syn_type_int16", inputMaskTensor->getName());
    }

    pNode squeezeSeqLenNode = NodeFactory::createNode({seqLenTensor2d},
                                                    {seqLenTensor},
                                                    nullptr,
                                                    NodeFactory::reshapeNodeTypeName,
                                                    "squeeze_seq_len");

    newNodes.push_back(reduceSumNode);
    newNodes.push_back(squeezeSeqLenNode);
}

bool RaggedSoftmaxFuser::fuseRaggedSoftmax(Graph* pattern)
{
    for (unsigned var = 0; var < (1 << NUM_INPUT_SWAPS_VARIATIONS); var++)
    {
        pattern->clear();
        if (!constructRaggedSoftmaxPattern(pattern, var))
        {
            LOG_ERR(GC, "failed to create ragged softmax pattern.");
            return false;
        }

        // find all matches for above patterns
        NodeSet matchingNodes = m_graph.matchPatternWithSingleOutputNode(pattern, NodeTypeMatchingFunc);
        unsigned fusions = 0;

        pTensor seqLenTensor;
        LOG_DEBUG(GC, "found {} patterns", matchingNodes.size());
        for (pNode lastNode : matchingNodes)
        {
            LOG_DEBUG(GC, "Checking pattern ending at node: {}", lastNode->getNodeName());
            pNode              reshape2Node = lastNode;
            NodeList           patternNodes;
            pTensor            tensorToNorm;
            pTensor            normalizedTensor;
            ns_Softmax::Params softmaxParams;
            pNode              inputMaskMulNode;
            pNode              inputMaskCastNode;
            pNode              inputMaskReshapeNode;

            if (!validatePattern(reshape2Node,
                                 patternNodes,
                                 tensorToNorm,
                                 normalizedTensor,
                                 softmaxParams,
                                 inputMaskMulNode,
                                 inputMaskCastNode,
                                 inputMaskReshapeNode,
                                 var))
            {
                LOG_WARN(GC, "Pattern failed");
                continue;
            }

            LOG_DEBUG(GC, "pattern verified");
            NodeList newNodes;

            // if it is the first fusion the input mask tensor need to be converted into seqLen tensor
            // it is being done using reduceSum that counts the 1's in the tensor.
            if (fusions == 0)
            {
                pTensor inputMaskTensor = inputMaskReshapeNode->getInput(0);
                createSeqLenTensor(newNodes, inputMaskTensor, seqLenTensor);
                if (seqLenTensor == nullptr)
                {
                    LOG_ERR(GC, "Failed creating seqLenTensor");
                    return false;
                }
            }

            // ragged softmax performs better on dim=3. so we need to transpose
            HB_ASSERT(softmaxParams.dim == 0, "dim must be 0");
            transposeRaggedSoftmaxTensors(tensorToNorm, normalizedTensor, fusions, newNodes);

            ns_Softmax::Params raggedSoftmaxParams = {3};

            std::string kernelName = "ragged_softmax";
            if (normalizedTensor->getElementType() != syn_type_na)
            {
                kernelName =
                    fmt::format("{}_{}", kernelName, getDtypeSuffixFromSynDataType(normalizedTensor->getElementType()));
            }

            pNode raggedSoftmax = NodeFactory::createNode({tensorToNorm, seqLenTensor},
                                                          {normalizedTensor},
                                                          &raggedSoftmaxParams,
                                                          kernelName,
                                                          fmt::format("ragged_softmax{}", fusions));
            newNodes.push_back(raggedSoftmax);

            /* Replacing the nodes: */
            if (GraphEditor::replaceNodes(m_graph, patternNodes, newNodes) != REPLACE_NODE_SUCCESS)
            {
                LOG_ERR(GC, "failed to replace ragged softmax pattern");
                return false;
            }
            else
            {
                LOG_DEBUG(GC, "fused pattern to ragged softmax node {}", raggedSoftmax->getNodeName());
            }

            fusions++;

            // in case no other node consumes those nodes outputs, they can be removed from the graph.
            if (inputMaskMulNode != nullptr && m_graph.getNumberOfTensorConsumers(inputMaskMulNode->getOutput(0)) == 0)
            {
                GraphEditor::removeNode(m_graph, inputMaskMulNode);
                LOG_DEBUG(GC, "Removed unused node: {}", inputMaskMulNode->getNodeName());
                if (inputMaskCastNode != nullptr)
                {
                    if (m_graph.getTensorConsumers(inputMaskCastNode->getOutput(0)).size() == 0)
                    {
                        GraphEditor::removeNode(m_graph, inputMaskCastNode);
                        LOG_DEBUG(GC, "Removed unused node: {}", inputMaskCastNode->getNodeName());
                        if (inputMaskReshapeNode != nullptr &&
                            m_graph.getNumberOfTensorConsumers(inputMaskReshapeNode->getOutput(0)) == 0)
                        {
                            GraphEditor::removeNode(m_graph, inputMaskReshapeNode);
                            LOG_DEBUG(GC, "Removed unused node: {}", inputMaskReshapeNode->getNodeName());
                        }
                    }
                }
                else if (inputMaskReshapeNode != nullptr &&
                         m_graph.getTensorConsumers(inputMaskReshapeNode->getOutput(0)).size() == 0)
                {
                    GraphEditor::removeNode(m_graph, inputMaskReshapeNode);
                    LOG_DEBUG(GC, "Removed unused node: {}", inputMaskReshapeNode->getNodeName());
                }
            }
        }
        LOG_DEBUG(GC, "completed {} fusions", fusions);
    }
    return true;
}