
#include "habana_graph.h"
#include "log_manager.h"
#include "passes.h"
#include "node_factory.h"
#include "graph_editor.h"
#include "graph_visualization.h"
#include "perf_lib_layer_params.h"
#include "layout.h"
#include "ragged_softmax_fuser_v2.h"
#include "data_type_utils.h"
#include "slice_node.h"
#include "synapse_common_types.h"
#include "types.h"

/* Input swapping variation for binary nodes in the Gelu pattern */
typedef enum {
    SWAP_BATCH_GEMM_MULT_INPUTS = 0,
    SWAP_SUB_GEMM_MULT_INPUTS,
    SWAP_ADD_INPUTS,
    NUM_INPUT_SWAPS_VARIATIONS /* Keep last */
}InputSwapsVariations;

bool RaggedSoftmaxFuserV2::constructRaggedSoftmaxPattern(Graph* pattern, unsigned int var)
{
    // todo SW-14510: check inputs order in symmetrical opeartors.

    bool patternStatus = true;

    /* right branch - add input port 0 */
    TensorPtr batchGemmIn1      = std::make_shared<Tensor>();
    TensorPtr batchGemmIn2      = std::make_shared<Tensor>();
    TensorPtr batchGemmOut      = std::make_shared<Tensor>();
    TensorPtr batchGemmMulOut   = std::make_shared<Tensor>();
    TensorPtr batchGemmMul2ndIn = std::make_shared<Tensor>();
    TensorPtr castBatchgemmOut  = std::make_shared<Tensor>();

    NodePtr batchGemmNode = NodeFactory::createNode({batchGemmIn1, batchGemmIn2}, {batchGemmOut}, nullptr,
                                                    NodeFactory::batchGemmNodeTypeName, "batch_gemm");
    patternStatus = patternStatus && pattern->addNode(batchGemmNode);

    NodePtr castBGNode = NodeFactory::createNode({batchGemmOut}, {castBatchgemmOut}, nullptr,
                                                 "cast", "cast_batch_gemm_output");
    patternStatus = patternStatus && pattern->addNode(castBGNode);

    NodePtr batchGemmDivNode;
    if ((var & (1 << SWAP_BATCH_GEMM_MULT_INPUTS)) == 0)
    {
        batchGemmDivNode = NodeFactory::createNode({castBatchgemmOut, batchGemmMul2ndIn}, {batchGemmMulOut}, nullptr,
                                                   "div", "batch_gemm_div");
    }
    else
    {
        batchGemmDivNode = NodeFactory::createNode({batchGemmMul2ndIn, castBatchgemmOut}, {batchGemmMulOut}, nullptr,
                                                   "div", "batch_gemm_div");
    }

    patternStatus = patternStatus && pattern->addNode(batchGemmDivNode);

    /* left branch add input port 1 */
    TensorPtr inputMask        = std::make_shared<Tensor>();
    TensorPtr inputMaskReshape = std::make_shared<Tensor>();
    TensorPtr subOut           = std::make_shared<Tensor>();
    TensorPtr sub1stInput      = std::make_shared<Tensor>();
    TensorPtr subMulOut        = std::make_shared<Tensor>();
    TensorPtr subMul2ndIn      = std::make_shared<Tensor>();
    TensorPtr cast1Out         = std::make_shared<Tensor>();
    TensorPtr cast2Out         = std::make_shared<Tensor>();
    TensorPtr cast3Out         = std::make_shared<Tensor>();

    NodePtr subNode = NodeFactory::createNode({sub1stInput, inputMaskReshape}, {subOut}, nullptr,
                                              "sub", "sub");
    patternStatus = patternStatus && pattern->addNode(subNode);
    NodePtr cast1Node = NodeFactory::createNode({subOut}, {cast1Out}, nullptr,
                                                "cast", "cast1");
    patternStatus = patternStatus && pattern->addNode(cast1Node);
    NodePtr mulNode;
    if ((var & (1 << SWAP_SUB_GEMM_MULT_INPUTS)) == 0)
    {
        mulNode = NodeFactory::createNode({cast1Out, subMul2ndIn}, {subMulOut}, nullptr,
                                          "mult", "sub_mul");
    }
    else
    {
        mulNode = NodeFactory::createNode({subMul2ndIn, cast1Out}, {subMulOut}, nullptr,
                                          "mult", "sub_mul");
    }

    patternStatus = patternStatus && pattern->addNode(mulNode);
    NodePtr cast2Node = NodeFactory::createNode({subMulOut}, {cast2Out}, nullptr,
                                                "cast", "cast2");
    patternStatus = patternStatus && pattern->addNode(cast2Node);

    /* main branch */
    TensorPtr addOut            = std::make_shared<Tensor>();
    TensorPtr softmaxOut        = std::make_shared<Tensor>();
    TensorPtr softmaxReshapeOut = std::make_shared<Tensor>();
    NodePtr addNode;

    if ((var & (1 << SWAP_ADD_INPUTS)) == 0)
    {
        addNode = NodeFactory::createNode({batchGemmMulOut, cast2Out}, {addOut}, nullptr,
                                          "add", "add");
    }
    else
    {
        addNode = NodeFactory::createNode({cast2Out, batchGemmMulOut}, {addOut}, nullptr,
                                          "add", "add");
    }

    patternStatus = patternStatus && pattern->addNode(addNode);

    NodePtr cast3Node = NodeFactory::createNode({addOut}, {cast3Out}, nullptr,
                                                "cast", "cast3");
    patternStatus = patternStatus && pattern->addNode(cast3Node);

    NodePtr softmaxNode = NodeFactory::createNode({cast3Out}, {softmaxOut}, nullptr,
                                                  "softmax_f32", "softmax");
    patternStatus = patternStatus && pattern->addNode(softmaxNode);

    NodePtr reshape2Node = NodeFactory::createNode({softmaxOut}, {softmaxReshapeOut}, nullptr,
                                                   NodeFactory::reshapeNodeTypeName, "reshape2");
    patternStatus = patternStatus && pattern->addNode(reshape2Node);

    return patternStatus;
}

bool RaggedSoftmaxFuserV2::validatePattern(NodePtr               reshape2Node,
                                           NodeList&             patternNodes,
                                           NodeList&             commonNodes,
                                           TensorPtr&            tensorToNorm,
                                           TensorPtr&            normalizedTensor,
                                           ns_Softmax::Params&   softmaxParams,
                                           TensorPtr&            inputMaskTensor,
                                           unsigned int          var)
{
    // TODO SW-126023 - uncomment out the following definition when moving mul and sub scalar inputs as static inputs
    // const unsigned EXPECTED_SUB_NODE_SCALAR_VAL     = 1;
    // float          EXPECTED_MAX_MUL_NODE_SCALAR_VAL = GCFG_RAGGED_SOFTMAX_OPT_AMP_VAL.value();

    if (reshape2Node == nullptr || reshape2Node->getNodeType() != Node::TYPE_INTERNAL_RESHAPE) return false;
    patternNodes.push_back(reshape2Node);

    TensorPtr reshape2Output = reshape2Node->getOutput(0);
    if (reshape2Output == nullptr) return false;
    normalizedTensor = reshape2Output;

    NodePtr softmaxNode = m_graph.getTensorProducer(reshape2Node->getInput(0));
    if (softmaxNode == nullptr) return false;
    std::shared_ptr<TPCNode> tpcNode = std::dynamic_pointer_cast<TPCNode>(softmaxNode);
    if (tpcNode == nullptr || tpcNode->getGUIDWithoutDtype() != "softmax") return false;
    softmaxParams = *(ns_Softmax::Params*)(tpcNode->getParams());
    if (softmaxParams.dim != 0) return false;
    patternNodes.push_back(softmaxNode);

    NodePtr addNode = nullptr;
    NodePtr cast1Node = m_graph.getTensorProducer(softmaxNode->getInput(0));
    if (cast1Node == nullptr) return false;
    tpcNode = std::dynamic_pointer_cast<TPCNode>(cast1Node);
    if (tpcNode == nullptr || tpcNode->getGUIDWithoutDtype() != "cast")
    {
        addNode = cast1Node;
    }
    else
    {
        patternNodes.push_back(cast1Node);
        addNode = m_graph.getTensorProducer(cast1Node->getInput(0));
    }

    unsigned int addInput1Index = ((var == 1) ? 0:1);

    if (addNode == nullptr) return false;
    tpcNode = std::dynamic_pointer_cast<TPCNode>(addNode);
    if (tpcNode == nullptr || tpcNode->getGUIDWithoutDtype() != "add") return false;
    patternNodes.push_back(addNode);

    NodePtr batchGemmMulNode = ((var & (1 << SWAP_ADD_INPUTS)) == 0) ? m_graph.getTensorProducer(addNode->getInput(0))
                                                                     : m_graph.getTensorProducer(addNode->getInput(1));

    tensorToNorm = batchGemmMulNode->getOutput(0);
    if (batchGemmMulNode == nullptr) return false;
    tpcNode = std::dynamic_pointer_cast<TPCNode>(batchGemmMulNode);
    if (tpcNode == nullptr || tpcNode->getGUIDWithoutDtype() != "div") return false;

    NodePtr batchGemmNode = nullptr;
    NodePtr cast2Node = ((var & (1 << SWAP_BATCH_GEMM_MULT_INPUTS)) == 0)
                              ? m_graph.getTensorProducer(batchGemmMulNode->getInput(0))
                              : m_graph.getTensorProducer(batchGemmMulNode->getInput(1));
    if (cast2Node == nullptr) return false;
    tpcNode = std::dynamic_pointer_cast<TPCNode>(cast2Node);
    if (tpcNode == nullptr || tpcNode->getGUIDWithoutDtype() != "cast")
    {
        batchGemmNode = cast2Node;
    }
    else
    {
        batchGemmNode = m_graph.getTensorProducer(cast2Node->getInput(0));
    }

    if (batchGemmNode == nullptr || batchGemmNode->getNodeType() != Node::TYPE_BATCH_GEMM) return false;

    ///////////////////////////////////////// ReduceSum (Common) Nodes /////////////////////////////////////////

    // TODO SW-126023 - uncomment out the following definition when moving mul and sub scalar inputs as static inputs
    // float scalarVal;
    NodePtr subMulNode = nullptr;
    NodePtr cast3Node = m_graph.getTensorProducer(addNode->getInput(addInput1Index));
    if (cast3Node == nullptr) return false;
    tpcNode = std::dynamic_pointer_cast<TPCNode>(cast3Node);
    if (tpcNode == nullptr || tpcNode->getGUIDWithoutDtype() != "cast")
    {
        subMulNode = cast3Node;
    }
    else
    {
        commonNodes.push_back(cast3Node);
        subMulNode = m_graph.getTensorProducer(cast3Node->getInput(0));
    }

    if (subMulNode == nullptr) return false;
    tpcNode = std::dynamic_pointer_cast<TPCNode>(subMulNode);
    if (tpcNode == nullptr || tpcNode->getGUIDWithoutDtype() != "mult") return false;
    int subInputIndex = ((var & (1 << SWAP_SUB_GEMM_MULT_INPUTS)) == 0) ? 0 : 1;
    // TODO SW-126023 - set mul scalar input as static tensor and enable the following check
    // if (!extractScalarFromStaticTensor(subMulNode->getInput((subInputIndex+1)%2), scalarVal)) return false;
    // if (scalarVal > EXPECTED_MAX_MUL_NODE_SCALAR_VAL) return false;
    commonNodes.push_back(subMulNode);

    NodePtr subNode = nullptr;
    NodePtr cast4Node = m_graph.getTensorProducer(subMulNode->getInput(subInputIndex));
    if (cast4Node == nullptr) return false;
    tpcNode = std::dynamic_pointer_cast<TPCNode>(cast4Node);
    if (tpcNode == nullptr || tpcNode->getGUIDWithoutDtype() != "cast")
    {
        subNode = cast4Node;
    }
    else
    {
        commonNodes.push_back(cast4Node);
        subNode = m_graph.getTensorProducer(cast4Node->getInput(0));
    }

    if (subNode == nullptr) return false;
    tpcNode = std::dynamic_pointer_cast<TPCNode>(subNode);
    if (tpcNode == nullptr || tpcNode->getGUIDWithoutDtype() != "sub") return false;
    // TODO SW-126023 - set sub scalar input as static tensor and enable the following check
    // if (!extractScalarFromStaticTensor(subNode->getInput(0), scalarVal)) return false;
    // if (scalarVal != EXPECTED_SUB_NODE_SCALAR_VAL) return false;
    commonNodes.push_back(subNode);

    NodePtr slice1Node = nullptr;
    NodePtr cast5Node = m_graph.getTensorProducer(subNode->getInput(1));
    if (cast5Node == nullptr) return false;
    tpcNode = std::dynamic_pointer_cast<TPCNode>(cast5Node);
    if (tpcNode == nullptr || tpcNode->getGUIDWithoutDtype() != "cast")
    {
        slice1Node = cast5Node;
    }
    else
    {
        commonNodes.push_back(cast5Node);
        slice1Node = m_graph.getTensorProducer(cast5Node->getInput(0));
    }

    if (slice1Node == nullptr || slice1Node->getNodeType() != Node::TYPE_SLICE) return false;
    commonNodes.push_back(slice1Node);

    NodePtr expandDims1Node = m_graph.getTensorProducer(slice1Node->getInput(0));
    if (expandDims1Node == nullptr || expandDims1Node->getNodeType() != Node::TYPE_INTERNAL_EXPAND_DIMS) return false;
    commonNodes.push_back(expandDims1Node);

    NodePtr expandDims2Node = m_graph.getTensorProducer(expandDims1Node->getInput(0));
    if (expandDims2Node == nullptr || expandDims2Node->getNodeType() != Node::TYPE_INTERNAL_EXPAND_DIMS) return false;
    commonNodes.push_back(expandDims2Node);

    NodePtr slice2Node = m_graph.getTensorProducer(expandDims2Node->getInput(0));
    if (slice2Node == nullptr || slice2Node->getNodeType() != Node::TYPE_SLICE) return false;
    commonNodes.push_back(slice2Node);

    inputMaskTensor = slice2Node->getInput(0);
    return true;
}

void RaggedSoftmaxFuserV2::transposeRaggedSoftmaxTensors(TensorPtr& tensorToNorm,
                                                         TensorPtr& normalizedTensor,
                                                         unsigned numFusions,
                                                         NodeList& newNodes)
{
    // need to tranpose between dim0 and dim3:
    gc::Permutation perm({1,2,3,0});

    synTransposeParams transposeParams;
    perm.getValues(transposeParams.permutation, MAX_DIMENSIONS_NUM);
    transposeParams.tensorDim = 4;

    TensorPtr transposedInput = tensorToNorm->clone();
    transposedInput->setName(fmt::format("{}_transposed", tensorToNorm->getName()));
    TSize newSizes[Tensor::c_tensorMaxDim];
    transposedInput->getAllSizesInElements(newSizes, Tensor::c_tensorMaxDim);
    perm.permuteShape<TSize>(newSizes, transposedInput->getDim());
    transposedInput->reshape(4U, newSizes, nullptr);

    NodePtr trBefore = NodeFactory::createNode({tensorToNorm},
                                               {transposedInput},
                                               &transposeParams,
                                               "transpose",
                                               fmt::format("ragged_softmax_transpose_in{}", numFusions));

    tensorToNorm = transposedInput;

    gc::Permutation invPerm(perm.getInversePermutation());
    synTransposeParams softmaxOutputTransposeParams;
    invPerm.getValues(softmaxOutputTransposeParams.permutation, MAX_DIMENSIONS_NUM);
    softmaxOutputTransposeParams.tensorDim = 4;

    TensorPtr  transposedOutput = transposedInput->clone();
    transposedOutput->setName(fmt::format("{}_transposed", normalizedTensor->getName()));
    transposedOutput->setAllQuantizationParams(normalizedTensor->getAllQuantizationParams());
    transposedOutput->setDynamicRange(normalizedTensor->getDynamicRange());
    NodePtr trAfter = NodeFactory::createNode({transposedOutput},
                                              {normalizedTensor},
                                              &softmaxOutputTransposeParams,
                                              "transpose",
                                              fmt::format("ragged_softmax_transpose_out{}", numFusions));

    normalizedTensor = transposedOutput;

    newNodes.push_back(trBefore);
    newNodes.push_back(trAfter);
}

void RaggedSoftmaxFuserV2::createSeqLenTensor(NodeList& newNodes, TensorPtr inputMaskTensor, TensorPtr& seqLenTensor)
{
    if (inputMaskTensor == nullptr) return;

    TSize seqLenSizes[] = {inputMaskTensor->getSizeInElements(1)};
    TSize seqLen2dSizes[] = {1, inputMaskTensor->getSizeInElements(1)};

    synQuantizationParams seqLenTensorQuant;
    seqLenTensorQuant.m_scale     = 1;
    seqLenTensorQuant.m_zp        = 0;
    seqLenTensorQuant.m_qDataType = syn_type_int16;

    TensorPtr seqLenTensor2d = std::make_shared<Tensor>(2U, seqLen2dSizes, syn_type_int16);
    seqLenTensor2d->setName("sequence_len_2d");
    seqLenTensor2d->setQuantizationParams(seqLenTensorQuant);

    seqLenTensor = std::make_shared<Tensor>(1U, seqLenSizes, syn_type_int16);
    seqLenTensor->setName("sequence_len");
    seqLenTensor->setQuantizationParams(seqLenTensorQuant);

    if (inputMaskTensor->getElementType() != syn_type_int16)
    {
        // cast normalize tensor to fp16
        const std::string castKernel =
            fmt::format("cast_{}_to_i16", getDtypeSuffixFromSynDataType(inputMaskTensor->getElementType()));
        TensorPtr  castOutput = inputMaskTensor->clone();
        castOutput->setName(fmt::format("{}_casted", inputMaskTensor->getName()));
        castOutput->setAllQuantizationParams(inputMaskTensor->getAllQuantizationParams());
        castOutput->setDynamicRange(inputMaskTensor->getDynamicRange());
        castOutput->setElementType(syn_type_int16);
        NodePtr castNode = NodeFactory::createNode({inputMaskTensor},
                                                   {castOutput},
                                                   nullptr,
                                                   castKernel,
                                                   fmt::format("cast_mask_tensor_{}", inputMaskTensor->getName()));
        newNodes.push_back(castNode);
        inputMaskTensor = castOutput;
    }

    ns_Reduction::Params params = {0};
    NodePtr reduceSumNode = NodeFactory::createNode({inputMaskTensor}, {seqLenTensor2d}, &params,
                                                    "reduce_sum_i16", "reduce_sum_input_mask");

    NodePtr squeezeSeqLenNode = NodeFactory::createNode({seqLenTensor2d},
                                                        {seqLenTensor},
                                                        nullptr,
                                                        NodeFactory::reshapeNodeTypeName,
                                                        "squeeze_seq_len");

    newNodes.push_back(reduceSumNode);
    newNodes.push_back(squeezeSeqLenNode);
}

bool RaggedSoftmaxFuserV2::fuseRaggedSoftmax(Graph* pattern)
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

        TensorPtr seqLenTensor;
        LOG_DEBUG(GC, "found {} patterns", matchingNodes.size());
        for (NodePtr lastNode : matchingNodes)
        {
            LOG_DEBUG(GC, "Checking pattern ending at node: {}", lastNode->getNodeName());
            NodePtr              reshape2Node = lastNode;
            NodeList             patternNodes;
            NodeList             commonNodes;
            TensorPtr            tensorToNorm;
            TensorPtr            normalizedTensor;
            ns_Softmax::Params   softmaxParams;
            TensorPtr            inputMaskTensor;

            if (!validatePattern(reshape2Node,
                                 patternNodes,
                                 commonNodes,
                                 tensorToNorm,
                                 normalizedTensor,
                                 softmaxParams,
                                 inputMaskTensor,
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

            std::string_view kernelName = "ragged_softmax_f16";
            if (tensorToNorm->getElementType() != syn_type_fp16)
            {
                // cast tensorToNorm to fp16
                const std::string castKernel =
                    fmt::format("cast_{}_to_f16", getDtypeSuffixFromSynDataType(tensorToNorm->getElementType()));
                TensorPtr  castOutput = tensorToNorm->clone();
                castOutput->setName(fmt::format("{}_casted", tensorToNorm->getName()));
                castOutput->setAllQuantizationParams(tensorToNorm->getAllQuantizationParams());
                castOutput->setDynamicRange(tensorToNorm->getDynamicRange());
                castOutput->setElementType(syn_type_fp16);
                NodePtr castNode = NodeFactory::createNode({tensorToNorm},
                                                           {castOutput},
                                                           nullptr,
                                                           castKernel,
                                                           fmt::format("cast_tensor_to_norm_{}", fusions));
                newNodes.push_back(castNode);
                tensorToNorm = castOutput;
            }

            if (normalizedTensor->getElementType() != syn_type_fp16)
            {
                // cast ragged_softmax output to bf16
                std::string_view castKernel = "cast_f16_to_bf16";
                TensorPtr        castInput  = normalizedTensor->clone();
                castInput->setName(fmt::format("{}_casted", normalizedTensor->getName()));
                castInput->setAllQuantizationParams(normalizedTensor->getAllQuantizationParams());
                castInput->setDynamicRange(normalizedTensor->getDynamicRange());
                castInput->setElementType(syn_type_fp16);
                normalizedTensor->setElementType(syn_type_bf16);
                NodePtr castNode = NodeFactory::createNode({castInput},
                                                           {normalizedTensor},
                                                           nullptr,
                                                           castKernel,
                                                           fmt::format("cast_normalized_tensor_{}", fusions));
                newNodes.push_back(castNode);
                normalizedTensor = castInput;
                normalizedTensor->setElementType(syn_type_fp16);
            }

            NodePtr raggedSoftmax = NodeFactory::createNode({tensorToNorm, seqLenTensor},
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

            for (pNode nodeToRemove : commonNodes)
            {
                if (nodeToRemove != nullptr && m_graph.getTensorConsumers(nodeToRemove->getOutput(0)).size() == 0)
                {
                    GraphEditor::removeNode(m_graph, nodeToRemove);
                    LOG_DEBUG(GC, "Removed unused node: {}", nodeToRemove->getNodeName());
                }
            }
        }
        LOG_DEBUG(GC, "completed {} fusions", fusions);
    }
    return true;
}