#include "nms_node.h"

#include "habana_graph.h"
#include "node_factory.h"
#include "perf_lib_layer_params.h"

#include "types_exception.h"
#include "utils.h"

/* This function gathers selected boxes according to the topkIndices
 * it will gather per batch of boxes and then will concat and output to topkBoxes*/
void NMSNode::gatherTopkBoxes(TensorPtr boxesTensor,
                              TensorPtr topkIndicesTensor,
                              TensorPtr topkBoxes, NodeList& retNodes)
{
    static const char* gatherKernelName_f32 = "gather_f32";

    unsigned K = topkIndicesTensor->getSizeInElements(0);
    unsigned C = topkIndicesTensor->getSizeInElements(1);
    unsigned N = topkIndicesTensor->getSizeInElements(2);
    unsigned B = boxesTensor->getSizeInElements(0);


    /* Step-1: split boxes and indices on N axis */

    TSize splittedBoxesSizes[Tensor::c_tensorMaxDim] = {B, 4, 1};
    TSize splittedIndicesSizes[Tensor::c_tensorMaxDim] = {K, C};

    /* Creating N tensors for boxes and indices to use as input to gather node */
    TensorVector splittedBoxesVector;
    TensorVector splittedIndicesVector;
    for (unsigned i = 0; i < N; ++i)
    {
        TensorPtr tmpSplittedBoxesTensor = std::make_shared<Tensor>(*(boxesTensor.get()));
        tmpSplittedBoxesTensor->reshape(3U, splittedBoxesSizes, nullptr);
        tmpSplittedBoxesTensor->setName(fmt::format("split_boxes{}_{}", i, m_name));
        splittedBoxesVector.push_back(tmpSplittedBoxesTensor);

        TensorPtr tmpSplittedIndicesTensor = std::make_shared<Tensor>(*(topkIndicesTensor.get()));
        tmpSplittedIndicesTensor->reshape(2U, splittedIndicesSizes, nullptr);
        tmpSplittedIndicesTensor->setName(fmt::format("split_indices{}_{}", i, m_name));
        splittedIndicesVector.push_back(tmpSplittedIndicesTensor);
    }


    if (N == 1)
    {
        /* If N == 1, we will reshape to 2D */
        NodePtr reshapeBoxesNode = NodeFactory::createNode({boxesTensor},
                                                           splittedBoxesVector,
                                                           nullptr,
                                                           NodeFactory::reshapeNodeTypeName,
                                                           fmt::format("reshape_boxes_for_gather_{}", m_name));
        retNodes.push_back(reshapeBoxesNode);

        NodePtr reshapeIndicesNode = NodeFactory::createNode({topkIndicesTensor},
                                                             splittedIndicesVector,
                                                             nullptr,
                                                             NodeFactory::reshapeNodeTypeName,
                                                             fmt::format("reshape_indices_for_gather_{}", m_name));
        retNodes.push_back(reshapeIndicesNode);
    }
    else
    {
        unsigned splitDim = 2;

        NodePtr splitBoxesNode = NodeFactory::createNode({boxesTensor},
                                                         splittedBoxesVector,
                                                         &splitDim,
                                                         NodeFactory::splitNodeInternalTypeName,
                                                         fmt::format("split_boxes_for_gather_{}", m_name));
        retNodes.push_back(splitBoxesNode);

        NodePtr splitIndicesNode = NodeFactory::createNode({topkIndicesTensor},
                                                           splittedIndicesVector,
                                                           &splitDim,
                                                           NodeFactory::splitNodeInternalTypeName,
                                                           fmt::format("split_indices_for_gather_{}", m_name));
        retNodes.push_back(splitIndicesNode);
    }

    /* Invoking N gather nodes, and Creating N topk boxes tensors for each gather node */
    /* We will later need to sqeeze the 1 size in dim=2 on the concated topk boxes */
    TSize splittedTopkBoxesSizes[Tensor::c_tensorMaxDim] = {K, C, 4, 1};
    TensorVector splittedTopkBoxesVector;
    for (unsigned i = 0; i < N; ++i)
    {
        TensorPtr tmpSplittedTopkBoxes = std::make_shared<Tensor>(*(splittedBoxesVector[i].get()));
        tmpSplittedTopkBoxes->reshape(4U, splittedTopkBoxesSizes, nullptr);
        tmpSplittedTopkBoxes->setName(fmt::format("split_topk_boxes{}_{}", i, m_name));
        splittedTopkBoxesVector.push_back(tmpSplittedTopkBoxes);

        ns_GatherKernel::Params gatherParams = {0};

        NodePtr gatherNodeTmp = NodeFactory::createNode({splittedBoxesVector[i], splittedIndicesVector[i]},
                                                        {tmpSplittedTopkBoxes},
                                                        nullptr,
                                                        gatherKernelName_f32,
                                                        fmt::format("gather_{}_{}", i, m_name));
        (std::dynamic_pointer_cast<TPCNode>(gatherNodeTmp))->storeParamsInBuffer(&gatherParams,
                                                                                 sizeof(ns_GatherKernel::Params));
        retNodes.push_back(gatherNodeTmp);
    }

    unsigned concatDim = 3;

    NodePtr concatTopkBoxesNode = NodeFactory::createNode(splittedTopkBoxesVector,
                                                          {topkBoxes},
                                                          &concatDim,
                                                          NodeFactory::concatenateNodeInternalTypeName,
                                                          fmt::format("concat_after_gather_{}", m_name));
    retNodes.push_back(concatTopkBoxesNode);
}

NMSNode::NMSNode(const TensorVector& inputs, const TensorVector& outputs, UserParams params, std::string_view name)
: MultiNode(inputs, outputs, name, Node::TYPE_NMS, SIF_NO_SUPPORT)
{
    setParams(params, sizeof(synNMSParams));
}

NodePtr NMSNode::createNode(const TensorVector& inputs,
                            const TensorVector& outputs,
                            UserParams          userParams,
                            std::string_view    guid,
                            std::string_view    name)
{
    return NodePtr(new NMSNode(inputs, outputs, userParams, name));
}

void NMSNode::setParams(UserParams userParams, unsigned int userParamsSize)
{
    if (userParams == nullptr)
    {
        LOG_ERR(HABANA_NODE, "NMSNode userParams is null");
        throw InvalidNodeParamsException(m_name, "userParams");
    }
    if (userParamsSize != sizeof(synNMSParams))
    {
        LOG_ERR(HABANA_NODE, "NMSNode userParams size is incorrect");
        throw InvalidNodeParamsSizeException(m_name, userParamsSize, sizeof(synNMSParams));
    }
    synNMSParams params =  *(synNMSParams*)userParams;
    LOG_TRACE(HABANA_NODE,
              "NMSNode name - {}, params - iouTh={}, maxOutSize={}, scoreTh={}",
              m_name,
              params.iouTh,
              params.maxOutSize,
              params.scoreTh);
    m_params = params;
}

NodePtr NMSNode::clone() const
{
    return NodePtr(new NMSNode(*this));
}

NodeList NMSNode::extract()
{
    NodeList retNodes;

    /* Kernel Names */
    static const char* filterAndSqueezeKernelName_f32 = "filter_and_squeeze_f32";
    static const char* nmsKernelName_f32              = "nms_f32";
    static const char* postNmsKernelName_i32          = "post_nms_i32";

    /* inputs tensors to NMS model */

    TensorPtr boxesTensor  = getInput(TENSOR_IFM);
    TensorPtr scoresTensor = getInput(1);

    /* Constant Values */
    TSize scoresSizes[Tensor::c_tensorMaxDim];
    scoresTensor->getAllSizesInElements(scoresSizes, Tensor::c_tensorMaxDim);

    TSize C = scoresSizes[1];
    TSize N = scoresSizes[2];
    TSize K = scoresSizes[0];

    /*
     * Intermediate tensors
     */

    /* filteredScores, filteredBoxIndices */
    TensorPtr filteredScoresTensor = std::make_shared<Tensor>(*(scoresTensor.get()));
    filteredScoresTensor->setName(fmt::format("{}filteredScores", m_name));
    TensorPtr filteredBoxIndicesTensor = std::make_shared<Tensor>(scoresTensor->getDim(), scoresSizes, syn_type_int32);
    filteredBoxIndicesTensor->setName(fmt::format("{}filteredBoxIndices", m_name));

    /* validCount_squeezed and validCount_sorted */
    TSize validCountSizes[Tensor::c_tensorMaxDim];
    for (int i = 0; i < scoresTensor->getDim() - 1; i++)
    {
        validCountSizes[i] = scoresSizes[i + 1];
    }
    TensorPtr validCountSqueezedTensor = std::make_shared<Tensor>(scoresTensor->getDim() - 1, validCountSizes, syn_type_int32);
    validCountSqueezedTensor->setName(fmt::format("{}validCount_squeezed", m_name));
    TensorPtr validCountSortedTensor = std::make_shared<Tensor>(scoresTensor->getDim() - 1, validCountSizes, syn_type_int32);
    validCountSortedTensor->setName(fmt::format("{}validCount_sorted", m_name));

    /* topkScores, topkIndices */
    TensorPtr topkScoresTensor = std::make_shared<Tensor>(*(filteredScoresTensor.get()));
    topkScoresTensor->setName(fmt::format("{}topkScores", m_name), true);
    TensorPtr topkIndicesTensor = std::make_shared<Tensor>(*(filteredBoxIndicesTensor.get()));
    topkIndicesTensor->setName(fmt::format("{}topkIndices", m_name));

    /* topk Boxes  {K,C,4,N}*/
    TSize topkBoxesSizes[Tensor::c_tensorMaxDim] = {K, C, 4, N, 1};
    TensorPtr topkBoxesTensor = std::make_shared<Tensor>(*(boxesTensor.get()));
    topkBoxesTensor->reshape(4U, topkBoxesSizes, nullptr);
    topkBoxesTensor->setName(fmt::format("{}topkBoxes", m_name));

    /* post nms box indices */
    TSize postNmsBoxIndicesSizes[Tensor::c_tensorMaxDim] = {K, C, N, 1, 1};
    TensorPtr postNmsBoxIndicesTensor = std::make_shared<Tensor>(*(topkIndicesTensor.get()));
    topkIndicesTensor->reshape(3U, postNmsBoxIndicesSizes, nullptr);
    postNmsBoxIndicesTensor->setName(fmt::format("{}postNmsBoxIndices", m_name));

    /* valid count 1D */
    /* TODO SW-6600: Check if this tensor is still requried */
    TSize validCount1dSizes[Tensor::c_tensorMaxDim] = {1, 1, 1, 1, 1};
    TensorPtr validCount1dTensor = std::make_shared<Tensor>(1U, validCount1dSizes, syn_type_int32);
    validCount1dTensor->setName(fmt::format("{}validCount1D", m_name), true);

    /* Step-1: filter_and_squeeze
     * input tensor is the scores
     * param is the score_threshold
     * output tensors: valid count, filtered scores, filtered box indices */

    ns_FilterAndSqueeze::Params filterAndSqueezeParams = {m_params.scoreTh};
    NodePtr                     filterAndSqueezeNode =
        NodeFactory::createNode({scoresTensor},
                                {filteredScoresTensor, filteredBoxIndicesTensor, validCountSqueezedTensor},
                                nullptr,
                                filterAndSqueezeKernelName_f32,
                                fmt::format("{}filterAndSqueeze", getNodeName()));
    (std::dynamic_pointer_cast<TPCNode>(filterAndSqueezeNode))->storeParamsInBuffer(&filterAndSqueezeParams, sizeof(ns_FilterAndSqueeze::Params));
    retNodes.push_back(filterAndSqueezeNode);

    /* Step-2: TOP-K */

    synBeamParams topkParams;
    topkParams.axis = 0;
    topkParams.bsw = K;
    NodePtr topkNode =
        NodeFactory::createNode({filteredScoresTensor, filteredBoxIndicesTensor, validCountSqueezedTensor},
                                {topkScoresTensor, topkIndicesTensor, validCountSortedTensor},
                                &topkParams,
                                "topk",
                                fmt::format("{}topk", getNodeName()));
    retNodes.push_back(topkNode);

    /* Step-3: Gather
     * 'gather' will be performed on each batch (n in N)
     * so split prior to the 'gather' and concat back after
     * input tensors: Boxes, topk k indices
     * output tensors: top k boxes */
    gatherTopkBoxes(boxesTensor, topkIndicesTensor, topkBoxesTensor, retNodes);

    /* Step-4: Decode (currently not implemented ) */

    /* Step-5: NMS */

    ns_Nms::Params nmsParams = {m_params.iouTh};

    NodePtr nmsNode = NodeFactory::createNode({topkBoxesTensor, topkIndicesTensor, validCountSortedTensor},
                                              {postNmsBoxIndicesTensor},
                                              nullptr,
                                              nmsKernelName_f32,
                                              fmt::format("{}nms", getNodeName()));
    (std::dynamic_pointer_cast<TPCNode>(nmsNode))->storeParamsInBuffer(&nmsParams, sizeof(ns_Nms::Params));

    retNodes.push_back(nmsNode);

    /* Step-6: Post NMS */

    ns_PostNms::Params postNmsParams = {m_params.maxOutSize};

    NodePtr postNmsNode = NodeFactory::createNode({postNmsBoxIndicesTensor, validCountSortedTensor},
                                                  {getOutput(TENSOR_OFM), validCount1dTensor},
                                                  nullptr,
                                                  postNmsKernelName_i32,
                                                  fmt::format("{}post_nms", getNodeName()));
    (std::dynamic_pointer_cast<TPCNode>(postNmsNode))->storeParamsInBuffer(&postNmsParams, sizeof(ns_PostNms::Params));

    retNodes.push_back(postNmsNode);

    return retNodes;
}

bool NMSNode::validateNodeForGraph(const HabanaGraph& g) const
{
    return false;
}

void NMSNode::printParamsRawData() const
{
    BaseClass::printParamsRawData((void*)&m_params, sizeof(m_params));
}
