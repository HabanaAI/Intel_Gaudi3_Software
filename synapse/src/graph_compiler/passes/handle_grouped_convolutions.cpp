#include "handle_grouped_convolutions.h"

#include "compilation_hal_reader.h"
#include "dedw_node.h"
#include "dedx_node.h"
#include "habana_graph.h"
#include "node_factory.h"
#include "pack_grouped_convolutions.h"
#include "types_exception.h"

const ConvBaseNode* GroupedConvolutionManager::getGroupedConvolution(const NodePtr& node)
{
    switch (node->getNodeType())
    {
        case Node::TYPE_DEDW:
        case Node::TYPE_DEDX:
        case Node::TYPE_CONVOLUTION:
        {
            auto convNode = static_cast<ConvBaseNode*>(node.get());
            if (convNode->getConvolutionParams().nGroups > 1)
            {
                return convNode;
            }
        }
        default:
            return nullptr;
    }
}

void GroupedConvolutionManager::setCurrentNode(const NodePtr& node)
{
    m_convNode = getGroupedConvolution(node);
}

bool GroupedConvolutionManager::canExtract() const
{
    return m_convNode != nullptr;
}

bool GroupedConvolutionManager::runHandleGroupedConvolutions(HabanaGraph& g)
{
    NodeVector nodes  = g.getExeSortedNodes();
    bool     result = true;
    for (const pNode& n : nodes)
    {
        setCurrentNode(n);
        if (!canExtract()) continue;
        if (!validateGroupedConvolutionNode())
        {
            result = false;
            break;
        }
        auto newNodes = extract(g);
        if (newNodes.empty())
        {
            result = false;
            break;
        }
        if (GraphEditor::replaceNodes(g, {n}, newNodes) != REPLACE_NODE_SUCCESS)
        {
            result = false;
            break;
        }
    }
    m_convNode = nullptr;
    return result;
}

/********************************** GroupedConvolutionManagerInference ******************************/
GroupedConvolutionManagerInference::GroupedConvolutionManagerInference(unsigned mmeVectorSize, bool runLogicalOp)
: m_mmeVectorSize(mmeVectorSize), m_runLogicalOp(runLogicalOp)
{
}

bool GroupedConvolutionManagerInference::validateGroupedConvolutionNode() const
{
    TensorPtr OFM     = m_convNode->getYOperand();
    TensorPtr IFM     = m_convNode->getXOperand();
    TensorPtr WGH     = m_convNode->getWOperand();
    TensorPtr BIAS    = m_convNode->getInput(TENSOR_BIAS);
    unsigned  nGroups = m_convNode->getConvolutionParams().nGroups;

    // Input channel dim should be divided into nGroups without remainder
    if (IFM->getSizeInElements(DIM_C) % nGroups != 0)
    {
        LOG_ERR(GC, "Validate grouped convolution node failed. IFM DIM_C {} % nGroup {} =! 0", IFM->getSizeInElements(DIM_C), nGroups);
        return false;
    }
    // Weights k dim (num of filters) should be divided into nGroups without remainder
    // C dim (common dim) should not be divided into nGroups since dim C is already divided into nGroups from user
    if (WGH->getSizeInElements(WEIGHT_DIM_K) % nGroups != 0)
    {
        LOG_ERR(GC, "Validate grouped convolution node failed. WGH WEIGHT_DIM_K {} % nGroup {} =! 0", WGH->getSizeInElements(WEIGHT_DIM_K), nGroups);
        return false;
    }
    // Bias should be divided into nGroups without remainder
    if (BIAS != nullptr)
    {
        if (BIAS->getSizeInElements(DIM_C) % nGroups != 0) {
            LOG_ERR(GC, "Validate grouped convolution node failed. BIAS DIM_C {} % nGroup {} =! 0",
                    BIAS->getSizeInElements(DIM_C), nGroups);
            return false;
        }
    }
    // Output c dim ( = k dim of weights) should be divided into nGroups without remainder
    if (OFM->getSizeInElements(DIM_C) % nGroups != 0)
    {
        LOG_ERR(GC, "Validate grouped convolution node failed. OFM DIM_C {} % nGroup {} =! 0", OFM->getSizeInElements(DIM_C), nGroups);
        return false;
    }

    return true;
}

template <typename DType>
static DType* PadWeightTensor(DType* pWeightData, DType zp, unsigned Q, unsigned R, unsigned S, unsigned C, unsigned K,
                              unsigned nGroups, unsigned kOffset, unsigned kStride)
{
    if (K % nGroups != 0)
    {
        LOG_ERR(GC, "Expected number of groups to evenly divide number of output channels");
        throw IllegalGroupParams();
    }
    if (C % nGroups != 0)
    {
        LOG_ERR(GC, "Expected number of groups to evenly divide number of input channels");
        throw IllegalGroupParams();
    }

    unsigned kPerGroup = K / nGroups;
    unsigned cPerGroup = C / nGroups;
    unsigned weightsSpatialSize =  R * Q * S;

    DType* pNewWeights = new DType[K * C * weightsSpatialSize];
    std::fill(pNewWeights, pNewWeights + weightsSpatialSize * C * K, zp);

    for (unsigned g = 0; g < nGroups; ++g)
    {
        for (unsigned spatialIdx = 0; spatialIdx < weightsSpatialSize; ++spatialIdx)
        {
            for (unsigned c = g * cPerGroup; c < (g + 1) * cPerGroup; ++c)
            {
                DType* src = pWeightData +
                             spatialIdx * cPerGroup * kStride +
                             (c % cPerGroup) * kStride +
                             kOffset +
                             g * kPerGroup;
                DType* dst = pNewWeights + spatialIdx * C * K + c * K + g * kPerGroup;
                memcpy(dst, src, kPerGroup * sizeof(DType));
            }
        }
    }
    return pNewWeights;
}

NodeList GroupedConvolutionManagerInference::extract(const HabanaGraph& g)
{
    TensorPtr IFM     = m_convNode->getXOperand();
    TensorPtr WGH     = m_convNode->getWOperand();
    TensorPtr OFM     = m_convNode->getYOperand();
    TensorPtr BIAS    = m_convNode->getInput(TENSOR_BIAS);
    unsigned  nGroups = m_convNode->getConvolutionParams().nGroups;

    unsigned C          = IFM->getSizeInElements(DIM_C);
    unsigned K          = WGH->getSizeInElements(WEIGHT_DIM_K);
    unsigned R          = WGH->getSizeInElements(WEIGHT_DIM_R);
    unsigned S          = WGH->getSizeInElements(WEIGHT_DIM_S);
    unsigned Q          = WGH->getSizeInElements(WEIGHT_DIM_Q);
    double zp           = WGH->getZeroPoint();

    unsigned vectorSize      = m_mmeVectorSize / IFM->getElementSizeInBytes();
    unsigned kPerGroup       = K / nGroups;
    unsigned cPerGroup       = C / nGroups;
    unsigned groupsPerVector = std::min(nGroups, std::max(1U, vectorSize / kPerGroup));

    unsigned                 nConvolutions   = div_round_up(nGroups, groupsPerVector);
    unsigned                 groupsRemainder = (K % (groupsPerVector * kPerGroup)) / kPerGroup;
    synConvolution3DParamsV2 newParams       = m_convNode->getConvolutionParams();
    newParams.nGroups                        = 1;

    TensorVector pIFMs;
    TensorVector pOFMs;

    TSize IFMSizes[Tensor::c_tensorMaxDim], OFMSizes[Tensor::c_tensorMaxDim], WGHSizes[Tensor::c_tensorMaxDim];
    IFM->getAllSizesInElements(IFMSizes, Tensor::c_tensorMaxDim);
    OFM->getAllSizesInElements(OFMSizes, Tensor::c_tensorMaxDim);
    WGH->getAllSizesInElements(WGHSizes, Tensor::c_tensorMaxDim);
    NodeList newNodes;

    for (unsigned conv = 0; conv < nConvolutions; ++conv)
    {
        bool remainderConv        = conv == nConvolutions - 1 && groupsRemainder > 0;
        unsigned groupsForCurConv = groupsPerVector;
        if (remainderConv)
        {
            groupsForCurConv = groupsRemainder;
        }
        IFMSizes[DIM_C]        = cPerGroup * groupsForCurConv;
        OFMSizes[DIM_C]        = kPerGroup * groupsForCurConv;
        WGHSizes[WEIGHT_DIM_K] = kPerGroup * groupsForCurConv;
        WGHSizes[WEIGHT_DIM_C] = cPerGroup * groupsForCurConv;

        TensorPtr curIFM = IFM, curOFM = OFM;
        std::string convName = m_convNode->getNodeName();
        if (nConvolutions > 1)
        {
            curIFM = std::make_shared<Tensor>(IFM->getDim(), IFMSizes, IFM->getElementType());
            curIFM->setName(IFM->getName() + "_subgroup_" + std::to_string(conv));
            curIFM->setAllQuantizationParams(IFM->getAllQuantizationParams());
            curIFM->setDynamicRange(IFM->getDynamicRange());

            curOFM = std::make_shared<Tensor>(OFM->getDim(), OFMSizes, OFM->getElementType());
            curOFM->setName(OFM->getName() + "_subgroup_" + std::to_string(conv));
            curOFM->setAllQuantizationParams(OFM->getAllQuantizationParams());
            curOFM->setDynamicRange(OFM->getDynamicRange());

            convName += "_subgroup_" + std::to_string(conv);
        }

        pIFMs.push_back(curIFM);
        pOFMs.push_back(curOFM);

        char* pCurWeights    = WGH->getData();
        unsigned kOffset = conv * groupsPerVector * kPerGroup;
        void* pNewWeights;

        try
        {
            switch (WGH->getElementType())
            {
                case syn_type_fixed:
                    pNewWeights = PadWeightTensor<int8_t>((int8_t*)pCurWeights, (int8_t)zp, Q, R, S,
                                                          cPerGroup * groupsForCurConv, kPerGroup * groupsForCurConv,
                                                          groupsForCurConv, kOffset, K);
                    break;

                case syn_type_uint8:
                    pNewWeights = PadWeightTensor<uint8_t>((uint8_t*)pCurWeights, (uint8_t)zp, Q, R, S,
                                                           cPerGroup * groupsForCurConv, kPerGroup * groupsForCurConv,
                                                           groupsForCurConv, kOffset, K);
                    break;

                case syn_type_int16:
                    pNewWeights = PadWeightTensor<int16_t>((int16_t*)pCurWeights, (int16_t)zp, Q, R, S,
                                                           cPerGroup * groupsForCurConv, kPerGroup * groupsForCurConv,
                                                           groupsForCurConv, kOffset, K);
                    break;

                default:
                    LOG_ERR(GC, "Unsupported weight data type for grouped convolution.");
                    throw IllegalGroupParams();
            }

        }
        catch (IllegalGroupParams& e)
        {
            LOG_ERR(GC, "Illegal param exception thrown during handling grouped convolution.");
            throw IllegalGroupParams(m_convNode->getNodeName());
        }

        TensorPtr curWGH = std::make_shared<Tensor>(WGH->getDim(), WGHSizes, WGH->getElementType());
        curWGH->bind(pNewWeights, true);
        curWGH->setName(WGH->getName() + "_weights_subgroup_" + std::to_string(conv));
        curWGH->setAllQuantizationParams(WGH->getAllQuantizationParams());
        curWGH->setDynamicRange(WGH->getDynamicRange());
        curWGH->setAsWeights();
        curWGH->setAsStaticParam(WGH->isStaticParam());

        TensorPtr curBias = nullptr;
        if (BIAS)
        {
            TSize BIASSizes[Tensor::c_tensorMaxDim];
            BIAS->getAllSizesInElements(BIASSizes, Tensor::c_tensorMaxDim);
            TSize curBiasSize = kPerGroup * groupsForCurConv;
            BIASSizes[DIM_C] = curBiasSize;
            char* curBiasData = new char[curBiasSize * sizeof(int32_t)];
            char* biasData = BIAS->getData();
            unsigned offset = kOffset * sizeof(int32_t);
            memcpy(curBiasData, biasData + offset, curBiasSize * sizeof(int32_t));
            curBias = std::make_shared<Tensor>(BIAS->getDim(), BIASSizes, BIAS->getElementType(), curBiasData);
            curBias->setName(BIAS->getName() + "_subgroup_" + std::to_string(conv));
            curBias->setAllQuantizationParams(BIAS->getAllQuantizationParams());
            curBias->setDynamicRange(BIAS->getDynamicRange());
            curBias->setAsBias();
            curBias->setShouldFreeBuffer(true);
        }
        NodePtr newConv = NodeFactory::createNode({curIFM, curWGH, curBias, nullptr}, {curOFM}, &newParams, NodeFactory::convolution3DNodeTypeName, convName);
        newNodes.push_back(newConv);
    }
    if (nConvolutions > 1)
    {
        unsigned splitDim = DIM_C;
        NodePtr  split    = NodeFactory::createNode({IFM},
                                                pIFMs,
                                                &splitDim,
                                                NodeFactory::splitNodeInternalTypeName,
                                                fmt::format("split_for_{}", m_convNode->getNodeName()));
        newNodes.push_back(split);
        /*
         * In goya2, gaudi1 and gaudi2 we have a specific predicate (PREDICATE_ID_LOGICAL_NODE_CREATED) for
         * handleLogicalOps pass. When we add split and concatenate nodes, we turn on this predicate and
         * handleLogicalOps pass runs again. In goya1, we do not use this predicate and in this pass we need to run the
         * logical operation 'manually' for split and concatenate logical nodes.
         */
        if (m_runLogicalOp)
        {
            split->runLogicalOperation();
        }
        unsigned concatDim = DIM_C;
        NodePtr  concat    = NodeFactory::createNode(pOFMs,
                                                 {OFM},
                                                 &concatDim,
                                                 NodeFactory::concatenateNodeInternalTypeName,
                                                 fmt::format("concat_for_{}", m_convNode->getNodeName()));
        newNodes.push_back(concat);
        if (m_runLogicalOp)
        {
            concat->runLogicalOperation();
        }
    }
    return newNodes;
}

/********************************** GroupedConvolutionManagerTraining ******************************/

bool GroupedConvolutionManagerTraining::validateGroupedConvolutionNode() const
{
    TensorPtr OFM  = m_convNode->getYOperand();
    TensorPtr IFM  = m_convNode->getXOperand();
    TensorPtr WGH  = m_convNode->getWOperand();
    TensorPtr BIAS = m_convNode->getNodeType() == Node::TYPE_CONVOLUTION ? m_convNode->getInput(TENSOR_BIAS) : nullptr;
    unsigned  nGroups = m_convNode->getConvolutionParams().nGroups;

    if (BIAS != nullptr)
    {
        LOG_ERR(GC, "Bias is not supported with grouped convolution for gaudi");
        return false;
    }
    // Input channel dim should be divided into nGroups without remainder
    if (IFM->getSizeInElements(DIM_C) % nGroups != 0)
    {
        LOG_ERR(GC, "Validate grouped convolution node failed. IFM DIM_C {} % nGroup {} =! 0", IFM->getSizeInElements(DIM_C), nGroups);
        return false;
    }
    // Weights k dim (num of filters) should be divided into nGroups without remainder
    // C dim (common dim) should not be divided into nGroups since dim C is already divided into nGroups from user
    if (WGH->getSizeInElements(WEIGHT_DIM_K) % nGroups != 0)
    {
        LOG_ERR(GC, "Validate grouped convolution node failed. WGH WEIGHT_DIM_K {} % nGroup {} =! 0", WGH->getSizeInElements(WEIGHT_DIM_K), nGroups);
        return false;
    }
    // Output c dim ( = k dim of weights) should be divided into nGroups without remainder
    if (OFM->getSizeInElements(DIM_C) % nGroups != 0)
    {
        LOG_ERR(GC, "Validate grouped convolution node failed. OFM DIM_C {} % nGroup {} =! 0", OFM->getSizeInElements(DIM_C), nGroups);
        return false;
    }

    return true;
}

void GroupedConvolutionManagerTraining::createSplitNode(const TensorVector& inputs,
                                                        const TensorVector& outputs,
                                                        unsigned            splitDim,
                                                        const std::string&  name,
                                                        NodeList&           newNodes,
                                                        bool                isShape)
{
    if (outputs.empty())
    {
        // Reuse existing split node
        return;
    }

    HB_ASSERT(inputs.size() == 1, "Expected one input for split node");
    synSplitParams splitParams;
    splitParams.axis = splitDim;
    NodePtr split =
        NodeFactory::createNode(inputs,
                                outputs,
                                &splitParams,
                                isShape ? NodeFactory::splitShapeNodeTypeName : NodeFactory::splitNodeInternalTypeName,
                                name);
    newNodes.push_back(split);

    m_splitNodesCache.emplace(split->getInput(0), split);
}

void GroupedConvolutionManagerTraining::createConcatNode(const TensorVector& inputs,
                                                         const TensorVector& outputs,
                                                         unsigned            concatDim,
                                                         const std::string&  name,
                                                         NodeList&           newNodes)
{
    HB_ASSERT(inputs.size() > 0, "Expected at least one input for concat node");
    HB_ASSERT(outputs.size() == 1, "Expected one output for concat node");

    synConcatenateParams concatParams;
    concatParams.axis = concatDim;
    NodePtr concat =
        NodeFactory::createNode(inputs, outputs, &concatParams, NodeFactory::concatenateNodeInternalTypeName, name);
    newNodes.push_back(concat);
}

/**
 * Split the split nodes for the inputs, and the concat node for the output, according to the guid.
 **/
void GroupedConvolutionManagerTraining::createSplitConcatNodes(const ConvBaseNode* n,
                                                               TensorVector&       pNewIFMs,
                                                               TensorVector&       pNewOFMs,
                                                               TensorVector&       pNewWGHs,
                                                               TensorVector&       pNewShapeTensors,
                                                               TensorPtr&          IFM,
                                                               TensorPtr&          OFM,
                                                               TensorPtr&          WGH,
                                                               TensorPtr&          shapeTensor,
                                                               NodeList&           newNodes)
{
    if (n->getNodeType() == Node::TYPE_CONVOLUTION)
    {
        // Split input on DIM_C
        createSplitNode({IFM}, pNewIFMs, DIM_C, fmt::format("split_ifm_for_{}", n->getNodeName()), newNodes);

        // Split weights on DIM_K (weights are already with updated DIM_C from user)
        createSplitNode({WGH}, pNewWGHs, WEIGHT_DIM_K, fmt::format("split_weights_for_{}", n->getNodeName()), newNodes);

        // Concat nGroup output tensors into a single output
        createConcatNode(pNewOFMs, {OFM}, DIM_C, fmt::format("concat_ofm_for_{}", n->getNodeName()), newNodes);
    }
    else if (n->getNodeType() == Node::TYPE_DEDW)
    {
        // Split IFM on DIM_C
        createSplitNode({IFM}, pNewIFMs, DIM_C, fmt::format("split_ifm_for_{}", n->getNodeName()), newNodes);

        // Split OFM on DIM_C
        createSplitNode({OFM}, pNewOFMs, DIM_C, fmt::format("split_ofm_for_{}", n->getNodeName()), newNodes);

        // Concat nGroup output tensors into a single output
        createConcatNode(pNewWGHs, {WGH}, WEIGHT_DIM_K, fmt::format("concat_wgh_for_{}", n->getNodeName()), newNodes);
    }
    else //if (n->getNodeType() == Node::TYPE_DEDX)
    {
        // Split WGH on WEIGHT_DIM_K
        createSplitNode({WGH}, pNewWGHs, WEIGHT_DIM_K, fmt::format("split_wgh_for_{}", n->getNodeName()), newNodes);

        // Split OFM on DIM_C
        createSplitNode({OFM}, pNewOFMs, DIM_C, fmt::format("split_ofm_for_{}", n->getNodeName()), newNodes);

        // Split shape tensor on DIM_C
        if (n->isDynamicShape())
        {
            createSplitNode({shapeTensor},
                            pNewShapeTensors,
                            DIM_C,
                            fmt::format("split_shape_for_{}", n->getNodeName()),
                            newNodes,
                            true);
        }

        // Concat nGroup output tensors into a single output
        createConcatNode(pNewIFMs, {IFM}, DIM_C, fmt::format("concat_ifm_for_{}", n->getNodeName()), newNodes);
    }
}

NodeList GroupedConvolutionManagerTraining::extract(const HabanaGraph& g)
{
    if (GCFG_ENABLE_GCONV_PACKING.value())
    {
        GroupedConvolutionPackingManager gConvPacker(m_convNode, g);
        NodeVector                       packGroupedConvNodes = gConvPacker.packGroupedConvNode();
        NodeList                         newNodes;
        for (const auto& currentNode : packGroupedConvNodes)
        {
            auto convNode = getGroupedConvolution(currentNode);
            if (convNode)
            {
                auto splitResult = splitGConvNodeToSingleGroups(convNode);
                if (splitResult.empty())
                {
                    return NodeList();
                }
                newNodes.splice(newNodes.end(), splitResult);
            }
            else
            {
                newNodes.push_back(move(currentNode));
            }
        }
        if (!packGroupedConvNodes.empty())
        {
            return newNodes;
        }
    }
    return splitGConvNodeToSingleGroups(m_convNode);
}

// Enable reuse of split nodes which have the same input: for example: DEDX + DEDW.
NodePtr GroupedConvolutionManagerTraining::tryReuseExistingSplitNode(const ConvBaseNode* origConv,
                                                                     const TensorPtr&    tensor,
                                                                     unsigned            splitDim,
                                                                     unsigned            numGroups)
{
    if (!GCFG_ENABLE_GCONV_SPLIT_NODES_REUSE.value())
    {
        return nullptr;
    }
    const auto& convInputs = origConv->getInputs();
    bool        isInput    = std::find(convInputs.begin(), convInputs.end(), tensor) != convInputs.end();
    bool        hasCtlInputs = !origConv->getControlInputs().empty();

    auto iter = m_splitNodesCache.find(tensor);
    if (isInput && !hasCtlInputs &&
        (iter != m_splitNodesCache.end()))  // Split node on the same input exists in the cache
    {
        auto cachedSplitNode = std::dynamic_pointer_cast<SplitNode>(iter->second);
        HB_ASSERT(cachedSplitNode != nullptr, "Expected split node");

        if ((cachedSplitNode->getNumOutputs() == numGroups) && (cachedSplitNode->getSplitDim() == splitDim))
        {
            // The cached split node has the same input and split dimension.
            // Since all the operands divided into numGroups on the split dimension (without remainder),
            // we can assume all the split outputs will have the same sizes.
            return cachedSplitNode;
        }
    }
    return nullptr;
}

/**
 * Each grouped conv (fwd) node should be replaced with the follwoing nodes:
 *  - split node to split input0 on dim c into num_of_groups inputs
 *  - split node to split input1 on dim k into num_of_groups inputs
 *  - num_of_groups conv nodes with params.nGroup=1
 *  - concat node to concat num_of_groups output tensors into one output
 * */
NodeList GroupedConvolutionManagerTraining::splitGConvNodeToSingleGroups(const ConvBaseNode* n)
{
    TensorPtr IFM         = n->getXOperand();
    TensorPtr WGH         = n->getWOperand();
    TensorPtr OFM         = n->getYOperand();
    TensorPtr shapeTensor = n->getShapeOperand();

    unsigned nGroups = n->getConvolutionParams().nGroups;
    unsigned packedGroups = m_convNode->getConvolutionParams().nGroups / n->getConvolutionParams().nGroups;
    unsigned kPerGroup = WGH->getSizeInElements(WEIGHT_DIM_K) / nGroups;
    unsigned cPerGroup = IFM->getSizeInElements(DIM_C) / nGroups;

    auto newParams    = n->getConvolutionParams();
    newParams.nGroups = 1;

    TensorVector pNewIFMs;
    TensorVector pNewOFMs;
    TensorVector pNewWGHs;
    TensorVector pNewShapeTensors;

    auto [ifmMinSizes, ifmMaxSizes] = IFM->getAllMinMaxSizesInElements();
    auto [wghMinSizes, wghMaxSizes] = WGH->getAllMinMaxSizesInElements();
    auto [ofmMinSizes, ofmMaxSizes] = OFM->getAllMinMaxSizesInElements();

    SizeArray shapeTensorMaxSizes;
    SizeArray shapeTensorMinSizes;
    if (shapeTensor)
    {
        shapeTensorMaxSizes = shapeTensor->getAllSizesInElements();
        shapeTensorMinSizes = shapeTensor->getAllMinimalSizesInElements();
    }
    NodeList newNodes;

    NodePtr existingIFMSplit = tryReuseExistingSplitNode(n, IFM, DIM_C, nGroups);
    NodePtr existingOFMSplit = tryReuseExistingSplitNode(n, OFM, DIM_C, nGroups);
    NodePtr existingWGHSplit = tryReuseExistingSplitNode(n, WGH, WEIGHT_DIM_K, nGroups);

    // Create nGroups nodes of the original node type
    for (unsigned groupIdx = 0; groupIdx < nGroups; ++groupIdx)
    {
        ifmMaxSizes[DIM_C]        = cPerGroup;
        ofmMaxSizes[DIM_C]        = kPerGroup;
        wghMaxSizes[WEIGHT_DIM_K] = kPerGroup;
        ifmMinSizes[DIM_C]        = cPerGroup;
        ofmMinSizes[DIM_C]        = kPerGroup;
        wghMinSizes[WEIGHT_DIM_K] = kPerGroup;
        if (shapeTensor)
        {
            shapeTensorMaxSizes[DIM_C] = cPerGroup;
            shapeTensorMinSizes[DIM_C] = cPerGroup;
        }

        TensorPtr pNewIFM;
        TensorPtr pNewOFM;
        TensorPtr pNewWGH;
        TensorPtr pNewShape;
        NodePtr   pNewNode;
        std::string nodeName = n->getNodeName() + "_sub_" + std::to_string(groupIdx);

        if (existingIFMSplit)
        {
            pNewIFM = existingIFMSplit->getOutput(groupIdx);
        }
        else
        {
            pNewIFM =
                std::make_shared<Tensor>(IFM->getDim(), ifmMaxSizes.data(), IFM->getElementType(), ifmMinSizes.data());
            pNewIFM->setName(IFM->getName() + "_ifm_subgroup_" + std::to_string(groupIdx));
            pNewIFM->setAllQuantizationParams(IFM->getAllQuantizationParams());
            pNewIFM->setDynamicRange(IFM->getDynamicRange());
            pNewIFMs.push_back(pNewIFM);
        }

        if (existingOFMSplit)
        {
            pNewOFM = existingOFMSplit->getOutput(groupIdx);
        }
        else
        {
            pNewOFM =
                std::make_shared<Tensor>(OFM->getDim(), ofmMaxSizes.data(), OFM->getElementType(), ofmMinSizes.data());
            pNewOFM->setName(OFM->getName() + "_ofm_subgroup_" + std::to_string(groupIdx));
            pNewOFM->setAllQuantizationParams(OFM->getAllQuantizationParams());
            pNewOFM->setDynamicRange(OFM->getDynamicRange());
            pNewOFMs.push_back(pNewOFM);
        }

        if (existingWGHSplit)
        {
            pNewWGH = existingWGHSplit->getOutput(groupIdx);
        }
        else
        {
            pNewWGH =
                std::make_shared<Tensor>(WGH->getDim(), wghMaxSizes.data(), WGH->getElementType(), wghMinSizes.data());
            pNewWGH->setName(WGH->getName() + "_weights_subgroup_" + std::to_string(groupIdx));
            pNewWGH->setAllQuantizationParams(WGH->getAllQuantizationParams());
            pNewWGH->setDynamicRange(WGH->getDynamicRange());
            pNewWGH->setAsWeights();
            pNewWGHs.push_back(pNewWGH);
        }

        if (shapeTensor)
        {
            pNewShape = std::make_shared<Tensor>(shapeTensor->getDim(),
                                                 shapeTensorMaxSizes.data(),
                                                 shapeTensor->getElementType(),
                                                 shapeTensorMinSizes.data());
            pNewShape->setName(shapeTensor->getName() + "_shape_subgroup_" + std::to_string(groupIdx));
            pNewShape->setAllQuantizationParams(shapeTensor->getAllQuantizationParams());
            pNewShape->setDynamicRange(shapeTensor->getDynamicRange());
            pNewShape->setTensorType(shapeTensor->getTensorType());
            pNewShapeTensors.push_back(pNewShape);
        }


        TensorVector inputs;
        TensorVector outputs;
        if (n->getNodeType() == Node::TYPE_CONVOLUTION)
        {
            inputs.push_back(pNewIFM);
            inputs.push_back(pNewWGH);
            outputs.push_back(pNewOFM);
        }
        else if (n->getNodeType() == Node::TYPE_DEDW)
        {
            inputs.push_back(pNewOFM);
            inputs.push_back(pNewIFM);
            outputs.push_back(pNewWGH);
        }
        else  // n->getNodeType() == Node::TYPE_DEDX
        {
            inputs.push_back(pNewOFM);
            inputs.push_back(pNewWGH);
            outputs.push_back(pNewIFM);
            if (pNewShape)
            {
                inputs.push_back(pNewShape);
            }
        }

        if (n->is3DConvolution())
        {
            pNewNode = NodeFactory::createNode(inputs, outputs, &newParams, n->getGUID(), nodeName);
        }
        else
        {
            synConvolutionParamsV2 params2D = MmeNode::convert3DconvTo2DconvStruct(newParams);
            pNewNode = NodeFactory::createNode(inputs, outputs, &params2D, n->getGUID(), nodeName);
        }
        pNewNode->getNodeAnnotation().mmeMetaData.groupPacking = packedGroups > 1;
        newNodes.push_back(pNewNode);
    }

    createSplitConcatNodes(n, pNewIFMs, pNewOFMs, pNewWGHs, pNewShapeTensors, IFM, OFM, WGH, shapeTensor, newNodes);
    return newNodes;
}

bool handleGroupedConvolutions(HabanaGraph &g)
{
    GroupedConvolutionManagerTraining groupedConvolutionMgr;
    return groupedConvolutionMgr.runHandleGroupedConvolutions(g);
}
