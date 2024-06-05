#include <vector>
#include "defs.h"
#include "fuse_transpose_mme.h"

#include "habana_graph.h"
#include "habana_pass.h"
#include "node_factory.h"
#include "graph_editor.h"
#include "synapse_common_types.h"
#include "tensor.h"
#include "passes/sram_management/slicing_utils.h"
#include "dma_transpose_node.h"
#include "slicing_brain.h"

bool TransposeFuserBase::isTransposeNode(const NodePtr& node) const
{
    // strided DMA via transpose is not use for transposing, just as a dma,
    // for example to bring data from dram to the MME operation in SRAM
    return node->isTranspose() && !std::dynamic_pointer_cast<StridedDMANodeViaTransposeNode>(node);
}

const TransposePermutationArrayVec& TransposeFuserBase::getSupportedPermutations(const NodePtr& mmeNode)
{
    static const TransposePermutationArrayVec gemmSupportedPermutations = {{TPD_Width, TPD_Channel}};
    // We fuse only transposes with permutation that switch between FCD TPD_Channel (0)_and next FCD TPD_Width (1).
    // All other dims should remain the same.
    static const TransposePermutationArrayVec batchGemmemmSupportedPermutations = {
        {TPD_Width, TPD_Channel, TPD_Height, TPD_Depth, TPD_Batch}};
    // WHBC, WHDBC corresponding to mme operations which can be converted to a gemm \ batch gemm
    static const TransposePermutationArrayVec convBasedSupportedPermutations = {
        {TPD_Width, TPD_Height, TPD_4Dim_Batch, TPD_Channel},
        {TPD_Width, TPD_Height, TPD_Depth, TPD_Batch, TPD_Channel}};

    // return the corresponding supported permutations based on node type
    if (Node::isGemmNode(mmeNode)) return gemmSupportedPermutations;
    if (Node::isBatchGemmNode(mmeNode)) return batchGemmemmSupportedPermutations;
    return convBasedSupportedPermutations;
}

bool TransposeFuserBase::isValidPermutation(const TransposePermutationArray&    permutation,
                                            const TransposePermutationArrayVec& supportedPermutations)
{
    bool isValidPermutation;
    for (const auto& supportedPermutation : supportedPermutations)
    {
        isValidPermutation = true;
        if (permutation.size() > supportedPermutation.size())
        {
            continue;
        }

        for (int i = 0; i < permutation.size(); ++i)
        {
            if (supportedPermutation[i] != permutation[i])
            {
                isValidPermutation = false;
                break;
            }
        }

        if (isValidPermutation) return true;
    }

    // Node permutation is not one of the supported permutations
    return false;
}

bool TransposeFuserBase::isValidTransposePermutation(const NodePtr& transposeNode, const NodePtr& mmeNode) const
{
    TransposePermutationArrayVec   supportedPermutations = getSupportedPermutations(mmeNode);
    std::shared_ptr<TransposeNode> internalTransposeNode = std::dynamic_pointer_cast<TransposeNode>(transposeNode);
    if (!internalTransposeNode)
    {
        std::shared_ptr<DMATransposeNode> dmaTransposeNode = std::dynamic_pointer_cast<DMATransposeNode>(transposeNode);
        if (!dmaTransposeNode)
        {
            std::shared_ptr<MmeTransposeNode> mmeTransposeNode =
                std::dynamic_pointer_cast<MmeTransposeNode>(transposeNode);
            if (!mmeTransposeNode) return false;
            LOG_TRACE(GC,
                      "{}: The transpose node is a MME transpose node = {}",
                      HLLOG_FUNC,
                      mmeTransposeNode->getNodeName());
            return isValidPermutation(mmeTransposeNode->permutation(), supportedPermutations);
        }

        LOG_TRACE(GC,
                  "{}: The transpose node is a DMA transpose node = {}",
                  HLLOG_FUNC,
                  dmaTransposeNode->getNodeName());
        return isValidPermutation(dmaTransposeNode->permutation(), supportedPermutations);
    }

    LOG_TRACE(GC,
              "{}: The transpose node is a regular transpose node = {}",
              HLLOG_FUNC,
              internalTransposeNode->getNodeName());
    return isValidPermutation(internalTransposeNode->permutation(), supportedPermutations);
}

bool TransposeFuser::canConvertToGemm(const NodePtr& mmeNode)
{
    // block masked bgemm fusion, as the masks are also transposed like the operands
    if (mmeNode->getNodeType() == Node::TYPE_MASKED_BATCH_GEMM)
    {
        return false;
    }

    if (Node::isGemmNode(mmeNode) || Node::isBatchGemmNode(mmeNode))
    {
        return true;
    }

    // Check is filter is (1,1), pad values equal to 0 and all other params equal to 1
    unsigned sIndex;
    unsigned rIndex;
    gc::Layout weightsLayout;
    TensorPtr  weights;
    if (mmeNode->getNodeType() == Node::TYPE_DEDW)
    {
        weightsLayout = mmeNode->getOutputLayouts()[TENSOR_OFM];
        weights       = mmeNode->getOutput(TENSOR_OFM);
    }
    else
    {
        weightsLayout = mmeNode->getInputLayouts()[TENSOR_WEIGHT];
        weights       = mmeNode->getInput(TENSOR_WEIGHT);
    }
    if (weightsLayout == gc::Layout(""))
    {
        sIndex = TPD_Weights_S;
        rIndex = TPD_Weights_R;
    }
    else
    {
        sIndex = weightsLayout.getIndexByName('S');
        rIndex = weightsLayout.getIndexByName('R');
    }

    unsigned sSize = weights->getSizeInElements(sIndex);
    unsigned rSize = weights->getSizeInElements(rIndex);

    if (sSize != 1 || rSize != 1)
    {
        return false;
    }
    bool canConvert = std::dynamic_pointer_cast<MmeNode>(mmeNode)->canBeConvertedToGEMM();
    canConvert      = canConvert && checkInputsDim(mmeNode);
    return canConvert;
}

bool TransposeFuser::checkInputsDim(const NodePtr& mmeNode)
{
    bool ifm    = SlicedOperandUtils::isTensor2D(mmeNode->getInput(TENSOR_IFM));
    bool weight = SlicedOperandUtils::isTensor2D(mmeNode->getInput(TENSOR_WEIGHT));

    return (ifm && weight);
}

void TransposeFuser::checkTransposeA(HabanaGraph& g, const NodePtr& mmeNode)
{
    NodePtr                      mmeProducerNodeA = g.getTensorProducer(mmeNode->getInput(TENSOR_IFM));
    NodePtr                      identityA        = nullptr;
    if (mmeProducerNodeA != nullptr && mmeProducerNodeA->getNodeType() == Node::TYPE_IDENTITY &&
        GCFG_ENABLE_FUSE_IDENTITY_TRANSPOSE_INTO_MME.value())
    {
        if (GraphEditor::canEliminateTensor(g, mmeProducerNodeA->getOutput(TENSOR_OFM)))
        {
            identityA        = mmeProducerNodeA;
            mmeProducerNodeA = g.getTensorProducer(mmeProducerNodeA->getInputs()[TENSOR_IFM]);
        }
        else
        {
            LOG_TRACE(GC,
                      "{}: Didn't fuse node {} inputA due to identity node {} that cannot be eliminated",
                      HLLOG_FUNC,
                      mmeNode->getNodeName(),
                      mmeProducerNodeA->getNodeName());
            mmeProducerNodeA = nullptr;
        }
    }
    if (mmeProducerNodeA != nullptr && isTransposeNode(mmeProducerNodeA))
    {
        if (isValidTransposePermutation(mmeProducerNodeA, mmeNode))
        {
            unsigned numConsumers = 1;
            if (identityA == nullptr)
            {
                numConsumers = g.getNumberOfTensorConsumers(mmeProducerNodeA->getOutput(TENSOR_OFM));
            }

            if (GraphEditor::canEliminateTensor(g, mmeProducerNodeA->getOutput(TENSOR_OFM), numConsumers))
            {
                m_actions |= TRANSPOSE_A_INPUT | REPLACE_A_INPUT;
                if (numConsumers == 1)
                {
                    m_actions |= REMOVE_TRANSPOSE_A_NODE;
                }
            }
            else
            {
                LOG_TRACE(GC,
                          "{}: Didn't fuse node {} inputA due to {} consumers",
                          HLLOG_FUNC,
                          mmeNode->getNodeName(),
                          mmeProducerNodeA->getNodeName());
            }
        }
        else
        {
            LOG_TRACE(GC,
                      "{}: Didn't fuse node {} inputA - current permutation isn't supported",
                      HLLOG_FUNC,
                      mmeNode->getNodeName());
        }
    }
    m_mmeProducerA = {mmeProducerNodeA, identityA};
}

void TransposeFuser::checkTransposeB(HabanaGraph& g, const NodePtr& mmeNode)
{
    NodePtr                      mmeProducerNodeB = g.getTensorProducer(mmeNode->getInput(TENSOR_WEIGHT));
    NodePtr                      identityB        = nullptr;
    if (mmeProducerNodeB != nullptr && mmeProducerNodeB->getNodeType() == Node::TYPE_IDENTITY &&
        GCFG_ENABLE_FUSE_IDENTITY_TRANSPOSE_INTO_MME.value())
    {
        if (GraphEditor::canEliminateTensor(g, mmeProducerNodeB->getOutput(TENSOR_OFM)))
        {
            identityB        = mmeProducerNodeB;
            mmeProducerNodeB = g.getTensorProducer(mmeProducerNodeB->getInputs()[TENSOR_IFM]);
        }
        else
        {
            LOG_TRACE(GC,
                      "{}: Didn't fuse node {} inputB due to identity node {} that cannot be eliminated",
                      HLLOG_FUNC,
                      mmeNode->getNodeName(),
                      mmeProducerNodeB->getNodeName());
            mmeProducerNodeB = nullptr;
        }
    }

    if (mmeProducerNodeB != nullptr && isTransposeNode(mmeProducerNodeB))
    {
        if (isValidTransposePermutation(mmeProducerNodeB, mmeNode))
        {
            unsigned numConsumers = 1;
            if (identityB == nullptr)
            {
                numConsumers = g.getTensorConsumers(mmeProducerNodeB->getOutput(TENSOR_OFM)).size();
            }

            if (GraphEditor::canEliminateTensor(g, mmeProducerNodeB->getOutput(TENSOR_OFM), numConsumers))
            {
                m_actions |= TRANSPOSE_B_INPUT | REPLACE_B_INPUT;
                if (numConsumers == 1)
                {
                    m_actions |= REMOVE_TRANSPOSE_B_NODE;
                }
            }
            else
            {
                LOG_TRACE(GC,
                          "{}: Didn't fuse node {} inputB due to {} consumers",
                          HLLOG_FUNC,
                          mmeNode->getNodeName(),
                          mmeProducerNodeB->getNodeName());
            }
        }
        else
        {
            LOG_TRACE(GC,
                      "{}: Didn't fuse node {} inputB - current permutation isn't supported",
                      HLLOG_FUNC,
                      mmeNode->getNodeName());
        }
    }
    m_mmeProducerB = {mmeProducerNodeB, identityB};
}

// Should be removed once [SW-95692] is done
// This function evaluates if a slicing strategy with double buffer is optional for the given gemm node
// Returns true if buffer size * 4 (factor) is smaller than half sram size
static bool isGemmSlicingOptimized(HabanaGraph& g, const NodePtr& mmeNode)
{
    bool isGaudi1Platform = g.getDeviceType() == synDeviceGaudi;
    if (!isGaudi1Platform)  // relevant for gaudi1 only
    {
        return true;
    }

    HB_ASSERT(Node::isGemmNode(mmeNode), "mmeNode is not GEMM");
    SlicingBrain dummyBrain(g);  // required to intialize SlicingBrain::knobs
    TensorPtr    inputA = mmeNode->getInput(0);
    TensorPtr    output = mmeNode->getOutput(0);
    synDataType  dType  = inputA->getElementType();

    std::shared_ptr<GEMMNode> gemmNode = std::dynamic_pointer_cast<GEMMNode>(mmeNode);
    TSize cd = gemmNode->getGEMMParams().transpose_a ? inputA->getSizeInElements(1) : inputA->getSizeInElements(0);

    TSize factor = 4;
    TSize minimalNonCdGeometry =
        std::min((TSize)(g.getHALReader()->getMmeMinimalWidthInElems(dType)), output->getSizeInElements(0)) +
        std::min((TSize)(g.getHALReader()->getMmeMinimalWidthInElems(dType)), output->getSizeInElements(1));
    return (cd * dataTypeSizeInBytes(dType) * minimalNonCdGeometry * factor) <
           SlicingBrain::knobs.maxSRAMCapInBytes / 2;
}

void TransposeFuser::checkTransposeOut(HabanaGraph& g, const NodePtr& mmeNode)
{
    NodePtr mmeConsumerNode = nullptr;
    NodePtr identity        = nullptr;

    // transpose on output is currently fused only in the case of batch gemm
    if (Node::isBatchGemmNode(mmeNode) || Node::isGemmNode(mmeNode))
    {
        NodeList mmeConsumerNodes = g.getTensorConsumers(mmeNode->getOutput(TENSOR_OFM));
        if (mmeConsumerNodes.size() == 1)
        {
            mmeConsumerNode = mmeConsumerNodes.front();
            if (mmeConsumerNode->getNodeType() == Node::TYPE_IDENTITY &&
                GCFG_ENABLE_FUSE_IDENTITY_TRANSPOSE_INTO_MME.value())
            {
                if (GraphEditor::canEliminateTensor(g, mmeConsumerNode->getOutput(TENSOR_OFM)))
                {
                    identity        = mmeConsumerNode;
                    mmeConsumerNodes = g.getTensorConsumers(mmeConsumerNode->getOutput(TENSOR_OFM));
                    mmeConsumerNode  = mmeConsumerNodes.empty() ? nullptr : mmeConsumerNodes.front();
                }
                else
                {
                    LOG_TRACE(GC,
                              "{}: Didn't fuse node {} output due to identity node {} that cannot be eliminated",
                              HLLOG_FUNC,
                              mmeNode->getNodeName(),
                              mmeConsumerNode->getNodeName());
                    mmeConsumerNode = nullptr;
                }
            }

            if (mmeConsumerNode && isTransposeNode(mmeConsumerNode) &&
                (Node::isBatchGemmNode(mmeNode) ||
                 (GCFG_ENABLE_FUSE_TRANSPOSE_TO_GEMM_OUTPUT.value() &&
                  isGemmSlicingOptimized(g, mmeNode))))  // TODO: remove once [SW-95692] is done
            {
                if (isValidTransposePermutation(mmeConsumerNode, mmeNode))
                {
                    if (GraphEditor::canEliminateTensor(g, mmeConsumerNode->getInput(TENSOR_IFM)))
                    {
                        m_actions |= SWAP_INPUTS | REPLACE_OUTPUT;
                        // We perform XOR with transposing inputs to aggregate transposes
                        m_actions ^= TRANSPOSE_A_INPUT | TRANSPOSE_B_INPUT;
                    }
                    else
                    {
                        LOG_TRACE(GC,
                                  "{}: Didn't fuse node {} output due to {} consumers",
                                  HLLOG_FUNC,
                                  mmeNode->getNodeName(),
                                  mmeConsumerNode->getNodeName());
                    }
                }
                else
                {
                    LOG_TRACE(GC,
                              "{}: Didn't fuse node {} output - current permutation isn't supported",
                              HLLOG_FUNC,
                              mmeNode->getNodeName());
                }
            }
        }
    }
    m_mmeConsumer = {mmeConsumerNode, identity};
}

bool TransposeFuser::fuseTransposeIntoMmeNode(HabanaGraph& g, const NodePtr& mmeNode)
{
    if (!canConvertToGemm(mmeNode))
    {
        LOG_TRACE(GC,
                  "{}: Didn't fuse node {} because it can't be converted to gemm",
                  HLLOG_FUNC,
                  mmeNode->getNodeName());
        return false;
    }

    m_actions = NOOP;
    checkTransposeA(g, mmeNode);
    checkTransposeB(g, mmeNode);
    checkTransposeOut(g, mmeNode);

    if (m_actions == NOOP)
    {
        // no fusion needed
        return false;
    }

    fuse(g, mmeNode);
    LOG_DEBUG(GC,
              "fuseTransposeMme: fused node {}, performed the following action(s): {}",
              mmeNode->getNodeName(),
              actionToStr());
    return true;
}

void TransposeFuser::fuse(HabanaGraph& graph, const NodePtr& mmeNode)
{
    // create a new GEMM node
    std::shared_ptr<GEMMNode> newGemmNode;
    if (Node::isGemmNode(mmeNode) || Node::isBatchGemmNode(mmeNode))
    {
        newGemmNode = std::dynamic_pointer_cast<GEMMNode>(mmeNode->clone());
    }
    else
    {
        pTensor gemmIfm     = mmeNode->getInput(TENSOR_IFM);
        pTensor gemmWeights = mmeNode->getInput(TENSOR_WEIGHT);
        NodePtr newNode     = NodeFactory::createNode({gemmIfm, gemmWeights},
                                                  {mmeNode->getOutput(TENSOR_OFM)},
                                                  nullptr,
                                                  NodeFactory::gemmNodeTypeName,
                                                  mmeNode->getNodeName());
        newGemmNode         = std::dynamic_pointer_cast<GEMMNode>(newNode);
    }
    synGEMMParams newGemmParams = newGemmNode->getGEMMParams();
    // fuse the transposes to the gemm node
    m_oldNodes.clear();
    m_newNodes.clear();

    doTransposeA(newGemmNode, newGemmParams);
    doTransposeB(newGemmNode, newGemmParams);
    doSwapInputs(newGemmNode, newGemmParams);
    doReplaceOutput(newGemmNode);

    MMENodePtr mmeNodePtr = std::dynamic_pointer_cast<MmeNode>(mmeNode);
    HB_ASSERT(mmeNodePtr, "could not downcast Node to MME Node");
    const MmeExpBias& MmeExpBias = mmeNodePtr->getMmeExpBias();
    newGemmNode->setMmeExpBias(MmeExpBias);

    newGemmNode->setGEMMParams(newGemmParams);
    // replace gemm with original node.
    m_oldNodes.push_back(mmeNode);
    m_newNodes.push_back(newGemmNode);
    // need to remove repeated transpose node for input in case both transposed operands
    // are the same, otherwise replaceNodes explodes.
    auto last = std::unique(m_oldNodes.begin(), m_oldNodes.end());
    m_oldNodes.erase(last, m_oldNodes.end());
    auto status = GraphEditor::replaceNodes(graph, m_oldNodes, m_newNodes);
    HB_ASSERT(status == REPLACE_NODE_SUCCESS, "failed fuseTransposeIntoMmeNode");
}

void TransposeFuser::doTransposeA(const std::shared_ptr<GEMMNode>& newGemmNode, synGEMMParams& newGemmParams)
{
    if ((m_actions & TRANSPOSE_A_INPUT) != 0)
    {
        newGemmParams.transpose_a = !newGemmParams.transpose_a;
        LOG_TRACE(GC, "{}: transposed OpA of mme node {}", HLLOG_FUNC, newGemmNode->getNodeName());
    }
    if ((m_actions & REPLACE_A_INPUT) != 0)
    {
        TensorPtr transposeInput = m_mmeProducerA[0]->getInput(TENSOR_IFM);
        newGemmNode->replaceInput(TENSOR_IFM, transposeInput);

        if ((m_actions & REMOVE_TRANSPOSE_A_NODE) != 0)
        {
            m_oldNodes.push_back(m_mmeProducerA[0]);
            if (m_mmeProducerA[1])
            {
                // producers[0] is transpose node, producers[1] is a node that comes between transpose and mme, that
                // will be eliminated
                m_oldNodes.push_back(m_mmeProducerA[1]);
            }
        }
        LOG_TRACE(GC,
                  "{}: Fused OpA transpose node {} into mme node {}",
                  HLLOG_FUNC,
                  m_mmeProducerA[0]->getNodeName(),
                  newGemmNode->getNodeName());
    }
}

void TransposeFuser::doTransposeB(const std::shared_ptr<GEMMNode>& newGemmNode, synGEMMParams& newGemmParams)
{
    if ((m_actions & TRANSPOSE_B_INPUT) != 0)
    {
        newGemmParams.transpose_b = !newGemmParams.transpose_b;
        LOG_TRACE(GC, "{}: transposed OpB of mme node {}", HLLOG_FUNC, newGemmNode->getNodeName());
    }
    if ((m_actions & REPLACE_B_INPUT) != 0)
    {
        TensorPtr transposeInput = m_mmeProducerB[0]->getInput(TENSOR_IFM);
        newGemmNode->replaceInput(TENSOR_WEIGHT, transposeInput);

        if ((m_actions & REMOVE_TRANSPOSE_B_NODE) != 0)
        {
            m_oldNodes.push_back(m_mmeProducerB[0]);

            if (m_mmeProducerB[1])
            {
                // producers[0] is transpose node, producers[1] is a node that comes between transpose and mme, that
                // will be eliminated
                m_oldNodes.push_back(m_mmeProducerB[1]);
            }
        }
        LOG_TRACE(GC,
                  "{}: Fused OpB transpose node {} into mme node {}",
                  HLLOG_FUNC,
                  m_mmeProducerB[0]->getNodeName(),
                  newGemmNode->getNodeName());
    }
}

void TransposeFuser::doSwapInputs(const std::shared_ptr<GEMMNode>& newGemmNode, synGEMMParams& newGemmParams)
{
    if ((m_actions & SWAP_INPUTS) != 0)
    {
        std::swap(newGemmParams.transpose_a, newGemmParams.transpose_b);

        TensorPtr xTensor = newGemmNode->getInput(TENSOR_IFM);
        TensorPtr wTensor = newGemmNode->getInput(TENSOR_WEIGHT);

        newGemmNode->replaceInput(TENSOR_WEIGHT, xTensor);
        newGemmNode->replaceInput(TENSOR_IFM, wTensor);
        LOG_TRACE(GC, "{}: Swap inputs of mme node {}", HLLOG_FUNC, newGemmNode->getNodeName());
    }
}

void TransposeFuser::doReplaceOutput(const std::shared_ptr<GEMMNode>& newGemmNode)
{
    if ((m_actions & REPLACE_OUTPUT) != 0)
    {
        if (m_mmeConsumer[1])
        {
            // consumer[0] is transpose node, consumer[1] is a node that comes between mme and transpose, that will be
            // eliminated
            m_oldNodes.push_back(m_mmeConsumer[1]);
        }

        TensorPtr transposeOutput = m_mmeConsumer[0]->getOutput(TENSOR_OFM);
        newGemmNode->replaceOutput(TENSOR_OFM, transposeOutput);
        m_oldNodes.push_back(m_mmeConsumer[0]);
        LOG_TRACE(GC,
                  "{}: Fused output transpose node {} into mme node {}",
                  HLLOG_FUNC,
                  m_mmeConsumer[0]->getNodeName(),
                  newGemmNode->getNodeName());
    }
}

// Find mme nodes with atleast one transpose before an input operand or after output operand.
NodeVector TransposeFuseCandidateFinder::findCandidates(HabanaGraph& graph)
{
    NodeVector candidates = {};
    for (auto& node : graph.getSortedMMENodes())
    {
        bool canFuse = checkInputCandidates(node, graph);
        if (!canFuse)
        {
            canFuse = checkOutputCandidates(node, graph);
        }

        if (canFuse)
        {
            candidates.push_back(node);
        }
    }
    return candidates;
}

bool TransposeFuseCandidateFinder::checkInputCandidates(const NodePtr& node, const HabanaGraph& graph)
{
    bool canFuse = false;
    for (auto input : node->getInputs())
    {
        NodePtr producer = graph.getTensorProducer(input);
        if (!producer) continue;

        if (producer->getNodeType() == Node::TYPE_IDENTITY && GCFG_ENABLE_FUSE_IDENTITY_TRANSPOSE_INTO_MME.value())
        {
            producer = graph.getTensorProducer(producer->getInput(0));
        }

        if (producer && isTransposeNode(producer))
        {
            canFuse = true;
            break;
        }
    }
    return canFuse;
}

bool TransposeFuseCandidateFinder::checkOutputCandidates(const NodePtr& node, const HabanaGraph& graph)
{
    auto outputConsumers = graph.getTensorConsumers(node->getOutput(0));

    // TODO - [SW-116827] support fuseTransposeMme with bias / Cin.
    auto mmeNodePtr = dynamic_cast<MmeNode*>(node.get());
    HB_ASSERT_PTR(mmeNodePtr);
    if (mmeNodePtr->hasBias() || mmeNodePtr->hasCin()) return false;

    if (outputConsumers.size() == 1)
    {
        NodePtr consumer = outputConsumers.front();
        if (consumer->getNodeType() == Node::TYPE_IDENTITY && GCFG_ENABLE_FUSE_IDENTITY_TRANSPOSE_INTO_MME.value())
        {
            auto identityConsumers = graph.getTensorConsumers(consumer->getOutput(0));
            if (identityConsumers.size() == 1)
            {
                consumer = identityConsumers.front();
            }
            else
            {
                return false;
            }
        }
        if (consumer && isTransposeNode(consumer))
        {
            return true;
        }
    }
    return false;
}

std::string TransposeFuser::actionToStr() const
{
    std::stringstream ss;
    if (m_actions & TRANSPOSE_A_INPUT)
    {
        ss << "Transpose input A ";
    }
    if (m_actions & TRANSPOSE_B_INPUT)
    {
        ss << "Transpose input B ";
    }
    if (m_actions & TRANSPOSE_OUTPUT)
    {
        ss << "Transpose output";
    }
    return ss.str();
}

bool fuseTransposeMme(HabanaGraph& g)
{
    // find mme nodes suitable for fusing
    TransposeFuseCandidateFinder finder;
    NodeVector                   candidates = finder.findCandidates(g);

    std::set<unsigned> fusedNodesIds;
    for (auto mmeNode : candidates)
    {
        if (MmeNode::isDmaOperation(mmeNode))
        {
            continue;
        }

        // verify MME node was not already fused in a previous pattern
        if (fusedNodesIds.find(mmeNode->getId()) != fusedNodesIds.end())
        {
            continue;
        }

        TransposeFuser fuser;
        bool isFused = fuser.fuseTransposeIntoMmeNode(g, mmeNode);
        if (isFused)
        {
            fusedNodesIds.insert(mmeNode->getId());
        }
    }
    if (fusedNodesIds.size() > 0)
    {
        g.turnOnPredicate(PREDICATE_ID_FUSED_NODE_TO_MME);
    }
    return true;
}
