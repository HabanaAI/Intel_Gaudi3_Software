#include <graph_editor.h>
#include <unistd.h>
#include "flatten_mme.h"
#include "log_manager.h"
#include "node_factory.h"
#include "habana_global_conf.h"
#include "slicing_utils.h"

bool MMENodeFlattener::execute()
{
    if (GCFG_SRAM_SLICER_MAX_CAPACITY_BYTES.value() == 0UL)
    {
        return true;
    }
    //An empty list just for testing
    std::vector<pSlicedOperand> slicedOperands;
    bool status = true;
    NodeVector                  nodes  = m_graph.getExeSortedNodes();
    for (const pNode& node : nodes)
    {
        if (canFlattenMMENode(node))
        {
            std::tie(status, std::ignore) = doFlatten(node, slicedOperands);
            if (!status)
            {
                LOG_ERR(GC, "Cannot flatten node - {}", node->getNodeName());
                break;
            }
        }
    }
    return status;
}

std::pair<bool, NodeSet> MMENodeFlattener::doFlatten(const NodePtr& node, std::vector<pSlicedOperand>& slicedOperands)
{
    LOG_TRACE(GC, "Trying to flatten Node - {}", node->getNodeName());
    GraphEditor::removeNode(m_graph, node);
    NodeSet newReshapeNodes {};
    for (pTensor& operand : node->getOperands())
    {
        if (!operand) continue;
        if (SlicedOperandUtils::isTensor2D(operand)) continue;
        const auto statusAndNewReshapeNode = doFlattenAndAddReshapeNode(node, operand, slicedOperands);
        if (statusAndNewReshapeNode.first /* status */ &&
            statusAndNewReshapeNode.second /* newReshapeNode */ != nullptr)
        {
            newReshapeNodes.insert(statusAndNewReshapeNode.second);
        }
    }
    // status is considered failure only if we can't reinsert the original node to the graph.
    // [CID: 42784] False positive - coverity ignores std::set default c'tor
    return std::make_pair(GraphEditor::addNode(m_graph, node), newReshapeNodes);
}

void MMENodeFlattener::copyOperandAttributes(const TensorPtr& operand, TensorPtr& flatTensor)
{
    if (operand->isShapeTensor())
    {
        flatTensor->setShapeTensor(operand->getTensorType());
    }

    // If operand is in sram, flattened tensor should also be in sram
    if (operand->inSram())
    {
        flatTensor->setTensorInSram();
    }

    flatTensor->setName(fmt::format("{}_flattened", operand->getName()));
}

std::pair<bool, NodePtr> MMENodeFlattener::doFlattenAndAddReshapeNode(const pNode&                 node,
                                                                      pTensor&                     operand,
                                                                      std::vector<pSlicedOperand>& slicedOperands)
{
    bool status = true;
    SizeArray flatShape = getFlattenShape(operand);
    SizeArray minSizes  = getFlattenShape(operand, true);
    bool isOutput = isOutputTensorOfNode(node, operand);
    LOG_DEBUG(GC, "Flattening {} tensor {} from shape {} to shape {}",
              isOutput ? "output" : "input", operand->getName(),
              toString(operand->getAllSizesInElements(), 'x'), toString(flatShape, 'x'));
    TensorPtr flatTensor(nullptr);
    NodePtr   reshapeNode(nullptr);

    if (!isOutput && !node->isDynamicShape())
    {
        flatTensor = getAlreadyFlattenTensor(operand, flatShape);
    }
    if (!flatTensor)
    {
        // When node is dynamic or reusable flattened tensor not found, create one and reshape accordingly
        LOG_TRACE(GC,"Adding reshape to {} tensor {} of node {}", isOutput ? "output" : "input",
                  operand->getName(), node->getNodeName());

        flatTensor =
            std::make_shared<Tensor>(TensorShape {operand->getDim(), flatShape, minSizes}, operand->getElementType());

        copyOperandAttributes(operand, flatTensor);
        std::string nodeName = fmt::format("reshape_{}", operand->getName());
        try
        {
            if (!isOutput)
            {
                reshapeNode = NodeFactory::createNode({operand},
                                                      {flatTensor},
                                                      nullptr,
                                                      NodeFactory::staticReshapeNodeTypeName,
                                                      nodeName);
                status      = GraphEditor::addNode(m_graph, reshapeNode);
            }
            else
            {
                reshapeNode = NodeFactory::createNode({flatTensor},
                                                      {operand},
                                                      nullptr,
                                                      NodeFactory::staticReshapeNodeTypeName,
                                                      nodeName);
                status      = GraphEditor::addNode(m_graph, reshapeNode);
            }
        }
        catch (std::exception& e)
        {
            LOG_WARN(GC, "caught exception creating reshape node - {}", e.what());
        }
    }
    if (!status)
    {
        LOG_WARN(GC, "Cannot flatten Tensor {}", operand->getName());
    }
    else
    {
        if (!slicedOperands.empty())
        {
            for (auto& slicedOperand : slicedOperands)
            {
                if (slicedOperand->originalTensor == operand)
                {
                    slicedOperand->originalTensor = flatTensor;
                }
            }
        }
        node->replaceTensor(operand, flatTensor);
    }
    return std::make_pair(status, reshapeNode);
}

SizeArray MMENodeFlattener::getFlattenShape(pTensor& operand, bool minSizes)
{
    static constexpr DimsIndex fcd = DIM_C /* = 0*/;
    SizeArray                  origDimSize =
        (minSizes == false) ? operand->getAllSizesInElements() : operand->getAllMinimalSizesInElements();
    unsigned loweredDimSize = multiplyElements(origDimSize.data() + getLoweredDim(),
                                               origDimSize.data() + operand->getDim());
    SizeArray flattenShape;
    flattenShape.fill(1);
    flattenShape[fcd] = origDimSize[fcd];
    flattenShape[getLoweredDim()] = loweredDimSize;
    return flattenShape;
}

pTensor MMENodeFlattener::getAlreadyFlattenTensor(pTensor origTensor, SizeArray& flatShape)
{
    pTensor flatTensor = nullptr;
    NodeList consumers = m_graph.getTensorConsumers(origTensor);
    for (const pNode& consumer : consumers)
    {
        if (consumer->getNodeType() == Node::TYPE_INTERNAL_RESHAPE)
        {
            pTensor flatTensorCandidate = consumer->getOutputs().front();
            if (flatTensorCandidate->getAllSizesInElements() == flatShape &&
                flatTensorCandidate->getDim() == origTensor->getDim())
            {
                flatTensor = flatTensorCandidate;
                break;
            }
        }
    }
    return flatTensor;
}

std::pair<bool, NodePtr>
MMENodeFlattener::createReshapeNode(pTensor& fromTensor, pTensor& toTensor, const std::string& origTensorName)
{
    std::pair<bool, NodePtr> ret(false, nullptr);
    TensorVector             input, output;
    std::string              nodeName  = fmt::format("reshape_{}", origTensorName);
    std::string_view         guid      = NodeFactory::reshapeNodeTypeName;
    void*                    pParams   = nullptr;
    unsigned                 paramSize = 0;
    synFlattenParams flattenParams{}; // For shape tensor case
    if (fromTensor->isShapeTensor())
    {
        // Shape tensor is always an input.
        flattenParams.axis = 0;
        pParams            = &flattenParams;
        paramSize          = sizeof flattenParams;
        guid               = NodeFactory::flattenShapeNodeTypeName;
    }
    try
    {
        ret.second = NodeFactory::createNode({fromTensor}, {toTensor}, pParams, paramSize, guid, nodeName);
        ret.first  = GraphEditor::addNode(m_graph, ret.second);
    }
    catch (std::exception& e)
    {
        LOG_WARN(GC, "caught exception creating reshape node - {}", e.what());
    }
    return ret;
}

bool MMENodeFlattener::canFlattenMMENode(const pNode& node)
{
    if (HabanaGraph::runsOnMME(node) && node->getNodeType() != Node::TYPE_INTERNAL_TRANSPOSE)
    {
        bool isBatchGemmNode = node->isBatchGemm();
        if ((isBatchGemmNode && !GCFG_ENABLE_BGEMM_FLATTEN_TO_GEMM_FOR_SLICING.value()) ||
            (!isBatchGemmNode && !GCFG_ENABLE_CONV_FLATTEN_TO_GEMM_FOR_SLICING.value()))
        {
            return false;
        }

        std::shared_ptr<MmeNode> mmeNode = std::dynamic_pointer_cast<MmeNode>(node);
        HB_ASSERT(mmeNode, "Could not downcast to MME node");
        return mmeNode->canBeConvertedToGEMM();
    }
    return false;
}

bool MMENodeFlattener::isOutputTensorOfNode(const pNode& node, const pTensor& tensor)
{
    const TensorVector& outputs = node->getOutputs();
    return (std::find(outputs.begin(), outputs.end(), tensor) != outputs.end());
}
