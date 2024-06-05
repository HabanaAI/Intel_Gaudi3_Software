#include "algorithm"
#include "data_type_utils.h"
#include "gc_protocol_utils.hpp"
#include "synapse_graph_wrapper.hpp"

using namespace gc_protocol;
// Static identity permutation to be used for all tensors with identity permutation.
static const uint8_t s_identityPerm[Tensor::c_tensorMaxNDim] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                                                 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};

bool SynapseWrapperBase::foreachInputTensor(uint64_t nodeId, gc_protocol::ProtocolInputTensorHandler& handler) const
{
    return foreachTensor<gc_protocol::ProtocolInputTensorHandler>(nodeId, handler, true);
}

bool SynapseWrapperBase::foreachOutputTensor(uint64_t nodeId, gc_protocol::ProtocolOutputTensorHandler& handler) const
{
    return foreachTensor<gc_protocol::ProtocolOutputTensorHandler>(nodeId, handler, false);
}
/*
 * Iterate on node's tensors.
 * Foreach tensor -
 *  1. Create ProtocolTensor from it
 *  2. invoke ProtocolInput/OutputTensorHandler on the IR Tensor (for example to create MLIR tensor from it)
 */
template<typename THandler>
inline bool SynapseWrapperBase::foreachTensor(uint64_t nodeId, THandler& handler, bool isInput) const
{
    if (m_currentNode == nullptr)
    {
        return false;
    }

    LOG_TRACE(GC_TRANSLATION,
              "Iterating {} tensors of node {}",
              isInput ? "input" : "output",
              m_currentNode->getNodeName());

    PermutationVector&  inputPermutations = m_currentNode->getNodeAnnotation().inputPermutations;
    const TensorVector& tensors           = isInput ? m_currentNode->getInputs() : m_currentNode->getOutputs();
    unsigned            tensorIdx         = 0;
    for (const TensorPtr& tensor : tensors)
    {
        if (!tensor)
        {
            handler.acceptTensor(gc_protocol::InvalidId);
            tensorIdx++;
            continue;
        }
        LOG_TRACE(GC_TRANSLATION, "Invoking handler's acceptTensor");
        if (!handler.acceptTensor(tensor->getId()))
        {
            LOG_TRACE(GC_TRANSLATION, "Handler didn't accept tensor {} with ID {}", tensor->getName(), tensor->getId());
            tensorIdx++;
            continue;
        }
        // create ir Tensor
        m_tmpTensor = {};
        LOG_TRACE(GC_TRANSLATION,
                  "Synapse Wrapper - Creating Protocol IR tensor from tensor id - {}, "
                  "name - {}, data type - {}, tensor index {}",
                  tensor->getId(),
                  tensor->getName(),
                  getStringFromSynDataType(tensor->getElementType()),
                  tensorIdx);

        m_tmpTensor.id              = tensor->getId();
        m_tmpTensor.name            = tensor->getName();
        m_tmpTensor.elementDataType = toTpcLibDataType(tensor->getElementType());
        m_tmpTensor.rank            = tensor->getDim();
        m_tmpTensor.strides         = tensor->getNStridesInBytes();
        m_tmpTensor.maxSizes        = tensor->getAllNSizesInElements().data();
        m_tmpTensor.minSizes        = tensor->getNMinimalSizesInElements().data();
        m_tmpTensor.pData           = tensor->isStaticParam() ? tensor->getAddress() : nullptr;

        saveTensorPermutations(inputPermutations, m_tmpTensor, tensorIdx, isInput);

        createProtocolSection(&m_tmpSection, tensor);
        m_tmpTensor.tensorSection = &m_tmpSection;

        createProtocolAttributes(&m_tmpAttributes, tensor, !isInput);
        // Quantization params are relevant for inference or 8BitFloat only.
        if (m_isInference || is8BitFloat(tensor->getElementType()))
        {
            createProtocolQuantizationParams(&m_tmpProtocolQuantParams, tensor);
        }
        else
        {
            // irrelevant in this case, filling with default values to save overhead time (eager optimizations)
            m_tmpProtocolQuantParams = {{0}, 1};
        }
        m_tmpAttributes.quantizationParams = &m_tmpProtocolQuantParams;
        m_tmpTensor.attributes             = &m_tmpAttributes;

        // invoke handler to translate to target IR (for example - MLIR operands / results /tensor_details
        LOG_TRACE(GC_TRANSLATION,
                  "Finished converting tensor - {} to Protocol IR, invoking tensor handler",
                  tensor->getName());

        if (!handler.handleTensor(m_tmpTensor))
        {
            LOG_WARN(GC_TRANSLATION,
                     "Handler failed to handle tensor {}, stopping {} tensors iteration",
                     tensor->getName(),
                     isInput ? "input" : "output");
            return false;
        }
        tensorIdx++;
    }
    LOG_TRACE(GC_TRANSLATION,
              "Finished iterating {} {} tensors of node {}",
              tensorIdx,
              isInput ? "input" : "output",
              m_currentNode->getNodeName());
    return true;
}

void SynapseWrapperBase::saveTensorPermutations(const PermutationVector& inputPermutations,
                                                ProtocolTensor&          irTensor,
                                                unsigned                 tensorIdx,
                                                bool                     isInput) const
{
    irTensor.perm = s_identityPerm;
    // Save m_currentNode input permutations in the IRTensors.
    if (isInput && !inputPermutations.empty() && !(inputPermutations[tensorIdx].isEmpty()))
    {
        const auto& permVec = inputPermutations[tensorIdx].getValues();
        irTensor.perm        = permVec.data();
    }
}

void SynapseWrapperBase::createProtocolNodeFromGcNode(const NodePtr& synNode, ProtocolNode& irNode) const
{
    const auto& params               = synNode->getParamsRawData();
    irNode.userParams.nodeParamsSize = params.size();
    irNode.userParams.nodeParams     = (void*)(params.data());
    irNode.guid                      = synNode->getGUID();
    irNode.name                      = synNode->getNodeName();
    irNode.useDeterministic          = synNode->getDeterministic();
    irNode.isShapeManipulationOp     = false;
    // toGlueDataType asserts if precision is syn_type_na.
    const auto precision             = synNode->getNodePrecision();
    irNode.precision                 = (precision == syn_type_na) ? tpc_lib_api::NUM_DATATYPES : toTpcLibDataType(precision);
}

/*
 * SynapseGraphWrapper functions
 */
unsigned SynapseGraphWrapper::getNumNodes() const
{
    return m_graph.getNumNodes();
}

/*
 * Iterate synapse graph nodes in execution order.
 * Foreach node -
 *  1. Create ProtocolNode from it
 *  2. invoke ProtocolNodeHandler on the IR Node (for example to create MLIR Op from it)
 */
bool SynapseGraphWrapper::foreachNode(ProtocolNodeHandler& handler) const
{
    LOG_DEBUG(GC_TRANSLATION, "Iterating synapse graph nodes, number of nodes - {}", m_graph.getNumNodes());
    std::string newGuid;
    for (const NodePtr& synNode : m_graph.getExeSortedNodes())
    {
        m_currentNode = synNode;
        synNodeId nodeId = m_currentNode->getId();
        if (!handler.acceptNode(nodeId))
        {
            LOG_TRACE(GC_TRANSLATION,
                      "Handler didn't accept node {} with ID {}",
                      synNode->getNodeName(),
                      nodeId);
            continue;
        }
        LOG_TRACE(GC_TRANSLATION,
                  "Synapse Wrapper - Creating Protocol IR Node from current node - {} , with id - {}, with guid - {}",
                  synNode->getNodeName(),
                  nodeId,
                  synNode->getGUID());
        // create ir Node
        m_tmpNode = {};
        m_tmpNode.id = nodeId;
        createProtocolNodeFromGcNode(synNode, m_tmpNode);
        // TODO: make sure that fuser gets the value from this field when tpc_fuser pass starts using ProtocolGraph
        m_tmpNode.replacedNodeIds.data =
            static_cast<const uint64_t*>(&synNode->getNodeAnnotation().originalComplexGuidId);
        m_tmpNode.replacedNodeIds.size = 1;

        LOG_TRACE(GC_TRANSLATION, "Finished calling accept node, invoking node handler");
        if (!handler.handleNode(m_tmpNode))
        {
            LOG_WARN(GC_TRANSLATION,
                     "Handler failed to handle node {} with id {}, stopping nodes iteration",
                     synNode->getNodeName(),
                     synNode->getId());
            return false;
        }
        LOG_TRACE(GC_TRANSLATION, "Finished handling node {}", synNode->getNodeName());
    }
    LOG_TRACE(GC_TRANSLATION, "Finished iterating graph nodes, number of nodes - {}", m_graph.getNumNodes());
    return true;
}

unsigned SynapseGraphWrapper::getNumInputTensors(uint64_t nodeId) const
{
    const NodePtr node = m_graph.getNodeByID(nodeId);
    HB_ASSERT(node != nullptr, "Node with id {} wasn't found in graph", nodeId);
    return node->getNumInputs();
}

unsigned SynapseGraphWrapper::getNumOutputTensors(uint64_t nodeId) const
{
    const NodePtr node = m_graph.getNodeByID(nodeId);
    HB_ASSERT(node != nullptr, "Node with id {} wasn't found in graph", nodeId);
    return node->getNumOutputs();
}

/*
 * SynapseNodeWrapper functions
 */

bool SynapseNodeWrapper::foreachNode(ProtocolNodeHandler& handler) const
{
    m_tmpNode = {};
    synNodeId nodeId = m_currentNode->getId();
    LOG_TRACE(GC_TRANSLATION, "Invoking handler's acceptNode");
    if (!handler.acceptNode(nodeId))
    {
        LOG_TRACE(GC_TRANSLATION,
                  "Handler didn't accept node {} with ID {}",
                  m_currentNode->getNodeName(),
                  m_currentNode->getId());
        return true;
    }
    m_tmpNode.id = nodeId;
    createProtocolNodeFromGcNode(m_currentNode, m_tmpNode);
    LOG_TRACE(GC_TRANSLATION, "Finished calling acceptNode, invoking node handler");
    if (!handler.handleNode(m_tmpNode))
    {
        LOG_WARN(GC_TRANSLATION,
                 "Handler failed to handle node {} with id {}, stopping nodes iteration",
                 m_currentNode->getNodeName(),
                 m_currentNode->getId());
        return false;
    }
    LOG_TRACE(GC_TRANSLATION, "Finished handling node {}", m_currentNode->getNodeName());
    return true;
}

unsigned SynapseNodeWrapper::getNumInputTensors(uint64_t nodeId) const
{
    HB_ASSERT(m_currentNode != nullptr, "Synapse node to be wrapped is null");
    HB_ASSERT(m_currentNode->getId() == nodeId,
              "Given id {} isn't the same as the original node ID {}",
              nodeId,
              m_currentNode->getId());
    return m_currentNode->getNumInputs();
}

unsigned SynapseNodeWrapper::getNumOutputTensors(uint64_t nodeId) const
{
    HB_ASSERT(m_currentNode != nullptr, "Synapse node to be wrapped is null");
    HB_ASSERT(m_currentNode->getId() == nodeId,
              "Given id {} isn't the same as the original node ID {}",
              nodeId,
              m_currentNode->getId());
    return m_currentNode->getNumOutputs();
}
