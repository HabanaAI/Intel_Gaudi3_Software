#include "habana_graph.h"

#include "bundle_plane_graph.h"
#include "data_type_utils.h"
#include "descriptor_generator.h"
#include "graph_compiler/print_cycles.h"
#include "graph_compiler/tpc_json_serializer.h"
#include "graph_serializers/serializer.h"
#include "graph_traits.h"
#include "graph_visualization.h"
#include "habana_global_conf.h"
#include "habana_nodes.h"
#include "habana_pass.h"
#include "hal_reader/hal_reader.h"
#include "infra/defs.h"
#include "infra/log_manager.h"
#include "infra/timer.h"
#include "json_utils.h"
#include "node_factory.h"
#include "node_pipeline_depth.h"
#include "node_predicate_handler.h"
#include "node_roi.h"
#include "node_tensor_accessor.h"
#include "node_utils.h"
#include "op_validator.h"
#include "pass_manager.h"
#include "brain_data.h"
#include "register_memory_coherence.h"
#include "scheduler.h"
#include "section_handle.hpp"
#include "tensor_roi.h"
#include "tensor.h"
#include "tpc_json_serializer.h"
#include "tpc_node.h"
#include "training_pass_registrator.h"
#include "types_exception.h"
#include "utils.h"

#include <algorithm>
#include <bitset>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <stack>
#include <unistd.h>

using namespace std;

// Init loggers
synapse::LogManager& logInit = synapse::LogManager::instance();

static std::shared_ptr<char> g_simulatedSram;
static std::shared_ptr<char> g_simulatedDram;

static gc::ops::OpValidationContext nodePtrToOpValidationContext(const NodePtr& n);

HabanaGraph::HabanaGraph(bool PassManagerNeeded)
{
    if (PassManagerNeeded)
    {
        m_passManager = std::make_unique<PassManager>();
    }
}

HabanaGraph::HabanaGraph(const HabanaGraph& other, bool copyAddresses /*false*/, bool keepMappings /*false*/)
: m_nodeROIs(std::map<pNode, std::list<NodeROI>*>()),
  m_recipeName(other.m_recipeName),
  m_recipeDebugID(other.m_recipeDebugID),
  m_graphTraits(other.m_graphTraits),
  m_annotation(other.m_annotation),
  m_numOfDmaIntermediates(other.m_numOfDmaIntermediates),
  m_ctrlDepWasConfigured(other.m_ctrlDepWasConfigured),
  m_dynamicNodeCount(other.m_dynamicNodeCount),
  m_inputInferenceLayouts(other.m_inputInferenceLayouts),
  m_userNodeTypePrecision(other.m_userNodeTypePrecision),
  m_nodeTypeMinPrecision(other.m_nodeTypeMinPrecision),
  m_bundlePlane(nullptr),
  m_compiled(other.m_compiled),
  m_preDataTypeSelection(other.m_preDataTypeSelection),
  m_numSigOutTensors(other.m_numSigOutTensors),
  m_constSectionTensors(other.m_constSectionTensors),
  m_initialPersistentTensors(other.m_initialPersistentTensors),
  m_deviceLimitationInfo(other.m_deviceLimitationInfo),
  m_nodeCostModel(other.m_nodeCostModel)
{
    if (other.getCodeGenerator() != nullptr)
    {
        m_codeGenerator = other.getCodeGenerator()->clone(this, copyAddresses);
    }
    copyNodesAndTensors(other, copyAddresses, true, keepMappings);
    m_debugMode = other.m_debugMode;
    m_annotation.streamGroupToSize.clear(); // do not copy streaming groups
    m_annotation.errors = ErrorStatus(); // do not copy error status
    if (other.m_passManager)
    {
        m_passManager = std::make_unique<PassManager>();
    }
}

HabanaGraph::~HabanaGraph()
{
    clear();
}

HabanaGraph& HabanaGraph::operator=(const HabanaGraph& other)
{
    if (this != &other)
    {
        // We cannot use the copy-and-swap idiom because it would invalidate the tokens
        // inside the nodes and tensors. Therefore, we must operate on the current object.
        clear();
        copyNodesAndTensors(other);
        m_setupNodes = other.m_setupNodes;
        m_recipeName = other.m_recipeName;
        m_recipeDebugID = other.m_recipeDebugID;
        m_debugMode = other.m_debugMode;
        m_numOfDmaIntermediates = other.m_numOfDmaIntermediates;
        m_annotation = GraphAnnotation(other.m_annotation);
        m_annotation.streamGroupToSize.clear(); // do not copy streaming groups
        m_annotation.errors = ErrorStatus(); // do not copy error status
        m_graphTraits = other.m_graphTraits;
        m_ctrlDepWasConfigured = other.m_ctrlDepWasConfigured;
        m_dynamicNodeCount = other.m_dynamicNodeCount;
        m_inputInferenceLayouts = other.m_inputInferenceLayouts;
        m_userNodeTypePrecision = other.m_userNodeTypePrecision;
        m_nodeTypeMinPrecision  = other.m_nodeTypeMinPrecision;
        m_compiled              = other.m_compiled;
        m_numSigOutTensors      = other.m_numSigOutTensors;
        m_constSectionTensors   = other.m_constSectionTensors;
        m_initialPersistentTensors = other.m_initialPersistentTensors;
        m_nodeCostModel            = other.m_nodeCostModel;
        if (other.m_passManager)
        {
            m_passManager = std::make_unique<PassManager>();
        }
        // sub-class handles m_compilationAttrs and command queues
    }
    return *this;
}

pPass HabanaGraph::addPass(pPass newPass)
{
    m_passManager->registerPass(newPass);
    return newPass;
}

void HabanaGraph::registerPassGroups()
{
    if (m_graphTraits != nullptr)
    {
        TrainingPassRegistrator().registerGroups(*this);
    }
}

HabanaGraphPtr HabanaGraph::duplicate(TensorPtrMappingVec& tensorsMap, NodeIdMappingVec& nodesMap)
{
    // m_clonedTensors and m_clonedNodes are only needed to provide the Duplicate SynapseAPI
    // caller with the appropriate mappings from the original graph to the duplicated graph.
    // So we can clear them once we've filled the user provided output destination, to avoid
    // keeping the tensors and nodes alive longer then needed.
    HabanaGraphPtr duplicateGraph = clone(false, true);
    auto&          clonedNodes    = duplicateGraph->m_clonedNodes;
    for (const auto& nodeMapping : clonedNodes)
    {
        nodesMap.emplace_back(nodeMapping.first->getId(), nodeMapping.second->getId());
    }
    clonedNodes.clear();
    auto& clonedTensorsMap = duplicateGraph->m_clonedTensors;
    std::unordered_map<Tensor*, TensorPtr> clonedTensorsRawPtrMap;
    clonedTensorsRawPtrMap.reserve(clonedTensorsMap.size());
    std::for_each(clonedTensorsMap.begin(), clonedTensorsMap.end(), [&clonedTensorsRawPtrMap](const auto& mapping) {
        clonedTensorsRawPtrMap.emplace(mapping.first.get(), mapping.second);
    });
    for (auto& [origTensor, newTensor] : tensorsMap)
    {
        auto tensorMappingIter = clonedTensorsRawPtrMap.find(origTensor);
        if (tensorMappingIter != clonedTensorsRawPtrMap.end())
        {
            newTensor = tensorMappingIter->second;
        }
        else
        {
            newTensor = origTensor->clone(false, true, true, TensorNameClonePolicy::COPY_NAME);
        }
    }
    clonedTensorsMap.clear();
    duplicateGraph->m_duplicatedTarget = true;
    return duplicateGraph;
}

std::optional<uint32_t> HabanaGraph::getNextTPCKernelUniqueId()
{
    return std::nullopt;
}

void HabanaGraph::clear()
{
    for (auto it : m_nodeROIs)
    {
        it.second->clear();
        delete it.second;
    }
    m_nodeROIs.clear();
    m_cacheExeSortedNodes.clear();
    Graph::clear();

    if (m_codeGenerator)
    {
        m_codeGenerator->clear();
    }
    m_ctrlDepWasConfigured = false;

    m_inputInferenceLayouts.clear();
    m_userNodeTypePrecision.clear();
    m_nodeTypeMinPrecision.clear();

    m_constSectionTensors.clear();

    m_initialPersistentTensors.clear();

    //clear node factory resources
    NodeFactory::getInstance().clear();

}

template <typename T>
NodesMap HabanaGraph::cloneNodes(const T& nodes, const TensorMap& clonedTensors)
{
    NodesMap nodesMapping;
    for (pNode n : nodes)
    {
        pNode newNode = n->clone();
        clonedTensorsReplacer(n, newNode, clonedTensors); //replace tensors with clones
        addValidatedNode(newNode);                        // insert node to graph
        nodesMapping[n] = newNode;
    }
    return nodesMapping;
}

void HabanaGraph::copyNodeROIs(const std::list<NodeROI>& origNodeRois,
                               std::list<NodeROI>&       destNodeRois,
                               const TensorMap&          tensorsMapping)
{
    for (const NodeROI& roi : origNodeRois)
    {
        NodeROI& newRoi = destNodeRois.emplace_back(roi);
        for (TensorROIVector* roiVec : {&newRoi.inputRois, &newRoi.outputRois})
        {
            for (TensorROI& r : *roiVec)
            {
                if (!r.m_parentTensor) continue;

                auto clonedParent = tensorsMapping.find(r.m_parentTensor);
                HB_ASSERT(clonedParent != tensorsMapping.end(),
                          "Graph copy: can't find clone of tensor {} ",
                          r.m_parentTensor->getName());
                r.m_parentTensor = clonedParent->second;
            }
        }
    }
}

void HabanaGraph::copyNodesAndTensors(const HabanaGraph& other,
                                      bool               copyAddresses /*false*/,
                                      bool               keepPersistent /*false*/,
                                      bool               keepMappings /*false*/)
{
    TensorMap clonedTensors = cloneTensors(other, copyAddresses, keepPersistent, keepMappings);
    NodesMap clonedNodes;
    // We strive that every copy made by this function will have execution order
    if (!other.m_cacheExeSortedNodes.empty())
    {
        clonedNodes = cloneNodes(other.m_cacheExeSortedNodes.get(), clonedTensors);
    }
    else
    {
        clonedNodes = cloneNodes(other.getNodes(), clonedTensors);
    }

    // Clone setup nodes
    for (const auto& n : other.m_setupNodes)
    {
        auto newNode = n->clone();
        clonedTensorsReplacer(n, newNode, clonedTensors); // replace tensors with clones
        addSetupNode(newNode); // insert node to graph
        clonedNodes[n] = newNode;
    }
    for (const auto& n : clonedNodes)
    {
        copyNodeROIs(*(other.GetNodeROIs(n.first)), *(this->GetNodeROIs(n.second)), clonedTensors);
    }

    if (keepMappings)
    {
        m_clonedTensors = std::move(clonedTensors);
        m_clonedNodes   = std::move(clonedNodes);
    }
}

HabanaDeviceType HabanaGraph::getNodeDebugDeviceType(const pNode& node) const
{
    return node->isDma() ? DEVICE_DMA_DRAM_SRAM_BIDIRECTIONAL : m_nodeUtility.getNodeDeviceType(node);
}

bool HabanaGraph::compile()
{
    return false;
}

bool HabanaGraph::execute()
{
    return false;
}

const NodeVector& HabanaGraph::getExeSortedNodes() const
{
    if (!m_cacheExeSortedNodes.empty())
    {
        return m_cacheExeSortedNodes.get();
    }
    bool result = generateExecutionSchedule();
    HB_ASSERT(result, "Failed to generate execution schedule");
    return m_cacheExeSortedNodes.get();
}

bool HabanaGraph::generateExecutionSchedule(Scheduler* scheduler) const
{
    if (!m_cacheExeSortedNodes.empty())
    {
        return true;
    }

    // validating that the graph is acyclic also with control dependencies relations.
    if (!isAcyclicGraph())
    {
        if (GCFG_CYCLE_PRINTING_LEVEL.value() >= CyclePrintLevel::PRINT_IF_CYCLE_FOUND)
        {
            this->printGraphCycles();
        }
        LOG_ERR(GC, "Invalid graph - a cycle was found in the graph");

        throw SynapseException("Invalid Graph: a cycle was found");
    }

    NodeList executionSchedule = scheduler->scheduleNodes();
    m_cacheExeSortedNodes.reserve(executionSchedule.size());
    for (const NodePtr& n : executionSchedule)
    {
        m_cacheExeSortedNodes.push_back(n);
    }
    return true;
}

bool HabanaGraph::generateExecutionSchedule() const
{
    Scheduler scheduler(this);
    return generateExecutionSchedule(&scheduler);
}

// Check if producer is directly connected to consumer (i.e. they are neighbors)
// or seperated by logical operation nodes only
bool HabanaGraph::areNeighborsIgnoreLogicals(const pNode& producer,
                                             const pNode& consumer) const
{
    HB_ASSERT_PTR(producer);
    HB_ASSERT_PTR(consumer);

    if (producer == consumer)
    {
        return false;
    }

    for (const pTensor& t : consumer->getInputs())
    {
        if (t == nullptr) continue;

        const pNode& consumerParent = getTensorProducer(t);

        if (consumerParent != nullptr)
        {
            // Allow only logical operation nodes to separate producer from consumer
            if (consumerParent->isLogicalOperation())
            {
                return areNeighborsIgnoreLogicals(producer, consumerParent);
            }
            if (producer == consumerParent)
            {
                return true;
            }
        }
    }
    return false;
}

uint32_t HabanaGraph::getNumTensorConsumersIgnoreLogicals(const pTensor& tensor) const
{
    uint32_t ret = 0;

    for (const auto& consumer : getTensorConsumers(tensor))
    {
        if (consumer->isLogicalOperation()) continue;
        ++ret;
    }

    return ret;
}

void HabanaGraph::PrintNodesAndOperands() const
{
    if (!LOG_LEVEL_AT_LEAST_DEBUG(GRAPH_DATA)) return;

    LOG_DEBUG(GRAPH_DATA, "Nodes and Operands of recipe:{}", getRecipeName());
    unsigned numBPs = 0;
    const NodeVector& exeSortedNodes = getExeSortedNodes();
    LOG_DEBUG(GRAPH_DATA, "Nodes in the graph: {}", exeSortedNodes.size());

    for (const pNode& n : exeSortedNodes)
    {
        if (n == nullptr) continue;
        n->print();
        printNodeAdditionalInfo(n);

        LOG_DEBUG(GRAPH_DATA, "Breakpoint before node: {}", numBPs);
        const auto* rois = GetNodeROIs(n);
        if (rois && disableBreakpointsForNonSignaling())
        {
            for (auto it: *rois)
            {
                if (it.numSignals > 0)
                    numBPs++;
            }
        }
        else
        {
            numBPs += rois ? rois->size() : 0;
        }
    }
    LOG_DEBUG(GRAPH_DATA, "Breakpoint after graph: {}", numBPs);
}

void HabanaGraph::invalidateExecutionSchedule() const
{
    m_cacheExeSortedNodes.clear();
}

bool HabanaGraph::validateGCOp(const NodePtr& node) const
{
    if (!GCFG_ENABLE_GC_NODES_VALIDATION_BY_OPS_DB.value()) return true;
    const auto sts =
        gc::ops::OpValidator::validateOp(node->getGUID(), nodePtrToOpValidationContext(node), getDeviceType());
    switch (sts)
    {
        case gc::ops::ValidationResult::GUID_NOT_FOUND:
        case gc::ops::ValidationResult::SUCCESS:
        {
            return true;
        }
        default:
        {
            return false;
        }
    }
}

bool HabanaGraph::validateNode(const NodePtr& node) const
{
    // Set graph traits first, as validation depends on it
    // if Node belongs to another graph - can't add node
    if (!node->setGraphTraits(m_graphTraits)) return false;

    if (!validateGCOp(node))
    {
        LOG_ERR(GC,
                "HabanaGraph::{} failed gc op validation for node: {}, guid: {}",
                HLLOG_FUNC,
                node->getNodeName(),
                node->getGUID());
        return false;
    }

    if (!node->validateNodeForGraph(*this))
    {
        LOG_ERR(GC, "HabanaGraph::validateNode unsupported node {} for hardware", node->getNodeName());
        return false;
    }

    if (!node->validateRMWSection(GCFG_RMW_SECTION_MAX_SIZE_BYTES.value()))
    {
        LOG_ERR(GC, "HabanaGraph::validateNode error in node {} - rmw section validation failed", node->getNodeName());
        return false;
    }

    // make sure node's input/output data type is supported
    // Validate that a node's output is not a constant
    if (m_graphTraits != nullptr && m_graphTraits->getHalReader() != nullptr)
    {
        for (const pTensor& input : node->getInputs())
        {
            if (input == nullptr) continue;
            if (!isTensorDataTypeValid(input))
            {
                LOG_ERR(GC,
                        "HabanaGraph::validateNode unsupported input data type {}",
                        getStringFromSynDataType(input->getElementType()));
                return false;
            }
        }

        for (const pTensor& output : node->getOutputs())
        {
            if (output == nullptr) continue;
            /*
                We want to prevent nodes from having constant outputs [SW-35108], however, some exceptions needed to be
                added due to:
                [SW-41345] Const tensor is output of an identity node resulting DMA writing the const data
                [SW-41577] Logical ops transferring input to output and keeping const state
                [SW-19883] cast insertion should not copy the isStatic property of the input tensor
                [SW-66287] intermediate zero sized tensors should be marked as constant.

            */
            if (output->isZeroSizedDataTensor()) continue;
            if (output->isStaticParam() && node->getGUID() != "memcpy" && node->getGUID() != "memcpy_dma" &&
                node->getGUID() != "" && !(node->getGUID().find("cast") != std::string::npos))
            {
                LOG_ERR(GC, "HabanaGraph::validateNode node output can not be a constant: {}", output->getName());
                return false;
            }
            if (!isTensorDataTypeValid(output))
            {
                LOG_ERR(GC,
                        "HabanaGraph::validateNode unsupported output data type {}",
                        getStringFromSynDataType(output->getElementType()));
                return false;
            }
        }
    }
    return true;
}

bool HabanaGraph::addNode(pNode node)
{
    if (!validateNode(node)) return false;

    triggerNewNodeTensorPredicates(node);

    return addValidatedNode(node);
}

void HabanaGraph::triggerNewNodeTensorPredicates(const NodePtr& node)
{
    const auto& memoryCoherence                   = getGraphAnnotation().memoryCoherence;
    auto        checkUnregisteredCoherencyTensors = [&](const TensorPtr& t) {
        if (t && memoryCoherence->isMemoryCoherencyTensor(t) && !memoryCoherence->doesCoherencyTensorExist(t))
        {
            turnOnPredicate(PREDICATE_ID_MEMORY_SECTION_TENSOR_CREATED);
        }
    };

    if (memoryCoherence)
    {
        runOnTensorsForType<Node::USAGE_OUTPUT>(node, Node::TENSOR_TYPE_DATA, checkUnregisteredCoherencyTensors);
        runOnTensorsForType<Node::USAGE_INPUT>(node, Node::TENSOR_TYPE_DATA, checkUnregisteredCoherencyTensors);
    }

    auto checkNodeStaticInputEliminate = [&](const TensorPtr& t) {
        if (t && t->isStaticParam())
        {
            turnOnPredicate(PREDICATE_ID_NODE_CREATED_CONST_INPUT);
        }
    };

    runOnTensorsForType<Node::USAGE_INPUT>(node, Node::TENSOR_TYPE_DATA, checkNodeStaticInputEliminate);
}

bool HabanaGraph::addValidatedNode(pNode node)
{
    if (!preProcessAddedNode(node))
    {
        return false;
    }

    // insert the node to the graph
    if (!Graph::addNode(node))
    {
        return false;
    }

    // insert node to BP graph
    if (m_bundlePlane)
    {
        m_bundlePlane->addNode(node);
    }

    postProcessAddedNode(node);
    return true;
}

void HabanaGraph::removeNode(pNode node, pNode newProducer)
{
    Graph::removeNode(node, newProducer);
    if (m_bundlePlane)
    {
        m_bundlePlane->removeNode(node, newProducer);
    }
    postProcessRemovedNode(node);
}

bool HabanaGraph::moveNodesToGraph(HabanaGraph& outputGraph)
{
    auto nodeList = getNodes();
    for (auto& node : nodeList)
    {
        GraphEditor::removeNode(*this, node);
        if (!GraphEditor::addNode(outputGraph, node))
        {
            LOG_ERR(SYN_GRAPH, "{}: Failed to add node {}.", HLLOG_FUNC, node->getGUID());
            return false;
        }
    }
    return true;
}

void HabanaGraph::replaceSemanticNodes(NodePtr oldNode, NodePtr newNode)
{
    bool result = validateNode(newNode);
    HB_ASSERT(result, "validation for new node {} failed!", newNode->getNodeName());
    HB_ASSERT(oldNode->isDynamicShape() == newNode->isDynamicShape(), "can't replace nodes-different dynamic shape");
    result = preProcessAddedNode(newNode);
    HB_ASSERT(result, "adding new node {} failed!", newNode->getNodeName());

    Graph::replaceSemanticNodes(oldNode, newNode);
    if (m_bundlePlane)
    {
        m_bundlePlane->replaceSemanticNodes(oldNode, newNode);
    }
    postProcessRemovedNode(oldNode);
    postProcessAddedNode(newNode);
}

bool HabanaGraph::preProcessAddedNode(const NodePtr& node) const
{
    // Update internal node caches that may rely on operands data. The addition of the node to the graph may be due to
    // editing which change those operands.
    node->updateCache();

    // if Node belongs to another graph - can't add node
    return node->setGraphTraits(m_graphTraits);
}

void HabanaGraph::postProcessAddedNode(const NodePtr& node)
{
    NodePredicateHandler predHandler(*this);
    node->accept(&predHandler);

    // create node ROIs
    if (m_nodeROIs.find(node) == m_nodeROIs.end())
    {
        m_nodeROIs[node] = new std::list<NodeROI>();
    }

    // invalidate cache
    invalidateExecutionSchedule();

    if (node->isDynamicShape())
    {
        m_dynamicNodeCount++;
    }
}

void HabanaGraph::postProcessRemovedNode(const NodePtr& node)
{
    node->setGraphTraits(nullptr);
    invalidateExecutionSchedule();

    if (node->isDynamicShape())
    {
        m_dynamicNodeCount--;
    }
}

void HabanaGraph::attachNodes(pNode from, pNode to, unsigned outputIndex, unsigned inputIndex)
{
    Graph::attachNodes(from, to, outputIndex, inputIndex);
    invalidateExecutionSchedule();
}

NodeList HabanaGraph::getRootNodes() const
{
    NodeList roots = Graph::getRootNodes();
    // filter out nodes with barriers
    auto it = roots.begin();
    while (it != roots.end())
    {
        if ((*it)->getNodeAnnotation().memorySpaceInfo.barriers.empty())
        {
            ++it; // node has no barrier, leave it
        }
        else
        {
            it = roots.erase(it); // node has a barrier, remove it
        }
    }
    return roots;
}


std::list<pTensor> HabanaGraph::getGraphInputs() const
{
    std::list<pTensor> ret;

    for (pTensor t : getTensors())
    {
        if (isInputTensor(t))
        {
            ret.push_back(t);
        }
    }
    return ret;
}

std::list<pTensor> HabanaGraph::getGraphIntermediates() const
{
    std::list<pTensor> ret;
    //Return all tensors that have consumer and producer
    for (pTensor t : getTensors())
    {
        if (!isInputTensor(t) && !isOutputTensor(t))
        {
            ret.push_back(t);
        }
    }
    return ret;
}

bool HabanaGraph::nodeHasROIs(const NodePtr& n) const
{
    auto it = m_nodeROIs.find(n);
    if (it == m_nodeROIs.end())
    {
        return false;
    }
    return !it->second->empty();
}

std::list<NodeROI>* HabanaGraph::GetNodeROIs(const NodePtr& n) const
{
    auto it = m_nodeROIs.find(n);
    if (it == m_nodeROIs.end())
    {
        LOG_ERR(GC, "No ROIs defined for node");
        return nullptr;
    }
    return it->second;
}

std::pair<unsigned, unsigned> HabanaGraph::getBreakpointsAndNodeROINr(const NodePtr& n) const
{
    std::pair<unsigned, unsigned> breakpointAndROINr = {};
    const auto*                   rois               = GetNodeROIs(n);
    if (rois)
    {
        if (disableBreakpointsForNonSignaling())
        {
            for (const NodeROI& roi : *rois)
            {
                if (roi.numSignals > 0)
                {
                    breakpointAndROINr.first++;
                }
            }
        }
        else
        {
            breakpointAndROINr.first = rois->size();
        }
        breakpointAndROINr.second = rois->size();
    }
    return breakpointAndROINr;
}

NodeVector HabanaGraph::getSortedMMENodes()
{
    auto fn       = [](const NodePtr& node) {return runsOnMME(node);};
    auto mmeNodes = getTopoSortedNodesCond(fn);
    return mmeNodes;
}

void HabanaGraph::incNumOfIntermediatesDmaNodes()
{
    m_numOfDmaIntermediates++;
}

void HabanaGraph::addSetupNode(pNode node)
{
    HB_ASSERT_PTR(node);
    HB_ASSERT(m_setupNodes.count(node) == 0, "Node already in graph");
    m_setupNodes.insert(node);
    if (m_nodeROIs.find(node) == m_nodeROIs.end())
    {
        m_nodeROIs[node] = new std::list<NodeROI>();
    }
}

bool HabanaGraph::isOutputTensor(const pTensor& t) const
{
    if (t != nullptr)
    {
        // Output tensors are tensors that have no consumers or their sole consumer is DMA node writing to the host
        const NodeList& consumers           = getTensorConsumers(t);
        bool            isConsumedByHostDMA = false;
        for (const pNode& cons : consumers)
        {
            if ((cons->isDma()) && (!isActivationDMA(cons)))
            {
                isConsumedByHostDMA = true;
                break;
            }
        }

        if (Graph::isOutputTensor(t) || isConsumedByHostDMA) return true;
    }
    return false;
}

bool HabanaGraph::isInputTensor(const pTensor& t) const
{
    if (t != nullptr)
    {
        // Input tensors are tensors that have no producer or their producer is DMA node reading from the host
        NodePtr producer = getTensorProducer(t);
        return producer == nullptr || (producer->isDma() && !isActivationDMA(producer));
    }
    return false;
}

NodeSet& HabanaGraph::getSetupNodes()
{
    return m_setupNodes;
}

const string& HabanaGraph::getRecipeName() const
{
    return m_recipeName;
}
void HabanaGraph::setRecipeName(std::string_view recipeName)
{
    m_recipeName = recipeName;
}

uint16_t HabanaGraph::getRecipeDebugId() const
{
    return m_recipeDebugID;
}

void HabanaGraph::setRecipeDebugId(uint16_t recipeDebugID)
{
    m_recipeDebugID = recipeDebugID;
}

const GraphAnnotation& HabanaGraph::getGraphAnnotation() const
{
    return m_annotation;
}

GraphAnnotation& HabanaGraph::getGraphAnnotation()
{
    return m_annotation;
}

bool HabanaGraph::runPassManager()
{
    bool res = true;
    try
    {
        res = m_passManager->run(*this);
    }
    catch(const std::exception& e)
    {
        LOG_ERR(GC, "Run Pass Manager failed: {}", e.what());
        res = false;
    }
    if (!res)
    {
        GraphVisualization::graphVisualizationPostOnFailure(*this);
        PrintNodesAndOperands();
        return false;
    }
    return true;
}

bool HabanaGraph::runPartialPasses(PassId stopBefore)
{
    bool res = false;
    try
    {
        res = m_passManager->runPartial(*this, stopBefore);
    }
    catch (const std::exception& e)
    {
        LOG_ERR(GC, "Run partial passes up to {} failed", stopBefore, e.what());
        res = false;
    }
    return res;
}

void HabanaGraph::setPassManager(std::unique_ptr<PassManager>& pm)
{
    m_passManager.swap(pm);
}

std::unique_ptr<PassManager> HabanaGraph::clonePassManager() const
{
    return m_passManager->clone();
}

unsigned HabanaGraph::getNumTpcEng() const
{
    unsigned maxNumOfTPCs = 8;

    if (m_graphTraits != nullptr)
    {
        maxNumOfTPCs = m_graphTraits->getHalReader()->getNumTpcEngines();
    }

    return countSetBits(GCFG_TPC_ENGINES_ENABLED_MASK.value() & m_graphTraits->getHalReader()->getTpcEnginesMask(), maxNumOfTPCs);
}

uint64_t HabanaGraph::getAvailableEnginesMask(HabanaDeviceType deviceType) const
{
    switch (deviceType)
    {
        case DEVICE_TPC:
            return GCFG_TPC_ENGINES_ENABLED_MASK.value() & m_graphTraits->getHalReader()->getTpcEnginesMask();
        case DEVICE_DMA_DRAM_SRAM_BIDIRECTIONAL:
            return m_graphTraits->getHalReader()->getInternalDmaEnginesMask();
        default:
            return 0;
    }
}

void HabanaGraph::resetMultibufferInfo(const TensorPtr& tensor)
{
    if (!tensor->getTensorAnnotation().origBigTensor) return;
    if (!tensor->getTensorAnnotation().nonPersistentSectionInfo.sectionId.is_set()) return;
    const auto& origBigTensor = tensor->getTensorAnnotation().origBigTensor;
    unsigned newSectionId;
    if (m_tensorToSectionId.find(origBigTensor) == m_tensorToSectionId.end())
    {
        newSectionId                       = getNextMemorySectionID(SectionIDGenerator::GC_ALLOCATED_SECTIONS);
        m_tensorToSectionId[origBigTensor] = newSectionId;
    }
    else
    {
        newSectionId = m_tensorToSectionId.find(origBigTensor)->second;
    }
    tensor->getTensorAnnotation().nonPersistentSectionInfo.sectionId = newSectionId;
    LOG_DEBUG(GC, "Reset tensor {} nonPersistentSectionInfo, new sectionId = {}", tensor->getName(), newSectionId);
}

bool HabanaGraph::isEngineDisabled(HabanaDeviceType deviceType, unsigned engineId) const
{
    if (deviceType == DEVICE_TPC)
    {
        return ((((GCFG_TPC_ENGINES_ENABLED_MASK.value() & m_graphTraits->getHalReader()->getTpcEnginesMask()) >> engineId) & 0x1) == 0);
    }
    if (deviceType == DEVICE_DMA_DRAM_SRAM_BIDIRECTIONAL)
    {
        return (((m_graphTraits->getHalReader()->getInternalDmaEnginesMask() >> engineId) & 0x1) == 0);
    }
    return false;
}

bool HabanaGraph::getMemoryOrientedCompilationEnabled() const
{
    return false;
}

bool HabanaGraph::getVisualizationStatus() const
{
    return (GCFG_GRAPH_VISUALIZATION.value() ||
            GCFG_ENABLE_GVD.value() ||
            GCFG_ENABLE_PARTIAL_GVD.value() ||
            GCFG_SRAM_SLICER_GRAPH_VISUALIZATION.value() ||
            GCFG_GRAPH_VISUALIZATION_COLLAPSE_BUNDLES.value() ||
            !GCFG_GRAPH_VISUALIZATION_START_TENSOR.value().empty() ||
            !GCFG_GRAPH_VISUALIZATION_END_TENSOR.value().empty());
}

void HabanaGraph::printGlobalConfigurations() const
{
    GlobalConfManager::instance().printGlobalConf(false);
}

void HabanaGraph::saveUsedGcfgFile()
{
    if (GCFG_CREATE_USED_CONFIGS_FILE.value())
    {
        string filename = getRecipeName() + ".used";
        GlobalConfManager::instance().flush(filename);
    }
}

void HabanaGraph::replaceCompilationPass(pPass newPass)
{
    m_passManager->addReplacementPass(newPass);
}

const GraphTraits& HabanaGraph::getTraits() const
{
    HB_ASSERT(m_graphTraits != nullptr, "m_graphTraits class member is not initialized");
    return *m_graphTraits;
}

pNode HabanaGraph::getNodeSharedPtr(const Node& node) const
{
    return const_cast<Node&>(node).shared_from_this();
}

bool HabanaGraph::isPersistentTensor(const TensorPtr& tensor) const
{
    return tensor->getMemoryDescriptor().m_isPersistent;
}

bool HabanaGraph::isUserManagedDram(const TensorPtr& tensor) const
{
    return tensor->isUserManagedDram();
}

// Go over given tensors, and add TensorData to the given graphTensorsData
bool HabanaGraph::getTensorData(const std::vector<std::pair<pTensor, uint32_t>>&  tensorsInfo,
                                TensorUsage                                       tensorLocation,
                                Graph::GraphTensorsData&                          graphTensorsData,
                                bool&                                             shouldBreakToEnqueue)
{
    if (graphTensorsData.size() != 0)
    {
        LOG_ERR(GC, "{}: Given TensorData is already set. Aborting operation!", HLLOG_FUNC);
        return false;
    }

    for (const std::pair<pTensor, uint32_t>& tensorInfo : tensorsInfo)
    {
        pTensor currTensor = tensorInfo.first;

        pTensor tensor = currTensor;
        if (currTensor->isHostAliasedTensor())
        {
            tensor = currTensor->getHostAliasTensor();
        }

        bool isEnforced         = tensor->isEnforcedOutput();
        bool isNotMasked        = !tensor->isMaskedOutput();
        bool isActivation       = !tensor->isModelParameter() && !tensor->isUnitMatrix();
        bool isNotAliasedTensor = !tensor->isAliasedTensor();

        if (isEnforced ||
            (isActivation && isNotMasked && (isNotAliasedTensor || tensor->isDenseLayout())))
        {
            Graph::TensorData  tensorData;
            pNode              node;
            bool               isIgnore = false;

            if (tensorLocation == TensorUsage::INPUT_TENSOR)
            {
                const auto consumers = getTensorConsumers(tensor);
                for (const auto& consumer : consumers)
                {
                    if (consumer != nullptr && HabanaGraph::runsOnTPC(consumer))
                    {
                        isIgnore = true;
                    }
                }
            }
            else
            {
                const auto& producer = getTensorProducer(tensor);
                if ((producer != nullptr) &&
                    (HabanaGraph::runsOnTPC(producer) || producer->getNodeType() == Node::TYPE_INTERNAL_FLATTEN ||
                     producer->getNodeType() == Node::TYPE_INTERNAL_PACKING))
                {
                    isIgnore = true;
                }

            }

            tensorData.ignoreBatchVerification = isIgnore;
            tensorData.name                    = tensor->getName();
            tensorData.dimensions              = tensor->getDim();
            auto sizes                         = tensor->getNSizesInElements();
            std::copy(sizes.begin(), sizes.end(), tensorData.dimensionsSize);

            tensorData.roiSize       = currTensor->getTensorROISize();
            tensorData.sliceOffset   = 0;
            tensorData.sliceSize     = currTensor->getTotalSizeInBytes();
            tensorData.firstDmaIndex = tensorInfo.second;

            if (currTensor->isHostAliasedTensor())
            {
                tensorData.sliceOffset = currTensor->getHostAliasOffset();
            }

            if (tensorLocation == TensorUsage::INPUT_TENSOR || tensorLocation == TensorUsage::OUTPUT_TENSOR)
            {
                unsigned batchPos = tensor->getBatchPos();
                if (batchPos == INVALID_BATCH_POS)
                {
                    // Print this only once
                    if (shouldBreakToEnqueue)
                    {
                        LOG_INFO(GC, "No batch position is given for Input/Output tensor. disabling break to enqueue.");
                        shouldBreakToEnqueue = false;
                    }
                    tensorData.batchSize = 0; // this is marked as unset
                    tensorData.sampleSize = 0; // sample sizes will be recalculated later
                }
                else if (batchPos != (tensorData.dimensions - 1)) // check if batch position is last
                {
                    // Print this only once
                    if (shouldBreakToEnqueue)
                    {
                        LOG_WARN(GC, "Batch position is not last. disabling break to enqueue.");
                        shouldBreakToEnqueue = false;
                    }
                    tensorData.batchSize = 0; // this is marked as unset
                    tensorData.sampleSize = 0; // sample sizes for I/O will be recalculated later
                }
                else
                {
                    tensorData.batchSize = tensor->getSizeInElements(batchPos);
                    uint32_t ret = 1;
                    for (uint32_t i = 0; i < tensorData.dimensions; ++i)
                    {
                        if (i != batchPos)
                        {
                            ret *= tensor->getSizeInElements(i);
                        }
                    }
                    tensorData.sampleSize = ret;
                }
            }
            else // Intermediate
            {
                tensorData.batchSize = 0;
                tensorData.sampleSize = multiplyElements(tensorData.dimensionsSize,
                                                         tensorData.dimensionsSize + tensorData.dimensions);
            }

            tensorData.elementType = tensor->getElementType();

            LOG_TRACE(GC, "Tensor Data (basic): name {}, sampleSize {}, batchSize {}, elementType {}, is-ignore {}",
                      tensorData.name, tensorData.sampleSize, tensorData.batchSize, tensorData.elementType,
                      tensorData.ignoreBatchVerification);

            tensorData.zp        = tensor->getZeroPoint();
            tensorData.scale     = tensor->getScale();
            tensorData.totalSize = multiplyElements(tensorData.dimensionsSize,
                                                    tensorData.dimensionsSize + tensorData.dimensions);

            graphTensorsData.push_back(tensorData);
        }
    }

    return true;
}

void HabanaGraph::SortedNodes::push_back(const NodePtr& n)
{
    // Each node maintains an index matching the final execution order.
    n->setExecutionOrderedIndex(size());
    NodeVector::push_back(n);
}

bool HabanaGraph::recalculateSampleSize(Graph::GraphTensorsData& inputTensorsData, Graph::GraphTensorsData& outputTensorsData)
{
    for (auto& tensorData: inputTensorsData)
    {
        tensorData.batchSize = 0;
        tensorData.sampleSize = tensorData.totalSize;
    }
    for (auto& tensorData: outputTensorsData)
    {
        tensorData.batchSize = 0;
        tensorData.sampleSize = tensorData.totalSize;
    }
    return true;
}

unsigned HabanaGraph::getRotateStripeWidth(std::shared_ptr<RotateNode>& rotateNode) const
{
    return getHALReader()->getRotateStripeWidth();
}

const std::shared_ptr<HalReader>& HabanaGraph::getHALReader() const
{
    static const std::shared_ptr<HalReader> NULL_HAL_READER;
    return m_graphTraits != nullptr ? m_graphTraits->getHalReader() : NULL_HAL_READER;
}

unsigned HabanaGraph::getDefaultPipelineDepth() const
{
    return GCFG_DEFAULT_PIPELINE_DEPTH.value();
}

unsigned HabanaGraph::getPipelineDepth(const pNode& node) const
{
    return getPipelineDepth(*node);
}

unsigned HabanaGraph::getPipelineDepth(const Node& node) const
{
    uint32_t          depth = getDefaultPipelineDepth();
    NodePipelineDepth visitor(*this);
    for (auto input : node.getInputs())
    {
        const pNode& producer = getTensorProducer(input);
        if (producer)
        {
            producer->accept(&visitor);
            depth = max(depth, visitor.m_pipelineDepth);
        }
    }

    depth = std::min(GCFG_MAX_DYNAMIC_PIPELINE_DEPTH.value(), (uint64_t)depth);

    LOG_INFO(GC, "Chosen pipeline depth for node {}: {}", node.getNodeName(), depth);

    return depth;
}

void HabanaGraph::setDefaultPipelineDepth(unsigned depth)
{
    GCFG_DEFAULT_PIPELINE_DEPTH.setValue(depth);
}

bool HabanaGraph::rerunPass(PassId id)
{
    return m_passManager->reRunPass(id);
}

bool HabanaGraph::turnOnPredicate(PredicateId id)
{
    return m_passManager ? m_passManager->turnOnPredicate(id) : false;
}



bool HabanaGraph::validateGraphTensorsAreAllocated() const
{
    bool valid = true;
    for (pTensor t : getTensors())
    {
        if (t == nullptr) continue;
        if (t->isControlEdge()) continue;
        if (t->isShapeTensor()) continue;
        if (t->getTensorAllocatedLocation() == UNDEFINED_LOCATION)
        {
            if (getNumberOfTensorConsumers(t) != 0 || getTensorProducer(t) != nullptr)
            {
                LOG_ERR(GC, "{}: Tensor {} was not allocated.", HLLOG_FUNC, t->getName());
                t->debugPrint();
                valid = false;
            }
        }
    }

    return valid;
}

bool HabanaGraph::validateGraphTensorsAreReset() const
{
    for (pTensor t : getTensors())
    {
        if (t == nullptr) continue;
        if (t->getTensorAllocatedLocation() != UNDEFINED_LOCATION)
        {
            LOG_ERR(GC,
                    "{}: Tensor {} is {} while no tensor should be allocated at this point.",
                    HLLOG_FUNC,
                    t->getName(),
                    t->getTensorLocationString());
            return false;
        }
    }

    return true;
}

bool HabanaGraph::validateMemorySection(const InternalSectionHandle* section) const
{
    HB_ASSERT(section != nullptr, "Unexpected empty section handle");

    // Default validation - backwards compatible sections: persistent non-RMW (or nullptr)
    // Graphs that allow other configuration should override this validation.

    CHECK_RET_FALSE(section->getPersistent(), "Only persistent sections are supported");
    CHECK_RET_FALSE(!section->getRMW(), "Only non-RMW sections are supported");

    return true;
}

bool HabanaGraph::doesNodeHaveDmaUpConsumer(pNode node) const
{
    const TensorVector& outputs = node->getOutputs();
    for (auto tensor : outputs)
    {
        std::shared_ptr<Tensor> outputTensor = tensor;
        const std::list<pNode> consumers = getTensorConsumers(outputTensor);
        for (auto consumer : consumers)
        {
            if (consumer != nullptr)
            {
                if (!consumer->isLogicalOperation())
                {
                    if (m_nodeUtility.getNodeDeviceType(consumer) == DEVICE_DMA_DEVICE_HOST)
                    {
                        return true;
                    }
                }
                else if (doesNodeHaveDmaUpConsumer(consumer))
                {
                    return true;
                }
            }
        }
    }

    return false;
}

bool HabanaGraph::pinningBufferSizeIsSet() const
{
    return (GCFG_PINNING_BUFFER_SIZE.value() > 0);
}

uint32_t HabanaGraph::getPinningBufferSize() const
{
    HB_ASSERT(pinningBufferSizeIsSet(), "Pinning buffer size is not set");
    return GCFG_PINNING_BUFFER_SIZE.value();
}

bool HabanaGraph::tensorsPinningDisabled() const
{
    return GCFG_DISABLE_TENSORS_PINNING.value();
}

bool HabanaGraph::prefetchingBufferSizeIsSet() const
{
    return (GCFG_PREFETCH_BUFFER_SIZE.value() > 0);
}

uint32_t HabanaGraph::getPrefetchingBufferSize() const
{
    HB_ASSERT(prefetchingBufferSizeIsSet(), "Prefetching buffer size is not set");
    return GCFG_PREFETCH_BUFFER_SIZE.value();
}

bool HabanaGraph::allocateAllInDramEnabled() const
{
    return GCFG_ALLOCATE_ALL_IN_DRAM.value();
}

NodeSet HabanaGraph::getBlockedNodes(const NodePtr& blockingNode) const
{
    NodeSet blockedNodes;

    for (pTensor t : blockingNode->getControlOutputs())
    {
        HB_ASSERT_PTR(t);
        auto consumers = getTensorConsumers(t);
        blockedNodes.insert(consumers.begin(), consumers.end());
    }
    return blockedNodes;
}

NodeSet HabanaGraph::getBlockingNodes(const NodePtr& blockedNode) const
{
    NodeSet blockingNodes;
    for (pTensor t : blockedNode->getControlInputs())
    {
        HB_ASSERT_PTR(t);
        blockingNodes.insert(getTensorProducer(t));
    }
    return blockingNodes;
}

NodeSet HabanaGraph::getBlockingNodes(const NodePtr& blockedNode, Tensor::ControlEdgeType controlType) const
{
    NodeSet blockingNodes;
    for (pTensor t : blockedNode->getControlInputs())
    {
        HB_ASSERT_PTR(t);
        if (t->getControlEdgeType() != controlType) continue;
        blockingNodes.insert(getTensorProducer(t));
    }
    return blockingNodes;
}

void HabanaGraph::addControlDependency(const NodeSet&          blockingSet,
                                       const NodeSet&          blockedSet,
                                       Tensor::ControlEdgeType controlType)
{
    HB_ASSERT(blockedSet.size() != 0, "blocked set size is 0");
    HB_ASSERT(blockingSet.size() != 0, "blocking set size is 0");

    m_ctrlDepWasConfigured = true;

    // need to connect an edge between each node in the blocking set to each node in the blocked set
    for (const NodePtr& blockingNode : blockingSet)
    {
        for (const NodePtr& blockedNode : blockedSet)
        {
            addControlDependency(blockingNode, blockedNode, controlType);
        }
    }
}

void HabanaGraph::addControlDependency(const NodePtr&          blockingNode,
                                       const NodePtr&          blockedNode,
                                       Tensor::ControlEdgeType controlType)
{
    m_ctrlDepWasConfigured = true;
    TensorPtr ctrlIn       = nullptr;

    for (const TensorPtr& t : blockingNode->getControlOutputs())
    {
        if (t->getControlEdgeType() == controlType)
        {
            ctrlIn = t;
        }
    }
    if (ctrlIn == nullptr)
    {
        ctrlIn = std::make_shared<Tensor>();
        ctrlIn->setName(blockingNode->getNodeName() + "_control_edge");
        ctrlIn->setAsControlEdge(controlType);
        blockingNode->addOutput(ctrlIn, Node::TENSOR_TYPE_CONTROL);
        addRelationship(ctrlIn, blockingNode, Node::USAGE_OUTPUT);
    }

    HB_ASSERT_PTR(ctrlIn);

    const TensorVector& blockTensors = blockedNode->getControlInputs();

    if (std::find(blockTensors.begin(), blockTensors.end(), ctrlIn) == blockTensors.end())
    {
        HB_ASSERT(blockingNode->getId() != blockedNode->getId(), "node is blocking itself");
        blockedNode->addInput(blockedNode->getNumInputs(Node::TENSOR_TYPE_CONTROL), ctrlIn, Node::TENSOR_TYPE_CONTROL);
        addRelationship(ctrlIn, blockedNode, Node::USAGE_INPUT);
        LOG_DEBUG(GC, "Adding control edge from: {}, to: {}", blockingNode->getNodeName(), blockedNode->getNodeName());
    }

    if (m_bundlePlane)
    {
        m_bundlePlane->addRelationshipInBP(ctrlIn, blockingNode, blockedNode);
    }
    invalidateExecutionSchedule();
}

void HabanaGraph::removeNodeControlDependencies(const NodePtr& node, Tensor::ControlEdgeType controlType)
{
    removeNodeInputControlDependencies(node, controlType);
    removeNodeOutputControlDependencies(node, controlType);
}

void HabanaGraph::removeNodeInputControlDependencies(const NodePtr& node, Tensor::ControlEdgeType controlType)
{
    TensorVector controlInputs = node->getControlInputs();
    for (const TensorPtr& t : controlInputs)
    {
        HB_ASSERT_PTR(t);
        if (t->getControlEdgeType() != controlType) continue;
        removeNodeControlDependency(node, t, Node::USAGE_INPUT);
    }
}

void HabanaGraph::removeNodeOutputControlDependencies(const NodePtr& node, Tensor::ControlEdgeType controlType)
{
    TensorVector controlOutputs = node->getControlOutputs();
    for (const TensorPtr& t : controlOutputs)
    {
        HB_ASSERT_PTR(t);
        if (t->getControlEdgeType() != controlType) continue;
        removeNodeControlDependency(node, t, Node::USAGE_OUTPUT);
    }
}

bool HabanaGraph::isControlDependencyBetweenNodes(const NodePtr& blocking, const NodePtr& blocked) const
{
    const TensorVector& blockingControlOutputs = blocking->getControlOutputs();
    const TensorVector& blockedControlInputs   = blocked->getControlInputs();

    return std::any_of(blockingControlOutputs.begin(),
                       blockingControlOutputs.end(),
                       [&blockedControlInputs](const auto& blockingControlOutput) {
                           return std::find(blockedControlInputs.begin(),
                                            blockedControlInputs.end(),
                                            blockingControlOutput) != blockedControlInputs.end();
                       });
}

void HabanaGraph::removeNodeControlDependency(const NodePtr& node, const TensorPtr& t, Node::eParamUsage usage)
{
    if (usage == Node::USAGE_INPUT)
    {
        LOG_DEBUG(GC, "removing input control edge {} from node: {}", t->getName(), node->getNodeName());
        node->removeInput(t, Node::TENSOR_TYPE_CONTROL);
        removeRelationship(t, node, Node::USAGE_INPUT);

        if (getNumberOfTensorConsumers(t) == 0)
        {
            NodePtr producer = getTensorProducer(t);
            HB_ASSERT_PTR(producer);
            producer->removeOutput(t, Node::TENSOR_TYPE_CONTROL);
            removeRelationship(t, producer, Node::USAGE_OUTPUT);
            if (m_bundlePlane)
            {
                m_bundlePlane->removeRelationshipInBP(t, producer, Node::USAGE_OUTPUT);
            }
        }
    }
    else
    {
        LOG_DEBUG(GC, "removing output control edge {} from node: {}", t->getName(), node->getNodeName());
        node->removeOutput(t, Node::TENSOR_TYPE_CONTROL);
        removeRelationship(t, node, Node::USAGE_OUTPUT);

        for (const NodePtr& consumer : getTensorConsumers(t))
        {
            consumer->removeInput(t, Node::TENSOR_TYPE_CONTROL);
            removeRelationship(t, consumer, Node::USAGE_INPUT);
            if (m_bundlePlane)
            {
                m_bundlePlane->removeRelationshipInBP(t, consumer, Node::USAGE_INPUT);
            }
        }
    }
    if (m_bundlePlane)
    {
        m_bundlePlane->removeRelationshipInBP(t, node, usage);
    }
    invalidateExecutionSchedule();
}

bool HabanaGraph::isControlDependencyConfigured()
{
    return m_ctrlDepWasConfigured;
}

void HabanaGraph::storeExecutionSchedule()
{
    HB_ASSERT(m_storedCacheExeSortedNodes.size() == 0, "nested storing is not allowed");
    m_storedCacheExeSortedNodes = m_cacheExeSortedNodes;
    Graph::storeTopologicalSort();
}

void HabanaGraph::restoreExecutionSchedule()
{
    m_cacheExeSortedNodes = m_storedCacheExeSortedNodes;
    m_storedCacheExeSortedNodes.clear();
    Graph::restoreTopologicalSort();
}

void HabanaGraph::clearStoredExecutionSchedule()
{
    m_storedCacheExeSortedNodes.clear();
}

static void printNodeCacheMetadata(const std::vector<CacheMetaData>& metadataEntries)
{
    for (size_t i = 0; i < metadataEntries.size(); i++)
    {
        const auto& md = metadataEntries[i];
        LOG_DEBUG(GRAPH_DATA,
                  "    metadata[{}]: directive: {}, class: {}, CME: {}, MCID: {}",
                  i,
                  md.print_directive(),
                  md.print_class(),
                  md.print_action(),
                  md.mcid);
    }
}

static void printNodeRoiCacheUsage(const NodeROI& roi)
{
    LOG_DEBUG(GRAPH_DATA, "  Inputs cache usage:");
    printNodeCacheMetadata(roi.inputsCacheMetaData);
    LOG_DEBUG(GRAPH_DATA, "  Outputs cache usage:");
    printNodeCacheMetadata(roi.outputsCacheMetaData);
}

void HabanaGraph::printNodeAdditionalInfo(const pNode& n) const
{
    if (!LOG_LEVEL_AT_LEAST_DEBUG(GRAPH_DATA)) return;
    if (GetNodeROIs(n) == nullptr) return;
    const auto nodeRois = GetNodeROIs(n);
    if (nodeRois != nullptr)
    {
        if (!nodeRois->empty() && getHALReader() && getHALReader()->isCacheSupported())
        {
            // nodeRois is populated means we're in post-graph printing of a physical node.
            printNodeRoiCacheUsage(nodeRois->front());
        }
        uint64_t readBytes  = n->getReadBytes(TENSOR_IN_DRAM, *nodeRois, getHALReader()->getCacheLineSizeInBytes());
        uint64_t writeBytes = n->getWriteBytes(TENSOR_IN_DRAM, *nodeRois, getHALReader()->getCacheLineSizeInBytes());
        LOG_DEBUG(GRAPH_DATA, "HBM/DRAM read: {} bytes ({} MB), HBM/DRAM write: {} bytes ({} MB)", readBytes, (float)readBytes / (1024 * 1024), writeBytes, (float)writeBytes / (1024 * 1024));
    }
    if (HabanaGraph::runsOnMME(n))
    {
        MMENodePtr mmeNode = std::dynamic_pointer_cast<MmeNode>(n);
        HB_ASSERT(mmeNode, "could not downcast Node to MME Node");
        const MmeExpBias& mmeExpBias = mmeNode->getMmeExpBias();
        LOG_DEBUG(GRAPH_DATA,
                  "MME Node Exp bias: fp8BiasIn: {}, fp8BiasIn2: {}, fp8BiasOut: {}",
                  mmeExpBias.fp8BiasIn[TENSOR_IFM],
                  mmeExpBias.fp8BiasIn.size() > TENSOR_WEIGHT ? std::to_string(mmeExpBias.fp8BiasIn[TENSOR_WEIGHT]) : "NA",
                  mmeExpBias.fp8BiasOut);
    }
}

tpc_lib_api::DeviceId HabanaGraph::deviceTypeToDeviceID(synDeviceType deviceType)
{
    return ::deviceTypeToDeviceID(deviceType);  // call function from utils
}

void HabanaGraph::setInputInferenceLayouts(std::map<TensorPtr, gc::Layout> inputInferenceLayouts)
{
    HB_ASSERT(inputInferenceLayouts.size() <= getGraphInputs().size(),
              "Inference input Layouts should be less or equal to number of graph "
              "inputs, since input layouts does not include model params");
    m_inputInferenceLayouts = inputInferenceLayouts;
}

void HabanaGraph::setUserNodeTypePrecision(const std::string& guid, synDataType precision)
{
    std::string_view guidWithoutDType = extractGUIDFromFullGUID(guid);
    std::string lower_guid(guidWithoutDType);
    std::transform(lower_guid.begin(), lower_guid.end(), lower_guid.begin(), ::tolower);

    LOG_DEBUG(GC, "Set user node type {} precision to {}", lower_guid, getStringFromSynDataType(precision));

    m_userNodeTypePrecision[lower_guid] = precision;
}

bool HabanaGraph::getUserNodeTypePrecision(const std::string& guid, synDataType& precision) const
{
    std::string_view guidWithoutDType = extractGUIDFromFullGUID(guid);
    std::string lower_guid(guidWithoutDType);
    std::transform(lower_guid.begin(), lower_guid.end(), lower_guid.begin(), ::tolower);

    auto itr = m_userNodeTypePrecision.find(lower_guid);

    if (itr != m_userNodeTypePrecision.end())
    {
        precision = itr->second;
        return true;
    }

    // return false if node type precision was not set
    return false;
}

synDataType HabanaGraph::getNodeTypeMinPrecision(const NodePtr& node)
{
    // For nodes of type add, if they have a scalar input, a float32 minPrecision will be returned.
    if (extractGUIDFromFullGUID(node->getGUID()) == NodeFactory::addNodeTypeName && hasScalarInput(node))
    {
        LOG_TRACE(DATA_TYPES, "Node {} has a scalar input, picking f32 as minPrecision", node->getNodeName());
        return syn_type_single;
    }
    return getNodeTypeMinPrecision(node->getGUID());
}

synDataType HabanaGraph::getNodeTypeMinPrecision(const std::string& guid)
{
    if (m_nodeTypeMinPrecision.empty())
    {
        LOG_DEBUG(GC, "Initialize node types min precision table");
        initNodeTypeMinPrecision();
    }

    std::string_view guidWithoutDType = extractGUIDFromFullGUID(guid);
    std::string lower_guid(guidWithoutDType);
    std::transform(lower_guid.begin(), lower_guid.end(), lower_guid.begin(), ::tolower);

    auto itr = m_nodeTypeMinPrecision.find(lower_guid);

    if (itr != m_nodeTypeMinPrecision.end())
    {
        return itr->second;
    }

    // Default min precision per node type is bf16
    return syn_type_bf16;
}

void HabanaGraph::setLayeredBrainData(std::unique_ptr<gc::layered_brain::LayeredBrainData>&& lbd)
{
    m_layeredBrainData = std::move(lbd);
}

gc::layered_brain::LayeredBrainData* HabanaGraph::getLayeredBrainData() const
{
    return m_layeredBrainData.get();
}

void HabanaGraph::constructBPGraph(bool useAnnotations, std::function<bool(const NodePtr&)> predicate)
{
    m_bundlePlane = unique_ptr<BundlePlane>(new BundlePlane(*this, useAnnotations, predicate));
}

void HabanaGraph::discardBPGraph()
{
    m_bundlePlane = nullptr;
}

BundlePlane* HabanaGraph::getBPGraph() const
{
    return m_bundlePlane.get();
}

bool HabanaGraph::isSupported64BitDataType(synDataType elementType) const
{
    static constexpr auto mask64BitDatatype = (syn_type_uint64 | syn_type_int64);
    bool                  is64BitDataType   = (elementType & mask64BitDatatype) != 0;
    return is64BitDataType && graphSupports64BitDataTypes();
}

bool HabanaGraph::isTensorDataTypeValid(const TensorPtr& tensor) const
{
    synDataType type = tensor->getElementType();
    return (m_graphTraits->getHalReader()->isSupportedDataType(type) || isSupported64BitDataType(type)) ||
           (type == syn_type_na && m_preDataTypeSelection) || tensor->isControlEdge();
}

void HabanaGraph::setInferenceMode(bool mode)
{
    if (!mode)
    {
        LOG_DEBUG(GC, "Set GCFG_SYNAPSE_DATA_TYPE_SELECTION=false, data type selction can be enabled only in "
                      "inference mode");
        GCFG_SYNAPSE_DATA_TYPE_SELECTION.setValue(false);
    }

    return m_graphTraits->setTrainingGraph(!mode);
}

void HabanaGraph::setBackoffFactor(double boFactor)
{
    return m_graphTraits->setBackoffFactor(boFactor);
}

void HabanaGraph::getSignalOutInfo(unsigned& numSigOutTensors, unsigned& numSigOutEngineTypes)
{
    std::set<HabanaDeviceType> engineTypes;
    unsigned                   count = 0;

    for (auto& tensor : getTensors())
    {
        if (tensor->getTensorIsExternal())
        {
            count++;
            for (auto& node : getRealProducers(tensor))
            {
                engineTypes.insert(m_nodeUtility.getNodeDeviceType(node));
            }
        }
    }

    numSigOutEngineTypes = engineTypes.size();
    numSigOutTensors     = count;
    m_numSigOutTensors   = numSigOutTensors;
}

void HabanaGraph::dumpTpcNodesDataToJson(uint32_t idx) const
{
    // Avoiding appending to existing JSONs since these can run over 1GB in filesize
    // and end up blowing up runtime and memory usage.
    if (!GCFG_DUMP_TPC_NODES_DATA_TO_JSON.value()) return;

    auto json = TpcJsonSerializer::serialize(*this);
    auto fname = GCFG_TPC_NODES_JSON_FILE.value() + "." + std::to_string(idx) + ".json";
    json_utils::jsonToFile(json, fname);
}

void HabanaGraph::dumpTpcNodesDataToJson()
{
    if (!GCFG_DUMP_TPC_NODES_DATA_TO_JSON.value()) return;

    auto json = TpcJsonSerializer::serialize(*this);
    json_utils::jsonToFile(json, GCFG_TPC_NODES_JSON_FILE.value());
}

static gc::ops::OpValidationContext nodePtrToOpValidationContext(const NodePtr& n)
{
    using namespace gc::ops;
    OpValidationContext ovc {};

    if (!n) return ovc;

    const auto& inputs = n->getInputs();
    ovc.getInputs().reserve(inputs.size());
    for (const auto& in : inputs)
    {
        ovc.getInputs().push_back(
            in == nullptr ? TensorValidationContext()
                          : TensorValidationContext(in->getDim(), in->getElementType(), in->getTensorType()));
    }

    const auto& outputs = n->getOutputs();
    ovc.getOutputs().reserve(outputs.size());
    for (const auto& out : outputs)
    {
        ovc.getOutputs().push_back(
            out == nullptr ? TensorValidationContext()
                           : TensorValidationContext(out->getDim(), out->getElementType(), out->getTensorType()));
    }
    return ovc;
}

static std::string getFilePath(const fs::path& recipeName, const fs::path& path, const std::string& postFix)
{
    const fs::path nonAbsRecipeName = recipeName.is_absolute() ? recipeName.string().substr(1) : recipeName.string();

    const fs::path recipeDir   = nonAbsRecipeName.parent_path();
    const fs::path fileName    = sanitizeFileName(nonAbsRecipeName.filename()) + postFix;
    const fs::path fullPath    = recipeDir.empty() ? path / fileName : path / recipeDir / fileName;
    const fs::path fullPathDir = fullPath.parent_path();

    if (!fs::is_directory(fullPathDir) || !fs::exists(fullPathDir))
    {
        fs::create_directories(fullPathDir);
    }

    return fullPath;
}

void HabanaGraph::dumpGraphToJson(graph_serializer::GraphState state, const std::string& name) const
{
    auto dumpPath = graph_serializer::getGraphStatePath(state);
    if (dumpPath.empty()) return;

    if (state == graph_serializer::GraphState::POST_PASS && !GCFG_DUMP_PASSES_FILTER.value().empty() &&
        GCFG_DUMP_PASSES_FILTER.value() != name)
        return;

    bool              isDir     = fs::is_directory(dumpPath);
    const std::string stateName = name.empty() ? graph_serializer::toString(state) : name;
    for (const auto& s : {graph_serializer::Serializers::PRE_GRAPH, graph_serializer::Serializers::POST_GRAPH})
    {
        auto serializer = graph_serializer::create(s);
        if (!serializer->supports(state)) continue;
        const std::string extention = serializer->name().empty()
                                          ? fmt::format(".{}.json", stateName)
                                          : fmt::format(".{}.{}.json", stateName, serializer->name());
        const std::string fileNameWithPid =
            isDir ? getFilePath(getRecipeName(), dumpPath, fmt::format(".{}{}", getpid(), extention))
                  : std::string(dumpPath);
        LOG_INFO(SYNREC, "dumping {} graph to file: {}", name, fileNameWithPid);
        serializer->serialize(*this, fileNameWithPid, getRecipeName(), !isDir);

        // temp w/a to avoid breaking apps that are dependent on previous file name (SW-167046)
        if (isDir)
        {
            const std::string fileName = getFilePath(getRecipeName(), dumpPath, extention);
            if (!fs::exists(fileName))
            {
                LOG_INFO(SYNREC, "dumping {} graph to file: {}", name, fileName);
                serializer->serialize(*this, fileName, getRecipeName(), !isDir);
            }
        }
    }
}

void HabanaGraph::setTensorsAlignment()
{
    unsigned cacheLineSize = getHALReader()->getCacheLineSizeInBytes();
    for (auto& tensor : getTensors())
    {
        if (tensor->isAliasedTensor()) continue;
        tensor->setTensorAlignment(cacheLineSize);
    }
}

void HabanaGraph::collectConstSectionAndPersistentTensors()
{
    for (const TensorPtr& t : getTensors())
    {
        if (t)
        {
            if (t->inConstSection())
            {
                m_constSectionTensors.push_back(t);
            }
            if (t->isPersistent())
            {
                m_initialPersistentTensors.insert(t);
            }
        }
    }
}

const std::map<unsigned, uint32_t>& HabanaGraph::getLogicalQueueToMaxExecutionIndex()
{
    if (m_logicalQueueToMaxExecutionIndex.empty())
    {
        setLogicalQueueToMaxExecutionIndex();
    }

    return m_logicalQueueToMaxExecutionIndex;
}

unsigned HabanaGraph::getNumNodesPreCompilation()
{
    return getNumNodes();
}

void HabanaGraph::setFP32LimitedDevice()
{
    m_deviceLimitationInfo.fp32Limited = true;
}

std::optional<std::pair<NodeCostModel::EngineType, double>>
HabanaGraph::getNodeExpectedDuration(const NodePtr& node) const
{
    if (m_nodeCostModel == nullptr) return std::nullopt;
    return m_nodeCostModel->getNodeExpectedDuration(node);
}