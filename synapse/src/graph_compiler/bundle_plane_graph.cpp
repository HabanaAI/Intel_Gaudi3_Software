#include "bundle_plane_graph.h"
#include "gaudi_scheduler.h"
#include "infra/defs.h"
#include "habana_graph.h"
#include "types.h"
#include <habana_nodes/conv_base_node.h>
#include <graph_editor.h>
#include "node_tensor_accessor.h"

#define BP_TRACE(...) LOG_TRACE(BP_GRAPH, __VA_ARGS__)
#define BP_DEBUG(...) LOG_DEBUG(BP_GRAPH, __VA_ARGS__)
#define BP_ERROR(...) LOG_ERR(BP_GRAPH, __VA_ARGS__)

NodeList BundlePlane::getOrigScheduleForBpNode(const std::shared_ptr<BundlePlaneNode>& bpNode)
{
    NodeList bundleOrigNodes(bpNode->getBundledNodes().begin(), bpNode->getBundledNodes().end());
    bundleOrigNodes.sort([](const NodePtr& n1, const NodePtr n2) {
        return n1->getExecutionOrderedIndex() < n2->getExecutionOrderedIndex();
    });
    return bundleOrigNodes;
}

NodeList BundlePlane::createOrigNodesScheduleFromBpgSchedule(const NodeList& bpgSchedule)
{
    BP_TRACE("{}: translate bundle plane nodes schedule to graph nodes schedule", HLLOG_FUNC);
    // Create the schedule of the orig nodes as follows:
    // replace every bundleNode in bpgSchedule with its sorted nodes
    NodeList schedule;
    for (const auto& n : bpgSchedule)
    {
        const auto bundlePlaneNode = std::dynamic_pointer_cast<BundlePlaneNode>(n);
        HB_ASSERT(bundlePlaneNode != nullptr,
                  "Expecting that all the nodes of the bpg schedule are of type BundlePlaneNode");
        if (!bundlePlaneNode->isBundle())
        {
            HB_ASSERT(bundlePlaneNode->getBundledNodes().size() == 1,
                      "Expecting that this bundlePlaneNode {} has only one orig node",
                      n->getNodeName());
            schedule.push_back(bundlePlaneNode->getBundledNodes()[0]);
        }
        else
        {
            schedule.splice(schedule.end(), getOrigScheduleForBpNode(bundlePlaneNode));
        }
    }
    return schedule;
}

// return true if both nodes are not in any bundle
bool BundlePlane::areSameBundle(const NodePtr& n1, const NodePtr& n2)
{
    return areSameBundle(n1->getNodeAnnotation().bundleInfo, n2->getNodeAnnotation().bundleInfo);
}

bool BundlePlane::areSameBundle(const Settable<BundleInfo>& info1, const Settable<BundleInfo>& info2)
{
    return (info1.is_set() == info2.is_set()) && (!info1.is_set() || (info2->bundleIndex == info1->bundleIndex));
}

bool BundlePlane::isPartOfBundle(const NodePtr& n)
{
    return n->getNodeAnnotation().bundleInfo.is_set();
}

class BundlePlane::BundlePlaneAccumulator
{
public:
    BundlePlaneAccumulator(const HabanaGraph& origGraph) : m_origGraph(origGraph) {}

    void setBundle(const Settable<BundleInfo>& nextInfo) { m_currentBundleInfo = nextInfo; }

    bool isSameBundle(const Settable<BundleInfo>& info) const { return areSameBundle(m_currentBundleInfo, info); }

    void resetAccumulator()
    {
        m_bundleNodes.clear();
        m_bundleInputs.clear();
        m_bundleOutputs.clear();
    }

    void accumulateNode(BundlePlane& bp, const NodePtr& node)
    {
        m_bundleNodes.push_back(node);
        accumulateInputs(bp, node);
        accumulateOutputs(bp, node);
    }

    void addBundleToBPGraph(BundlePlane& bp)
    {
        if (m_currentBundleInfo.is_set())
        {
            // create new bundle node, and add to bundle-plane
            NodePtr newBPNode = NodePtr(new BundlePlaneNode({m_bundleInputs.begin(), m_bundleInputs.end()},
                                                            {m_bundleOutputs.begin(), m_bundleOutputs.end()},
                                                            m_bundleNodes,
                                                            m_currentBundleInfo.value()));
            bp.m_bpGraph->addNode(newBPNode);
            for (const NodePtr& n : m_bundleNodes)
            {
                bp.m_OGNodeToBPNode[n->getId()] = newBPNode;
            }
            bp.m_bundleIdxToBundleNode[m_currentBundleInfo->bundleIndex] = newBPNode;
        }
    }

private:
    void accumulateInputs(BundlePlane& bp, const NodePtr& node)
    {
        runOnTensorsForType<Node::USAGE_INPUT>(node, Node::TENSOR_TYPE_ALL, [&](const TensorPtr& in) {
            if (in == nullptr) return;
            const NodePtr& producer = m_origGraph.getTensorProducer(in);
            if (producer == nullptr) return;
            if (!areSameBundle(node, producer))  // otherwise, intermediate tensor within the bundle
            {
                m_bundleInputs.insert(bp.insertBPTensor(in));  // set as bundle input
            }
        });
    }

    void accumulateOutputs(BundlePlane& bp, const NodePtr& node)
    {
        runOnTensorsForType<Node::USAGE_OUTPUT>(node, Node::TENSOR_TYPE_ALL, [&](const TensorPtr& out) {
            if (out == nullptr) return;
            const auto& consumers   = m_origGraph.getTensorConsumers(out);
            bool        outOfBundle = std::any_of(consumers.begin(), consumers.end(), [&](const NodePtr& n) {
                return !areSameBundle(n, node);
            });
            if (outOfBundle)  // there exists a consumer that is not part of the bundle
            {
                m_bundleOutputs.insert(bp.insertBPTensor(out));
            }
        });
    }

    const HabanaGraph&   m_origGraph;
    NodeVector           m_bundleNodes;
    TensorSet            m_bundleInputs;
    TensorSet            m_bundleOutputs;
    Settable<BundleInfo> m_currentBundleInfo;
};

bool BundlePlane::validateBPGraph(const HabanaGraph& origGraph) const
{
    if (!getBundlePlaneGraph()->isAcyclicGraph())
    {
        BP_ERROR("circularity found in newly created BP graph");
        return false;
    }

    std::unordered_set<unsigned> bundleIDs;
    for (const NodePtr& n : origGraph.getNodes())
    {
        if (!n || (m_predicate && !m_predicate(n))) continue;
        const auto& info = n->getNodeAnnotation().bundleInfo;
        if (info.is_set()) bundleIDs.insert(info->bundleIndex);
    }

    unsigned numBundles = 0;
    for (const NodePtr& n : getBundlePlaneGraph()->getNodes())
    {
        if (downcastToBundlePlaneNode(n)->isBundle()) numBundles++;
    }
    if (numBundles != bundleIDs.size())
    {
        BP_ERROR("no match in number of bundles at creation of bundle plane");
        return false;
    }
    return true;
}

void BundlePlane::buildFromExistingAnnotations(HabanaGraph& origGraph)
{
    const NodeVector& sortedNodes = origGraph.getExeSortedNodes();  // assuming bundle nodes are scheduled in order
    BundlePlane::BundlePlaneAccumulator accumulator(origGraph);
    for (const NodePtr& node : sortedNodes)
    {
        if (m_predicate && !m_predicate(node)) continue;
        const auto& bundleInfo = node->getNodeAnnotation().bundleInfo;
        if (isPartOfBundle(node))
        {
            if (!accumulator.isSameBundle(bundleInfo))  // finish creating previous bundle
            {
                accumulator.addBundleToBPGraph(*this);
                accumulator.resetAccumulator();
                accumulator.setBundle(bundleInfo);
            }

            // start accumulating nodes for next bundle
            accumulator.accumulateNode(*this, node);
        }
        else
        {
            addBPNodeFromOGNode(node);  // no a bundle node, use existing method
        }
    }

    // finish up last bundle
    accumulator.addBundleToBPGraph(*this);
    HB_ASSERT(validateBPGraph(origGraph), "bp graph validation failed");
}

BundlePlane::BundlePlane() : m_bpGraph(new Graph()) {}

BundlePlane::BundlePlane(HabanaGraph& origGraph, bool useAnnotations, std::function<bool(const NodePtr&)> predicate) : m_bpGraph(new Graph()), m_predicate(predicate)
{
    if (useAnnotations)
    {
        buildFromExistingAnnotations(origGraph);
    }
    else
    {
        buildInitialBPGraph(origGraph);
    }
}

void BundlePlane::createBundleFromNodes(const NodeVector& ogNodes, const BundleInfo& bundleInfo)
{
    HB_ASSERT(!ogNodes.empty(), "did not get any nodes to add to bundle");
    HB_ASSERT(m_bundleIdxToBundleNode.find(bundleInfo.bundleIndex) == m_bundleIdxToBundleNode.end(),
              "bundle {} already exists in bp graph!",
              bundleInfo.bundleIndex);

    NodeVector ogBPNodes;
    for (const NodePtr& n : ogNodes)
    {
        const NodePtr& bpNode = getBundlePlaneRepresentation(n);
        HB_ASSERT(!downcastToBundlePlaneNode(bpNode)->isBundle(),
                  "node {} is already part of {}!",
                  n->getNodeName(),
                  bpNode->getNodeName());
        ogBPNodes.push_back(bpNode);
    }

    fuseBundles(ogBPNodes, bundleInfo);
}

bool BundlePlane::addNodeToBundle(const NodePtr& ogNode, const BundleInfo& bundleInfo)
{
    auto iter = m_bundleIdxToBundleNode.find(bundleInfo.bundleIndex);
    if (iter == m_bundleIdxToBundleNode.end())
    {
        return addNewBundle(ogNode, bundleInfo);
    }
    else
    {
        const NodePtr& bpCandidate = getBundlePlaneRepresentation(ogNode);
        HB_ASSERT(downcastToBundlePlaneNode(iter->second)->getBundleIndex() == bundleInfo.bundleIndex,
                  "bundle info mismatch between node {} and {}",
                  ogNode->getNodeName(),
                  iter->second->getNodeName());
        if (downcastToBundlePlaneNode(bpCandidate)->isBundle())
        {
            BP_ERROR("Fusing 2 bundles together is not supported");
            return false;
        }
        fuseBundles({bpCandidate, iter->second}, bundleInfo);
        return true;
    }
}

bool BundlePlane::addNewBundle(const NodePtr& ogNode, const BundleInfo& bundleInfo)
{
    auto  nodeRepresentation = getBundlePlaneRepresentation(ogNode);
    auto* bpNode             = downcastToBundlePlaneNode(nodeRepresentation);
    HB_ASSERT_PTR(bpNode);

    if (bpNode->isBundle())
    {
        return false;
    }
    bpNode->setBundle(bundleInfo);
    m_bundleIdxToBundleNode[bundleInfo.bundleIndex] = nodeRepresentation;
    BP_DEBUG("");
    BP_DEBUG("Set {} as {}", ogNode->getNodeName(), bpNode->getNodeName());
    return true;
}

void BundlePlane::fuseBundles(const NodeVector& ogBPNodes, const BundleInfo& bundleInfo)
{
    NodeVector ogNodes;
    for (const NodePtr& bpNode : ogBPNodes)
    {
        const auto& bundleNodes = downcastToBundlePlaneNode(bpNode)->getBundledNodes();
        ogNodes.insert(ogNodes.end(), bundleNodes.begin(), bundleNodes.end());
    }

    TensorSet externalInputs;
    TensorSet externalOutputs;
    std::tie(externalInputs, externalOutputs) = getExternalTensors(ogBPNodes);

    NodePtr newBPNode = NodePtr(new BundlePlaneNode({externalInputs.begin(), externalInputs.end()},
                                                    {externalOutputs.begin(), externalOutputs.end()},
                                                    ogNodes,
                                                    bundleInfo));
    for (const NodePtr& n : ogNodes)
    {
        m_OGNodeToBPNode[n->getId()] = newBPNode;
    }
    m_bundleIdxToBundleNode[bundleInfo.bundleIndex] = newBPNode;

    for (const NodePtr& n : ogBPNodes)
    {
        m_bpGraph->removeNode(n);
    }
    bool result = m_bpGraph->addNode(newBPNode);

    HB_ASSERT(result, "not able to add bp node {}", newBPNode->getNodeName());
    HB_DEBUG_VALIDATE(m_bpGraph->isAcyclicGraph());
    BP_DEBUG("");
    BP_DEBUG("Fused {} nodes into {}", ogNodes.size(), newBPNode->getNodeName());
    tracePrint();
}

// get all external inputs and outputs of a set of bp nodes.
// i.e., if these nodes were fused together, what would be there tensors comming out/in of this fused node
std::pair<TensorSet, TensorSet> BundlePlane::getExternalTensors(const NodeVector& bpNodes) const
{
    TensorSet externalInputs;
    TensorSet externalOutputs;
    for (const NodePtr& n : bpNodes)
    {
        for (const TensorPtr& t : n->getInputs())
        {
            if (!t) continue;
            // mark tensor as internal if its producer is one of the bpNodes. otherwise, mark it as external.
            const NodePtr& prod = m_bpGraph->getTensorProducer(t);
            if (!prod || (std::find(bpNodes.begin(), bpNodes.end(), prod) == bpNodes.end()))
            {
                externalInputs.insert(t);
            }
        }
        for (const TensorPtr& t : n->getOutputs())
        {
            if (!t) continue;
            // mark tensor as internal if all its consumers are part of the bpNodes. otherwise, mark it as external.
            const auto& consumers = m_bpGraph->getTensorConsumers(t);
            if (consumers.empty() || std::any_of(consumers.begin(), consumers.end(), [&](const NodePtr& cons) {
                    return std::find(bpNodes.begin(), bpNodes.end(), cons) == bpNodes.end();
                }))
            {
                externalOutputs.insert(t);
            }
        }
    }
    return std::make_pair(externalInputs, externalOutputs);
}

uint64_t BundlePlane::getNumberOfPaths(const NodePtr& ogSource, const NodePtr& ogTarget) const
{
    const auto& bpSource = getBundlePlaneRepresentation(ogSource);
    const auto& bpTarget = getBundlePlaneRepresentation(ogTarget);
    return m_bpGraph->getNumberOfPaths(bpSource, bpTarget);
}

bool BundlePlane::validateCandidate(const NodePtr&    candidate,
                                    const TensorPtr&  stitchedTensor,
                                    const NodeVector& acceptedNodes) const
{
    // W.I.P!
    // Ignore this method for now.

    auto bpCandidate = getBundlePlaneRepresentation(candidate);
    for (const auto& acceptedNode : acceptedNodes)
    {
        auto bpAccepted = getBundlePlaneRepresentation(acceptedNode);
        if (m_bpGraph->getNumberOfPaths(bpCandidate, bpAccepted) > 1)
        {
            return false;
        }
        if (m_bpGraph->getNumberOfPaths(bpAccepted, bpCandidate) > 0)
        {
            return false;
        }
    }
    return true;
}
void BundlePlane::unbundleNode(const NodePtr& ogNode)
{
    NodePtr bpNode         = getBundlePlaneRepresentation(ogNode);
    auto*   bpNodeDowncast = downcastToBundlePlaneNode(bpNode);
    if (!bpNodeDowncast->isBundle()) return;
    HB_ASSERT(bpNodeDowncast->isBundle(), "Expecting node {} belongs to a bundle", ogNode->getNodeName());

    // if this is a bundle of 1 node, no need to actually remove from the graph.
    if (bpNodeDowncast->getBundledNodes().size() == 1)
    {
        m_bundleIdxToBundleNode.erase(bpNodeDowncast->getBundleIndex());
        bpNodeDowncast->unsetBundle();
    }
    else
    {
        // bundle consists of more than one node
        // remove bundle and re-create w/o ogNode
        const auto ogNodeBundleInfo = ogNode->getNodeAnnotation().bundleInfo;
        NodeVector bundleNodesCopy  = bpNodeDowncast->getBundledNodes();
        bundleNodesCopy.erase(std::remove(bundleNodesCopy.begin(), bundleNodesCopy.end(), ogNode));
        removeBundle(ogNode);  // dismantles bundle then adds each og node to the bp graph as a standalone bp node

        // Reconstructing the bundle node by node exposes the graph to be dependent on the order
        // nodes were bundled, hence reconstruction is done in a single step in which fuseBundles is called once.
        createBundleFromNodes(bundleNodesCopy, ogNodeBundleInfo.value());
    }
}

void BundlePlane::removeBundle(const NodePtr& ogNode)
{
    NodePtr bpNode         = getBundlePlaneRepresentation(ogNode);
    auto*   bpNodeDowncast = downcastToBundlePlaneNode(bpNode);
    HB_ASSERT(bpNodeDowncast->isBundle(), "node {} is not part of a bundle!", ogNode->getNodeName());

    m_bundleIdxToBundleNode.erase(bpNodeDowncast->getBundleIndex());

    // if this is a bundle of 1 node, no need to actually remove from the graph.
    if (bpNodeDowncast->getBundledNodes().size() == 1)
    {
        bpNodeDowncast->unsetBundle();
        return;
    }

    m_bpGraph->removeNode(bpNode);
    for (const NodePtr& n : bpNodeDowncast->getBundledNodes())
    {
        m_OGNodeToBPNode.erase(n->getId());
    }

    BP_DEBUG("Removing {}", bpNode->getNodeName());
    for (const NodePtr& n : bpNodeDowncast->getBundledNodes())
    {
        addBPNodeFromOGNode(n);
        BP_DEBUG("Added {} instead of removed bundle", n->getNodeName());
    }
}

void BundlePlane::buildInitialBPGraph(const HabanaGraph& graph)
{
    for (const auto& ogNode : graph.getNodes())
    {
        addBPNodeFromOGNode(ogNode);
    }
    BP_DEBUG("Bundle Plane Graph built ({} nodes)", m_bpGraph->getNumNodes());
    tracePrint(graph);
}

void BundlePlane::addBPNodeFromOGNode(const NodePtr& ogNode)
{
    HB_ASSERT(m_OGNodeToBPNode.find(ogNode->getId()) == m_OGNodeToBPNode.end(),
              "Original graph node: {} re-inserted to bundle-plane graph",
              ogNode->getNodeName());

    TensorVector inputs;
    TensorVector outputs;
    runOnTensorsForType<Node::USAGE_INPUT>(ogNode, Node::TENSOR_TYPE_ALL, [&](const TensorPtr& input) {
        if (!input) return;
        const TensorPtr& t = insertBPTensor(input);
        if (std::find(inputs.begin(), inputs.end(), t) != inputs.end()) return;
        inputs.push_back(t);
    });
    runOnTensorsForType<Node::USAGE_OUTPUT>(ogNode, Node::TENSOR_TYPE_ALL, [&](const TensorPtr& output) {
        if (!output) return;
        outputs.push_back(insertBPTensor(output));
    });
    NodePtr bpNode                    = NodePtr(new BundlePlaneNode(inputs, outputs, ogNode));
    m_OGNodeToBPNode[ogNode->getId()] = bpNode;
    m_bpGraph->addNode(bpNode);
}

const bool BundlePlane::hasBundlePlaneRepresentation(const NodePtr& ogNode) const
{
    return m_OGNodeToBPNode.find(ogNode->getId()) != m_OGNodeToBPNode.end();
}

const NodePtr& BundlePlane::getBundlePlaneRepresentation(const NodePtr& ogNode) const
{
    HB_ASSERT(m_OGNodeToBPNode.find(ogNode->getId()) != m_OGNodeToBPNode.end(),
              "Original graph node: {} not found in the bundle-plane",
              ogNode->getNodeName());

    const NodePtr& bpNode = m_OGNodeToBPNode.at(ogNode->getId());
    HB_ASSERT_PTR(bpNode.get());

    return bpNode;
}

const TensorPtr& BundlePlane::getBundlePlaneRepresentation(const TensorPtr& ogTensor) const
{
    HB_ASSERT(m_OGTensorToBPTensor.find(ogTensor->getId()) != m_OGTensorToBPTensor.end(),
              "Original graph tensor: {} not found in the bundle plane",
              ogTensor->getName());

    const TensorPtr& bpTensor = m_OGTensorToBPTensor.at(ogTensor->getId());
    HB_ASSERT_PTR(bpTensor.get());

    return bpTensor;
}

TensorPtr BundlePlane::insertBPTensor(const TensorPtr& ogTensor)
{
    auto iter = m_OGTensorToBPTensor.find(ogTensor->getId());
    if (iter == m_OGTensorToBPTensor.end())
    {
        auto iterAndIsInserted = m_OGTensorToBPTensor.insert(
            std::make_pair(ogTensor->getId(),
                           ogTensor->clone(true /*copyAddress*/, false /*copyData*/, true /*keepPersistent*/)));

        HB_ASSERT(iterAndIsInserted.second,
                  "Something went wrong - couldn't insert a tensor despite not finding the key");

        iter = iterAndIsInserted.first;
    }
    return iter->second;
}

void BundlePlane::tracePrint() const
{
    if (!LOG_LEVEL_AT_LEAST_TRACE(BP_GRAPH)) return;

    for (const auto& node : m_bpGraph->getNodes())
    {
        HB_ASSERT_PTR(node);
        BP_TRACE("");
        BP_TRACE("Node: {}", node->getNodeName());
        int idx = 0;
        for (const auto& input : node->getInputs())
        {
            if (!input) continue;
            BP_TRACE("IN[{}]:  {}", idx++, input->getName());
        }
        idx = 0;
        for (const auto& output : node->getOutputs())
        {
            if (!output) continue;
            BP_TRACE("OUT[{}]: {}", idx++, output->getName());
        }
    }

    BP_TRACE("");
}

void BundlePlane::tracePrint(const HabanaGraph& origGraph) const
{
    if (!LOG_LEVEL_AT_LEAST_TRACE(BP_GRAPH)) return;
    tracePrint();
    BP_TRACE("Nodes map:");
    for (const auto& idAndNode : m_OGNodeToBPNode)
    {
        const auto& ogNode = origGraph.getNodeByID(idAndNode.first);
        HB_ASSERT_PTR(ogNode);
        BP_TRACE("{} -> {}", ogNode->getNodeName(), idAndNode.second->getNodeName());
    }

    BP_TRACE("");
    BP_TRACE("Tensors map:");
    const auto& allOGTensors = origGraph.getTensors();
    for (const auto& idAndTensor : m_OGTensorToBPTensor)
    {
        auto iter = std::find_if(allOGTensors.begin(), allOGTensors.end(), [&](const TensorPtr& tensor) {
            return (tensor && tensor->getId() == idAndTensor.first);
        });
        BP_TRACE("{} -> {}", iter != allOGTensors.end() ? (*iter)->getName() : "NULL", idAndTensor.second->getName());
    }
}

BundlePlaneNode* BundlePlane::downcastToBundlePlaneNode(const NodePtr& bundlePlaneRepresentation)
{
    auto* pNode = dynamic_cast<BundlePlaneNode*>(bundlePlaneRepresentation.get());
    HB_ASSERT_PTR(pNode);
    return pNode;
}

void BundlePlane::addNode(const NodePtr& node)
{
    if (m_freeze) return;
    HB_ASSERT(m_OGNodeToBPNode.find(node->getId()) == m_OGNodeToBPNode.end(),
              "node {} already exists in BP!",
              node->getNodeName());
    addBPNodeFromOGNode(node);
    if (node->getNodeAnnotation().bundleInfo.is_set())
    {
        addNodeToBundle(node, node->getNodeAnnotation().bundleInfo.value());
    }
}

void BundlePlane::removeNode(const NodePtr& node, NodePtr newProducer)
{
    if (m_freeze) return;
    if (newProducer)
    {
        newProducer = getBundlePlaneRepresentation(newProducer);
    }

    auto it = m_OGNodeToBPNode.find(node->getId());
    HB_ASSERT(it != m_OGNodeToBPNode.end(), "node {} doesn't exist in BP!", node->getNodeName());
    auto* bpNode = downcastToBundlePlaneNode(it->second);

    if (bpNode->isBundle() && bpNode->getBundledNodes().size() != 1)  // remove node from existing bundle
    {
        m_bpGraph->removeNode(it->second, newProducer);
        bpNode->removeNodeFromBundle(node, m_OGTensorToBPTensor);
        m_bpGraph->addNode(it->second);
    }
    else  // bundle node consisting of a single original node
    {
        if (bpNode->isBundle())
        {
            m_bundleIdxToBundleNode.erase(bpNode->getBundleIndex());
        }

        m_bpGraph->removeNode(it->second, newProducer);
    }
    m_OGNodeToBPNode.erase(it);
}

void BundlePlane::replaceSemanticNodes(const NodePtr& oldNode, const NodePtr& newNode)
{
    if (m_freeze) return;
    auto it = m_OGNodeToBPNode.find(oldNode->getId());
    HB_ASSERT(it != m_OGNodeToBPNode.end(), "node {} doesn't exist in BP!", oldNode->getNodeName());
    HB_ASSERT(m_OGNodeToBPNode.find(newNode->getId()) == m_OGNodeToBPNode.end(),
              "node {} already exists in BP!",
              newNode->getNodeName());

    downcastToBundlePlaneNode(it->second)->replaceNodeInBundle(oldNode, newNode);
    m_OGNodeToBPNode[newNode->getId()] = it->second;
    m_OGNodeToBPNode.erase(it);
}

void BundlePlane::freezeGraph()
{
    HB_ASSERT(m_freeze == false, "BP graph already frozen");
    m_freeze = true;
}

void BundlePlane::unfreezeGraph()
{
    HB_ASSERT(m_freeze == true, "BP graph already not frozen");
    m_freeze = false;
}

void BundlePlane::addRelationshipInBP(const TensorPtr& origTensor,
                                      const NodePtr&   origProducer,
                                      const NodePtr&   origConsumer)
{
    if (m_freeze) return;

    NodePtr bpBlockingNode = getBundlePlaneRepresentation(origProducer);
    NodePtr bpBlockedNode  = getBundlePlaneRepresentation(origConsumer);

    // If the 2 Nodes are located in the same bundle, they will be represented as a single node in BP.
    // Thus we can't add control relationship between them in BP (Tensor control would be input and output of same node)
    if (bpBlockedNode == bpBlockingNode) return;

    const TensorPtr& bpTensor = insertBPTensor(origTensor);

    const auto& outputs = bpBlockingNode->getOutputs();
    const auto& inputs  = bpBlockedNode->getInputs();

    if (std::find(outputs.begin(), outputs.end(), bpTensor) == outputs.end())
    {
        bpBlockingNode->addOutput(bpTensor);
        m_bpGraph->addRelationship(bpTensor, bpBlockingNode, Node::USAGE_OUTPUT);
    }
    if (std::find(inputs.begin(), inputs.end(), bpTensor) == inputs.end())
    {
        bpBlockedNode->addInput(bpBlockedNode->getNumInputs(), bpTensor);
        m_bpGraph->addRelationship(bpTensor, bpBlockedNode, Node::USAGE_INPUT);
    }
}

void BundlePlane::removeRelationshipInBP(const TensorPtr& origTensor, const NodePtr& origNode, Node::eParamUsage usage)
{
    if (m_freeze) return;
    const TensorPtr& bpTensor = insertBPTensor(origTensor);
    NodePtr          bpNode   = getBundlePlaneRepresentation(origNode);
    const auto&      tensors  = usage == Node::USAGE_INPUT ? bpNode->getInputs() : bpNode->getOutputs();
    if (std::find(tensors.begin(), tensors.end(), bpTensor) == tensors.end())
    {
        return;  // inner-bundle tensor
    }

    if (usage == Node::USAGE_INPUT)
    {
        bpNode->removeInput(bpTensor);
    }
    else
    {
        bpNode->removeOutput(bpTensor);
    }
    m_bpGraph->removeRelationship(bpTensor, bpNode, usage);
}
