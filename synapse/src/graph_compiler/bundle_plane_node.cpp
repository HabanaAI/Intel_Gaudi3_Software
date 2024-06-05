#include "bundle_plane_node.h"
#include "infra/defs.h"
#include "node_tensor_accessor.h"
#include "types.h"
#include <algorithm>

BundlePlaneNode::BundlePlaneNode(const TensorVector& inputs, const TensorVector& outputs, const NodePtr& ogNode)
: BaseType(inputs, outputs, ogNode->getNodeName(), TYPE_DEBUG, SHAPE_FUNC_MAX_ID)
{
    m_origNodes.push_back(ogNode);
    m_annotation.flashAttentionInfo = ogNode->getNodeAnnotation().flashAttentionInfo;
}

BundlePlaneNode::BundlePlaneNode(const TensorVector& inputs,
                                 const TensorVector& outputs,
                                 const NodeVector&   ogNodes,
                                 const BundleInfo&   bundleInfo)
: BaseType(inputs, outputs, ogNodes.front()->getNodeName(), TYPE_DEBUG, SHAPE_FUNC_MAX_ID),
  m_bundleInfo(bundleInfo),
  m_origNodes(ogNodes)
{
    m_annotation.flashAttentionInfo = ogNodes.front()->getNodeAnnotation().flashAttentionInfo;
}

NodePtr BundlePlaneNode::clone() const
{
    return NodePtr(new BundlePlaneNode(*this));
}

bool BundlePlaneNode::validateNodeForGraph(const HabanaGraph& g) const
{
    return true;
}

bool BundlePlaneNode::isBundle() const
{
    return m_bundleInfo.is_set();
}

void BundlePlaneNode::setBundle(const BundleInfo& bundleInfo)
{
    m_bundleInfo.set(bundleInfo);
    setName(fmt::format("Bundle{}", m_bundleInfo->bundleIndex));
}

void BundlePlaneNode::unsetBundle()
{
    m_bundleInfo.unset();
    if (!m_origNodes.empty())
    {
        setName(m_origNodes.front()->getNodeName());
    }
}

void BundlePlaneNode::addNodeToBundle(const NodePtr& ogNode)
{
    HB_ASSERT(isBundle(), "Can't fuse into non-bundle BP node.");
    m_origNodes.push_back(ogNode);
}

// remove a single node from a bundle node
void BundlePlaneNode::removeNodeFromBundle(const NodePtr&                                 ogNode,
                                           const std::unordered_map<uint32_t, TensorPtr>& OGTensorToBPTensor)
{
    HB_ASSERT(m_origNodes.size() > 1, "cannot delete node from bundle of size 1");
    auto it = std::find(m_origNodes.begin(), m_origNodes.end(), ogNode);
    HB_ASSERT(it != m_origNodes.end(), "original node not found in bundle plane node!");
    m_origNodes.erase(it);

    runOnTensorsForType<Node::USAGE_INPUT>(ogNode, TENSOR_TYPE_ALL, [&](const TensorPtr& in) {
        if (!in) return;
        auto bpTensorIt = OGTensorToBPTensor.find(in->getId());
        if (bpTensorIt == OGTensorToBPTensor.end()) return;  // original ternsor doesn't have a BP representation

        auto inputIt = std::find(m_inputs.begin(), m_inputs.end(), bpTensorIt->second);
        if (inputIt == m_inputs.end()) return;  // probably a inner-bundle tensor

        bool unique = std::all_of(m_origNodes.begin(), m_origNodes.end(), [&](const NodePtr& n) {
            const TensorVector& inputs        = n->getInputs();
            const TensorVector& controlInputs = n->getControlInputs();
            return std::find(inputs.begin(), inputs.end(), in) == inputs.end() &&
                   std::find(controlInputs.begin(), controlInputs.end(), in) == controlInputs.end();
        });
        if (unique)  // if the deleted node is the only node in the bundle consuming this tensor
        {
            m_inputs.erase(inputIt);
        }
    });

    runOnTensorsForType<Node::USAGE_OUTPUT>(ogNode, TENSOR_TYPE_ALL, [&](const TensorPtr& out) {
        if (!out) return;
        auto bpTensorIt = OGTensorToBPTensor.find(out->getId());
        if (bpTensorIt == OGTensorToBPTensor.end()) return;

        auto outputIt = std::find(m_outputs.begin(), m_outputs.end(), bpTensorIt->second);
        if (outputIt == m_outputs.end()) return;  // probably a inner-bundle tensor
        m_outputs.erase(outputIt);
    });
}

bool BundlePlaneNode::isLogicalOperation() const
{
    return std::all_of(m_origNodes.begin(), m_origNodes.end(), [](const NodePtr& n) {
        return n->isLogicalOperation();
    });
}

bool BundlePlaneNode::isSplit() const
{
    return std::all_of(m_origNodes.begin(), m_origNodes.end(), [](const NodePtr& n){
        return n->isSplit();
    });
}

const NodeVector& BundlePlaneNode::getBundledNodes() const
{
    return m_origNodes;
}

void BundlePlaneNode::replaceNodeInBundle(const NodePtr& oldNode, const NodePtr& newNode)
{
    auto it = std::find(m_origNodes.begin(), m_origNodes.end(), oldNode);
    HB_ASSERT(it != m_origNodes.end(), "node {} not found in bundle!", oldNode->getNodeName());
    m_origNodes[it - m_origNodes.begin()] = newNode;
}

unsigned BundlePlaneNode::getBundleIndex() const
{
    return m_bundleInfo->bundleIndex;
}
