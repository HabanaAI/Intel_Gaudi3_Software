#pragma once

#include <node.h>
#include <node_visitor.h>

class BundlePlaneNode : public Node
{
    DEFINE_VISITOR_METHOD

    using BaseType = Node;

public:
    BundlePlaneNode(const TensorVector& inputs, const TensorVector& outputs, const NodePtr& ogNode);

    BundlePlaneNode(const TensorVector& inputs,
                    const TensorVector& outputs,
                    const NodeVector&   ogNodes,
                    const BundleInfo&   bundleInfo);

    BundlePlaneNode(const BundlePlaneNode& other) = default;

    NodePtr clone() const override;

    bool validateNodeForGraph(const HabanaGraph& g) const override;

    bool isBundle() const;

    void setBundle(const BundleInfo& bundleInfo);

    void unsetBundle();

    void addNodeToBundle(const NodePtr& ogNode);

    void removeNodeFromBundle(const NodePtr& ogNode, const std::unordered_map<uint32_t, TensorPtr>& OGTensorToBPTensor);

    void replaceNodeInBundle(const NodePtr& oldNode, const NodePtr& newNode);

    const NodeVector& getBundledNodes() const;

    unsigned getBundleIndex() const;

    bool isLogicalOperation() const override;

    bool isSplit() const override;

private:
    Settable<BundleInfo> m_bundleInfo;
    NodeVector           m_origNodes;
};