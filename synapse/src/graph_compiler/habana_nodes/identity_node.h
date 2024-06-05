#pragma once

#include "logical_op_node.h"
#include "node_visitor.h"

/**
 * This node makes sure output is identical to input
 */
class IdentityNode : public LogicalOpNode
{
    DEFINE_VISITOR_METHOD
    friend class NodeFactory;

public:
    typedef LogicalOpNode BaseClass;

    virtual NodePtr clone() const override;

    virtual bool validateNode() const override;

    virtual bool validateDynamicShapes() const override;

    virtual NStrideArray calculateAliasStrides(unsigned idx) const override;

    virtual void runLogicalOperation() const override;

    virtual bool isAliasStrided() const override { return !getRealTensor()->isDenseLayout(); }

    virtual bool canSwapAliasDirection() const override { return true; }

    virtual bool canHandleStridedRealTensor() const override { return true; }

    bool isRedundantNode() const override { return !m_persistent; }

    void markPersistent() { m_persistent = true; };

    static void runLogicalOperation(const TensorPtr& real, const TensorPtr& alias);

protected:
    IdentityNode(const TensorVector& inputs,
                 const TensorVector& outputs,
                 std::string_view    name = "",
                 Node::eNodeType     type = Node::TYPE_IDENTITY);

private:
    static NodePtr createNode(const TensorVector& inputs,
                              const TensorVector& outputs,
                              UserParams          userParams,
                              std::string_view    guid,
                              std::string_view    name);
    // If persistent - it won't be removed when removing redundant nodes
    bool m_persistent = false;

};
