#pragma once

#include "slice_logical_node.h"
#include "node_visitor.h"

class LogicalSliceInsertNode : public LogicalSliceNode
{
    DEFINE_VISITOR_METHOD
    friend class NodeFactory;

public:
    typedef LogicalSliceNode BaseClass;

    LogicalSliceInsertNode(const TensorVector& inputs,
                           const TensorVector& outputs,
                           UserParams          params,
                           std::string_view    name);
    NodePtr clone() const override;

    bool validateNode() const override;
    bool isRedundantNode() const override { return false; };

    virtual NStrideArray calculateAliasStrides(unsigned idx) const override;
    void runLogicalOperation() const override;

protected:
    bool isAliasStrided(unsigned idx) const override;

private:
    static NodePtr createNode(const TensorVector& inputs,
                              const TensorVector& outputs,
                              UserParams          userParams,
                              std::string_view    guid,
                              std::string_view    name);
};