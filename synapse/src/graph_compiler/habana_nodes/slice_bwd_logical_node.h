#pragma once

#include "slice_logical_node.h"
#include "node_visitor.h"

class LogicalSliceBwdNode : public LogicalSliceNode
{
    DEFINE_VISITOR_METHOD
    friend class NodeFactory;

public:
    typedef LogicalSliceNode BaseClass;

    LogicalSliceBwdNode(const TensorVector& inputs,
                        const TensorVector& outputs,
                        UserParams          params,
                        std::string_view    name);
    NodePtr clone() const override;

    bool validateNode() const override;

    virtual NStrideArray calculateAliasStrides(unsigned idx) const override;
    void runLogicalOperation() const override;

private:
    static NodePtr createNode(const TensorVector& inputs,
                              const TensorVector& outputs,
                              UserParams          userParams,
                              std::string_view    guid,
                              std::string_view    name);
};