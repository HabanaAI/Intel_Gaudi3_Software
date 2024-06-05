#pragma once

#include "types.h"
#include "node.h"
#include "node_visitor.h"

class FrobeniusNormNode : public Node
{
    DEFINE_VISITOR_METHOD
    friend class NodeFactory;

public:
    using BaseClass = Node;

    NodePtr clone() const override;
    bool    validateNode() const override;
    bool    validateNodeForGraph(const HabanaGraph& g) const override;

private:
    FrobeniusNormNode(const TensorVector& in, const TensorVector& out, std::string_view name);

    static NodePtr createNode(const TensorVector& inputs,
                              const TensorVector& outputs,
                              UserParams          userParams,
                              std::string_view    guid,
                              std::string_view    name);
};
