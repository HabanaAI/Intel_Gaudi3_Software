#pragma once

#include "node.h"
#include "node_visitor.h"

class MomentsNode : public Node
{
    DEFINE_VISITOR_METHOD
    friend class NodeFactory;

public:
    virtual NodePtr  clone()        const override;
    virtual bool     validateNode() const override;
    virtual bool     validateNodeForGraph(const HabanaGraph& g) const override;

private:
    MomentsNode(const TensorVector& in, const TensorVector& out, std::string_view name);

    static NodePtr createNode(const TensorVector& inputs,
                              const TensorVector& outputs,
                              UserParams          userParams,
                              std::string_view    guid,
                              std::string_view    name);
};