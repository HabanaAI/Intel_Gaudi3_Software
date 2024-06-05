#pragma once

#include "node.h"
#include "node_visitor.h"

class MemcpyNode : public Node
{
    DEFINE_VISITOR_METHOD
    friend class NodeFactory;

public:
    typedef Node BaseClass;
    virtual NodePtr  clone()        const override;
    virtual bool     validateNode() const override;
    virtual bool     validateNodeForGraph(const HabanaGraph& g) const override;
    virtual bool     isNode64BitCompatible() const override;

private:
    MemcpyNode(const TensorVector& in, const TensorVector& out, std::string_view name);

    static NodePtr createNode(const TensorVector& inputs,
                              const TensorVector& outputs,
                              UserParams          userParams,
                              std::string_view    guid,
                              std::string_view    name);
};
