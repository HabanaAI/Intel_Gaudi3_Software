#pragma once

#include "node.h"

class HabanaGraph;

class MultiNode : public Node
{
public:
    typedef Node BaseClass;

    MultiNode(const TensorVector& inputs,
              const TensorVector& outputs,
              std::string_view    name,
              eNodeType           type,
              ShapeFuncID         sifId = SHAPE_FUNC_MAX_ID);

    virtual bool     RunOnCpu() override;
    virtual bool     RunOnCpu(const HabanaGraph& g) override;

    struct MultiNodeDependency
    {
        NodePtr blocking;
        NodePtr blocked;
        Tensor::ControlEdgeType type;
    };

    using MultiNodeDependencies = SmallVector< MultiNodeDependency, 4>;

    virtual NodeList extract() = 0;
    virtual NodeList extract(const HabanaGraph&) { return extract(); }
    virtual NodeList extract(const HabanaGraph& g, MultiNodeDependencies& deps) { return extract(g); }
    virtual bool     isMultiNode() const override { return true; };
    virtual bool     isDataMovementMultiNode() const { return false; };
};
