#pragma once

#include "logical_op_node.h"

template <typename LOGICAL>
class ShapeOperationNode : public LOGICAL
{
protected:
    using BaseClass = LOGICAL;

    template <typename ... Args>
    ShapeOperationNode(Args&& ... args)
    : LOGICAL(std::forward<Args>(args)...)
    {
        static_assert(std::is_base_of<LogicalOpNode, LOGICAL>::value,
                      "Expects template param to be a subclass of LogicalOpNode");
    }

public:
    ~ShapeOperationNode() override = default;

    void runLogicalOperation()          const override { }
    bool isAliasStrided() const override { return false; }
    bool canSwapAliasDirection()        const override { return false; }
    bool canHandleStridedRealTensor()   const override { return true; }
    bool isShapeOperation()             const override { return true; }
    bool isRedundantNode()             const override { return false; }
};
