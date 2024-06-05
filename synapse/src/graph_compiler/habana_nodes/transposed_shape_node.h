#pragma once

#include "shape_op_node.h"
#include "node_visitor.h"
#include "transpose_node.h"

// Extract transposed Shape Node is responsible to create a transposed shape tensor out of another tensor.
// It can be used to save the shape of a tensor which data is no longer needed for the purpose of shape inference.

class TransposedShapeNode : public ShapeOperationNode<LogicalTransposeNode>
{
    DEFINE_VISITOR_METHOD
    friend class NodeFactory;

public:
    using BaseClass = ShapeOperationNode<LogicalTransposeNode>;

    ~TransposedShapeNode() override = default;

    bool validateNode() const override;
    NodePtr clone() const override;
    virtual bool RunOnCpu() override {return true;};
    virtual bool isRedundantNode() const override {return false;};
    virtual void setParams(UserParams userParams, unsigned userParamsSize) override;

private:
    TransposedShapeNode(const TensorPtr& input,
                        const TensorPtr& output,
                        UserParams       params,
                        unsigned         paramsSize,
                        std::string_view name);

    static NodePtr createNode(const TensorVector& inputs,
                              const TensorVector& outputs,
                              UserParams          userParams,
                              unsigned            userParamsSize,
                              std::string_view    guid,
                              std::string_view    name);
};
