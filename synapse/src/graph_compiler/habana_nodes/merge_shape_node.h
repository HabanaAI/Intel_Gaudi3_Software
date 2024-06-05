#pragma once

#include "shape_op_node.h"
#include "node_visitor.h"
#include "sif/shape_inference_metadata.h"

class MergeShapesNode : public ShapeOperationNode<LogicalOpNode>
{
    DEFINE_VISITOR_METHOD
    friend class NodeFactory;

public:
    using BaseClass = ShapeOperationNode<LogicalOpNode>;

    bool validateNode() const override;

    NodePtr clone() const override;

    virtual SifNodeParams getShapeInferenceFunctionUserParams() override;
    virtual size_t        getShapeInferenceFunctionUserParamsSize() const override;

    bool isShapeOperation() const override { return true; }
    virtual void setParams(UserParams userParams, unsigned userParamsSize) override;

private:
    MergeShapesNode(const TensorVector& inputs,
                    const TensorVector& outputs,
                    UserParams          userParams,
                    std::string_view    name);

    static std::string paramsToString(const SifMergeShapesMetadata& params);

    static NodePtr createNode(const TensorVector& inputs,
                              const TensorVector& outputs,
                              UserParams          userParams,
                              std::string_view    guid,
                              std::string_view    name);

    SifMergeShapesMetadata m_params;
};
