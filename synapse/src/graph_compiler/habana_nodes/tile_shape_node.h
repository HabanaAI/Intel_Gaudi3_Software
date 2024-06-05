#pragma once

#include "node_visitor.h"
#include "logical_op_node.h"
#include "shape_inference_metadata.h"
#include "shape_op_node.h"
#include "perf_lib_layer_params.h"

// The node input is dynamic shape tensor, and tile params such that for every dim:
//      iMin[dim]    * params.repeat[dim] = oMin[dim].
//      iMax[dim]    * params.repeat[dim] = oMax[dim].
//      iActual[dim] * params.repeat[dim] = oActual[dim].
class TileShapeNode : public ShapeOperationNode<LogicalOpNode>
{
    DEFINE_VISITOR_METHOD
    friend class NodeFactory;

public:
    typedef ShapeOperationNode<LogicalOpNode> BaseClass;

    virtual NodePtr clone() const override;
    virtual bool    validateNode() const override;

    virtual SifNodeParams getShapeInferenceFunctionUserParams() override;
    virtual size_t        getShapeInferenceFunctionUserParamsSize() const override { return sizeof(m_params); }

private:
    TileShapeNode(const TensorVector& inputs,
                  const TensorVector& outputs,
                  UserParams          userParams,
                  unsigned            userParamsSize,
                  std::string_view    name = "");

    virtual void setParams(UserParams userParams, unsigned userParamsSize) override;

    static NodePtr createNode(const TensorVector& inputs,
                              const TensorVector& outputs,
                              UserParams          userParams,
                              unsigned            userParamsSize,
                              std::string_view    guid,
                              std::string_view    name);

    ns_TileKernel::ParamsV2 m_params;
};