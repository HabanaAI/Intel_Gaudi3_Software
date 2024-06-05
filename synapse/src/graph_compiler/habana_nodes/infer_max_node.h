#pragma once

#include "logical_op_node.h"
#include "node_visitor.h"
#include "shape_inference_metadata.h"

/**
 * input is dynamic tensor, and the output is static tensor with the input max sizes
 * there is optional output shape tensor that extracts the input shape
 */
class InferMaxShapeNode : public LogicalOpNode
{
    DEFINE_VISITOR_METHOD
    friend class NodeFactory;

public:
    typedef LogicalOpNode BaseClass;

    virtual NodePtr clone() const override;
    virtual bool    validateNode() const override;
    virtual bool    validateDynamicShapes() const override;
    virtual void    runLogicalOperation() const override;
    virtual void    setParams(UserParams userParams, unsigned userParamsSize) override;
    virtual bool    isAliasStrided() const override { return !getRealTensor()->isDenseLayout(); }
    virtual bool    canSwapAliasDirection() const override { return true; }
    virtual bool    canHandleStridedRealTensor() const override { return true; }

    virtual NStrideArray calculateAliasStrides(unsigned idx) const override;

    virtual SifNodeParams getShapeInferenceFunctionUserParams() override;
    virtual size_t        getShapeInferenceFunctionUserParamsSize() const override { return sizeof(m_sifMetadata); }

private:
    InferMaxShapeNode(const TensorVector& inputs,
                      const TensorVector& outputs,
                      UserParams          userParams,
                      std::string_view    name = "",
                      Node::eNodeType     type = Node::TYPE_INFER_MAX_SHAPE);

    static NodePtr createNode(const TensorVector& inputs,
                              const TensorVector& outputs,
                              UserParams          userParams,
                              std::string_view    guid,
                              std::string_view    name);

    SifInferMaxShapeMetadata m_sifMetadata;
};
