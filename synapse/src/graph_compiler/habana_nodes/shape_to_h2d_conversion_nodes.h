#pragma once
// temporary nodes to convert IDSTs to H2D tensors for nodes that use them
#include "logical_op_node.h"
#include "node_visitor.h"

class StridedOpsConversionNode : public LogicalOpNode
{
    DEFINE_VISITOR_METHOD
    friend class NodeFactory;

public:
    typedef LogicalOpNode BaseClass;

    StridedOpsConversionNode(const TensorVector& inputs, const TensorVector& outputs, std::string_view name);

    virtual NodePtr clone() const override;
    virtual bool    validateNode() const override;
    virtual bool    isH2DManipulationNode() const override { return true; }
    virtual void    runLogicalOperation() const override {}

protected:
    virtual SifNodeParams getShapeInferenceFunctionUserParams() override { return nullptr; }
    virtual size_t        getShapeInferenceFunctionUserParamsSize() const override { return 0; }

private:
    static NodePtr createNode(const TensorVector& inputs,
                              const TensorVector& outputs,
                              UserParams          userParams,
                              std::string_view    guid,
                              std::string_view    name);
};

class SliceConversionNode : public LogicalOpNode   // converts _from_ H2D _to_ shape tensors
{
    DEFINE_VISITOR_METHOD
    friend class NodeFactory;

public:
    typedef LogicalOpNode BaseClass;

    SliceConversionNode(const TensorVector& inputs, const TensorVector& outputs, std::string_view name);

    virtual NodePtr clone() const override;
    virtual bool    validateNode() const override;
    virtual bool    isH2DManipulationNode() const override { return true; }
    virtual void    runLogicalOperation() const override {}
    virtual bool    isDynamicShape() const override { return true; } // because H2D is input here

protected:
    virtual SifNodeParams getShapeInferenceFunctionUserParams() override { return nullptr; }
    virtual size_t        getShapeInferenceFunctionUserParamsSize() const override { return 0; }

private:
    static NodePtr createNode(const TensorVector& inputs,
                              const TensorVector& outputs,
                              UserParams          userParams,
                              std::string_view    guid,
                              std::string_view    name);
};