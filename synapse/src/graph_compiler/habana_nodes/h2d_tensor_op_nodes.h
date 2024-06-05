#pragma once

#include "logical_op_node.h"
#include "shape_inference_metadata.h"
#include "node_visitor.h"
#include <string_view>

class H2DTensorOpNode : public LogicalOpNode
{
public:
    typedef LogicalOpNode BaseClass;

    H2DTensorOpNode(const TensorVector& inputs, const TensorVector& outputs, std::string_view name, ShapeFuncID sifId);

    virtual bool validateNode() const override;
    virtual bool isH2DManipulationNode() const override { return true; }
    virtual void runLogicalOperation() const override {}
};

class DynamicStridedDmaExpandH2DNode : public H2DTensorOpNode
{
    DEFINE_VISITOR_METHOD
    friend class NodeFactory;

public:
    DynamicStridedDmaExpandH2DNode(const TensorVector& inputs,
                                   const TensorVector& outputs,
                                   unsigned            dim,
                                   std::string_view    name);

    virtual NodePtr clone() const override;
    virtual bool    validateNode() const override;

protected:
    virtual SifNodeParams getShapeInferenceFunctionUserParams() override { return (SifNodeParams)&m_expandDim; }
    virtual size_t        getShapeInferenceFunctionUserParamsSize() const override { return sizeof(m_expandDim); }

private:
    unsigned m_expandDim;

    static NodePtr createNode(const TensorVector& inputs,
                              const TensorVector& outputs,
                              UserParams          userParams,
                              std::string_view    guid,
                              std::string_view    name);
};

class DynamicStridedDmaReinterpretH2DNode : public H2DTensorOpNode
{
    DEFINE_VISITOR_METHOD
    friend class NodeFactory;

public:
    DynamicStridedDmaReinterpretH2DNode(const TensorVector& inputs,
                                        const TensorVector& outputs,
                                        unsigned            dim,
                                        std::string_view    name);

    virtual NodePtr clone() const override;
    virtual bool    validateNode() const override;

protected:
    virtual SifNodeParams getShapeInferenceFunctionUserParams() override { return (SifNodeParams)&m_factor; }
    virtual size_t        getShapeInferenceFunctionUserParamsSize() const override { return sizeof(m_factor); }

private:
    SifReinterpretH2DMetadata m_factor;

    static NodePtr createNode(const TensorVector& inputs,
                              const TensorVector& outputs,
                              UserParams          userParams,
                              std::string_view    guid,
                              std::string_view    name);
};

class DynamicSliceDmaExpandH2DNode : public H2DTensorOpNode
{
    DEFINE_VISITOR_METHOD
    friend class NodeFactory;

public:
    DynamicSliceDmaExpandH2DNode(const TensorVector& inputs,
                                 const TensorVector& outputs,
                                 unsigned            dim,
                                 std::string_view    name);

    virtual NodePtr clone() const override;
    virtual bool    validateNode() const override;

protected:
    virtual SifNodeParams getShapeInferenceFunctionUserParams() override { return (SifNodeParams)&m_expandDim; }
    virtual size_t        getShapeInferenceFunctionUserParamsSize() const override { return sizeof(m_expandDim); }

private:
    unsigned m_expandDim;

    static NodePtr createNode(const TensorVector& inputs,
                              const TensorVector& outputs,
                              UserParams          userParams,
                              std::string_view    guid,
                              std::string_view    name);
};

class TransposeSliceH2DNode : public H2DTensorOpNode
{
    DEFINE_VISITOR_METHOD
    friend class NodeFactory;

public:
    TransposeSliceH2DNode(const TensorVector& inputs,
                     const TensorVector& outputs,
                     UserParams          userParams,
                     unsigned            userParamsSize,
                     std::string_view    name);

    virtual NodePtr clone() const override;
    virtual bool    validateNode() const override;

    virtual void setParams(UserParams userParams, unsigned userParamsSize) override;

protected:
    virtual SifNodeParams getShapeInferenceFunctionUserParams() override { return (SifNodeParams)&m_sifMetadata; }
    virtual size_t        getShapeInferenceFunctionUserParamsSize() const override { return sizeof(m_sifMetadata); }

private:
    TransposePermutationArray m_permutation;
    SifTransposeMetadata      m_sifMetadata;

    static NodePtr createNode(const TensorVector& inputs,
                              const TensorVector& outputs,
                              UserParams          userParams,
                              unsigned            userParamsSize,
                              std::string_view    guid,
                              std::string_view    name);
};
