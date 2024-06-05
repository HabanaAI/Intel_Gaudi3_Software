#pragma once

#include "logical_op_node.h"
#include "node_visitor.h"
#include "shape_inference_metadata.h"

/**
 * This node makes sure that input element size * FCD size is equal to output element size * FCD size
 * and all other dimensions sizes are same
 */
class ReinterpretCastNode : public LogicalOpNode
{
    DEFINE_VISITOR_METHOD
    friend class NodeFactory;

public:
    typedef LogicalOpNode BaseClass;

    virtual NodePtr clone() const override;
    virtual bool    validateNode() const override;
    virtual void    runLogicalOperation() const override;
    virtual bool    isAliasStrided() const override { return !getRealTensor()->isDenseLayout(); }
    virtual bool    canSwapAliasDirection() const override { return true; }
    virtual bool    canHandleStridedRealTensor() const override { return true; }

    virtual NStrideArray calculateAliasStrides(unsigned idx) const override;

    virtual synDataType getRequiredInputType(uint32_t tensorIdx) const override;
    virtual synDataType getRequiredOutputType(uint32_t tensorIdx) const override;

    virtual SifNodeParams getShapeInferenceFunctionUserParams() override { return (SifNodeParams)(&m_sifMetadata); }
    virtual size_t        getShapeInferenceFunctionUserParamsSize() const override { return sizeof(m_sifMetadata); }

private:
    ReinterpretCastNode(const TensorVector& inputs,
                        const TensorVector& outputs,
                        std::string_view    name = "",
                        Node::eNodeType     type = Node::TYPE_REINTERPRET_CAST);

    static NodePtr createNode(const TensorVector& inputs,
                              const TensorVector& outputs,
                              UserParams          userParams,
                              std::string_view    guid,
                              std::string_view    name);

    SifReinterpretCastMetadata m_sifMetadata;
};
