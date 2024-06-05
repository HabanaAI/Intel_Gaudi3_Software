#pragma once

#include "logical_op_node.h"
#include "node_visitor.h"

/**
 * This node replace nodes with binding input reuse property, only during the "handle logical operation" pass.
 * It get unbounded number of operands, and two indices, one for an output and one for an input,
 * and behave like identity node between these tensors.
 */
class OperandReuseInternalLogicalNode : public LogicalOpNode
{
    DEFINE_VISITOR_METHOD

public:
    typedef LogicalOpNode BaseClass;

    virtual bool         validateNode() const override { return true; };
    virtual bool         isRedundantNode() const override { return false; };
    virtual bool         canHandleStridedRealTensor() const override { return true; }
    virtual NStrideArray calculateAliasStrides(unsigned idx) const override;
    virtual bool         canSwapAliasDirection() const override { return false; }
    virtual NodePtr      clone() const override;
    virtual void         runLogicalOperation() const override;
    virtual TensorPtr    getRealTensor() const override;
    virtual TensorVector getAliasTensors() const override;

    virtual uint64_t      getShapeInferenceFunctionVersion() const override;
    virtual SifNodeParams getShapeInferenceFunctionUserParams() override;
    virtual size_t        getShapeInferenceFunctionUserParamsSize() const override;

    const TensorPtr& getAliasTensor() const;

    OperandReuseInternalLogicalNode(const NodePtr& node, const unsigned inputIndex, const unsigned outputIndex);

    NodePtr getOriginalNode() const { return m_originalNode; }

private:
    const unsigned m_inputIndex;
    const unsigned m_outputIndex;
    const NodePtr  m_originalNode;
};
