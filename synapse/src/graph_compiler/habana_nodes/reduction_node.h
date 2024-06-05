#pragma once

#include "logical_op_node.h"
#include "node_visitor.h"

class ReductionNode : public LogicalOpNode
{
    DEFINE_VISITOR_METHOD
    friend class NodeFactory;

public:
    typedef LogicalOpNode BaseClass;

    virtual NodePtr clone() const override;
    virtual bool  validateNode() const override;
    void          runLogicalOperation() const override;

    virtual NStrideArray calculateAliasStrides(unsigned idx) const override;

    virtual TensorShape getInputShape(const TensorShape& output, uint32_t outputIndex, uint32_t inputIdx) const override;
    virtual bool        validateNodeForGraph(const HabanaGraph&) const override;

    ReductionOperation getReductionOperation() const;

    virtual bool canHandleStridedRealTensor() const override { return true; }

    void printParamsRawData() const override;

    bool linkConsumedMemsetShape(const HabanaGraph& graph) const;

    virtual void setParams(UserParams userParams, unsigned userParamsSize) override;

protected:
    virtual bool isAliasStrided() const override { return !getRealTensor()->isDenseLayout(); }

    gc::access_pattern::NodeAccessPatternPtr generateNodeAccessPattern() const override;

private:
    ReductionOperation m_reductionOperation;

    ReductionNode(const TensorVector& in, const TensorVector& out, UserParams params, std::string_view name);

    static NodePtr createNode(const TensorVector& inputs,
                              const TensorVector& outputs,
                              UserParams          userParams,
                              std::string_view    guid,
                              std::string_view    name);
};
