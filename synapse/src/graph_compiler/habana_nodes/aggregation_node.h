#pragma once

#include "logical_op_node.h"
#include "node_visitor.h"

class AggregationNode : public LogicalOpNode
{
    DEFINE_VISITOR_METHOD
public:
    typedef LogicalOpNode BaseClass;
    virtual ~AggregationNode() = default;

    virtual bool validateNode() const override;
    virtual void runLogicalOperation() const override;
    virtual bool isRedundantNode() const override;
    virtual void printParamsRawData() const override;

    unsigned getAggregationDim() const { return m_aggDim; }

    virtual bool canHandleStridedRealTensor() const override { return true; }

    virtual bool isAliasStrided() const override;

    virtual NStrideArray calculateAliasStrides(unsigned idx) const override;

    static bool validateAggregation(const Node*     node,
                                    const TensorVector& aggTensorsVec,
                                    const TensorVector& aggregations,
                                    unsigned            aggDim);

    void permuteParams(const PermutationVector& inputPermutations) override;

    virtual void setParams(UserParams userParams, unsigned int userParamsSize) override;

protected:
    AggregationNode(const TensorVector& inputs,
                    const TensorVector& outputs,
                    std::string_view    name,
                    AliasDirection      direction,
                    eNodeType           type,
                    ShapeFuncID         sifId,
                    UserParams          userParams);

    unsigned m_aggDim;
};
