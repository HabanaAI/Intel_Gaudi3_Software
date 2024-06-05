#pragma once

#include "logical_op_node.h"
#include "slice_node.h"
#include "node_visitor.h"

class LogicalSliceNode : public LogicalOpNode
{
    DEFINE_VISITOR_METHOD
public:
    typedef LogicalOpNode BaseClass;

    LogicalSliceNode(const TensorVector& inputs,
                     const TensorVector& outputs,
                     UserParams          params,
                     AliasDirection      direction,
                     std::string_view    name,
                     eNodeType           nodeType,
                     ShapeFuncID         sifId);

    bool    validateNode() const override;
    NodePtr clone() const override = 0;

    bool         RunOnCpu() override;
    void         runLogicalOperation() const override = 0;
    bool         isRedundantNode() const override;
    bool         canHandleStridedRealTensor() const override { return true; }

    std::pair<NStrideArray, uint64_t> calcStridesAndOffset(const TensorPtr& real) const;

    void                             runSlice(TensorPtr real, TensorPtr aliased) const;
    SliceNode::SliceNodeStaticParams getParams() const { return m_params; };

    void printParamsRawData() const override;
    void setParams(UserParams userParams, unsigned userParamsSize) override;

protected:
    bool          isAliasStrided() const override { return true; }
    bool          isAliasStrided(unsigned idx) const override { return isAliasStrided(); }
    SifNodeParams getShapeInferenceFunctionUserParams() override;
    size_t        getShapeInferenceFunctionUserParamsSize() const override;

    SliceNode::SliceNodeStaticParams m_params;
};