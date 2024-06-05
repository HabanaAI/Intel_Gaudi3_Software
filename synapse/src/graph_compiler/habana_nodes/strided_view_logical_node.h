#pragma once

#include "logical_op_node.h"
#include "node_visitor.h"
#include "synapse_common_types.hpp"

class LogicalStridedViewNode : public LogicalOpNode
{
    DEFINE_VISITOR_METHOD
    friend class NodeFactory;

public:
    typedef LogicalOpNode BaseClass;

    virtual NodePtr clone() const override;
    virtual bool    validateNode() const override;
    void            runLogicalOperation() const override;

    virtual bool canHandleStridedRealTensor() const override { return false; }
    virtual NStrideArray calculateAliasStrides(unsigned idx) const override;

    const synStridedOpParams& getParams() const { return m_params; }

    void printParamsRawData() const override;

    LogicalStridedViewNode(const TensorVector& in, const TensorVector& out, UserParams params, std::string_view name);

    virtual void setParams(UserParams userParams, unsigned userParamsSize) override;

protected:
    virtual bool isAliasStrided() const override { return true; }

private:
    synStridedOpParams m_params;

    static NodePtr createNode(const TensorVector& inputs,
                              const TensorVector& outputs,
                              UserParams          userParams,
                              std::string_view    guid,
                              std::string_view    name);
};
