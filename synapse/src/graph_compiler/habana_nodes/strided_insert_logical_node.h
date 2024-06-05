#pragma once

#include "logical_op_node.h"
#include "node_visitor.h"
#include "synapse_common_types.hpp"

class LogicalStridedInsertNode : public LogicalOpNode
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

    void printParamsRawData() const override;
    virtual void setParams(UserParams userParams, unsigned userParamsSize) override;
    const synStridedOpParams& getParams() const { return m_params; }

    LogicalStridedInsertNode(const TensorVector& in, const TensorVector& out, UserParams params, std::string_view name);

    enum StridedInsertInputs
    {
        ORIGINAL_TENSOR = 0,
        INSERT_TENSOR   = 1
    };

protected:
    virtual bool isAliasStrided(unsigned idx) const override { return idx == INSERT_TENSOR; }

private:
    synStridedOpParams m_params;

    static NodePtr createNode(const TensorVector& inputs,
                              const TensorVector& outputs,
                              UserParams          userParams,
                              std::string_view    guid,
                              std::string_view    name);
};