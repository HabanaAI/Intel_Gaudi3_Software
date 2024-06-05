#pragma once

#include "logical_op_node.h"
#include "node_visitor.h"

using MultiInsertParams = std::vector<uint64_t>;

class MultiInsertNode : public LogicalOpNode
{
    DEFINE_VISITOR_METHOD
    friend class NodeFactory;

public:
    typedef LogicalOpNode BaseClass;

    virtual NodePtr clone() const override;
    virtual bool    validateNode() const override;
    void            runLogicalOperation() const override;

    virtual bool canHandleStridedRealTensor() const override { return false; }

    void printParamsRawData() const override;

    MultiInsertNode(const TensorVector& in, const TensorVector& out, UserParams params, std::string_view name);

    enum StridedInsertInputs
    {
        ORIGINAL_TENSOR = 0,
    };
    virtual void setParams(UserParams userParams, unsigned userParamsSize) override;

protected:
    virtual bool isAliasStrided() const override { return false; }

private:
    MultiInsertParams m_inputOffsets;

    static NodePtr createNode(const TensorVector& inputs,
                              const TensorVector& outputs,
                              UserParams          userParams,
                              std::string_view    guid,
                              std::string_view    name);
};
