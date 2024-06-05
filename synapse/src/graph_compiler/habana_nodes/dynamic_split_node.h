#pragma once

#include "aggregation_node.h"

class DynamicSplitNode : public AggregationNode
{
    DEFINE_VISITOR_METHOD
    friend class NodeFactory;

public:
    using BaseClass = AggregationNode;

    virtual NodePtr clone() const override;
    bool            RunOnCpu() override;
    unsigned        getSplitDim() const { return m_aggDim; }
    virtual bool    validateNode() const override;
    virtual void    setParams(UserParams userParams, unsigned int userParamsSize) override;

protected:
    virtual bool          isAliasStrided() const override { return true; }
    virtual SifNodeParams getShapeInferenceFunctionUserParams() override;
    virtual size_t        getShapeInferenceFunctionUserParamsSize() const override;

    DynamicSplitNode(const TensorVector& inputs,
                     const TensorVector& outputs,
                     UserParams          userParams,
                     std::string_view    name,
                     eNodeType           type = Node::TYPE_INTERNAL_SPLIT);

private:
    std::vector<uint8_t> m_sifMetadataBuffer;

    static NodePtr createNode(const TensorVector& inputs,
                              const TensorVector& outputs,
                              UserParams          userParams,
                              std::string_view    guid,
                              std::string_view    name);
};
