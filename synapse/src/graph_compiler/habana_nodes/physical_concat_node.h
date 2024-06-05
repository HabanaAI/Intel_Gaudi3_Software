#pragma once

#include "habana_nodes.h"
#include "node_visitor.h"

class PhysicalConcatNode : public MultiNode
{
    DEFINE_VISITOR_METHOD
    friend class NodeFactory;

public:
    using BaseClass = MultiNode;

    virtual bool          validateNode() const override;
    virtual bool          validateNodeForGraph(const HabanaGraph& g) const override;
    virtual NodeList      extract() override;
    virtual NodePtr       clone() const override;
    virtual void          setParams(UserParams userParams, unsigned userParamsSize) override;

    SifNodeParams getShapeInferenceFunctionUserParams() override;
    size_t        getShapeInferenceFunctionUserParamsSize() const override;

protected:
    PhysicalConcatNode(const TensorVector& in, const TensorVector& out, std::string_view name, UserParams userParams);
    unsigned m_concatDim;
    Settable<SifConcatenateMetadata> m_metadata;

private:
    static NodePtr createNode(const TensorVector& inputs,
                              const TensorVector& outputs,
                              UserParams          userParams,
                              std::string_view    guid,
                              std::string_view    name);
};
