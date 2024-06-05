#pragma once

#include "multi_node.h"
#include "node_visitor.h"

class NMSNode : public MultiNode
{
    DEFINE_VISITOR_METHOD
    friend class NodeFactory;

public:
    typedef MultiNode BaseClass;

    virtual NodePtr clone() const override;

    virtual NodeList extract() override;

    virtual bool validateNodeForGraph(const HabanaGraph& g) const override;

    virtual void setParams(UserParams userParams, unsigned userParamsSize) override;

    void printParamsRawData() const override;

private:

    void gatherTopkBoxes(TensorPtr boxesTensor, TensorPtr topkIndicesTensor, TensorPtr topkBoxes, NodeList& retNodes);

    synNMSParams m_params;

    NMSNode(const TensorVector& inputs, const TensorVector& outputs, UserParams params, std::string_view name);

    static NodePtr createNode(const TensorVector& inputs,
                              const TensorVector& outputs,
                              UserParams          userParams,
                              std::string_view    guid,
                              std::string_view    name);
};
