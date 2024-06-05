#pragma once

#include "node_visitor.h"
#include "habana_nodes.h"
#include "habana_graph.h"
#include "graph_traits.h"

class DynamicRangeNode : public Node
{
    DEFINE_VISITOR_METHOD
public:
    DynamicRangeNode(const TensorVector& inputs,
                     const TensorVector& outputs,
                     std::string_view    name,
                     UserParams          userParams);
    static NodePtr  createNode(const TensorVector& inputs,
                               const TensorVector& outputs,
                               UserParams          userParams,
                               std::string_view    guid,
                               std::string_view    name);
    virtual bool    validateNodeForGraph(const HabanaGraph& g) const override;
    virtual bool    validateNode() const override;
    bool            setRange(double min, double max);
    DynamicRange    getRange() const;
    virtual NodePtr clone() const override;
    virtual void    setParams(UserParams userParams, unsigned userParamsSize) override;

private:
    DynamicRange m_dynamicRange;
};
