#pragma once

#include "defs.h"
#include "graph_traits.h"
#include "habana_nodes.h"
#include "habana_graph.h"
#include "logical_op_node.h"
#include "multi_node.h"
#include "node.h"
#include "dma_node.h"
#include <string_view>

class LogicalBroadcastNode : public LogicalOpNode
{
    DEFINE_VISITOR_METHOD
public:
    typedef LogicalOpNode BaseClass;

    NodePtr         clone() const override;
    bool            validateNode() const override;
    bool            RunOnCpu() override;
    void            runLogicalOperation() const override;
    bool            isRedundantNode() const override;

    LogicalBroadcastNode(const TensorVector& inputs, const TensorVector& outputs, const std::string& name);

protected:
    bool isAliasStrided() const override { return true; }
};

class FcdBroadcastNode : public MultiNode
{
    DEFINE_VISITOR_METHOD
public:
    typedef MultiNode BaseClass;

    FcdBroadcastNode(const TensorVector& inputs, const TensorVector& outputs, const std::string& name);

    NodePtr          clone() const override;
    bool             validateNode() const override;
    bool             RunOnCpu() override;
    NodeList         extract() override
    {
        HB_ASSERT(false, "Habana graph required");
        return {};
    }
    NodeList extract(const HabanaGraph& g) override;

    bool validateNodeForGraph(const HabanaGraph&) const override;

    bool isDataMovementMultiNode() const override { return true; };

private:
    void             extractNodesWithReducedDims(const TensorPtr& input,
                                                 const TensorPtr& output,
                                                 const TensorPtr& shapeTensor,
                                                 NodeList&        nodes) const;
    void             reduceDims(TensorPtr& input,
                                TensorPtr& output,
                                TensorPtr& shapeTensor,
                                NodeList&  nodes,
                                unsigned   maxTensorDim) const;
    TensorPtr        handleShapeTensor(const TensorPtr& shapeTensor, NodeList& nodes) const;
    static TensorPtr createFlattenedTensor(const TensorPtr& tensor, unsigned dims);

    NodeList splitToFcdAndRegularBroadcasts() const;
};

class BroadcastNode : public MultiNode
{
    DEFINE_VISITOR_METHOD
    friend class NodeFactory;
    friend class BroadcastNodeCreator;

public:
    typedef MultiNode BaseClass;

    NodePtr          clone() const override;
    bool             validateNode() const override;
    bool             RunOnCpu() override;
    NodeList         extract() override;
    bool             validateNodeForGraph(const HabanaGraph&) const override;

    static bool validateBroadcast(const Node* node);

    bool isDataMovementMultiNode() const override { return true; };

private:
    BroadcastNode(const TensorVector& inputs, const TensorVector& outputs, std::string_view name);

    static NodePtr createNode(const TensorVector& inputs,
                              const TensorVector& outputs,
                              UserParams          userParams,
                              std::string_view    guid,
                              std::string_view    name);
};

class DMABroadcastNode : public DMANode
{
    DEFINE_VISITOR_METHOD

public:
    static const unsigned MAX_SUPPORTED_DIM = 2;
    NodePtr               clone() const override;
    bool                  validateNode() const override;
    bool                  validateNodeForGraph(const HabanaGraph& g) const override;

    DMA_OP_TYPE getOpType() const override;

    bool RunOnCpu() override;
    bool isBroadcast() const override { return true; }

    std::string getNodeTypeStr() const override { return "DmaBroadcast"; }

    DMABroadcastNode(const TensorVector& in, const TensorVector& out, const std::string& name);

    void replaceOutput(unsigned index, const TensorPtr& newTensor) override;

    bool canHandleStridedOutput(synDeviceType device = synDeviceTypeInvalid) const override { return false; }
};
