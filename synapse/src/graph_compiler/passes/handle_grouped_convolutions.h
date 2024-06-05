#pragma once

#include "convolution_node.h"
#include "habana_graph.h"
#include <unordered_map>

class GroupedConvolutionManager
{
public:
    GroupedConvolutionManager() : m_convNode(nullptr) {}
    GroupedConvolutionManager(const NodePtr& node) { setCurrentNode(node); };
    virtual bool runHandleGroupedConvolutions(HabanaGraph& g);
    bool         canExtract() const;

private:
    void setCurrentNode(const NodePtr& node);

protected:
    static const ConvBaseNode* getGroupedConvolution(const NodePtr& node);
    virtual bool               validateGroupedConvolutionNode() const = 0;
    virtual NodeList           extract(const HabanaGraph& g)          = 0;
    const ConvBaseNode*        m_convNode;
};

class GroupedConvolutionManagerTraining : public GroupedConvolutionManager
{
public:
    GroupedConvolutionManagerTraining() : GroupedConvolutionManager() {}
    GroupedConvolutionManagerTraining(const NodePtr& node) : GroupedConvolutionManager(node) {}
    virtual bool     validateGroupedConvolutionNode() const override;
    virtual NodeList extract(const HabanaGraph& g) override;

private:
    void     createSplitConcatNodes(const ConvBaseNode* n,
                                    TensorVector&       pNewIFMs,
                                    TensorVector&       pNewOFMs,
                                    TensorVector&       pNewWGHs,
                                    TensorVector&       pNewShapeTensors,
                                    TensorPtr&          IFM,
                                    TensorPtr&          OFM,
                                    TensorPtr&          WGH,
                                    TensorPtr&          shapeTensor,
                                    NodeList&           newNodes);
    NodeList splitGConvNodeToSingleGroups(const ConvBaseNode* n);
    void     createSplitNode(const TensorVector& inputs,
                             const TensorVector& outputs,
                             unsigned            splitDim,
                             const std::string&  name,
                             NodeList&           newNodes,
                             bool                isShape = false);
    void    createConcatNode(const TensorVector& inputs,
                             const TensorVector& outputs,
                             unsigned            concatDim,
                             const std::string&  name,
                             NodeList&           newNodes);
    NodePtr  tryReuseExistingSplitNode(const ConvBaseNode* origConv,
                                       const TensorPtr&    tensor,
                                       unsigned            splitDim,
                                       unsigned            numGroups);

    // Holds the split nodes added so far to enable reuse.
    // Maps from split input to split node.
    std::unordered_map<TensorPtr, NodePtr> m_splitNodesCache;
};

class GroupedConvolutionManagerInference : public GroupedConvolutionManager
{
public:
    GroupedConvolutionManagerInference(unsigned mmeVectorSize, bool runLogicalOp);

protected:
    virtual bool     validateGroupedConvolutionNode() const override;
    virtual NodeList extract(const HabanaGraph& g) override;

private:
    unsigned m_mmeVectorSize;
    bool     m_runLogicalOp;
};