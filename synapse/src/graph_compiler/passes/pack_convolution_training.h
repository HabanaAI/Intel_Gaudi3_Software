#pragma once

#include "types.h"
#include "synapse_common_types.h"
#include "pack_convolution.h"

class ConvolutionPackingManagerTraining : public ConvolutionPackingManager
{
public:
    ConvolutionPackingManagerTraining(HabanaGraph& g) : ConvolutionPackingManager(g) {};
    ~ConvolutionPackingManagerTraining() {};

protected:
    virtual bool packingEnabled() const override;
    void         packWeights(const MMENodePtr& convNode,
                             const TensorPtr&  origWeights,
                             unsigned          stride,
                             unsigned          packingFactor) override;
    virtual bool shouldBlockPacking(const synConvolution3DParamsV2& convParams,
                                    const TensorPtr&                wTensor,
                                    const TensorPtr&                outputTensor,
                                    bool                            isBwd) const override;
    virtual unsigned
    minCDElementsForFullMmeUtil(synDataType inType, synDataType outType, unsigned outputHeight) const override;
    virtual void packOutput(const NodePtr& node, const TensorPtr& outTensor, unsigned packingFactor) override;
    virtual void applyChangeToGraph(const NodePtr& node) override;
    virtual void resetTemporaryState() override;

    NodePtr   m_packingNode;
    TensorPtr m_shapeTensorInputToRemove;
};

class EagerConvolutionPackingManagerTraining final : public ConvolutionPackingManagerTraining
{
public:
    EagerConvolutionPackingManagerTraining(HabanaGraph& g, const NodePtr& node)
    : ConvolutionPackingManagerTraining(g), m_node(node)
    {
    }
    bool                        canExtract() const;
    std::pair<NodePtr, NodePtr> extract();

protected:
    virtual void applyChangeToGraph(const NodePtr& node) override;
    virtual void resetTemporaryState() override {}

private:
    mutable uint32_t m_packingFactor = 1;
    NodePtr          m_node;
};