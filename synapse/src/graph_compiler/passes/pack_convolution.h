#pragma once

#include "habana_graph.h"
#include "types.h"
class ConvolutionPackingManager
{
public:
    virtual ~ConvolutionPackingManager() = default;
    explicit ConvolutionPackingManager(HabanaGraph& g) : m_graph(g) {}
    bool packConvolutionNodes();

protected:
    bool         isCandidateForPacking(const NodePtr& node) const;
    virtual bool packingEnabled() const = 0;
    virtual void handleSameTensorDiffPackingFactor(const NodePtr& node, unsigned packingFactor) {};
    virtual void packConvNode(const NodePtr& node);
    unsigned     getKernelWidthAfterPacking(unsigned packingFactor, unsigned kernelWidth, unsigned strideWidth) const;
    unsigned     getStrideWidthAfterPacking(unsigned packingFactor, unsigned strideWidth) const;
    virtual bool packingFactorIsValid(unsigned                  packingFactor,
                                      const TensorPtr&          inputTensor,
                                      const TensorPtr&          wTensor,
                                      const TensorPtr&          outputTensor,
                                      synConvolution3DParamsV2& convParams,
                                      bool                      isBwd) const;
    unsigned     choosePackingFactor(const NodePtr& node) const;
    virtual bool shouldBlockPacking(const synConvolution3DParamsV2& convParams,
                                    const TensorPtr&                wTensor,
                                    const TensorPtr&                outputTensor,
                                    bool                            isBwd) const;
    // TODO SW-47037 - move this func to new class with MME HAL logic
    virtual unsigned
                 minCDElementsForFullMmeUtil(synDataType inType, synDataType outType, unsigned outputHeight) const = 0;
    virtual void packWeights(const MMENodePtr& node, const TensorPtr& t, unsigned stride, unsigned packingFactor) = 0;
    virtual void packOutput(const NodePtr& node, const TensorPtr& outTensor, unsigned packingFactor);
    virtual void     applyChangeToGraph(const NodePtr& node) = 0;
    virtual void     resetTemporaryState()                   = 0;
    virtual unsigned getTpcLoweringPackingFactor(const MMENodePtr& node) const { return 0; };
    void             reshapeOutputTensor(const TensorPtr& t, unsigned packingFactor);
    void             updatePackedOutputSizes(SizeArray& sizes, unsigned packingFactor);

    HabanaGraph&                                       m_graph;
    NodePtr                                            m_reshapeNodeOfm;
};
