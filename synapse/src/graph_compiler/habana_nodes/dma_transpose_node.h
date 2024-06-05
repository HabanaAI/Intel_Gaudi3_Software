#pragma once

#include "habana_nodes.h"
#include "node_visitor.h"
#include "transpose_permutation.h"
#include "transpose_utils.h"
#include "dma_node.h"

class SplitStrategy;

class DMATransposeNode : public DMANode
{
    DEFINE_VISITOR_METHOD
    friend class NodeFactory;

    static const TransposePermutationArray s_permutation;

public:
    virtual NodePtr  clone()        const override;
    virtual bool     validateNode() const override;
    virtual bool     validateNodeForGraph(const HabanaGraph& g) const override;

    static bool isSupported(synDeviceType type) { return true; }

    virtual bool isTranspose() const override { return true; };
    virtual bool isROIDynamic(const NodeROI* roi) const override { return false; };

    virtual DMA_OP_TYPE getOpType() const override;
    void                setIsFullyUtilized(bool fullyUtilized) { m_isFullyUtilized = fullyUtilized; }
    bool                isFullyUtilized() { return m_isFullyUtilized; }

    const std::shared_ptr<SplitStrategy>& getSplitStrategy();

    virtual bool RunOnCpu() override;
    const TransposePermutationArray& permutation() const { return m_permutation; }

    std::string getNodeTypeStr() const override { return "DmaTranspose"; }

    virtual bool canHandleStridedOutput(synDeviceType device = synDeviceTypeInvalid) const override { return false; }
    virtual void setParams(UserParams userParams, unsigned userParamsSize) override;

protected:
    virtual SifNodeParams getShapeInferenceFunctionUserParams() override;
    virtual size_t getShapeInferenceFunctionUserParamsSize() const override;
    DMATransposeNode(const TensorPtr& in, const TensorPtr& out, std::string_view name);

    gc::access_pattern::NodeAccessPatternPtr generateNodeAccessPattern() const override;

    virtual void replaceOutput(unsigned index, const TensorPtr& newTensor) override;

private:
    static NodePtr createNode(const TensorVector& inputs,
                              const TensorVector& outputs,
                              UserParams          userParams,
                              std::string_view    guid,
                              std::string_view    name);

    pSifTransposeMetadata          m_sifMetadata;
    std::shared_ptr<SplitStrategy> m_splitStrategy;
    TransposePermutationArray      m_permutation;
    bool                           m_isFullyUtilized = false;
};

class StridedDMANodeViaTransposeNode : public DMATransposeNode
{
    DEFINE_VISITOR_METHOD
    friend class StridedMemcpyViaTransposeEngineStrategy;
    using Node = DMATransposeNode;

    StridedDMANodeViaTransposeNode(const TensorPtr& in, const TensorPtr& out, std::string_view name)
    : DMATransposeNode(in, out, name)
    {
        TransposePermutationArray params = getIdentityPermutation(2);
        setParams((void*)&params, sizeof(params));
    }

public:
    virtual NodePtr clone() const override { return NodePtr(new StridedDMANodeViaTransposeNode(*this)); }
    virtual bool    validateNode() const override;

    std::string getNodeTypeStr() const override { return "StridedDmaViaTranspose"; }
};
