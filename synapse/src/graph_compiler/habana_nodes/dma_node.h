#pragma once

#include "node.h"
#include "habana_nodes.h"
#include "synapse_common_types.h"
#include "synapse_types.h"
#include "node_visitor.h"
#include "tensor.h"

class DMANode : public Node
{
    DEFINE_VISITOR_METHOD
public:
    typedef Node BaseClass;

    DMANode(const TensorPtr& t,
            std::string_view name,
            DMA_TYPE         dmaType = DMA_TYPE_DOWNSTREAM,
            ShapeFuncID      sifId   = SHAPE_FUNC_MAX_ID);
    DMANode(const TensorPtr& input,
            const TensorPtr& output,
            std::string_view name,
            DMA_TYPE         dmaType,
            ShapeFuncID      sifId = SHAPE_FUNC_MAX_ID);
    DMANode(const TensorVector& input,
            const TensorVector& output,
            std::string_view    name,
            DMA_TYPE            dmaType,
            ShapeFuncID         sifId = SHAPE_FUNC_MAX_ID);

    virtual NodePtr     clone() const override;
    virtual void        print() const override;
    const DMA_TYPE&     getDmaType() const { return m_dmaType; }
    virtual void        setDmaType(DMA_TYPE type) { m_dmaType = type; }
    virtual std::string_view getEngineTypeStr() const override;
    virtual bool        validateNode() const override;
    virtual NodeROI     generateRoi() const override;
    virtual bool        isLinearDma() const;
    virtual bool        isMemset() const override;
    bool                isPrefetch() const { return m_dmaType == DMA_TYPE_PREFETCH_STATIC_TENSORS; }
    virtual DMA_OP_TYPE getOpType() const;
    virtual uint64_t    parallelLevel() const;
    virtual void        setParallelLevel(uint64_t pLevel);
    virtual uint64_t    dispatcherIndex() const;
    virtual void        setDispatcherIndex(uint64_t pLevel);
    virtual uint64_t    chunkSizeInBytes() const;

    virtual bool                isDynamicMemoryOp() const { return false; }
    virtual DYNAMIC_MEM_OP_TYPE getDynamicMemoryOpType() const { return DMA_OP_NONE; }

    virtual bool          isBroadcast() const { return false; }
    virtual bool          canHaveAdditionalInputs() const { return false; }
    virtual bool          isROIDynamic(const NodeROI* roi) const override;
    virtual bool          validateNodeForGraph(const HabanaGraph& g) const override;
    bool                  isNodeTwoDimensionalStrided() const;
    DimVector             getSplitDimensionsOrder();
    HabanaDeviceType      getNodeDeviceType() const override;

protected:
    bool isTensorTwoDimensionalStrided(TensorPtr t) const;

    DMA_TYPE m_dmaType;
    uint64_t m_parallelLevel   = 1;
    uint64_t m_dispatcherIndex = 0;

private:
    static inline bool isTensorInputForDMANode(DMA_TYPE dmaType)
    {
        return dmaType == DMA_TYPE_UPSTREAM || dmaType == DMA_TYPE_INTERMEDIATES;
    }
    void generateRoiForStridedMemcpy(NodeROI& roi) const;
};
