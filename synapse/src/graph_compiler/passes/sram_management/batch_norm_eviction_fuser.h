#pragma once

#include "habana_graph.h"
#include "bundle.h"
#include "slicing_strategy.h"
#include "slice_mapping.h"
#include "types.h"

// Objects of this class identify batch_norm_stageN nodes in the bundle and intermediate tensors that needs eviction and
// add optional outputs to the BN kernels to perform the eviction.
class BatchNormStagesEvictionFuser
{
public:
    BatchNormStagesEvictionFuser(HabanaGraph& graph, const pBundle& bundle, const SlicingStrategyPtr& strategy);

    // Adds the relevant evictions through BN optional outputs. Modify the graph!
    void fuseEvictions(bool stage1Fwd = true,
                       bool stage2Fwd = true,
                       bool stage1bwd = true);  // Until fusing for each stage has been proved to improve runtime,
                                                // enable not fusing some stage
private:
    using SlicedOperandList  = std::list<pSlicedOperand>;
    using InOutSlicedOpLists = std::pair<SlicedOperandList, SlicedOperandList>;

    HabanaGraph&       m_graph;
    NodeSet            m_bundleNodes;
    SlicingStrategyPtr m_strategy;

    static const unsigned BN1_FWD_IFM_INPUT_IDX       = 0;
    static const unsigned BN1_FWD_IFM_COPY_OUTPUT_IDX = 1;
    static const unsigned BN1_BWD_GRAD_OUT_IDX        = 0;
    static const unsigned BN1_BWD_GRAD_OUT_COPY_IDX   = 1;

    void               checkAndFuseBN1FwdEviction(NodePtr bn1Fwd);
    TensorPtr          fuseBN1FwdEviction(NodePtr& bn1Fwd, const TensorPtr& bnIfm);
    InOutSlicedOpLists getBN1SlicedOperands(const pSlicedOperand& bn1SlicedIfm) const;
    void replaceBN1IfmInStrategy(const NodePtr& bn1Fwd, const TensorPtr& bnIfm, const TensorPtr& bnIfmCopy);

    static const unsigned BN2_FWD_OFM_IDX      = 0;
    static const unsigned BN2_FWD_OFM_COPY_IDX = 1;

    void               checkAndFuseBN2FwdEviction(NodePtr bn2Fwd);
    TensorPtr          fuseBN2FwdEviction(NodePtr& bn2Fwd, const TensorPtr& bnOfm);
    InOutSlicedOpLists getBN2SlicedOperands(const pSlicedOperand& bn2SlicedOfm) const;
    void replaceBN2OfmInStrategy(const NodePtr& bn2Fwd, const TensorPtr& bnOfm, const TensorPtr& bnOfmCopy);

    bool           requiresEviction(const TensorPtr& tensor) const;
    NodeVector     findBundledTensorConsumers(const TensorPtr& tensor) const;
    pSlicedOperand generateEvictedSlicedOperand(pSlicedOperand orig, const TensorPtr& newIntermediateTensor);
    void           reInstantiateTpcNode(NodePtr& node) const;
};