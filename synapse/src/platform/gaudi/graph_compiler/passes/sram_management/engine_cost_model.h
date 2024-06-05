#pragma once

#include <unordered_map>
#include <unordered_set>
#include "graph_compiler/passes/sram_management/bundle.h"
#include "cost_model.h"
#include "platform/gaudi/graph_compiler/descriptor_generator.h"
#include "graph_compiler/passes/sram_management/tensor_slicer.h"
#include "graph_compiler/passes/sram_management/bundle_slicer.h"
#include "graph_compiler/passes/sram_management/mme_geometry.h"
#include "graph_compiler/passes/sram_management/mme_slicing_strategy.h"

namespace gaudi
{
using CostEngine = CostModel::Cost::Engine;

// evaluate mme node cost (execution time, hbm traffic)
class MMECostModel : public CostModel
{
public:
    MMECostModel(const HalReader& halReader) : m_halReader(halReader) {}
    Cost calcCost(const pNode& node,
                  const SliceReferenceList& inputs,
                  const SliceReferenceList& outputs) const override;

protected:
    uint64_t calcProcessingTime(const pNode& node,
                              const SliceReferenceList& inputs,
                              const SliceReferenceList& outputs) const;
    // return estimated hbm traffic in bytes
    uint64_t calcHBMTraffic(const pNode& node,
                            const SliceReferenceList& inputs,
                            const SliceReferenceList& outputs) const;

    MmeCommon::MmeStrategy getMMEStrategy(const pNode&                 node,
                                          const pSliceReference&       inputASliceRef,
                                          const pSliceReference&       outputSliceRef,
                                          const MmeCommon::EMmeOpType& operationType) const;

    void getMMEGeometryInElements(const MmeCommon::MmeStrategy& mmeStrategy,
                                  const synDataType&            finalElementType,
                                  unsigned&                     mmeGeometryHeightElements,
                                  unsigned&                     mmeGeometryWidthElements) const;

    pSliceReference getSbReusedOperand(const MmeCommon::MmeStrategy& mmeStrategy,
                                       const pSliceReference&        opA,
                                       const pSliceReference&        opB) const;

    bool isAlignedToCL(const pSliceReference& sliceRef) const;

    float getUnalignedPenaltyFactor(const MmeCommon::MmeStrategy& mmeStrategy,
                                    const SliceReferenceList&     inputs,
                                    unsigned                      widthActivations,
                                    unsigned                      heightActivations) const;

    uint64_t getBatchFactor(const pNode&                  node,
                            const MmeCommon::MmeStrategy& mmeStrategy,
                            const MmeCommon::EMmeOpType&  operationType,
                            const pSliceReference&        output,
                            unsigned                      heightOutputElements,
                            unsigned                      widthOutputElements) const;

    const HalReader& m_halReader;
    mutable std::unordered_map<pNode, MmeDimController> m_dimControlers;
};


class TPCCostModel : public CostModel
{
public:
    TPCCostModel(const HalReader& halReader) : m_halReader(halReader) {}
    Cost calcCost(const pNode& node,
                  const SliceReferenceList& inputs,
                  const SliceReferenceList& outputs) const override;
protected:
    uint64_t calcHBMTraffic(const pNode& node,
                            const SliceReferenceList& inputs,
                            const SliceReferenceList& outputs) const;

    const HalReader& m_halReader;
};

// Cost model to evaluate DMA "fetch" operations (bringing inputs from hbm to sram)
class DmaFetchCostModel : public CostModel
{
public:
    DmaFetchCostModel(const HabanaGraph& graph);

    Cost calcCost(const pNode& node,
                  const SliceReferenceList& inputs,
                  const SliceReferenceList& outputs) const override;

protected:
    uint64_t calcHBMTraffic(const SliceReferenceList& sliceRefs, bool isInput) const;

    const HabanaGraph& m_graph;

    // We use the same cache as the tensor slicer does to be able to mimic accurately the double buffering logic.
    // Since we don't really want to create real tensors, we have a cache of booleans (true will indicate existence in
    // the cache).
    using dmaCostModelCache = TensorSlicerCache<bool>;
    mutable std::map<pSlicedOperand, dmaCostModelCache, SlicedOperand::SliceOperandComp> m_cacheMap;
};

// Cost model to evaluate DMA "eviction" operations (copying tensors from sram to hbm if needed).
// Important note: while the class is called "DMA", it also evaluates some TPC operations (cast nodes, to cast outputs
// which are "partials" and are originally bf16 back to their original data type).
class DmaEvictionCostModel : public CostModel
{
public:
    DmaEvictionCostModel(const HabanaGraph& graph, const pBundle& bundle, const pMmeSlicingStrategy& strategy);

    Cost
    calcCost(const pNode& node, const SliceReferenceList& inputs, const SliceReferenceList& outputs) const override;

protected:
    std::pair<CostEngine, uint64_t> calcHBMTraffic(const SliceReferenceList& outputs) const;

    const HabanaGraph& m_graph;
    NodeSet            m_bundleNodeSet;

    // This set is used in order to "remember" which outputs were already evacuated, in order not to count the same
    // slice twice, as can happen i.e. in case of partials, where we'll have 2 different operations, both having "the
    // same" output.
    mutable std::unordered_set<pSliceReference, SliceReference::Hasher, SliceReference::IsEqual> m_evacuatedSliceRefSet;
};


} //namespace gaudi
