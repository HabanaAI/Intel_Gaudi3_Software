
#pragma once

#include "brain_data.h"
#include "types.h"

namespace gc::layered_brain
{
// The reduction handler is responsible to identify the cases where a slice has multiple producers
// (multiple ISRs with the same output ISMR) and create a reduction node for it.
class ReductionHandler
{
public:
    ReductionHandler(const NodeSet& requireCast, const NodeSet& requireMemset)
    : m_bundleIdx(std::nullopt), m_requireCast(requireCast), m_requireMemset(requireMemset) {};

    ReductionHandler(BundleIdx bundleIdx, const NodeSet& requireCast, const NodeSet& requireMemset)
    : m_bundleIdx(bundleIdx), m_requireCast(requireCast), m_requireMemset(requireMemset) {};

    void addProducerForTensorSlice(const TensorPtr& slice,
                                   const NodePtr&   sliceProducer,
                                   const NodePtr&   origProducer,
                                   unsigned         producerOutputIdx);  // The output idx of the produced tensor

    NodeVector createReductionNodes() const;

private:
    struct SlicedTensorProducers
    {
        NodePtr    origProducer;
        NodeVector sliceProducers;
        unsigned   outputIdx;
    };

    bool     requiresReduction(const SlicedTensorProducers& producers) const;
    unsigned getReductionOp(const NodePtr& origProducer) const;

    // Converts the reduction tensors to F32 and add cast to the original data-type.
    // Returns the new created cast node.
    NodePtr handleReductionDataType(const NodePtr& reductionNode) const;
    bool    requiresCastForReduction(const TensorPtr& tensor, const NodePtr& origProducer) const;

    bool    requiresMemset(const NodePtr& origProducer) const;
    NodePtr addMemsetForReduction(const NodePtr& reduction) const;

    std::string getNodeNamePrefix() const;

    const std::optional<BundleIdx>                       m_bundleIdx;
    const NodeSet                                        m_requireCast;
    const NodeSet                                        m_requireMemset;
    TensorToItemOrderedMap<SlicedTensorProducers>        m_slicedTensorProducers;
    static constexpr synDataType                         HIGH_PRECISION_DATA_TYPE_FOR_REDUCTION = syn_type_single;
};

}  // namespace gc::layered_brain