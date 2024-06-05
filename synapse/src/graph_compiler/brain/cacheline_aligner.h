#pragma once
#include <unordered_map>
#include <memory>
#include "habana_graph.h"
#include "types.h"
#include "brain_data.h"

using namespace gc::layered_brain;
/**
 * @brief Recieves a graph and a map of bpt clone tensors to a bool
 *        indicating whether the original cloned tensor is presistent (bpt clones are persistent by definition).
 *        For each join output that is not graph persistent with fcd stride unaligned to cacheline,
 *        align the fcd stride to cacheline if size increase is less than
 *        a configurable value.
 */
class CachelineAligner
{
public:
    CachelineAligner(HabanaGraph& g, const BPTClonePersistenceMap& bptClonePersistence);
    virtual ~CachelineAligner() = default;

    void run();

protected:
    void alignFcdStrideToCacheLine(const TensorPtr& bpt, const NStrideArray& alignedStrides, TSize alignedSize);

    NStrideArray::value_type getAlignedTensorSizeInBytes(const TensorPtr&    bpt,
                                                         const NStrideArray& alignedStrides) const;

    bool isAlignedFcdStride(const NStrideArray& strides) const;

    bool isGraphPersistent(const TensorPtr& t) const;

    unsigned getHalfCachelineSize() const { return m_halfCachelineSize; }

    unsigned getCachelineSize() const { return m_cachelineSize; }

    unsigned         getAlignedFcdStrideSizeInBytes(const NStrideArray::value_type& fcdStrideInBytes,
                                                    unsigned                        alignTargetSizeInBytes) const;
    virtual unsigned getAlignmentTargetSize(const NStrideArray::value_type& fcdStrideInBytes) const;

    NStrideArray getCachelineAlignedStrides(unsigned            rank,
                                            const NSizeArray&   sizes,
                                            const NStrideArray& initialStrides,
                                            unsigned            alignTargetSize) const;

    HabanaGraph&                  m_graph;
    const BPTClonePersistenceMap& m_bptClonePersistence;
    const unsigned                m_cachelineSize;
    const unsigned                m_halfCachelineSize;
    static constexpr unsigned     FCD_DIM = 0;
};

/**
 * @brief Extends the basic cacheline aligner by attempting to align eligible bpt fcd strides
 *        to half cacheline size instead of to full cacheline when fcd stride is smaller than
 *        half cacheline.
 */
class SmallFCDHalfCachelineAligner : public CachelineAligner
{
public:
    SmallFCDHalfCachelineAligner(HabanaGraph& g, const BPTClonePersistenceMap& bptProducers)
    : CachelineAligner(g, bptProducers)
    {
    }

protected:
    virtual unsigned getAlignmentTargetSize(const NStrideArray::value_type& fcdStrideInBytes) const override;
};