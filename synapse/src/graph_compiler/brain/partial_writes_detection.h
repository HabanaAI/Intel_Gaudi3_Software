#pragma once

#include "habana_graph.h"

namespace gc::layered_brain
{
// Check if the tensor's producer writes it with partial cache line writes.
// Define decisions to improve partial cache writes performance.
// Assumes running on the sliced graph
class PartialWritesDetector
{
public:
    struct PartialsDecision
    {
        bool                          warmupCache        = false;
        bool                          allocInSingleDCore = false;
        std::optional<CacheDirective> cacheDirective;
    };

    PartialWritesDetector(const TensorPtr& output, const HabanaGraph& g);

    // Returns a decision if the tensor is valid. If unsupported tensor is detected, returns nullopt to fail the
    // strategy selection
    std::optional<PartialsDecision> checkTensor() const;

private:
    PartialsDecision checkUnsliceableTensor() const;
    PartialsDecision checkTensorSizeAndStrides() const;

    bool shouldHandleNode(const NodePtr& node) const;

    bool isAllRequired() const;
    bool isRmw() const;
    bool isPartialWrite() const;
    bool isIsmeAligned() const;
    bool isSizeFullCLMult(TSize bytes, bool allowExactHalfCL = false) const;
    bool isSizeCLAligned(TSize bytes, bool allowExactHalfCL = false) const;
    bool isTpcSparseAccess() const;
    bool isTensorLargerThanCache() const;
    bool isReducedTensor() const;
    bool perforatedOnNonReducedDim() const;
    bool isSliceOfBPT() const;

    TSize getFcdIsmeInBytes() const;

    void setWarmupCacheRequired(PartialsDecision& decision) const;
    void setAllocFullTensorInSingleDcore(PartialsDecision& decision) const;
    void setWriteInNoAlloc(PartialsDecision& decision) const;
    void setWriteInAllocD(PartialsDecision& decision) const;

    TensorPtr          m_output;
    const HabanaGraph& m_graph;
    NodePtr            m_producer;
    unsigned           m_outputIndex;
};

}  // namespace gc::layered_brain