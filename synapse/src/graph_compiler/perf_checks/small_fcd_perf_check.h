#pragma once

#include "gc_perf.h"

class Tensor;

// This perf check report about tensors with small fcd which is unaligned to cache line
// Readers: report if there are many readers (consumers or mme fetches) that may read with low utilization
// Writer: report if the producer will write the tensor with low utilization
class SmallFcdPerfCheck : public GcPerf::GcPerfCheck
{
public:
    void                          run(const std::shared_ptr<Node>& node) const override;
    std::string_view              name() const override { return "SmallFcdPerfCheck"; }
    static GcPerf::GcPerfCheckPtr createPerfCheck(const HabanaGraph& g);

private:
    SmallFcdPerfCheck(const HabanaGraph& g);
    unsigned getNumberOfReads(const std::shared_ptr<Tensor>& t) const;
    void     performReadChecks(const std::shared_ptr<Tensor>& t, const TSize& fcdSize) const;
    void
    performWriteChecks(const std::shared_ptr<Node>& node, const std::shared_ptr<Tensor>& t, const TSize& fcdSize) const;

    TSize m_cacheLineSizeInBytes;
};