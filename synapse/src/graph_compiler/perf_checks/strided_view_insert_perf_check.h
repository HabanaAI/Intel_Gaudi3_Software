#pragma once

#include "gc_perf.h"

// This perf check report about strided view/insert operations.
// in general these operations should be converted to other less general operations at bridge level.
// If these strided view/insert operations are non dense, these might cause performance issues.
class StridedOpPerfCheck : public GcPerf::GcPerfCheck
{
public:
    void                          run(const std::shared_ptr<Node>& node) const override;
    std::string_view              name() const override { return "StridedOpPerfCheck"; }
    static GcPerf::GcPerfCheckPtr createPerfCheck(const HabanaGraph& g);

private:
    StridedOpPerfCheck(const HabanaGraph& g);
};